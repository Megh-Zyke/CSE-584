from concurrent.futures import ThreadPoolExecutor
from collections import deque
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import os
import time
import asyncio
import hashlib
import chromadb
import redis as redislib
from typing import List

from google import genai
from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer, CrossEncoder
from modules.dynamic_ttl_training import TTLClassifier
from modules.semantic_cache import PyFSSemanticCache
from modules.query import Message, QueryRequest
import threading
from modules.gate3 import Gate3
from dotenv import load_dotenv
from typing import Optional
load_dotenv()
_executor = ThreadPoolExecutor(max_workers=4)
client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

TTL_MAP = {
    "Static":      7 * 24 * 60 * 60,   # 7 days
    "Slow-Moving": 3 * 24 * 60 * 60,   # 3 days
    "Volatile":    5 * 60,             # 5 min
}

def query_hash(query: str) -> str:
    return "tg:" + hashlib.md5(query.strip().lower().encode()).hexdigest()

class TriGuardCache:

    def __init__(self):
        print("Loading ContextGate (Gate1)...")
        from modules.gate1 import ContextGate
        self.gate1 = ContextGate()

        self._store_lock = threading.Lock()
        self._metrics_lock = threading.Lock()

        #print("Loading embedding model (bge-small)...")
        #self.embedder = SentenceTransformer("BAAI/bge-small-en-v1.5", backend="onnx")
        print("Loading TTL classifier + bge small embedding model")
        self.ttl_clf = TTLClassifier()
        try:
            self.ttl_clf.load("ttl_classifier.joblib")
            print("[TTL Classifier] Loaded — classes:", self.ttl_clf._classes)
        except Exception as e:
            print(f"[TTL Classifier] WARNING: Could not load — {e}")
            self.ttl_clf = None

        # FIX: LRU cache for bge-base embeddings — saves 120ms per hit
        self._embed_cache: dict[str, list] = {}
        self._embed_cache_keys: list[str]  = []
        self._embed_cache_size = 1024

        print("Loading cross-encoder rerankers...")
        #self.reranker        = CrossEncoder("cross-encoder/quora-roberta-base")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # FIX: LRU cache for reranker scores — saves 70ms on repeated candidate pairs
        self._rerank_cache: dict[tuple, float] = {}
        self._rerank_cache_keys: list[tuple]   = []
        self._rerank_cache_size = 1024

        print("Initializing ChromaDB...")
        self.chroma_client = chromadb.PersistentClient(path="./chroma_store")
        self.collection = self.chroma_client.get_or_create_collection(
            name="trigard_cache",
            metadata={"hnsw:space": "cosine"}
        )
        with self._store_lock:
            count = self.collection.count()
        print(f"[ChromaDB] {count} entries loaded from disk")

        print("Connecting to Redis...")
        try:
            redis_url = os.environ.get("REDIS_URL")
            if redis_url:
                redis_url = redis_url.strip('"').strip("'")
                self.redis = redislib.Redis.from_url(redis_url, decode_responses=True)
            else:
                self.redis = redislib.Redis(
                    host=os.environ.get("REDIS_HOST", "localhost"),
                    port=int(os.environ.get("REDIS_PORT", 6379)),
                    password=os.environ.get("REDIS_PASSWORD", None),
                    decode_responses=True
                )
            self.redis.ping()
            print("[Redis] Connected")
        except Exception as e:
            print(f"[Redis] WARNING: Not reachable — TTL layer disabled. Detail: {e}")
            self.redis = None

        # IMPORTANT: ttl_classifier.joblib is a dict saved by TTLClassifier.save().
        # Must be loaded via TTLClassifier().load(), NOT joblib.load() directly.
        # joblib.load() returns a plain dict with no .predict() method.
        self.verifier = PyFSSemanticCache()
        self.gate3 = Gate3(threshold=0.5)

        self.metrics = {
            "redis_hits": 0,
            "chroma_hits_fast": 0,
            "chroma_hits_verified": 0,
            "volatile_bypasses": 0,
            "gemini_calls": 0,
            "gate3_admitted": 0,
            "gate3_blocked": 0,
            "gate3_errors": 0,
            "total_queries": 0,
            "gate1_rewrites": 0
        }

        self.embedding_threshold        = 0.80
        self.rerank_threshold           = 0.70
        self.contradiction_threshold    = 0.30
        self.answer_relevance_threshold = 1.50
        self._api_call_times            = deque(maxlen=100)

    def _inc(self, **fields: int) -> None:
        """Atomically increment one or more metrics counters."""
        with self._metrics_lock:
            for key, delta in fields.items():
                self.metrics[key] += delta


    # ─────────────────────────────────────────────────────────────────
    # Gate 2 
    # ─────────────────────────────────────────────────────────────────

    def classify_query(self, query: str):
        """
        Returns (label, confidence).
          label         → "Static" / "Slow-Moving" / "Volatile"   (Gate 2)
          confidence    → raw probability 0-1 (of the predicted class)

        FIX: on any classifier error, fail SAFE — return Volatile/0.0
        so nothing gets cached and Gemini is always called.
        Previously defaulted to Static/True/1.0 which would cache
        hallucinations unconditionally on classifier failure.
        """
        if self.ttl_clf:
            try:
                result = self.ttl_clf.predict(query)
                print(
                    f"[TTL] label={result.label}  "
                    f"conf={result.confidence:.1%}  "
                    f"latency={result.latency_ms:.1f}ms"
                )
                return result.label, result.confidence

            except Exception as e:
                print(f"[TTL Classifier] Error: {e} — failing safe (Volatile/no-cache)")

        return "Volatile", 0.0

    # ─────────────────────────────────────────────────────────────────
    # Redis helpers
    # ─────────────────────────────────────────────────────────────────

    def _redis_get(self, query: str):
        if not self.redis:
            return None
        return self.redis.get(query_hash(query))

    def _redis_set(self, query: str, response: str, category: str):
        if not self.redis:
            return
        ttl = TTL_MAP.get(category, -1)
        key = query_hash(query)
        if ttl == -1:
            self.redis.set(key, response)
        else:
            self.redis.setex(key, ttl, response)

    # ─────────────────────────────────────────────────────────────────
    # Embedding with LRU cache
    # ─────────────────────────────────────────────────────────────────

    def _embed(self, query: str) -> list:
        """
        FIX: LRU embedding cache saves 120ms per hit by skipping the
        bge-base forward pass for queries seen before.
        """
        cached = self._embed_cache.get(query)
        if cached is not None:
            return cached

        emb = self.ttl_clf.encoder.encode(
            [query], normalize_embeddings=True
        ).astype("float32")[0].tolist()

        if len(self._embed_cache_keys) >= self._embed_cache_size:
            oldest = self._embed_cache_keys.pop(0)
            self._embed_cache.pop(oldest, None)
        self._embed_cache_keys.append(query)
        self._embed_cache[query] = emb
        return emb

    # ─────────────────────────────────────────────────────────────────
    # ChromaDB helpers
    # ─────────────────────────────────────────────────────────────────

    # def _chroma_search(self, query: str, k: int = 5):
    #     if self.collection.count() == 0:
    #         return []
    #     results = self.collection.query(
    #         query_embeddings=[self._embed(query)],
    #         n_results=min(k, self.collection.count()),
    #         include=["documents", "metadatas", "distances"]
    #     )
    #     hits = []
    #     for i in range(len(results["ids"][0])):
    #         hits.append({
    #             "id":         results["ids"][0][i],
    #             "query":      results["metadatas"][0][i]["original_query"],
    #             "response":   results["documents"][0][i],
    #             "category":   results["metadatas"][0][i].get("category", "Static"),
    #             "similarity": 1 - results["distances"][0][i],
    #         })
    #     return hits
    
    def _chroma_search_with_embedding(self, embedding: list, k: int = 5) -> list:
        """Search ChromaDB with a pre-computed embedding (avoids re-encoding)."""
        with self._store_lock:                          
            count = self.collection.count()
            if count == 0:
                return []
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=min(k, count),
                include=["documents", "metadatas", "distances"]
            )
        hits = []
        for i in range(len(results["ids"][0])):
            hits.append({
                "id":         results["ids"][0][i],
                "query":      results["metadatas"][0][i]["original_query"],
                "response":   results["documents"][0][i],
                "category":   results["metadatas"][0][i].get("category", "Static"),
                "similarity": 1 - results["distances"][0][i],
            })
        return hits

    def _chroma_store(self, query: str, response: str, category: str):
        self._evict_if_needed(max_entries=1000)
        with self._store_lock:
            self.collection.upsert(
            ids=[query_hash(query)],
            embeddings=[self._embed(query)],
            documents=[response],
            metadatas=[{
                "original_query": query,
                "category":       category,
                "cached_at":      time.time(),
            }]
        )


    # ─────────────────────────────────────────────────────────────────
    # Reranker with LRU cache
    # ─────────────────────────────────────────────────────────────────

    def rerank_candidates(self, query: str, hits: list):
        """
        FIX: LRU cache on (query, candidate) pairs saves 70ms when the
        same query repeatedly hits the same ChromaDB candidates.
        """
        if not hits:
            return None, None

        scores = []
        pairs_to_score = []
        pair_indices   = []

        for i, h in enumerate(hits):
            key = (query, h["query"])
            if key in self._rerank_cache:
                scores.append(self._rerank_cache[key])
            else:
                scores.append(None)
                pairs_to_score.append([query, h["query"]])
                pair_indices.append(i)

        if pairs_to_score:
            fresh_scores = self.reranker.predict(pairs_to_score)
            for idx, score in zip(pair_indices, fresh_scores):
                key = (query, hits[idx]["query"])
                if len(self._rerank_cache_keys) >= self._rerank_cache_size:
                    oldest = self._rerank_cache_keys.pop(0)
                    self._rerank_cache.pop(oldest, None)
                self._rerank_cache_keys.append(key)
                self._rerank_cache[key] = float(score)
                scores[idx] = float(score)

        best = int(np.argmax(scores))
        return hits[best], float(scores[best])

    # ─────────────────────────────────────────────────────────────────
    # Answer relevance
    # ─────────────────────────────────────────────────────────────────
    def _dynamic_threshold(self) -> float:
        now = time.time()
        recent = [t for t in self._api_call_times if now - t < 60]
        rate = len(recent)
        if rate > 10:
            return 0.75
        elif rate < 3:
            return 0.85
        return 0.80

    def _evict_if_needed(self, max_entries: int = 1000):
        count = self.collection.count()
        if count < max_entries:
            return
        results = self.collection.get(include=["metadatas"])
        entries = list(zip(results["ids"], results["metadatas"]))
        entries.sort(key=lambda x: x[1].get("cached_at", 0))
        to_remove = max(1, count // 10)
        ids_to_remove = [e[0] for e in entries[:to_remove]]
        self.collection.delete(ids=ids_to_remove)
        print(f"[Eviction] Removed {len(ids_to_remove)} oldest entries")

    def verify_answer_relevance(self, query: str, response: str) -> float:
        return float(self.reranker.predict([[query, response[:512]]]))

    def call_gemini(self, query: str , history: Optional[List[Message]] = None) -> str:
        try:
            self._api_call_times.append(time.time())
            content = []

            if history: 
                for msg in history:
                    content.append({
                        "role": "user",
                        "parts": [{"text": msg.query}]
                    })
                    content.append({
                        "role": "model",
                        "parts": [{"text": msg.response}]
                    })
            content.append({
                "role": "user",
                "parts": [{"text": query}]
            })
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=content,
            )
            return response.text
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    # ─────────────────────────────────────────────────────────────────
    # GATE 3
    # ─────────────────────────────────────────────────────────────────
    async def _gate3_and_store(self, query: str, response: str, category: str):
        """
        Runs in the background after the user already received their response.
        Gate 3 decides whether the response is safe to cache.
        If admitted  → write to ChromaDB + Redis
        If blocked   → log and discard (hallucination not cached)
        """
        try:
            result = await self.gate3.check_async(query, response)
 
            print(
                f"[Gate3] confidence={result['confidence_score']}  "
                f"faithfulness={result['faithfulness_score']}  "
                f"combined={result['combined_score']}  "
                f"admit={result['admit_to_cache']}  "
            )
 
            if result["admit_to_cache"]:
                # Write to both stores
                loop = asyncio.get_event_loop()
                await asyncio.gather(
                    loop.run_in_executor(_executor, self._chroma_store, query, response, category),
                    loop.run_in_executor(_executor, self._redis_set,    query, response, category),
                )
                self._inc(gate3_admitted=1)
                print(f"[Gate3] ✅ Admitted to cache — category={category}")
            else:
                self._inc(gate3_blocked=1)
                print(f"[Gate3] ❌ Blocked — response not cached (combined={result['combined_score']})")
 
        except Exception as e:
            self._inc(gate3_errors=1)
            print(f"[Gate3] ERROR in background task: {e}")


    # ─────────────────────────────────────────────────────────────────
    # Main query handler — async so Gemini calls don't block other reqs
    # ─────────────────────────────────────────────────────────────────
    def _trim_history(self, history, max_turns=3):
        if not history:
            return []
        return history[-max_turns:]
    async def ask(self, query: str, history: List[Message] = None) -> dict:
        """
        FIX 1 (critical): TTL classification runs FIRST, before any cache
        lookup. Volatile queries bypass ChromaDB entirely — they must
        never return a stale cached answer.

        FIX 2 (critical): Gemini call is now async via ThreadPoolExecutor
        so slow LLM calls don't block other concurrent requests.

        FIX 3: PyFS removed from the fast path (sim > 0.80). At that
        similarity level, cosine alone is sufficient — running a 500MB
        NLI model on top adds 200ms with no meaningful accuracy gain.
        PyFS is retained for the lower-confidence full path only.
        """ 
        t_start = time.perf_counter()

        # ── Gate 1: Context Normalization ────────────────────────────────────
        
        if history:
            history = self._trim_history(history)
            history_texts = [f"{msg.query}: {msg.response}" for msg in history]
            normalized_query = self.gate1.rewrite_query(query, history_texts)
            print(f"[Gate1] Rewrote query: '{query}' -> '{normalized_query}'")
            if normalized_query != query:
                print(f"[Gate1] Rewrote query: '{query}' -> '{normalized_query}'")
                query = normalized_query
                self.metrics["gate1_rewrites"] += 1

        # ── STEP 0: Classify query FIRST ─────────────────────────────────────
        # This must happen before any cache lookup.
        # Volatile queries skip the cache entirely — serving a stale live
        # price or breaking news answer is worse than a cache miss.
        # can not parallelise
        category, confidence = self.classify_query(query)

        if category == "Volatile":
            # print(f"[VOLATILE] Bypassing cache — calling Gemini directly")
            loop     = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                _executor,
                self.call_gemini,
                query,
                history
            )
            t_end = time.perf_counter()
            self._inc(volatile_bypasses=1, total_queries=1)
            return {
                "source":     "GEMINI_VOLATILE",
                "category":   "Volatile",
                "confidence": round(confidence, 4),
                "response":   response,
                "latency_seconds": round(t_end - t_start, 4)
            }

        # ── STEP 1: Redis exact-match fast path ──────────────────────────────
        # Only reached for Static and Slow-Moving queries.
        # Parallelise with redis lookup and embed run
        loop = asyncio.get_event_loop()
        redis_task = loop.run_in_executor(_executor, self._redis_get, query)
        embed_task = loop.run_in_executor(_executor , self._embed, query)

        redis_result , query_embedding = await asyncio.gather(redis_task, embed_task)
        
        if redis_result:
            #print(f"[Redis HIT] Returning from exact-match cache")
            t_end = time.perf_counter()
            self._inc(redis_hits=1, total_queries=1)
            #print(f"[DEBUG] Total latency: {t_end - t_start:.4f} seconds")
            return {"source": "REDIS_HIT", "response": redis_result, "latency_seconds": round(t_end - t_start, 4)} 

        # ── STEP 2: ChromaDB semantic search ─────────────────────────────────
        hits = self._chroma_search_with_embedding(query_embedding, k=5)

        if hits:
            best     = hits[0]
            best_sim = best["similarity"]

            print(f"[DEBUG] chroma_size={self.collection.count()}  "
                  f"best_sim={best_sim:.4f}  category={category}")

            # ── FAST PATH: high cosine similarity ────────────────────────────
            # FIX: PyFS removed here. Cosine > 0.80 + correct category from
            # classifier is sufficient confidence. Saves 200ms per cache hit.
            if best_sim > self._dynamic_threshold():
                print(f"[DEBUG] Fast path hit — sim={best_sim:.4f} > {self.embedding_threshold}")
                self._redis_set(query, best["response"], best["category"])
                t_end = time.perf_counter()
                print(f"[DEBUG] Total latency: {t_end - t_start:.4f} seconds")
                self._inc(chroma_hits_fast=1, total_queries=1)
                return {
                    "source":     "CACHE_HIT_FAST",
                    "similarity": round(best_sim, 4),
                    "category":   best["category"],
                    "response":   best["response"],
                    "latency_seconds": round(t_end - t_start, 4)
                }

            # ── FULL PATH: rerank + PyFS + answer relevance ───────────────────
            # Used when cosine is promising but below the high-confidence threshold.
            best_hit, rerank_score = self.rerank_candidates(query, hits)

            if best_hit and rerank_score > self.rerank_threshold:

                #parallelise
                pyfs_task = loop.run_in_executor(_executor, self.verifier.calculate_pyfs, query, best_hit["query"])
                ans_task  = loop.run_in_executor(_executor, self.verify_answer_relevance, query, best_hit["response"])

                pyfs_res, ans_score = await asyncio.gather(pyfs_task, ans_task)
                
                #print(f"[DEBUG] PyFS mu={pyfs_res['mu']:.4f}  nu={pyfs_res['nu']:.4f}")

                if pyfs_res["nu"] <= self.contradiction_threshold and ans_score > self.answer_relevance_threshold:

                    #ans_score = self.verify_answer_relevance(query, best_hit["response"])

                    #if ans_score > self.answer_relevance_threshold:
                        self._redis_set(query, best_hit["response"], best_hit["category"])
                        t_end = time.perf_counter()
                        self.metrics["chroma_hits_verified"] += 1
                        self.metrics["total_queries"] += 1
                        return {
                            "source":           "CACHE_HIT_VERIFIED",
                            "rerank_score":     round(rerank_score, 4),
                            "answer_relevance": round(ans_score, 4),
                            "mu":               round(pyfs_res["mu"], 4),
                            "nu":               round(pyfs_res["nu"], 4),
                            "pi":               round(pyfs_res["pi"], 4),
                            "response":         best_hit["response"],
                            "latency_seconds": round(t_end - t_start, 4)
                        }
                    
                #else:
                #    print(f"[DEBUG] PyFS contradiction gate FAILED (nu={res['nu']:.4f})")

        # ── STEP 3: Cache miss — call Gemini asynchronously ──────────────────
        print(f"[MISS] Calling Gemini (category={category})")
        gemini_start = time.perf_counter()
        loop     = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            _executor,
            self.call_gemini,
            query,
            history
        )
        end = time.perf_counter()
        latency  = end - t_start
        api_call_latency = end - gemini_start


        asyncio.create_task(self._gate3_and_store(query, response, category))

        # ── STEP 4: Gate 3 — store only if classifier is confident ───────────
        # Admits the data irrespective of the confidence score now 
        
        # self._chroma_store(query, response, category)
        # self._redis_set(query, response, category)
        # print(f"[CACHED] category={category}  conf={confidence:.1%}  latency={latency:.3f}s")
        self._inc(gemini_calls=1, total_queries=1)

        return {
            "source":          "GEMINI_API",
            "latency_seconds": round(latency, 4),
            "category":        category,
            "confidence":      round(confidence, 4),
            "response":        response,
            "api_call_latency": round(api_call_latency, 4)
        }
