import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import os
import time
import hashlib
import uvicorn
import traceback

import chromadb
import redis as redislib

from google import genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Meghanand's TTL classifier — must be imported as class, not raw joblib
from train_transformer import TTLClassifier

load_dotenv()

API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found")

client = genai.Client(api_key=API_KEY)

app = FastAPI(title="Tri-Guard Semantic Cache API", version="4.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str


# ─── TTL in seconds — labels match TTLClassifier exactly ──────────────────────
TTL_MAP = {
    "Static":       -1,                 # Never expires
    "Slow-Moving":  30 * 24 * 60 * 60, # 30 days
    "Volatile":     5 * 60,            # 5 minutes
}


def query_hash(query: str) -> str:
    """Stable hash key for Redis exact lookup."""
    return "tg:" + hashlib.md5(query.strip().lower().encode()).hexdigest()


# -----------------------------------------------------
# PyFS Semantic Verifier (unchanged from original)
# -----------------------------------------------------

class PyFSSemanticCache:

    def __init__(self, model_name="cross-encoder/stsb-roberta-base"):
        print("Initializing PyFS STS Verifier...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def _nli_probs(self, a, b):
        inputs = self.tokenizer(a, b, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        score = float(torch.sigmoid(logits).cpu().numpy().flatten()[0])
        return score, 1.0 - score, 0.0

    def calculate_pyfs(self, q1, q2):
        mu1, nu1, ne1 = self._nli_probs(q1, q2)
        mu2, nu2, ne2 = self._nli_probs(q2, q1)
        mu      = (mu1 + mu2) / 2
        nu      = (nu1 + nu2) / 2
        neutral = (ne1 + ne2) / 2
        pi      = np.sqrt(max(0, 1 - (mu**2 + nu**2)))
        return {
            "score":   float((mu**2) - (nu**2)),
            "mu":      float(mu),
            "nu":      float(nu),
            "pi":      float(pi),
            "neutral": float(neutral)
        }


# -----------------------------------------------------
# TriGuard Cache — ChromaDB + Redis replacing FAISS
# -----------------------------------------------------

class TriGuardCache:

    def __init__(self):

        # ── Embedding + rerankers (unchanged from original) ──
        print("Loading embedding model...")
        self.embedder        = SentenceTransformer("BAAI/bge-base-en-v1.5")

        print("Loading cross-encoder rerankers...")
        self.reranker        = CrossEncoder("cross-encoder/quora-roberta-base")
        self.answer_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # ── ChromaDB: replaces FAISS index + self.store ──
        # Persistent — cache survives server restarts
        print("Initializing ChromaDB...")
        self.chroma_client = chromadb.PersistentClient(path="./chroma_store")
        self.collection = self.chroma_client.get_or_create_collection(
            name="trigard_cache",
            metadata={"hnsw:space": "cosine"}
        )
        print(f"[ChromaDB] {self.collection.count()} entries loaded from disk")

        # ── Redis: dynamic TTL enforcement (Gate 2) ──
        print("Connecting to Redis...")
        try:
            self.redis = redislib.Redis(host="localhost", port=6379, decode_responses=True)
            self.redis.ping()
            print("[Redis] Connected")
        except redislib.exceptions.ConnectionError:
            print("[Redis] WARNING: Not reachable — TTL layer disabled")
            self.redis = None

        # ── TTL Classifier: Meghanand's trained model (Gate 2 + Gate 3) ──
        #
        # IMPORTANT: ttl_classifier.joblib is a dict saved by TTLClassifier.save().
        # It must be loaded via TTLClassifier().load(), NOT joblib.load() directly.
        # Calling joblib.load() returns a plain dict with no .predict() method.
        #
        # TTLClassifier.predict(query) returns a TTLResult with:
        #   .label          → "Static" / "Slow-Moving" / "Volatile"   (Gate 2)
        #   .ttl            → seconds (inf for Static)
        #   .confidence     → probability 0-1
        #   .admit_to_cache → True if confidence >= 0.90              (Gate 3)
        #   .latency_ms     → inference time
        print("Loading TTL classifier (Gate 2 + Gate 3)...")
        self.ttl_clf = TTLClassifier()
        try:
            self.ttl_clf.load("ttl_classifier.joblib")
            print("[TTL Classifier] Loaded — classes:", self.ttl_clf._classes)
        except Exception as e:
            print(f"[TTL Classifier] WARNING: Could not load — {e}")
            self.ttl_clf = None

        # ── PyFS verifier (unchanged) ──
        self.verifier = PyFSSemanticCache()

        # ── Thresholds (unchanged from original) ──
        self.embedding_threshold        = 0.80
        self.rerank_threshold           = 0.70
        self.contradiction_threshold    = 0.30
        self.answer_relevance_threshold = 1.50

    # -------------------------------------------------
    # Gate 2 + Gate 3 — TTL Classification
    # -------------------------------------------------

    def classify_query(self, query: str):
        """
        Runs Meghanand's TTLClassifier on a query.

        Returns:
            label         : "Static" / "Slow-Moving" / "Volatile"  → Gate 2
            admit_to_cache: True if confidence >= 90%               → Gate 3
            confidence    : raw probability score
        """
        if self.ttl_clf:
            try:
                result = self.ttl_clf.predict(query)
                print(
                    f"[TTL] label={result.label}  "
                    f"conf={result.confidence:.1%}  "
                    f"admit={result.admit_to_cache}  "
                    f"latency={result.latency_ms:.1f}ms"
                )
                return result.label, result.admit_to_cache, result.confidence
            except Exception as e:
                print(f"[TTL Classifier] Error: {e} — defaulting to Static/admit")
        # Fallback: treat as Static and always admit
        return "Static", True, 1.0

    # -------------------------------------------------
    # Redis helpers
    # -------------------------------------------------

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
            self.redis.set(key, response)         # Static — no expiry
        else:
            self.redis.setex(key, ttl, response)  # Volatile/Slow-Moving — auto expire

    # -------------------------------------------------
    # ChromaDB helpers
    # -------------------------------------------------

    def _embed(self, query: str):
        return self.embedder.encode(
            [query], normalize_embeddings=True
        ).astype("float32")[0].tolist()

    def _chroma_search(self, query: str, k: int = 5):
        """Search ChromaDB — returns list of hit dicts sorted by similarity."""
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_embeddings=[self._embed(query)],
            n_results=min(k, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        hits = []
        for i in range(len(results["ids"][0])):
            hits.append({
                "id":         results["ids"][0][i],
                "query":      results["metadatas"][0][i]["original_query"],
                "response":   results["documents"][0][i],
                "category":   results["metadatas"][0][i].get("category", "Static"),
                "similarity": 1 - results["distances"][0][i]   # distance → similarity
            })
        return hits

    def _chroma_store(self, query: str, response: str, category: str):
        """Persist entry to ChromaDB."""
        self.collection.upsert(
            ids=[query_hash(query)],
            embeddings=[self._embed(query)],
            documents=[response],
            metadatas=[{
                "original_query": query,
                "category":       category,
                "cached_at":      time.time()
            }]
        )

    # -------------------------------------------------
    # Rerank candidates (unchanged logic, adapted for ChromaDB hits)
    # -------------------------------------------------

    def rerank_candidates(self, query, hits):
        if not hits:
            return None, None
        pairs  = [[query, h["query"]] for h in hits]
        scores = self.reranker.predict(pairs)
        best   = int(np.argmax(scores))
        return hits[best], float(scores[best])

    # -------------------------------------------------
    # Answer relevance (unchanged)
    # -------------------------------------------------

    def verify_answer_relevance(self, query, response):
        return float(self.answer_reranker.predict([[query, response[:512]]]))

    # -------------------------------------------------
    # Gemini API (unchanged)
    # -------------------------------------------------

    def call_gemini(self, query):
        try:
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=query
            )
            return response.text
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # -------------------------------------------------
    # Main Query Handler
    # -------------------------------------------------

    def ask(self, query: str):

        # ── Step 1: Redis exact-match fast path ──────────────────────────────
        # Checks if this exact query was recently cached with a valid TTL
        redis_result = self._redis_get(query)
        if redis_result:
            print(f"[Redis HIT] Returning instantly from TTL cache")
            return {"source": "REDIS_HIT", "response": redis_result}

        # ── Step 2: ChromaDB semantic search ─────────────────────────────────
        # Finds the most semantically similar cached query
        hits = self._chroma_search(query, k=5)

        if hits:
            best     = hits[0]
            best_sim = best["similarity"]

            print(f"[DEBUG] chroma size  : {self.collection.count()}")
            print(f"[DEBUG] best_sim     : {best_sim:.4f}")
            print(f"[DEBUG] embed thresh : {self.embedding_threshold}")

            # ── FAST PATH: top-1 cosine hit passes threshold ──────────────
            if best_sim > self.embedding_threshold:

                res = self.verifier.calculate_pyfs(query, best["query"])

                print(f"[DEBUG] cached_query : {best['query']}")
                print(f"[DEBUG] mu={res['mu']:.4f}  nu={res['nu']:.4f}  pi={res['pi']:.4f}")
                print(f"[DEBUG] contradiction: {res['nu']:.4f} <= {self.contradiction_threshold} = {res['nu'] <= self.contradiction_threshold}")

                if res["nu"] <= self.contradiction_threshold:
                    # Refresh Redis TTL for this exact query
                    self._redis_set(query, best["response"], best["category"])
                    return {
                        "source":     "CACHE_HIT_FAST",
                        "similarity": best_sim,
                        "mu":         res["mu"],
                        "nu":         res["nu"],
                        "pi":         res["pi"],
                        "response":   best["response"]
                    }

            # ── FULL PATH: rerank all 5 candidates ────────────────────────
            best_hit, rerank_score = self.rerank_candidates(query, hits)

            if best_hit and rerank_score > self.rerank_threshold:

                res = self.verifier.calculate_pyfs(query, best_hit["query"])

                if res["nu"] <= self.contradiction_threshold:

                    ans_score = self.verify_answer_relevance(query, best_hit["response"])

                    if ans_score > self.answer_relevance_threshold:
                        self._redis_set(query, best_hit["response"], best_hit["category"])
                        return {
                            "source":           "CACHE_HIT_VERIFIED",
                            "rerank_score":     rerank_score,
                            "answer_relevance": ans_score,
                            "mu":               res["mu"],
                            "nu":               res["nu"],
                            "pi":               res["pi"],
                            "response":         best_hit["response"]
                        }
                    else:
                        print("[DEBUG] Answer relevance gate FAILED")
                else:
                    print("[DEBUG] PyFS contradiction gate FAILED")

        # ── Step 3: Cache miss — call Gemini ─────────────────────────────────
        start    = time.time()
        response = self.call_gemini(query)
        latency  = time.time() - start

        # ── Step 4: Gate 2 + Gate 3 via TTLClassifier ────────────────────────
        #
        # classify_query() returns:
        #   category      → TTL bucket (Gate 2: Static/Slow-Moving/Volatile)
        #   admit         → True if confidence >= 90% (Gate 3: quality check)
        #   confidence    → raw probability
        #
        # Only store in cache if Gate 3 passes (admit=True).
        # This prevents hallucinated or low-confidence answers from being cached.
        category, admit, confidence = self.classify_query(query)

        if admit:
            self._chroma_store(query, response, category)
            self._redis_set(query, response, category)
            print(f"[CACHED] category={category}  confidence={confidence:.1%}  latency={latency:.3f}s")
        else:
            print(f"[NOT CACHED] Gate 3 failed — confidence {confidence:.1%} below 90% threshold")

        return {
            "source":          "GEMINI_API",
            "latency_seconds": round(latency, 4),
            "category":        category,
            "admit_to_cache":  admit,
            "confidence":      round(confidence, 4),
            "response":        response
        }


# -----------------------------------------------------
# Initialize + Routes
# -----------------------------------------------------

cache = TriGuardCache()


@app.post("/ask")
def ask_question(request: QueryRequest):
    return cache.ask(request.query)


@app.get("/stats")
def cache_stats():
    return {
        "chroma_entries": cache.collection.count(),
        "redis_entries":  len(cache.redis.keys("tg:*")) if cache.redis else 0
    }


if __name__ == "__main__":
    uvicorn.run("temp:app", host="0.0.0.0", port=8000, reload=True)
