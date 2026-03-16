import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import os
import time
import joblib
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


# ─── TTL config ────────────────────────────────────────────────────────────────
TTL_MAP = {
    "Static":       -1,                  # Infinity — never expires
    "Slow-Moving":  30 * 24 * 60 * 60,  # 30 days in seconds
    "Volatile":     5 * 60,             # 5 minutes in seconds
}


def query_hash(query: str) -> str:
    """Short hash key for Redis exact lookup."""
    return "tg:" + hashlib.md5(query.strip().lower().encode()).hexdigest()


# -----------------------------------------------------
# PyFS Semantic Verifier (unchanged)
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
        mu = (mu1 + mu2) / 2
        nu = (nu1 + nu2) / 2
        neutral = (ne1 + ne2) / 2
        pi = np.sqrt(max(0, 1 - (mu**2 + nu**2)))
        pyfs_score = (mu**2) - (nu**2)
        return {
            "score": float(pyfs_score),
            "mu": float(mu),
            "nu": float(nu),
            "pi": float(pi),
            "neutral": float(neutral)
        }


# -----------------------------------------------------
# TriGuard Cache — FAISS replaced with ChromaDB + Redis
# -----------------------------------------------------

class TriGuardCache:

    def __init__(self):

        # ── Embedding + rerankers (unchanged) ──
        print("Loading embedding model...")
        self.embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")

        print("Loading cross-encoder reranker...")
        self.reranker        = CrossEncoder("cross-encoder/quora-roberta-base")
        self.answer_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # ── ChromaDB (replaces FAISS + self.store) ──
        print("Initializing ChromaDB...")
        self.chroma_client = chromadb.PersistentClient(path="./chroma_store")
        self.collection = self.chroma_client.get_or_create_collection(
            name="trigard_cache",
            metadata={"hnsw:space": "cosine"}
        )
        print(f"[ChromaDB] {self.collection.count()} entries loaded from disk")

        # ── Redis (dynamic TTL layer) ──
        print("Connecting to Redis...")
        try:
            self.redis = redislib.Redis(host="localhost", port=6379, decode_responses=True)
            self.redis.ping()
            print("[Redis] Connected")
        except redislib.exceptions.ConnectionError:
            print("[Redis] WARNING: Not reachable — TTL layer disabled")
            self.redis = None

        # ── TTL classifier (Meghanand's trained model) ──
        print("Loading TTL classifier...")
        try:
            self.ttl_classifier = joblib.load("ttl_classifier.joblib")
            print("[TTL Classifier] Loaded successfully")
        except Exception as e:
            print(f"[TTL Classifier] WARNING: Could not load — {e}")
            self.ttl_classifier = None

        # ── Thresholds (unchanged) ──
        self.embedding_threshold        = 0.80
        self.rerank_threshold           = 0.7
        self.contradiction_threshold    = 0.3
        self.answer_relevance_threshold = 1.5

        # ── PyFS verifier (unchanged) ──
        self.verifier = PyFSSemanticCache()

    # -------------------------------------------------
    # TTL Classification
    # -------------------------------------------------

    def classify_ttl(self, query: str) -> str:
        """Use Meghanand's trained classifier to get TTL category."""
        if self.ttl_classifier:
            try:
                label = self.ttl_classifier.predict([query])[0]
                return label  # "Static" / "Slow-Moving" / "Volatile"
            except Exception as e:
                print(f"[TTL] Classifier error: {e} — defaulting to Static")
        return "Static"

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
            self.redis.set(key, response)
        else:
            self.redis.setex(key, ttl, response)

    # -------------------------------------------------
    # ChromaDB helpers
    # -------------------------------------------------

    def _embed(self, query: str):
        return self.embedder.encode(
            [query], normalize_embeddings=True
        ).astype("float32")[0].tolist()

    def _chroma_search(self, query: str, k: int = 5):
        """Search ChromaDB — returns list of (idx, similarity, metadata) tuples."""
        if self.collection.count() == 0:
            return []

        embedding = self._embed(query)
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(k, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        hits = []
        for i in range(len(results["ids"][0])):
            similarity = 1 - results["distances"][0][i]  # cosine distance → similarity
            hits.append({
                "id":       results["ids"][0][i],
                "query":    results["metadatas"][0][i]["original_query"],
                "response": results["documents"][0][i],
                "category": results["metadatas"][0][i].get("category", "Static"),
                "similarity": similarity
            })
        return hits

    def _chroma_store(self, query: str, response: str, category: str):
        """Persist a new entry to ChromaDB."""
        entry_id  = query_hash(query)
        embedding = self._embed(query)

        self.collection.upsert(
            ids=[entry_id],
            embeddings=[embedding],
            documents=[response],
            metadatas=[{
                "original_query": query,
                "category":       category,
                "cached_at":      time.time()
            }]
        )

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
    # Answer Relevance Verification (unchanged)
    # -------------------------------------------------

    def verify_answer_relevance(self, query, response):
        return float(self.answer_reranker.predict([[query, response[:512]]]))

    # -------------------------------------------------
    # Rerank Candidates
    # -------------------------------------------------

    def rerank_candidates(self, query, hits):
        """Rerank ChromaDB hits using the cross-encoder."""
        if not hits:
            return None, None

        pairs  = [[query, h["query"]] for h in hits]
        scores = self.reranker.predict(pairs)

        best_pos   = int(np.argmax(scores))
        best_score = float(scores[best_pos])

        return hits[best_pos], best_score

    # -------------------------------------------------
    # Main Query Handler
    # -------------------------------------------------

    def ask(self, query: str):

        # ── Step 1: Redis exact-match fast path ──────────────────────────────
        redis_result = self._redis_get(query)
        if redis_result:
            print(f"[Redis HIT] Returning instantly from TTL cache")
            return {
                "source":   "REDIS_HIT",
                "response": redis_result
            }

        # ── Step 2: ChromaDB semantic search ─────────────────────────────────
        hits = self._chroma_search(query, k=5)

        if hits:
            best_hit    = hits[0]
            best_sim    = best_hit["similarity"]
            best_query  = best_hit["query"]
            best_resp   = best_hit["response"]
            best_cat    = best_hit["category"]

            print(f"[DEBUG] chroma size  : {self.collection.count()}")
            print(f"[DEBUG] best_sim     : {best_sim:.4f}")
            print(f"[DEBUG] embed thresh : {self.embedding_threshold}")

            # ── FAST PATH ────────────────────────────────────────────────────
            if best_sim > self.embedding_threshold:

                res = self.verifier.calculate_pyfs(query, best_query)

                print(f"[DEBUG] cached_query : {best_query}")
                print(f"[DEBUG] mu={res['mu']:.4f}  nu={res['nu']:.4f}  pi={res['pi']:.4f}")
                print(f"[DEBUG] contradiction: {res['nu']:.4f} <= {self.contradiction_threshold} = {res['nu'] <= self.contradiction_threshold}")

                if res["nu"] <= self.contradiction_threshold:
                    self._redis_set(query, best_resp, best_cat)
                    return {
                        "source":     "CACHE_HIT_FAST",
                        "similarity": best_sim,
                        "mu":         res["mu"],
                        "nu":         res["nu"],
                        "pi":         res["pi"],
                        "response":   best_resp
                    }

            # ── FULL PATH ────────────────────────────────────────────────────
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

        # ── Step 3: Gemini API call ───────────────────────────────────────────
        start    = time.time()
        response = self.call_gemini(query)
        latency  = time.time() - start

        # ── Step 4: Classify TTL and store ───────────────────────────────────
        category = self.classify_ttl(query)
        self._chroma_store(query, response, category)
        self._redis_set(query, response, category)

        print(f"[Gemini] Stored as '{category}' | latency {latency:.3f}s")

        return {
            "source":          "GEMINI_API",
            "latency_seconds": round(latency, 4),
            "category":        category,
            "response":        response
        }


# -----------------------------------------------------
# Initialize Cache
# -----------------------------------------------------

cache = TriGuardCache()


@app.post("/ask")
def ask_question(request: QueryRequest):
    return cache.ask(request.query)


if __name__ == "__main__":
    uvicorn.run(
        "temp:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
