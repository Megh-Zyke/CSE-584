import os
import hashlib
import uvicorn
from concurrent.futures import ThreadPoolExecutor

from ollama import AsyncClient
from google import genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from modules.trigaurd import TriGuardCache
from modules.query import QueryRequest

load_dotenv()

API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment")

client = genai.Client(api_key=API_KEY)

app = FastAPI(title="Tri-Guard Semantic Cache API", version="5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_executor = ThreadPoolExecutor(max_workers=4)

TTL_MAP = {
    "Static":      -1,
    "Slow-Moving": 30 * 24 * 60 * 60,
    "Volatile":    5 * 60,
}


def query_hash(query: str) -> str:
    return "tg:" + hashlib.md5(query.strip().lower().encode()).hexdigest()


cache = TriGuardCache()


@app.post("/ask")
async def ask_question(request: QueryRequest):
    return await cache.ask(request.query, request.history)


@app.get("/stats")
def cache_stats():
    redis_count = 0
    if cache.redis:
        try:
            cursor, keys = cache.redis.scan(match="tg:*", count=100)
            redis_count = len(keys)
        except Exception:
            redis_count = -1

    total = max(1, cache.metrics["total_queries"])
    cache_hits = (
        cache.metrics["redis_hits"] +
        cache.metrics["chroma_hits_fast"] +
        cache.metrics["chroma_hits_verified"]
    )

    return {
        "chroma_entries":        cache.collection.count(),
        "redis_keys":            redis_count,
        "embed_lru_entries":     len(cache._embed_cache),
        "rerank_lru_entries":    len(cache._rerank_cache),
        "pyfs_lru_entries":      len(cache.verifier._cache),
        "metrics":               cache.metrics,
        "hit_rate_percent":      round(cache_hits / total * 100, 2),
        "api_reduction_percent": round(
            (1 - cache.metrics["gemini_calls"] / total) * 100, 2
        )
    }


@app.get("/health")
def health():
    return {
        "status":         "ok",
        "ttl_classifier": cache.ttl_clf is not None,
        "redis":          cache.redis is not None,
        "chroma_entries": cache.collection.count(),
        "models": {
            "embedder":       "bge-small-en-v1.5  (384-dim, cache vectors)",
            "ttl_classifier": "bge-small-en-v1.5 (384-dim, TTL classification)",
            "reranker":       "ms-marco-MiniLM-L-6-v2",
            "pyfs_verifier":  "stsb-roberta-base (low-sim path only)",
            "gate3":          "qwen2.5:1.5b via Ollama (confidence + faithfulness)",
        }
    }


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)