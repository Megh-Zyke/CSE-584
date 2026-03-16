"""
TriGuard DB Layer — Prudvi's contribution
Integrates ChromaDB (persistent vector store) + Redis (dynamic TTL)
into the existing PyFS semantic cache pipeline.

How it fits into the pipeline:
    Query
      ↓
    [Redis]     → exact hash hit? return instantly (with TTL check)
      ↓ miss
    [ChromaDB]  → semantic search → best candidate query
      ↓ candidate found
    [PyFS]      → contradiction check on (new_query, candidate_query)
      ↓ pyfs passes
    CACHE HIT   → return stored answer
      ↓ pyfs fails or no candidate
    LLM API     → get fresh answer
      ↓
    [Classifier] → Static / Slow-Moving / Volatile
      ↓
    [ChromaDB + Redis] → store with dynamic TTL
"""

import hashlib
import time
import chromadb
import redis
from sentence_transformers import SentenceTransformer

# Import the existing PyFS class from the repo
from pyFS import PyFSSemanticCache


# ─── TTL config (matches the project dataset exactly) ─────────────────────────
TTL_MAP = {
    "static":      -1,         # Infinity — never expires
    "slow-moving": 30 * 24 * 60 * 60,   # 30 days in seconds
    "volatile":    5 * 60,     # 5 minutes in seconds
}

# Keywords used to classify incoming queries (from TTL_Lookup_Guide sheet)
VOLATILE_KEYWORDS = [
    "now", "right now", "today", "currently", "live", "real-time",
    "stock price", "current price", "latest news", "breaking"
]
SLOW_KEYWORDS = [
    "latest version", "current version", "recommended", "policy",
    "regulation", "law", "salary", "pricing", "cost", "fee"
]


def classify_query(query: str) -> str:
    """Classify query into Static / Slow-Moving / Volatile."""
    q = query.lower()
    if any(kw in q for kw in VOLATILE_KEYWORDS):
        return "volatile"
    if any(kw in q for kw in SLOW_KEYWORDS):
        return "slow-moving"
    return "static"


def query_hash(query: str) -> str:
    """Create a short hash key for exact Redis lookup."""
    return "tg:" + hashlib.md5(query.strip().lower().encode()).hexdigest()


# ─── Main Pipeline Class ───────────────────────────────────────────────────────

class TriGuardDBLayer:
    """
    Wraps the existing PyFS pipeline with persistent ChromaDB storage
    and Redis-based dynamic TTL eviction.

    Usage:
        cache = TriGuardDBLayer()
        result = cache.ask("What is the capital of France?", llm_fn=your_llm_call)
    """

    def __init__(
        self,
        chroma_collection: str = "trigard_cache",
        chroma_path: str = "./chroma_store",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        similarity_threshold: float = 0.75,
        pyfs_threshold: float = 0.35,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        print("[TriGuard] Initializing DB Layer...")

        # ── ChromaDB (persistent vector store) ──
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name=chroma_collection,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"[ChromaDB] Connected — {self.collection.count()} entries in cache")

        # ── Redis (TTL fast-path) ──
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        try:
            self.redis.ping()
            print("[Redis] Connected")
        except redis.exceptions.ConnectionError:
            print("[Redis] WARNING: Redis not reachable — TTL layer disabled")
            self.redis = None

        # ── Embedding model (same as test_chroma.py) ──
        print(f"[Embeddings] Loading {embedding_model}...")
        self.embedder = SentenceTransformer(embedding_model)

        # ── PyFS scorer (existing class from pyFS.py) ──
        print("[PyFS] Loading NLI model...")
        self.pyfs = PyFSSemanticCache(threshold=pyfs_threshold)

        self.similarity_threshold = similarity_threshold
        print("[TriGuard] Ready.\n")

    # ─── Public API ────────────────────────────────────────────────────────────

    def ask(self, query: str, llm_fn=None) -> dict:
        """
        Main entry point. Pass any query and an optional LLM function.

        llm_fn: callable that takes a query string and returns an answer string.
                If None, returns a placeholder on cache miss.

        Returns dict with keys:
            answer, source ("redis_hit" | "chroma_hit" | "llm"), category, pyfs_result
        """

        # ── Step 1: Redis exact-match fast path ──────────────────────────────
        redis_result = self._redis_get(query)
        if redis_result:
            print(f"[Redis HIT] Returning cached answer instantly")
            return {
                "answer": redis_result,
                "source": "redis_hit",
                "category": None,
                "pyfs_result": None
            }

        # ── Step 2: ChromaDB semantic search ─────────────────────────────────
        candidate = self._chroma_search(query)

        if candidate:
            cached_query   = candidate["query"]
            cached_answer  = candidate["answer"]
            cached_category = candidate["category"]

            # ── Step 3: PyFS contradiction check ─────────────────────────────
            hit, pyfs_res = self.pyfs.is_cache_hit(query, cached_query)
            print(f"[PyFS] mu={pyfs_res['mu_similarity']:.3f}  "
                  f"nu={pyfs_res['nu_conflict']:.3f}  "
                  f"pi={pyfs_res['pi_hesitation']:.3f}  "
                  f"score={pyfs_res['pyfs_score']:.3f}  hit={hit}")

            if hit:
                print(f"[ChromaDB HIT] Matched: '{cached_query}'")
                # Refresh Redis so next identical query is even faster
                self._redis_set(query, cached_answer, cached_category)
                return {
                    "answer": cached_answer,
                    "source": "chroma_hit",
                    "category": cached_category,
                    "pyfs_result": pyfs_res
                }
            else:
                print(f"[PyFS REJECT] Contradiction detected — going to LLM")
        else:
            print(f"[ChromaDB MISS] No similar entry found — going to LLM")
            pyfs_res = None

        # ── Step 4: LLM call ──────────────────────────────────────────────────
        if llm_fn:
            answer = llm_fn(query)
        else:
            answer = f"[LLM placeholder for: {query}]"

        # ── Step 5: Classify and store in cache ───────────────────────────────
        category = classify_query(query)
        self._store(query, answer, category)

        print(f"[LLM] Answer cached as '{category}'")
        return {
            "answer": answer,
            "source": "llm",
            "category": category,
            "pyfs_result": pyfs_res
        }

    def cache_stats(self) -> dict:
        """Return current cache size info."""
        chroma_count = self.collection.count()
        redis_count = len(self.redis.keys("tg:*")) if self.redis else 0
        return {
            "chroma_entries": chroma_count,
            "redis_entries": redis_count,
        }

    def clear_cache(self):
        """Wipe everything — use carefully."""
        self.chroma_client.delete_collection("trigard_cache")
        self.collection = self.chroma_client.get_or_create_collection(
            name="trigard_cache",
            metadata={"hnsw:space": "cosine"}
        )
        if self.redis:
            for key in self.redis.keys("tg:*"):
                self.redis.delete(key)
        print("[TriGuard] Cache cleared.")

    # ─── Internal helpers ──────────────────────────────────────────────────────

    def _embed(self, text: str):
        return self.embedder.encode(text, normalize_embeddings=True).tolist()

    def _chroma_search(self, query: str) -> dict | None:
        """Search ChromaDB for the most similar cached query."""
        if self.collection.count() == 0:
            return None

        embedding = self._embed(query)
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=1,
            include=["documents", "metadatas", "distances"]
        )

        if not results["ids"][0]:
            return None

        distance = results["distances"][0][0]
        similarity = 1 - distance  # ChromaDB cosine distance → similarity

        if similarity < self.similarity_threshold:
            print(f"[ChromaDB] Best match similarity {similarity:.3f} below threshold {self.similarity_threshold}")
            return None

        return {
            "query":    results["metadatas"][0][0]["original_query"],
            "answer":   results["documents"][0][0],
            "category": results["metadatas"][0][0]["category"],
            "similarity": similarity
        }

    def _store(self, query: str, answer: str, category: str):
        """Store a new entry in ChromaDB and Redis."""
        entry_id = query_hash(query)
        embedding = self._embed(query)

        # ChromaDB — persistent semantic store
        self.collection.upsert(
            ids=[entry_id],
            embeddings=[embedding],
            documents=[answer],
            metadatas=[{
                "original_query": query,
                "category": category,
                "cached_at": time.time()
            }]
        )

        # Redis — TTL fast-path
        self._redis_set(query, answer, category)

    def _redis_get(self, query: str) -> str | None:
        if not self.redis:
            return None
        return self.redis.get(query_hash(query))

    def _redis_set(self, query: str, answer: str, category: str):
        if not self.redis:
            return
        ttl = TTL_MAP.get(category, -1)
        key = query_hash(query)
        if ttl == -1:
            self.redis.set(key, answer)        # No expiry for static
        else:
            self.redis.setex(key, ttl, answer) # Auto-expire for volatile/slow


# ─── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cache = TriGuardDBLayer()

    # Dummy LLM function (replace with real Gemini/OpenAI call)
    def dummy_llm(query):
        answers = {
            "What is the capital of France?": "Paris is the capital of France.",
            "What is the stock price of Apple?": "Apple stock is currently $189.50.",
            "What is Python?": "Python is a popular programming language.",
        }
        return answers.get(query, f"Answer to: {query}")

    print("=" * 60)
    print("TEST 1: First call — should go to LLM (cache miss)")
    r = cache.ask("What is the capital of France?", llm_fn=dummy_llm)
    print(f"Answer: {r['answer']}  |  Source: {r['source']}\n")

    print("=" * 60)
    print("TEST 2: Rephrased query — should hit ChromaDB + PyFS")
    r = cache.ask("Which city is the capital of France?", llm_fn=dummy_llm)
    print(f"Answer: {r['answer']}  |  Source: {r['source']}\n")

    print("=" * 60)
    print("TEST 3: Exact same query — should hit Redis instantly")
    r = cache.ask("What is the capital of France?", llm_fn=dummy_llm)
    print(f"Answer: {r['answer']}  |  Source: {r['source']}\n")

    print("=" * 60)
    print("TEST 4: Volatile query — gets 5-min TTL")
    r = cache.ask("What is the stock price of Apple right now?", llm_fn=dummy_llm)
    print(f"Answer: {r['answer']}  |  Source: {r['source']}  |  Category: {r['category']}\n")

    print("=" * 60)
    print("Cache Stats:", cache.cache_stats())
