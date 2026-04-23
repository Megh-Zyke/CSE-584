TriGuard Semantic Cache
CSE 584 — University of Michigan
Team: Aman Soni, Meghanand Gejjela, Prudvi Sai Padidala
==========================================================

==========================================================
SECTION 1: USAGE
==========================================================

STARTING THE SERVER
-------------------
Ensure Redis is running and Ollama is serving qwen2.5:1.5b before
starting the server. See INSTALL.txt for setup instructions.

    # Terminal 1 — Redis
    redis-server --port 6379 --bind 127.0.0.1 &

    # Terminal 2 — Ollama
    ollama serve &

    # Terminal 3 — TriGuard server
    python3 server.py

The server runs at http://localhost:8000.


API ENDPOINTS
-------------

POST /ask
    Submit a query to the semantic cache.

    Request body (JSON):
        {
            "query": "What is the speed of light?",
            "history": [
                {
                    "query": "Tell me about Einstein",
                    "response": "Einstein developed the theory of relativity..."
                }
            ]
        }

    The "history" field is optional. When provided (up to last 3 turns),
    Gate 1 uses it to resolve pronouns and ambiguous references before
    embedding the query.

    Example using curl:
        curl -X POST http://localhost:8000/ask \
             -H "Content-Type: application/json" \
             -d '{"query": "What is the speed of light?"}'

    Example with context:
        curl -X POST http://localhost:8000/ask \
             -H "Content-Type: application/json" \
             -d '{
               "query": "When was it founded?",
               "history": [
                 {"query": "Tell me about Google", "response": "Google was founded in 1998..."}
               ]
             }'

    Response fields:
        source          — One of: CACHE_HIT_FAST, CACHE_HIT_VERIFIED,
                          REDIS_HIT, GEMINI_API, GEMINI_VOLATILE
        response        — The answer text
        latency_seconds — End-to-end latency
        category        — TTL category (Static / Slow-Moving / Volatile)
        similarity      — Cosine similarity to cached query (if cache hit)
        mu, nu, pi      — PyFS scores (if full verification path taken)


GET /stats
    Returns real-time cache statistics.

        curl http://localhost:8000/stats

    Response includes:
        chroma_entries       — Number of entries in ChromaDB
        redis_keys           — Number of keys in Redis
        hit_rate_percent     — Overall cache hit rate since startup
        api_reduction_percent — Fraction of queries served without LLM call
        metrics              — Full breakdown: redis_hits, chroma_hits_fast,
                               chroma_hits_verified, volatile_bypasses,
                               gemini_calls, gate3_admitted, gate3_blocked,
                               gate1_rewrites


GET /health
    Returns model load status and ChromaDB entry count.

        curl http://localhost:8000/health

    Response includes which models are loaded (embedder, reranker,
    pyfs_verifier, gate3) and whether Redis and the TTL classifier
    are available.


GET /gate3/log
    Returns the last 100 Gate 3 admission/rejection decisions.

        curl http://localhost:8000/gate3/log


RUNNING THE EVALUATION
----------------------
With the server running:

    python3 evaluate.py

The script performs two phases:
  Phase 1: Sends 150 queries from triaguard_cache_set.csv to populate
           the ChromaDB and Redis caches.
  Phase 2: Sends 200 test queries from triaguard_test_set.csv and
           measures hit rate, staleness, Gate 3 integrity, and latency.

Output files (saved to project root):
    evaluation_results_<timestamp>.csv     — Per-query results
    evaluation_summary_<timestamp>.json    — Aggregated metrics
    wrong_cache_hits_<timestamp>.csv       — Cache hits scored below
                                             0.70 cosine vs ground truth

TRAINING THE TTL CLASSIFIER (OPTIONAL)
---------------------------------------
A pre-trained classifier (ttl_classifier.joblib) is included.
To retrain from scratch:

    python3 modules/dynamic_ttl_training.py \
        --train datasets/synthetic_ttl_training_dataset.xlsx

To classify a single query:

    python3 modules/dynamic_ttl_training.py \
        --query "What is the current stock price of Apple?"

To run batch classification from a text file (one query per line):

    python3 modules/dynamic_ttl_training.py \
        --batch my_queries.txt

SMALL DATASET FOR QUICK TESTING
---------------------------------
The following small set of queries can be used to verify the system
is working without running the full evaluation:

    # Static query — should be cached after first call
    curl -X POST http://localhost:8000/ask \
         -H "Content-Type: application/json" \
         -d '{"query": "What is the Pythagorean theorem?"}'

    # Send paraphrase — should return a cache hit
    curl -X POST http://localhost:8000/ask \
         -H "Content-Type: application/json" \
         -d '{"query": "Explain the Pythagorean theorem in math."}'

    # Volatile query — should bypass cache entirely
    curl -X POST http://localhost:8000/ask \
         -H "Content-Type: application/json" \
         -d '{"query": "What is the current stock price of TSLA?"}'

    # Check stats after the above three queries
    curl http://localhost:8000/stats


==========================================================
SECTION 2: SOURCE CODE OVERVIEW
==========================================================

PROJECT STRUCTURE
-----------------
CSE-584-main/
├── server.py                     — FastAPI server and API endpoint definitions
├── evaluate.py                   — Two-phase evaluation script
├── trigard_db_layer.py           — Early prototype DB layer (not used in evaluation)
├── requirements.txt              — Pinned Python dependencies
├── ttl_classifier.joblib         — Pre-trained TTL classifier weights
├── triaguard_cache_set.csv       — 150 queries used for Phase 1 cache population
├── triaguard_test_set.csv        — 200 queries used for Phase 2 evaluation
├── datasets/
│   ├── synthetic_ttl_training_dataset.xlsx  — Training data for TTL classifier
│   ├── evaluation_dataset.xlsx             — Full labeled evaluation dataset
│   └── training.xlsx                       — Additional training data
└── modules/
    ├── trigaurd.py               — Core pipeline: TriGuardCache class
    ├── gate1.py                  — Gate 1: Context normalization (FLAN-T5)
    ├── gate3.py                  — Gate 3: Async SLM verification (Qwen)
    ├── dynamic_ttl_training.py   — TTL classifier training and inference
    ├── semantic_cache.py         — PyFS contradiction detection (stsb-roberta)
    ├── context_normalizer.py     — Alternative Ollama-based context normalizer
    ├── query.py                  — Pydantic request/response models
    └── __init__.py


MAJOR FUNCTIONS AND WHERE TO FIND THEM
---------------------------------------

1. MAIN QUERY PIPELINE
   File: modules/trigaurd.py
   Class: TriGuardCache
   Method: ask() — line 419

   This is the central orchestrator. For every incoming query it:
   (a) Passes the query through Gate 1 (pronoun resolution)
   (b) Classifies the query into Static/Slow-Moving/Volatile via Gate 2
   (c) Bypasses cache if Volatile; otherwise runs Redis exact-match lookup
   (d) On Redis miss, runs ChromaDB semantic search with pre-computed embedding
   (e) Applies fast-path (cosine > dynamic threshold) or full-path
       (reranker → PyFS → answer relevance) verification
   (f) On cache miss, calls GPT-4o and asynchronously runs Gate 3


2. GATE 1 — CONTEXT NORMALIZATION
   File: modules/gate1.py
   Class: ContextGate
   Method: rewrite_query() — line 20

   Uses google/flan-t5-base to rewrite ambiguous queries using the
   last N turns of conversation history. A heuristic pre-filter
   (_needs_rewriting(), line 13) scans for pronoun signals (it, he,
   she, they, that, this, etc.) before invoking the T5 forward pass,
   saving latency when no rewrite is needed.


3. GATE 2 — TTL TEMPORAL CLASSIFICATION
   File: modules/dynamic_ttl_training.py
   Class: TTLClassifier
   Methods:
     train()   — line 182  (trains logistic regression on BGE embeddings)
     predict() — line 315  (classifies a single query, returns TTLResult)
     load()    — line 296  (loads ttl_classifier.joblib from disk)

   The classifier encodes queries with BAAI/bge-small-en-v1.5 (384-dim,
   ONNX-accelerated) and feeds the embedding into a logistic regression
   model trained with 80/20 stratified split and 5-fold cross-validation.
   Confidence below 0.40 triggers a safe fallback to Volatile.

   TTL values used by the live cache (modules/trigaurd.py lines 28-31):
     Static:      7 days  (7 * 24 * 60 * 60 seconds)
     Slow-Moving: 3 days  (3 * 24 * 60 * 60 seconds)
     Volatile:    5 minutes (5 * 60 seconds) — bypasses cache entirely


4. REDIS EXACT-MATCH CACHE
   File: modules/trigaurd.py
   Methods: _redis_get() — line 172, _redis_set() — line 177

   Uses MD5 hash of the normalized query (prefix "tg:") as the Redis key.
   TTLs are enforced via Redis's setex command. Falls back silently if
   Redis is unavailable.


5. CHROMADB SEMANTIC SEARCH
   File: modules/trigaurd.py
   Methods:
     _embed()                        — line 191  (BGE embedding with LRU cache)
     _chroma_search_with_embedding() — line 234  (HNSW cosine search, k=5)
     _chroma_store()                 — line 256  (upsert to ChromaDB)
     _evict_if_needed()              — line 330  (evicts oldest 10% when >1000 entries)
     _dynamic_threshold()            — line 319  (returns 0.70/0.75/0.80 based on load)

   Embedding results are cached in an application-layer LRU dict
   (_embed_cache, size 1024) to avoid re-running the encoder forward pass.


6. RERANKER
   File: modules/trigaurd.py
   Method: rerank_candidates() — line 281

   Uses cross-encoder/ms-marco-MiniLM-L-6-v2 to score each of the k=5
   ChromaDB candidates against the query. Results are cached in an LRU
   dict (_rerank_cache, size 1024) to avoid rescoring repeated pairs.
   Only candidates with rerank score > 0.70 proceed to PyFS verification.


7. PYFS CONTRADICTION DETECTION
   File: modules/semantic_cache.py
   Class: PyFSSemanticCache
   Method: calculate_pyfs() — line 24

   Uses cross-encoder/stsb-roberta-base to run bidirectional NLI
   scoring between the incoming query and the cached query. Maps the
   sigmoid logit to Pythagorean Fuzzy Set parameters:
     mu = sigma(logit)       — degree of semantic agreement
     nu = 1 - sigma(logit)  — degree of contradiction
     pi = sqrt(1 - mu^2 - nu^2) — hesitancy/uncertainty

   A cache hit is granted only when nu <= 0.30 AND answer relevance > 1.50.
   Results cached in LRU dict (_cache, size 512).

   Answer relevance is verified by:
   File: modules/trigaurd.py
   Method: verify_answer_relevance() — line 342
   Uses ms-marco-MiniLM-L-6-v2 to score (query, cached_response) relevance.


8. GATE 3 — ASYNC SLM VERIFICATION
   File: modules/gate3.py
   Class: Gate3
   Methods:
     check_async() — line 151  (main async entry point)
     _reusability() — line 142, _confidence() — line 134,
     _faithfulness() — line 138

   Uses qwen2.5:1.5b via local Ollama to evaluate fresh LLM responses
   across three parallel signals before storing them in the cache:
     Reusability (weight 0.5): Would future users benefit from this?
     Confidence  (weight 0.3): Is the response factually accurate?
     Faithfulness (weight 0.2): Does it directly answer without hallucinating?

   Combined score = 0.5*reusability + 0.3*confidence + 0.2*faithfulness
   Gate 3 is launched via asyncio.create_task() AFTER the user has
   received their response, adding zero user-facing latency.

   A heuristic pre-filter (_fails_basic_cacheability(), line 76) blocks
   responses containing phrases like "please provide more context" or
   "it depends" without invoking the SLM.


9. FASTAPI SERVER AND ENDPOINTS
   File: server.py
   Endpoints:
     POST /ask       — line 76  (calls cache.ask())
     GET  /gate3/log — line 80  (returns last 100 Gate 3 decisions)
     GET  /stats     — line 84  (returns cache metrics and hit rates)
     GET  /health    — line 115 (returns model load status)

   The server is initialized at line 44. The TriGuardCache instance
   (cache) is created at line 71 and shared across all requests.


10. EVALUATION SCRIPT
    File: evaluate.py
    Function: run_evaluation() — line 116

    Phase 1 (lines 136-172): Reads triaguard_cache_set.csv and sends
    150 queries to /ask to populate the cache. Saves per-query results
    to phase1_results_<timestamp>.csv.

    Phase 2 (lines 174-217): Reads triaguard_test_set.csv and sends
    200 test queries to /ask. Records source (cache hit vs LLM call),
    predicted TTL category, latency, and cached response.

    Phase 3 (lines 219-330): Computes all metrics: effective hit rate,
    staleness rate, Gate 3 integrity, latency speedup, and TTL accuracy.
    Also runs wrong-hit analysis using all-MiniLM-L6-v2 to compare
    cached responses against ground-truth reference answers.


