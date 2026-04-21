# TriGuard Semantic Cache

A 3-stage, uncertainty-aware semantic cache for GenAI APIs built for CSE 584. TriGuard reduces redundant LLM API calls by intelligently serving cached responses for semantically equivalent queries — while using Pythagorean Fuzzy Sets (PyFS) to catch contradictions that naive cosine similarity misses.

---

## What It Does

Standard semantic caches fail in two ways: they serve stale responses to volatile queries (e.g., stock prices), and they incorrectly match contradictory queries that happen to have similar embeddings (e.g., "increase the dosage" vs "decrease the dosage"). TriGuard addresses both with a 4-stage cascade pipeline:

```
Query
  ↓
[Gate 1 — Context Normalization]   FLAN-T5 resolves pronouns using conversation history
  ↓
[Gate 2 — Dynamic TTL]             BGE-Small + Logistic Regression → Static / Slow-Moving / Volatile
  ↓ (Volatile → skip to Gemini)
[Redis]                            Exact hash lookup with TTL-based eviction
  ↓ miss
[ChromaDB]                         Dense cosine retrieval (BGE-Small embeddings, HNSW index)
  ↓ candidate found
[Reranker + PyFS]        ms-marco reranker + stsb-roberta contradiction detection
  ↓ passes
[Answer Relevance]       ms-marco scores (query, cached response) for relevance
  ↓ cache hit
RETURN CACHED RESPONSE

  ↓ any gate fails
[Gemini API]                       Fresh response generated
  ↓
[Gate 3 — Async SLM Judge]         Qwen via Ollama evaluates reusability + faithfulness + confidence
  ↓ admitted
[ChromaDB + Redis]                 Response stored with TTL
```

### The Verification Gates

| Gate | Model | Role | Threshold |
|------|-------|------|-----------|
| 1 — Cosine | BAAI/bge-base-en-v1.5 | Dense retrieval | > 0.80 |
| 1b — PyFS | stsb-roberta-base | Contradiction detection | nu ≤ 0.30, mu ≥ 0.65 |
| 2 — Rerank | ms-marco-MiniLM-L-6-v2 | Q-Q paraphrase | > 0.70 |
| 3 — Answer | ms-marco-MiniLM-L-6-v2 | Query-passage relevance | > 1.50 |

### Why PyFS Instead of Similarity

NLI/STS models score paraphrase question pairs low on similarity but near-zero on contradiction. TriGuard gates on **absence of contradiction** (low `nu`) rather than presence of similarity (high score) — this is more stable and catches semantic opposites that cosine similarity misses.

```python
score = torch.sigmoid(logits).squeeze().item()
mu = score          # membership — how similar
nu = 1.0 - score    # non-membership — how contradictory
pi = sqrt(1 - mu² - nu²)  # hesitancy — model uncertainty

# Gate passes if: no contradiction AND sufficient similarity
if nu <= 0.30 and mu >= 0.65:
    # cache hit
```

### TTL Classification

Queries are classified into three temporal tiers before cache lookup:

| Category | TTL | Examples |
|----------|-----|---------|
| Static | ∞ (never expires) | Laws of physics, historical facts, legal definitions |
| Slow-Moving | 30 days | Library versions, regulations, clinical guidelines |
| Volatile | 5 minutes | Stock prices, live scores, breaking news |

Volatile queries bypass the cache entirely — they always go to Gemini.

---

## Project Directory

```
.
├── server.py                      
├── evaluate.py                
├── trigard_db_layer.py            
├── requirements.txt               
├── modules/
│   ├── trigaurd.py                
│   ├── semantic_cache.py          
│   ├── dynamic_ttl_training.py    
│   ├── gate1.py                   
│   ├── gate3.py                   
│   ├── query.py                  
│   └── __init__.py
├── triaguard_cache_set.csv        
├── triaguard_test_set.csv         
├── ttl_classifier.joblib         
├── chroma_store/                  
```
---

## Models Used

| Model | Purpose |
|-------|---------|
| `BAAI/bge-small-en-v1.5` | Query embedding for retrieval and TTL classification |
| `cross-encoder/stsb-roberta-base` | PyFS bidirectional NLI scorer |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker + answer relevance gate |
| `google/flan-t5-base` | Gate 1 context-aware query rewriter |
| `qwen2.5:1.5b` (via Ollama) | Gate 4 async SLM judge |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ask` | Submit a query (with optional conversation history) |
| `GET` | `/stats` | Hit rate, source breakdown, LRU cache sizes |
| `GET` | `/health` | Model load status, ChromaDB entry count |
| `GET` | `/gate3/log` | Last 100 Gate 3 admission/rejection decisions |

### Request Format

```json
{
  "query": "What is the speed of light?",
  "history": [
    { "query": "Tell me about Einstein", "response": "Einstein developed relativity..." }
  ]
}
```
## Dependencies

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Ollama is required for Gate 3 (async SLM judge). Install and pull the model:

## Set up

```bash
redis-server --port 6379 --bind 127.0.0.1 &
module load ollama
ollama pull qwen2.5:1.5b
ollama pull llama3.2
ollama serve &
```

```python
python server.py 
#Server runs at localhost:8000
```

```python
python evaluation.py
```
The evaluation scripts sets up the chromaDB and redis and runs inference from the `triaguard_cache_set.csv` and `triaguard_test_set.csv` and saves the necessary logs and results in the directory