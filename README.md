# TriGuard Semantic Cache CSE 584 Project

A 4-stage semantic cache for GenAI APIs using Pythagorean Fuzzy Sets (PyFS) for uncertainty-aware verification.

---

## Pipeline

```
Query → [BGE cosine] → [PyFS contradiction] → [Quora reranker] → [PyFS contradiction] → [ms-marco answer] → CACHE_HIT
                                                                                                            ↓ (any gate fails)
                                                                                                        Gemini API
```

| Gate | Model | Task | Threshold |
|------|-------|------|-----------|
| 1 — cosine | BAAI/bge-base-en-v1.5 | Dense retrieval | > 0.82 |
| 1b — PyFS | stsb-roberta-base | Contradiction detection | nu <= 0.35, mu >= 0.65 |
| 2 — rerank | quora-roberta-base | Q-Q paraphrase | > 0.7 |
| 3 — PyFS | stsb-roberta-base | Contradiction check | nu <= 0.35, mu >= 0.65 |
| 4 — answer | ms-marco-MiniLM-L-6-v2 | Query-passage relevance | > 1.5 |

---

## PyFS

```python
score = torch.sigmoid(logits).squeeze().item()  # stsb regression output
mu = score          # membership — how similar
nu = 1.0 - score    # non-membership — how different
pi = sqrt(1 - mu² - nu²)  # hesitancy — uncertainty

# Gate condition — contradiction detector, not similarity detector
if nu <= 0.35 and mu >= 0.65:  # passes
```

**Why contradiction detection and not similarity:** NLI/STS models score question-question pairs low even for paraphrases. But `nu` (contradiction) stays near zero on valid paraphrase pairs. Gating on absence of contradiction is stable; gating on presence of similarity is not.

**pi is currently unused** — when pi > 0.5 the model is uncertain. Future: async LLM judge on high-pi pairs (Krites 2026).

---

## Why each model

| Model | Why |
|-------|-----|
| bge-base-en-v1.5 | Dense retrieval embeddings. Normalised → cosine via dot product in FAISS. |
| stsb-roberta-base | Trained on STS-B + QQP. Regression output maps cleanly to mu/nu. |
| quora-roberta-base | 400k Quora Question Pairs. Purpose-built for Q-Q paraphrase detection. |
| ms-marco-MiniLM-L-6-v2 | MS MARCO passage retrieval. Correct task for (query, response) Gate 4. |

---

## Compared to published work

| System | Verification | Uncertainty | Threshold | Eviction |
|--------|-------------|-------------|-----------|----------|
| GPTCache 2023 | 1 — cosine | None | Fixed | LRU |
| MeanCache 2024 | 1 — cosine | None | Fixed | LRU |
| LangCache 2025 | 1 — fine-tuned embed | None | Fixed per domain | LRU + TTL |
| SISO Aug 2025 | 1 — centroid cosine | None | **Dynamic** | Semantic LFU |
| Liu et al. Aug 2025 | 1 — cost-aware | Mismatch cost | **Bandit-learned** | Bandit |
| Krites Feb 2026 | 2 — cosine + async LLM | None | Fixed + grey zone | Static |
| SphereLFU Feb 2026 | 1 — cosine | None | Fixed | **KDE prototype** |
| **TriGuard v4.3** | **4 — cascade** | **PyFS mu/nu/pi** | 3 fixed | None yet |

**Unique to TriGuard:** 4-stage cascade, PyFS uncertainty quantification, answer-level gate.  
**Gaps:** no eviction, fixed thresholds, no persistence, no metrics.

---

## Future Roadmap

### P1 — SISO dynamic threshold (~20 lines)
```python
from collections import deque

self._api_call_times = deque()

def _dynamic_threshold(self):
    now = time.time()
    while self._api_call_times and now - self._api_call_times[0] > 60:
        self._api_call_times.popleft()
    rate = len(self._api_call_times)
    return 0.78 if rate > 10 else 0.88 if rate < 3 else 0.82

# Before Gemini call:
self._api_call_times.append(time.time())
```

### P2 — SphereLFU eviction
```python
import math

def _sphere_lfu_score(self, idx):
    e = self.store[idx]
    age_hours = (time.time() - e["cached_at"]) / 3600
    return e.get("hits", 0) / (1 + math.log1p(age_hours))

def _evict_if_needed(self, max_size=5000):
    if len(self.store) >= max_size:
        evict_idx = int(np.argmin([self._sphere_lfu_score(i) for i in range(len(self.store))]))
        self.store.pop(evict_idx)
        self._rebuild_index()
```

### P3 — Krites async LLM judge on high-pi pairs
```python
# After PyFS, before returning miss:
if res["pi"] > 0.5:
    threading.Thread(
        target=self._async_judge,
        args=(query, cached_query, self.store[best_idx]["response"]),
        daemon=True
    ).start()

def _async_judge(self, query, cached_query, cached_response):
    prompt = f"Q1: {query}\nQ2: {cached_query}\nSame answer? YES or NO."
    if "YES" in self.call_gemini(prompt).upper():
        # Register as alias for future hits
        vec = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        self.index.add(vec)
        self.store.append({"original_query": query, "response": cached_response, "cached_at": time.time(), "hits": 0})
```

### P4 — Liu mismatch cost by query type
```python
def _risk_level(self, query):
    high_risk = ["dosage", "mg", "diagnosis", "legal", "how many", "what year", "who is the current"]
    return "high" if any(s in query.lower() for s in high_risk) else "normal"

# In ask():
risk = self._risk_level(query)
contra_thresh = 0.15 if risk == "high" else self.contradiction_threshold
mu_floor = 0.80 if risk == "high" else 0.65
```

### P5 — LangCache fine-tuning (future)
Once real traffic accumulates, fine-tune BGE using contrastive loss on cache hit pairs (positives) and miss pairs (hard negatives). Makes `embedding_threshold` a stable constant instead of a hand-tuned number.

---

## Debug

Add to `ask()` when hits are missing:

```python
print(f"[DEBUG] index={self.index.ntotal} sim={best_sim:.4f} thresh={self.embedding_threshold}")
print(f"[DEBUG] mu={res['mu']:.4f} nu={res['nu']:.4f} pi={res['pi']:.4f}")
print(f"[DEBUG] nu check: {res['nu']:.4f} <= {self.contradiction_threshold} = {res['nu'] <= self.contradiction_threshold}")
```

| Output | Cause | Fix |
|--------|-------|-----|
| `index=1, sim=1.0` | Second query is identical string | Send different phrasing |
| `sim=0.84xx, thresh=0.85` | Cosine just below threshold | Lower `embedding_threshold` to 0.82 |
| `nu=0.28xx <= 0.25 = False` | Contradiction threshold too tight | Raise `contradiction_threshold` to 0.35 |
| `mu=0.63xx >= 0.65 = False` | mu floor too high | Lower mu floor to 0.60 |

---

## Known gaps

- No `threading.Lock()` — race condition on concurrent requests
- No eviction — memory grows unbounded (implement P2)
- No persistence — cache resets on restart (`pickle` + `faiss.write_index`)
- No metrics endpoint — no visibility into hit rate or latency
- Fixed thresholds — implement P1 for workload-adaptive behaviour