import re
import time
import asyncio
import requests
from concurrent.futures import ThreadPoolExecutor


# ── Ollama config ────────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"

CONFIDENCE_PROMPT = """You are a cache quality evaluator.

Query: "{query}"
Response: "{response}"

Should this response be cached for future reuse?
A response is worth caching if it is factually accurate, complete, and unlikely to change over time.

Rate the caching confidence from 0.0 to 1.0.
0.0 = should NOT be cached (wrong, incomplete, or time-sensitive)
1.0 = should definitely be cached (accurate, complete, stable)

Output only a single number like 0.8"""

FAITHFULNESS_PROMPT = """You are a cache quality evaluator.

Query: "{query}"
Response: "{response}"

Does this response faithfully and directly answer the query without unsupported claims?
A faithful response is safe to cache and serve to future users asking the same question.

Rate the faithfulness for caching from 0.0 to 1.0.
0.0 = unfaithful (hallucinated, speculative, or off-topic — do NOT cache)
1.0 = fully faithful (directly answers the query — safe to cache)

Output only a single number like 0.8"""

class Gate3:
    def __init__(self, threshold=0.64, confidence_weight=0.6, faithfulness_weight=0.4):
        """
        Both signals equally weighted: combined = 0.5 * confidence + 0.5 * faithfulness
        threshold: minimum combined score to admit response to cache
        """
        self.threshold = threshold
        self.confidence_weight = confidence_weight
        self.faithfulness_weight = faithfulness_weight
        # two threads — one per Ollama call, both fire in parallel

        self._executor = ThreadPoolExecutor(max_workers=2)
        print("[Gate3] Ready — dual SLM mode (confidence + faithfulness)")

    # ── Shared Ollama caller ──────────────────────────────────────────────────
    def _call_ollama(self, prompt: str, label: str) -> tuple[float | None, float]:
        t0 = time.perf_counter()
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model":   OLLAMA_MODEL,
                    "prompt":  prompt,
                    "stream":  False,
                    "options": {"temperature": 0.0},
                },
                timeout=30,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "").strip()
            print(f"[Gate3] {label}: raw='{raw}'")  # ← add here
            score = self._parse_float(raw)
            if score is None:
                print(f"[Gate3] {label}: could not parse float from '{raw}'")
            return score, time.perf_counter() - t0

        except requests.exceptions.ConnectionError:
            print("[Gate3] Ollama not reachable — is it running? (`ollama serve`)")
            return None, time.perf_counter() - t0
        except Exception as e:
            print(f"[Gate3] {label} call failed: {e}")
            return None, time.perf_counter() - t0

    def _get_confidence_sync(self, query: str, response: str) -> tuple[float | None, float]:
        prompt = CONFIDENCE_PROMPT.format(query=query, response=response)
        return self._call_ollama(prompt, "confidence")

    def _get_faithfulness_sync(self, query: str, response: str) -> tuple[float | None, float]:
        prompt = FAITHFULNESS_PROMPT.format(query=query, response=response)
        return self._call_ollama(prompt, "faithfulness")

    @staticmethod
    def _parse_float(text: str) -> float | None:
        match = re.search(r"\b(1\.0+|0?\.\d+)\b", text)
        if match:
            return max(0.0, min(1.0, float(match.group())))
        return None

    # ── Async wrappers ────────────────────────────────────────────────────────
    async def _confidence_async(self, loop, query: str, response: str):
        return await loop.run_in_executor(
            self._executor, self._get_confidence_sync, query, response
        )

    async def _faithfulness_async(self, loop, query: str, response: str):
        return await loop.run_in_executor(
            self._executor, self._get_faithfulness_sync, query, response
        )

    # ── Parallel check ────────────────────────────────────────────────────────
    async def check_async(self, query: str, response: str) -> dict:
        loop    = asyncio.get_event_loop()

        # both Ollama calls fire at the same time
        (confidence_score, _), (faithfulness_score, _) = \
            await asyncio.gather(
                self._confidence_async(loop, query, response),
                self._faithfulness_async(loop, query, response),
            )



        # scoring — handle partial failures gracefully
        if confidence_score is not None and faithfulness_score is not None:
            combined = self.confidence_weight * confidence_score + self.faithfulness_weight * faithfulness_score
            mode     = "confidence+faithfulness (parallel)"
        elif confidence_score is not None:
            combined = confidence_score
            mode     = "confidence_only (faithfulness unavailable)"
        elif faithfulness_score is not None:
            combined = faithfulness_score
            mode     = "faithfulness_only (confidence unavailable)"
        else:
            # both failed — fail safe, do not cache
            combined = 0.0
            mode     = "failed (ollama unavailable) — blocking cache write"

        return {
            "confidence_score":   round(confidence_score,   4) if confidence_score   is not None else None,
            "faithfulness_score": round(faithfulness_score, 4) if faithfulness_score is not None else None,
            "combined_score":     round(combined, 4),
            "admit_to_cache": combined >= self.threshold,
            "mode": mode,
        }

    # ── Sync convenience wrapper ──────────────────────────────────────────────
    def check(self, query: str, response: str) -> dict:
        return asyncio.run(self.check_async(query, response))