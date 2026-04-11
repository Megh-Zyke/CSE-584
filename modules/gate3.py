import re
import time
import asyncio
import requests
from concurrent.futures import ThreadPoolExecutor


# ── Ollama config ────────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:3b"

CONFIDENCE_PROMPT = """You answered this query: "{query}"
Your answer was: "{response}"

How confident are you this answer is factually correct and complete?
Reply with ONLY a single float between 0.0 and 1.0. Nothing else."""

FAITHFULNESS_PROMPT = """Query: "{query}"
Response: "{response}"

Does the response directly and faithfully answer the query without
introducing unsupported or speculative claims?
Reply with ONLY a single float between 0.0 (unfaithful) and 1.0 (faithful). Nothing else."""


class Gate3:
    def __init__(self, threshold: float = 0.5):
        """
        Both signals equally weighted: combined = 0.5 * confidence + 0.5 * faithfulness
        threshold: minimum combined score to admit response to cache
        """
        self.threshold = threshold
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
        wall_t0 = time.perf_counter()

        # both Ollama calls fire at the same time
        (confidence_score, conf_latency), (faithfulness_score, faith_latency) = \
            await asyncio.gather(
                self._confidence_async(loop, query, response),
                self._faithfulness_async(loop, query, response),
            )

        wall_latency = time.perf_counter() - wall_t0

        # scoring — handle partial failures gracefully
        if confidence_score is not None and faithfulness_score is not None:
            combined = 0.5 * confidence_score + 0.5 * faithfulness_score
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
            "admit_to_cache":     combined >= self.threshold,
            "mode":               mode,
        }

    # ── Sync convenience wrapper ──────────────────────────────────────────────
    def check(self, query: str, response: str) -> dict:
        return asyncio.run(self.check_async(query, response))