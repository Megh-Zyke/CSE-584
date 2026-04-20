import re
import time
import asyncio
import requests
from concurrent.futures import ThreadPoolExecutor


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"
CONFIDENCE_PROMPT = """You are a cache quality evaluator.

Query: "{query}"
Response: "{response}"

Should this response be cached for future reuse?

A response is worth caching ONLY if:
- it is factually accurate
- it is complete
- it is stable over time
- it directly answers the query

If the response asks for clarification or does not provide a concrete answer, return 0.0.

Output ONLY a number between 0.0 and 1.0.
"""

FAITHFULNESS_PROMPT = """You are a cache quality evaluator.

Query: "{query}"
Response: "{response}"

Does this response faithfully and directly answer the query?

A response is ONLY faithful if:
- it directly answers the query
- it does not hallucinate or speculate
- it provides useful information

If the response asks for clarification or does not answer the query, return 0.0.

Output ONLY a number between 0.0 and 1.0.
"""

REUSABILITY_PROMPT = """You are a cache quality evaluator.

Query: "{query}"
Response: "{response}"

Would this response be useful for future users asking the SAME query?

Reject responses that:
- ask for clarification
- are generic
- do not provide a concrete answer

Return a score from 0.0 to 1.0.

Output ONLY a number.
"""


# ── Gate3 Class ──────────────────────────────────────────────────────────────
class Gate3:
    def __init__(self, threshold=0.75):
        """
        Combined score:
        0.5 * reusability + 0.3 * confidence + 0.2 * faithfulness
        """
        self.threshold = threshold
        self._executor = ThreadPoolExecutor(max_workers=3)

        print("[Gate3] Ready — triple signal mode (reusability + confidence + faithfulness)")

    # ── Heuristic filters (FAST + IMPORTANT) ──────────────────────────────────
    def _fails_basic_cacheability(self, response: str) -> bool:
        r = response.lower()

        patterns = [
            "please provide more context",
            "i need more information",
            "could you clarify",
            "can you clarify",
            "it depends",
            "unclear",
            "which one do you mean",
        ]

        if any(p in r for p in patterns):
            return True

        return False

    # ── Ollama caller ─────────────────────────────────────────────────────────
    def _call_ollama(self, prompt: str, label: str):
        t0 = time.perf_counter()
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0},
                },
                timeout=30,  # reduced timeout
            )
            resp.raise_for_status()

            raw = resp.json().get("response", "").strip()
            print(f"[Gate3] {label}: raw='{raw}'")

            score = self._parse_float(raw)

            if score is None:
                print(f"[Gate3] {label}: parse failed")
                return 0.0, time.perf_counter() - t0

            return score, time.perf_counter() - t0

        except Exception as e:
            print(f"[Gate3] {label} failed: {e}")
            return None, time.perf_counter() - t0

    # ── Parsing ───────────────────────────────────────────────────────────────
    @staticmethod
    def _parse_float(text: str):
        match = re.search(r"\b(1\.0+|0?\.\d+)\b", text)
        if match:
            return max(0.0, min(1.0, float(match.group())))
        return None

    # ── Signal wrappers ───────────────────────────────────────────────────────
    def _confidence(self, query, response):
        prompt = CONFIDENCE_PROMPT.format(query=query, response=response)
        return self._call_ollama(prompt, "confidence")

    def _faithfulness(self, query, response):
        prompt = FAITHFULNESS_PROMPT.format(query=query, response=response)
        return self._call_ollama(prompt, "faithfulness")

    def _reusability(self, query, response):
        prompt = REUSABILITY_PROMPT.format(query=query, response=response)
        return self._call_ollama(prompt, "reusability")

    # ── Async wrappers ────────────────────────────────────────────────────────
    async def _run_async(self, func, loop, query, response):
        return await loop.run_in_executor(self._executor, func, query, response)

    # ── Main check ────────────────────────────────────────────────────────────
    async def check_async(self, query: str, response: str):
        loop = asyncio.get_event_loop()

        # 🚫 HARD FILTER (no LLM call)
        if self._fails_basic_cacheability(response):
            return {
                "reusability_score": 0.0,
                "confidence_score": 0.0,
                "faithfulness_score": 0.0,
                "combined_score": 0.0,
                "admit_to_cache": False,
                "mode": "blocked: heuristic"
            }

        # parallel execution
        (r_score, _), (c_score, _), (f_score, _) = await asyncio.gather(
            self._run_async(self._reusability, loop, query, response),
            self._run_async(self._confidence, loop, query, response),
            self._run_async(self._faithfulness, loop, query, response),
        )

        # fallback handling
        r_score = r_score if r_score is not None else 0.0
        c_score = c_score if c_score is not None else 0.0
        f_score = f_score if f_score is not None else 0.0

        combined = (
            0.5 * r_score +
            0.3 * c_score +
            0.2 * f_score
        )

        return {
            "reusability_score": round(r_score, 4),
            "confidence_score": round(c_score, 4),
            "faithfulness_score": round(f_score, 4),
            "combined_score": round(combined, 4),
            "admit_to_cache": combined >= self.threshold,
            "mode": "triple-signal"
        }

    # ── Sync wrapper ──────────────────────────────────────────────────────────
    def check(self, query: str, response: str):
        return asyncio.run(self.check_async(query, response))

if __name__ == "__main__":
    gate = Gate3(threshold=0.75)

    # Test cases
    test_cases = [
        {
            "query": "What is the capital of France?",
            "response": "The capital of France is Paris."
        },
        {
            "query": "Explain quantum computing",
            "response": "It depends on what aspect you are interested in. Could you clarify?"
        },
        {
            "query": "Who won the FIFA World Cup 2018?",
            "response": "France won the FIFA World Cup in 2018."
        },
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n=== Test Case {i} ===")
        print(f"Query: {case['query']}")
        print(f"Response: {case['response']}")

        result = gate.check(case["query"], case["response"])

        print("\nResult:")
        for k, v in result.items():
            print(f"{k}: {v}")