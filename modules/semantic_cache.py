import numpy as np
import torch 
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class PyFSSemanticCache:

    def __init__(self, model_name="cross-encoder/stsb-roberta-base"):
        print("Initializing PyFS STS Verifier...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        # FIX: LRU cache for PyFS results — saves 200ms per repeated pair
        self._cache: dict[tuple, dict] = {}
        self._cache_keys: list[tuple]  = []
        self._cache_size = 512

    def _nli_probs(self, a, b):
        inputs = self.tokenizer(a, b, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        score = float(torch.sigmoid(logits).cpu().numpy().flatten()[0])
        return score, 1.0 - score, 0.0

    def calculate_pyfs(self, q1, q2):
        # FIX: check cache before running expensive bidirectional NLI
        cache_key = (q1, q2)
        if cache_key in self._cache:
            return self._cache[cache_key]

        mu1, nu1, _ = self._nli_probs(q1, q2)
        mu2, nu2, _ = self._nli_probs(q2, q1)
        mu      = (mu1 + mu2) / 2
        nu      = (nu1 + nu2) / 2
        neutral = 0.0
        pi      = np.sqrt(max(0, 1 - (mu**2 + nu**2)))
        result  = {
            "score":   float((mu**2) - (nu**2)),
            "mu":      float(mu),
            "nu":      float(nu),
            "pi":      float(pi),
            "neutral": float(neutral),
        }

        if len(self._cache_keys) >= self._cache_size:
            oldest = self._cache_keys.pop(0)
            self._cache.pop(oldest, None)
        self._cache_keys.append(cache_key)
        self._cache[cache_key] = result
        return result