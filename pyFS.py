import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class PyFSSemanticCache:
    def __init__(self, model_name="cross-encoder/nli-deberta-v3-base", threshold=0.35):
        print("Initializing PyFS NLI Cross-Encoder...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        self.model.eval()

        # label mapping
        self.label_map = {v.lower(): k for k, v in self.model.config.id2label.items()}

        self.threshold = threshold

    def _nli_probs(self, a, b):
        """Return entailment, contradiction, neutral probabilities"""

        inputs = self.tokenizer(
            a,
            b,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        entail = probs[self.label_map["entailment"]]
        contra = probs[self.label_map["contradiction"]]
        neutral = probs[self.label_map["neutral"]]

        return entail, contra, neutral

    def calculate_pyfs(self, q1: str, q2: str):
        """
        Bidirectional NLI + PyFS scoring
        """

        # forward
        mu1, nu1, ne1 = self._nli_probs(q1, q2)

        # reverse (important!)
        mu2, nu2, ne2 = self._nli_probs(q2, q1)

        # symmetric averaging
        mu = (mu1 + mu2) / 2
        nu = (nu1 + nu2) / 2
        neutral = (ne1 + ne2) / 2

        # Pythagorean fuzzy hesitation
        pi = np.sqrt(max(0, 1 - (mu**2 + nu**2)))

        # PyFS score
        pyfs_score = (mu**2) - (nu**2)

        return {
            "mu_similarity": float(mu),
            "nu_conflict": float(nu),
            "neutral": float(neutral),
            "pi_hesitation": float(pi),
            "pyfs_score": float(pyfs_score),
        }

    def is_cache_hit(self, q1, q2):
        """
        Decide whether cached result can be reused
        """

        res = self.calculate_pyfs(q1, q2)

        score = res["pyfs_score"]

        return score > self.threshold, res


if __name__ == "__main__":

    pyfs = PyFSSemanticCache()

    print("\n--- Test 1: Paraphrase ---")

    q1 = "What is the capital of France?"
    q2 = "What is the capital of France?"

    hit, res = pyfs.is_cache_hit(q1, q2)

    print(res)
    print("Cache Hit:", hit)

    print("\n--- Test 2: Different meaning ---")

    q1 = "What is the price of Apple stock?"
    q2 = "The stock market is falling today."

    hit, res = pyfs.is_cache_hit(q1, q2)

    print(res)
    print("Cache Hit:", hit)