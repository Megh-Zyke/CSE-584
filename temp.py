import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import os
import time
import faiss
import uvicorn
import traceback

from google import genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

API_KEY = os.environ.get("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found")

client = genai.Client(api_key=API_KEY)

app = FastAPI(title="Tri-Guard Semantic Cache API", version="4.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str


# -----------------------------------------------------
# PyFS Semantic Verifier
# -----------------------------------------------------

class PyFSSemanticCache:

    def __init__(self, model_name="cross-encoder/stsb-roberta-base"):

        print("Initializing PyFS STS Verifier...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        self.model.eval()

    def _nli_probs(self, a, b):

        inputs = self.tokenizer(
            a,
            b,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        with torch.no_grad():
            logits = self.model(**inputs).logits

        score = float(torch.sigmoid(logits).cpu().numpy().flatten()[0])

        mu = score
        nu = 1.0 - score
        neutral = 0.0

        return mu, nu, neutral

    def calculate_pyfs(self, q1, q2):

        mu1, nu1, ne1 = self._nli_probs(q1, q2)
        mu2, nu2, ne2 = self._nli_probs(q2, q1)

        mu = (mu1 + mu2) / 2
        nu = (nu1 + nu2) / 2
        neutral = (ne1 + ne2) / 2

        pi = np.sqrt(max(0, 1 - (mu**2 + nu**2)))

        pyfs_score = (mu**2) - (nu**2)

        return {
            "score": float(pyfs_score),
            "mu": float(mu),
            "nu": float(nu),
            "pi": float(pi),
            "neutral": float(neutral)
        }


# -----------------------------------------------------
# TriGuard Cache
# -----------------------------------------------------

class TriGuardCache:

    def __init__(self):

        print("Loading embedding model...")
        self.embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")

        print("Loading cross-encoder reranker...")
        self.reranker        = CrossEncoder("cross-encoder/quora-roberta-base")   # Q-Q gates
        self.answer_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        self.dimension = self.embedder.get_sentence_embedding_dimension()

        self.index = faiss.IndexFlatIP(self.dimension)

        self.store = []

        self.embedding_threshold = 0.8
        self.rerank_threshold = 0.7

        # PyFS now used as contradiction detector
        self.contradiction_threshold = 0.3

        # answer relevance threshold
        self.answer_relevance_threshold = 1.5

        self.verifier = PyFSSemanticCache()

    # -------------------------------------------------
    # Gemini API
    # -------------------------------------------------

    def call_gemini(self, query):

        try:

            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=query
            )

            return response.text

        except Exception as e:

            raise HTTPException(
                status_code=500,
                detail=str(e)
            )

    # -------------------------------------------------
    # Answer Relevance Verification
    # -------------------------------------------------

    def verify_answer_relevance(self, query, response):
        score = float(
            self.answer_reranker.predict([[query, response[:512]]])  # was self.reranker
        )
        return score

    # -------------------------------------------------
    # Rerank Candidates
    # -------------------------------------------------

    def rerank_candidates(self, query, candidate_indices):

        pairs = []
        valid_indices = []

        for idx in candidate_indices:

            if idx == -1:
                continue

            cached_query = self.store[idx]["original_query"]

            pairs.append([query, cached_query])

            valid_indices.append(idx)

        if not pairs:
            return None, None

        scores = self.reranker.predict(pairs)

        best_pos = int(np.argmax(scores))
        best_idx = valid_indices[best_pos]
        best_score = float(scores[best_pos])

        return best_idx, best_score

    # -------------------------------------------------
    # Main Query Handler
    # -------------------------------------------------

    def ask(self, query):

        query_vec = self.embedder.encode(
            [query],
            normalize_embeddings=True
        ).astype("float32")

        if self.index.ntotal > 0:

            try:

                distances, indices = self.index.search(query_vec, k=5)

                candidate_indices = indices[0]

                best_sim = distances[0][0]
                best_idx = candidate_indices[0]

                # -------------------------------
                # FAST PATH
                # -------------------------------
                print(f"[DEBUG] index size   : {self.index.ntotal}")
                print(f"[DEBUG] best_sim     : {best_sim:.4f}")
                print(f"[DEBUG] best_idx     : {best_idx}")
                print(f"[DEBUG] embed thresh : {self.embedding_threshold}")

                if best_idx != -1 and best_sim > self.embedding_threshold:

                    cached_query = self.store[best_idx]["original_query"]

                    res = self.verifier.calculate_pyfs(query, cached_query)
                    
                    # ADD THESE:
                    print(f"[DEBUG] cached_query : {cached_query}")
                    print(f"[DEBUG] mu={res['mu']:.4f}  nu={res['nu']:.4f}  pi={res['pi']:.4f}")
                    print(f"[DEBUG] contradiction check: {res['nu']:.4f} <= {self.contradiction_threshold} = {res['nu'] <= self.contradiction_threshold}")


                    # PyFS used as contradiction detector
                    if res["nu"] <= self.contradiction_threshold:

                        return {
                            "source": "CACHE_HIT_FAST",
                            "similarity": float(best_sim),
                            "mu": res["mu"],
                            "nu": res["nu"],
                            "pi": res["pi"],
                            "response": self.store[best_idx]["response"]
                        }

                # -------------------------------
                # FULL PATH
                # -------------------------------

                best_idx, rerank_score = self.rerank_candidates(
                    query,
                    candidate_indices
                )

                if best_idx is not None and rerank_score > self.rerank_threshold:

                    cached_query = self.store[best_idx]["original_query"]

                    res = self.verifier.calculate_pyfs(query, cached_query)

                    # contradiction gate
                    if res["nu"] <= self.contradiction_threshold:

                        response = self.store[best_idx]["response"]

                        ans_score = self.verify_answer_relevance(query, response)

                        if ans_score > self.answer_relevance_threshold:

                            return {
                                "source": "CACHE_HIT_VERIFIED",
                                "rerank_score": rerank_score,
                                "answer_relevance": ans_score,
                                "mu": res["mu"],
                                "nu": res["nu"],
                                "pi": res["pi"],
                                "response": response
                            }
                        else:
                            print("[DEBUG] FAST PATH: PyFS contradiction gate FAILED")
                    else:
                        print(f"[DEBUG] FAST PATH SKIPPED: sim {best_sim:.4f} did not exceed {self.embedding_threshold}")


            except Exception as e:

                print(f"[ERROR] Retrieval failed: {e}")
                traceback.print_exc()

        # -------------------------------
        # CALL GEMINI
        # -------------------------------

        start = time.time()

        response = self.call_gemini(query)

        latency = time.time() - start

        vec = self.embedder.encode(
            [query],
            normalize_embeddings=True
        ).astype("float32")

        self.index.add(vec)

        self.store.append({
            "original_query": query,
            "response": response,
            "cached_at": time.time()
        })

        return {
            "source": "GEMINI_API",
            "latency_seconds": round(latency, 4),
            "response": response
        }


# -----------------------------------------------------
# Initialize Cache
# -----------------------------------------------------

cache = TriGuardCache()


@app.post("/ask")
def ask_question(request: QueryRequest):

    return cache.ask(request.query)


if __name__ == "__main__":

    uvicorn.run(
        "temp:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )