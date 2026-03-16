from __future__ import annotations

import os
import re
import time
import logging
from dataclasses import dataclass, field
from typing import Optional
import torch

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ttl_classifier")


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# TTL values in seconds
TTL_MAP: dict[str, float] = {
    "Static":      float("inf"),
    "Slow-Moving": 60 * 60 * 24 * 30,   # 30 days
    "Volatile":    60 * 5,               # 5 minutes
}

# Human-readable TTL labels (for display / logging)
TTL_DISPLAY: dict[str, str] = {
    "Static":      "∞  (never expires)",
    "Slow-Moving": "30 days",
    "Volatile":    "5 minutes",
}

# Encoder model — fast 384-dim model, good accuracy/latency tradeoff
ENCODER_NAME   = "BAAI/bge-small-en-v1.5"
MODEL_SAVE_PATH = "ttl_classifier.joblib"

CONFIDENCE_THRESHOLD = 0.60

FALLBACK_LABEL = "Volatile"

CACHE_ADMISSION_THRESHOLD = 0.90
# ─────────────────────────────────────────────────────────────────────────────
# Inference result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TTLResult:
    """
    Full inference result returned by TTLClassifier.predict().

    Fields
    ------
    label           : Predicted temporal class
    ttl             : TTL in seconds (float("inf") for Static)
    ttl_display     : Human-readable TTL string
    confidence      : Classifier probability for the predicted class (0–1)
    admit_to_cache  : True if Gate 3 confidence >= CACHE_ADMISSION_THRESHOLD
    stage           : Which pipeline stage produced this result
    all_probs       : Full probability dict across all classes
    latency_ms      : Total inference time in milliseconds
    """
    label          : str
    ttl            : float
    ttl_display    : str
    confidence     : float
    admit_to_cache : bool
    stage          : str
    all_probs      : dict[str, float] = field(default_factory=dict)
    latency_ms     : float = 0.0

    def __str__(self) -> str:
        cache_flag = "✓ CACHE" if self.admit_to_cache else "✗ NO CACHE"
        return (
            f"Label      : {self.label}\n"
            f"TTL        : {self.ttl_display}\n"
            f"Confidence : {self.confidence:.1%}\n"
            f"Cache gate : {cache_flag}\n"
            f"Stage      : {self.stage}\n"
            f"Latency    : {self.latency_ms:.1f} ms\n"
            f"All probs  : " +
            "  ".join(f"{k}={v:.1%}" for k, v in self.all_probs.items())
            
        )


# ─────────────────────────────────────────────────────────────────────────────
# TTL Classifier
# ─────────────────────────────────────────────────────────────────────────────

class TTLClassifier:
    """
    Three-stage TTL classifier for Tri-Guard semantic cache.

    Usage
    -----
    # Train once:
        clf = TTLClassifier()
        clf.train("synthetic_ttl_training_dataset.xlsx")
        clf.save()

    # Inference (load saved model):
        clf = TTLClassifier()
        clf.load()
        result = clf.predict("what is bitcoin trading at right now")
        print(result)

    # Batch inference:
        results = clf.predict_batch(["query1", "query2", ...])
    """

    def __init__(
        self,
        encoder_name: str = ENCODER_NAME,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        fallback_label: str = FALLBACK_LABEL,
        cache_admission_threshold: float = CACHE_ADMISSION_THRESHOLD,
    ):
        self.encoder_name              = encoder_name
        self.confidence_threshold      = confidence_threshold
        self.fallback_label            = fallback_label
        self.cache_admission_threshold = cache_admission_threshold

        self.encoder       : Optional[SentenceTransformer] = None
        self.clf           : Optional[LogisticRegression]  = None
        self.label_encoder : LabelEncoder                  = LabelEncoder()
        self.is_trained    : bool                          = False

        self._classes      : list[str]                     = []

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_encoder(self) -> None:
        if self.encoder is None:
            log.info("Loading encoder: %s", self.encoder_name)
            self.encoder = SentenceTransformer(self.encoder_name, backend="onnx") 

            self.encoder.encode(["warmup"], normalize_embeddings=True)

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of strings; single strings are wrapped automatically."""
        return self.encoder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=8,
        )

    def _make_result(
        self,
        label: str,
        probs: Optional[np.ndarray],
        stage: str,
        latency_ms: float,
    ) -> TTLResult:
        confidence = float(probs.max()) if probs is not None else 1.0

        all_probs: dict[str, float] = {}
        if probs is not None:
            for i, cls in enumerate(self._classes):
                all_probs[cls] = float(probs[i])
        else:
            all_probs = {c: (1.0 if c == label else 0.0) for c in TTL_MAP}

        return TTLResult(
            label          = label,
            ttl            = TTL_MAP[label],
            ttl_display    = TTL_DISPLAY[label],
            confidence     = confidence,
            admit_to_cache = confidence >= self.cache_admission_threshold,
            stage          = stage,
            all_probs      = all_probs,
            latency_ms     = latency_ms,
        )

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        data_path: str,
        question_col: str = "Question",
        label_col: str = "Temporal_Category",
        test_size: float = 0.2,
        cv_folds: int = 5,
    ) -> None:
        """
        Train the classifier from an Excel dataset.
        """
        self._load_encoder()

        # ── Load data ─────────────────────────────────────────────────────────
        log.info("Loading dataset: %s", data_path)
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        else:
            df = pd.read_excel(data_path)
            
        before = len(df)
        df = df.drop_duplicates(subset=[question_col]).reset_index(drop=True)
        log.info("Loaded %d rows (%d duplicates removed)", len(df), before - len(df))

        queries = df[question_col].tolist()
        labels  = df[label_col].tolist()

        dist = pd.Series(labels).value_counts().to_dict()
        log.info("Class distribution: %s", dist)

        # ── Encode ────────────────────────────────────────────────────────────
        log.info("Encoding %d queries (this takes ~1-2 min on CPU)...", len(queries))
        embeddings = self.encoder.encode(
            queries,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=64,
        )
        log.info("Embeddings shape: %s", embeddings.shape)

        encoded_labels = self.label_encoder.fit_transform(labels)
        self._classes  = list(self.label_encoder.classes_)

        # ── Train / test split ────────────────────────────────────────────────
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, encoded_labels,
            test_size=test_size,
            stratify=encoded_labels,
            random_state=42,
        )

        log.info(
            "Split — train: %d  test: %d",
            len(X_train), len(X_test),
        )

        # ── Fit classifier ────────────────────────────────────────────────────
        log.info("Training LogisticRegression (class_weight=balanced)...")
        self.clf = LogisticRegression(
            max_iter=2000,
            C=1.0,
            solver="lbfgs",
            class_weight="balanced",
            random_state=42,
        )
        self.clf.fit(X_train, y_train)
        self.is_trained = True

        # ── Hold-out evaluation ───────────────────────────────────────────────
        y_pred = self.clf.predict(X_test)
        print("\n" + "─" * 55)
        print("  HOLD-OUT TEST SET EVALUATION")
        print("─" * 55)
        print(classification_report(
            y_test, y_pred,
            target_names=self._classes,
            digits=4,
        ))

        # ── 5-fold cross-validation ───────────────────────────────────────────
        log.info("Running %d-fold stratified cross-validation...", cv_folds)
        skf    = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_acc = []
        for fold, (tr_idx, val_idx) in enumerate(skf.split(embeddings, encoded_labels)):
            fold_clf = LogisticRegression(
                max_iter=2000, C=1.0, solver="lbfgs",
                class_weight="balanced",
                random_state=42,
            )
            fold_clf.fit(embeddings[tr_idx], encoded_labels[tr_idx])
            acc = fold_clf.score(embeddings[val_idx], encoded_labels[val_idx])
            cv_acc.append(acc)
            log.info("  Fold %d  accuracy = %.4f", fold + 1, acc)

        print(f"\n  Cross-validation ({cv_folds}-fold):  "
              f"mean={np.mean(cv_acc):.4f}  std={np.std(cv_acc):.4f}")
        print("─" * 55 + "\n")

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str = MODEL_SAVE_PATH) -> None:
        """Serialize classifier + label encoder to disk."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet.")
        joblib.dump({
            "clf":                     self.clf,
            "label_encoder":           self.label_encoder,
            "classes":                 self._classes,
            "encoder_name":            self.encoder_name,
            "confidence_threshold":    self.confidence_threshold,
            "fallback_label":          self.fallback_label,
            "cache_admission_threshold": self.cache_admission_threshold,
        }, path)
        log.info("Model saved → %s", path)

    def load(self, path: str = MODEL_SAVE_PATH) -> None:
        payload = joblib.load(path)
        self.clf                       = payload["clf"]
        self.label_encoder             = payload["label_encoder"]

        self._classes = payload.get(
            "classes",
            list(self.label_encoder.classes_)   # fallback: derive from encoder
        )

        self.encoder_name              = payload.get("encoder_name", ENCODER_NAME)
        self.confidence_threshold      = payload.get("confidence_threshold", CONFIDENCE_THRESHOLD)
        self.fallback_label            = payload.get("fallback_label", FALLBACK_LABEL)
        self.cache_admission_threshold = payload.get("cache_admission_threshold", CACHE_ADMISSION_THRESHOLD)
        self.is_trained                = True
        self._load_encoder()
        log.info("Model loaded ← %s  |  classes: %s", path, self._classes)

    # ── Single-query inference ────────────────────────────────────────────────

    def predict(self, query: str) -> TTLResult:
        """
        Classify a single query and return a full TTLResult.
        """
        if not self.is_trained:
            raise RuntimeError("Call train() or load() first.")

        t0 = time.perf_counter()

        # ── Stage 1: embedding + classifier ──────────────────────────────────
        embedding = self._encode([query])
        probs     = self.clf.predict_proba(embedding)[0]

        top_idx   = int(probs.argmax())
        top_label = self.label_encoder.inverse_transform([top_idx])[0]
        top_conf  = float(probs.max())


        # ── Stage 2: confidence gate ──────────────────────────────────────────
        if top_conf < self.confidence_threshold:
            latency = (time.perf_counter() - t0) * 1000
            # Use fallback but keep the real probs for transparency
            return self._make_result(self.fallback_label, probs, "fallback_low_confidence", latency)

        latency = (time.perf_counter() - t0) * 1000
        return self._make_result(top_label, probs, "embedding_classifier", latency)

    # ── Batch inference ───────────────────────────────────────────────────────

    def predict_batch(self, queries: list[str]) -> list[TTLResult]:
        """
        Classify a list of queries efficiently using batched encoding.
        """
        if not self.is_trained:
            raise RuntimeError("Call train() or load() first.")

        t0 = time.perf_counter()

        results: list[Optional[TTLResult]] = [None] * len(queries)
        embed_indices: list[int]           = []

        # Stage 1 pass — resolve prefilter hits immediately
        for i, q in enumerate(queries):
            embed_indices.append(i)

        # Stage 2 pass — batch-encode remaining queries
        if embed_indices:
            batch_queries = [queries[i] for i in embed_indices]
            embeddings    = self._encode(batch_queries)
            probs_batch   = self.clf.predict_proba(embeddings)

            for j, i in enumerate(embed_indices):
                probs     = probs_batch[j]
                top_idx   = int(probs.argmax())
                top_label = self.label_encoder.inverse_transform([top_idx])[0]
                top_conf  = float(probs.max())

                if top_conf < self.confidence_threshold:
                    results[i] = self._make_result(
                        self.fallback_label, probs, "fallback_low_confidence", 0.0
                    )
                else:
                    results[i] = self._make_result(
                        top_label, probs, "embedding_classifier", 0.0
                    )

        # Stamp total batch latency proportionally
        total_ms = (time.perf_counter() - t0) * 1000
        per_query = total_ms / len(queries)
        for r in results:
            r.latency_ms = per_query

        return results

    # ── Evaluation / Testing ──────────────────────────────────────────────────

    def evaluate(self, data_path: str, question_col: str = "Question", label_col: str = "Temporal_Category") -> None:
        """
        Run inference on a labelled dataset and print performance metrics.
        """
        if not self.is_trained:
            raise RuntimeError("Call train() or load() first before testing.")

        log.info("Loading test dataset: %s", data_path)
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        else:
            df = pd.read_excel(data_path)

        if question_col not in df.columns or label_col not in df.columns:
            raise ValueError(f"Dataset must contain '{question_col}' and '{label_col}' columns.")

        # Drop any empty rows to avoid inference crashes
        df = df.dropna(subset=[question_col, label_col])
        queries = df[question_col].tolist()
        true_labels = df[label_col].tolist()

        log.info("Running batch inference on %d queries...", len(queries))
        
        # Track time for latency metrics
        t0 = time.perf_counter()
        results = self.predict_batch(queries)
        total_time = time.perf_counter() - t0
        
        pred_labels = [r.label for r in results]

        print("\n" + "═" * 55)
        print("  TESTING DATASET EVALUATION")
        print("═" * 55)
        print(f"Total Queries : {len(queries)}")
        print(f"Total Time    : {total_time:.2f} seconds")
        print(f"Avg Latency   : {(total_time / len(queries)) * 1000:.1f} ms / query\n")
        
        print(classification_report(
            true_labels, pred_labels,
            digits=4,
        ))
        print("═" * 55 + "\n")

    # ── Convenience accessors ─────────────────────────────────────────────────

    def get_ttl(self, query: str) -> float:
        """Return just the TTL value in seconds."""
        return self.predict(query).ttl

    def should_cache(self, query: str) -> tuple[bool, TTLResult]:
        """
        Gate 3 integration point for the caching layer.
        """
        result = self.predict(query)
        return result.admit_to_cache, result


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _print_banner() -> None:
    print("\n" + "═" * 55)
    print("  Tri-Guard  ·  Dynamic TTL Classifier  ·  Gate 2")
    print("═" * 55)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Tri-Guard TTL Classifier — train or run inference"
    )
    parser.add_argument(
        "--train",
        metavar="XLSX",
        help="Path to training Excel/CSV file. Trains and saves model.",
    )
    parser.add_argument(
        "--model",
        default=MODEL_SAVE_PATH,
        help=f"Path to saved model (default: {MODEL_SAVE_PATH})",
    )
    parser.add_argument(
        "--query",
        metavar="TEXT",
        help="Classify a single query and exit.",
    )
    parser.add_argument(
        "--batch",
        metavar="FILE",
        help="Path to plain-text file with one query per line.",
    )
    parser.add_argument(
        "--testing",
        metavar="FILE",
        help="Path to labeled test dataset (CSV/XLSX) to evaluate model accuracy.",
    )
    args = parser.parse_args()

    clf = TTLClassifier()

    # ── Train mode ────────────────────────────────────────────────────────────
    if args.train:
        clf.train(args.train)
        clf.save(args.model)
        return

    # ── Load saved model ──────────────────────────────────────────────────────
    if not os.path.exists(args.model):
        print(f"[ERROR] No saved model at '{args.model}'.")
        print("        Run with --train <dataset.xlsx> first.")
        return

    clf.load(args.model)
    _print_banner()

    # ── Testing / Evaluation mode ─────────────────────────────────────────────
    if args.testing:
        clf.evaluate(args.testing)
        return

    # ── Single query mode ─────────────────────────────────────────────────────
    if args.query:
        result = clf.predict(args.query)
        print(f"\nQuery: {args.query}\n")
        print(result)
        return

    # ── Batch mode ────────────────────────────────────────────────────────────
    if args.batch:
        with open(args.batch) as f:
            queries = [line.strip() for line in f if line.strip()]

        print(f"\nRunning batch inference on {len(queries)} queries...\n")
        results = clf.predict_batch(queries)

        print(f"{'Query':<60}  {'Label':<13}  {'Conf':>6}  {'TTL':<12}  {'Cache':<7}  Stage")
        print("─" * 120)
        for q, r in zip(queries, results):
            cache_flag = "✓" if r.admit_to_cache else "✗"
            q_display  = (q[:57] + "...") if len(q) > 60 else q
            print(
                f"{q_display:<60}  {r.label:<13}  {r.confidence:>6.1%}"
                f"  {r.ttl_display:<12}  {cache_flag:<7}  {r.stage}"
            )
        return

    # ── Interactive REPL mode ─────────────────────────────────────────────────
    print("\nType a query to classify. Commands: 'quit' to exit.\n")

    while True:
        try:
            raw = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not raw:
            continue
        if raw.lower() in ("quit", "exit", "q"):
            break

        result = clf.predict(raw)
        print()
        print(result)
        print()


if __name__ == "__main__":
    main()