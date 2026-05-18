"""Ensemble meta-classifier combining 5 detection signals."""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_score, recall_score, f1_score


def finbert_score(finbert_model, content):
    """Run FinBERT sentiment analysis, return float in [-1, 1]."""
    if finbert_model is None:
        return 0.0
    try:
        res = finbert_model(content[:512])[0]
        label = res["label"]
        if label == "POSITIVE":
            return 1.0
        elif label == "NEGATIVE":
            return -1.0
        else:
            return 0.0
    except Exception:
        return 0.0


def compute_ece(probabilities, labels, n_bins=10):
    """Expected Calibration Error."""
    if len(probabilities) == 0:
        return 1.0
    probs = np.array(probabilities)
    labels = np.array(labels)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if not np.any(mask):
            continue
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        ece += np.abs(bin_acc - bin_conf) * mask.sum() / len(probs)
    return float(ece)


def plot_calibration_curve(probabilities, labels, save_path="./plots/calibration_curve.png"):
    """Reliability diagram: confidence vs accuracy."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    probs = np.array(probabilities)
    labels = np.array(labels)
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_accs = []
    bin_counts = []
    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        count = mask.sum()
        bin_counts.append(count)
        bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
        if count > 0:
            bin_accs.append(labels[mask].mean())
        else:
            bin_accs.append(0.0)

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.plot(bin_centers, bin_accs, "o-", label="Ensemble", linewidth=2)
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Calibration Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


class EnsembleDetector:
    """Meta-classifier combining 5 detection signals.

    Signals:
        1. FinBERT sentiment (System 1)
        2. LLM CoT verdict + confidence (System 2)
        3. CoT structured flags (5 binary)
        4. TF-IDF heuristic (Phase 1 baseline)
        5. RAG contradiction score

    Feature vector: 11 dimensions.
    """

    def __init__(self, rag_retriever=None, finbert_model=None, heuristic_baseline=None):
        self.rag_retriever = rag_retriever
        self.finbert_model = finbert_model
        self.heuristic_baseline = heuristic_baseline
        self.classifier = None

    def _tfidf_score(self, content):
        """Get TF-IDF heuristic score (class + decision function value)."""
        if self.heuristic_baseline is None:
            return 0, 0.0
        vectorizer, clf = self.heuristic_baseline
        X = vectorizer.transform([content])
        cls = int(clf.predict(X)[0])
        # Use decision_function for continuous score
        if hasattr(clf, "decision_function"):
            score = float(clf.decision_function(X)[0])
        else:
            score = float(cls)
        return cls, score

    def _rag_contradiction_score(self, headline):
        """Compute RAG contradiction score: 1 - max similarity with retrieved docs."""
        if self.rag_retriever is None:
            return 0.5
        try:
            ctx_results, _ = self.rag_retriever.retrieve(headline)
            if not ctx_results:
                return 0.5
            max_sim = max(score for _, score in ctx_results)
            return float(1.0 - max_sim)
        except Exception:
            return 0.5

    def collect_features(self, headline, cot_result=None):
        """Collect all signals and return 11-dim feature vector.

        Args:
            headline: Input headline text.
            cot_result: Dict from cot_parser.parse_cot_output (or None to use defaults).

        Returns:
            np.ndarray of shape (11,).
        """
        if cot_result is None:
            cot_result = {
                "verdict": 0,
                "confidence": 0.5,
                "contradiction_flag": 0,
                "entity_mismatch": 0,
                "temporal_inconsistency": 0,
                "metric_implausibility": 0,
                "source_unverifiable": 0,
            }

        # Signal 1: FinBERT sentiment
        fb = finbert_score(self.finbert_model, headline)

        # Signal 2: LLM verdict + confidence
        llm_v = cot_result.get("verdict", 0)
        llm_c = cot_result.get("confidence", 0.5)

        # Signal 3: CoT flags
        c_flags = [
            cot_result.get("contradiction_flag", 0),
            cot_result.get("entity_mismatch", 0),
            cot_result.get("temporal_inconsistency", 0),
            cot_result.get("metric_implausibility", 0),
            cot_result.get("source_unverifiable", 0),
        ]
        n_flags = sum(c_flags)

        # Signal 4: TF-IDF heuristic
        h_cls, h_score = self._tfidf_score(headline)

        # Signal 5: RAG contradiction score
        rag_cs = self._rag_contradiction_score(headline)

        return np.array([
            fb,           # 0: finbert_sentiment
            llm_v,        # 1: llm_verdict
            llm_c,        # 2: llm_confidence
            h_cls,        # 3: tfidf_class
            h_score,      # 4: tfidf_score
            rag_cs,       # 5: rag_contradiction
            c_flags[0],   # 6: cot_contradiction_flag
            c_flags[1],   # 7: cot_entity_mismatch
            c_flags[2],   # 8: cot_temporal_inconsistency
            c_flags[3],   # 9: cot_metric_implausibility
            n_flags,      # 10: n_cot_flags
        ], dtype=np.float64)

    def train(self, feature_vectors, labels):
        """Train meta-classifier with Platt scaling calibration.

        Args:
            feature_vectors: List or 2D array of shape (n_samples, 11).
            labels: List or 1D array of shape (n_samples,).
        """
        X = np.array(feature_vectors)
        y = np.array(labels)

        base_clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        self.classifier = CalibratedClassifierCV(base_clf, method="sigmoid", cv=3)

        # For small datasets, cv=3 may fail if classes are missing in a fold.
        # Fallback: cv=min(3, smallest_class_count)
        unique, counts = np.unique(y, return_counts=True)
        min_class = counts.min()
        cv = min(3, min_class)
        if cv < 2:
            cv = 2

        self.classifier = CalibratedClassifierCV(
            LogisticRegression(C=1.0, max_iter=1000, random_state=42),
            method="sigmoid",
            cv=cv,
        )
        self.classifier.fit(X, y)

    def predict_proba(self, feature_vector):
        """Return P(class=1) for a single feature vector."""
        if self.classifier is None:
            return 0.5
        X = np.array(feature_vector).reshape(1, -1)
        return float(self.classifier.predict_proba(X)[0, 1])

    def predict(self, feature_vector, threshold=0.5):
        """Return class 0/1 for a single feature vector."""
        proba = self.predict_proba(feature_vector)
        return 1 if proba >= threshold else 0
