"""Final paper-validation audit script.

Reads saved raw outputs from results/final_validation/ and computes:
- Paired bootstrap 95% CIs for F1, precision, recall
- F1 difference CIs (Voting+RAG vs Single-Shot) with significance test
- Latency-normalized gain (ΔF1/s)
- Source leakage analysis
- Near-dup leakage detection
- Paper-ready tables
"""

import json
import os
import sys
import numpy as np
from collections import Counter, defaultdict
from typing import List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.schemas import Verdict, VerificationResult
from src.metrics import compute_confusion_matrix, classification_metrics

SEED = 42
N_BOOTSTRAP = 10000
ALPHA = 0.05

RAW_DIR = "results/final_validation/raw_outputs"
REPORT_PATH = "results/final_validation/report.json"

ARCHITECTURES = ["single_shot", "voting_n3", "moa_rag", "voting_n3_rag"]
ARCH_LABELS = {
    "single_shot": "Single-Shot",
    "voting_n3": "Voting N=3",
    "moa_rag": "MoA+RAG",
    "voting_n3_rag": "Voting N=3+RAG",
}

# ── Load raw results ──────────────────────────────────────────────────────

def load_raw(domain: str, arch: str) -> Tuple[List[VerificationResult], List[Verdict], List[dict]]:
    """Load raw predictions for a domain/architecture.

    Returns:
        (results, truths, item_metadata)
    """
    path = os.path.join(RAW_DIR, f"{domain}_{arch}.json")
    with open(path) as f:
        data = json.load(f)

    # Get truth labels from items
    items = data.get("items", data.get("test_items", []))
    truths_raw = [it.get("ground_truth", None) for it in items]

    # Map string truth to Verdict
    truth_map = {"REAL": Verdict.REAL, "FAKE": Verdict.FAKE,
                 "ESCALATE": Verdict.ESCALATE, "EXAGGERATED": Verdict.EXAGGERATED,
                 "None": None, None: None}
    truths = []
    for t in truths_raw:
        if isinstance(t, str):
            truths.append(truth_map.get(t, None))
        elif isinstance(t, int):
            truths.append(Verdict(t))
        else:
            truths.append(None)

    # Get predictions from results
    results_list = data.get("results", [])
    item_by_id = {it["id"]: it for it in items}

    results = []
    for r in results_list:
        iid = r["item_id"]
        verdict_str = r["verdict"]
        if isinstance(verdict_str, str):
            verdict = getattr(Verdict, verdict_str.upper(),
                           truth_map.get(verdict_str, Verdict.REAL))
        elif isinstance(verdict_str, int):
            verdict = Verdict(verdict_str)
        else:
            verdict = Verdict.REAL
        vr = VerificationResult(
            item_id=iid,
            verdict=verdict,
            confidence=r.get("confidence", 0.5),
            latency_s=r.get("latency_s", 0.0),
        )
        results.append(vr)

    # Map truths ordered by results
    truth_by_id = {}
    for it in items:
        t = it.get("ground_truth", None)
        if isinstance(t, str):
            truth_by_id[it["id"]] = truth_map.get(t, None)
        elif isinstance(t, int):
            truth_by_id[it["id"]] = Verdict(t)
        else:
            truth_by_id[it["id"]] = None

    ordered_truths = [truth_by_id.get(r.item_id, None) for r in results]
    # Replace None with REAL (same as script does)
    ordered_truths = [t if t is not None else Verdict.REAL for t in ordered_truths]

    item_meta = [item_by_id.get(r.item_id, {}) for r in results]

    return results, ordered_truths, item_meta


def compute_metrics(results, truths):
    """Compute F1, precision, recall from results and truths."""
    cm = compute_confusion_matrix(results, truths)
    n = cm.tp + cm.fp + cm.tn + cm.fn
    precision = cm.tp / (cm.tp + cm.fp) if (cm.tp + cm.fp) > 0 else 0.0
    recall = cm.tp / (cm.tp + cm.fn) if (cm.tp + cm.fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = cm.fp / (cm.fp + cm.tn) if (cm.fp + cm.tn) > 0 else 0.0
    fnr = cm.fn / (cm.fn + cm.tp) if (cm.fn + cm.tp) > 0 else 0.0
    accuracy = (cm.tp + cm.tn) / n if n > 0 else 0.0
    return {
        "f1": f1, "precision": precision, "recall": recall,
        "fpr": fpr, "fnr": fnr, "accuracy": accuracy,
        "tp": cm.tp, "fp": cm.fp, "tn": cm.tn, "fn": cm.fn,
    }


def bootstrap_ci(results_list, truths_list, metric_fn, n_iter=N_BOOTSTRAP, alpha=ALPHA):
    """Compute paired bootstrap CI for a metric over multiple architectures.

    Args:
        results_list: List of result lists, one per architecture
        truths_list: List of truth lists (same length, paired by index)
        metric_fn: Function that takes (results, truths) and returns a float
        n_iter: Number of bootstrap iterations

    Returns:
        (lower, upper) tuple
    """
    n = len(truths_list[0])
    rng = np.random.RandomState(SEED)
    vals = []
    for _ in range(n_iter):
        idx = rng.choice(n, n, replace=True)
        v = metric_fn(
            [results_list[0][i] for i in idx],
            [truths_list[0][i] for i in idx],
        )
        vals.append(v)
    vals = sorted(vals)
    return (float(np.percentile(vals, alpha / 2 * 100)),
            float(np.percentile(vals, (1 - alpha / 2) * 100)))


def bootstrap_diff(results_a, truths_a, results_b, truths_b,
                   metric_fn, n_iter=N_BOOTSTRAP, alpha=ALPHA):
    """Bootstrap CI for difference (metric_a - metric_b).

    Paired by index (same items for both architectures).
    """
    n = len(truths_a)
    rng = np.random.RandomState(SEED)
    diffs = []
    for _ in range(n_iter):
        idx = rng.choice(n, n, replace=True)
        va = metric_fn([results_a[i] for i in idx], [truths_a[i] for i in idx])
        vb = metric_fn([results_b[i] for i in idx], [truths_b[i] for i in idx])
        diffs.append(va - vb)
    diffs = sorted(diffs)
    return (float(np.percentile(diffs, alpha / 2 * 100)),
            float(np.percentile(diffs, (1 - alpha / 2) * 100)))


def f1_fn(r, t):
    cm = compute_confusion_matrix(r, t)
    p = cm.tp / (cm.tp + cm.fp) if (cm.tp + cm.fp) > 0 else 0.0
    rc = cm.tp / (cm.tp + cm.fn) if (cm.tp + cm.fn) > 0 else 0.0
    return 2 * p * rc / (p + rc) if (p + rc) > 0 else 0.0


def prec_fn(r, t):
    cm = compute_confusion_matrix(r, t)
    return cm.tp / (cm.tp + cm.fp) if (cm.tp + cm.fp) > 0 else 0.0


def rec_fn(r, t):
    cm = compute_confusion_matrix(r, t)
    return cm.tp / (cm.tp + cm.fn) if (cm.tp + cm.fn) > 0 else 0.0


# ── Main audit ─────────────────────────────────────────────────────────────

def audit_domain(domain: str, n_items: int):
    print(f"\n{'=' * 72}")
    print(f"  AUDIT: {domain.upper()} (N={n_items})")
    print(f"{'=' * 72}")

    # Load all architectures
    arch_data = {}
    for arch in ARCHITECTURES:
        results, truths, meta = load_raw(domain, arch)
        arch_data[arch] = {"results": results, "truths": truths, "meta": meta}
        n_fake = sum(1 for t in truths if t == Verdict.FAKE)
        n_real = sum(1 for t in truths if t == Verdict.REAL)
        print(f"  Loaded {arch}: {len(results)} predictions ({n_fake}F/{n_real}R)")

    # ── 1. Metrics ──
    print(f"\n  ── Point Metrics ──")
    headers = ["Architecture", "F1", "Precision", "Recall", "FPR", "FNR", "Acc", "FP", "FN"]
    header_line = "  " + " | ".join(f"{h:>14}" for h in headers)
    print(header_line)
    print(f"  " + "-" * 110)

    all_metrics = {}
    for arch in ARCHITECTURES:
        m = compute_metrics(arch_data[arch]["results"], arch_data[arch]["truths"])
        all_metrics[arch] = m
        label = ARCH_LABELS[arch]
        fp = m["tp"] + m["fp"]
        print(f"  {label:>14s} | {m['f1']:>10.4f}  {m['precision']:>10.4f}  {m['recall']:>10.4f}  "
              f"{m['fpr']:>10.4f}  {m['fnr']:>10.4f}  {m['accuracy']:>10.4f}  "
              f"{fp:>4d}  {m['fn']:>4d}")

    # ── 2. Bootstrap CIs ──
    print(f"\n  ── 95% Bootstrap CIs (N={N_BOOTSTRAP}) ──")
    for metric_name, metric_fn, prec in [("F1", f1_fn, 4), ("Precision", prec_fn, 4), ("Recall", rec_fn, 4)]:
        print(f"  {metric_name}:")
        for arch in ARCHITECTURES:
            r = arch_data[arch]["results"]
            t = arch_data[arch]["truths"]
            lo, hi = bootstrap_ci([r], [t], metric_fn)
            point = all_metrics[arch][metric_name.lower()]
            print(f"    {ARCH_LABELS[arch]:>14s}: {point:.{prec}f}  CI=({lo:.{prec}f}, {hi:.{prec}f})")

    # ── 3. Voting+RAG vs Single-Shot difference ──
    print(f"\n  ── Voting N=3+RAG vs Single-Shot ──")
    for metric_name, metric_fn, prec in [("F1", f1_fn, 4), ("Precision", prec_fn, 4), ("Recall", rec_fn, 4)]:
        r_a = arch_data["voting_n3_rag"]["results"]
        t_a = arch_data["voting_n3_rag"]["truths"]
        r_b = arch_data["single_shot"]["results"]
        t_b = arch_data["single_shot"]["truths"]

        lo, hi = bootstrap_diff(r_a, t_a, r_b, t_b, metric_fn)
        point_a = all_metrics["voting_n3_rag"][metric_name.lower()]
        point_b = all_metrics["single_shot"][metric_name.lower()]
        diff = point_a - point_b

        significant = lo > 0 or hi < 0
        print(f"    {metric_name}: VRAG={point_a:.{prec}f}  SS={point_b:.{prec}f}  "
              f"Δ={diff:.{prec}f}  CI=({lo:.{prec}f}, {hi:.{prec}f})  "
              f"{'✓ SIGNIFICANT' if significant else '✗ NOT significant'}")

    # ── 4. Latency-normalized gain ──
    print(f"\n  ── Latency-Performance Tradeoff ──")
    # Load latencies from report
    with open(REPORT_PATH) as f:
        report = json.load(f)
    dom = report["domains"][domain]

    ss_lat = dom["single_shot"]["latency"]["mean"]
    ss_f1 = all_metrics["single_shot"]["f1"]

    header_line2 = "  " + " | ".join(f"{h:>14}" for h in headers)
    print(header_line2)
    print(f"  " + "-" * 110)

    for arch in ARCHITECTURES:
        lat = dom[arch]["latency"]["mean"]
        f1 = all_metrics[arch]["f1"]
        df1 = f1 - ss_f1
        dlat = lat - ss_lat
        eff = df1 / dlat if dlat > 0 else float('inf')
        label = ARCH_LABELS[arch]
        print(f"  {label:>14s} | {f1:>10.4f}  {lat:>8.2f}s  {df1:>+10.4f}  {dlat:>+10.2f}s  {eff:>+10.4f}")


def audit_source_leakage():
    """Check if explicit source names affect results."""
    print(f"\n{'=' * 72}")
    print(f"  AUDIT: SOURCE LEAKAGE")
    print(f"{'=' * 72}")

    import pandas as pd

    # Load finance CSV
    df = pd.read_csv("data/raw/finance/financial_news.csv")

    # Check for source mentions in titles
    source_kw = ["reuters", "bloomberg", "ap ", "cnn", "fox news", "breitbart",
                 "huffington", "politico", "washington post", "nyt ", "wsj ",
                 "cnbc", "msnbc", "bbc", "the guardian", "usa today"]

    for label_name, lbl in [("REAL(0)", 0), ("FAKE(1)", 1)]:
        sub = df[df["label"] == lbl]
        mentions = sum(1 for t in sub["title"].dropna() if any(kw in t.lower() for kw in source_kw))
        print(f"  {label_name}: {mentions}/{len(sub)} titles mention source ({mentions/len(sub)*100:.1f}%)")

    print()

    # Check: do REAL articles mention Reuters more?
    reuters_real = sum(1 for t in df[df["label"]==0]["title"].dropna() if "reuters" in t.lower())
    reuters_fake = sum(1 for t in df[df["label"]==1]["title"].dropna() if "reuters" in t.lower())
    print(f"  'Reuters' in title: REAL={reuters_real}, FAKE={reuters_fake}")

    # Load validation items and check source mentions
    items_real = df[df["label"] == 0].sample(n=25, random_state=42)
    items_fake = df[df["label"] == 1].sample(n=25, random_state=42)
    val_titles = pd.concat([items_fake, items_real])["title"]

    for kw in ["reuters", "bloomberg", "ap ", "washington", "politico", "cnn", "fox"]:
        cnt = sum(1 for t in val_titles if kw in t.lower())
        print(f"  '{kw}' in validation sample: {cnt}/50")

    print()
    print("  → If performance differs by source mention, model may be exploiting source names")
    print("  → If model performs equally on Reuters-mentioning and non-mentioning articles, no leakage")


def audit_duplicate_leakage():
    """Check for near-duplicate articles between test and corpus."""
    print(f"\n{'=' * 72}")
    print(f"  AUDIT: DUPLICATE / NEAR-DUPLICATE LEAKAGE")
    print(f"{'=' * 72}")

    import sys; sys.path.insert(0, '.')
    from src.final_validation import _sample_balanced

    items, corpus = _sample_balanced("finance")

    # Check exact title matches
    test_titles = set(it.claim_text.lower().strip() for it in items)
    corpus_titles = set(it.claim_text.lower().strip() for it in corpus)

    exact_overlap = test_titles & corpus_titles
    print(f"  Test items: {len(items)}")
    print(f"  Corpus items: {len(corpus)}")
    print(f"  Exact title overlap: {len(exact_overlap)}")
    if exact_overlap:
        for t in list(exact_overlap)[:5]:
            print(f"    '{t[:80]}'")

    # Check near-duplicate: high word overlap
    # Near-dup check with explicit listing
    near_dups = []
    for tt in test_titles:
        twords = set(tt.split())
        for ct in corpus_titles:
            cwords = set(ct.split())
            if len(twords) > 0 and len(cwords) > 0:
                jaccard = len(twords & cwords) / len(twords | cwords)
                if jaccard > 0.7:
                    near_dups.append((tt, ct, jaccard))
                    break

    print(f"  Near-dup overlap (Jaccard>0.7): {len(near_dups)}")
    for td, cd, j in near_dups[:5]:
        print(f"    J={j:.2f} TEST: \"{td[:80]}\"")
        print(f"           CORPUS: \"{cd[:80]}\"")
    if near_dups:
        # Check if these affect results
        print(f"  → Manual review recommended for {len(near_dups)} near-duplicate pairs")
        print(f"  → Impact: potential inflation of RAG metrics for those items")
    else:
        print(f"  → No significant near-dup leakage detected")


def compute_tfidf_baseline():
    """Compute simple TF-IDF + Logistic Regression baseline."""
    print(f"\n{'=' * 72}")
    print(f"  AUDIT: TF-IDF BASELINE")
    print(f"{'=' * 72}")

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    import pandas as pd

    df = pd.read_csv("data/raw/finance/financial_news.csv")
    titles = df["title"].fillna("").values
    labels = df["label"].values

    vec = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vec.fit_transform(titles)

    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    scores = cross_val_score(clf, X, labels, cv=5, scoring="f1")
    print(f"  TF-IDF + LogisticRegression on ALL {len(df)} titles:")
    print(f"  F1 = {scores.mean():.4f} ± {scores.std():.4f} (5-fold CV)")

    scores_acc = cross_val_score(clf, X, labels, cv=5, scoring="accuracy")
    print(f"  Acc = {scores_acc.mean():.4f} ± {scores_acc.std():.4f} (5-fold CV)")
    print()

    # Also on healthcare
    health_path = "data/raw/health/health_headlines.csv"
    if not os.path.exists(health_path):
        health_path = "data/raw/healthcare/health_headlines.csv"
    df_h = pd.read_csv(health_path, encoding="utf-8-sig")
    if "headline" in df_h.columns:
        h_titles = df_h["headline"].fillna("").values
    else:
        h_titles = df_h["text"].fillna("").values
    h_labels = df_h["label"].values

    vec_h = TfidfVectorizer(stop_words="english", max_features=5000)
    X_h = vec_h.fit_transform(h_titles)

    clf_h = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    scores_h = cross_val_score(clf_h, X_h, h_labels, cv=5, scoring="f1")
    print(f"  Healthcare TF-IDF baseline:")
    print(f"  F1 = {scores_h.mean():.4f} ± {scores_h.std():.4f} (5-fold CV)")
    print()

    # Also fit on full finance data and eval on the validation sample
    print(f"  TF-IDF trained on ALL 3731 finance articles, evaluated on N=50 val sample:")
    from src.final_validation import _sample_balanced
    test_items_f, _ = _sample_balanced("finance")
    val_texts_f = [it.claim_text for it in test_items_f]
    val_labels_f = [int(it.ground_truth == Verdict.FAKE) for it in test_items_f]

    vec_f = TfidfVectorizer(stop_words="english", max_features=5000)
    X_full_f = vec_f.fit_transform(titles)
    clf_f = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf_f.fit(X_full_f, labels)

    from sklearn.metrics import f1_score
    X_val_f = vec_f.transform(val_texts_f)
    preds_f = clf_f.predict(X_val_f)
    val_f1_f = f1_score(val_labels_f, preds_f)
    print(f"    Finance val sample F1: {val_f1_f:.4f}")

    # Same for healthcare
    print(f"\n  Healthcare: TF-IDF trained on all {len(df_h)} articles, evaluated on N=40 val sample:")
    test_items_h, _ = _sample_balanced("healthcare")
    val_texts_h = [it.claim_text for it in test_items_h]
    val_labels_h = [int(it.ground_truth == Verdict.FAKE) for it in test_items_h]

    vec_h2 = TfidfVectorizer(stop_words="english", max_features=5000)
    X_full_h = vec_h2.fit_transform(h_titles)
    clf_h2 = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf_h2.fit(X_full_h, h_labels)

    X_val_h = vec_h2.transform(val_texts_h)
    preds_h = clf_h2.predict(X_val_h)
    val_f1_h = f1_score(val_labels_h, preds_h)
    print(f"    Healthcare val sample F1: {val_f1_h:.4f}")


# ── Run ──
if __name__ == "__main__":
    print("=" * 72)
    print("  FINAL PAPER-VALIDATION AUDIT")
    print("=" * 72)
    print(f"  Date: 2026-07-11")
    print(f"  Model: deepseek-v4-flash")
    print(f"  Bootstrap iterations: {N_BOOTSTRAP}")
    print(f"  Alpha: {ALPHA}")

    audit_domain("finance", 50)
    audit_domain("healthcare", 40)

    print(f"\n{'=' * 72}")
    print(f"  CROSS-DOMAIN SUMMARY")
    print(f"{'=' * 72}")

    # Load report for cross-domain data
    with open(REPORT_PATH) as f:
        report = json.load(f)

    for arch in ARCHITECTURES:
        label = ARCH_LABELS[arch]
        f1f = report["domains"]["finance"][arch]["metrics"]["f1"]
        f1h = report["domains"]["healthcare"][arch]["metrics"]["f1"]
        mean_f1 = (f1f + f1h) / 2
        print(f"  {label:>14s}: Finance F1={f1f:.4f}, Health F1={f1h:.4f}, Mean={mean_f1:.4f}")

    audit_source_leakage()
    audit_duplicate_leakage()
    compute_tfidf_baseline()

    print(f"\n{'=' * 72}")
    print(f"  AUDIT COMPLETE")
    print(f"{'=' * 72}")
