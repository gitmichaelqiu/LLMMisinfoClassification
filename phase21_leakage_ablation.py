"""Phase 21: Baseline Leakage Auditing & Ablation.

Performs two analyses on the classical baselines (LR, GBDT):
1. Feature importance analysis to check if synthetic templates leak FAKE class
2. Retraining with key lexical markers ablated to measure real vs shortcut performance

Output:
  - plots/phase_21_leakage_diagnostic.png
  - output/phase_21_ablation_metrics.json
"""

import os, sys, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

os.makedirs("./output", exist_ok=True)
os.makedirs("./plots", exist_ok=True)

def _load_dataset(target_size=None):
    """Minimal copy of main.load_combined_dataset() avoiding full imports."""
    synth_path = "./input/headlines.csv"
    if os.path.exists(synth_path):
        synth_df = pd.read_csv(synth_path)
        synth_df["source"] = "synthetic"
        synth_df["content"] = synth_df["headline"]
    else:
        synth_df = pd.DataFrame(columns=["content", "label", "source"])

    kaggle_path = "./input/kaggle_fake_news_FULL.csv"
    if os.path.exists(kaggle_path):
        kaggle_df = pd.read_csv(kaggle_path).dropna(subset=["text", "label"])
        kaggle_df["source"] = "kaggle"
        kaggle_df["content"] = kaggle_df["text"]
    else:
        kaggle_df = pd.DataFrame(columns=["content", "label", "source"])

    if target_size:
        kaggle_target = target_size - len(synth_df)
        per_class = max(1, kaggle_target // 2)
        kaggle_sample = pd.concat([
            kaggle_df[kaggle_df["label"] == 0].sample(n=min(len(kaggle_df[kaggle_df["label"] == 0]), per_class), random_state=42),
            kaggle_df[kaggle_df["label"] == 1].sample(n=min(len(kaggle_df[kaggle_df["label"] == 1]), per_class), random_state=42),
        ], ignore_index=True)
        return pd.concat([synth_df[["content", "label", "source"]], kaggle_sample[["content", "label", "source"]]], ignore_index=True)
    return pd.concat([synth_df[["content", "label", "source"]], kaggle_df[["content", "label", "source"]]], ignore_index=True)

df = _load_dataset(target_size=5000)
df["content"] = df["content"].fillna("")
X_raw = df["content"].values
y = df["label"].values
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train: {len(X_train)}  Test: {len(X_test)}  Fake rate: {y.mean():.2%}")

# ── Ablation: lexical markers ──────────────────────────────────
# These are the most suspicious shortcut features in synthetic data
from src.system0_filter import System0Filter
s0 = System0Filter(enabled=True)
PANIC_SET = {w.lower() for w in s0.PANIC_KEYWORDS}
ENTITY_SET = {w.lower() for w in s0.HIGH_IMPACT_ENTITIES}
SHORT_PANIC = {w for w in PANIC_SET if len(w) > 3}
SHORT_ENTITY = {w for w in ENTITY_SET if len(w) > 3}

def ablate_text(text, remove_panic=True, remove_entities=True):
    """Remove lexical shortcut markers from text."""
    if not isinstance(text, str):
        return ""
    words = text.split()
    kept = []
    for w in words:
        wl = w.lower().strip(".,!?;:'\"()[]{}")
        if remove_panic and wl in SHORT_PANIC:
            continue
        if remove_entities and wl in SHORT_ENTITY:
            continue
        kept.append(w)
    return " ".join(kept)

X_train_ablated = np.array([ablate_text(t) for t in X_train])
X_test_ablated = np.array([ablate_text(t) for t in X_test])

total_orig = sum(len(t.split()) for t in X_train if isinstance(t, str))
total_abl = sum(len(t.split()) for t in X_train_ablated if isinstance(t, str))
print(f"\nAblation removed {total_orig - total_abl} words (panic keywords + entity names)")

# ── Train classifiers ──────────────────────────────────────────
results = {}

for name, X_tr, X_te in [("original", X_train, X_test), ("ablated", X_train_ablated, X_test_ablated)]:
    print(f"\n{'='*50}\n{name.upper()}\n{'='*50}")

    for clf_name, clf, vec_params in [
        ("TF-IDF+LR", LogisticRegression(max_iter=1000, random_state=42), {"max_features": 5000, "ngram_range": (1, 2)}),
        ("GBDT", GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                             subsample=0.8, random_state=42), {"max_features": 3000, "ngram_range": (1, 2)}),
    ]:
        vec = TfidfVectorizer(**vec_params)
        X_tr_vec = vec.fit_transform(X_tr)
        X_te_vec = vec.transform(X_te)

        clf.fit(X_tr_vec, y_train)
        preds = clf.predict(X_te_vec)

        metrics = {
            "accuracy_pct": round(float(accuracy_score(y_test, preds) * 100), 2),
            "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, preds, zero_division=0)), 4),
        }
        results[f"{name}_{clf_name}"] = metrics
        print(f"  {clf_name:15s}  F1={metrics['f1_score']:.4f}  "
              f"Prec={metrics['precision']:.4f}  Rec={metrics['recall']:.4f}  "
              f"Acc={metrics['accuracy_pct']:.1f}%")

# ── Ablation impact ────────────────────────────────────────────
print(f"\n{'='*50}\nABLATION IMPACT\n{'='*50}")
for clf_name in ["TF-IDF+LR", "GBDT"]:
    orig = results[f"original_{clf_name}"]
    ablated = results[f"ablated_{clf_name}"]
    delta = orig["f1_score"] - ablated["f1_score"]
    print(f"  {clf_name:15s}: F1 delta = {delta:+.4f} "
          f"(original={orig['f1_score']:.4f} ablated={ablated['f1_score']:.4f})")
    ablated["f1_delta"] = round(delta, 4)

# ── Feature importance plot (GBDT coefficients) ────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

vec = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X_tr_vec = vec.fit_transform(X_train)
gbdt = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
gbdt.fit(X_tr_vec, y_train)

importances = gbdt.feature_importances_
top_n = 25
top_indices = np.argsort(importances)[-top_n:]
feature_names = np.array(vec.get_feature_names_out())

fig, ax = plt.subplots(figsize=(12, 8))
colors = ["red" if any(p in feature_names[i].lower() for p in ["bankrupt", "fraud", "explos",
            "crash", "hack", "indict", "arrest", "insolvent", "collapse", "terror"])
          else "steelblue" for i in top_indices]
ax.barh(range(top_n), importances[top_indices], color=colors, alpha=0.8, edgecolor="white")
ax.set_yticks(range(top_n))
ax.set_yticklabels(feature_names[top_indices], fontsize=9)
ax.set_xlabel("Feature Importance (GBDT)")
ax.set_title("Top 25 GBDT Features — Panic Keywords Highlighted in Red",
            fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3, axis="x")
n_panic = sum(1 for c in colors if c == "red")
ax.text(0.95, 0.02, f"Panic keywords in top-25: {n_panic}",
        transform=ax.transAxes, ha="right", fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8))

leak_path = "./plots/phase_21_leakage_diagnostic.png"
fig.tight_layout()
fig.savefig(leak_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"\nLeakage diagnostic plot -> {leak_path}")

# ── Save results ───────────────────────────────────────────────
results["ablation_config"] = {
    "n_features_original": X_tr_vec.shape[1],
    "panic_keywords_removed": len(SHORT_PANIC),
    "entity_keywords_removed": len(SHORT_ENTITY),
}
save_path = "./output/phase_21_ablation_metrics.json"
with open(save_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Ablation metrics -> {save_path}")
