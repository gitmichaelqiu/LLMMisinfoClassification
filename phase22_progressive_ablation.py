"""Phase 22: Progressive Leakage Ablation Curve.

Retrains GBDT and TF-IDF+LR under 5 progressively aggressive ablations:
  1. No ablation (baseline)
  2. Panic keywords removed
  3. Panic keywords + entity names removed
  4. POS-tag template normalization (remove template artifacts)
  5. Paraphrase-invariant (semantic-only via paraphrase proxies)

Generates F1 vs. ablation aggressiveness curve.
"""

import os, json, warnings, re
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

os.makedirs("./output", exist_ok=True)
os.makedirs("./plots", exist_ok=True)

# ── Load data ──────────────────────────────────────────────────
synth = pd.read_csv("./input/headlines.csv")
kaggle = pd.read_csv("./input/kaggle_fake_news_FULL.csv").dropna(subset=["text","label"])
synth["text"] = synth["headline"]
kaggle["text"] = kaggle["text"]

# Balance
min_n = min(len(synth), 2000)
idf = pd.concat([
    synth.sample(n=min_n, random_state=42),
    kaggle.sample(n=min_n, random_state=42),
], ignore_index=True)
idf["label"] = idf["label"].astype(int)

train_df, test_df = train_test_split(idf, test_size=0.2, stratify=idf["label"], random_state=42)
print(f"Train: {len(train_df)}  Test: {len(test_df)}")

# ── Ablation functions ─────────────────────────────────────────
from src.system0_filter import System0Filter
s0 = System0Filter(enabled=True)
PANIC = {w.lower().strip(".,!?;:'\"") for w in s0.PANIC_KEYWORDS if len(w) > 3}
ENTITIES = {w.lower().strip(".,!?;:'\"") for w in s0.HIGH_IMPACT_ENTITIES if len(w) > 3}

# Template marker words (from synthetic template patterns)
TEMPLATE_MARKERS = {
    "reports", "announces", "posts", "surges", "plunges", "soars",
    "drops", "rises", "falls", "gains", "declines", "increases",
    "acquires", "partners", "secures", "receives", "confirms",
    "expects", "plans", "launches", "unveils", "introduces",
    "quarter", "fiscal", "annual", "guidance", "forecast", "outlook",
    "dividend", "buyback", "offering", "listing", "rating", "upgrade",
    "downgrade", "target", "estimate", "consensus", "analyst",
    "billion", "million", "percent", "growth", "momentum",
}

EXCLUDE_PATTERNS = re.compile(
    r'\b(?:' + '|'.join(re.escape(w) for w in sorted(TEMPLATE_MARKERS, key=len, reverse=True)) + r')\b',
    re.IGNORECASE
)

def ablate_none(text):
    return text

def ablate_panic(text):
    return " ".join(w for w in text.split() if w.lower().strip(".,!?;:'\"") not in PANIC)

def ablate_entities(text):
    return " ".join(w for w in text.split() if w.lower().strip(".,!?;:'\"") not in PANIC | ENTITIES)

def ablate_template(text):
    # Remove both lexical markers and template-pattern words
    text = " ".join(w for w in text.split() if w.lower().strip(".,!?;:'\"") not in PANIC | ENTITIES)
    return EXCLUDE_PATTERNS.sub("", text)

def ablate_semantic(text):
    # Heaviest: remove all structural/syntactic markers, keep only content words > 4 chars
    text = ablate_template(text)
    return " ".join(w for w in text.split() if len(w.strip(".,!?;:'\"")) > 4 and not any(
        c.isdigit() for c in w))

ABLATIONS = [
    ("1. None (baseline)", ablate_none),
    ("2. Panic keywords", ablate_panic),
    ("3. + Entities", ablate_entities),
    ("4. + Template markers", ablate_template),
    ("5. Semantic-only", ablate_semantic),
]

# ── Train & evaluate ───────────────────────────────────────────
results = []

for clf_name, clf, vparams, cparams in [
    ("TF-IDF+LR", LogisticRegression, {"max_features": 5000, "ngram_range": (1, 2)},
     {"random_state": 42, "max_iter": 1000}),
    ("GBDT", GradientBoostingClassifier, {"max_features": 3000, "ngram_range": (1, 2)},
     {"random_state": 42, "n_estimators": 200, "max_depth": 4, "learning_rate": 0.1}),
]:
    print(f"\n{clf_name}:")
    model_f1s = []
    for ab_name, ab_fn in ABLATIONS:
        X_tr = np.array([ab_fn(t) for t in train_df["text"].fillna("")])
        X_te = np.array([ab_fn(t) for t in test_df["text"].fillna("")])

        vec = TfidfVectorizer(**vparams)
        X_tr_vec = vec.fit_transform(X_tr)
        model = clf(**cparams)
        model.fit(X_tr_vec, train_df["label"])
        X_te_vec = vec.transform(X_te)
        preds = model.predict(X_te_vec)
        f1 = f1_score(test_df["label"], preds, zero_division=0)

        model_f1s.append(f1)
        print(f"  {ab_name:30s}: F1={f1:.4f}")

    results.append({"model": clf_name, "f1_values": model_f1s})

# ── Plot ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))
x = np.arange(len(ABLATIONS))
width = 0.35

colors = {"TF-IDF+LR": "#1f77b4", "GBDT": "#d62728"}
for i, r in enumerate(results):
    ax.bar(x + i * width - width/2, r["f1_values"], width, label=r["model"],
           color=colors[r["model"]], alpha=0.85, edgecolor="white")
    for j, v in enumerate(r["f1_values"]):
        ax.text(j + i * width - width/2, v + 0.005, f"{v:.3f}",
                ha="center", fontsize=9, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels([a[0] for a in ABLATIONS], rotation=25, ha="right", fontsize=9)
ax.set_ylabel("F1 Score", fontsize=12)
ax.set_title("Progressive Leakage-Ablation: F1 vs. Feature Removal Aggressiveness",
            fontsize=13, fontweight="bold")
ax.legend(loc="lower left", framealpha=0.9)
ax.grid(True, alpha=0.3, axis="y")
ax.set_ylim([0, 1.05])

curve_path = "./plots/phase_22_leakage_curve.png"
fig.tight_layout()
fig.savefig(curve_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"\nLeakage curve -> {curve_path}")

# ── Save ───────────────────────────────────────────────────────
output = {"phase": "22_progressive_ablation", "results": results,
          "ablation_levels": [a[0] for a in ABLATIONS]}
out_path = "./output/phase_22_progressive_ablation.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"Results -> {out_path}")
