"""Phase 24: Verify Ablation Cliff (Stage 5.5).

Implements Stage 5.5: ablate residual non-keyword vocabulary while
retaining original keyword tokens. Maps F1 against fractional
vocabulary retention curve (90% down to 0%).
"""

import os, json, warnings, re, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

os.makedirs("./output", exist_ok=True)

# Data
synth = pd.read_csv("./input/headlines.csv"); kaggle = pd.read_csv("./input/kaggle_fake_news_FULL.csv").dropna(subset=["text","label"])
n = min(len(synth), 300, len(kaggle))
s = synth[["headline","label"]].rename(columns={"headline":"text"}).sample(n=min(len(synth),n), random_state=42)
k0 = kaggle[kaggle["label"]==0].sample(n=min(len(kaggle[kaggle["label"]==0]), n//2), random_state=42)
k1 = kaggle[kaggle["label"]==1].sample(n=min(len(kaggle[kaggle["label"]==1]), n//2), random_state=42)
df = pd.concat([s,k0[["text","label"]],k1[["text","label"]]], ignore_index=True)
df["label"]=df["label"].astype(int)
train,test = train_test_split(df,test_size=0.2,stratify=df["label"],random_state=42)

# Ablation functions
from src.system0_filter import System0Filter
s0 = System0Filter(enabled=True)
PANIC = {w.lower().strip(".,!?;:'\"") for w in s0.PANIC_KEYWORDS if len(w) > 3}
ENTITIES = {w.lower().strip(".,!?;:'\"") for w in s0.HIGH_IMPACT_ENTITIES if len(w) > 3}
TEMPLATE_MARKERS = {"reports","announces","posts","surges","plunges","soars","drops","rises","falls",
    "gains","declines","acquires","partners","secures","receives","confirms","expects","plans",
    "launches","unveils","introduces","quarter","fiscal","annual","guidance","forecast","outlook",
    "dividend","buyback","offering","listing","rating","upgrade","downgrade","target","estimate",
    "consensus","analyst","billion","million","percent","growth","momentum"}
ALL_MARKERS = PANIC | ENTITIES | {w.lower() for w in TEMPLATE_MARKERS}

def ablate_retain_keywords(text, retention_pct=1.0):
    """Remove random fraction of NON-keyword vocabulary, retain keywords."""
    words = text.split()
    kept = []
    rng = np.random.default_rng(42)
    for w in words:
        wl = w.lower().strip(".,!?;:'\"")
        if wl in ALL_MARKERS or len(wl) <= 3:
            kept.append(w)  # always keep keywords and short words
        elif rng.random() < retention_pct:
            kept.append(w)
    return " ".join(kept)

# Stage 5.5: retain only keywords (remove ALL non-keyword vocabulary)
def ablate_stage55(text):
    return ablate_retain_keywords(text, retention_pct=0.0)

# Vocabulary retention curve
retention_levels = [1.0, 0.75, 0.50, 0.25, 0.10, 0.05, 0.01, 0.0]
print("Vocabulary Retention Curve:")
results = []
for model_name, clf, vp, cp in [
    ("LR", LogisticRegression, {"max_features":5000}, {"max_iter":1000,"random_state":42}),
    ("GBDT", GradientBoostingClassifier, {"max_features":3000}, {"n_estimators":200,"max_depth":4,"random_state":42}),
]:
    model_f1s = []
    for ret in retention_levels:
        def ablate_fn(t, r=ret): return ablate_retain_keywords(t, r)
        X_tr = np.array([ablate_fn(t) for t in train["text"].fillna("")])
        X_te = np.array([ablate_fn(t) for t in test["text"].fillna("")])
        vec = TfidfVectorizer(**vp)
        X_tr_v = vec.fit_transform(X_tr)
        model = clf(**cp)
        model.fit(X_tr_v, train["label"])
        X_te_v = vec.transform(X_te)
        f1 = f1_score(test["label"], model.predict(X_te_v), zero_division=0)
        model_f1s.append(f1)
        print(f"  {model_name:5s} retention={ret:.0%}: F1={f1:.4f}")
    results.append({"model":model_name,"f1_values":model_f1s})

# Also run Stage 5.5 (0% retention but with explicit keyword preservation)
print("\nStage 5.5 (keywords-only):")
for model_name, clf, vp, cp in [
    ("LR", LogisticRegression, {"max_features":5000}, {"max_iter":1000,"random_state":42}),
    ("GBDT", GradientBoostingClassifier, {"max_features":3000}, {"n_estimators":200,"max_depth":4,"random_state":42}),
]:
    X_tr = np.array([ablate_stage55(t) for t in train["text"].fillna("")])
    X_te = np.array([ablate_stage55(t) for t in test["text"].fillna("")])
    vec = TfidfVectorizer(**vp)
    model = clf(**cp)
    model.fit(vec.fit_transform(X_tr), train["label"])
    f1 = f1_score(test["label"], model.predict(vec.transform(X_te)), zero_division=0)
    print(f"  {model_name:5s} Stage 5.5: F1={f1:.4f}")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(10,6))
for r in results:
    ax.plot([l*100 for l in retention_levels], r["f1_values"], marker="o", label=r["model"], lw=2)
ax.axhline(0.5, color="gray", ls=":", alpha=0.5, label="Random")
ax.set_xlabel("Vocabulary Retention (%)"); ax.set_ylabel("F1 Score")
ax.set_title("Ablation Cliff: F1 vs Vocabulary Retention", fontweight="bold")
ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlim([-5,105])
plt.tight_layout(); plt.savefig("./plots/phase_24_ablation_cliff.png", dpi=200); plt.close()
print("\nPlot -> ./plots/phase_24_ablation_cliff.png")

out={"phase":"24_ablation_cliff","retention_levels":retention_levels,"results":results}
with open("./output/phase_24_ablation_cliff.json","w") as f: json.dump(out,f,indent=2)
print("Saved -> ./output/phase_24_ablation_cliff.json")
