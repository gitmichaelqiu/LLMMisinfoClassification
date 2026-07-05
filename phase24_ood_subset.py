"""Phase 24: Post-Cutoff OOD Subset Evaluation."""
import os, json, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

os.makedirs("./output", exist_ok=True)

# Balanced training data
synth = pd.read_csv("./input/headlines.csv")
kaggle = pd.read_csv("./input/kaggle_fake_news_FULL.csv").dropna(subset=["text","label"])
n = min(len(synth), 200, len(kaggle))
s = synth[["headline","label"]].rename(columns={"headline":"text"}).sample(n=min(len(synth),n), random_state=42)
k0 = kaggle[kaggle["label"]==0].sample(n=min(len(kaggle[kaggle["label"]==0]), n//2), random_state=42)
k1 = kaggle[kaggle["label"]==1].sample(n=min(len(kaggle[kaggle["label"]==1]), n//2), random_state=42)
train, _ = train_test_split(pd.concat([s,k0[["text","label"]],k1[["text","label"]]], ignore_index=True),
                           test_size=0.2, stratify=pd.concat([s,k0[["text","label"]],k1[["text","label"]]])["label"], random_state=42)
train["label"] = train["label"].astype(int)
print(f"Train: {len(train)} samples")

# OOD examples with dates/subcategories
PC = [
    ("margins expanding on earnings call",0,"2024-03","earnings_call"),
    ("merger arb spread tightened to 20bps",0,"2024-06","merger_arb"),
    ("Q3 deferred revenue accelerating",0,"2024-08","reg_filing"),
    ("short interest covering this week",0,"2024-11","market_intel"),
    ("activist pushes for board changes",0,"2025-01","activist"),
    ("gamma squeeze setup identical to 2021",1,"2024-04","social_media"),
    ("CEO selling personal shares rumor",1,"2024-07","social_media"),
    ("unverified SEC charges document",1,"2024-10","social_media"),
    ("blog alleges fake revenue",1,"2025-01","blog_rumor"),
    ("layoffs announced on Blind",1,"2024-02","blind_post"),
]
PR = [
    ("quarterly results looking solid",0,"2023-06","earnings_call"),
    ("management confident on transcript",0,"2023-09","earnings_call"),
    ("Bloomberg merger close by quarter",0,"2023-03","merger_arb"),
    ("analyst price target upward",0,"2023-11","market_intel"),
    ("due diligence progressing",0,"2023-05","merger_arb"),
    ("CEO about to resign rumor",1,"2023-04","social_media"),
    ("massive undisclosed loss rumor",1,"2023-08","social_media"),
    ("WSB claims Chapter 11 filing",1,"2023-02","social_media"),
    ("leaked document overstated revenue",1,"2023-10","blog_rumor"),
    ("Blind employee posts layoff rumor",1,"2023-07","blind_post"),
]

post = pd.DataFrame([{"text":e[0],"label":e[1]} for e in PC])
pre = pd.DataFrame([{"text":e[0],"label":e[1]} for e in PR])

# Train
vec = TfidfVectorizer(max_features=3000); X=vec.fit_transform(train["text"].fillna(""))
lr=LogisticRegression(max_iter=1000,random_state=42).fit(X,train["label"])

results={}
for nm,df in [("pre",pre),("post",post)]:
    Xt=vec.transform(df["text"].fillna("")); p=lr.predict(Xt)
    f1=f1_score(df["label"],p,zero_division=0)
    results[nm]=round(f1,4); print(f"  {nm}: F1={f1:.4f}")

# Subcategories
sc={}
for e in PC+PR:
    sc.setdefault(e[3],[]).append(e[0])
sc_r={}
for cat,txts in sc.items():
    if len(txts)<2: continue
    labs=[1 if any(w in t.lower() for w in["rumor","claims","alleges","anonymous","random","someone","unverified","blog","fraud","fake","leaked","overstated"]) else 0 for t in txts]
    Xt=vec.transform(txts); p=lr.predict(Xt)
    sc_r[cat]=round(f1_score(labs,p,zero_division=0),4)
    print(f"  {cat:15s}: F1={sc_r[cat]:.4f}")

out={"phase":"24_ood_subset","results":results,"subcategories":sc_r}
with open("./output/phase_24_ood_subset.json","w") as f: json.dump(out,f,indent=2)
print(f"Saved -> ./output/phase_24_ood_subset.json")
