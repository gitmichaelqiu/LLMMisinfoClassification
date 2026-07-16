"""Continuation script: runs only remaining experiments.

Completed in previous run (finance):
  - TF-IDF, SS OFF/ON, Voting N=1/3/5/7 OFF/ON, MoA RAG OFF

Remaining:
  - Finance: MoA RAG ON
  - Healthcare: ALL experiments (TF-IDF, SS OFF/ON, Voting OFF/ON, MoA OFF/ON)
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

for _var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
    os.environ.pop(_var, None)
os.environ["no_proxy"] = "*"
os.environ["NO_PROXY"] = "*"

from src.metrics import compute_confusion_matrix, compute_ece
from src.schemas import Verdict, VerificationItem, VerificationResult

load_dotenv()

SEED = 42
MODEL = "deepseek-v4-flash"
TEMPERATURE = 0.7
MAX_TOKENS = 512
MAX_CONCURRENCY = 2000
OUTPUT_DIR = "results/final_1000_validation"
COST_PER_CALL = 0.00015
PPV_BASE_RATES = [0.001, 0.01, 0.05, 0.10, 0.25, 0.50]
COST_RATIOS = [(1, 1), (1, 5), (1, 10), (1, 25)]

FINANCE_TEST = "data/raw/finance/finance_test_500.csv"
FINANCE_CORPUS = "data/raw/finance/finance_corpus.csv"
COVID_TEST = "data/raw/health/covid_test_500.csv"
COVID_CORPUS = "data/raw/health/covid_corpus.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "raw_outputs"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)


def _llm_call(system_prompt, user_prompt, model=MODEL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS):
    import httpx
    from openai import OpenAI
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    http_client = httpx.Client(proxy=None, timeout=httpx.Timeout(180.0, connect=30.0), follow_redirects=True)
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1", http_client=http_client)
    start = time.time()
    try:
        resp = client.chat.completions.create(
            model=model, temperature=temperature, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        )
        return resp.choices[0].message.content or "", time.time() - start
    except Exception as e:
        raise RuntimeError(f"API call failed: {e}")
    finally:
        http_client.close()


_VERDICT_RE = re.compile(r"Verdict:\s*(REAL|FAKE|ESCALATE|EXAGGERATED)", re.IGNORECASE)
_CONFIDENCE_RE = re.compile(r"Confidence:\s*(\d+)", re.IGNORECASE)
_VERDICT_MAP = {"REAL": Verdict.REAL, "FAKE": Verdict.FAKE, "ESCALATE": Verdict.ESCALATE, "EXAGGERATED": Verdict.EXAGGERATED}


def _parse_response(raw, item_id, latency_s, extra_metadata=None):
    verdict = Verdict.REAL
    vm = _VERDICT_RE.search(raw)
    if vm:
        verdict = _VERDICT_MAP.get(vm.group(1).upper(), Verdict.REAL)
    confidence = 0.5
    cm = _CONFIDENCE_RE.search(raw)
    if cm:
        confidence = int(cm.group(1)) / 100.0
    return VerificationResult(
        item_id=item_id, verdict=verdict, confidence=max(0.0, min(1.0, confidence)),
        latency_s=latency_s, evidence=[], metadata=extra_metadata or {},
    )


def load_test_csv(path, domain):
    with open(path, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    items = []
    for r in rows:
        text = r.get("headline", r.get("title", r.get("tweet", r.get("text", ""))))
        raw_label = r.get("label", "").strip()
        if raw_label in ("0", "1"):
            gt = Verdict.FAKE if raw_label == "1" else Verdict.REAL
        elif raw_label.lower() in ("real", "fake"):
            gt = Verdict.FAKE if raw_label.lower() == "fake" else Verdict.REAL
        else:
            continue
        items.append(VerificationItem.create(claim_text=text, ground_truth=gt, metadata={"domain": domain}))
    return items


def load_corpus_csv(path, domain):
    with open(path, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    items = []
    for r in rows:
        text = r.get("headline", r.get("title", r.get("tweet", r.get("text", ""))))
        raw_label = r.get("label", "").strip()
        if raw_label in ("0", "1"):
            gt = Verdict.FAKE if raw_label == "1" else Verdict.REAL
        elif raw_label.lower() in ("real", "fake"):
            gt = Verdict.FAKE if raw_label.lower() == "fake" else Verdict.REAL
        else:
            continue
        items.append(VerificationItem.create(claim_text=text, ground_truth=gt, metadata={"domain": domain}))
    return items


def _run_parallel(callables, desc=""):
    n = len(callables)
    results = [None] * n
    with ThreadPoolExecutor(max_workers=min(MAX_CONCURRENCY, n)) as ex:
        fut_to_idx = {ex.submit(fn): i for i, fn in enumerate(callables)}
        for fut in as_completed(fut_to_idx):
            idx = fut_to_idx[fut]
            try:
                results[idx] = fut.result()
            except Exception:
                results[idx] = None
    return results


# Same prompts as main script
CANONICAL_SYSTEM = (
    "You are an Information Authenticity Verifier. Determine whether a given claim "
    "is authentic (REAL) or contains misinformation (FAKE).\n\n"
    "Analyze the claim for:\n"
    "1. Internal consistency\n"
    "2. Plausibility\n"
    "3. Specificity\n"
    "4. Temporal consistency\n\n"
    "Output in EXACTLY this format:\n"
    "Verdict: REAL or FAKE or ESCALATE\n"
    "Confidence: <0-100>\n"
    "Flags: [contradiction|implausibility|inconsistency|none]\n"
    "Reasoning: <one-sentence rationale>"
)

MOA_SUPPORTER = (
    "You are Agent 1 (The Supporter). Argue the claim is REAL.\n\n"
    "Build the strongest case for authenticity:\n"
    "1. Internal consistency\n"
    "2. Plausibility\n"
    "3. Factual alignment with known information\n"
    "4. Source credibility signals\n\n"
    "Output:\n"
    "Verdict: REAL or UNCERTAIN\n"
    "Confidence: <0-100>\n"
    "Reasoning: <your argument>"
)

MOA_SKEPTIC = (
    "You are Agent 2 (The Skeptic). Argue the claim is FAKE.\n\n"
    "Build the strongest case against authenticity:\n"
    "1. Logical contradictions\n"
    "2. Implausibility\n"
    "3. Inconsistencies\n"
    "4. Red flags or hallmarks of misinformation\n\n"
    "Output:\n"
    "Verdict: FAKE or UNCERTAIN\n"
    "Confidence: <0-100>\n"
    "Reasoning: <your argument>"
)

MOA_JUDGE = (
    "You are the Judge. Read both arguments and deliver a final verdict.\n\n"
    "You have analyses from:\n"
    "- The Supporter (argues REAL)\n"
    "- The Skeptic (argues FAKE)\n\n"
    "Weigh evidence QUALITY, not just presence.\n"
    "Default to REAL if Supporter has stronger evidence.\n"
    "Default to FAKE if Skeptic has clear contradictions.\n"
    "Default to ESCALATE if arguments are balanced or uncertain.\n\n"
    "Output in EXACTLY this format:\n"
    "Verdict: REAL or FAKE or ESCALATE\n"
    "Confidence: <0-100>\n"
    "Flags: [...]\n"
    "Reasoning: <one-sentence rationale>"
)

MOA_SUPPORTER_RAG = (
    "You are Agent 1 (The Supporter). Argue the claim is REAL.\n\n"
    "You have access to RETRIEVED EVIDENCE from a knowledge corpus. Use it to build your case.\n\n"
    "Build the strongest case for authenticity:\n"
    "1. Does any evidence support the claim?\n"
    "2. Source credibility\n"
    "3. Factual alignment\n"
    "4. Plausibility\n\n"
    "Output:\n"
    "Verdict: REAL or UNCERTAIN\n"
    "Evidence: <bulleted list>\n"
    "Confidence: <0-100>"
)

MOA_SKEPTIC_RAG = (
    "You are Agent 2 (The Skeptic). Argue the claim is FAKE.\n\n"
    "You have access to RETRIEVED EVIDENCE. Scrutinize it for contradictions.\n\n"
    "Build the strongest case against authenticity:\n"
    "1. Does any evidence contradict the claim?\n"
    "2. Implausibility\n"
    "3. Inconsistencies\n"
    "4. Red flags\n\n"
    "Output:\n"
    "Verdict: FAKE or UNCERTAIN\n"
    "Evidence: <bulleted list>\n"
    "Confidence: <0-100>"
)

MOA_JUDGE_RAG = (
    "You are the Judge. Read both arguments and the retrieved evidence, then deliver a final verdict.\n\n"
    "You have analyses from:\n"
    "- The Supporter (argues REAL, with evidence)\n"
    "- The Skeptic (argues FAKE, with evidence)\n"
    "Evidence from the knowledge corpus is available.\n\n"
    "Weigh evidence QUALITY, not just presence.\n"
    "Default to REAL if Supporter has stronger evidence.\n"
    "Default to FAKE if Skeptic has clear contradictions.\n"
    "Default to ESCALATE if arguments are balanced or uncertain.\n\n"
    "Output in EXACTLY this format:\n"
    "Verdict: REAL or FAKE or ESCALATE\n"
    "Confidence: <0-100>\n"
    "Flags: [...]\n"
    "Reasoning: <one-sentence rationale>"
)


def _build_retriever(corpus):
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts = [it.claim_text for it in corpus]
    vec = TfidfVectorizer(stop_words="english", max_features=5000)
    return vec, vec.fit_transform(texts), texts


def _retrieve(claim, vec, tfidf, texts, top_k=5):
    from sklearn.metrics.pairwise import cosine_similarity
    qv = vec.transform([claim])
    sims = cosine_similarity(qv, tfidf).flatten()
    parts = []
    for idx in np.argsort(sims)[::-1][:top_k]:
        if sims[idx] > 0:
            parts.append(f"[Doc {idx}] (rel={sims[idx]:.3f})\n{texts[idx]}\n")
    return "".join(parts)[:2000]


def _make_user_prompt(claim_text, rag_vec=None, rag_tfidf=None, rag_texts=None):
    if rag_vec is not None:
        evidence = _retrieve(claim_text, rag_vec, rag_tfidf, rag_texts)
        if evidence:
            return f"Claim to verify:\n{claim_text}\n\n── Retrieved Evidence ──\n{evidence}\n────────────────────\n\nIs this claim REAL or FAKE?"
    return f"Claim to verify:\n{claim_text}\n\nIs this claim REAL or FAKE?"


# TF-IDF baseline
def run_tfidf_baseline(items, corpus):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    corpus_texts = [it.claim_text for it in corpus]
    corpus_labels = [1 if it.ground_truth == Verdict.FAKE else 0 for it in corpus]
    clf = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=10000)),
        ("lr", LogisticRegression(class_weight="balanced", random_state=SEED, max_iter=1000)),
    ])
    clf.fit(corpus_texts, corpus_labels)
    results = []
    for item in items:
        pred = clf.predict([item.claim_text])[0]
        prob = clf.predict_proba([item.claim_text])[0]
        verdict, confidence = (Verdict.FAKE, prob[1]) if pred == 1 else (Verdict.REAL, prob[0])
        results.append(VerificationResult(
            item_id=item.id, verdict=verdict, confidence=float(confidence),
            latency_s=0.001, evidence=[], metadata={"architecture": "tfidf_baseline"},
        ))
    return results


def run_single_shot(items, rag_on=False, corpus=None):
    vec = tfidf = texts = None
    if rag_on and corpus:
        vec, tfidf, texts = _build_retriever(corpus)
    def make_call(item):
        up = _make_user_prompt(item.claim_text, vec, tfidf, texts)
        raw, lat = _llm_call(CANONICAL_SYSTEM, up)
        return _parse_response(raw, item.id, lat, {"architecture": "single_shot", "rag": rag_on})
    raw = _run_parallel([lambda it=item: make_call(it) for item in items], "SS")
    return [r if r else VerificationResult(item_id=it.id, verdict=Verdict.REAL, confidence=0.5, latency_s=0, metadata={"error": "failure"}) for r, it in zip(raw, items)]


def run_voting_all_n(items, rag_on=False, corpus=None, n_total=7):
    vec = tfidf = texts = None
    if rag_on and corpus:
        vec, tfidf, texts = _build_retriever(corpus)
    callables = []
    for item in items:
        up = _make_user_prompt(item.claim_text, vec, tfidf, texts)
        for v in range(n_total):
            def _vote(it=item, u=up, vidx=v):
                txt, lat = _llm_call(CANONICAL_SYSTEM, u)
                return (it.id, vidx, _parse_response(txt, it.id, lat, {"voter_idx": vidx, "architecture": "voting", "rag": rag_on}))
            callables.append(_vote)
    raw = _run_parallel(callables, f"Voting N={n_total}")
    item_voters_all = defaultdict(list)
    for r in raw:
        if r is not None:
            _, vidx, vr = r
            item_voters_all[vr.item_id].append(vr)
    for iid in item_voters_all:
        item_voters_all[iid].sort(key=lambda x: x.metadata.get("voter_idx", 0))
    result_map = {}
    for N in [1, 3, 5, 7]:
        threshold = N // 2 + 1
        aggregated, per_item_voters = [], []
        for item in items:
            voters = item_voters_all.get(item.id, [])[:N]
            per_item_voters.append(voters)
            if not voters:
                aggregated.append(VerificationResult(item_id=item.id, verdict=Verdict.REAL, confidence=0.5, latency_s=0, metadata={"architecture": f"voting_n{N}", "error": "no_voters"}))
                continue
            counter = Counter(v.verdict for v in voters)
            mc = counter.most_common(1)
            verdict = mc[0][0] if (mc and mc[0][1] >= threshold) else Verdict.ESCALATE
            aggregated.append(VerificationResult(
                item_id=item.id, verdict=verdict, confidence=float(np.mean([v.confidence for v in voters])),
                latency_s=max(v.latency_s for v in voters),
                evidence=[f"N={N}: {dict(counter)}"],
                metadata={"architecture": f"voting_n{N}", "rag": rag_on, "n_voters": len(voters), "threshold": threshold,
                          "verdict_distribution": {k.name: v for k, v in counter.items()},
                          "disagreement_rate": 1.0 - (mc[0][1] / len(voters)) if mc else 1.0},
            ))
        result_map[N] = (aggregated, per_item_voters)
    return result_map


def run_moa(items, rag_on=False, corpus=None):
    vec = tfidf = texts = None
    if rag_on and corpus:
        vec, tfidf, texts = _build_retriever(corpus)
    evidence_map = {}
    p1_callables = []
    for item in items:
        up = _make_user_prompt(item.claim_text, vec, tfidf, texts)
        evidence_map[item.id] = up
        sp = MOA_SUPPORTER_RAG if rag_on else MOA_SUPPORTER
        sk = MOA_SKEPTIC_RAG if rag_on else MOA_SKEPTIC
        def _sup(iid=item.id, u=up, p=sp):
            txt, lat = _llm_call(p, u)
            return ("supporter", iid, txt, lat)
        def _ske(iid=item.id, u=up, p=sk):
            txt, lat = _llm_call(p, u)
            return ("skeptic", iid, txt, lat)
        p1_callables.append(_sup); p1_callables.append(_ske)
    raw_p1 = _run_parallel(p1_callables, "MoA P1")
    supporter_out, skeptic_out = {}, {}
    for result in raw_p1:
        if result is None: continue
        role, iid, text, lat = result
        if role == "supporter": supporter_out[iid] = (text, lat)
        else: skeptic_out[iid] = (text, lat)
    judge_prompt = MOA_JUDGE_RAG if rag_on else MOA_JUDGE
    p2_callables = []
    for item in items:
        up = evidence_map.get(item.id, f"Claim to verify:\n{item.claim_text}")
        sup_text, sup_lat = supporter_out.get(item.id, ("Error", 0))
        ske_text, ske_lat = skeptic_out.get(item.id, ("Error", 0))
        p1_lat = max(sup_lat, ske_lat)
        ctx = f"{up}\n\n── Supporter's Analysis ──\n{sup_text}\n\n── Skeptic's Analysis ──\n{ske_text}\n"
        def _judge(iid=item.id, c=ctx, p1=p1_lat):
            txt, lat = _llm_call(judge_prompt, c)
            vr = _parse_response(txt, iid, lat, {"architecture": "moa", "rag": rag_on})
            vr.latency_s = p1 + lat
            return (iid, vr)
        p2_callables.append(_judge)
    raw_p2 = _run_parallel(p2_callables, "MoA P2")
    return [vr for r in raw_p2 if r is not None for _, vr in [r]]


# ── Metrics ─────────────────────────────────────────────────────────────

def _latency_stats(latencies):
    if not latencies: return {"mean": 0, "median": 0, "p95": 0, "min": 0, "max": 0}
    return {"mean": float(np.mean(latencies)), "median": float(np.median(latencies)),
            "p95": float(np.percentile(latencies, 95)), "min": float(min(latencies)), "max": float(max(latencies))}


def _bootstrap_ci(items, results, metric_fn, n_iter=2000, alpha=0.05):
    truths = [it.ground_truth or Verdict.REAL for it in items]
    paired = [(r, t) for r, t in zip(results, truths)]
    if len(paired) < 2: return (0.0, 0.0)
    rng = np.random.RandomState(SEED)
    vals = []
    for _ in range(n_iter):
        idx = rng.choice(len(paired), len(paired), replace=True)
        try: vals.append(metric_fn([paired[i][0] for i in idx], [paired[i][1] for i in idx]))
        except Exception: continue
    if len(vals) < 2: return (0.0, 0.0)
    return (float(np.percentile(vals, alpha/2*100)), float(np.percentile(vals, (1-alpha/2)*100)))


def _f1_metric(r, t):
    cm = compute_confusion_matrix(r, t)
    p = cm.tp / (cm.tp + cm.fp) if (cm.tp + cm.fp) else 0.0
    r_ = cm.tp / (cm.tp + cm.fn) if (cm.tp + cm.fn) else 0.0
    return 2 * p * r_ / (p + r_) if (p + r_) else 0.0


def _prec_metric(r, t):
    cm = compute_confusion_matrix(r, t)
    return cm.tp / (cm.tp + cm.fp) if (cm.tp + cm.fp) else 0.0


def _rec_metric(r, t):
    cm = compute_confusion_matrix(r, t)
    return cm.tp / (cm.tp + cm.fn) if (cm.tp + cm.fn) else 0.0


def evaluate_architecture(items, results, arch_name, total_api_calls):
    truths = [it.ground_truth or Verdict.REAL for it in items]
    latencies = [r.latency_s for r in results if r.latency_s > 0]
    section = {"n": len(items), "n_fake": sum(1 for it in items if it.ground_truth == Verdict.FAKE), "n_real": sum(1 for it in items if it.ground_truth == Verdict.REAL)}
    try:
        from src.metrics import classification_metrics
        m = classification_metrics(results, truths)
        section["metrics"] = {"precision": m.precision, "recall": m.recall, "f1": m.f1, "fpr": m.fpr, "fnr": m.fnr, "accuracy": m.accuracy, "n_total": m.n_total}
        section["verdict_distribution"] = dict(Counter(r.verdict.name for r in results))
        n_esc = sum(1 for r in results if r.verdict == Verdict.ESCALATE)
        section["escalate_rate"] = n_esc / len(results) if results else 0.0
        correct = [r.verdict == t for r, t in zip(results, truths)]
        ece_val, _, _, _ = compute_ece([r.confidence for r in results], correct, n_bins=5)
        section["ece"] = ece_val
        section["ppv"] = [{"base_rate": br, "ppv": round((m.recall * br) / (m.recall * br + m.fpr * (1-br)), 6) if (m.recall * br + m.fpr * (1-br)) > 0 else 0.0} for br in PPV_BASE_RATES]
        section["expected_cost"] = {f"FP{cfp}_FN{cfn}": [{"base_rate": br, "expected_cost": round((1-br) * m.fpr * cfp + br * m.fnr * cfn, 6)} for br in PPV_BASE_RATES] for cfp, cfn in COST_RATIOS}
        section["confidence_intervals"] = {"f1_95": list(_bootstrap_ci(items, results, _f1_metric)), "precision_95": list(_bootstrap_ci(items, results, _prec_metric)), "recall_95": list(_bootstrap_ci(items, results, _rec_metric))}
    except Exception as e:
        section["metrics_error"] = str(e)
    section["latency"] = _latency_stats(latencies)
    section["total_api_calls"] = total_api_calls
    return section


def analyze_voter_agreement(items, per_item_voters, n_voters):
    truths = [it.ground_truth or Verdict.REAL for it in items]
    stats = {"total_items": len(items), "n_voters": n_voters, "unanimous": 0, "majority_flip_v1": 0,
             "corrected_by_voting": 0, "damaged_by_voting": 0, "pairwise_agreement": 0.0,
             "majority_verdicts": {}, "voter1_verdicts": {}, "voter_fake_counts": []}
    pairwise_agree, pairwise_total = 0, 0
    threshold = n_voters // 2 + 1
    for i, voters in enumerate(per_item_voters):
        if not voters or len(voters) < n_voters: continue
        gt = truths[i] if i < len(truths) else None
        verdicts = [v.verdict for v in voters]; v1 = verdicts[0]
        if len(set(verdicts)) == 1: stats["unanimous"] += 1
        counter = Counter(verdicts); majority_v, majority_c = counter.most_common(1)[0]
        if majority_c >= threshold: stats["majority_verdicts"][majority_v.name] = stats["majority_verdicts"].get(majority_v.name, 0) + 1
        else: stats["majority_verdicts"]["ESCALATE"] = stats["majority_verdicts"].get("ESCALATE", 0) + 1
        stats["voter1_verdicts"][v1.name] = stats["voter1_verdicts"].get(v1.name, 0) + 1
        maj_v = majority_v if majority_c >= threshold else Verdict.ESCALATE
        if v1 != Verdict.ESCALATE and maj_v != Verdict.ESCALATE and maj_v != v1 and n_voters > 1: stats["majority_flip_v1"] += 1
        for a in range(len(verdicts)):
            for b in range(a+1, len(verdicts)):
                pairwise_total += 1
                if verdicts[a] == verdicts[b]: pairwise_agree += 1
        if gt is not None:
            if v1 != gt and maj_v == gt: stats["corrected_by_voting"] += 1
            if v1 == gt and maj_v != gt and n_voters > 1: stats["damaged_by_voting"] += 1
    stats["pairwise_agreement"] = pairwise_agree / pairwise_total if pairwise_total > 0 else 0
    for vi in range(n_voters):
        stats["voter_fake_counts"].append(sum(1 for voters in per_item_voters if vi < len(voters) and voters[vi].verdict == Verdict.FAKE))
    return stats


def save_raw_output(domain, arch_name, items, results, output_dir):
    path = os.path.join(output_dir, "raw_outputs", f"{domain}_{arch_name}.json")
    with open(path, "w") as f:
        json.dump({"architecture": arch_name, "domain": domain, "model": MODEL,
                   "results": [{"item_id": r.item_id, "verdict": r.verdict.name, "confidence": r.confidence, "latency_s": r.latency_s, "metadata": r.metadata} for r in results]}, f, indent=2, default=str)
    return path


def print_metrics(section, label=""):
    m = section.get("metrics", {})
    ci = section.get("confidence_intervals", {})
    ci_str = f"  F1 CI=({ci.get('f1_95',[0,0])[0]:.3f},{ci.get('f1_95',[0,0])[1]:.3f})" if ci.get("f1_95") else ""
    print(f"    {label:30s} F1={m.get('f1',0):.4f}  P={m.get('precision',0):.4f}  R={m.get('recall',0):.4f}  FPR={m.get('fpr',0):.4f}  FNR={m.get('fnr',0):.4f}  Acc={m.get('accuracy',0):.4f}  ESC={section.get('escalate_rate',0):.2%}  ECE={section.get('ece',0):.4f}  Lat={section.get('latency',{}).get('mean',0):.1f}s{ci_str}")
    sys.stdout.flush()


def load_existing_results(domain, output_dir):
    """Load existing raw result files from a previous run."""
    prefix = f"{domain}_"
    results = {}
    raw_dir = os.path.join(output_dir, "raw_outputs")
    if not os.path.isdir(raw_dir):
        return results
    for fname in os.listdir(raw_dir):
        if not fname.startswith(prefix) or not fname.endswith(".json"):
            continue
        arch_name = fname[len(prefix):-5]  # strip domain_ prefix and .json
        with open(os.path.join(raw_dir, fname)) as f:
            data = json.load(f)
        # Reconstruct VerificationResult objects
        loaded = []
        for rd in data.get("results", []):
            meta = dict(rd.get("metadata", {}) or {})
            vr = VerificationResult(
                item_id=rd["item_id"],
                verdict=Verdict[rd["verdict"]],
                confidence=rd.get("confidence", 0.5),
                latency_s=rd.get("latency_s", 0),
                evidence=[],
                metadata=meta,
            )
            loaded.append(vr)
        results[arch_name] = loaded
        print(f"  Loaded existing: {arch_name} ({len(loaded)} items)")
    return results


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    global_start = time.time()
    print("=" * 70)
    print("FINAL 1000-ITEM VALIDATION — CONTINUATION")
    print("=" * 70)
    print(f"Model: {MODEL}, Temperature: {TEMPERATURE}, Concurrency: {MAX_CONCURRENCY}")
    sys.stdout.flush()

    # Verify API
    print("\nVerifying API...")
    try:
        raw, lat = _llm_call("You are a test assistant.", "Reply with OK.")
        print(f"  API OK ({lat:.1f}s)")
    except Exception as e:
        print(f"  API ERROR: {e}"); return

    # Load test data
    print("\nLoading data...")
    finance_test = load_test_csv(FINANCE_TEST, "finance")
    finance_corpus = load_corpus_csv(FINANCE_CORPUS, "finance")
    covid_test = load_test_csv(COVID_TEST, "healthcare")
    covid_corpus = load_corpus_csv(COVID_CORPUS, "healthcare")
    print(f"  Finance: {len(finance_test)} test, {len(finance_corpus)} corpus")
    print(f"  COVID:   {len(covid_test)} test, {len(covid_corpus)} corpus")

    # Load existing results
    print("\nLoading existing results...")
    finance_existing = load_existing_results("finance", OUTPUT_DIR)
    covid_existing = load_existing_results("healthcare", OUTPUT_DIR)

    all_results = {"finance": {}, "healthcare": {}}
    total_api_calls = 0

    # ═══ FINANCE ═══════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("FINANCE — running MoA RAG ON (only missing experiment)")
    print(f"{'='*65}")

    # Re-evaluate existing finance results
    for arch, raw_results in finance_existing.items():
        api_count = len(raw_results)
        # Estimate API calls from architecture name
        if "voting_n7" in arch: api_count = len(finance_test) * 7
        elif "voting_n5" in arch: api_count = len(finance_test) * 7  # generated all 7 together
        elif "voting_n3" in arch: api_count = len(finance_test) * 7
        elif "voting_n1" in arch: api_count = len(finance_test) * 7
        elif "moa" in arch: api_count = len(finance_test) * 3
        elif "single_shot" in arch: api_count = len(finance_test)
        elif "tfidf" in arch: api_count = 0

        section = evaluate_architecture(finance_test, raw_results, arch, api_count)
        section["_cached_results"] = raw_results
        all_results["finance"][arch] = section
        print_metrics(section, arch)

    # MoA RAG ON for finance
    if "moa_rag_on" not in finance_existing:
        print(f"\n  MoA RAG ON (finance)..."); sys.stdout.flush()
        t0 = time.time()
        moa_on = run_moa(finance_test, rag_on=True, corpus=finance_corpus)
        elapsed = time.time() - t0
        arch = "moa_rag_on"
        all_results["finance"][arch] = evaluate_architecture(finance_test, moa_on, arch, len(finance_test)*3)
        all_results["finance"][arch]["elapsed_s"] = round(elapsed, 1)
        all_results["finance"][arch]["_cached_results"] = moa_on
        save_raw_output("finance", arch, finance_test, moa_on, OUTPUT_DIR)
        print_metrics(all_results["finance"][arch], "MoA RAG ON")
        total_api_calls += len(finance_test) * 3
    else:
        print(f"  MoA RAG ON already exists, skipping")

    # ═══ HEALTHCARE ═══════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"HEALTHCARE — running all experiments")
    print(f"{'='*65}")

    items, corpus = covid_test, covid_corpus
    total_api_calls_health = 0

    # Check which healthcare results already exist
    existing_health = set(covid_existing.keys())
    missing_health = [k for k in ["tfidf_baseline","single_shot_rag_off","single_shot_rag_on",
                                   "voting_n1_rag_off","voting_n3_rag_off","voting_n5_rag_off","voting_n7_rag_off",
                                   "voting_n1_rag_on","voting_n3_rag_on","voting_n5_rag_on","voting_n7_rag_on",
                                   "moa_rag_off","moa_rag_on"] if k not in existing_health]

    if existing_health:
        print(f"\n  Found {len(existing_health)} existing healthcare results, missing: {missing_health}")
        for arch, raw_results in covid_existing.items():
            api_est = len(items) * 7 if "voting" in arch else len(items) * 3 if "moa" in arch else len(items) if "single_shot" in arch else 0
            all_results["healthcare"][arch] = evaluate_architecture(items, raw_results, arch, api_est)
            all_results["healthcare"][arch]["_cached_results"] = raw_results
            print_metrics(all_results["healthcare"][arch], arch)

    # TF-IDF
    if "tfidf_baseline" not in all_results["healthcare"]:
        print(f"\n  TF-IDF Baseline..."); sys.stdout.flush()
        t0 = time.time()
        r = run_tfidf_baseline(items, corpus)
        all_results["healthcare"]["tfidf_baseline"] = evaluate_architecture(items, r, "tfidf_baseline", 0)
        all_results["healthcare"]["tfidf_baseline"]["_cached_results"] = r
        all_results["healthcare"]["tfidf_baseline"]["elapsed_s"] = time.time() - t0
        print_metrics(all_results["healthcare"]["tfidf_baseline"], "TF-IDF Baseline")

    # Single-Shot OFF/ON
    for rag_flag, rag_label in [(False,"RAG OFF"),(True,"RAG ON")]:
        arch = f"single_shot_{rag_label.lower().replace(' ','_')}"
        if arch not in all_results["healthcare"]:
            print(f"\n  Single-Shot {rag_label}..."); sys.stdout.flush()
            t0 = time.time()
            r = run_single_shot(items, rag_on=rag_flag, corpus=corpus if rag_flag else None)
            all_results["healthcare"][arch] = evaluate_architecture(items, r, arch, len(items))
            all_results["healthcare"][arch]["elapsed_s"] = time.time() - t0
            all_results["healthcare"][arch]["_cached_results"] = r
            save_raw_output("healthcare", arch, items, r, OUTPUT_DIR)
            print_metrics(all_results["healthcare"][arch], f"SS {rag_label}")
            total_api_calls_health += len(items)

    # Voting (7 outputs -> N=1,3,5,7)
    for rag_flag, rag_label in [(False,"RAG OFF"),(True,"RAG ON")]:
        rl = rag_label.lower().replace(' ','_')
        arch_base = f"voting_n{{N}}_{rl}"
        all_exist = all(f"voting_n{N}_{rl}" in all_results["healthcare"] for N in [1,3,5,7])

        if not all_exist:
            print(f"\n  Voting {rag_label} (7 outputs per item)..."); sys.stdout.flush()
            t0 = time.time()
            voting_map = run_voting_all_n(items, rag_on=rag_flag, corpus=corpus if rag_flag else None, n_total=7)
            v_elapsed = time.time() - t0
            api_calls = len(items) * 7
            for N in [1, 3, 5, 7]:
                agg, per_voter = voting_map[N]
                arch = f"voting_n{N}_{rl}"
                all_results["healthcare"][arch] = evaluate_architecture(items, agg, arch, api_calls)
                all_results["healthcare"][arch]["elapsed_s"] = round(v_elapsed, 1)
                all_results["healthcare"][arch]["_cached_results"] = agg
                all_results["healthcare"][arch]["voter_agreement"] = analyze_voter_agreement(items, per_voter, N)
                save_raw_output("healthcare", arch, items, agg, OUTPUT_DIR)
                print_metrics(all_results["healthcare"][arch], f"Voting N={N} {rag_label}")
            total_api_calls_health += api_calls
        else:
            print(f"  Voting {rag_label} already complete")

    # MoA OFF/ON
    for rag_flag, rag_label in [(False,"RAG OFF"),(True,"RAG ON")]:
        arch = f"moa_{rag_label.lower().replace(' ','_')}"
        if arch not in all_results["healthcare"]:
            print(f"\n  MoA {rag_label}..."); sys.stdout.flush()
            t0 = time.time()
            r = run_moa(items, rag_on=rag_flag, corpus=corpus if rag_flag else None)
            all_results["healthcare"][arch] = evaluate_architecture(items, r, arch, len(items)*3)
            all_results["healthcare"][arch]["elapsed_s"] = time.time() - t0
            all_results["healthcare"][arch]["_cached_results"] = r
            save_raw_output("healthcare", arch, items, r, OUTPUT_DIR)
            print_metrics(all_results["healthcare"][arch], f"MoA {rag_label}")
            total_api_calls_health += len(items) * 3

    # Update total
    all_results["healthcare"]["_total_api_calls"] = total_api_calls_health

    # ══════════════════════════════════════════════════════════════════
    #  GENERATE FULL REPORT
    # ══════════════════════════════════════════════════════════════════

    total_runtime = time.time() - global_start
    total_calls = total_api_calls + total_api_calls_health

    # Print all tables
    print(f"\n\n{'='*70}")
    print("VOTING SIZE SENSITIVITY (RAG OFF)")
    print(f"{'='*70}")
    for domain in ["finance","healthcare"]:
        table = {}
        for N in [1,3,5,7]:
            key = f"voting_n{N}_rag_off"
            if key in all_results[domain]:
                s = all_results[domain][key]
                table[N] = {"rag_off": {"metrics": s.get("metrics",{}), "latency": s.get("latency",{}),
                                         "escalate_rate": s.get("escalate_rate",0), "ece": s.get("ece",0),
                                         "agreement": s.get("voter_agreement",{}),
                                         "confidence_intervals": s.get("confidence_intervals",{})}}
        print(f"\n  {domain.upper()}:")
        h = f"  {'N':6s} {'F1':8s} {'P':8s} {'R':8s} {'FPR':8s} {'FNR':8s} {'Acc':8s} {'ESC':7s} {'ECE':7s} {'Unanim':7s} {'PairAgr':7s} {'Lat':6s}"
        print(h); print(f"  {'-'*len(h.strip())}")
        for N in [1,3,5,7]:
            if N in table:
                s, m, lt = table[N]["rag_off"], table[N]["rag_off"]["metrics"], table[N]["rag_off"]["latency"]
                a = s.get("agreement",{})
                print(f"  {N:6d} {m.get('f1',0):.4f}  {m.get('precision',0):.4f}  {m.get('recall',0):.4f}  {m.get('fpr',0):.4f}  "
                      f"{m.get('fnr',0):.4f}  {m.get('accuracy',0):.4f}  {s.get('escalate_rate',0):.2%}  {s.get('ece',0):.4f}  "
                      f"{a.get('unanimous',0)/max(a.get('total_items',1),1):.2%}  {a.get('pairwise_agreement',0):.4f}  {lt.get('mean',0):.1f}s")

    print(f"\n\n{'='*70}")
    print("FINAL ARCHITECTURE COMPARISON")
    print(f"{'='*70}")
    cells = [("tfidf_baseline","TF-IDF"),("single_shot_rag_off","SS OFF"),("single_shot_rag_on","SS ON"),
             ("voting_n1_rag_off","V1 OFF"),("voting_n1_rag_on","V1 ON"),
             ("voting_n3_rag_off","V3 OFF"),("voting_n3_rag_on","V3 ON"),
             ("voting_n5_rag_off","V5 OFF"),("voting_n5_rag_on","V5 ON"),
             ("voting_n7_rag_off","V7 OFF"),("voting_n7_rag_on","V7 ON"),
             ("moa_rag_off","MoA OFF"),("moa_rag_on","MoA ON")]

    for domain in ["finance","healthcare"]:
        print(f"\n  {domain.upper()} Architecture Comparison:")
        h = f"  {'Architecture':12s} {'F1':8s} {'P':8s} {'R':8s} {'FPR':8s} {'FNR':8s} {'Acc':8s} {'ESC':7s} {'ECE':7s} {'Lat':6s} {'Calls':6s}"
        print(h); print(f"  {'-'*len(h.strip())}")
        for key, label in cells:
            s = all_results[domain].get(key,{})
            if not s or not s.get("metrics"): continue
            m, lt = s["metrics"], s["latency"]
            print(f"  {label:12s} {m.get('f1',0):.4f}  {m.get('precision',0):.4f}  {m.get('recall',0):.4f}  "
                  f"{m.get('fpr',0):.4f}  {m.get('fnr',0):.4f}  {m.get('accuracy',0):.4f}  "
                  f"{s.get('escalate_rate',0):.2%}  {s.get('ece',0):.4f}  "
                  f"{lt.get('mean',0):.1f}s  {s.get('total_api_calls',0):5d}")

    print(f"\n  Cross-domain mean F1:")
    for key, label in cells:
        f1s = [all_results[d].get(key,{}).get("metrics",{}).get("f1",0) for d in ["finance","healthcare"]]
        if any(f1s):
            print(f"    {label:12s} Fin={f1s[0]:.4f}  Health={f1s[1]:.4f}  Mean={float(np.mean(f1s)):.4f}")

    # Summary
    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"Total runtime: {total_runtime:.0f}s ({total_runtime/60:.1f} min)")
    print(f"Total API calls (continuation): ~{total_calls}")
    print(f"Results: {OUTPUT_DIR}/")
    sys.stdout.flush()

    # Save report
    report = {"metadata": {"experiment": "final_1000_validation", "model": MODEL, "temperature": TEMPERATURE,
                           "timestamp": datetime.now().isoformat(), "total_runtime_s": total_runtime},
              "data_summary": {"finance": {"test": 500, "corpus": len(finance_corpus)},
                                "healthcare": {"test": 500, "corpus": len(covid_corpus)}}}
    for domain in ["finance","healthcare"]:
        report[domain] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in all_results[domain].items()}
    with open(os.path.join(OUTPUT_DIR, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report: {OUTPUT_DIR}/report.json")


if __name__ == "__main__":
    main()
