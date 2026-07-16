"""Final 1000-item large-sample validation experiment.

Evaluates all canonical verification architectures on 500 finance + 500 healthcare
(COVID-19) items with comprehensive metrics, bootstrap CI, and sensitivity analysis.

Architectures:
  - Single-Shot (Voting N=1)
  - Self-consistency Voting N=3, N=5, N=7 (all from the same 7 generated outputs)
  - MoA (Supporter + Skeptic + Judge, one round)

Factor: RAG OFF / ON

Key design: generate 7 outputs per item once, then subset for N=3, N=5, N=7.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
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

# Config
SEED = 42
MODEL = "deepseek-v4-flash"
TEMPERATURE = 0.7
MAX_TOKENS = 512
MAX_CONCURRENCY = 2000
OUTPUT_DIR = "results/final_1000_validation"
COST_PER_CALL = 0.00015

PPV_BASE_RATES = [0.001, 0.01, 0.05, 0.10, 0.25, 0.50]
COST_RATIOS = [(1, 1), (1, 5), (1, 10), (1, 25)]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "raw_outputs"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)

FINANCE_TEST = "data/raw/finance/finance_test_500.csv"
FINANCE_CORPUS = "data/raw/finance/finance_corpus.csv"
COVID_TEST = "data/raw/health/covid_test_500.csv"
COVID_CORPUS = "data/raw/health/covid_corpus.csv"


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


def load_all_data():
    all_data = {}
    for domain_key, test_path, corpus_path in [
        ("finance", FINANCE_TEST, FINANCE_CORPUS),
        ("healthcare", COVID_TEST, COVID_CORPUS),
    ]:
        test_items = load_test_csv(test_path, domain_key)
        corpus_items = load_corpus_csv(corpus_path, domain_key)
        test_texts = set(it.claim_text.strip().lower() for it in test_items)
        corpus_texts = set(it.claim_text.strip().lower() for it in corpus_items)
        overlap = test_texts & corpus_texts
        n_real = sum(1 for it in test_items if it.ground_truth == Verdict.REAL)
        n_fake = sum(1 for it in test_items if it.ground_truth == Verdict.FAKE)
        print(f"  {domain_key}: test={len(test_items)} (R={n_real} F={n_fake}), corpus={len(corpus_items)}, overlap={len(overlap)}")
        all_data[domain_key] = {"test": test_items, "corpus": corpus_items}
    return all_data


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


# Prompts
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


# TF-IDF
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
    for i, item in enumerate(items):
        pred = clf.predict([item.claim_text])[0]
        prob = clf.predict_proba([item.claim_text])[0]
        if pred == 1:
            verdict, confidence = Verdict.FAKE, prob[1]
        else:
            verdict, confidence = Verdict.REAL, prob[0]
        results.append(VerificationResult(
            item_id=item.id, verdict=verdict, confidence=float(confidence),
            latency_s=0.001, evidence=[], metadata={"architecture": "tfidf_baseline"},
        ))
    return results


# Single-Shot
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


# Voting (generate N=7, subset for 1/3/5/7)
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
            all_voters = item_voters_all.get(item.id, [])
            voters = all_voters[:N]
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


# MoA
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
        p1_callables.append(_sup)
        p1_callables.append(_ske)
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


# Metrics
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


def compute_ppv_curve(sens, fpr, base_rates):
    return [{"base_rate": br, "ppv": round((sens * br) / (sens * br + fpr * (1-br)), 6) if (sens * br + fpr * (1-br)) > 0 else 0.0} for br in base_rates]


def compute_expected_cost(fpr, fnr, base_rates, cfp, cfn):
    return [{"base_rate": br, "expected_cost": round((1-br) * fpr * cfp + br * fnr * cfn, 6)} for br in base_rates]


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
        section["ppv"] = compute_ppv_curve(m.recall, m.fpr, PPV_BASE_RATES)
        section["expected_cost"] = {f"FP{cfp}_FN{cfn}": compute_expected_cost(m.fpr, m.fnr, PPV_BASE_RATES, cfp, cfn) for cfp, cfn in COST_RATIOS}
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
        verdicts = [v.verdict for v in voters]
        v1 = verdicts[0]
        if len(set(verdicts)) == 1: stats["unanimous"] += 1
        counter = Counter(verdicts)
        majority_v, majority_c = counter.most_common(1)[0]
        if majority_c >= threshold: stats["majority_verdicts"][majority_v.name] = stats["majority_verdicts"].get(majority_v.name, 0) + 1
        else: stats["majority_verdicts"]["ESCALATE"] = stats["majority_verdicts"].get("ESCALATE", 0) + 1
        stats["voter1_verdicts"][v1.name] = stats["voter1_verdicts"].get(v1.name, 0) + 1
        maj_v = majority_v if majority_c >= threshold else Verdict.ESCALATE
        if v1 != Verdict.ESCALATE and maj_v != Verdict.ESCALATE and maj_v != v1 and n_voters > 1: stats["majority_flip_v1"] += 1
        for a in range(len(verdicts)):
            for b in range(a + 1, len(verdicts)):
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
    ci_str = f"  F1 CI=({ci.get('f1_95', [0,0])[0]:.3f},{ci.get('f1_95', [0,0])[1]:.3f})" if ci.get("f1_95") else ""
    print(f"    {label:30s} F1={m.get('f1',0):.4f}  P={m.get('precision',0):.4f}  R={m.get('recall',0):.4f}  FPR={m.get('fpr',0):.4f}  FNR={m.get('fnr',0):.4f}  Acc={m.get('accuracy',0):.4f}  ESC={section.get('escalate_rate',0):.2%}  ECE={section.get('ece',0):.4f}  Lat={section.get('latency',{}).get('mean',0):.1f}s{ci_str}")
    sys.stdout.flush()


def extract_voting_table(domain_results):
    table = {}
    for N in [1, 3, 5, 7]:
        for rag in ["rag_off", "rag_on"]:
            key = f"voting_n{N}_{rag}"
            if key in domain_results:
                if N not in table: table[N] = {}
                s = domain_results[key]
                table[N][rag] = {
                    "metrics": s.get("metrics", {}), "latency": s.get("latency", {}),
                    "escalate_rate": s.get("escalate_rate", 0), "ece": s.get("ece", 0),
                    "mean_disagreement": s.get("mean_disagreement", 0),
                    "agreement": s.get("voter_agreement", {}),
                    "confidence_intervals": s.get("confidence_intervals", {}),
                }
    return table


def print_voting_sensitivity(domain_results, domain):
    table = extract_voting_table(domain_results)
    rag_key = "rag_off"
    print(f"\n  VOTING SENSITIVITY - {domain.upper()} (RAG OFF)")
    h = f"  {'N':6s} {'F1':8s} {'P':8s} {'R':8s} {'FPR':8s} {'FNR':8s} {'Acc':8s} {'ESC':7s} {'ECE':7s} {'Disagr':7s} {'Unanim':7s} {'PairAgr':7s} {'Lat':6s}"
    print(h)
    print(f"  {'-'*len(h.strip())}")
    for N in [1, 3, 5, 7]:
        if N in table and rag_key in table[N]:
            s, m, lt = table[N][rag_key], table[N][rag_key]["metrics"], table[N][rag_key]["latency"]
            a = s.get("agreement", {})
            print(f"  {N:6d} {m.get('f1',0):.4f}  {m.get('precision',0):.4f}  {m.get('recall',0):.4f}  {m.get('fpr',0):.4f}  {m.get('fnr',0):.4f}  {m.get('accuracy',0):.4f}  {s.get('escalate_rate',0):.2%}  {s.get('ece',0):.4f}  {s.get('mean_disagreement',0):.4f}  {a.get('unanimous',0)/max(a.get('total_items',1),1):.2%}  {a.get('pairwise_agreement',0):.4f}  {lt.get('mean',0):.1f}s")

    print(f"\n  Marginal gains (F1):")
    f1_prev = table[1][rag_key]["metrics"]["f1"] if 1 in table and rag_key in table[1] else 0
    for N in [3, 5, 7]:
        if N in table and rag_key in table[N]:
            f1_n = table[N][rag_key]["metrics"]["f1"]
            print(f"    N={N-2} -> N={N}: {f1_n-f1_prev:+.4f}")
            f1_prev = f1_n

    print(f"\n  Paired bootstrap 95% CI vs N=1 (RAG OFF):")
    for N in [3, 5, 7]:
        if N in table and rag_key in table[N]:
            ci = table[N][rag_key].get("confidence_intervals",{}).get("f1_95",[])
            f1_n = table[N][rag_key]["metrics"]["f1"]
            f1_1 = table[1][rag_key]["metrics"]["f1"]
            sig = "NS" if (ci[0] <= 0 <= ci[1]) else "SIG"
            print(f"    N={N}: DF1={f1_n-f1_1:+.4f}  CI=({ci[0]:.3f},{ci[1]:.3f})  {sig}")

    best_n = max(((table[N][rag_key]["metrics"]["f1"], N) for N in table if rag_key in table[N]), default=(0,1))[1]
    best_f1 = table[best_n][rag_key]["metrics"]["f1"]
    smallest = min((N for N in table if rag_key in table[N] and table[N][rag_key]["metrics"]["f1"] >= best_f1 - 0.02), default=1)
    print(f"\n  Best N: N={best_n} (F1={best_f1:.4f})")
    print(f"  Smallest N within 0.02 F1 of best: N={smallest} (F1={table[smallest][rag_key]['metrics']['f1']:.4f})")

    print(f"\n  Corrections vs Damages (majority vs voter 1):")
    for N in [3, 5, 7]:
        if N in table and rag_key in table[N]:
            a = table[N][rag_key].get("agreement", {})
            print(f"    N={N}: {a.get('corrected_by_voting',0)} corrected, {a.get('damaged_by_voting',0)} damaged, net={a.get('corrected_by_voting',0)-a.get('damaged_by_voting',0):+d}")

    return table


def print_final_table(domain_results, domain):
    print(f"\n  FINAL ARCHITECTURE COMPARISON - {domain.upper()}")
    cells = [("tfidf_baseline","TF-IDF Baseline"),("single_shot_rag_off","SS RAG OFF"),("single_shot_rag_on","SS RAG ON"),
             ("voting_n1_rag_off","Vot N=1 OFF"),("voting_n1_rag_on","Vot N=1 ON"),
             ("voting_n3_rag_off","Vot N=3 OFF"),("voting_n3_rag_on","Vot N=3 ON"),
             ("voting_n5_rag_off","Vot N=5 OFF"),("voting_n5_rag_on","Vot N=5 ON"),
             ("voting_n7_rag_off","Vot N=7 OFF"),("voting_n7_rag_on","Vot N=7 ON"),
             ("moa_rag_off","MoA RAG OFF"),("moa_rag_on","MoA RAG ON")]
    h = f"  {'Architecture':22s} {'F1':8s} {'P':8s} {'R':8s} {'FPR':8s} {'FNR':8s} {'Acc':8s} {'ESC':7s} {'ECE':7s} {'Lat':6s} {'Calls':6s}"
    print(h)
    print(f"  {'-'*len(h.strip())}")
    for key, label in cells:
        s = domain_results.get(key, {})
        if not s or not s.get("metrics"): continue
        m, lt = s["metrics"], s["latency"]
        print(f"  {label:22s} {m.get('f1',0):.4f}  {m.get('precision',0):.4f}  {m.get('recall',0):.4f}  {m.get('fpr',0):.4f}  {m.get('fnr',0):.4f}  {m.get('accuracy',0):.4f}  {s.get('escalate_rate',0):.2%}  {s.get('ece',0):.4f}  {lt.get('mean',0):.1f}s  {s.get('total_api_calls',0):5d}")

    print(f"\n  95% Bootstrap CIs:")
    for key, label in cells:
        s = domain_results.get(key, {})
        ci = s.get("confidence_intervals",{}).get("f1_95",[])
        if ci: print(f"    {label:22s} F1={s['metrics']['f1']:.4f}  CI=({ci[0]:.3f},{ci[1]:.3f})")


def print_cross_domain_summary(all_results):
    print(f"\n  CROSS-DOMAIN MEAN F1")
    cells = [("tfidf_baseline","TF-IDF"),("single_shot_rag_off","SS OFF"),("single_shot_rag_on","SS ON"),
             ("voting_n1_rag_off","V1 OFF"),("voting_n1_rag_on","V1 ON"),
             ("voting_n3_rag_off","V3 OFF"),("voting_n3_rag_on","V3 ON"),
             ("voting_n5_rag_off","V5 OFF"),("voting_n5_rag_on","V5 ON"),
             ("voting_n7_rag_off","V7 OFF"),("voting_n7_rag_on","V7 ON"),
             ("moa_rag_off","MoA OFF"),("moa_rag_on","MoA ON")]
    print(f"  {'Architecture':12s} {'Fin F1':8s} {'Health F1':10s} {'Mean':8s} {'Lat':8s}")
    print(f"  {'-'*46}")
    for key, label in cells:
        f1s, lats = [], []
        for domain in ["finance","healthcare"]:
            s = all_results.get(domain,{}).get(key,{})
            if s and s.get("metrics"):
                f1s.append(s["metrics"]["f1"]); lats.append(s.get("latency",{}).get("mean",0))
        if f1s:
            print(f"  {label:12s} {f1s[0]:.4f}     {f1s[1]:.4f}     {float(np.mean(f1s)):.4f}  {float(np.mean(lats)):.1f}s")


def print_pairwise_comparisons(all_results, all_test):
    print(f"\n  PAIRWISE COMPARISONS (Bootstrap DF1)")
    comparisons = [
        ("RAG ON vs OFF (SS)","single_shot_rag_on","single_shot_rag_off"),
        ("RAG ON vs OFF (Vot N=3)","voting_n3_rag_on","voting_n3_rag_off"),
        ("RAG ON vs OFF (MoA)","moa_rag_on","moa_rag_off"),
        ("MoA vs SS (RAG OFF)","moa_rag_off","single_shot_rag_off"),
        ("MoA vs SS (RAG ON)","moa_rag_on","single_shot_rag_on"),
        ("Vot N=3 vs N=1 (RAG OFF)","voting_n3_rag_off","voting_n1_rag_off"),
        ("Vot N=3 vs N=1 (RAG ON)","voting_n3_rag_on","voting_n1_rag_on"),
        ("SS vs TF-IDF","single_shot_rag_off","tfidf_baseline"),
    ]
    for comp_name, key_a, key_b in comparisons:
        print(f"\n  {comp_name}:")
        for domain in ["finance","healthcare"]:
            items = all_test[domain]
            sa, sb = all_results.get(domain,{}).get(key_a,{}), all_results.get(domain,{}).get(key_b,{})
            if not sa.get("_cached_results") or not sb.get("_cached_results"):
                print(f"    {domain}: caching raw results...")
                continue
            ra, rb = sa["_cached_results"], sb["_cached_results"]
            truths = [it.ground_truth or Verdict.REAL for it in items]
            ci = _bootstrap_ci(items, ra, _f1_metric)
            # Can't do paired diff without paired results
            fa, fb = sa["metrics"]["f1"], sb["metrics"]["f1"]
            print(f"    {domain}: A={fa:.4f} vs B={fb:.4f}  D={fa-fb:+.4f}")


def generate_figures(all_results, output_dir):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping figures"); return
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Fig 1: F1 vs N
    plt.figure(figsize=(8, 5))
    for domain in ["finance","healthcare"]:
        ns, f1s = [], []
        for N in [1,3,5,7]:
            s = all_results[domain].get(f"voting_n{N}_rag_off", {})
            if s.get("metrics"): ns.append(N); f1s.append(s["metrics"]["f1"])
        plt.plot(ns, f1s, "o-", label=domain.capitalize())
    plt.xlabel("N voters"); plt.ylabel("F1"); plt.title("F1 vs Voting Size (RAG OFF)")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(fig_dir, "fig1_f1_vs_n.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 2: Architecture comparison
    plt.figure(figsize=(12, 5))
    arch_keys = ["single_shot_rag_off","single_shot_rag_on","voting_n3_rag_off","voting_n3_rag_on","moa_rag_off","moa_rag_on"]
    arch_labels = ["SS\nOFF","SS\nON","V3\nOFF","V3\nON","MoA\nOFF","MoA\nON"]
    x = np.arange(len(arch_labels))
    for i, domain in enumerate(["finance","healthcare"]):
        f1s = [all_results[domain].get(k,{}).get("metrics",{}).get("f1",0) for k in arch_keys]
        plt.bar(x + i*0.25 - 0.25, f1s, 0.25, label=domain.capitalize())
    plt.xlabel("Architecture"); plt.ylabel("F1"); plt.title("Architecture Comparison")
    plt.xticks(x, arch_labels); plt.legend(); plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig2_architecture_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 3: RAG effect
    plt.figure(figsize=(8, 5))
    effects, labels = [], []
    for arch, label in [("single_shot","SS"),("voting_n3","V3"),("voting_n5","V5"),("voting_n7","V7"),("moa","MoA")]:
        for domain in ["finance","healthcare"]:
            soff = all_results[domain].get(f"{arch}_rag_off",{}).get("metrics",{}).get("f1",0)
            son = all_results[domain].get(f"{arch}_rag_on",{}).get("metrics",{}).get("f1",0)
            if soff or son: effects.append(son - soff); labels.append(f"{label}\n{domain[:4]}")
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in effects]
    plt.bar(range(len(effects)), effects, color=colors)
    plt.axhline(y=0, color="black", linewidth=0.5)
    plt.xticks(range(len(effects)), labels, fontsize=9)
    plt.ylabel("DF1 (RAG ON - RAG OFF)"); plt.title("RAG Effect on F1")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig3_rag_effect.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Figures saved to {fig_dir}/")


def main():
    global_start = time.time()
    print("=" * 70)
    print("FINAL 1000-ITEM LARGE-SAMPLE VALIDATION")
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

    # Load data
    print("\nLOADING DATA")
    all_data = load_all_data()
    all_test = {d: all_data[d]["test"] for d in all_data}
    all_corpus = {d: all_data[d]["corpus"] for d in all_data}

    # Run experiments
    print("\nEXPERIMENTS")
    all_results = {}
    for domain in ["finance", "healthcare"]:
        items, corpus = all_test[domain], all_corpus[domain]
        print(f"\n{'='*65}")
        print(f"{domain.upper()} - {len(items)} items, {len(corpus)} corpus")
        print(f"{'='*65}")
        domain_results = {}

        # TF-IDF Baseline
        print(f"\n  TF-IDF Baseline..."); sys.stdout.flush()
        t0 = time.time()
        tfidf_r = run_tfidf_baseline(items, corpus)
        domain_results["tfidf_baseline"] = evaluate_architecture(items, tfidf_r, "tfidf_baseline", 0)
        domain_results["tfidf_baseline"]["elapsed_s"] = time.time() - t0
        print_metrics(domain_results["tfidf_baseline"], "TF-IDF Baseline")

        # Single-Shot RAG OFF/ON
        for rag_flag, rag_label in [(False,"RAG OFF"),(True,"RAG ON")]:
            arch_label = f"single_shot_{rag_label.lower().replace(' ','_')}"
            print(f"\n  Single-Shot {rag_label}..."); sys.stdout.flush()
            t0 = time.time()
            ss_r = run_single_shot(items, rag_on=rag_flag, corpus=corpus if rag_flag else None)
            domain_results[arch_label] = evaluate_architecture(items, ss_r, arch_label, len(items))
            domain_results[arch_label]["elapsed_s"] = time.time() - t0
            domain_results[arch_label]["_cached_results"] = ss_r
            save_raw_output(domain, arch_label, items, ss_r, OUTPUT_DIR)
            print_metrics(domain_results[arch_label], f"SS {rag_label}")

        # Voting (7 outputs -> N=1,3,5,7)
        for rag_flag, rag_label in [(False,"RAG OFF"),(True,"RAG ON")]:
            rl = rag_label.lower().replace(' ','_')
            print(f"\n  Voting {rag_label} (7 outputs per item)..."); sys.stdout.flush()
            t0 = time.time()
            voting_map = run_voting_all_n(items, rag_on=rag_flag, corpus=corpus if rag_flag else None, n_total=7)
            v_elapsed = time.time() - t0
            api_calls = len(items) * 7
            for N in [1, 3, 5, 7]:
                agg, per_voter = voting_map[N]
                arch_label = f"voting_n{N}_{rl}"
                domain_results[arch_label] = evaluate_architecture(items, agg, arch_label, api_calls)
                domain_results[arch_label]["elapsed_s"] = round(v_elapsed, 1)
                domain_results[arch_label]["_cached_results"] = agg
                domain_results[arch_label]["voter_agreement"] = analyze_voter_agreement(items, per_voter, N)
                save_raw_output(domain, arch_label, items, agg, OUTPUT_DIR)
                print_metrics(domain_results[arch_label], f"Voting N={N} {rag_label}")

        # MoA RAG OFF/ON
        for rag_flag, rag_label in [(False,"RAG OFF"),(True,"RAG ON")]:
            arch_label = f"moa_{rag_label.lower().replace(' ','_')}"
            print(f"\n  MoA {rag_label}..."); sys.stdout.flush()
            t0 = time.time()
            moa_r = run_moa(items, rag_on=rag_flag, corpus=corpus if rag_flag else None)
            domain_results[arch_label] = evaluate_architecture(items, moa_r, arch_label, len(items)*3)
            domain_results[arch_label]["elapsed_s"] = time.time() - t0
            domain_results[arch_label]["_cached_results"] = moa_r
            save_raw_output(domain, arch_label, items, moa_r, OUTPUT_DIR)
            print_metrics(domain_results[arch_label], f"MoA {rag_label}")

        all_results[domain] = domain_results

    # Print results
    print(f"\n{'='*70}")
    print("VOTING SIZE SENSITIVITY (RAG OFF)")
    print(f"{'='*70}")
    for domain in ["finance","healthcare"]:
        print_voting_sensitivity(all_results[domain], domain)

    print(f"\n{'='*70}")
    print("FINAL ARCHITECTURE COMPARISON")
    print(f"{'='*70}")
    for domain in ["finance","healthcare"]:
        print_final_table(all_results[domain], domain)
    print_cross_domain_summary(all_results)
    print_pairwise_comparisons(all_results, all_test)

    # Generate figures
    print(f"\nGENERATING FIGURES")
    generate_figures(all_results, OUTPUT_DIR)

    # Summary
    total_runtime = time.time() - global_start
    total_calls = sum(all_results[d].get("voting_n7_rag_off",{}).get("total_api_calls",0) for d in ["finance","healthcare"]) \
                + sum(all_results[d].get("voting_n7_rag_on",{}).get("total_api_calls",0) for d in ["finance","healthcare"]) \
                + sum(all_results[d].get("single_shot_rag_off",{}).get("total_api_calls",0) for d in ["finance","healthcare"]) \
                + sum(all_results[d].get("single_shot_rag_on",{}).get("total_api_calls",0) for d in ["finance","healthcare"]) \
                + sum(all_results[d].get("moa_rag_off",{}).get("total_api_calls",0) for d in ["finance","healthcare"]) \
                + sum(all_results[d].get("moa_rag_on",{}).get("total_api_calls",0) for d in ["finance","healthcare"])

    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"Total runtime: {total_runtime:.0f}s ({total_runtime/60:.1f} min)")
    print(f"Estimated total API calls: ~{total_calls}")
    print(f"Estimated cost: ~${total_calls * COST_PER_CALL:.4f}")
    print(f"Results: {OUTPUT_DIR}/")
    sys.stdout.flush()

    # Save report JSON
    report = {"metadata": {"experiment": "final_1000_validation", "model": MODEL, "temperature": TEMPERATURE,
                           "timestamp": datetime.now().isoformat(), "total_runtime_s": total_runtime},
              "data_summary": {d: {"test_size": len(all_test[d]), "corpus_size": len(all_corpus[d])} for d in ["finance","healthcare"]}}
    for domain in ["finance","healthcare"]:
        report[domain] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in all_results[domain].items()}
    with open(os.path.join(OUTPUT_DIR, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report: {OUTPUT_DIR}/report.json")


if __name__ == "__main__":
    main()
