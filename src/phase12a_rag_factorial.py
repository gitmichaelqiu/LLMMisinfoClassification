"""Phase 12a: Complete the 2x3 factorial matrix — Voting+RAG and MoA+RAG.

Evaluates Voting N=3 and MoA with evidence-augmented prompts (same TF-IDF
retrieval as the SS+RAG prototype). Tests whether stronger verifiers benefit
from generic evidence retrieval.

~180 DeepSeek API calls, est. ~$0.027.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
from dotenv import load_dotenv

for _var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
    os.environ.pop(_var, None)
os.environ["no_proxy"] = "*"
os.environ["NO_PROXY"] = "*"

from src.datasets import load_dataset
from src.metrics import classification_metrics, compute_confusion_matrix, compute_ece, compute_ppv
from src.prompts import SINGLE_SHOT_SYSTEM, SINGLE_SHOT_USER, format_user_prompt
from src.schemas import Verdict, VerificationItem, VerificationResult

load_dotenv()

SAMPLE_SIZE = 10
SEED = 42
MODEL = "deepseek-v4-flash"
TEMPERATURE = 0.0
MAX_TOKENS = 512
DOMAINS = ["finance", "healthcare", "political"]
ORIGINAL_RAW_DIR = os.path.join("output", "real_eval", "raw_outputs")
OUTPUT_DIR = "output/phase12a_rag_factorial"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── LLM helpers (from repair_rerun.py) ────────────────────────────

def _make_openai_client():
    from openai import OpenAI
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key or api_key.startswith("placeholder"):
        print("  WARNING: No valid DEEPSEEK_API_KEY — using mock responses")
        return None
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")


_client = _make_openai_client()


def _llm_call(system_prompt: str, user_prompt: str, model: str = MODEL,
              temperature: float = TEMPERATURE, max_tokens: int = MAX_TOKENS) -> Tuple[str, float]:
    start = time.time()
    if _client is None:
        time.sleep(0.1)
        return ("Verdict: REAL\nConfidence: 50\nFlags: none\nReasoning: Mock mode.", time.time() - start)
    try:
        resp = _client.chat.completions.create(
            model=model, temperature=temperature, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}],
        )
        raw = resp.choices[0].message.content or ""
    except Exception as e:
        raw = f"LLM error: {e}"
    return (raw, time.time() - start)


def _parse_response(raw: str, item_id: str, latency_s: float,
                    extra_metadata: Optional[dict] = None) -> VerificationResult:
    verdict_map = {"REAL": Verdict.REAL, "FAKE": Verdict.FAKE, "ESCALATE": Verdict.ESCALATE}
    text = raw.strip()
    verdict = Verdict.REAL
    confidence = 0.5
    flags = []
    reasoning = ""
    for line in text.split("\n"):
        ul = line.upper()
        if ul.startswith("VERDICT:"):
            v = ul.replace("VERDICT:", "").strip().split()[0] if ul.replace("VERDICT:", "").strip() else "REAL"
            verdict = verdict_map.get(v, Verdict.REAL)
        elif ul.startswith("CONFIDENCE:"):
            try:
                confidence = float(ul.replace("CONFIDENCE:", "").strip().rstrip("%")) / 100.0
            except ValueError:
                confidence = 0.5
        elif ul.startswith("FLAGS:"):
            flags = [f.strip() for f in ul.replace("FLAGS:", "").strip().split(",") if f.strip()]
        elif ul.startswith("REASONING:"):
            reasoning = line[len("REASONING:"):].strip()
    return VerificationResult(
        item_id=item_id, verdict=verdict, confidence=max(0.0, min(1.0, confidence)),
        latency_s=latency_s, evidence=[],
        metadata={"architecture": extra_metadata.get("architecture", "unknown"),
                  **(extra_metadata or {})},
    )


# ── Data loading (from repair_rerun.py) ───────────────────────────

def _build_rag_corpus_from_raw(domain: str) -> Tuple[List[dict], List[dict]]:
    """Build test items and RAG corpus from saved raw outputs.

    Returns (test_items, corpus_items) where each item is a dict
    with 'id', 'claim', 'ground_truth' keys.
    """
    raw_path = os.path.join(ORIGINAL_RAW_DIR, f"{domain}_single_shot.json")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw output not found: {raw_path}")
    with open(raw_path) as f:
        data = json.load(f)

    items_raw = data.get("items", [])
    test_items = []
    for it in items_raw:
        claim = it.get("claim", "")
        gt_raw = it.get("ground_truth")
        if gt_raw and gt_raw != "None":
            gt = gt_raw
        else:
            gt = None
        test_items.append({"id": it.get("id", ""), "claim": claim, "ground_truth": gt})

    # Build corpus from all other available raw files in the domain
    corpus_items = []
    seen_ids = {it["id"] for it in test_items}
    for fname in os.listdir(ORIGINAL_RAW_DIR):
        if not fname.startswith(domain):
            continue
        if fname.endswith("_single_shot.json") and domain in fname:
            continue  # already loaded test items from this file
        fpath = os.path.join(ORIGINAL_RAW_DIR, fname)
        try:
            with open(fpath) as f:
                d = json.load(f)
            for cit in d.get("items", []):
                if cit.get("id") not in seen_ids:
                    corpus_items.append({"id": cit.get("id", ""), "claim": cit.get("claim", ""),
                                          "ground_truth": cit.get("ground_truth")})
                    seen_ids.add(cit.get("id"))
        except:
            pass

    # If corpus is too small, include the test items themselves (for retrieval diversity)
    if len(corpus_items) < 20:
        # Add all items from all domains as general corpus
        for fname in os.listdir(ORIGINAL_RAW_DIR):
            if not fname.endswith("_single_shot.json"):
                continue
            fpath = os.path.join(ORIGINAL_RAW_DIR, fname)
            try:
                with open(fpath) as f:
                    d = json.load(f)
                for cit in d.get("items", []):
                    if cit.get("id") not in seen_ids:
                        corpus_items.append({"id": cit.get("id", ""), "claim": cit.get("claim", ""),
                                              "ground_truth": cit.get("ground_truth")})
                        seen_ids.add(cit.get("id"))
            except:
                pass

    return test_items, corpus_items


# ── TF-IDF Retriever ──────────────────────────────────────────────

def _retrieve_evidence(claim_text: str, corpus: List[dict],
                       vec, tfidf_matrix, top_k: int = 3) -> str:
    from sklearn.metrics.pairwise import cosine_similarity
    query_vec = vec.transform([claim_text])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = np.argsort(sims)[::-1][:top_k]
    context_parts = []
    for idx in top_indices:
        if sims[idx] <= 0:
            continue
        doc = corpus[idx]
        doc_text = doc["claim"]
        context_parts.append(
            f"[Document {doc['id']}] (relevance={sims[idx]:.3f})\n{doc_text}\n"
        )
    return "".join(context_parts)[:2000]


# ── Voting+RAG ────────────────────────────────────────────────────

VOTER_NEUTRAL_RAG = """You are a Neutral Evidence Verifier. Examine the following claim for evidence patterns without bias.

You have access to RETRIEVED EVIDENCE from a knowledge corpus. Use it alongside your own analysis.

Analyze systematically:
1. Retrieved evidence: Does it support or contradict the claim?
2. Internal consistency: Are the claims logically self-consistent?
3. Plausibility: Is the magnitude within reasonable bounds?

Weigh supporting and contradicting evidence equally.

Output in EXACTLY this format:
Verdict: REAL or FAKE or ESCALATE
Confidence: <0-100>
Flags: [contradiction|implausibility|inconsistency|none]
Reasoning: <one-sentence rationale>"""

VOTER_SKEPTIC_RAG = """You are a Skeptical Misinformation Detector. Your job is to scrutinize the claim for specific red flags.

You have access to RETRIEVED EVIDENCE from a knowledge corpus. Scrutinize whether the evidence contradicts the claim.

Look for:
1. Retrieved evidence: Does any evidence directly contradict the claim?
2. Logical contradictions or impossibilities
3. Exaggerated or implausible magnitudes/numbers
4. Hallmarks of misinformation: emotional manipulation, lack of specifics

Only flag as FAKE when you find CONCRETE, SPECIFIC evidence.
If the claim seems plausible with no clear red flags, report REAL.

Output in EXACTLY this format:
Verdict: REAL or FAKE or ESCALATE
Confidence: <0-100>
Flags: [contradiction|implausibility|inconsistency|none]
Reasoning: <one-sentence rationale>"""

VOTER_CALIBRATED_RAG = """You are a Base-Rate-Calibrated Verifier. You understand that most news is genuine.

You have access to RETRIEVED EVIDENCE from a knowledge corpus. Compare the claim against this evidence.

Remember that genuine news is the majority. Before flagging FAKE, ensure evidence against it is clear.
Consider:
1. Does the retrieved evidence support or contradict the claim?
2. Is there clear evidence this is false, or is it merely surprising?
3. Would a reasonable person with domain knowledge accept this claim?

Output in EXACTLY this format:
Verdict: REAL or FAKE or ESCALATE
Confidence: <0-100>
Flags: [contradiction|implausibility|inconsistency|none]
Reasoning: <one-sentence rationale>"""

VOTER_RAG_PROMPTS = [
    ("neutral", VOTER_NEUTRAL_RAG),
    ("skeptical", VOTER_SKEPTIC_RAG),
    ("calibrated", VOTER_CALIBRATED_RAG),
]


def _run_voting_rag(items: List[dict],
                    corpus: List[dict]) -> List[VerificationResult]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    corpus_texts = [it["claim"] for it in corpus]
    vec = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = vec.fit_transform(corpus_texts)

    results = []
    for item in items:
        start = time.time()
        claim = item["claim"]
        evidence = _retrieve_evidence(claim, corpus, vec, tfidf_matrix)
        retrieval_lat = time.time() - start
        evidence_prefix = "\n\n── Retrieved Evidence ──\n" + evidence + "\n────────────────────\n" if evidence else ""

        voter_outputs = []
        for role_name, system_prompt in VOTER_RAG_PROMPTS:
            user_prompt = f"Claim to verify:\n{claim}{evidence_prefix}\nIs this claim REAL or FAKE?"
            try:
                raw, lat = _llm_call(system_prompt, user_prompt, temperature=0.3)
                parsed = _parse_response(raw, item["id"], lat,
                                         extra_metadata={"voter_role": role_name, "architecture": "voting_n3_rag"})
            except Exception as e:
                parsed = VerificationResult(
                    item_id=item["id"], verdict=Verdict.REAL, confidence=0.5,
                    latency_s=0, evidence=[f"Error: {e}"],
                    metadata={"voter_role": role_name, "architecture": "voting_n3_rag"},
                )
            voter_outputs.append(parsed)

        total_latency = time.time() - start

        # Aggregation: votes for FAKE vs REAL, ESCALATE as abstain
        votes_fake = sum(1 for v in voter_outputs if v.verdict == Verdict.FAKE)
        votes_real = sum(1 for v in voter_outputs if v.verdict == Verdict.REAL)
        votes_esc = sum(1 for v in voter_outputs if v.verdict == Verdict.ESCALATE)
        avg_conf = np.mean([v.confidence for v in voter_outputs])

        if votes_fake >= 2:
            verdict = Verdict.FAKE
        elif votes_real >= 2:
            verdict = Verdict.REAL
        else:
            verdict = Verdict.ESCALATE

        evidence_log = [
            f"voting_rag: FAKE={votes_fake} REAL={votes_real} ESCALATE={votes_esc} avg_conf={avg_conf:.2f}",
            *[f"  [{v.metadata.get('voter_role','?')}] {v.verdict.name} (conf={v.confidence:.2f})" for v in voter_outputs],
        ]
        results.append(VerificationResult(
            item_id=item["id"], verdict=verdict, confidence=avg_conf,
            latency_s=total_latency, evidence=evidence_log,
            metadata={
                "architecture": "voting_n3_rag",
                "votes": {"FAKE": votes_fake, "REAL": votes_real, "ESCALATE": votes_esc},
                "retrieval_hits": len(evidence),
            },
        ))
    return results


# ── MoA+RAG ───────────────────────────────────────────────────────

MOA_BELIEVER_RAG = """You are Agent 1 (The Believer) in an information verification debate.
Your role is to find evidence SUPPORTING the authenticity of the claim. Argue the claim is REAL.

You have access to RETRIEVED EVIDENCE from a knowledge corpus. Use it to build your case.

Build the strongest possible case for authenticity:
1. Retrieved evidence: Does any evidence support the claim?
2. Source credibility: Could this originate from a legitimate source?
3. Factual alignment: Does the claim match known facts?
4. Plausibility: Is the magnitude reasonable?

Output:
Verdict: REAL or UNCERTAIN
Evidence: <bulleted list of supporting points>
Confidence: <0-100>"""

MOA_SKEPTIC_RAG = """You are Agent 2 (The Skeptic) in an information verification debate.
Your role is to find evidence that the claim is FALSE or misleading. Argue the claim is FAKE.

You have access to RETRIEVED EVIDENCE from a knowledge corpus. Scrutinize it for contradictions.

Build the strongest possible case against authenticity:
1. Retrieved evidence: Does any evidence contradict the claim?
2. Implausibility: Are specific numbers or claims unrealistic?
3. Inconsistencies: Are there logical or temporal inconsistencies?
4. Red flags: Hallmarks of misinformation?

Output:
Verdict: FAKE or UNCERTAIN
Evidence: <bulleted list of problematic points>
Confidence: <0-100>"""

MOA_RISK_OFFICER_RAG = """You are Agent 3 (The Risk Officer) in an information verification debate.

You have received analyses from two agents:
- The Believer argues the claim is REAL (with evidence, including retrieved corpus evidence).
- The Skeptic argues the claim is FAKE (with evidence, including retrieved corpus evidence).

Your job is to evaluate BOTH arguments critically, considering the retrieved evidence, and deliver a final verdict.

IMPORTANT:
1. Weigh evidence QUALITY, not just presence.
2. Consider whether the retrieved evidence supports or contradicts the claim.
3. Default to REAL when Believer provides stronger evidence.
4. Default to FAKE when Skeptic provides clear contradictions.
5. Default to ESCALATE when evidence is balanced or weak.

Output your final verdict in EXACTLY this format:
Verdict: REAL or FAKE or ESCALATE
Confidence: <0-100>
Flags: [contradiction|implausibility|inconsistency|none]
Reasoning: <one-sentence rationale>"""


def _run_moa_rag(items: List[dict],
                 corpus: List[dict]) -> List[VerificationResult]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    corpus_texts = [it["claim"] for it in corpus]
    vec = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = vec.fit_transform(corpus_texts)

    results = []
    for item in items:
        start = time.time()
        claim = item["claim"]
        item_id = item["id"]
        evidence = _retrieve_evidence(claim, corpus, vec, tfidf_matrix)
        retrieval_lat = time.time() - start
        evidence_prefix = "\n\n── Retrieved Evidence ──\n" + evidence + "\n────────────────────\n" if evidence else ""
        user_prompt = "Claim to verify:\n" + claim + evidence_prefix

        # Believer
        try:
            bel_raw, bel_lat = _llm_call(MOA_BELIEVER_RAG, user_prompt)
        except Exception as e:
            bel_raw = f"Error: {e}"

        # Skeptic
        try:
            skt_raw, skt_lat = _llm_call(MOA_SKEPTIC_RAG, user_prompt)
        except Exception as e:
            skt_raw = f"Error: {e}"

        # Risk Officer
        ro_context = (
            "Claim to verify:\n" + claim + "\n\n"
            "── Retrieved Evidence ──\n" + evidence + "\n────────────────────\n"
            "── Believer's Analysis ──\n" + bel_raw + "\n\n"
            "── Skeptic's Analysis ──\n" + skt_raw + "\n"
        )
        try:
            ro_raw, ro_lat = _llm_call(MOA_RISK_OFFICER_RAG, ro_context)
        except Exception as e:
            ro_raw = f"Error: {e}"

        total_latency = time.time() - start
        r = _parse_response(ro_raw, item["id"], total_latency, extra_metadata={
            "architecture": "moa_rag",
            "believer_raw": bel_raw[:200],
            "skeptic_raw": skt_raw[:200],
        })
        r.evidence = [
            f"moa_rag: Believer then Skeptic then Risk Officer",
            f"  Believer: {bel_raw[:100]}",
            f"  Skeptic: {skt_raw[:100]}",
            f"  RO verdict: {r.verdict.name} (conf={r.confidence:.2f})",
        ]
        results.append(r)
    return results


# ── Evaluation ────────────────────────────────────────────────────

def evaluate(domain: str, arch_name: str, runner_fn: Callable) -> dict:
    print(f"\n  [{domain}] Running {arch_name}...")
    items, corpus = _build_rag_corpus_from_raw(domain)
    if not corpus:
        print(f"    WARNING: Empty corpus for {domain} — results may be degraded")
    print(f"    Items: {len(items)}, Corpus: {len(corpus)} items")

    results = runner_fn(items, corpus)

    # Build confusion matrix against ground truth
    VERDICT_MAP = {"FAKE": Verdict.FAKE, "REAL": Verdict.REAL}
    tp = fp = tn = fn = 0
    for r in results:
        gt_raw = next((it["ground_truth"] for it in items if it["id"] == r.item_id), None)
        if gt_raw is None:
            continue
        gt = VERDICT_MAP.get(gt_raw)
        if gt is None:
            continue
        if r.verdict == Verdict.FAKE and gt == Verdict.FAKE:
            tp += 1
        elif r.verdict == Verdict.FAKE and gt == Verdict.REAL:
            fp += 1
        elif r.verdict == Verdict.REAL and gt == Verdict.REAL:
            tn += 1
        elif r.verdict == Verdict.REAL and gt == Verdict.FAKE:
            fn += 1
        elif r.verdict == Verdict.ESCALATE and gt == Verdict.REAL:
            tn += 1
        elif r.verdict == Verdict.ESCALATE and gt == Verdict.FAKE:
            fn += 1

    n_gt = sum(1 for it in items if it["ground_truth"] is not None)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    acc = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0

    latencies = [r.latency_s for r in results]
    confidences = [r.confidence for r in results]
    verdicts = Counter(r.verdict.name for r in results)

    ece_val = compute_ece(np.array(confidences), np.array([
        1 if r.verdict == Verdict.FAKE else 0 for r in results
    ]))
    # compute_ece returns Tuple[ece, bin_confidences, bin_accuracies, bin_counts]
    ece = ece_val[0] if isinstance(ece_val, (tuple, list)) else ece_val

    metrics = {
        "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
        "fpr": round(fpr, 4), "fnr": round(fnr, 4), "accuracy": round(acc, 4),
        "n_total": len(results), "n_with_gt": n_gt,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "latency": {"mean": round(float(np.mean(latencies)), 2), "median": round(float(np.median(latencies)), 2),
                     "min": round(float(min(latencies)), 2), "max": round(float(max(latencies)), 2)},
        "ece": round(ece, 4),
        "verdict_distribution": dict(verdicts),
    }
    print(f"    F1={metrics['f1']:.4f}  P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  "
          f"FPR={metrics['fpr']:.4f}  Lat={metrics['latency']['mean']:.1f}s  "
          f"ECE={metrics['ece']:.4f}")
    return metrics


def main():
    print("=" * 60)
    print("Phase 12a: 2x3 Factorial — Voting+RAG and MoA+RAG")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Domains: {DOMAINS}")
    print(f"Items/domain: {SAMPLE_SIZE}")

    if _client is None:
        print("  ⚠️  No API key — using mock responses (results will be non-informative)")
    else:
        print("  ✅ API key found — making real DeepSeek calls")

    report = {
        "metadata": {
            "phase": "12a",
            "model": MODEL,
            "timestamp": datetime.utcnow().isoformat(),
            "sample_size_per_domain": SAMPLE_SIZE,
            "total_api_calls_estimated": len(DOMAINS) * SAMPLE_SIZE * 6,  # 3 voters + 3 moa agents
        },
        "domains": {},
    }

    for domain in DOMAINS:
        report["domains"][domain] = {}

        try:
            metrics = evaluate(domain, "voting_n3_rag", _run_voting_rag)
            report["domains"][domain]["voting_n3_rag"] = {"metrics": metrics, "skipped": False}
        except Exception as e:
            print(f"    ERROR: {e}")
            report["domains"][domain]["voting_n3_rag"] = {"skipped": True, "error": str(e)}

        try:
            metrics = evaluate(domain, "moa_rag", _run_moa_rag)
            report["domains"][domain]["moa_rag"] = {"metrics": metrics, "skipped": False}
        except Exception as e:
            print(f"    ERROR: {e}")
            report["domains"][domain]["moa_rag"] = {"skipped": True, "error": str(e)}

    # Save report
    report_path = os.path.join(OUTPUT_DIR, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to {report_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("PHASE 12a: RESULTS SUMMARY")
    print("=" * 60)
    for domain in DOMAINS:
        print(f"\n  {domain.upper()}:")
        for arch in ["voting_n3_rag", "moa_rag"]:
            d = report["domains"][domain].get(arch, {})
            if d.get("skipped"):
                print(f"    {arch}: SKIPPED ({d.get('error', '')})")
            else:
                m = d.get("metrics", {})
                print(f"    {arch}: F1={m.get('f1',0):.4f}  P={m.get('precision',0):.4f}  "
                      f"R={m.get('recall',0):.4f}  FPR={m.get('fpr',0):.4f}  "
                      f"Lat={m.get('latency',{}).get('mean',0):.1f}s")

    print("\nDone.")


if __name__ == "__main__":
    main()
