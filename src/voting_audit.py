"""Voting audit: establish baseline equivalence and voter agreement analysis.

Fixes the temperature mismatch between Single-Shot and Voting N=1.
Both use the same temperature (0.7), same prompt, same everything.

Captures per-voter predictions for agreement analysis.
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
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

for _var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
    os.environ.pop(_var, None)
os.environ["no_proxy"] = "*"
os.environ["NO_PROXY"] = "*"

from src.metrics import compute_confusion_matrix, classification_metrics
from src.schemas import Verdict, VerificationItem, VerificationResult

load_dotenv()

# ── Canonical config ──────────────────────────────────────────────────────

SEED = 42
MODEL = "deepseek-v4-flash"
TEMPERATURE = 0.7          # canonical: same for SS and Voting
MAX_TOKENS = 512
MAX_CONCURRENCY = 400
OUTPUT_DIR = "results/voting_audit"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "raw_outputs"), exist_ok=True)

FINANCE_CSV = os.path.join("data", "raw", "finance", "financial_news.csv")
HEALTH_CSV = os.path.join("data", "raw", "health", "health_headlines.csv")

SAMPLE_CONFIG = {
    "finance": {"n": 50, "n_real": 25, "n_fake": 25},
    "healthcare": {"n": 40, "n_real": 20, "n_fake": 20},
}


# ── API Call ────────────────────────────────────────────────────────────────

def _llm_call(sys_prompt: str, user_prompt: str) -> Tuple[str, float]:
    import httpx
    from openai import OpenAI

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    http_client = httpx.Client(
        proxy=None,
        timeout=httpx.Timeout(180.0, connect=30.0),
        follow_redirects=True,
    )
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        http_client=http_client,
    )

    start = time.time()
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        elapsed = time.time() - start
        return resp.choices[0].message.content or "", elapsed
    except Exception as e:
        elapsed = time.time() - start
        raise RuntimeError(f"API call failed after {elapsed:.1f}s: {e}")
    finally:
        http_client.close()


# ── Canonical Verifier Prompt ─────────────────────────────────────────────

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


# ── Parser ────────────────────────────────────────────────────────────────

_VERDICT_RE = re.compile(r"Verdict:\s*(REAL|FAKE|ESCALATE|EXAGGERATED)", re.IGNORECASE)
_CONFIDENCE_RE = re.compile(r"Confidence:\s*(\d+)", re.IGNORECASE)
_VERDICT_MAP = {
    "REAL": Verdict.REAL, "FAKE": Verdict.FAKE,
    "ESCALATE": Verdict.ESCALATE, "EXAGGERATED": Verdict.EXAGGERATED,
}


def _parse_response(raw: str, item_id: str, latency_s: float,
                    extra_meta: Optional[Dict] = None) -> VerificationResult:
    verdict = Verdict.REAL
    vm = _VERDICT_RE.search(raw)
    if vm:
        verdict = _VERDICT_MAP.get(vm.group(1).upper(), Verdict.REAL)
    confidence = 0.5
    cm = _CONFIDENCE_RE.search(raw)
    if cm:
        confidence = int(cm.group(1)) / 100.0
    return VerificationResult(
        item_id=item_id, verdict=verdict,
        confidence=max(0.0, min(1.0, confidence)),
        latency_s=latency_s, evidence=[], metadata=extra_meta or {},
    )


# ── Data Loading ──────────────────────────────────────────────────────────

def _load_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _sample_balanced(domain: str) -> Tuple[List[VerificationItem], List[VerificationItem]]:
    rng = np.random.RandomState(SEED)
    cfg = SAMPLE_CONFIG[domain]
    path = FINANCE_CSV if domain == "finance" else HEALTH_CSV
    rows = _load_csv_rows(path)

    real = [r for r in rows if r.get("label", "").strip() == "0"]
    fake = [r for r in rows if r.get("label", "").strip() == "1"]

    n_real = min(cfg["n_real"], len(real))
    n_fake = min(cfg["n_fake"], len(fake))

    rng.shuffle(real)
    rng.shuffle(fake)

    sel_real = real[:n_real]
    sel_fake = fake[:n_fake]

    def to_item(r: Dict[str, str], gt: Verdict) -> VerificationItem:
        claim = r.get("headline", r.get("title", r.get("text", "")))
        return VerificationItem.create(
            claim_text=claim, ground_truth=gt,
            metadata={"domain": domain, "source": "csv_sample"},
        )

    test_items = [to_item(r, Verdict.FAKE) for r in sel_fake] + [to_item(r, Verdict.REAL) for r in sel_real]
    rng.shuffle(test_items)

    corpus_items = [to_item(r, Verdict.FAKE) for r in fake[n_fake:]] + [to_item(r, Verdict.REAL) for r in real[n_real:]]
    if len(corpus_items) > 1000:
        rng.shuffle(corpus_items)
        corpus_items = corpus_items[:1000]

    print(f"  [{domain}] test={len(test_items)} (R={n_real} F={n_fake}), corpus={len(corpus_items)}")
    return test_items, corpus_items


# ── Parallel ──────────────────────────────────────────────────────────────

def _run_parallel(callables: List[Callable], desc: str = "") -> List[Any]:
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


# ══════════════════════════════════════════════════════════════════════════
#  VOTING RUNNER (per-voter capture)
# ══════════════════════════════════════════════════════════════════════════

def run_voting(items: List[VerificationItem], n_voters: int,
               ) -> Tuple[List[VerificationResult], List[List[VerificationResult]]]:
    """Run self-consistency voting.

    Returns:
        (aggregated_results, per_voter_results) where per_voter_results[i]
        is a list of n_voters VerificationResult for item i.
    """
    callables = []
    for item in items:
        up = f"Claim to verify:\n{item.claim_text}\n\nIs this claim REAL or FAKE?"
        for v in range(n_voters):
            def _vote(it=item, u=up, vidx=v):
                txt, lat = _llm_call(CANONICAL_SYSTEM, u)
                return (it.id, vidx, _parse_response(txt, it.id, lat, {
                    "voter_idx": vidx, "n_voters": n_voters,
                }))
            callables.append(_vote)

    raw = _run_parallel(callables, f"N={n_voters}")

    # Group per-item per-voter
    item_voters: Dict[str, List[VerificationResult]] = defaultdict(list)
    for r in raw:
        if r is not None:
            _, vidx, vr = r
            item_voters[vr.item_id].append(vr)

    threshold = n_voters // 2 + 1
    aggregated = []
    per_item_voters = []

    for item in items:
        voters = sorted(item_voters.get(item.id, []), key=lambda x: x.metadata.get("voter_idx", 0))
        if not voters:
            aggregated.append(VerificationResult(
                item_id=item.id, verdict=Verdict.REAL, confidence=0.5, latency_s=0,
                metadata={"error": "no_voters"},
            ))
            continue

        counter = Counter(v.verdict for v in voters)
        mc = counter.most_common(1)
        if mc and mc[0][1] >= threshold:
            verdict = mc[0][0]
        else:
            verdict = Verdict.ESCALATE

        per_item_voters.append(voters)
        avg_conf = float(np.mean([v.confidence for v in voters]))
        max_lat = max(v.latency_s for v in voters)

        aggregated.append(VerificationResult(
            item_id=item.id, verdict=verdict, confidence=avg_conf,
            latency_s=max_lat,
            evidence=[f"N={n_voters}: {dict(counter)}"],
            metadata={
                "n_voters": n_voters, "threshold": threshold,
                "verdict_distribution": {k.name: v for k, v in counter.items()},
                "disagreement_rate": 1.0 - (mc[0][1] / len(voters)) if mc else 1.0,
            },
        ))

    return aggregated, per_item_voters


# ══════════════════════════════════════════════════════════════════════════
#  ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

def compute_metrics(items, results):
    truths = [it.ground_truth or Verdict.REAL for it in items]
    m = classification_metrics(results, truths)
    n_esc = sum(1 for r in results if r.verdict == Verdict.ESCALATE)
    latencies = [r.latency_s for r in results if r.latency_s > 0]
    return {
        "f1": m.f1, "precision": m.precision, "recall": m.recall,
        "fpr": m.fpr, "fnr": m.fnr, "accuracy": m.accuracy,
        "escalate_rate": n_esc / len(results) if results else 0,
        "mean_latency": float(np.mean(latencies)) if latencies else 0,
        "confusion": {"tp": m.confusion.tp, "fp": m.confusion.fp,
                       "tn": m.confusion.tn, "fn": m.confusion.fn},
    }


def analyze_voter_agreement(items, per_item_voters, n_voters):
    """Detailed voter agreement analysis."""
    truths = [it.ground_truth or Verdict.REAL for it in items]
    ver_map = {Verdict.REAL: 0, Verdict.FAKE: 1, Verdict.ESCALATE: 2}

    stats = {
        "total_items": len(items),
        "n_voters": n_voters,
        "unanimous": 0,
        "majority_flip_v1": 0,
        "corrected_by_voting": 0,
        "damaged_by_voting": 0,
        "pairwise_agreement": 0.0,
        "voter1_verdicts": Counter(),
        "majority_verdicts": Counter(),
        "voter_fake_counts": [],
    }

    pairwise_agree = 0
    pairwise_total = 0

    for i, voters in enumerate(per_item_voters):
        if not voters:
            continue
        gt = truths[i] if i < len(truths) else None
        verdicts = [v.verdict for v in voters]
        v1 = verdicts[0]

        # Unanimous?
        if len(set(verdicts)) == 1:
            stats["unanimous"] += 1

        # Majority
        counter = Counter(verdicts)
        majority_v, majority_c = counter.most_common(1)[0]
        threshold = n_voters // 2 + 1
        if majority_c >= threshold:
            stats["majority_verdicts"][majority_v.name] += 1
        else:
            stats["majority_verdicts"]["ESCALATE"] += 1

        stats["voter1_verdicts"][v1.name] += 1

        # Flip?
        if v1 != Verdict.ESCALATE and majority_v != Verdict.ESCALATE:
            if majority_v != v1 and n_voters > 1:
                stats["majority_flip_v1"] += 1

        # Pairwise agreement
        for a in range(len(verdicts)):
            for b in range(a + 1, len(verdicts)):
                pairwise_total += 1
                if verdicts[a] == verdicts[b]:
                    pairwise_agree += 1

        # Ground-truth comparisons
        if gt is not None:
            majority_verdict = majority_v if majority_c >= threshold else Verdict.ESCALATE
            # Compare: voter 1 alone vs majority
            v1_alone = v1
            # Was voter 1 correct?
            majority_threshold = n_voters // 2 + 1
            maj_v = majority_v if majority_c >= majority_threshold else Verdict.ESCALATE

            # "Corrected by voting": voter 1 was wrong, majority is right
            if v1_alone != gt and maj_v == gt:
                stats["corrected_by_voting"] += 1
            # "Damaged by voting": voter 1 was right, majority is wrong
            if v1_alone == gt and maj_v != gt and n_voters > 1:
                stats["damaged_by_voting"] += 1

    stats["pairwise_agreement"] = pairwise_agree / pairwise_total if pairwise_total > 0 else 0
    # Per-voter FAKE counts
    for vi in range(n_voters):
        fc = sum(1 for voters in per_item_voters if vi < len(voters) and voters[vi].verdict == Verdict.FAKE)
        stats["voter_fake_counts"].append(fc)

    return stats


def save_results(domain, n_voters, items, aggregated, per_item_voters):
    """Save raw outputs including per-voter predictions."""
    out = {
        "architecture": f"voting_n{n_voters}",
        "domain": domain,
        "model": MODEL,
        "temperature": TEMPERATURE,
        "items": [{"id": it.id, "claim": it.claim_text[:200],
                    "ground_truth": it.ground_truth.name if it.ground_truth else None}
                   for it in items],
        "aggregated": [{
            "item_id": r.item_id, "verdict": r.verdict.name,
            "confidence": r.confidence, "latency_s": r.latency_s,
            "metadata": r.metadata,
        } for r in aggregated],
        "per_voter": [[{
            "item_id": vr.item_id, "verdict": vr.verdict.name,
            "confidence": vr.confidence, "latency_s": vr.latency_s,
            "voter_idx": vr.metadata.get("voter_idx", -1),
        } for vr in voters] for voters in per_item_voters],
    }
    path = os.path.join(OUTPUT_DIR, "raw_outputs", f"{domain}_{n_voters}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    return path


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("VOTING BASELINE AUDIT")
    print("=" * 70)
    print(f"Model: {MODEL}")
    print(f"Temperature: {TEMPERATURE} (same for all runs)")
    print(f"Prompt: CANONICAL_SYSTEM (same for all voters)")
    print(f"Aggregation: majority (threshold=N//2+1), else ESCALATE")
    sys.stdout.flush()

    # Verify API
    print("\nVerifying API...")
    try:
        txt, lat = _llm_call("You are a test assistant.", "Reply with OK.")
        print(f"  OK ({lat:.1f}s)")
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    # Load data
    all_test = {}
    for domain in ["finance", "healthcare"]:
        test, _ = _sample_balanced(domain)
        all_test[domain] = test

    # Run N=1,3,5,7
    n_values = [1, 3, 5, 7]
    all_results = {}

    for domain in ["finance", "healthcare"]:
        print(f"\n{'─' * 60}")
        print(f"{domain.upper()} — N={1,3,5,7}")
        print(f"{'─' * 60}")

        domain_results = {}
        for n in n_values:
            print(f"\n  N={n}...")
            sys.stdout.flush()

            start_t = time.time()
            aggregated, per_voter = run_voting(all_test[domain], n)
            elapsed = time.time() - start_t

            metrics = compute_metrics(all_test[domain], aggregated)
            agreement = analyze_voter_agreement(all_test[domain], per_voter, n)
            path = save_results(domain, n, all_test[domain], aggregated, per_voter)

            domain_results[n] = {
                "metrics": metrics,
                "agreement": agreement,
                "elapsed_s": round(elapsed, 1),
                "per_voter_count": sum(len(v) for v in per_voter),
                "raw_path": path,
            }

            m = metrics
            a = agreement
            print(f"    F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} "
                  f"FPR={m['fpr']:.4f} Acc={m['accuracy']:.4f} ESC={m['escalate_rate']:.2%} "
                  f"Lat={m['mean_latency']:.1f}s")
            print(f"    Agreement: unanimous={a['unanimous']}/{a['total_items']} "
                  f"({a['unanimous']/a['total_items']:.1%}) "
                  f"pairwise={a['pairwise_agreement']:.4f} "
                  f"flip_v1={a['majority_flip_v1']} "
                  f"corrected={a['corrected_by_voting']} "
                  f"damaged={a['damaged_by_voting']}")
            print(f"    V1 verdicts: {dict(a['voter1_verdicts'])}")
            print(f"    Voter FAKE counts: {a['voter_fake_counts']}")
            sys.stdout.flush()

        all_results[domain] = domain_results

    # ── Print corrected Stage 1 table ──
    print(f"\n{'=' * 70}")
    print("CORRECTED STAGE 1: VOTING SENSITIVITY")
    print(f"(All at temperature={TEMPERATURE}, same prompt)")
    print(f"{'=' * 70}")

    for domain in ["finance", "healthcare"]:
        print(f"\n{domain.upper()}:")
        h = f"{'N':6s} {'F1':8s} {'P':8s} {'R':8s} {'FPR':8s} {'Acc':8s} {'ESC':7s} {'Lat':7s} {'Unanim':7s} {'PairAgr':7s} {'Corr':5s} {'Dam':5s}"
        print(h)
        print("-" * len(h))
        for n in n_values:
            r = all_results[domain][n]
            m = r["metrics"]
            a = r["agreement"]
            print(f"{n:6d} {m['f1']:.4f}  {m['precision']:.4f}  {m['recall']:.4f}  "
                  f"{m['fpr']:.4f}  {m['accuracy']:.4f}  {m['escalate_rate']:.2%}  "
                  f"{m['mean_latency']:.1f}s  {a['unanimous']/a['total_items']:.2%}  "
                  f"{a['pairwise_agreement']:.4f}  {a['corrected_by_voting']:3d}  {a['damaged_by_voting']:3d}")

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("SUMMARY: DOES N>1 ADD VALUE?")
    print(f"{'=' * 70}")

    for domain in ["finance", "healthcare"]:
        print(f"\n{domain.upper()}:")
        n1_f1 = all_results[domain][1]["metrics"]["f1"]
        print(f"  N=1 F1 = {n1_f1:.4f} (baseline)")
        for n in [3, 5, 7]:
            r = all_results[domain][n]
            m = r["metrics"]
            a = r["agreement"]
            print(f"  N={n} F1 = {m['f1']:.4f}  Δ={m['f1'] - n1_f1:+.4f}  "
                  f"unanimous={a['unanimous']}/{a['total_items']} "
                  f"corrected={a['corrected_by_voting']} damaged={a['damaged_by_voting']}")

    # Final determination
    print(f"\n{'─' * 40}")
    print("DETERMINATION")
    print(f"{'─' * 40}")
    for domain in ["finance", "healthcare"]:
        improvements = 0
        regressions = 0
        for n in [3, 5, 7]:
            r = all_results[domain][n]
            a = r["agreement"]
            improvements += a["corrected_by_voting"]
            regressions += a["damaged_by_voting"]
        print(f"  {domain}: total corrected={improvements}, total damaged={regressions}")
        if improvements > regressions:
            print(f"    → Voting adds marginal value (net +{improvements - regressions} correct)")
        elif improvements == regressions:
            print(f"    → Voting is neutral")
        else:
            print(f"    → Voting is harmful (net {improvements - regressions} correct)")

    # Compute total API calls
    total_api = 0
    for domain in ["finance", "healthcare"]:
        nitems = len(all_test[domain])
        for n in n_values:
            total_api += nitems * n
    print(f"\n  Total API calls: {total_api}")
    print(f"  Estimated cost: ${total_api * 0.00015:.4f}")


if __name__ == "__main__":
    main()
