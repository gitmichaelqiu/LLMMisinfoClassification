"""First real-API evaluation across domains and architectures.

THREAD-SAFE SEQUENTIAL VERSION.
The OpenAI client is NOT thread-safe, so we make all calls sequentially
with fresh httpx clients per call, avoiding ThreadPoolExecutor issues.

Runs small controlled samples (10 per domain) through real DeepSeek API calls
for single-shot, voting N=3, MoA, and RAG (where corpus available). Reports
metrics, MoA degeneracy diagnostics, failure analysis, and clear separation
between real API results, mock baselines, and policy-only outcomes.

Usage:
    python -m src.real_evaluation --output output/real_eval_results.json

Requires DEEPSEEK_API_KEY in .env or environment.
"""

from __future__ import annotations

import csv
import json
import os
import random
import re
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

# Bypass system proxy (Clash V-Ninja on port 7890) for DeepSeek API calls
for _var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
    os.environ.pop(_var, None)
os.environ["no_proxy"] = "*"
os.environ["NO_PROXY"] = "*"

# Lazy imports for thread safety
from src.datasets import load_dataset
from src.llm_clients import MockClient
from src.metrics import (
    classification_metrics,
    compute_confusion_matrix,
    compute_ece,
    compute_ppv,
)
from src.prompts import (
    SINGLE_SHOT_SYSTEM,
    SINGLE_SHOT_USER,
    RAG_SYSTEM,
    RAG_USER,
    VOTING_VARIATIONS,
    format_user_prompt,
)
from src.schemas import (
    Verdict,
    VerificationItem,
    VerificationResult,
    VerifierConfig,
)

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────────

SAMPLE_SIZE = 10  # items per domain
SEED = 42
MODEL = "deepseek-v4-flash"
TEMPERATURE = 0.0
MAX_TOKENS = 512

DOMAINS = ["finance", "healthcare", "political"]

KAGGLE_FAKE_NEWS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "raw", "political", "kaggle_fake_news.csv"
)
FINANCIAL_NEWS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "raw", "finance", "financial_news.csv"
)

PPV_BASE_RATES = [0.001, 0.01, 0.05, 0.10, 0.25, 0.50]


# ── Thread-safe API helpers ─────────────────────────────────────────────────

def _make_openai_client():
    """Create a fresh OpenAI client (thread-safe, no shared state)."""
    import httpx
    from openai import OpenAI

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    http_client = httpx.Client(
        proxy=None,
        timeout=httpx.Timeout(60.0, connect=15.0),
        follow_redirects=True,
    )
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        http_client=http_client,
    )


def _llm_call(system_prompt: str, user_prompt: str, model: str = MODEL,
              temperature: float = TEMPERATURE, max_tokens: int = MAX_TOKENS) -> Tuple[str, float]:
    """Make a single LLM API call. Returns (raw_text, latency_s)."""
    client = _make_openai_client()
    start = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        elapsed = time.time() - start
        return resp.choices[0].message.content or "", elapsed
    except Exception as e:
        elapsed = time.time() - start
        raise RuntimeError(f"API call failed after {elapsed:.1f}s: {e}")


# ── Response Parser (reusing SingleShotVerifier's patterns) ────────────────

_VERDICT_PATTERN = re.compile(r"Verdict:\s*(REAL|FAKE|ESCALATE|EXAGGERATED)", re.IGNORECASE)
_CONFIDENCE_PATTERN = re.compile(r"Confidence:\s*(\d+)", re.IGNORECASE)
_FLAGS_PATTERN = re.compile(r"Flags:\s*\[([^\]]*)\]", re.IGNORECASE)
_REASONING_PATTERN = re.compile(r"Reasoning:\s*(.+)", re.IGNORECASE)

_VERDICT_MAP = {
    "REAL": Verdict.REAL,
    "FAKE": Verdict.FAKE,
    "ESCALATE": Verdict.ESCALATE,
    "EXAGGERATED": Verdict.EXAGGERATED,
}


def _parse_response(raw: str, item_id: str, latency_s: float,
                    extra_metadata: Optional[Dict[str, Any]] = None) -> VerificationResult:
    """Parse a raw LLM response into a VerificationResult."""
    verdict_match = _VERDICT_PATTERN.search(raw)
    verdict = Verdict.REAL
    if verdict_match:
        key = verdict_match.group(1).upper()
        verdict = _VERDICT_MAP.get(key, Verdict.REAL)

    confidence_match = _CONFIDENCE_PATTERN.search(raw)
    confidence = 0.5
    if confidence_match:
        confidence = int(confidence_match.group(1)) / 100.0

    flags_match = _FLAGS_PATTERN.search(raw)
    flags = flags_match.group(1) if flags_match else ""

    reasoning_match = _REASONING_PATTERN.search(raw)
    reasoning = reasoning_match.group(1) if reasoning_match else ""

    evidence = []
    if flags:
        evidence.append(f"Flags: {flags}")
    if reasoning:
        evidence.append(reasoning)

    metadata = {
        "model": MODEL,
        "prompt_template": "real_eval",
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return VerificationResult(
        item_id=item_id,
        verdict=verdict,
        confidence=min(max(confidence, 0.0), 1.0),
        latency_s=latency_s,
        evidence=evidence,
        metadata=metadata,
    )


# ── Data Loaders ────────────────────────────────────────────────────────────


@dataclass
class DomainData:
    name: str
    test_items: List[VerificationItem]
    rag_corpus: Optional[List[VerificationItem]] = None
    rag_notes: str = ""


def _load_finance_items() -> DomainData:
    """Load 10 balanced finance items from synthetic fallback."""
    adapter = load_dataset("finance")
    all_items = adapter.load()
    rng = random.Random(SEED)
    fake = [it for it in all_items if it.ground_truth == Verdict.FAKE]
    real = [it for it in all_items if it.ground_truth == Verdict.REAL]
    k = min(SAMPLE_SIZE // 2, len(fake), len(real))
    test_items = rng.sample(fake, k) + rng.sample(real, k)
    rng.shuffle(test_items)

    rag_corpus: List[VerificationItem] = []
    rag_notes = ""
    try:
        with open(FINANCIAL_NEWS_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                text = row.get("text", row.get("title", ""))
                if len(text) > 50:
                    rag_corpus.append(VerificationItem.create(
                        claim_text=text[:2000],
                        ground_truth=Verdict.REAL,
                        metadata={"domain": "finance", "source": "financial_news_.csv"},
                    ))
                    if i >= 199:
                        break
        rag_notes = f"Corpus: {len(rag_corpus)} real financial news articles"
    except Exception as e:
        rag_notes = f"RAG skipped: {e}"

    return DomainData(name="finance", test_items=test_items,
                      rag_corpus=rag_corpus or None, rag_notes=rag_notes)


def _load_healthcare_items() -> DomainData:
    """Load 10 balanced healthcare items from health_headlines.csv."""
    adapter = load_dataset("healthcare")
    all_items = adapter.load()
    rng = random.Random(SEED)
    fake = [it for it in all_items if it.ground_truth == Verdict.FAKE]
    real = [it for it in all_items if it.ground_truth == Verdict.REAL]
    k = min(SAMPLE_SIZE // 2, len(fake), len(real))
    test_items = rng.sample(fake, k) + rng.sample(real, k)
    rng.shuffle(test_items)
    test_ids = {it.id for it in test_items}
    rag_corpus = [it for it in all_items if it.id not in test_ids]
    return DomainData(name="healthcare", test_items=test_items, rag_corpus=rag_corpus,
                      rag_notes=f"Corpus: {len(rag_corpus)} healthcare items")


def _load_political_items() -> DomainData:
    """Load 10 balanced political items from kaggle_fake_news_FULL.csv."""
    rng = random.Random(SEED)
    try:
        with open(KAGGLE_FAKE_NEWS_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            lines = list(reader)
        fake, real = [], []
        for row in lines:
            title = row.get("title", row.get("text", ""))
            label_raw = row.get("label", "0")
            if not title:
                continue
            try:
                label = int(label_raw.strip())
            except (ValueError, AttributeError):
                continue
            item = VerificationItem.create(
                claim_text=title[:500],
                ground_truth=Verdict.FAKE if label == 1 else Verdict.REAL,
                metadata={"domain": "political", "source": "kaggle_fake_news_FULL.csv"},
            )
            (fake if label == 1 else real).append(item)
        k = min(SAMPLE_SIZE // 2, len(fake), len(real))
        test_items = rng.sample(fake, k) + rng.sample(real, k)
        rng.shuffle(test_items)
        test_ids = {it.id for it in test_items}
        rag_corpus = [it for it in fake + real if it.id not in test_ids][:200]
        return DomainData(name="political", test_items=test_items, rag_corpus=rag_corpus,
                          rag_notes=f"Corpus: {len(rag_corpus)} political items")
    except Exception as e:
        adapter = load_dataset("political")
        all_items = adapter.load()
        fake = [it for it in all_items if it.ground_truth == Verdict.FAKE]
        real = [it for it in all_items if it.ground_truth == Verdict.REAL]
        k = min(SAMPLE_SIZE // 2, len(fake), len(real))
        test_items = rng.sample(fake, k) + rng.sample(real, k)
        rng.shuffle(test_items)
        return DomainData(name="political", test_items=test_items,
                          rag_notes=f"RAG skipped: {e}")


DOMAIN_LOADERS = {
    "finance": _load_finance_items,
    "healthcare": _load_healthcare_items,
    "political": _load_political_items,
}


# ── Verifier implementations (thread-safe, sequential) ─────────────────────


def _run_single_shot(items: List[VerificationItem]) -> List[VerificationResult]:
    """Sequential single-shot verification."""
    results = []
    for item in items:
        user_prompt = format_user_prompt(SINGLE_SHOT_USER, claim_text=item.claim_text,
                                         context=item.context or "")
        try:
            raw, lat = _llm_call(SINGLE_SHOT_SYSTEM, user_prompt)
            r = _parse_response(raw, item.id, lat)
        except Exception as e:
            r = VerificationResult(item_id=item.id, verdict=Verdict.REAL,
                                   confidence=0.5, latency_s=0, evidence=[str(e)],
                                   metadata={"error": str(e)})
        results.append(r)
    return results


def _run_voting(items: List[VerificationItem], n_voters: int = 3) -> List[VerificationResult]:
    """Sequential voting: make N calls per item (no thread pool), majority aggregation."""
    results = []
    for item in items:
        start = time.time()
        item_results = []
        user_prompt = format_user_prompt(SINGLE_SHOT_USER, claim_text=item.claim_text,
                                         context=item.context or "")
        for i in range(n_voters):
            variation = VOTING_VARIATIONS[i % len(VOTING_VARIATIONS)]
            try:
                raw, lat = _llm_call(variation, user_prompt, temperature=0.3)
                parsed = _parse_response(raw, item.id, lat)
            except Exception as e:
                parsed = VerificationResult(item_id=item.id, verdict=Verdict.REAL,
                                            confidence=0.5, latency_s=0,
                                            metadata={"error": f"voter_{i}: {e}"})
            item_results.append(parsed)

        total_latency = time.time() - start

        # Majority aggregation
        verdicts = [r.verdict for r in item_results]
        counter = Counter(verdicts)
        most_common = counter.most_common(1)
        final_verdict = most_common[0][0] if most_common else Verdict.REAL
        avg_conf = float(np.mean([r.confidence for r in item_results]))
        agreement = max(counter.values()) / len(verdicts) if verdicts else 0.0

        evidence = [
            f"Voters: {n_voters}",
            f"Aggregation: majority",
            f"Agreement rate: {agreement:.2f}",
            f"Verdict distribution: {dict(counter)}",
        ]
        results.append(VerificationResult(
            item_id=item.id, verdict=final_verdict, confidence=avg_conf,
            latency_s=total_latency, evidence=evidence,
            metadata={"model": MODEL, "n_voters": n_voters, "agreement_rate": agreement},
        ))
    return results


MOA_BELIEVER_SYSTEM = """You are Agent 1 (The Believer) in an information verification debate.
Your role is to find evidence SUPPORTING the authenticity of the claim. You argue the claim is REAL and should NOT be flagged.
Analyze the claim and build the strongest possible case for authenticity.
Be thorough — list specific supporting evidence.
Output:
Verdict: REAL or UNCERTAIN
Evidence: <bulleted list of supporting points>
Confidence: <0-100>"""

MOA_SKEPTIC_SYSTEM = """You are Agent 2 (The Skeptic) in an information verification debate.
Your role is to find evidence that the claim is FALSE or misleading. You argue the claim is FAKE and should be flagged.
Analyze the claim and build the strongest possible case against authenticity.
Be thorough — list specific problematic evidence.
Output:
Verdict: FAKE or UNCERTAIN
Evidence: <bulleted list of problematic points>
Confidence: <0-100>"""

MOA_RISK_OFFICER_SYSTEM = """You are Agent 3 (The Risk Officer) in an information verification debate.
You have received analyses from two agents: The Believer argues the claim is REAL; The Skeptic argues the claim is FAKE.
Evaluate both arguments, weigh the evidence, and deliver a final verdict.
Output your final verdict in EXACTLY this format:
Verdict: REAL or FAKE or ESCALATE
Confidence: <0-100>
Flags: [contradiction|implausibility|inconsistency|none]
Reasoning: <one-sentence rationale weighing both sides>"""


def _run_moa(items: List[VerificationItem]) -> List[VerificationResult]:
    """Sequential MoA: Believer → Skeptic → Risk Officer per item."""
    results = []
    for item in items:
        start = time.time()
        user_prompt = f"Claim to verify:\n{item.claim_text}\n\nContext:\n{item.context or 'None provided.'}"

        try:
            bel_raw, bel_lat = _llm_call(MOA_BELIEVER_SYSTEM, user_prompt)
            skep_raw, skep_lat = _llm_call(MOA_SKEPTIC_SYSTEM, user_prompt)
        except Exception as e:
            results.append(VerificationResult(
                item_id=item.id, verdict=Verdict.REAL, confidence=0.5,
                latency_s=time.time() - start, evidence=[f"MoA agent error: {e}"],
                metadata={"error": str(e)},
            ))
            continue

        synthesis_prompt = (
            f"Claim to verify:\n{item.claim_text}\n\n"
            f"=== Believer's Analysis (pro-REAL) ===\n{bel_raw}\n\n"
            f"=== Skeptic's Analysis (pro-FAKE) ===\n{skep_raw}\n\n"
            f"Based on both analyses above, deliver your final verdict."
        )

        try:
            ro_raw, ro_lat = _llm_call(MOA_RISK_OFFICER_SYSTEM, synthesis_prompt)
        except Exception as e:
            results.append(VerificationResult(
                item_id=item.id, verdict=Verdict.REAL, confidence=0.5,
                latency_s=time.time() - start,
                evidence=[f"Risk Officer error: {e}"],
                metadata={"error": str(e), "believer_output": bel_raw[:200],
                          "skeptic_output": skep_raw[:200]},
            ))
            continue

        total_latency = time.time() - start
        r = _parse_response(ro_raw, item.id, total_latency,
                            extra_metadata={"verifier_type": "moa",
                                            "believer_output": bel_raw[:200],
                                            "skeptic_output": skep_raw[:200]})
        # Add debate evidence
        bel_conf = re.search(r"Confidence:\s*(\d+)", bel_raw)
        skep_conf = re.search(r"Confidence:\s*(\d+)", skep_raw)
        r.evidence = [
            f"Believer confidence: {bel_conf.group(1) if bel_conf else 'unknown'}",
            f"Skeptic confidence: {skep_conf.group(1) if skep_conf else 'unknown'}",
        ] + r.evidence
        results.append(r)
    return results


def _run_rag(items: List[VerificationItem], corpus: List[VerificationItem]) -> List[VerificationResult]:
    """Sequential RAG: retrieve from corpus, then LLM call."""
    # Build a simple TF-IDF retriever (no sentence-transformers dependency)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    corpus_texts = [it.claim_text for it in corpus]
    corpus_ids = [it.id for it in corpus]
    vec = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = vec.fit_transform(corpus_texts)

    results = []
    for item in items:
        start = time.time()

        # Retrieve
        query_vec = vec.transform([item.claim_text])
        sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_k = 3
        top_indices = np.argsort(sims)[::-1][:top_k]

        hits = []
        retrieval_lat = 0.0
        for idx in top_indices:
            if sims[idx] <= 0:
                continue
            hits.append((corpus_ids[idx], corpus_texts[idx], float(sims[idx])))

        retrieval_lat = time.time() - start

        # Format context
        context_parts = []
        for doc_id, doc_text, score in hits:
            context_parts.append(f"[Document {doc_id}] (relevance={score:.3f})\n{doc_text}\n")
        context = "".join(context_parts)[:2000]

        # LLM call
        user_prompt = format_user_prompt(RAG_USER, claim_text=item.claim_text, context=context)
        try:
            raw, llm_lat = _llm_call(RAG_SYSTEM, user_prompt)
        except Exception as e:
            results.append(VerificationResult(
                item_id=item.id, verdict=Verdict.REAL, confidence=0.5,
                latency_s=time.time() - start, evidence=[f"RAG LLM error: {e}"],
                metadata={"retrieval_hits": len(hits), "error": str(e)},
            ))
            continue

        total_latency = time.time() - start
        r = _parse_response(raw, item.id, total_latency, extra_metadata={
            "retriever_type": "sparse", "top_k": top_k,
            "retrieval_hits": len(hits),
            "evidence_ids": [doc_id for doc_id, _, _ in hits],
            "relevance_scores": [s for _, _, s in hits],
            "retrieval_latency_s": round(retrieval_lat, 4),
            "llm_latency_s": round(llm_lat, 4),
            "context_length_chars": len(context),
        })
        # Prepend retrieved evidence
        retrieval_evidence = [
            f"[Retrieved] doc={doc_id} score={s:.3f}: {text[:120]}"
            for doc_id, text, s in hits
        ]
        r.evidence = retrieval_evidence + r.evidence
        results.append(r)
    return results


def _run_mock_baseline(items: List[VerificationItem]) -> List[VerificationResult]:
    """Mock baseline (always FAKE with 0.85 confidence)."""
    mock = MockClient(fixed_verdict="FAKE", fixed_confidence=85)
    from src.verifier_single_shot import SingleShotVerifier
    verifier = SingleShotVerifier(config=VerifierConfig(model=MODEL), client=mock)
    return verifier.verify_batch(items)


# ── Metrics helpers ─────────────────────────────────────────────────────────


def _latency_stats(latencies: List[float]) -> Dict[str, float]:
    if not latencies:
        return {"mean": 0, "median": 0, "p95": 0, "min": 0, "max": 0}
    arr = sorted(latencies)
    return {"mean": float(np.mean(arr)), "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)), "min": float(min(arr)), "max": float(max(arr))}


def _compute_ppv_table(tp: int, fp: int, tn: int, fn: int) -> List[Dict[str, float]]:
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 1.0
    ppvs = compute_ppv(sensitivity=sens, specificity=spec, base_rates=PPV_BASE_RATES)
    return [{"base_rate": br, "ppv": ppv, "npv": npv} for br, ppv, npv in ppvs
            if br in PPV_BASE_RATES]


def find_failures(results: List[VerificationResult], items: List[VerificationItem]) -> List[Dict[str, Any]]:
    failures = []
    for r, it in zip(results, items):
        if it.ground_truth is None:
            continue
        if r.verdict != it.ground_truth:
            ft = "FN" if it.ground_truth == Verdict.FAKE else "FP"
            failures.append({"item_id": it.id, "claim": it.claim_text[:150],
                             "domain": it.metadata.get("domain", ""), "type": ft,
                             "truth": it.ground_truth.name, "predicted": r.verdict.name,
                             "confidence": r.confidence, "latency_s": r.latency_s})
    return failures


def diagnose_moa_degeneracy(results: List[VerificationResult], items: List[VerificationItem],
                            domain: str) -> Dict[str, Any]:
    fake_preds = sum(1 for r in results if r.verdict == Verdict.FAKE)
    total = len(results)
    fake_fraction = fake_preds / total if total else 0.0
    base_rate = sum(1 for it in items if it.ground_truth == Verdict.FAKE) / len(items)
    cm = compute_confusion_matrix(results, [it.ground_truth or Verdict.REAL for it in items])
    prec = cm.tp / (cm.tp + cm.fp) if (cm.tp + cm.fp) else 0.0

    believer_confs, skeptic_confs = [], []
    for r in results:
        bel = r.metadata.get("believer_output", "")
        skep = r.metadata.get("skeptic_output", "")
        m = re.search(r"Confidence:\s*(\d+)", bel)
        if m:
            believer_confs.append(int(m.group(1)))
        m = re.search(r"Confidence:\s*(\d+)", skep)
        if m:
            skeptic_confs.append(int(m.group(1)))

    return {"domain": domain, "fake_prediction_rate": fake_fraction,
            "test_set_fake_base_rate": base_rate, "precision": prec,
            "precision_above_base_rate": prec > base_rate,
            "likely_degenerate": prec <= base_rate,
            "avg_believer_confidence": float(np.mean(believer_confs)) if believer_confs else None,
            "avg_skeptic_confidence": float(np.mean(skeptic_confs)) if skeptic_confs else None,
            "believer_skeptic_divergence": (
                float(np.mean(skeptic_confs) - np.mean(believer_confs))
                if believer_confs and skeptic_confs else None)}


# ── Evaluation runner ──────────────────────────────────────────────────────


@dataclass
class EvalResult:
    architecture: str
    domain: str
    skipped: bool = False
    skip_reason: str = ""
    error: str = ""
    results: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    latency: Dict[str, float] = field(default_factory=dict)
    verdict_distribution: Dict[str, int] = field(default_factory=dict)
    ppv_table: List[Dict[str, float]] = field(default_factory=list)
    ece: float = 0.0
    failure_examples: List[Dict[str, Any]] = field(default_factory=list)
    raw_output_path: str = ""


def run_evaluation(arch_name: str, domain_data: DomainData,
                   runner_fn: Callable, output_dir: str,
                   api_count: int = 0) -> Tuple[EvalResult, int]:
    """Run an architecture on a domain and return metrics."""
    domain = domain_data.name
    items = domain_data.test_items
    result = EvalResult(architecture=arch_name, domain=domain)

    print(f"\n{'='*60}")
    print(f"  [{domain}] Running {arch_name} on {len(items)} items...")
    print(f"{'='*60}")

    try:
        raw_results = runner_fn(items)
    except Exception as e:
        traceback.print_exc()
        result.error = str(e)
        return result, 0

    latencies = []
    result_list = []
    for r in raw_results:
        latencies.append(r.latency_s)
        result_list.append({
            "item_id": r.item_id, "verdict": r.verdict.name,
            "confidence": r.confidence, "latency_s": r.latency_s,
            "evidence": r.evidence[:3],
            "metadata": {k: v for k, v in r.metadata.items()
                         if k not in ("raw_output_first_100",)},
        })

    result.results = result_list
    result.latency = _latency_stats(latencies)

    truths = [it.ground_truth or Verdict.REAL for it in items]
    try:
        cm = compute_confusion_matrix(raw_results, truths)
        metrics = classification_metrics(raw_results, truths)
        result.metrics = asdict(metrics)
        result.verdict_distribution = dict(Counter(r.verdict.name for r in raw_results))
        result.ppv_table = _compute_ppv_table(cm.tp, cm.fp, cm.tn, cm.fn)
        result.failure_examples = find_failures(raw_results, items)
        correct = [r.verdict == t for r, t in zip(raw_results, truths)]
        ece, _, _, _ = compute_ece([r.confidence for r in raw_results], correct, n_bins=5)
        result.ece = ece
    except Exception as e:
        result.error = f"Metrics error: {e}"

    # Save raw outputs
    os.makedirs(os.path.join(output_dir, "raw_outputs"), exist_ok=True)
    raw_path = os.path.join(output_dir, "raw_outputs", f"{domain}_{arch_name}.json")
    with open(raw_path, "w") as f:
        json.dump({"architecture": arch_name, "domain": domain, "model": MODEL,
                   "items": [{"id": it.id, "claim": it.claim_text[:200],
                              "ground_truth": it.ground_truth.name if it.ground_truth else None}
                             for it in items],
                   "results": [{"item_id": r.item_id, "verdict": r.verdict.name,
                                "confidence": r.confidence, "latency_s": r.latency_s,
                                "evidence": r.evidence, "metadata": r.metadata}
                               for r in raw_results]},
                  f, indent=2, default=str)
    result.raw_output_path = raw_path

    m = result.metrics
    print(f"  └─ F1={m.get('f1', 0):.4f}  P={m.get('precision', 0):.4f}  "
          f"R={m.get('recall', 0):.4f}  "
          f"Lat={result.latency.get('mean', 0):.1f}s  ECE={result.ece:.4f}")
    if result.failure_examples:
        fn_c = sum(1 for f in result.failure_examples if f['type'] == 'FN')
        fp_c = sum(1 for f in result.failure_examples if f['type'] == 'FP')
        print(f"     Failures: {len(result.failure_examples)} ({fn_c} FN, {fp_c} FP)")

    return result, api_count or len(items)


# ── Main ────────────────────────────────────────────────────────────────────


def main(output_dir: str = "output/real_eval"):
    os.makedirs(output_dir, exist_ok=True)

    config = VerifierConfig(model=MODEL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)

    # Verify connectivity
    print("Verifying DeepSeek API connectivity...")
    try:
        raw, lat = _llm_call("You are a test assistant.", "Reply with a single word: OK.")
        print(f"  API OK — response: {raw[:50]} ({lat:.1f}s)")
    except Exception as e:
        print(f"  API ERROR: {e}")
        print("  Falling back to mock mode.")
        return

    # Load data
    print(f"\n{'='*60}")
    print("  Loading data for all domains...")
    print(f"{'='*60}")
    loaded_data: Dict[str, DomainData] = {}
    for domain_name in DOMAINS:
        data = DOMAIN_LOADERS[domain_name]()
        loaded_data[domain_name] = data
        n_fake = sum(1 for it in data.test_items if it.ground_truth == Verdict.FAKE)
        n_real = sum(1 for it in data.test_items if it.ground_truth == Verdict.REAL)
        print(f"  {domain_name}: {len(data.test_items)} test items "
              f"(F:{n_fake} R:{n_real})  RAG: {data.rag_notes}")

    # Architecture runners
    arch_runners: Dict[str, Tuple[Callable, int]] = {
        "single_shot": (_run_single_shot, SAMPLE_SIZE),
        "voting_n3": (_run_voting, SAMPLE_SIZE * 3),
        "moa": (_run_moa, SAMPLE_SIZE * 3),
        "rag": (None, SAMPLE_SIZE),  # handled separately
    }

    domain_results: Dict[str, Dict[str, EvalResult]] = {}
    all_api_counts: Dict[str, int] = {}

    for domain_name in DOMAINS:
        data = loaded_data[domain_name]
        domain_results[domain_name] = {}

        for arch_name, (runner_fn, api_count) in arch_runners.items():
            if arch_name == "rag":
                if data.rag_corpus and len(data.rag_corpus) >= 10:
                    def make_rag_runner(corpus):
                        return lambda its: _run_rag(its, corpus)
                    result, _ = run_evaluation("rag", data,
                                               make_rag_runner(data.rag_corpus),
                                               output_dir, api_count)
                else:
                    result = EvalResult(architecture="rag", domain=domain_name,
                                        skipped=True, skip_reason=data.rag_notes or "No corpus")
            else:
                result, _ = run_evaluation(arch_name, data, runner_fn, output_dir, api_count)

            domain_results[domain_name][arch_name] = result
            all_api_counts[arch_name] = all_api_counts.get(arch_name, 0) + api_count

    # Mock baselines
    print(f"\n{'='*60}")
    print("  Running mock baselines...")
    print(f"{'='*60}")
    for domain_name in DOMAINS:
        data = loaded_data[domain_name]
        result, _ = run_evaluation("mock_baseline", data, _run_mock_baseline, output_dir)
        domain_results[domain_name]["mock_baseline"] = result

    # ── Build report ──
    report = {
        "metadata": {
            "model": MODEL,
            "timestamp": datetime.utcnow().isoformat(),
            "sample_size_per_domain": SAMPLE_SIZE,
            "total_api_calls": sum(all_api_counts.values()),
            "api_calls_by_architecture": all_api_counts,
            "estimated_cost_usd": sum(all_api_counts.values()) * 0.00015,
        },
        "domains": {},
        "summary": {},
    }

    for domain_name in DOMAINS:
        arch_results = domain_results[domain_name]
        domain_section = {}
        for arch_name, er in arch_results.items():
            section = {"skipped": er.skipped, "skip_reason": er.skip_reason,
                       "error": er.error, "metrics": er.metrics, "latency": er.latency,
                       "verdict_distribution": er.verdict_distribution,
                       "ece": er.ece, "ppv": er.ppv_table,
                       "failure_count": len(er.failure_examples),
                       "failures": er.failure_examples[:5],
                       "raw_output_path": er.raw_output_path}
            domain_section[arch_name] = section

        # MoA degeneracy
        if "moa" in arch_results and not arch_results["moa"].skipped and not arch_results["moa"].error:
            moa_res = [VerificationResult(item_id=r["item_id"], verdict=Verdict[r["verdict"]],
                                          confidence=r["confidence"], latency_s=r["latency_s"],
                                          evidence=r.get("evidence", []),
                                          metadata=r.get("metadata", {}))
                       for r in arch_results["moa"].results]
            truth_map = {it.id: it.ground_truth for it in loaded_data[domain_name].test_items}
            moa_items = [VerificationItem(id=it.id, claim_text="", ground_truth=truth_map.get(it.id))
                         for it in loaded_data[domain_name].test_items
                         if it.id in truth_map and truth_map[it.id] is not None]
            moa_res_filtered = [r for r in moa_res if r.item_id in {it.id for it in moa_items}]
            if moa_res_filtered and moa_items:
                domain_section["moa_degeneracy"] = diagnose_moa_degeneracy(
                    moa_res_filtered, moa_items, domain_name)

        report["domains"][domain_name] = domain_section

    # Cross-domain summary
    summary: Dict[str, Any] = {}
    for arch_name in ["single_shot", "voting_n3", "moa", "rag"]:
        f1s = {d: domain_results[d][arch_name].metrics.get("f1", 0)
               for d in DOMAINS if arch_name in domain_results[d]
               and not domain_results[d][arch_name].skipped
               and not domain_results[d][arch_name].error}
        if f1s:
            vals = list(f1s.values())
            summary[f"{arch_name}_cross_domain_f1"] = {"values": f1s,
                "mean": float(np.mean(vals)), "std": float(np.std(vals)),
                "range": max(vals) - min(vals)}
    summary["architecture_ranking_by_domain"] = {
        d: dict(sorted({a: domain_results[d][a].metrics.get("f1", 0)
                        for a in ["single_shot", "voting_n3", "moa", "rag"]
                        if a in domain_results[d] and not domain_results[d][a].skipped
                        and not domain_results[d][a].error}.items(),
                       key=lambda x: x[1], reverse=True))
        for d in DOMAINS
    }
    summary["mock_vs_real"] = {
        d: {"mock_f1": domain_results[d]["mock_baseline"].metrics.get("f1", 0),
            "real_f1": domain_results[d]["single_shot"].metrics.get("f1", 0)}
        for d in DOMAINS if "mock_baseline" in domain_results[d] and "single_shot" in domain_results[d]
    }
    summary["voting_improvement"] = {
        d: {"single_shot_f1": domain_results[d]["single_shot"].metrics.get("f1", 0),
            "voting_n3_f1": domain_results[d]["voting_n3"].metrics.get("f1", 0),
            "improvement": domain_results[d]["voting_n3"].metrics.get("f1", 0)
                          - domain_results[d]["single_shot"].metrics.get("f1", 0),
            "voting_better": domain_results[d]["voting_n3"].metrics.get("f1", 0)
                            > domain_results[d]["single_shot"].metrics.get("f1", 0)}
        for d in DOMAINS
    }
    all_fails = []
    for d in DOMAINS:
        for an, er in domain_results[d].items():
            for f in er.failure_examples:
                f["architecture"] = an
                all_fails.append(f)
    all_fails.sort(key=lambda x: x.get("confidence", 1.0))
    summary["top_failures"] = all_fails[:10]
    report["summary"] = summary

    report_path = os.path.join(output_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to {report_path}")

    # ── Print compact report ──
    print(f"\n{'='*60}")
    print("  COMPACT REPORT")
    print(f"{'='*60}")

    print(f"\n## Real-API Evaluation Report")
    print(f"\n**Model**: {MODEL}")
    print(f"**Total API calls**: {sum(all_api_counts.values())}")
    print(f"**Estimated cost**: ~${sum(all_api_counts.values()) * 0.00015:.4f}")
    print(f"**Sample size**: {SAMPLE_SIZE} per domain")

    for domain in DOMAINS:
        print(f"\n### {domain.upper()}")
        headers = ["Architecture", "F1", "Prec", "Rec", "FPR", "FNR", "ECE", "Mean Lat"]
        rows = []
        for arch_name in ["single_shot", "voting_n3", "moa", "rag", "mock_baseline"]:
            er = domain_results[domain].get(arch_name)
            if er is None:
                continue
            if er.skipped:
                rows.append([arch_name, "SKIPPED", "", "", "", "", "", er.skip_reason[:40]])
                continue
            if not er.metrics:
                rows.append([arch_name, "ERROR", er.error[:40], "", "", "", "", ""])
                continue
            m = er.metrics
            rows.append([arch_name,
                         f"{m.get('f1', 0):.4f}", f"{m.get('precision', 0):.4f}",
                         f"{m.get('recall', 0):.4f}", f"{m.get('fpr', 0):.4f}",
                         f"{m.get('fnr', 0):.4f}", f"{er.ece:.4f}",
                         f"{er.latency.get('mean', 0):.1f}s"])

        col_widths = [max(len(str(r[i])) for r in rows + [headers]) for i in range(len(headers))]
        header_line = "  " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        print(f"\n  {header_line}")
        print(f"  " + "-" * len(header_line))
        for row in rows:
            print("  " + " | ".join(str(c).ljust(col_widths[i]) for i, c in enumerate(row)))

    # MoA degeneracy
    print(f"\n### MoA Degeneracy Diagnostics")
    for domain in DOMAINS:
        dd = report["domains"][domain].get("moa_degeneracy", {})
        if dd:
            flag = "⚠️  DEGENERATE" if dd.get("likely_degenerate") else "✅ NOT DEGENERATE"
            print(f"  {domain}: FAKE pred rate={dd['fake_prediction_rate']:.2%} "
                  f"(base={dd['test_set_fake_base_rate']:.2%}) "
                  f"prec={dd['precision']:.4f} — {flag}")
            if dd.get("avg_believer_confidence") is not None:
                print(f"    Believer avg={dd['avg_believer_confidence']:.1f}  "
                      f"Skeptic avg={dd['avg_skeptic_confidence']:.1f}  "
                      f"div={dd['believer_skeptic_divergence']:.1f}")

    # Voting improvement
    print(f"\n### Voting N=3 vs Single-Shot")
    for domain in DOMAINS:
        vi = summary.get("voting_improvement", {}).get(domain, {})
        if vi:
            d = "↑ BETTER" if vi["voting_better"] else "↓ WORSE"
            print(f"  {domain}: SS F1={vi['single_shot_f1']:.4f} → "
                  f"V3 F1={vi['voting_n3_f1']:.4f} ({d})")

    # Top failures
    print(f"\n### Top Failure Modes")
    for i, f in enumerate(summary.get("top_failures", [])[:5]):
        print(f"  {i+1}. [{f['type']}] {f.get('domain','?')}/{f.get('architecture','?')}: "
              f"\"{f['claim'][:80]}...\" "
              f"(conf={f['confidence']:.2f}, truth={f['truth']}, pred={f['predicted']})")

    # PPV
    print(f"\n### PPV at Operational Base Rates (Single-Shot)")
    for domain in DOMAINS:
        ppvs = report["domains"][domain].get("single_shot", {}).get("ppv", [])
        if ppvs:
            print(f"  {domain}: " + " | ".join(
                f"P(F)={p['base_rate']:.1%}: PPV={p['ppv']:.1%}" for p in ppvs))
    print()


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "output/real_eval"
    main(output_dir)
