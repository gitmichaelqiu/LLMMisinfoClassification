"""Architecture runners: Single-Shot, Voting, MoA, and TF-IDF baseline."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

from src.api import _llm_call, _parse_response, _run_parallel
from src.config import SEED
from src.prompts import (
    CANONICAL_SYSTEM,
    MOA_JUDGE,
    MOA_JUDGE_RAG,
    MOA_SKEPTIC,
    MOA_SKEPTIC_RAG,
    MOA_SUPPORTER,
    MOA_SUPPORTER_RAG,
)
from src.retrieval import build_retriever, make_user_prompt
from src.schemas import Verdict, VerificationItem, VerificationResult


# ══════════════════════════════════════════════════════════════════
#  TF-IDF + Logistic Regression baseline
# ═════════════════════════════════════════════════════════════════
def run_tfidf_baseline(
    items: list[VerificationItem],
    corpus: list[VerificationItem],
) -> list[VerificationResult]:
    """Train a TF-IDF + LogisticRegression classifier on the corpus and
    evaluate on *items*.  No LLM calls — cheap sklearn baseline."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    corpus_texts = [it.claim_text for it in corpus]
    corpus_labels = [1 if it.ground_truth == Verdict.FAKE else 0 for it in corpus]

    clf = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=10000)),
        (
            "lr",
            LogisticRegression(
                class_weight="balanced", random_state=SEED, max_iter=1000
            ),
        ),
    ])
    clf.fit(corpus_texts, corpus_labels)

    results: list[VerificationResult] = []
    for item in items:
        pred = clf.predict([item.claim_text])[0]
        prob = clf.predict_proba([item.claim_text])[0]
        if pred == 1:
            verdict, confidence = Verdict.FAKE, prob[1]
        else:
            verdict, confidence = Verdict.REAL, prob[0]
        results.append(
            VerificationResult(
                item_id=item.id,
                verdict=verdict,
                confidence=float(confidence),
                latency_s=0.001,
                evidence=[],
                metadata={"architecture": "tfidf_baseline"},
            )
        )
    return results


# ══════════════════════════════════════════════════════════════════
#  Single-Shot
# ═════════════════════════════════════════════════════════════════
def run_single_shot(
    items: list[VerificationItem],
    rag_on: bool = False,
    corpus: list[VerificationItem] | None = None,
) -> list[VerificationResult]:
    """Single LLM call per item."""
    vec = tfidf = texts = None
    if rag_on and corpus:
        vec, tfidf, texts = build_retriever(corpus)

    def make_call(item: VerificationItem) -> VerificationResult:
        up = make_user_prompt(item.claim_text, vec, tfidf, texts)
        raw, lat = _llm_call(CANONICAL_SYSTEM, up)
        return _parse_response(
            raw, item.id, lat, {"architecture": "single_shot", "rag": rag_on}
        )

    raw = _run_parallel(
        [lambda it=item: make_call(it) for item in items], "SS"
    )
    return [
        (
            r
            if r
            else VerificationResult(
                item_id=it.id,
                verdict=Verdict.REAL,
                confidence=0.5,
                latency_s=0,
                metadata={"error": "failure"},
            )
        )
        for r, it in zip(raw, items)
    ]


# ══════════════════════════════════════════════════════════════════
#  Voting  (generate N=7,  subset for 1/3/5/7)
# ═════════════════════════════════════════════════════════════════
def run_voting_all_n(
    items: list[VerificationItem],
    rag_on: bool = False,
    corpus: list[VerificationItem] | None = None,
    n_total: int = 7,
) -> dict[int, tuple[list[VerificationResult], list[list[VerificationResult]]]]:
    """Generate *n_total* voter outputs per item, then aggregate for
    N ∈ {1, 3, 5, 7}.

    Returns a dict ``{N: (aggregated_results, per_item_voters)}``.
    """
    vec = tfidf = texts = None
    if rag_on and corpus:
        vec, tfidf, texts = build_retriever(corpus)

    callables: list[Any] = []
    for item in items:
        up = make_user_prompt(item.claim_text, vec, tfidf, texts)
        for v in range(n_total):
            def _vote(it: VerificationItem = item, u: str = up, vidx: int = v):
                txt, lat = _llm_call(CANONICAL_SYSTEM, u)
                return (
                    it.id,
                    vidx,
                    _parse_response(
                        txt,
                        it.id,
                        lat,
                        {"voter_idx": vidx, "architecture": "voting", "rag": rag_on},
                    ),
                )
            callables.append(_vote)

    raw = _run_parallel(callables, f"Voting N={n_total}")

    # Group by item, sort by voter index
    item_voters_all: dict[str, list[VerificationResult]] = defaultdict(list)
    for r in raw:
        if r is not None:
            _, vidx, vr = r
            item_voters_all[vr.item_id].append(vr)
    for iid in item_voters_all:
        item_voters_all[iid].sort(key=lambda x: x.metadata.get("voter_idx", 0))

    result_map: dict[int, tuple[list[VerificationResult], list[list[VerificationResult]]]] = {}
    for N in [1, 3, 5, 7]:
        threshold = N // 2 + 1
        aggregated: list[VerificationResult] = []
        per_item_voters: list[list[VerificationResult]] = []
        for item in items:
            all_voters = item_voters_all.get(item.id, [])
            voters = all_voters[:N]
            per_item_voters.append(voters)
            if not voters:
                aggregated.append(
                    VerificationResult(
                        item_id=item.id,
                        verdict=Verdict.REAL,
                        confidence=0.5,
                        latency_s=0,
                        metadata={
                            "architecture": f"voting_n{N}",
                            "error": "no_voters",
                        },
                    )
                )
                continue
            counter = Counter(v.verdict for v in voters)
            mc = counter.most_common(1)
            verdict = (
                mc[0][0] if (mc and mc[0][1] >= threshold) else Verdict.ESCALATE
            )
            aggregated.append(
                VerificationResult(
                    item_id=item.id,
                    verdict=verdict,
                    confidence=float(np.mean([v.confidence for v in voters])),
                    latency_s=max(v.latency_s for v in voters),
                    evidence=[f"N={N}: {dict(counter)}"],
                    metadata={
                        "architecture": f"voting_n{N}",
                        "rag": rag_on,
                        "n_voters": len(voters),
                        "threshold": threshold,
                        "verdict_distribution": {
                            k.name: v for k, v in counter.items()
                        },
                        "disagreement_rate": (
                            1.0 - (mc[0][1] / len(voters)) if mc else 1.0
                        ),
                    },
                )
            )
        result_map[N] = (aggregated, per_item_voters)
    return result_map


# ══════════════════════════════════════════════════════════════════
#  Mixture of Agents
# ═════════════════════════════════════════════════════════════════
def run_moa(
    items: list[VerificationItem],
    rag_on: bool = False,
    corpus: list[VerificationItem] | None = None,
) -> list[VerificationResult]:
    """MoA: Supporter + Skeptic (concurrent) → Judge (sequential)."""
    vec = tfidf = texts = None
    if rag_on and corpus:
        vec, tfidf, texts = build_retriever(corpus)

    evidence_map: dict[str, str] = {}
    p1_callables: list[Any] = []
    for item in items:
        up = make_user_prompt(item.claim_text, vec, tfidf, texts)
        evidence_map[item.id] = up
        sp = MOA_SUPPORTER_RAG if rag_on else MOA_SUPPORTER
        sk = MOA_SKEPTIC_RAG if rag_on else MOA_SKEPTIC

        def _sup(iid: str = item.id, u: str = up, p: str = sp):
            txt, lat = _llm_call(p, u)
            return ("supporter", iid, txt, lat)

        def _ske(iid: str = item.id, u: str = up, p: str = sk):
            txt, lat = _llm_call(p, u)
            return ("skeptic", iid, txt, lat)

        p1_callables.append(_sup)
        p1_callables.append(_ske)

    raw_p1 = _run_parallel(p1_callables, "MoA P1")

    supporter_out: dict[str, tuple[str, float]] = {}
    skeptic_out: dict[str, tuple[str, float]] = {}
    for result in raw_p1:
        if result is None:
            continue
        role, iid, text, lat = result
        if role == "supporter":
            supporter_out[iid] = (text, lat)
        else:
            skeptic_out[iid] = (text, lat)

    judge_prompt = MOA_JUDGE_RAG if rag_on else MOA_JUDGE
    p2_callables: list[Any] = []
    for item in items:
        up = evidence_map.get(item.id, f"Claim to verify:\n{item.claim_text}")
        sup_text, sup_lat = supporter_out.get(item.id, ("Error", 0))
        ske_text, ske_lat = skeptic_out.get(item.id, ("Error", 0))
        p1_lat = max(sup_lat, ske_lat)
        ctx = (
            f"{up}\n\n── Supporter's Analysis ──\n{sup_text}\n\n"
            f"── Skeptic's Analysis ──\n{ske_text}\n"
        )

        def _judge(iid: str = item.id, c: str = ctx, p1: float = p1_lat):
            txt, lat = _llm_call(judge_prompt, c)
            vr = _parse_response(
                txt, iid, lat, {"architecture": "moa", "rag": rag_on}
            )
            vr.latency_s = p1 + lat
            return (iid, vr)

        p2_callables.append(_judge)

    raw_p2 = _run_parallel(p2_callables, "MoA P2")
    return [vr for r in raw_p2 if r is not None for _, vr in [r]]
