"""CSV data loading for test sets and knowledge corpora."""

from __future__ import annotations

import csv

from src.schemas import Verdict, VerificationItem


def _parse_label(raw_label: str) -> Verdict | None:
    """Parse a label string into a Verdict, or return None if unparsable."""
    raw = raw_label.strip()
    if raw in ("0", "1"):
        return Verdict.FAKE if raw == "1" else Verdict.REAL
    if raw.lower() in ("real", "fake"):
        return Verdict.FAKE if raw.lower() == "fake" else Verdict.REAL
    return None


def _pick_text(row: dict) -> str:
    """Extract the claim text from a CSV row, trying common column names."""
    return row.get("headline", row.get("title", row.get("tweet", row.get("text", ""))))


def load_test_csv(path: str, domain: str) -> list[VerificationItem]:
    """Load a CSV test set into ``VerificationItem`` objects."""
    with open(path, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    items: list[VerificationItem] = []
    for r in rows:
        gt = _parse_label(r.get("label", ""))
        if gt is None:
            continue
        items.append(
            VerificationItem.create(
                claim_text=_pick_text(r),
                ground_truth=gt,
                metadata={"domain": domain},
            )
        )
    return items


def load_corpus_csv(path: str, domain: str) -> list[VerificationItem]:
    """Load a CSV knowledge corpus into ``VerificationItem`` objects."""
    with open(path, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    items: list[VerificationItem] = []
    for r in rows:
        gt = _parse_label(r.get("label", ""))
        if gt is None:
            continue
        items.append(
            VerificationItem.create(
                claim_text=_pick_text(r),
                ground_truth=gt,
                metadata={"domain": domain},
            )
        )
    return items


def load_all_data(
    finance_test_path: str,
    finance_corpus_path: str,
    covid_test_path: str,
    covid_corpus_path: str,
) -> dict[str, dict[str, list[VerificationItem]]]:
    """Load both domains and print a summary."""
    all_data: dict[str, dict[str, list[VerificationItem]]] = {}
    for domain_key, test_path, corpus_path in [
        ("finance", finance_test_path, finance_corpus_path),
        ("healthcare", covid_test_path, covid_corpus_path),
    ]:
        test_items = load_test_csv(test_path, domain_key)
        corpus_items = load_corpus_csv(corpus_path, domain_key)
        test_texts = set(it.claim_text.strip().lower() for it in test_items)
        corpus_texts = set(it.claim_text.strip().lower() for it in corpus_items)
        overlap = test_texts & corpus_texts
        n_real = sum(1 for it in test_items if it.ground_truth == Verdict.REAL)
        n_fake = sum(1 for it in test_items if it.ground_truth == Verdict.FAKE)
        print(
            f"  {domain_key}: test={len(test_items)} "
            f"(R={n_real} F={n_fake}), "
            f"corpus={len(corpus_items)}, overlap={len(overlap)}"
        )
        all_data[domain_key] = {"test": test_items, "corpus": corpus_items}
    return all_data
