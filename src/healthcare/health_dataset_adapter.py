"""Healthcare dataset adapter.

Loads healthcare/medical news headlines and maps them into the unified
VerificationItem schema. Dataset sourced from health_headlines.csv.

Labels: 0=REAL, 1=FAKE.
"""

from __future__ import annotations

import csv
import os
from typing import List, Optional

from src.datasets import DatasetAdapter
from src.schemas import Verdict, VerificationItem


class HealthDatasetAdapter(DatasetAdapter):
    """Adapter for healthcare/medical news verification.

    Loads from CSV files in data/raw/health/.

    Args:
        path: Base data directory. Defaults to ``data/`` relative to this file.
    """

    def __init__(self, path: Optional[str] = None):
        base = path or os.path.join(os.path.dirname(__file__), "..", "..", "data")
        self._search_dir = os.path.join(base, "raw", "health")

    def load(self) -> List[VerificationItem]:
        """Load all items from the health dataset.

        Returns:
            List of VerificationItem objects.
        """
        if not os.path.isdir(self._search_dir):
            return self._synthetic_fallback()

        items: List[VerificationItem] = []
        for fname in sorted(os.listdir(self._search_dir)):
            if not fname.endswith(".csv"):
                continue
            fpath = os.path.join(self._search_dir, fname)
            try:
                items.extend(self._load_csv(fpath, fname))
            except Exception as e:
                print(f"Warning: could not load {fpath}: {e}")

        if not items:
            items = self._synthetic_fallback()
        return items

    def _load_csv(self, fpath: str, fname: str) -> List[VerificationItem]:
        """Load items from a CSV file."""
        items: List[VerificationItem] = []
        with open(fpath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                headline = row.get("headline", row.get("text", row.get("claim", "")))
                label_raw = row.get("label", row.get("ground_truth", ""))
                label = self._parse_label(label_raw)
                items.append(
                    VerificationItem.create(
                        claim_text=headline,
                        ground_truth=label,
                        metadata={
                            "domain": "healthcare",
                            "source": fname,
                        },
                    )
                )
        return items

    @staticmethod
    def _parse_label(label_raw: str) -> Optional[Verdict]:
        """Parse a label string into a Verdict."""
        label = label_raw.strip().upper()
        if label in ("0", "REAL"):
            return Verdict.REAL
        elif label in ("1", "FAKE"):
            return Verdict.FAKE
        return None

    def _synthetic_fallback(self) -> List[VerificationItem]:
        """Return a small synthetic set when no CSV is available."""
        fake_headlines = [
            "Miracle drug cures all cancers in single clinical trial of 3 patients",
            "WHO confirms vaccines cause autism in landmark study retracted 3 times",
            "FDA approves untested gene therapy based on CEO's personal testimony",
        ]
        real_headlines = [
            "CDC reports 15% reduction in hospital-acquired infections following hand hygiene campaign",
            "WHO publishes updated guidelines for malaria prevention in sub-Saharan Africa",
            "Moderna announces Phase III trial results for seasonal influenza vaccine",
        ]
        items = [
            VerificationItem.create(
                claim_text=h, ground_truth=Verdict.FAKE,
                metadata={"domain": "healthcare", "source": "synthetic_fallback"},
            )
            for h in fake_headlines
        ]
        items.extend(
            VerificationItem.create(
                claim_text=h, ground_truth=Verdict.REAL,
                metadata={"domain": "healthcare", "source": "synthetic_fallback"},
            )
            for h in real_headlines
        )
        return items
