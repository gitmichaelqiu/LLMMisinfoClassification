"""Political statements dataset adapter.

Adapts political fact-check statements into the unified VerificationItem
schema. Loads from CSV files in data/raw/political/ or falls back to
synthetic examples when no CSV is available.

Labels: 0=REAL, 1=FAKE.
"""

from __future__ import annotations

import csv
import os
from typing import List, Optional

from src.datasets import DatasetAdapter
from src.schemas import Verdict, VerificationItem


class PoliticalDatasetAdapter(DatasetAdapter):
    """Adapter for political fact-check statements.

    Loads from CSV files in data/raw/political/. Falls back to synthetic
    examples when no CSV files are found.

    Args:
        path: Base data directory. Defaults to ``data/`` relative to this file.
    """

    def __init__(self, path: Optional[str] = None):
        base = path or os.path.join(os.path.dirname(__file__), "..", "..", "data")
        self._search_dir = os.path.join(base, "raw", "political")

    def load(self) -> List[VerificationItem]:
        """Load all items from the political dataset.

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
        with open(fpath, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                headline = row.get(
                    "title",
                    row.get("headline", row.get("text", row.get("claim", ""))),
                )
                label_raw = row.get("label", row.get("ground_truth", ""))
                label = self._parse_label(label_raw)
                items.append(
                    VerificationItem.create(
                        claim_text=headline,
                        ground_truth=label,
                        metadata={
                            "domain": "political",
                            "source": fname,
                            "source_file": fname,
                        },
                    )
                )
        return items

    @staticmethod
    def _parse_label(label_raw: str) -> Optional[Verdict]:
        """Parse a label string into a Verdict."""
        label = label_raw.strip().upper()
        if label in ("0", "REAL", "REAL/HOLD"):
            return Verdict.REAL
        elif label in ("1", "FAKE", "FAKE/INTERVENE"):
            return Verdict.FAKE
        return None

    def _synthetic_fallback(self) -> List[VerificationItem]:
        """Generate synthetic political fact-check items."""
        template = [
            ("Unemployment rate drops to historic 2.1% following new trade deal", False),
            ("Congress passes bipartisan infrastructure bill with 78-22 vote", False),
            ("Federal budget deficit decreases for third consecutive fiscal year", False),
            ("Voter turnout reaches 68% in latest general election, highest in decades", False),
            ("Supreme Court upholds constitutionality of campaign finance disclosure law", False),
            ("Opposition candidate caught orchestrating widespread voter fraud rings", True),
            ("Foreign government secretly funded 40% of incumbent's re-election campaign", True),
            ("New tax bill would eliminate income tax for 95% of households, say analysts", True),
            ("Election results reversed after mysterious late-night ballot dump in swing state", True),
            ("Leaked memo reveals government planning to monitor all social media messages", True),
        ]
        items: List[VerificationItem] = []
        for headline, is_fake in template:
            ground_truth = Verdict.FAKE if is_fake else Verdict.REAL
            items.append(
                VerificationItem.create(
                    claim_text=headline,
                    ground_truth=ground_truth,
                    metadata={
                        "domain": "political",
                        "source": "synthetic_fallback",
                        "is_fake": is_fake,
                    },
                )
            )
        return items
