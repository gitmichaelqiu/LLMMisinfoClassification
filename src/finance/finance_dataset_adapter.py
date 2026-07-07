"""Finance dataset adapter.

Loads the legacy financial news dataset (synthetic + real) and maps it into
the unified VerificationItem schema. Serves as the template for all future
domain adapters.
"""

from __future__ import annotations

import csv
import os
from typing import List, Optional

from src.datasets import DatasetAdapter
from src.schemas import Verdict, VerificationItem


class FinanceDatasetAdapter(DatasetAdapter):
    """Adapter for legacy finance news verification datasets.

    Loads from CSV files with columns: headline, label, domain, source.
    Labels: 0=REAL, 1=FAKE.

    Search paths (in order):
      - data/raw/finance/     (domain-specific raw CSVs)
      - data/synthetic/       (generated synthetic CSVs)
    """

    def load(self) -> List[VerificationItem]:
        """Load all items from the finance dataset.

        Searches in order:
        1. data/raw/finance/     — domain-specific raw files
        2. data/synthetic/       — generated synthetic files

        Does NOT scan data/raw/ broadly, avoiding cross-domain
        contamination with health or other datasets.

        Returns:
            List of VerificationItem objects.
        """
        items = []
        base = os.path.join(os.path.dirname(__file__), "..", "..", "data")
        search_dirs = [
            os.path.join(base, "raw", "finance"),
            os.path.join(base, "synthetic"),
        ]

        for data_dir in search_dirs:
            if not os.path.isdir(data_dir):
                continue
            for fname in sorted(os.listdir(data_dir)):
                if not fname.endswith(".csv"):
                    continue
                fpath = os.path.join(data_dir, fname)
                try:
                    items.extend(self._load_csv(fpath, fname))
                except Exception as e:
                    print(f"Warning: could not load {fpath}: {e}")

        # If no CSV files found, load synthetic items from generate_dataset
        if not items:
            items = self._load_synthetic_fallback()

        return items

    def _load_csv(self, fpath: str, fname: str) -> List[VerificationItem]:
        """Load items from a CSV file."""
        items = []
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
                            "domain": "finance",
                            "source": fname,
                            "source_file": fname,
                        },
                    )
                )
        return items

    def _load_synthetic_fallback(self) -> List[VerificationItem]:
        """Generate synthetic items when no CSV files are available.

        This fallback generates signal-only items to allow the framework
        to run and be tested even without dataset files on disk.
        """
        items = []
        fake_headlines = [
            "Breaking: Federal Reserve declares all bank deposits void effective immediately",
            "SEC announces Apple is under investigation for accounting fraud",
            "Elon Musk announces Tesla is filing for bankruptcy",
            "JP Morgan reports $50 billion trading loss due to algorithm error",
            "Amazon to lay off 40% of workforce in cost-cutting measure",
        ]
        real_headlines = [
            "Federal Reserve holds interest rates steady at 5.25%",
            "Apple reports Q4 revenue of $89.5 billion, beating estimates",
            "Tesla delivers 435,000 vehicles in Q3 2024",
            "Microsoft announces $10 billion share buyback program",
            "Goldman Sachs upgrades S&P 500 target to 6,000",
        ]
        for h in fake_headlines:
            items.append(
                VerificationItem.create(
                    claim_text=h, ground_truth=Verdict.FAKE,
                    metadata={"domain": "finance", "source": "synthetic_fallback"},
                )
            )
        for h in real_headlines:
            items.append(
                VerificationItem.create(
                    claim_text=h, ground_truth=Verdict.REAL,
                    metadata={"domain": "finance", "source": "synthetic_fallback"},
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
        elif label in ("2", "ESCALATE", "GREY_SWAN/ESCALATE"):
            return Verdict.ESCALATE
        elif label in ("3", "EXAGGERATED", "EXAGGERATED/PARTIAL_REDUCE"):
            return Verdict.EXAGGERATED
        return None
