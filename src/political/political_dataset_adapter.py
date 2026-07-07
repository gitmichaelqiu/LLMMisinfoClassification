"""Political statements dataset adapter.

Adapts political fact-check statements into the unified VerificationItem
schema. Uses a synthetic fallback since the project does not include a
real political dataset — this validates the framework's ability to
represent a non-finance, non-health domain.

Labels: 0=REAL, 1=FAKE.
"""

from __future__ import annotations

from typing import List, Optional

from src.datasets import DatasetAdapter
from src.schemas import Verdict, VerificationItem


class PoliticalDatasetAdapter(DatasetAdapter):
    """Adapter for political fact-check statements.

    Currently uses a synthetic fallback set covering common political
    misinformation patterns (economic claims, election integrity,
    policy proposals).

    Args:
        path: Unused (reserved for future CSV loading).
    """

    def __init__(self, path: Optional[str] = None):
        self._path = path

    def load(self) -> List[VerificationItem]:
        """Load political statements dataset.

        Returns:
            List of VerificationItem objects.
        """
        return self._synthetic_fallback()

    def _synthetic_fallback(self) -> List[VerificationItem]:
        """Generate synthetic political fact-check items."""
        template = [
            # (headline, is_fake)
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
