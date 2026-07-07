"""Base dataset adapter and loading utilities.

Defines the DatasetAdapter ABC that all domain adapters must implement,
and the load_dataset() factory function.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from src.schemas import Verdict, VerificationItem


class DatasetAdapter(ABC):
    """Abstract base for domain-specific dataset adapters.

    Each domain adapter maps its raw data into the unified VerificationItem
    schema, providing train/test splits and dataset metadata.
    """

    @abstractmethod
    def load(self) -> List[VerificationItem]:
        """Load and return all items in the dataset."""
        ...

    def train_test_split(
        self,
        test_size: float = 0.3,
        seed: int = 42,
    ) -> Tuple[List[VerificationItem], List[VerificationItem]]:
        """Split items into train and test sets.

        Args:
            test_size: Fraction of items to use for testing.
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (train_items, test_items).
        """
        items = self.load()
        n = len(items)
        n_test = max(1, int(n * test_size))

        import random
        rng = random.Random(seed)
        indices = list(range(n))
        rng.shuffle(indices)

        test_idx = set(indices[:n_test])
        train = [items[i] for i in indices if i not in test_idx]
        test = [items[i] for i in indices if i in test_idx]
        return train, test

    def item_counts(self) -> Dict[str, int]:
        """Return dataset statistics.

        Returns:
            Dict with keys: 'total', 'real', 'fake', 'escalate', 'exaggerated', 'unlabeled'.
        """
        items = self.load()
        counts = {"total": len(items), "real": 0, "fake": 0, "escalate": 0, "exaggerated": 0, "unlabeled": 0}
        for item in items:
            if item.ground_truth == Verdict.REAL:
                counts["real"] += 1
            elif item.ground_truth == Verdict.FAKE:
                counts["fake"] += 1
            elif item.ground_truth == Verdict.ESCALATE:
                counts["escalate"] += 1
            elif item.ground_truth == Verdict.EXAGGERATED:
                counts["exaggerated"] += 1
            else:
                counts["unlabeled"] += 1
        return counts


def load_dataset(
    domain: str = "finance",
    path: Optional[str] = None,
    **kwargs,
) -> DatasetAdapter:
    """Factory function: load a dataset adapter for the given domain.

    Args:
        domain: Domain name ("finance", "healthcare", etc.).
        path: Optional path to dataset files.
        **kwargs: Additional adapter-specific arguments.

    Returns:
        A DatasetAdapter instance for the specified domain.
    """
    if domain == "finance":
        from src.finance.finance_dataset_adapter import FinanceDatasetAdapter
        return FinanceDatasetAdapter(path=path, **kwargs)
    elif domain in ("healthcare", "health"):
        from src.healthcare.health_dataset_adapter import HealthDatasetAdapter
        return HealthDatasetAdapter(path=path, **kwargs)
    elif domain in ("political", "politics"):
        from src.political.political_dataset_adapter import PoliticalDatasetAdapter
        return PoliticalDatasetAdapter(path=path, **kwargs)
    else:
        raise ValueError(
            f"Unknown domain: {domain}. "
            f"Available: finance, healthcare, political"
        )
