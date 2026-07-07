"""Finance case-study subpackage.

This package contains the finance-specific implementations of framework
components: dataset adapter, metrics, and case-study orchestration.
"""

from src.finance.finance_dataset_adapter import FinanceDatasetAdapter
from src.finance.finance_metrics import FinanceMetrics

__all__ = ["FinanceDatasetAdapter", "FinanceMetrics"]
