"""Tests for src/finance/finance_dataset_adapter.py."""
from __future__ import annotations

import os
import tempfile

import pytest

from src.datasets import DatasetAdapter, load_dataset
from src.finance.finance_dataset_adapter import FinanceDatasetAdapter
from src.schemas import Verdict


class TestFinanceDatasetAdapter:
    """Unit tests for FinanceDatasetAdapter."""

    def test_csv_loaded(self):
        """With CSV files in data/raw/finance/, loads them."""
        adapter = FinanceDatasetAdapter()
        items = adapter.load()
        assert len(items) > 1000  # the real dataset has ~45k rows
        assert any(it.metadata.get("source") != "synthetic_fallback" for it in items[:10])

    def test_items_have_valid_labels(self):
        """All loaded items have REAL or FAKE ground truth."""
        adapter = FinanceDatasetAdapter()
        items = adapter.load()
        for item in items:
            assert item.ground_truth in (Verdict.REAL, Verdict.FAKE)

    def test_item_counts(self):
        adapter = FinanceDatasetAdapter()
        counts = adapter.item_counts()
        assert counts["total"] > 1000
        assert counts["real"] > 0
        assert counts["fake"] > 0
        assert counts["unlabeled"] == 0

    def test_train_test_split_default(self):
        adapter = FinanceDatasetAdapter()
        counts = adapter.item_counts()
        train, test = adapter.train_test_split(test_size=0.3)
        assert len(train) + len(test) == counts["total"]

    def test_train_test_split_seed_reproducible(self):
        """Same seed produces the same ground_truth ordering."""
        adapter = FinanceDatasetAdapter()
        t1, s1 = adapter.train_test_split(test_size=0.5, seed=42)
        t2, s2 = adapter.train_test_split(test_size=0.5, seed=42)
        assert [i.ground_truth for i in s1] == [i.ground_truth for i in s2]
        assert [i.ground_truth for i in t1] == [i.ground_truth for i in t2]

    def test_is_dataset_adapter(self):
        adapter = FinanceDatasetAdapter()
        assert isinstance(adapter, DatasetAdapter)

    def test_load_dataset_factory(self):
        adapter = load_dataset("finance")
        assert isinstance(adapter, FinanceDatasetAdapter)

    def test_load_dataset_unknown_domain(self):
        with pytest.raises(ValueError, match="Unknown domain"):
            load_dataset("nonexistent")

    def test_synthetic_fallback_no_csv_dir(self):
        """When data directory doesn't exist, falls back to synthetic items."""
        with tempfile.TemporaryDirectory() as tmp:
            adapter = FinanceDatasetAdapter(path=tmp)
            items = adapter.load()
        assert len(items) == 10
        real = sum(1 for i in items if i.ground_truth == Verdict.REAL)
        fake = sum(1 for i in items if i.ground_truth == Verdict.FAKE)
        assert real == 5
        assert fake == 5
        for item in items:
            assert item.metadata["domain"] == "finance"
            assert item.metadata["source"] == "synthetic_fallback"

    def test_synthetic_fallback_empty_csv_dir(self):
        """When data directory exists but has no CSVs, falls back to synthetic."""
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "raw", "finance"), exist_ok=True)
            os.makedirs(os.path.join(tmp, "synthetic"), exist_ok=True)
            adapter = FinanceDatasetAdapter(path=tmp)
            items = adapter.load()
        assert len(items) == 10
