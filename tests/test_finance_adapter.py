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

    def test_synthetic_fallback_counts(self):
        """Without CSV files, fallback returns 10 items (5 real, 5 fake)."""
        adapter = FinanceDatasetAdapter()
        items = adapter.load()
        assert len(items) == 10
        real_count = sum(1 for i in items if i.ground_truth == Verdict.REAL)
        fake_count = sum(1 for i in items if i.ground_truth == Verdict.FAKE)
        assert real_count == 5
        assert fake_count == 5

    def test_synthetic_fallback_metadata(self):
        """Fallback items carry domain='finance' and source='synthetic_fallback'."""
        adapter = FinanceDatasetAdapter()
        items = adapter.load()
        for item in items:
            assert item.metadata["domain"] == "finance"
            assert item.metadata["source"] == "synthetic_fallback"

    def test_item_counts_synthetic(self):
        adapter = FinanceDatasetAdapter()
        counts = adapter.item_counts()
        assert counts["total"] == 10
        assert counts["real"] == 5
        assert counts["fake"] == 5
        assert counts["unlabeled"] == 0

    def test_train_test_split_default(self):
        adapter = FinanceDatasetAdapter()
        train, test = adapter.train_test_split(test_size=0.3)
        assert len(train) + len(test) == 10
        # 0.3 of 10 = 3 test, 7 train
        assert len(test) == 3
        assert len(train) == 7

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

    def test_parse_label_real(self):
        assert FinanceDatasetAdapter._parse_label("0") == Verdict.REAL
        assert FinanceDatasetAdapter._parse_label("REAL") == Verdict.REAL
        assert FinanceDatasetAdapter._parse_label("REAL/HOLD") == Verdict.REAL

    def test_parse_label_fake(self):
        assert FinanceDatasetAdapter._parse_label("1") == Verdict.FAKE
        assert FinanceDatasetAdapter._parse_label("FAKE") == Verdict.FAKE
        assert FinanceDatasetAdapter._parse_label("FAKE/INTERVENE") == Verdict.FAKE

    def test_parse_label_escalate(self):
        assert FinanceDatasetAdapter._parse_label("2") == Verdict.ESCALATE
        assert FinanceDatasetAdapter._parse_label("ESCALATE") == Verdict.ESCALATE

    def test_parse_label_exaggerated(self):
        assert FinanceDatasetAdapter._parse_label("3") == Verdict.EXAGGERATED

    def test_parse_label_unknown(self):
        assert FinanceDatasetAdapter._parse_label("") is None
        assert FinanceDatasetAdapter._parse_label("banana") is None

    def test_load_csv_skips_non_csv_files(self):
        """Files without .csv extension are ignored."""
        with tempfile.TemporaryDirectory() as tmp:
            readme = os.path.join(tmp, "README.md")
            with open(readme, "w") as f:
                f.write("# not a csv")
            adapter = FinanceDatasetAdapter(path=tmp)
            items = adapter.load()
        # No CSV found → fallback (10 items)
        assert len(items) == 10

    def test_load_csv_reads_small_file(self):
        """A tiny valid CSV in the raw/finance/ dir should be loaded."""
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "raw", "finance"))
            csv_path = os.path.join(tmp, "raw", "finance", "test.csv")
            with open(csv_path, "w") as f:
                f.write("headline,label\n")
                f.write("Some real news,0\n")
                f.write("Some fake news,1\n")
            adapter = FinanceDatasetAdapter(path=tmp)
            items = adapter.load()
        assert len(items) == 2
        assert items[0].ground_truth == Verdict.REAL
        assert items[1].ground_truth == Verdict.FAKE
        assert items[0].ground_truth == Verdict.REAL
        assert items[1].ground_truth == Verdict.FAKE

    def test_does_not_load_health_data(self):
        """Finance adapter must NOT load health_headlines.csv from data/raw/health/."""
        adapter = FinanceDatasetAdapter()
        items = adapter.load()
        for item in items:
            assert "health" not in item.metadata.get("source_file", "")
