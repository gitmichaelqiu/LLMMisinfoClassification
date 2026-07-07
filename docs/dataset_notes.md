# Dataset Notes

## Schema

All datasets in the framework conform to the `VerificationItem` schema
defined in `src/schemas.py`:

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | Unique identifier |
| `claim_text` | str | The claim or headline to verify |
| `context` | Optional[str] | Supporting context |
| `metadata` | dict | Domain, source, timestamp, etc. |
| `ground_truth` | Optional[Verdict] | Known label (None if unlabeled) |

## Dataset Adapter Pattern

Each domain implements `DatasetAdapter` (from `src/datasets.py`):
```
class MyDomainAdapter(DatasetAdapter):
    def load(self) -> List[VerificationItem]: ...
```

The adapter handles:
- File format parsing (CSV, JSON, etc.)
- Label mapping (raw labels → Verdict enum)
- Data source documentation

## Known Datasets

### Finance (Phase 2)

| Source | Format | Size | Balance | Status |
|--------|--------|------|---------|--------|
| Synthetic headlines | CSV | ~200 | ~50/50 | Active |
| Financial news dataset (Kaggle) | CSV | ~50K | Unknown | Legacy (in archive) |
| Historical hoaxes | JSON | 7 | Mixed | Legacy (in archive) |

The `FinanceDatasetAdapter` has a synthetic fallback that provides 10
signal items (5 FAKE, 5 REAL) when no dataset files are present on disk.

### Data Directory Structure

```
data/
├── raw/           # Raw external datasets (not committed if large)
├── processed/     # Cleaned/featurized data
└── synthetic/     # Generated synthetic events
```

Large datasets (Kaggle CSVs, model weights) are not stored in the
repository. See `data/raw/legacy_finance/README.md` if present for
download instructions.
