# Research Log

## Overview

This log tracks research activities, literature review, architectural decisions,
and experimental findings throughout the General AI Information Verification
Framework development.

---

## 2026-07-07 — Repository Restructure

**Activity:** Cleaned main branch to establish the general verification framework.
Archived all old finance-specific code to `archive/verification-arbitrage-pre-general-framework`.

**Key decisions:**
- Finance is the first case-study domain, not the exclusive focus.
- All verifier architectures share a unified schema (`src/schemas.py`).
- Dataset adapters abstract domain-specific logic behind a common ABC.
- Legacy code moved to `src/legacy/` with README explaining provenance.

**References:**
- [Architecture Notes](architecture_notes.md)
- [Dataset Notes](dataset_notes.md)

---

## 2026-07-07 — Phase 0: Repository Sanity Check & Reproducibility Setup

**Activity:** Established project infrastructure for the new framework.

**Deliverables:**
- Pinned `requirements.txt` with exact versions (numpy, openai, scikit-learn, etc.)
- `Makefile` with install/test/clean targets
- `pyproject.toml` with project metadata and pytest config
- `.env.example` with API key template
- `tests/conftest.py` with shared fixtures
- `tests/test_schemas.py` — 13 unit tests for schemas module

**Verification:**
- `pip install -r requirements.txt` and `pip install -e .` both pass
- `python -c "import src"` succeeds
- `python -m pytest tests/ -v` — 13/13 passed

**Key decisions:**
- `IntEnum` used for Verdict (JSON-serializable, comparable to ints)
- Removed heavy unused deps (transformers, hftbacktest, vectorbt)
- sentence-transformers kept optional (needed only for dense RAG retriever)

---

## Template for Future Entries

```
## YYYY-MM-DD — <Title>

**Activity:** <what was done>

**Key decisions:**
- <decision 1>
- <decision 2>

**References:**
- [Architecture Notes](architecture_notes.md)
- [Finance Results](finance_results_log.md)
```
