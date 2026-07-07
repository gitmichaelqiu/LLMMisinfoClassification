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
