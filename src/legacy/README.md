# Legacy Modules (Archived — Not Actively Maintained)

These modules were carried over from the original Verification Arbitrage
(finance flash-crash) project. They are **not actively used** in the new
General AI Information Verification Framework but are preserved for reference,
historical tracing, and potential refactoring into new framework modules.

## Module Inventory

| Module | Original Purpose | Future Reuse |
|--------|-----------------|--------------|
| `base_rate_analysis.py` | Bayesian PPV analysis | Core logic refactored into `src/base_rate.py` |
| `cot_parser.py` | CoT output parsing | Core logic refactored into `src/verifier_single_shot.py` |
| `cross_lingual.py` | Cross-lingual (Nikkei, DAX, Hang Seng) | Reference for Phase 9 cross-domain |
| `crypto_domain.py` | Cryptocurrency stress test | Reference for Phase 9 new domains |
| `edgar_rag_retriever.py` | SEC EDGAR filing RAG | Reference for Phase 2 finance adapter |
| `health_dataset.py` | Healthcare domain evaluation | Reference for Phase 9 healthcare adapter |
| `moa_agents.py` | MoA debate (Believer/Skeptic/Risk Officer) | Refactored into `src/verifier_moa.py` |
| `prompts.py` | Old finance-specific prompts | Superseded by `src/prompts.py` (domain-agnostic) |
| `rag_retriever.py` | Dual-source RAG (corpus + social stream) | Reference for Phase 6 RAG verifier |
| `red_team_generator.py` | Adversarial red-team generation | Reference for future adversarial testing |
| `xbrl_verifier.py` | XBRL financial statement verification | Reference for Phase 2 finance adapter |

## Intention

- Some modules will be **refactored** into new framework files as phases progress.
- Others serve as **documentation** of what worked (and what didn't) in the
  finance-specific prototype.
- No imports from legacy/ are used in the new framework code.
