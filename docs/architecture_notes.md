# Architecture Notes

## Overview

The General AI Information Verification Framework is a domain-agnostic
verification system with pluggable verifier architectures and hybrid
risk-based decision policies.

## Core Abstractions

```
VerificationItem → Verifier → VerificationResult → HybridPolicy → RiskAction
                       ↑
                VerifierConfig
```

### VerificationItem
Input claim with optional context, metadata, and ground truth.
Defined in `src/schemas.py`.

### Verifier (ABC)
Takes a `VerificationItem`, produces a `VerificationResult`.
Implemented as:
- `SingleShotVerifier` — one LLM call (Phase 3)
- `VotingVerifier` — N parallel LLM calls, majority vote (Phase 4)
- `MoAVerifier` — Believer/Skeptic/Risk Officer debate (Phase 5)
- `RAGVerifier` — retrieval-augmented LLM call (Phase 6)

### HybridPolicy
Takes a `VerificationResult` + base rate + cost matrix → `RiskAction`.
Implements cost-sensitive decision rule (Phase 7).

### DatasetAdapter (ABC)
Maps domain-specific raw data into `VerificationItem` list.
Implemented as `FinanceDatasetAdapter` (Phase 2).

## Key Design Decisions

### 1. Schema-first
All data structures are defined in `src/schemas.py` before any verifier
code. This ensures all verifiers produce compatible output.

### 2. Domain as first-class concept
Domain adapters are the unit of generalization. Each domain provides:
- Dataset (labeled claims)
- VerifierRole descriptions (for prompts)
- DomainContext for prompt generation

### 3. Mock mode for development
All verifiers work with MockClient when no API key is available,
enabling offline development and CI testing.

### 4. Structured experiment logging
Every run produces a timestamped JSON file with git commit, config,
metrics, and artifact paths. No manual record-keeping.

## Directory Layout

```
src/
├── schemas.py              # Core data structures
├── datasets.py             # DatasetAdapter ABC + factory
├── metrics.py              # Classification metrics
├── base_rate.py            # Bayesian PPV / cost thresholds
├── llm_clients.py          # LLM API wrappers (OpenAI, DeepSeek, Mock)
├── prompts.py              # Domain-agnostic prompt templates
├── verifier_single_shot.py # Single-shot verifier
├── verifier_voting.py      # Voting ensemble verifier
├── verifier_moa.py         # MoA debate verifier
├── verifier_rag.py         # RAG verifier
├── hybrid_policy.py        # Cost-sensitive policy
├── logging_utils.py        # Experiment logging
├── legacy/                 # Archived modules from original project
└── finance/                # Finance case-study implementation
```

## Future Considerations

- **Cross-domain adapter API**: Standardize what each domain must implement
  (Phase 9).
- **Streaming verifier**: For real-time claim streams with sliding windows.
- **Human-in-the-loop**: Formalize ESCALATE action handling.
- **Calibration**: Platt-scaling or temperature scaling for confidence.
