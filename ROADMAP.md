# General AI Information Verification Framework — Roadmap

A domain-agnostic framework for verifying AI-generated or human-authored information claims using configurable verifier architectures (single-shot, voting, MoA debate, RAG-augmented) with hybrid risk-based decision policies.

Finance is the first case-study domain (Phase 2). Cross-domain readiness is validated in Phase 9.

```mermaid
graph LR
    P0[Phase 0: Reproducibility] --> P1[Phase 1: Unified Schema & Metrics]
    P1 --> P2[Phase 2: Finance Dataset Adapter]
    P2 --> P3[Phase 3: Single-Shot Verifier]
    P3 --> P4[Phase 4: Voting Verifier]
    P4 --> P5[Phase 5: MoA/Debate Verifier]
    P5 --> P6[Phase 6: RAG Verifier]
    P6 --> P7[Phase 7: Hybrid Policy]
    P7 --> P8[Phase 8: Sensitivity Analysis]
    P8 --> P9[Phase 9: Cross-Domain Readiness]
```

---

## Phase 0: Repository Sanity Check & Reproducibility Setup

**Objective:** Verify that the repository is in a known-good state — dependencies install cleanly, legacy modules are quarantined, docs scaffold exists, and a single command can run the first end-to-end smoke test.

### Files to create/edit
- `requirements.txt` — audit existing, pin exact versions, add `pytest`, `pytest-cov`
- `Makefile` — common targets: `install`, `test`, `lint`, `clean`
- `pyproject.toml` — project metadata, tool configs (pytest, ruff/black)
- `tests/conftest.py` — shared fixtures
- `tests/test_schemas.py` — verify schema validation
- `docs/research_log.md` — project research log
- `docs/architecture_notes.md` — architectural decisions
- `.env.example` — template for secrets (API keys)

### Implementation tasks
1. Freeze `requirements.txt` with pinned versions
2. Create `Makefile` with `install`, `test`, `clean` targets
3. Create `pyproject.toml` with project metadata
4. Verify `pip install -e .` works
5. Add `.env.example` with `OPENAI_API_KEY` and `DEEPSEEK_API_KEY`
6. Create `tests/conftest.py` with `pytest` fixtures

### Experiments to run
- `pip install -r requirements.txt && python -c "import src; print('OK')"`
- `python -m pytest tests/ -v`

### Metrics to report
- Test pass/fail count
- Dependency installation time

### Expected output artifacts
- `requirements.txt` (pinned)
- `Makefile`
- `pyproject.toml`
- `tests/conftest.py`
- `docs/research_log.md`
- `docs/architecture_notes.md`

### Copy back to ChatGPT
```
Phase 0 complete. Repository is reproducible. Dependencies pinned. Test suite passes.
Makefile targets: install, test, lint, clean.
```

---

## Phase 1: Unified Verification Schema & Metrics

**Objective:** Define the canonical data structures (schemas, verdicts, evidence) and metrics computation (precision, recall, F1, FPR, FNR, PPV, NPV, calibration error) that all verifier architectures will share.

### Files to create/edit
- `src/__init__.py` — package exports
- `src/schemas.py` — `VerificationItem`, `Verdict`, `VerificationResult`, `VerifierConfig`
- `src/metrics.py` — `compute_confusion_matrix()`, `classification_metrics()`, `calibration_metrics()`
- `src/logging_utils.py` — structured result logging, ledger helpers
- `tests/test_schemas.py` — schema validation tests
- `tests/test_metrics.py` — metrics correctness tests
- `docs/dataset_notes.md` — document dataset schema

### Implementation tasks
1. Define `VerificationItem` dataclass (id, claim_text, context, metadata, ground_truth)
2. Define `Verdict` enum (REAL, FAKE, ESCALATE, EXAGGERATED) with confidence
3. Define `VerificationResult` dataclass (item_id, verdict, confidence, latency, evidence)
4. Define `VerifierConfig` dataclass (model, temperature, max_tokens, prompt_template)
5. Implement `compute_confusion_matrix()` with integer counts (TP, FP, TN, FN)
6. Implement `classification_metrics()` returning precision, recall, F1, FPR, FNR
7. Implement `calibration_metrics()` with ECE computation
8. Implement `compute_ppv()` with Bayesian base-rate sweep
9. Create `logging_utils.py` with `ExperimentLogger` that saves structured JSON results
10. Write unit tests for all schema validation and metric computations

### Experiments to run
- `python -c "from src.schemas import Verdict; print(Verdict.FAKE.value)"`
- `python -m pytest tests/test_schemas.py tests/test_metrics.py -v`

### Metrics to report
- Unit test coverage for schemas and metrics modules

### Expected output artifacts
- `src/schemas.py`
- `src/metrics.py`
- `src/logging_utils.py`
- `tests/test_schemas.py`
- `tests/test_metrics.py`
- `docs/dataset_notes.md`

### Copy back to ChatGPT
```
Phase 1 complete. Unified schemas defined (VerificationItem, Verdict, VerificationResult, VerifierConfig).
Metrics module implements confusion matrix, precision/recall/F1/FPR/FNR, ECE calibration, and Bayesian PPV.
All verified by unit tests.
```

---

## Phase 2: Finance Case-Study Dataset Adapter

**Objective:** Adapt the legacy finance news dataset into the unified schema. Build a dataset adapter that produces `VerificationItem` objects from both synthetic and real financial news sources. This adapter serves as the template for future domain adapters.

### Files to create/edit
- `src/datasets.py` — base `DatasetAdapter` ABC, `load_dataset()`, train/test split
- `src/finance/__init__.py` — finance subpackage
- `src/finance/finance_dataset_adapter.py` — `FinanceDatasetAdapter( DatasetAdapter)`
- `src/finance/finance_metrics.py` — finance-specific metric wrappers (crossover thresholds, P&L estimation)
- `src/finance/finance_case_study.py` — case-study orchestration script
- `tests/test_finance_adapter.py` — adapter tests
- `docs/finance_results_log.md` — structured experiment log (template)

### Implementation tasks
1. Implement `DatasetAdapter` ABC with `load()`, `train_test_split()`, `item_counts()` methods
2. Implement `FinanceDatasetAdapter` that loads synthetic + real financial news
3. Map legacy labels (FAKE/REAL) to `Verdict` enum
4. Implement `load_dataset()` factory function
5. Create `finance_case_study.py` CLI that loads adapter, runs through a verifier, logs results
6. Structure `finance_results_log.md` with experiment template (see docs section)

### Experiments to run
- `python -c "from src.finance.finance_dataset_adapter import FinanceDatasetAdapter; d = FinanceDatasetAdapter(); print(d.item_counts())"`
- `python -m pytest tests/test_finance_adapter.py -v`

### Metrics to report
- Dataset size, class balance, train/test split sizes

### Expected output artifacts
- `src/datasets.py`
- `src/finance/finance_dataset_adapter.py`
- `src/finance/finance_metrics.py`
- `src/finance/finance_case_study.py`
- `tests/test_finance_adapter.py`
- `docs/finance_results_log.md`

### Copy back to ChatGPT
```
Phase 2 complete. FinanceDatasetAdapter maps legacy financial news data into unified VerificationItem schema.
Adapter produces N train / M test items with documented class balance.
Template established for future domain adapters.
```

---

## Phase 3: Single-Shot LLM Verifier

**Objective:** Implement the simplest verifier — a single LLM call with a system prompt. Establish the baseline LLM verification performance against the finance dataset. This is the "fast path" verifier.

### Files to create/edit
- `src/llm_clients.py` — `LLMClient` ABC, `OpenAIClient`, `DeepSeekClient`
- `src/verifier_single_shot.py` — `SingleShotVerifier` with CoT parsing
- `src/prompts.py` — update with domain-agnostic prompt templates
- `src/base_rate.py` — Bayesian PPV/NPV, base-rate analysis utilities
- `tests/test_llm_clients.py` — client tests (mock mode)
- `tests/test_verifier_single_shot.py` — end-to-end verifier tests

### Implementation tasks
1. Implement `LLMClient` ABC with `generate()` and `generate_batch()` methods
2. Implement `OpenAIClient` and `DeepSeekClient` wrappers
3. Implement mock mode: fallback to heuristic classifier when no API key
4. Implement `SingleShotVerifier` with configurable system prompt
5. Implement CoT parser that extracts verdict + confidence + evidence from LLM output
6. Update `src/prompts.py` with domain-agnostic verification prompts
7. Implement `base_rate.py` with Bayesian update and PPV computation
8. Run end-to-end verification on finance dataset

### Experiments to run
- `python -m src.verifier_single_shot --model gpt-4o-mini --test-size 50`
- `python -m src.verifier_single_shot --model deepseek-chat --test-size 50`
- `python -m pytest tests/test_verifier_single_shot.py tests/test_llm_clients.py -v`

### Metrics to report
- Precision, Recall, F1, FPR, FNR (per model)
- Mean latency (per model)
- Confusion matrix (raw integer counts)
- PPV at tested base rates
- Calibration ECE

### Expected output artifacts
- `src/llm_clients.py`
- `src/verifier_single_shot.py`
- `src/prompts.py` (updated)
- `src/base_rate.py`
- `tests/test_llm_clients.py`
- `tests/test_verifier_single_shot.py`
- `results/phase03_single_shot_metrics.json`
- Log entry in `docs/finance_results_log.md`

### Copy back to ChatGPT
```
Phase 3 complete. Single-shot verifier implemented with OpenAIClient and DeepSeekClient.
Baseline metrics: Prec=XX, Rec=XX, F1=XX, FPR=XX, FNR=XX at N=XX.
ECE=XX, PPV at 5% base rate = XX%.
```

---

## Phase 4: Voting Verifier

**Objective:** Implement a voting ensemble verifier that calls the LLM N times (default 5) with varied prompts or temperatures and aggregates verdicts by majority vote. Compare precision/recall trade-off against single-shot baseline.

### Files to create/edit
- `src/verifier_voting.py` — `VotingVerifier` with configurable N, aggregation strategy
- `tests/test_verifier_voting.py` — voting verifier tests

### Implementation tasks
1. Implement `VotingVerifier` that spawns N parallel LLM calls
2. Support multiple aggregation strategies: majority, weighted (by confidence), unanimous
3. Support prompt variation (different framing per voter) or temperature variation
4. Implement tie-breaking rules
5. Run comparison against single-shot baseline on finance dataset
6. Analyze latency vs. accuracy trade-off

### Experiments to run
- `python -m src.verifier_voting --n-voters 3 --test-size 50`
- `python -m src.verifier_voting --n-voters 5 --test-size 50`
- `python -m src.verifier_voting --n-voters 7 --test-size 50`
- Compare results with Phase 3 single-shot

### Metrics to report
- Precision, Recall, F1, FPR, FNR (per N)
- Mean latency per N
- Confusion matrix per N
- Latency-precision Pareto frontier
- Variance across voters (agreement rate)

### Expected output artifacts
- `src/verifier_voting.py`
- `tests/test_verifier_voting.py`
- `results/phase04_voting_metrics.json`
- Log entry in `docs/finance_results_log.md`

### Copy back to ChatGPT
```
Phase 4 complete. Voting verifier implemented with N=3,5,7.
Best F1=XX at N=X with latency=XXs.
Pareto frontier shows diminishing returns after N=X.
```

---

## Phase 5: MoA/Debate Verifier & Degeneracy Tests

**Objective:** Implement a Mixture-of-Agents (MoA) debate verifier with Believer/Skeptic/Risk Officer roles. Conduct systematic degeneracy tests — always-FAKE bias, always-REAL bias, base-rate tracking, precision-above-base-rate checks — to validate that the debate architecture produces genuine reasoning rather than superficial patterns.

### Files to create/edit
- `src/verifier_moa.py` — `MoAVerifier` with role-based agents
- `tests/test_verifier_moa.py` — degeneracy tests, base-rate checks

### Implementation tasks
1. Implement `MoAVerifier` with configurable roles
2. Implement Believer (pro-authenticity), Skeptic (pro-hoax), Risk Officer (synthesis) prompts
3. Implement concurrent execution (Believer + Skeptic in parallel, Risk Officer after)
4. Implement degeneracy test suite:
   - Always-FAKE check: precision vs base rate at 0% bot intensity
   - Always-REAL check: recall on known-fake items
   - Base-rate tracking: does precision significantly exceed P(FAKE)?
5. Run comparison against single-shot and voting baselines

### Experiments to run
- `python -m src.verifier_moa --test-size 50`
- Degeneracy tests: `python -m pytest tests/test_verifier_moa.py -v`

### Metrics to report
- Precision, Recall, F1, FPR, FNR
- Latency breakdown (Believer, Skeptic, synthesis)
- Degeneracy test results (pass/fail per test)
- Precision vs. base rate comparison

### Expected output artifacts
- `src/verifier_moa.py`
- `tests/test_verifier_moa.py`
- `results/phase05_moa_metrics.json`
- Log entry in `docs/finance_results_log.md`

### Copy back to ChatGPT
```
Phase 5 complete. MoA verifier with Believer/Skeptic/Risk Officer roles.
Degeneracy tests: always-FAKE=[PASS/FAIL], always-REAL=[PASS/FAIL],
precision > P(FAKE)=[PASS/FAIL]. MoA F1=XX vs single-shot F1=XX.
```

---

## Phase 6: RAG-Based Verifier

**Objective:** Implement a retrieval-augmented verifier that supplements the LLM call with relevant context from a knowledge corpus. This grounds verification in external evidence rather than relying solely on parametric knowledge.

### Files to create/edit
- `src/verifier_rag.py` — `RAGVerifier` with corpus retrieval + LLM synthesis
- `tests/test_verifier_rag.py` — RAG verifier tests

### Implementation tasks
1. Implement `RAGVerifier` that retrieves relevant context before LLM call
2. Build light-weight corpus index (sentence embeddings + FAISS or simple TF-IDF)
3. Implement context window management (truncation, relevance scoring)
4. Support configurable retriever: dense (embedding) or sparse (TF-IDF)
5. Run comparison: RAG vs. non-RAG single-shot
6. Analyze context quality metrics (retrieval precision, context utilization)

### Experiments to run
- `python -m src.verifier_rag --retriever dense --test-size 50`
- `python -m src.verifier_rag --retriever sparse --test-size 50`
- Compare with Phase 3 (no-RAG baseline)

### Metrics to report
- Precision, Recall, F1, FPR, FNR (dense vs sparse)
- Retrieval latency
- Context relevance score
- Improvement over no-RAG baseline

### Expected output artifacts
- `src/verifier_rag.py`
- `tests/test_verifier_rag.py`
- `results/phase06_rag_metrics.json`
- Log entry in `docs/finance_results_log.md`

### Copy back to ChatGPT
```
Phase 6 complete. RAG verifier with dense and sparse retrievers.
F1 improvement over no-RAG baseline: +XX (dense), +XX (sparse).
Retrieval latency: XXs (dense), XXs (sparse).
```

---

## Phase 7: Hybrid Policy Based on Misinformation Base Rate & Risk

**Objective:** Implement a hybrid policy that selects or weights verifier outputs based on context — the estimated base rate of misinformation, the cost asymmetry of false positives vs. false negatives, and the verifier's confidence calibration. This replaces the old "three-tier abstention" with a mathematically principled decision rule.

### Files to create/edit
- `src/hybrid_policy.py` — `HybridPolicy` decision rule
- `src/base_rate.py` — extend with cost-sensitive decision thresholds
- `tests/test_hybrid_policy.py` — policy tests

### Implementation tasks
1. Formalize the decision problem: choose action (HOLD / HEDGE / REVERSE / ESCALATE) to minimize expected cost
2. Implement `HybridPolicy` that takes verifier output + base rate estimate + cost matrix → action
3. Implement cost-sensitive threshold computation (asymmetric loss)
4. Implement base rate estimation from stream statistics or prior
5. Implement meta-policy: which verifier to deploy based on estimated base rate
6. Run end-to-end: dataset → verifier → policy → action

### Experiments to run
- `python -m src.hybrid_policy --test-size 50 --cost-fp 1.0 --cost-fn 10.0`
- Sweep cost asymmetry ratios (1:1, 1:5, 1:10, 1:25, 1:100)

### Metrics to report
- Expected cost per decision (under various cost ratios)
- Action distribution (% HOLD vs % HEDGE vs % REVERSE vs % ESCALATE)
- Cost savings vs. "always reverse" and "always hold" baselines
- Decision threshold values at each cost ratio

### Expected output artifacts
- `src/hybrid_policy.py`
- `src/base_rate.py` (extended)
- `tests/test_hybrid_policy.py`
- `results/phase07_hybrid_policy_metrics.json`
- Log entry in `docs/finance_results_log.md`

### Copy back to ChatGPT
```
Phase 7 complete. Hybrid policy minimizes expected cost under asymmetric loss.
At FP:FN=1:10, decision threshold = XX%, expected cost = $XX vs
always-reverse cost = $XX. Action distribution: HOLD=XX%, HEDGE=XX%,
REVERSE=XX%, ESCALATE=XX%.
```

---

## Phase 8: Sensitivity Analysis

**Objective:** Systematically vary all key parameters and measure the impact on verification performance. Identify which parameters drive results and establish safe operating ranges. This replaces the old domain-specific microstructure sensitivity with a framework-agnostic analysis.

### Files to create/edit
- `src/sensitivity_analysis.py` — parameter sweeps, sensitivity metrics
- `tests/test_sensitivity.py` — sensitivity test harness

### Implementation tasks
1. Define parameter grid: model (gpt-4o-mini, gpt-4o, deepseek-chat, claude-sonnet), temperature (0.0, 0.3, 0.7), prompt template (terse, detailed, CoT), verifier architecture (single, voting, MoA, RAG), dataset balance (50/50, 75/25, 90/10)
2. Implement sweep harness that runs each config and collects metrics
3. Implement sensitivity ranking (which parameters have largest effect on F1?)
4. Implement interaction detection (do certain parameters interact?)
5. Generate summary table of best/worst configurations

### Experiments to run
- Full grid search (may be large — support `--quick` mode with subset)
- `python -m src.sensitivity_analysis --quick --test-size 30`

### Metrics to report
- Sensitivity ranking (F1 variance explained per parameter)
- Best configuration: (model, temperature, prompt, verifier)
- Worst configuration
- Interaction effects between parameters

### Expected output artifacts
- `src/sensitivity_analysis.py`
- `tests/test_sensitivity.py`
- `results/phase08_sensitivity_metrics.json`
- Log entry in `docs/finance_results_log.md`

### Copy back to ChatGPT
```
Phase 8 complete. Sensitivity analysis across 4 parameters.
Top driver: verifier architecture (XX% variance explained).
Best config: {model}+{verifier}+{prompt} with F1=XX.
Safe operating range: temperature [0.0, 0.3], base rate [2%, 50%].
```

---

## Phase 9: Cross-Domain Readiness Check

**Objective:** Validate that the framework generalizes beyond finance by testing on at least two additional domains (e.g., healthcare claims, political statements, or product reviews). Document what domain-specific adaptations were needed and propose a domain-adapter API standard.

### Files to create/edit
- `src/domain_adapter.py` — `DomainAdapter` ABC and registry
- Finance adapter moves to `src/finance/finance_dataset_adapter.py` (update if needed)
- New: `src/healthcare/` or adapt existing health dataset
- New: `src/political/` or adapt third-party political fact-check dataset
- `tests/test_cross_domain.py` — cross-domain evaluation tests

### Implementation tasks
1. Formalize `DomainAdapter` ABC: what methods must every domain implement?
2. If healthcare dataset exists, adapt it; otherwise source a public dataset
3. Source or create a political/statements dataset
4. Run single-shot verifier on both new domains
5. Document domain-specific prompt adjustments needed
6. Propose domain-adapter API standard in `docs/architecture_notes.md`

### Experiments to run
- `python -m src.verifier_single_shot --domain healthcare --test-size 50`
- `python -m src.verifier_single_shot --domain political --test-size 50`
- Compare cross-domain F1 variance

### Metrics to report
- F1 per domain
- F1 variance across domains
- Domain-specific prompt sensitivity
- Dataset quality metrics (label consistency, ambiguity rate)

### Expected output artifacts
- `src/domain_adapter.py`
- Healthcare adapter
- Political adapter
- `tests/test_cross_domain.py`
- `results/phase09_cross_domain_metrics.json`
- Log entry in `docs/finance_results_log.md`

### Copy back to ChatGPT
```
Phase 9 complete. Framework validated across 3 domains: finance (F1=XX),
healthcare (F1=XX), political (F1=XX). Cross-domain F1 variance = XX%.
Domain adapter API standardized. Framework is domain-general.
```

---

# Post-Phase-9 Research Roadmap

**Context:** After completing the general verification framework (Phases 0–9), we conducted a real-API pilot evaluation (240 DeepSeek calls, ~$0.036) comparing Single-Shot, Voting N=3, MoA, and RAG on finance, healthcare, and political domains (N=10 each). The results revealed a critical framing gap: all four architectures are individually well-studied in prior work (FEVER, Self-Consistency, Mixture-of-Agents, Self-RAG). A paper that simply compares architectures would not be novel.

**Pivot:** The likely core contribution should be **cost-aware, base-rate-aware, latency-aware verifier routing** — selecting which architecture to deploy based on the operating context (base rate of misinformation, cost asymmetry of errors, available latency budget). The phases below reflect this pivot.

```mermaid
graph LR
    P9[Phase 9: Cross-Domain] --> P10[Phase 10: Lit. Map & Bib.]
    P10 --> P11[Phase 11: Base-Rate-Stratified Analysis]
    P11 --> P12[Phase 12: Routing Policy Implementation]
    P12 --> P13[Phase 13: Small Statistical Scaling]
    P12 -.-> P14[Phase 14: RAG Redesign (Optional)]
    P13 --> P15[Phase 15: Final Consolidation]
    P14 --> P15
```

---

## Phase 10: Literature-Grounded Evidence Map & Bibliography

**Objective:** Build a structured, annotated bibliography organized by the six research dimensions this work touches. This provides the literature backing for the routing-pivot thesis and prevents the paper from claiming novelty in well-studied areas.

**Status:** Draft completed in `docs/meeting_evidence_map.md` (see Literature Context section and Bibliography). This phase formalizes it.

### Organization dimensions
1. **Benchmarks & datasets** — FEVER, SciFact, AVeriTeC, custom financial/political sets
2. **Architecture baselines** — single-shot (zero/few-shot CoT), self-consistency/voting (Wang et al., 2022), debate/MoA (Irving et al., 2018; Wang et al., 2024; Du et al., 2023), RAG (Lewis et al., 2020; Self-RAG: Asai et al., 2023)
3. **Calibration & abstention** — confidence calibration (Hendrycks & Gimpel, 2016), selective classification (Varshney et al., 2022)
4. **Cost-aware LLM routing** — FrugalGPT (Chen et al., 2023), model cascade routing
5. **Class imbalance & PPV** — Provost & Fawcett (1997), Chawla et al. (2002), Bayesian PPV in low-prevalence settings
6. **Finance-specific verification** — flash-crash hoax literature, market microstructure impact of misinformation

### Files to create/edit
- `docs/literature_map.md` — structured bibliography with annotations
- `docs/meeting_evidence_map.md` — already has the Literature Context section; expand with per-reference annotations

### Implementation tasks
1. Classify each citation into the six dimensions above
2. Annotate: what each reference establishes + what gap remains for this work
3. Map existing empirical findings to gaps in the literature

### Expected output artifacts
- `docs/literature_map.md` (structured, annotated bibliography)
- `docs/meeting_evidence_map.md` (updated with full annotations)

### Copy back to Professor
```
Phase 10 complete. Literature organized into 6 dimensions.
Gap confirmed: prior work studies architectures in isolation,
not cost/base-rate/latency-aware routing between them.
```

---

## Phase 11: Base-Rate-Stratified Analysis Using Existing Real API Outputs

**Objective:** Determine the optimal verifier architecture at each base-rate regime using already-collected real API outputs. No new API calls. This is the **most important next experiment** — it answers whether the routing hypothesis has empirical support.

**Rationale for doing this first:** Prior to scaling to larger N, we need to know whether the crossover effect (different architectures optimal at different base rates) exists in our data. If it does not — if one architecture dominates universally — then the routing contribution collapses and we should report the simpler finding.

### Data sources (all exist, no new API calls)
- `output/repair_rerun/report.json` — 240 real DeepSeek calls, 4 architectures × 3 domains × 10 items
- `output/real_eval/report.json` — original real API run (same items, different seeds)
- Raw outputs in `output/repair_rerun/raw_outputs/` and `output/real_eval/raw_outputs/`

### Base-rate regimes to test
| Segment | P(Fake) | Rationale |
|---------|---------|-----------|
| Very low | 0.1% | Real-world baseline; PPV collapse zone |
| Low | 1% | Crossover boundary for most classifiers |
| Medium | 5% | Near Phase-22 crossover estimate (5.8%) |
| Medium-high | 10% | Upper bound of "typical" crisis base rate |
| High | 25% | Active misinformation campaign |
| Balanced | 50% | Standard ML evaluation regime |

### Latency budgets to test
| Budget | Rationale |
|--------|-----------|
| <5s | HFT-compatible (Single-Shot, RAG) |
| <15s | Swing-trading compatible (Voting N=3) |
| <30s | Fundamental verification (MoA) |

### What to compute per (base rate, latency budget, architecture)
- **PPV / NPV** — Bayesian, using measured TPR and FPR from real API data
- **Expected per-item cost** — using asymmetric loss (sweep cost_fp:cost_fn = 1:1, 1:10, 1:100)
- **Optimal action** — hold, hedge, reverse, or abstain given the expected cost
- **Architecture ranking** — which verifier minimizes expected cost at this base rate

### Implementation approach
- Downsample existing predictions to simulate each base rate (no new labels needed)
- Apply Bayes' rule: PPV = (sens × prev) / (sens × prev + (1 − spec) × (1 − prev))
- Sweep across all 6 base rates × 3 latency budgets × 3 cost ratios = 54 configurations

### Expected output artifacts
- `output/phase11_routing_analysis.json` — structured results
- Crossover tables: optimal architecture × base rate for each cost ratio
- Decision boundaries: where routing policy switches architectures

### Copy back to Professor
```
Phase 11 complete. Crossover analysis on existing real API outputs.
At base rates below X%, Single-Shot is optimal; above X%, Voting dominates.
At crisis base rates (>25%), MoA's calibration advantage justifies its latency.
Routing hypothesis has [empirical support / requires revision].
```

---

## Phase 12: Routing Policy Implementation

**Objective:** Implement a context-aware routing policy that selects verifier architecture and action based on operating conditions. This is the likely core contribution of the work.

**Policy signature:**
```
input:  base_rate_estimate, cost_ratio, latency_budget,
        evidence_availability, verifier_confidence
output: selected_verifier (single_shot | voting | moa | rag),
        action (hold | hedge | reverse | abstain | escalate),
        confidence threshold
```

### Policy design
1. **Base-rate gating**: If P(Fake) < threshold, use Single-Shot + abstention (avoid FP costs)
2. **Latency gating**: If budget < 5s, exclude Voting and MoA
3. **Cost-ratio gating**: If cost_fn >> cost_fp, prefer high-recall verifier (Voting/MoA)
4. **Evidence gating**: If RAG evidence is available and relevant, use RAG
5. **Confidence gating**: If confidence < threshold, escalate to next-tier verifier
6. **Abstention**: If no verifier meets minimum expected-cost criteria, output ESCALATE

### Implementation tasks
1. Implement `RoutingPolicy` class with the above decision logic
2. Implement policy evaluation: compare routing vs. each fixed architecture
3. Sweep policy thresholds to find optimal configuration
4. Document policy decision boundaries (see Phase 11 output)

### Files to create/edit
- `src/routing_policy.py` — `RoutingPolicy` class
- `src/base_rate.py` — extend with streaming base-rate estimation (already done)
- `tests/test_routing_policy.py` — policy tests

### Experiments to run
- `python -m src.routing_policy --mode evaluate` — compare routing vs. fixed architectures
- `python -m src.routing_policy --mode threshold_sweep` — find optimal decision boundaries

### Metrics to report
- Expected cost: routing policy vs. best fixed architecture
- Action distribution: how often does the policy select each verifier?
- Latency compliance: % of decisions within budget
- Cost savings: routing vs. "always use Voting" baseline

### Expected output artifacts
- `src/routing_policy.py`
- `tests/test_routing_policy.py`
- `output/phase12_routing_metrics.json`

### Copy back to Professor
```
Phase 12 complete. Routing policy selects verifier based on base rate,
cost ratio, and latency budget. At low base rates, Single-Shot + abstention
is optimal. At medium base rates, Voting is preferred. Routing achieves
XX% cost savings vs. best fixed architecture.
```

---

## Phase 13: Small Statistical Scaling

**Objective:** Scale the most relevant architectures to larger sample sizes to establish statistical confidence. This phase runs **only after** Phase 11/12 confirms which architectures matter for the routing policy.

**Why not scale first:** Scaling all architectures to N=50 before knowing the routing thesis would waste API calls on architectures that may not be relevant (e.g., RAG if it does not improve routing outcomes). Phase 11 tells us which ones to scale.

### Likely design (subject to Phase 11/12 results)
- Scale **Single-Shot** (N=50/domain) × 3 domains = 150 API calls (~$0.023)
- Scale **Voting N=3** (N=50/domain) × 3 domains = 150 API calls (~$0.023)
- Possibly **MoA** only if Phase 11/12 shows a crossover regime where MoA is optimal
- Possibly **exclude political** if Phase 11/12 confirms it is unreliable

### Metrics to report
- 95% confidence intervals on all metrics (F1, PPV, ECE, latency)
- Statistical significance tests: does Voting beat Single-Shot at N=50?
- Updated routing decision boundaries with confidence intervals

### Expected output artifacts
- Updated raw outputs in `output/phase13_scaling/`
- Updated metrics with confidence intervals

### Copy back to Professor
```
Phase 13 complete. [Architecture(s)] scaled to N=50/domain.
F1: Single-Shot = XX ± XX [95% CI], Voting N=3 = XX ± XX [95% CI].
Routing policy decision boundaries stable within [range].
```

---

## Phase 14: RAG Redesign (Optional)

**Objective:** Redesign the RAG verifier with curated, source-aware evidence retrieval. **Only if the professor determines RAG is important for the paper's contribution** — otherwise skip or defer.

**Background:** Our Phase 6/repair RAG uses generic TF-IDF retrieval from an all-REAL news corpus. This produced F1=0.413 (worst architecture) with 80% ESCALATE on finance and political. Prior work (FEVER, SciFact, Self-RAG) uses curated evidence with source reliability labels — which we did not attempt.

### Design alternatives to test
1. **Curated contradiction corpus** — include known-fake examples with contradiction signals
2. **Source reliability labels** — annotate each retrieved document with source trustworthiness
3. **Temporal validity** — retrieve only documents within relevant time windows
4. **Retrieve-on-demand (Self-RAG style)** — retrieve only when the verifier identifies an evidence gap

### Minimum viable test
- Build a balanced corpus (50% real articles, 50% known-fake headlines with contextual debunking)
- Re-run RAG evaluator (same prompt, better retrieval corpus)
- Compare: does ESCALATE rate drop from 80% to <30%?

### Expected output artifacts
- Updated RAG corpus (curated)
- `output/phase14_rag_redesign.json` — before/after comparison

### Copy back to Professor
```
Phase 14 [optional] complete. Curated evidence corpus reduces RAG ESCALATE
rate from 80% to XX%. RAG F1 improves from 0.413 to XX. Design decision:
[include RAG in paper / defer to future work].
```

---

## Phase 15: Final Experiment Consolidation

**Objective:** Consolidate all experiment outputs into final tables and figures. Generate the artifacts needed for the paper's empirical sections.

**Scope:** Tables and figures only. No paper writing.

### Deliverables
1. **Cross-domain metrics table** — F1, PPV, ECE, latency per architecture per domain (with CIs from Phase 13)
2. **Routing policy decision table** — optimal architecture × base rate × cost ratio × latency budget
3. **PPV curve chart** — PPV vs. P(Fake) for each architecture with crossover thresholds annotated
4. **Latency-Precision Pareto frontier** — mean latency vs. F1 for each architecture with latency-budget zones
5. **Cost savings bar chart** — expected cost: routing policy vs. each fixed architecture
6. **Action distribution table** — % hold / hedge / reverse / abstain / escalate per architecture per base rate

### Files to create/edit
- `src/final_visualizations.py` — plot generation script
- `output/phase15_final_metrics.json` — all consolidated metrics

### Copy back to Professor
```
Phase 15 complete. Final tables and figures generated.
6 deliverables ready: cross-domain metrics, routing table, PPV curves,
Pareto frontier, cost savings, action distributions.
Artifacts at output/phase15_final_metrics.json and plots/.
```
