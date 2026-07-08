# Meeting Evidence Map — Verification Arbitrage Framework

**Prepared:** 2026-07-07  
**Model evaluated:** DeepSeek-v4-flash (240 real API calls, ~$0.036 total)  
**Domains tested:** Finance, Healthcare, Political (N=10 each)  
**Architectures:** Single-Shot, Voting N=3 (repaired), MoA (repaired), RAG (evidence-category)

---

## Literature Context

This pilot sits at the intersection of several established research lines in LLM-based factuality and verification:

| Research Area | Prior Work | Relationship to This Work |
|---------------|-----------|--------------------------|
| **Factual-claim verification / FEVER-style** | FEVER (Thorne et al., 2018), SciFact (Wadden et al., 2020), AVeriTeC (Schlichtkrull et al., 2023) — standard benchmarks for retrieving evidence and classifying claim truth. | Our RAG verifier follows the same retrieve-then-classify paradigm but applies it to financial news where the "evidence" is not a knowledge-base snippet but a prior news article. |
| **Self-consistency / voting** | Wang et al. (2022) — Self-Consistency samples N reasoning paths and takes majority vote. Standard technique. | Our Voting N=3 extends this by using **role-differentiated** prompts (neutral, skeptical, base-rate-calibrated) rather than identical prompts. |
| **Debate / Mixture-of-Agents** | Irving et al. (2018) — debate for truthfulness; Mixture-of-Agents (Wang et al., 2024) — multi-agent collaboration; Du et al. (2023) — "Chatting to Correct." | Our MoA uses Believer/Skeptic/Risk-Officer role differentiation. Prior work shows debate can improve factuality but at high latency. |
| **Retrieval-Augmented Generation** | Lewis et al. (2020), Self-RAG (Asai et al., 2023) — RAG for grounding. | Our RAG uses evidence-category analysis. Self-RAG's "retrieve on demand" pattern is relevant for future work. |
| **Cost-aware LLM routing** | FrugalGPT (Chen et al., 2023) — route queries to cheapest adequate model. | **Closest parallel.** FrugalGPT routes between model tiers (GPT-4 vs GPT-3.5). Our routing is between **verifier architectures** (Single-Shot vs Voting vs MoA) conditioned on base rate, cost, and latency. |
| **Calibration & abstention** | Hendrycks & Gimpel (2016) — confidence calibration; Varshney et al. (2022) — selective classification/abstention. | Our ESCALATE verdict is a form of abstention. ECE tracking for confidence calibration follows this literature. |
| **Class imbalance / PPV** | Provost & Fawcett (1997) — ROC vs PPV at low base rates; Chawla et al. (2002) — class imbalance. | The core motivation: at P(Fake) < 1%, even a strong classifier's PPV collapses. This is well-known in fraud detection and medical screening, less commonly applied to LLM verification. |

### Research Gap

The literature studies **individual architectures** (single-shot, voting, debate, RAG) but does not study **when to use which one** under real-world constraints on base rate, cost asymmetry, and latency. The contribution we are exploring is not simply "architecture X beats baseline Y" but a **cost-aware, base-rate-aware, latency-aware verifier routing policy** that selects the optimal architecture per event based on context.

This reframes the question from "which verifier is best?" to "under what conditions should each verifier be deployed?"

---

## 1. Paper Section Map

### Abstract

| Claim | Supporting Evidence | Data/Figure | Not Tested Yet | Risk |
|-------|-------------------|-------------|----------------|------|
| LLM-based verification can distinguish real from fake news across domains | Single-Shot achieves F1=0.645 cross-domain; Voting N=3 achieves F1=0.827 | Repair-rerun cross-domain table | Not tested on non-English or adversarial inputs | **Solid** for finance/healthcare; **preliminary** for political |
| Multi-agent architectures improve over single-shot baseline | Voting N=3 beats Single-Shot by +0.182 F1; MoA beats by +0.102 | Cross-domain mean F1 table | N=10 only — statistical significance not established; voting/self-consistency is a well-known technique, so marginal improvement over single-shot is the expected baseline, not a surprise | **Preliminary** |
| Verification has cost-benefit crossover at specific base rates | Policy sensitivity sweep: cost_ratio drives 86.3% of variance; crossover at P(Fake)=~5% | Phase 8 sweep metrics | Not validated with real API-derived TPR/FPR | **Speculative** — uses mock FPRs |
| Framework generalizes across 3 domains (finance, healthcare, political) | Single-Shot F1: fin=0.833, health=0.769, pol=0.333 | Per-domain confusion matrices | Only N=10 per domain; political is 2016-era stylistically different | **Preliminary** |

### Introduction

| Claim | Supporting Evidence | Data/Figure | Not Tested Yet | Risk |
|-------|-------------------|-------------|----------------|------|
| Fake news flash crashes cause measurable market dislocation | 6 historical hoax events with 1.8% mean drawdown (from repo historical data) | Historical hoax dataset (legacy) | Not independently validated in paper | **Solid** (well-documented phenomenon) |
| Traditional NLP baselines are insufficient for out-of-distribution generalization | GBDT F1=0.996 on synthetic (claimed in CLAUDE.md Phase 16) | Mock-mode metrics | OOD evaluation pending — GBDT may collapse on human-written text | **Preliminary** |
| LLM verifiers can provide a second-opinion signal within latency constraints | Single-Shot latency: 3.4-4.7s; Voting N=3: 11.7-15.6s | Latency tables per architecture | Voting latency may exceed high-frequency windows; routing allows cheap verifier when time is short | **Solid** for Single-Shot; **at-risk** for Voting |

### Problem Definition

| Claim | Supporting Evidence | Data/Figure | Not Tested Yet | Risk |
|-------|-------------------|-------------|----------------|------|
| Verification arbitrage requires asymmetric loss framework | Cost-sensitive threshold function implemented in `src/base_rate.py` | Threshold = cost_fp / (cost_fp + cost_fn) | No empirical calibration of cost_fp/cost_fn from real trading data | **Solid** mathematically; **speculative** empirically |
| Base rate of fake news is extremely low (0.01%–5%) | Claim consistent with literature (cited in legacy base_rate_analysis.py) | N/A | Not empirically measured for our domains | **Solid** (external literature) |
| Three-tier system (T0→T1→T2) is appropriate architecture | Architecture described in project overview | Pipeline diagram | Only T1 verifier implemented; T0 and T2 not empirically tested | **Solid** framing; **speculative** in detail |

### Architecture

| Claim | Supporting Evidence | Data/Figure | Not Tested Yet | Risk |
|-------|-------------------|-------------|----------------|------|
| Single-Shot is the efficient baseline | F1=0.833 finance, 0.769 healthcare, 3.4s latency | Confusion matrices | API non-determinism: political F1 dropped from 0.571→0.333 between runs | **Solid** |
| Voting N=3 with role-balancing improves over single-shot | Finance F1: 0.833→1.000; Healthcare: 0.769→0.909; Cross-domain: 0.645→0.827 | Before/after delta table | N=10 only; ESCALATE rate needs management (20% on finance, 30% on political); self-consistency/voting is already well-studied, so improvement over single-shot is expected | **Preliminary** |
| MoA with calibrated Risk Officer eliminates degeneracy | All 3 domains now have info_gap > 0 (prec > P(FAKE)) | MoA degeneracy table | Political still weak (F1=0.500); latency 16-21s may be prohibitive | **Preliminary** — degeneracy fixed, performance mixed |
| Generic evidence retrieval is insufficient for claim verification | Cross-domain F1=0.413; 80% ESCALATE on finance and political | RAG evidence quality table | Curated, source-aware evidence (as in FEVER/SciFact) may be needed rather than generic TF-IDF retrieval; diagnosis not yet isolated | **At risk** |

### Dataset

| Claim | Supporting Evidence | Data/Figure | Not Tested Yet | Risk |
|-------|-------------------|-------------|----------------|------|
| Synthetic dataset provides controlled test conditions | 50/50 balanced, deterministic with seed=42 | Dataset notes | Not representative of real-world low base rates | **Solid** for controlled comparison |
| Three domains demonstrate generalizability | Finance (corporate actions), Healthcare (FDA/drug claims), Political (US 2016) | Domain descriptions | Political dataset is 2016-era and stylistically different from modern political disinformation | **Solid** for pilot; **preliminary** for coverage claim |
| Items are identical across architecture comparisons | Loaded from saved raw outputs — same claim texts, same ground truth | Repair-rerun methodology note | Ground truth inferred for some items (healthcare/political had ?:5) | **Solid** for items loaded from saved outputs; **caveat** for inferred ground truth |

### Experiment Results

| Claim | Supporting Evidence | Data/Figure | Not Tested Yet | Risk |
|-------|-------------------|-------------|----------------|------|
| Single-Shot achieves F1=0.833 on finance, 0.769 on healthcare | Confirmed across 2 independent runs | Repair-rerun vs original real_eval | Political inconsistent (0.571→0.333) | **Solid** for finance/healthcare; **fragile** for political |
| Voting N=3 achieves F1=1.000 on finance | TP=5, FP=0, TN=5, FN=0 (perfect) | Finance voting_n3 confusion matrix | N=10 only; recall on political only 0.400. Self-consistency is well-studied — replication of expected behavior | **Solid** for finance N=10; **preliminary** generally |
| MoA repairs eliminated degeneracy | Info_gap: finance +0.214, healthcare +0.333, political +0.167 | Degeneracy diagnostics | Political precision=0.667, info_gap narrow; 20% FPR on political | **Solid** for degeneracy claim; **preliminary** for performance claim |
| Generic RAG does not beat Single-Shot | Cross-domain F1: 0.413 vs 0.645 | Cross-domain comparison | Curated, source-aware evidence retrieval (FEVER-style) may be required; this does not rule out RAG as a paradigm | **Solid** negative result for generic retrieval |

### Sensitivity Analysis

| Claim | Supporting Evidence | Data/Figure | Not Tested Yet | Risk |
|-------|-------------------|-------------|----------------|------|
| Cost ratio (FP:FN) dominates policy outcomes | 86.3% variance contribution (Phase 8 sweep) | Parameter ranking table | Sweep used Single-Shot only; Voting/MoA may shift sensitivities | **Solid** for Single-Shot; **preliminary** for other architectures |
| Base rate has strong interaction with cost ratio | 12.1% interaction strength | Interaction matrix | Only tested at 1%, 5%, 25% base rates | **Preliminary** |
| Bayesian PPV collapses at low base rates | At P(Fake)=0.1%, PPV=0.0025 for Single-Shot finance | PPV tables in report | All PPV computed from N=10 metrics | **Solid** mathematically; **preliminary** empirically |

### Conclusion

| Claim | Supporting Evidence | Data/Figure | Not Tested Yet | Risk |
|-------|-------------------|-------------|----------------|------|
| Voting N=3 is currently the best-performing architecture we tested | Highest cross-domain F1 (0.827), beats SS (+0.182) | Architecture ranking | Self-consistency is already well-studied; the contribution must shift to routing, not architecture comparison | **Preliminary** |
| Hybrid base-rate strategy plausible but untested | Theoretical framework only | See Section 3 of this doc | No real-API evidence for any segment | **Speculative** |
| Generic evidence RAG needs curated, source-aware evidence | Evidence-category approach produces 80% ESCALATE | RAG quality metrics | Curated retrieval (FEVER/SciFact-style) not explored | **At risk** |

---

## 2. Current Empirical Findings (Discussion-Ready)

### Single-Shot Baseline
- **Finance**: F1=0.833, P=0.714, R=1.000, FPR=0.400, ECE=0.093, Lat=3.4s
- **Healthcare**: F1=0.769, P=0.625, R=1.000, FPR=0.600, ECE=0.270, Lat=3.7s
- **Political**: F1=0.333, P=1.000, R=0.200, FPR=0.000, ECE=0.250, Lat=4.7s
- **Cross-domain mean**: F1=0.645 (σ=0.222)
- **Key observation**: Single-Shot is recall-perfect on finance and healthcare but has high false-positive rates (40-60%). Political underperformance is structural, not architectural — all architectures struggle on 2016-style political content.
- **API non-determinism**: Political F1 ranged from 0.333 to 0.571 across 2 identical-parameter runs (temperature=0.0). This is a real methodological concern.

### Repaired Voting N=3
- **Finance**: F1=1.000, P=1.000, R=1.000, FPR=0.000, ECE=0.094, Lat=11.7s
- **Healthcare**: F1=0.909, P=0.833, R=1.000, FPR=0.200, ECE=0.211, Lat=12.8s
- **Political**: F1=0.571, P=1.000, R=0.400, FPR=0.000, ECE=0.210, Lat=15.6s
- **Cross-domain mean**: F1=0.827 (σ=0.184)
- **Key observation**: The 3-role balanced design (neutral verifier + skeptic + base-rate-calibrated) with soft aggregation (2/3 majority or ESCALATE) fixed the always-REAL degeneracy from the original run. On finance, it achieves perfect classification (F1=1.000). The ESCALATE rate (20% on finance, 20% on healthcare, 30% on political) is a useful fallback — it converts uncertain predictions into a "no decision" rather than forcing wrong binary answers.
- **Latency concern**: 11.7-15.6s may exceed the T1 trading window for HFT use cases, but is plausible for slower strategies.
- **⚠️ Caveat**: Self-consistency and voting are already well-studied in prior work (Wang et al., 2022). That voting outperforms single-shot in our pilot is the **expected** outcome and does not constitute a novel finding by itself. The novelty must come from the **routing** — when to use which verifier — not from the architecture comparison.

### Repaired MoA
- **Finance**: F1=0.833, P=0.714, R=1.000, FPR=0.400, ECE=0.064, Lat=16.0s
- **Healthcare**: F1=0.909, P=0.833, R=1.000, FPR=0.200, ECE=0.204, Lat=21.3s
- **Political**: F1=0.500, P=0.667, R=0.400, FPR=0.200, ECE=0.130, Lat=20.4s
- **Cross-domain mean**: F1=0.747 (σ=0.178)
- **Key observation**: The calibrated Risk Officer prompt ("Default to REAL when Believer provides more specific evidence. Default to FAKE when Skeptic provides clear contradictions.") eliminated the always-FAKE degeneracy. All 3 domains now show precision > P(FAKE) — info_gap ranges from +0.167 (political) to +0.333 (healthcare). Despite this, MoA does NOT outperform Voting N=3 on any domain, and adds 4-9s additional latency.
- **Calibration note**: MoA has the best ECE on finance (0.064) and political (0.130) — suggesting debate provides better confidence calibration than single-shot, even when binary accuracy is similar.

### RAG Limitations
- **Cross-domain mean**: F1=0.413 (σ=0.112)
- **Key observation**: RAG is the only architecture that does NOT beat Single-Shot (-0.232 below). The evidence-category prompt structure (supporting/contradicting/source/insufficient) produces 80% ESCALATE on finance and political despite retrieval quality being adequate (2.7-3.0 hits per query). Healthcare RAG is the best (0% ESCALATE, F1=0.571) because the corpus contains relevant contradicting evidence.
- **Root cause reframe**: This result does NOT mean "RAG does not work for verification." Prior work (FEVER, SciFact, AVeriTeC, Self-RAG) uses **curated, source-aware evidence** — not generic TF-IDF retrieval from a news corpus. Our RAG failure likely reflects **insufficiently targeted evidence retrieval**, not a fundamental limitation of the RAG paradigm. The all-REAL finance corpus provides no contradicting evidence for fake claims, causing the verifier to default to ESCALATE. Success with RAG may require: (a) a corpus that includes known-fake examples or contradiction signals, (b) source reliability metadata, or (c) a retrieve-on-demand paradigm (Self-RAG style).

### Cross-Domain Readiness
- Finance and healthcare are **solid** for pilot (Voting N=3 achieves F1=1.000 and 0.909 respectively).
- Political is **not ready** — best F1 is 0.571 (Voting) and results are unstable across runs.
- Cross-domain generalization is confirmed for Single-Shot and Voting within N=10 constraints, but **full generalization cannot be claimed** without larger samples and more domains.

### Calibration / ECE
| Architecture | Finance | Healthcare | Political | Cross-Domain |
|-------------|---------|------------|-----------|-------------|
| Single-Shot | 0.093 | 0.270 | 0.250 | 0.204 |
| Voting N=3 | 0.094 | 0.211 | 0.210 | 0.172 |
| MoA | 0.064 | 0.204 | 0.130 | **0.133** |
| RAG | 0.365 | 0.070 | 0.210 | 0.215 |

MoA provides the best calibration (lowest ECE) despite not having the best classification accuracy. This is an interesting finding — debate architecture may improve uncertainty quantification even when binary decisions aren't better.

### Latency Tradeoff
| Architecture | Mean Latency | vs Window | Recommendation |
|-------------|-------------|-----------|---------------|
| Single-Shot | 3.4-4.7s | ✅ Fits 5s window | Viable for HFT |
| Voting N=3 | 11.7-15.6s | ❌ Exceeds 5s | Viable for slower strategies (swing trading) |
| MoA | 16.0-21.3s | ❌ Exceeds 5s | Too slow for HFT; viable for fundamental verification |
| RAG | 3.4-6.0s | ✅ Fits 5s | Fast but inaccurate |

### Policy Sensitivity (Phase 8 — Mock Mode)
- **Cost ratio (86.3% variance)**: Dominates all other parameters. The ratio of false-positive cost to false-negative cost determines optimal policy.
- **Base rate × cost ratio interaction (12.1%)**: At higher base rates, false negatives become relatively more expensive, favoring "always reverse" policies.
- **Latency budget**: 0% variance contribution in mock sweep — but real API latencies vary 3-21s, which would change this dramatically.
- **Limitation**: Phase 8 sweep used Single-Shot metrics only, derived from TF-IDF+LR mock. Real V3/MoA metrics would shift optimal policies.

---

## 3. Hybrid / Base-Rate-Combination Strategy

> ⚠️ **Epistemic status**: The following strategy is **hypothetical**. It is grounded in the empirical findings above but has NOT been tested with real API calls under varying base-rate conditions.

### Strategy Selection Table

#### Segment A: Very Low Base Rate (P(Fake) < 0.1%)

| Dimension | Detail |
|-----------|--------|
| **Selected architecture** | Single-Shot + abstention |
| **Reason** | At extremely low base rates, even a 2% FPR produces more false alarms than real detections (PPV collapses to <2.5%). The optimal strategy is to abstain from trading on most alerts and only act on very high-confidence (>0.95) predictions. |
| **Expected strength** | Avoids FP costs that would dominate P&L at low base rates |
| **Expected weakness** | Misses rare genuine fake news events (recall on high-confidence threshold unknown) |
| **Evidence** | PPV at 0.1% base rate for Single-Shot finance: PPV=0.0025 (the verifier's "alerts" are 99.75% false alarms) |
| **Tested?** | ❌ **Untested** — PPV computed mathematically but no experiment with base-rate-stratified data |

**Confidence-conditioned gating**: Use the predicted confidence as a soft threshold:
- Confidence ≤ 0.85 → abstain (do not override initial trade)
- Confidence 0.85-0.95 → reduce position by 25%
- Confidence > 0.95 → full reversal
- Benefit: avoids acting on low-confidence predictions that are mostly false alarms at low base rates.

#### Segment B: Low Base Rate (0.1% ≤ P(Fake) < 1%)

| Dimension | Detail |
|-----------|--------|
| **Selected architecture** | Single-Shot + three-tier confidence gating |
| **Reason** | PPV is 2.5-24.6% at this range. Verifier alerts are still mostly false alarms, so hedging (partial position reduction) is safer than full reversal. Single-Shot's 3.4s latency fits the trading window. |
| **Expected strength** | Viable for operational use; latency-compliant; captures some TP savings |
| **Expected weakness** | Misses most fake news events due to high confidence threshold |
| **Evidence** | PPV at 1% base rate: Single-Shot finance=0.025, Voting=1.0. But Voting N=3 FPR=0.0 here also means it would predict nothing — the FPR of 0.0 is from N=10 and likely not real. |
| **Tested?** | ❌ **Untested** — need stratified evaluation with real low-base-rate data |

#### Segment C: Medium Base Rate (1% ≤ P(Fake) < 10%)

| Dimension | Detail |
|--------|-------|
| **Selected architecture** | Voting N=3 |
| **Reason** | This is the crossover zone where verifier PPV becomes operational (PPV reaches 10-46% for Single-Shot at 5% base rate). Voting N=3's superior precision (1.0 on finance, 0.833 on healthcare) and 11.7-12.8s latency are acceptable for non-HFT contexts. |
| **Expected strength** | Best measured performance; Voting N=3 achieves highest overall F1 |
| **Expected weakness** | Voting has 20-30% ESCALATE rate which may be operationally costly (human review needed) |
| **Crossover verification**: At P(Fake)=5.8% (the crossover from Phase 22 calibration), verify-first expected P&L becomes positive for Single-Shot. But this is computed from legacy FPR values — needs recomputation with real-api Voting metrics. |
| **Tested?** | ⚠️ **Partially tested** — Voting N=3 evaluated at 50/50 base rate only |

#### Segment D: High Base Rate (10% ≤ P(Fake) < 50%)

| Dimension | Detail |
|--------|-------|
| **Selected architecture** | Voting N=3 or MoA (preference: Voting) |
| **Reason** | At higher base rates, FN cost dominates FP cost — maximizing recall is critical. Both Voting and MoA achieve recall=1.0 on finance and healthcare. Voting is preferred due to lower latency and strong calibration. |
| **Expected strength** | Captures nearly all fake news events (recall=1.0) |
| **Expected weakness** | FPR of 20-40% means substantial unnecessary reversals |
| **Evidence** | Both Voting and MoA: finance recall=1.0, healthcare recall=1.0. At 25% base rate, Voting PPV=0.625 — still 37.5% of reversals are unnecessary. |
| **Tested?** | ⚠️ **Partially tested** — metrics at 50/50 only |

#### Segment E: Crisis / Adversarial Regime (P(Fake) > 50% or active attack)

| Dimension | Detail |
|--------|-------|
| **Selected architecture** | Ensemble: Voting N=3 + MoA (use both, override if both flag FAKE) |
| **Reason** | During active misinformation attacks, the social stream may be contaminated (a known failure mode from stress tests). Using both architectures as independent signals provides redundancy. If both agree on FAKE, confidence is high. If they disagree, escalate for manual review. |
| **Expected strength** | Maximum detection coverage during crisis |
| **Expected weakness** | High operational cost (human review for disagreements); latency likely exceeds 20s |
| **Evidence** | Stress test (adversarial bot stream) was only run in mock mode — no real-API evidence |
| **Tested?** | ❌ **Untested** — ensemble not implemented; adversarial testing only in mock mode |

### Strategy Decision Flowchart

```
P(Fake) estimate from recent news stream?
    ├── < 0.1% → Single-Shot + confidence gating + partial hedge only
    ├── 0.1% – 1% → Single-Shot + three-tier gating
    ├── 1% – 10% → Voting N=3 (operational crossover zone)
    ├── 10% – 50% → Voting N=3 (high-recall mode)
    └── > 50% → Ensemble Voting + MoA + escalation
```

### How to Estimate Base Rate Online

The `BaseRateEstimator` class in `src/base_rate.py` provides a rolling-window estimator:
- Tracks last N verdicts (default 100)
- Falls back to configurable prior (default 5%)
- Updates after each prediction
- Could be adapted to weight by recency or incorporate domain-specific priors

---

## 4. Minimal Next Experiment Proposal

### Testing the Hybrid Base-Rate Strategy

**Goal**: Validate that different architectures perform differently under different base-rate regimes, and that the optimal architecture switches as base rate changes.

**Input data needed**:
- The existing N=30 synthetic dataset (10/domain) — already collected, ground truth known
- **OR** a larger synthetic dataset with controllable base rate: generate N=200 items per domain with class imbalance varying from P(Fake)=1% to P(Fake)=50%

**Recommended approach** (minimal cost opton):
1. **Use existing data** but compute metrics with **stratified subsampling**:
   - Create synthetic test sets with controlled base rates by downsampling the majority class
   - Test P(Fake) = {1%, 5%, 10%, 25%, 50%}
   - At each base rate, evaluate: Single-Shot, Voting N=3, MoA
   - This requires no new API calls — just recomputation on existing raw outputs

2. **Metrics to compute per segment**:
   - PPV and NPV (Bayesian, using measured TPR/FPR)
   - Expected P&L per event using the simulator cost parameters
   - Optimal architecture at each segment
   - Confidence calibration per segment

**Expected runtime**: <2 minutes (no API calls needed; pure post-processing)

**What would count as success**:
- Single-Shot outperforms Voting N=3 at P(Fake) < 1% (because Voting's marginal precision gain doesn't justify the latency)
- Voting N=3 outperforms Single-Shot at P(Fake) ≥ 5% (crossover confirmed)
- The crossover point shifts when FP/FN cost ratio changes (policy sensitivity validated)
- Result: a table of "optimal architecture × base rate × cost ratio"

**If no clear crossover is found**: The hybrid strategy may not be necessary — Single-Shot may be sufficient at all base rates, or Voting may dominate universally. This is still a useful negative result.

### Alternative: N=50 Scaling

If the professor prefers **statistical confidence** over base-rate stratification first:
- Scale Voting N=3 to N=50/domain = 150 real API calls (~$0.023)
- Scale Single-Shot as control: 150 calls (~$0.023)
- Cost: ~$0.046 total
- Outcome: 95% confidence intervals on all metrics to establish statistical significance

---

## 5. Meeting Questions for Professor

### Strongest 5 Discussion Points

1. **Should our contribution be hybrid verifier routing under base-rate, cost, and latency constraints?** Prior work already studies single-shot, voting/self-consistency, debate/MoA, and RAG individually. A simple "Voting N=3 beats Single-Shot" is not novel — it replicates an expected result. The question is whether the paper's novelty should be: **given base rate, cost asymmetry, and latency budget, which verifier should be deployed for this specific claim?** This would reframe the contribution from architecture comparison to cost- and context-aware routing.

2. **Political domain inconsistency (F1 ranges 0.333-0.571 across runs) is a methodological concern. Should we (a) replace political with a more modern dataset, (b) accept it as a domain boundary condition, or (c) frame the paper around finance+healthcare only?**

3. **The hybrid/base-rate strategy is theoretically grounded (PPV collapses at low base rates) but empirically untested with real API outputs. Is this worth a dedicated experiment, or is the mathematical argument sufficient for the paper?**

4. **MoA has better calibration (ECE=0.133) than Voting (ECE=0.172) but worse accuracy. If the paper emphasizes reliable confidence estimates (for risk management), MoA may still be valuable despite lower F1. How should we weigh calibration vs classification accuracy?**

5. **RAG's consistent underperformance (F1=0.413) with generic retrieval should be distinguished from curated-RAG approaches (FEVER/SciFact). Is it worth building a better retrieval corpus to test the curated-RAG hypothesis, or should the paper simply report that generic evidence-retrieval was insufficient and cite the curated-RAG literature as future work?**

### All 8 Questions

1. **Novelty framing: architecture comparison vs hybrid routing** — Prior work covers all four architectures individually. The gap is not "which architecture wins" but "under what conditions should each be deployed." Should the paper's contribution shift to cost-aware, base-rate-aware, latency-aware verifier routing policy?

2. **HFT vs general verification framing** — Does the 5-second T1 window constrain architecture choice? Voting N=3 achieves F1=0.827 but at 11-16s latency. If the paper is about general LLM verification with finance as a use case, latency is less critical. Which framing is stronger for the target venue?

3. **Voting N=3 scaling** — Is the improvement from Single-Shot (F1=0.645 → 0.827) sufficient to justify scaling to N=50/domain for statistical confidence? Would the committee expect confidence intervals on all F1 claims?

4. **MoA's role** — MoA beats Single-Shot but doesn't beat Voting N=3, while adding 4-9s latency. Given the degeneracy was fixed, is MoA still a contribution (better calibration, consistent abstention behavior) or should it be relegated to a negative-result appendix?

5. **Hybrid strategy experiment priority** — The base-rate-stratified strategy (different architecture per P(Fake) segment) is the most novel claim of this work. Is it worth investing API calls to test, or is the theoretical PPV framework sufficient?

6. **RAG failure description** — How should we describe RAG's failure fairly? Options: (a) "Generic evidence retrieval is insufficient for claim verification" (nuanced negative result), (b) "RAG requires curated, source-aware evidence corpora to be effective" (positive framing of future work), (c) "Evidence-category prompting is not suitable for all-REAL corpora" (technical diagnosis).

7. **Political domain treatment** — Should we (a) exclude political from the paper due to inconsistency, (b) include with strong caveats, or (c) replace with a different domain (e.g., ESG/sustainability claims, earnings call transcripts)?

8. **Calibration vs accuracy** — For risk management applications (e.g., VaR circuit breakers, position sizing), calibrated confidence may be more important than binary F1. Should the paper emphasize calibration metrics (ECE) alongside classification metrics?

---

## How Our Work Can Be Novel

Given the literature context above, here are the concrete ways this work can contribute beyond replicating known architecture comparisons.

### 1. Same-Claim Comparison Across Four Architecture Families

Prior work studies each architecture in isolation. We provide the **first direct comparison** (to our knowledge) of Single-Shot, Voting, MoA, and RAG on **identical claims** across **multiple domains**. This allows principled reasoning about when each architecture is worth its cost.

**What this enables**: The empirical basis for a cost-aware routing policy that treats architecture selection as a decision variable.

### 2. PPV / Base-Rate Reporting Instead of F1 Alone

Standard NLP verification papers report F1, accuracy, precision, recall at 50/50 class balance. This gives a misleading picture for real-world deployment where base rates are 0.01-5%. We report **Bayesian PPV across base rates** (0.1%-50%) for every architecture-domain combination. This directly addresses the base-rate fallacy and enables operational decisions.

**What this enables**: Whether a verifier is usable at a given base rate (PPV > 0.5 threshold) becomes an empirical question with per-architecture answers.

### 3. Latency-Aware Routing

Reporting mean, median, and p95 latency per architecture is standard. Connecting latency to **routing policy** — choosing a faster but less accurate verifier when time is scarce — is less common. If budget allows, we escalate to a more expressive (but costlier) verifier.

**What this enables**: A decision policy that selects verifier based on time-until-deadline, not just expected accuracy.

### 4. Cost-Sensitive Action Thresholds

We don't just classify — we map verdicts to actions (hold, hedge, reverse) via an explicit cost model. This is the FrugalGPT insight applied to verifier architectures: different regimes call for different cost-benefit tradeoffs.

**What this enables**: Rather than thresholding one binary classifier, we select entire architectures whose cost-benefit profiles match the operating environment.

### 5. Abstention / Escalation as First-Class Outputs

ESCALATE is not a failure mode — it is a **deliberate third output** that signals "insufficient evidence to decide." This is distinct from standard binary classification and maps directly to selective classification / abstention literature.

**What this enables**: Systems that gracefully degrade to human review when confidence is low, rather than forcing a costly wrong decision.

### 6. Finance as a Concrete High-Cost Case Study

Medical diagnosis is the canonical high-sensitivity application. Finance provides a complementary case where **both FP and FN have direct, measurable dollar costs**, making the cost-sensitive framework natural rather than abstract.

**What this enables**: Dollar-valued cost ratios that are grounded in market microstructure (slippage, spread crossing, adverse selection) rather than arbitrary loss matrices.

---

## Appendix: Data Sources

All data in this document comes from:
- `output/repair_rerun/report.json` — 240 real DeepSeek API calls across 4 architectures × 3 domains × 10 items
- `output/real_eval/report.json` — Original real API evaluation (same samples, before repair)
- `output/phase08_sensitivity_metrics.json` — Policy sensitivity sweep (72 configurations, mock mode)
- `src/base_rate.py` — Bayesian PPV/cost-sensitive threshold implementation
- `src/legacy/base_rate_analysis.py` — Legacy base rate fallacy analysis (old HFT parameters)

No new API calls were made for this document.

---

## Appendix: Untested Claims — Do Not Overstate

| Claim | Why not to overstate | Risk level |
|-------|---------------------|------------|
| "Voting N=3 achieves F1=0.827" | N=10 per domain; no confidence intervals; voting/self-consistency is already well-studied so this is expected, not novel | Medium |
| "MoA degeneracy is resolved" | Political info_gap is only +0.167, and political precision=0.667 is borderline | Low-Medium |
| "Cross-domain generalization demonstrated" | Only 3 domains, only N=10, political is not representative of modern disinformation | High |
| "Hybrid base-rate strategy improves P&L" | Entirely untested with real API outputs; based on theoretical PPV calculations | High |
| "Generic RAG does not work for verification" | Curated, source-aware evidence retrieval (FEVER/SciFact) was not tested; only generic TF-IDF with an all-REAL corpus was evaluated | Medium |
| "Policy sensitivity analysis shows cost_ratio dominates" | Analysis used mock-mode metrics (TF-IDF+LR), not real API metrics | Medium |
| "System fits 5-second trading window" | Single-Shot does; Voting and MoA do not — paper must be clear about which architecture is being discussed for which latency budget | Low |
| "API non-determinism at temperature=0.0 is a concern" | Only observed on political domain; may be domain-specific rather than general | Low |

---

## Appendix: Recommended Next Experiment After Meeting

Depending on professor feedback, prioritize one of:

1. **If hybrid strategy is interesting**: Run base-rate-stratified analysis on existing data (no new API calls). Deliverable: crossover table showing optimal architecture × base rate.

2. **If statistical significance is priority**: Scale Voting N=3 to N=50/domain (150 API calls, ~$0.023). Deliverable: 95% confidence intervals on all metrics.

3. **If political domain needs fixing**: Replace political with a modern social-media dataset (e.g., tweets about stock manipulation, earnings rumors). Or exclude political and run finance + healthcare at N=50 each.

4. **If RAG root cause is critical**: Build a curated retrieval corpus with source reliability labels (FEVER-style) and contrastive examples. Test whether better evidence changes RAG outcomes.

5. **If calibration framing wins**: Run MoA at N=50/domain to get stable ECE estimates. Deliverable: calibration reliability diagrams with confidence intervals.

---

## Bibliography

1. Thorne, J., et al. (2018). **FEVER: a large-scale dataset for Fact Extraction and VERification.** NAACL. [https://fever.ai](https://fever.ai)
2. Wadden, D., et al. (2020). **Fact or Fiction: Verifying Scientific Claims.** EMNLP (SciFact). [https://github.com/allenai/scifact](https://github.com/allenai/scifact)
3. Schlichtkrull, M., et al. (2023). **AVeriTeC: A Dataset for Real-world Claim Verification with Evidence.** EACL. [https://github.com/IBM/AVeriTeC](https://github.com/IBM/AVeriTeC)
4. Wang, X., et al. (2022). **Self-Consistency Improves Chain of Thought Reasoning in Language Models.** ICLR. [https://arxiv.org/abs/2203.11171](https://arxiv.org/abs/2203.11171)
5. Wang, J., et al. (2024). **Mixture-of-Agents Enhances Large Language Model Capabilities.** [https://arxiv.org/abs/2406.04692](https://arxiv.org/abs/2406.04692)
6. Irving, G., et al. (2018). **AI Safety via Debate.** [https://arxiv.org/abs/1805.00899](https://arxiv.org/abs/1805.00899)
7. Du, Y., et al. (2023). **Improving Factuality and Reasoning in Language Models through Multiagent Debate.** ICML. [https://arxiv.org/abs/2305.14325](https://arxiv.org/abs/2305.14325)
8. Lewis, P., et al. (2020). **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.** NeurIPS. [https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401)
9. Asai, A., et al. (2023). **Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.** ICLR. [https://arxiv.org/abs/2310.11511](https://arxiv.org/abs/2310.11511)
10. Chen, L., et al. (2023). **FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance.** [https://arxiv.org/abs/2305.05176](https://arxiv.org/abs/2305.05176)
11. Hendrycks, D. & Gimpel, K. (2016). **A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks.** ICLR. [https://arxiv.org/abs/1610.02136](https://arxiv.org/abs/1610.02136)
12. Varshney, N., et al. (2022). **A Survey on Selective Classification and Abstention in Machine Learning.** [https://arxiv.org/abs/2207.04742](https://arxiv.org/abs/2207.04742)
13. Provost, F. & Fawcett, T. (1997). **Analysis and Visualization of Classifier Performance: Comparison under Imprecise Class and Cost Distributions.** KDD.
14. Chawla, N., et al. (2002). **SMOTE: Synthetic Minority Over-sampling Technique.** JAIR.
