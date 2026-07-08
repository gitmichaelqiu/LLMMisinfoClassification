# Meeting Evidence Map — Verification Arbitrage Framework

**Prepared:** 2026-07-07  
**Model evaluated:** DeepSeek-v4-flash (240 real API calls, ~$0.036 total)  
**Domains tested:** Finance, Healthcare, Political (N=10 each)  
**Architectures:** Single-Shot, Voting N=3 (repaired), MoA (repaired), RAG (evidence-category)

```mermaid
flowchart LR
    subgraph T0["T₀: Detection (not implemented)"]
        A[("News Feed")]
        B["Fast NLP Trigger<br>(FinBERT / GBDT)"]
        A --> B
    end
    subgraph T1["T₁: Verifier (this work)"]
        C["Verifier Router<br>Latency check"]
        D["Single-Shot<br>(<5s budget)"]
        E["Voting N=3<br>(≥5s budget)"]
        F["MoA / RAG<br>(special cases)"]
        C -- "<5s" --> D
        C -- "≥5s" --> E
        C -- "research" --> F
    end
    subgraph T2["T₂: Human Review (not implemented)"]
        G["Manual verification<br>(~300s)"]
    end
    B --> C
    D --> H{Action}
    E --> H
    F --> H
    H -->|"FAKE"| I["Reverse / Hedge"]
    H -->|"REAL"| J["Hold"]
    H -->|"ESCALATE"| K["Partial hedge → T₂"]
    K --> G
```

*System architecture: T₀ fast trigger → T₁ LLM verifier (this work, latency-routed) → T₂ human review. Dashed lines show unimplemented components.*

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

```mermaid
quadrantChart
    title Research Positioning: Verifier Complexity vs Operating Regime
    x-axis "Low base rate / Tight latency" --> "High base rate / Loose latency"
    y-axis "Simple (single LLM call)" --> "Complex (multi-agent / debate)"
    quadrant-1 "Over-engineered\n(not worth cost)"
    quadrant-2 "Multi-Architecture Target\n(our routing gap)"
    quadrant-3 "Baseline Sufficient\n(no routing needed)"
    quadrant-4 "Under-powered\n(can't meet accuracy)"
    Single-Shot: [0.2, 0.15]
    Voting-N-3: [0.6, 0.45]
    MoA: [0.7, 0.80]
    RAG: [0.4, 0.35]
    "Routing Policy (gap)": [0.5, 0.55]
```

*The quadrant chart shows where each architecture lives. The routing gap is the ability to move between quadrants as operating conditions change.*

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

### Phase 11: Base-Rate-Stratified Routing Analysis (Real API Outputs)

**No new API calls.** Computed PPV curves, expected costs, and optimal architecture across 72 regimes (6 base rates × 4 cost ratios × 3 latency budgets) using existing real API outputs.

| Architecture | F1 | Precision | Recall | FPR | Latency | ESCALATE | Regimes Won |
|---|---|---|---|---|---|---|---|
| **Single-Shot** | 0.645 | 0.780 | 0.733 | 0.333 | 3.9s | 3% | 10 (14%) |
| **Voting N=3** | **0.827** | **0.944** | 0.800 | **0.067** | 13.3s | 23% | 30 (42%) |
| MoA | 0.747 | 0.738 | 0.800 | 0.267 | 19.2s | 10% | 0 (0%) |
| RAG | 0.413 | 1.000 | 0.267 | 0.000 | 4.3s | 53% | 32 (44%)* |

*\*RAG's 44% win rate is an ESCALATE abstention artifact — see Key Finding #5.*

**Key findings:**
1. **Voting N=3 is the best architecture overall** (F1=0.827, P=0.944, FPR=0.067) but requires >5s latency budget (13.3s mean).
2. **Single-Shot is the practical choice for <5s latency budgets** (F1=0.645, latency=3.9s) — the only architecture with mean latency under 5s.
3. **MoA never uniquely wins** — Pareto-dominated by Voting N=3 (lower F1, higher latency, similar recall).
4. **RAG's apparent dominance at low base rates (44% of regimes) is an artifact.** FPR=0.000 is achieved by ESCALATE-ing on 53% of items (treating ESCALATE as cost-free correct rejection). Real recall = 0.267. If ESCALATE carries operational cost (manual review, delayed decision), RAG wins zero regimes.
5. **No base-rate-sensitive architecture switching.** The optimal architecture is stable across base rates (0.1%–50%) and cost ratios (FP:FN 1:1–1:25) within each latency tier.
6. **Latency budget is the binding constraint** and the sole driver of architecture selection.

**Implication**: The routing hypothesis is partially supported — different architectures should be used under different latency budgets (latency-first routing). But true base-rate-sensitive architecture switching (different architecture at different P(Fake) levels) was not observed. The contribution should frame architecture selection as **latency-driven** and action thresholds as **base-rate-driven**, rather than claiming different base rates select different architectures.

```mermaid
pie title Regime Win Distribution (72 configurations)
    "Voting N=3 — 30 (42%)" : 30
    "Single-Shot — 10 (14%)" : 10
    "RAG* — 32 (44%)" : 32
```

*\*RAG's 44% is an ESCALATE abstention artifact — see finding #4. MoA wins 0 regimes.*

---

## 3. Hybrid / Latency-Aware Routing Strategy

> ⚠️ **Epistemic status**: The following strategy is **hypothetical in parts** but informed by Phase 11 analysis of 240 real API outputs. The architecture selection rule is empirically grounded; the action-threshold component is theoretically motivated but untested with real API outputs under varying base-rate conditions.

### Phase 11 Finding: Latency-First Routing

Phase 11 (base-rate-stratified analysis on 240 existing real API outputs) found:
- **Architecture selection is driven by latency budget, not base rate or cost ratio.**
- Within each latency tier, a single architecture dominates across all tested base rates (0.1%–50%) and cost ratios (FP:FN from 1:1 to 1:25).

| Latency Budget | Feasible Architectures | Optimal | F1 | Why |
|---|---|---|---|---|
| **<5s** | Single-Shot (3.9s), RAG (4.3s) | **Single-Shot** | 0.645 | RAG's apparent wins are ESCALATE artifacts (see below) |
| **≥5s, <15s** | Voting N=3 (13.3s), Single-Shot, RAG | **Voting N=3** | 0.827 | Dominates on accuracy, precision, and expected cost |
| **≥15s** | All architectures feasible | **Voting N=3** | 0.827 | MoA adds latency without accuracy gain; RAG non-competitive |

**Key negative result**: No base-rate-sensitive architecture switching was observed. The optimal architecture within each latency tier is stable across base rates 0.1%–50% and FP:FN ratios 1:1–1:25. This means:
- The **architecture** is selected by latency budget (how much time is available before the trading decision deadline).
- The **base rate** and **FP:FN cost ratio** determine the **intervention threshold and action policy** (how to act on the verifier's output), not which verifier to call.

### Revised Strategy: Two-Level Decision

#### Level 1: Architecture Selection (Latency-Driven)

```mermaid
flowchart TD
    Q["Time until trading deadline?"]
    Q -->|"<5s"| SS["Single-Shot<br>F1=0.645, Lat=3.9s<br>Feasible: ✅"]
    Q -->|"≥5s"| V3["Voting N=3<br>F1=0.827, Lat=13.3s<br>Feasible: ✅"]
    SS --> ACT["Action Policy<br>(Level 2)"]
    V3 --> ACT
    MOA["MoA: Pareto-dominated by Voting<br>Lat=19.2s, F1=0.747, Wins=0"] -.->|"Not recommended"| ACT
    RAG["RAG: ESCALATE artifact<br>Lat=4.3s, F1=0.413"] -.->|"Not recommended"| ACT
```

- **Single-Shot (<5s)**: Cross-domain F1=0.645, P=0.780, R=0.733, Lat=3.9s. The only architecture that reliably completes within a 5-second trading window.
- **Voting N=3 (≥5s)**: Cross-domain F1=0.827, P=0.944, R=0.800, Lat=13.3s. Dominates on all accuracy metrics when latency permits.
- **MoA**: Never uniquely optimal — Pareto-dominated by Voting N=3 (lower F1, higher latency). Dropped from routing consideration.
- **RAG**: Wins zero regimes when ESCALATE carries any operational cost. The current TF-IDF retrieval pipeline is insufficient.

#### Level 2: Action Policy (Base-Rate and Cost-Ratio Driven)

Once the verifier architecture is selected, the action threshold and position sizing are determined by the estimated base rate and FP:FN cost ratio:

| Base Rate Regime | Action Rule | Rationale |
|---|---|---|
| **Very Low (<0.1%)** | High-confidence gating: only act on P(fake) > 0.95 | PPV collapse at low base rates makes most "alerts" false alarms regardless of architecture |
| **Low (0.1%–1%)** | Moderate-confidence gating: hedge (partial reduction) on P(fake) > 0.85, full reverse on >0.95 | PPV 2.5%–24.6% even for Single-Shot; most positive predictions are still FP |
| **Medium (1%–10%)** | Standard asymmetric threshold: reverse when P(fake) > cost_fp/(cost_fp+cost_fn) | PPV becomes operational (10%–46+%); cost-sensitive threshold selects the optimal trade-off |
| **High (>10%)** | Low threshold: reverse on weak signals | FN cost dominates; maximizing recall is more important than precision |

### Action Policy Decision Tree

```mermaid
flowchart TD
    V["Verdict from Verifier"]
    V -->|"FAKE"| FC{Confidence ≥ threshold?}
    V -->|"REAL"| RH["Hold position<br>No action needed"]
    V -->|"ESCALATE"| ESC{Hedge or Abstain?}

    FC -->|"Yes"| REV["Reverse position<br>Full reversal"]
    FC -->|"No"| HEDGE["Hedge: partial reduction<br>Sized by confidence"]

    ESC -->|"Low P(Fake)"| ABSTAIN["Abstain: do nothing<br>Avoid FP cost at low base rate"]
    ESC -->|"Medium P(Fake)"| PARTIAL["Partial hedge:<br>Reduce 25-50% of position"]
    ESC -->|"High P(Fake)"| REVIEW["Route to T₂ manual review<br>while holding reduced position"]

    subgraph THRESH["Threshold Selection"]
        T1["Base Rate < 0.1%: threshold=0.95"]
        T2["Base Rate 0.1-1%: threshold=0.85"]
        T3["Base Rate 1-10%: cost-sensitive threshold<br>= cost_fp / (cost_fp + cost_fn)"]
        T4["Base Rate > 10%: low threshold<br>Maximize recall"]
    end
    
    THRESH -.->|"Determines<br>confidence gate"| FC
```

A single architecture (Voting N=3 or Single-Shot) with this **confidence-gated action policy** achieves the same effect as the previously hypothesized multi-architecture segments. The `BaseRateEstimator` class in `src/base_rate.py` provides a rolling-window estimator to track the current base rate and adjust thresholds online.

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

1. **Should the contribution be latency-aware verification routing plus base-rate-aware action thresholds, rather than base-rate-driven architecture switching?** Phase 11 found no evidence that different base rates select different architectures — the optimal architecture is stable across 0.1%–50% base rates within each latency tier. Architecture choice is driven by latency budget (<5s → Single-Shot, ≥5s → Voting N=3). The base rate and cost ratio determine the **action policy** (confidence threshold, position sizing), not which verifier to call. This is an important correction to the previously hypothesized multi-architecture base-rate strategy. Should the paper frame the contribution as: *latency-driven verifier selection + base-rate-aware intervention thresholding*?

2. **Political domain inconsistency (F1 ranges 0.333-0.571 across runs) is a methodological concern. Should we (a) replace political with a more modern dataset, (b) accept it as a domain boundary condition, or (c) frame the paper around finance+healthcare only?**

3. **Voting N=3 dominates all latency-permissive regimes** (F1=0.827, P=0.944, FPR=0.067, 42% of regimes optimal). But self-consistency/voting is well-studied. Is Voting N=3's dominance a sufficient result for the paper, or does the paper need a novel contribution like the latency-aware routing policy to differentiate from prior work?

4. **MoA has better calibration (ECE=0.133) than Voting (ECE=0.172) but worse accuracy and never uniquely wins any regime.** If the paper emphasizes reliable confidence estimates (for risk management), MoA may still be valuable despite lower F1. But with zero regimes where MoA is optimal, should we (a) include MoA as a calibration-focused architecture, (b) relegate it to a negative-result appendix, or (c) drop it entirely?

5. **RAG's consistent underperformance (F1=0.413) with generic retrieval** should be distinguished from curated-RAG approaches (FEVER/SciFact). Is it worth building a better retrieval corpus to test the curated-RAG hypothesis, or should the paper simply report that generic evidence-retrieval was insufficient and cite the curated-RAG literature as future work?**

### All 9 Questions

1. **Novelty: latency-aware routing + base-rate-aware thresholds vs architecture comparison** — Phase 11 found no base-rate-sensitive architecture switching. The optimal architecture is stable across base rates 0.1%–50% within each latency tier. Should the paper frame the contribution as: (a) latency-driven verifier routing + base-rate-aware action thresholds, or (b) is that too incremental for the target venue? This is the most important framing decision.

2. **Which research direction should we prioritize** given Phase 11's negative result (no base-rate-sensitive architecture switching)? Options:
   - **A. Latency-aware verifier routing** — Formalize the two-tier routing policy (SS for <5s, Voting for ≥5s). Compare against fixed-architecture baselines. This is the "safe" direction: clear experiment, well-defined contribution, but is it novel enough?
   - **B. Base-rate-aware action thresholding** — Keep one architecture (Voting N=3) but focus on the confidence-gated action thresholds and PPV-based abstention policy. This is a decision-theoretic contribution rather than a systems contribution.
   - **C. Larger statistical validation of Voting N=3** — Scale to N=50/domain, establish significance bounds, produce reliable figures. This is empirical rigor but less conceptual novelty.
   - **D. RAG redesign** — Build a curated, source-aware retrieval corpus (FEVER-style) to test whether RAG's failure is the retrieval paradigm or the evidence quality. This is highest-risk but could recover the RAG architecture path.
   - **E. Some combination** — which two or three?

3. **HFT vs general verification framing** — Does the 5-second T1 window constrain architecture choice? Voting N=3 achieves F1=0.827 but at 11-16s latency. If the paper is about general LLM verification with finance as a use case, latency is less critical. Which framing is stronger for the target venue?

4. **Voting N=3 scaling** — Is the improvement from Single-Shot (F1=0.645 → 0.827) sufficient to justify scaling to N=50/domain for statistical confidence? Phase 11 already confirms Voting N=3 dominance across 72 regimes — would CIs change the story?

5. **MoA's role after Phase 11** — MoA now wins zero regimes. It never outperforms Voting N=3 on any regime. Should MoA be (a) dropped from the paper, (b) kept as a calibration-focused negative result (ECE=0.133 vs Voting 0.172), or (c) kept as evidence that debate architectures add latency without accuracy benefit?

6. **Hybrid strategy experiment priority after Phase 11** — Phase 11 is already done (zero API calls, 72 regimes analyzed). The next experiment depends on direction A/B/C/D from question 2 above. Which direction should we fund with API calls?

7. **RAG failure description after Phase 11** — RAG's "wins" in 44% of regimes are now known to be ESCALATE artifacts. Should we report: (a) "RAG's FPR=0 is achieved by 53% abstention — not a real advantage," (b) "RAG wins no regimes when ESCALATE carries cost," or (c) "The Phase 11 analysis demonstrates that generic retrieval RAG does not improve over Single-Shot"?

8. **Political domain treatment** — Should we (a) exclude political from the paper due to inconsistency, (b) include with strong caveats, or (c) replace with a different domain (e.g., ESG/sustainability claims, earnings call transcripts)?

9. **Calibration vs accuracy** — For risk management applications (e.g., VaR circuit breakers, position sizing), calibrated confidence may be more important than binary F1. Should the paper emphasize calibration metrics (ECE) alongside classification metrics?

---

## How Our Work Can Be Novel

Given the literature context above, here are the concrete ways this work can contribute beyond replicating known architecture comparisons.

```mermaid
flowchart LR
    subgraph STANDARD["Standard NLP Verification Paper"]
        S1["Evaluate at 50/50 class balance"]
        S2["Report F1, precision, recall"]
        S3["Single domain or corpus"]
        S4["One architecture (usually)"]
    end
    subgraph OURS["Our Contribution"]
        O1["PPV across base rates<br>0.1% to 50%"]
        O2["Latency-aware routing<br><5s → SS, ≥5s → Voting"]
        O3["3 domains: finance,<br>healthcare, political"]
        O4["4 architectures compared<br>on identical claims"]
    end
    S1 -.->|"Fails at<br>deployment"| O1
    S2 -.->|"Missing<br>operational view"| O1
    S3 -.->|"Limited<br>generalization"| O3
    S4 -.->|"No routing<br>insight"| O2
```

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
| "Hybrid base-rate strategy improves P&L" | Entirely untested with real API outputs; based on theoretical PPV calculations. Phase 11 found no evidence of base-rate-sensitive architecture switching — the routing strategy must be latency-first, not base-rate-driven | High |
| "Different base-rate regimes choose different verifier architectures" | Phase 11 found the optimal architecture is stable across base rates 0.1%–50% within each latency tier. Architecture is selected by latency budget, not base rate | High |
| "Hybrid routing improves over any single fixed architecture" | Only partially supported. Voting N=3 dominates all ≥5s regimes. The improvement comes from switching to Single-Shot when latency is tight (<5s), not from base-rate switching. Within each tier, one architecture is universally best | Medium-High |
| "RAG wins some regimes" | RAG's 44% regime win rate is an ESCALATE abstention artifact. FPR=0.000 comes from ESCALATE-ing on 53% of items, treated as cost-free correct rejections. If ESCALATE carries any operational cost, RAG wins zero regimes | High |
| "Generic RAG does not work for verification" | Curated, source-aware evidence retrieval (FEVER/SciFact) was not tested; only generic TF-IDF with an all-REAL corpus was evaluated | Medium |
| "Policy sensitivity analysis shows cost_ratio dominates" | Analysis used mock-mode metrics (TF-IDF+LR), not real API metrics | Medium |
| "System fits 5-second trading window" | Single-Shot does; Voting and MoA do not — paper must be clear about which architecture is being discussed for which latency budget | Low |
| "API non-determinism at temperature=0.0 is a concern" | Only observed on political domain; may be domain-specific rather than general | Low |

---

## Appendix: Recommended Next Experiment After Meeting

**Phase 11 is complete** (base-rate-stratified analysis on 240 existing API outputs, 72 regimes). Depending on professor feedback on question 2 (the A/B/C/D direction), prioritize one of:

1. **If professor prefers Latency-Aware Routing (A)**: Implement Phase 12 — `RoutingPolicy` with latency-first rule. Compare two-tier routing (SS for <5s, Voting for ≥5s) against fixed-architecture baselines. Deliverable: routing performance table with expected P&L.

2. **If professor prefers Action Thresholding (B)**: Focus on confidence-gated abstention policy using Voting N=3 only. Compute optimal thresholds from cost-sensitive framework. Deliverable: PPV-operating curves with threshold recommendations.

3. **If professor prefers Statistical Validation (C)**: Scale Voting N=3 to N=50/domain (150 real API calls, ~$0.023). Deliverable: 95% confidence intervals on F1, PPV, and expected cost.

4. **If professor prefers RAG Redesign (D)**: Build a curated retrieval corpus with source reliability labels (FEVER-style) and contrastive examples. Test whether better evidence changes RAG's outcomes.

5. **If professor prefers Combination (E)**: Recommend A+B (routing + action thresholds) as the most impactful combined direction — they form a complete system (choose verifier → choose action).

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
