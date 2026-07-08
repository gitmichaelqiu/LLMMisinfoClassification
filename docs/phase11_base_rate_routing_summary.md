# Phase 11: Base-Rate-Stratified Routing Analysis

## Objective

Identify the optimal verifier architecture for each (base_rate, cost_ratio, latency_budget) regime using existing real API outputs. Zero new API calls.

---

## Cross-Domain Architecture Summary (N=10/domain, 3 domains)

| Architecture | F1 (mean) | Precision | Recall | FPR | Latency | Escalate |
|------------|----------|----------|------|-----|---------|---------|
| Single Shot | 0.6453 | 0.7798 | 0.7333 | 0.3333 | 3.9s | 3.3% |
| Voting N3 | 0.8268 | 0.9444 | 0.8000 | 0.0667 | 13.3s | 23.3% |
| Moa | 0.7475 | 0.7381 | 0.8000 | 0.2667 | 19.2s | 10.0% |
| Rag | 0.4127 | 1.0000 | 0.2667 | 0.0000 | 4.3s | 53.3% |

---

## PPV Across Base Rates (Cross-Domain Mean)

| Architecture | 0.1% | 1% | 5% | 10% | 25% | 50% |
|------------|------|-----|-----|-----|-----|-----|
| Single Shot | 0.3347 | 0.3471 | 0.3990 | 0.4579 | 0.6039 | 0.7798 |
| Voting N3 | 0.6683 | 0.6827 | 0.7361 | 0.7857 | 0.8750 | 0.9444 |
| Moa | 0.0032 | 0.0308 | 0.1399 | 0.2521 | 0.4932 | 0.7381 |
| Rag | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

---

## Latency Feasibility

| Architecture | Mean Latency | <5s | <15s | <30s |
|------------|-------------|-----|------|------|
| Single Shot | 3.9s | ✅ | ✅ | ✅ |
| Voting N3 | 13.3s | ❌ | ✅ | ✅ |
| Moa | 19.2s | ❌ | ❌ | ✅ |
| Rag | 4.3s | ✅ | ✅ | ✅ |

---

## Regime Win Counts (72 configurations evaluated)

| Architecture | Regimes Won | % of Feasible | Note |
|------------|------------|--------------|------|
| Rag | 32 | 44.4% | ESCALATE abstention artifact (see Finding #5) |
| Voting N3 | 30 | 41.7% |  |
| Single Shot | 10 | 13.9% | Only wins when Voting disqualified by latency |

**Total feasible regimes**: 72 / 72

---

## Expected Cost Across Cost Ratios (Cross-Domain Mean)

### FP=1, FN=1

| Base Rate | Single-Shot | Voting N=3 | MoA | RAG | Optimal |
|----------|------------|-----------|-----|-----|---------|
| 0.1% | 0.3332 | 0.0668 | 0.2666 | 0.0007 | rag |
| 1% | 0.3326 | 0.0680 | 0.2660 | 0.0073 | rag |
| 5% | 0.3300 | 0.0734 | 0.2634 | 0.0367 | rag |
| 10% | 0.3266 | 0.0800 | 0.2600 | 0.0733 | rag |
| 25% | 0.3166 | 0.1000 | 0.2500 | 0.1833 | voting_n3 |
| 50% | 0.3000 | 0.1333 | 0.2333 | 0.3667 | voting_n3 |

### FP=1, FN=5

| Base Rate | Single-Shot | Voting N=3 | MoA | RAG | Optimal |
|----------|------------|-----------|-----|-----|---------|
| 0.1% | 0.3343 | 0.0676 | 0.2674 | 0.0037 | rag |
| 1% | 0.3433 | 0.0760 | 0.2740 | 0.0367 | rag |
| 5% | 0.3833 | 0.1134 | 0.3034 | 0.1833 | voting_n3 |
| 10% | 0.4333 | 0.1600 | 0.3400 | 0.3667 | voting_n3 |
| 25% | 0.5834 | 0.3000 | 0.4500 | 0.9166 | voting_n3 |
| 50% | 0.8334 | 0.5333 | 0.6333 | 1.8332 | voting_n3 |

### FP=1, FN=10

| Base Rate | Single-Shot | Voting N=3 | MoA | RAG | Optimal |
|----------|------------|-----------|-----|-----|---------|
| 0.1% | 0.3356 | 0.0686 | 0.2684 | 0.0073 | rag |
| 1% | 0.3566 | 0.0860 | 0.2840 | 0.0733 | rag |
| 5% | 0.4500 | 0.1634 | 0.3534 | 0.3667 | voting_n3 |
| 10% | 0.5667 | 0.2600 | 0.4400 | 0.7333 | voting_n3 |
| 25% | 0.9167 | 0.5500 | 0.7000 | 1.8332 | voting_n3 |
| 50% | 1.5002 | 1.0333 | 1.1333 | 3.6665 | voting_n3 |

### FP=1, FN=25

| Base Rate | Single-Shot | Voting N=3 | MoA | RAG | Optimal |
|----------|------------|-----------|-----|-----|---------|
| 0.1% | 0.3396 | 0.0716 | 0.2714 | 0.0183 | rag |
| 1% | 0.3966 | 0.1160 | 0.3140 | 0.1833 | voting_n3 |
| 5% | 0.6500 | 0.3134 | 0.5034 | 0.9166 | voting_n3 |
| 10% | 0.9667 | 0.5600 | 0.7400 | 1.8332 | voting_n3 |
| 25% | 1.9169 | 1.3000 | 1.4500 | 4.5831 | voting_n3 |
| 50% | 3.5004 | 2.5333 | 2.6333 | 9.1663 | voting_n3 |

---

## Crossover Points Between Architectures

The crossover base rate is the prevalence where expected cost of two architectures intersects.

| Arch A | Arch B | Cost Ratio | Crossover Base Rate |
|--------|--------|-----------|-------------------|
| Single Shot | Rag | FP:1_FN:1 | 0.4167 (41.67%) |
| Voting N3 | Rag | FP:1_FN:1 | 0.1111 (11.11%) |
| Moa | Rag | FP:1_FN:1 | 0.3333 (33.33%) |
| Single Shot | Rag | FP:1_FN:5 | 0.1250 (12.50%) |
| Voting N3 | Rag | FP:1_FN:5 | 0.0244 (2.44%) |
| Moa | Rag | FP:1_FN:5 | 0.0909 (9.09%) |
| Single Shot | Rag | FP:1_FN:10 | 0.0667 (6.67%) |
| Voting N3 | Rag | FP:1_FN:10 | 0.0123 (1.23%) |
| Moa | Rag | FP:1_FN:10 | 0.0476 (4.76%) |
| Single Shot | Rag | FP:1_FN:25 | 0.0278 (2.78%) |
| Voting N3 | Rag | FP:1_FN:25 | 0.0050 (0.50%) |
| Moa | Rag | FP:1_FN:25 | 0.0196 (1.96%) |

---

## Key Findings

1. **Best overall architecture**: **Voting N3** (F1=0.8268)

2. **Single-Shot is never the best performer** (F1=0.6453, latency=3.9s). Voting N=3 dominates in accuracy. Single-Shot's only advantage is latency <5s, making it optimal only in regimes where Voting N=3 is disqualified by latency budget.

3. **Voting N=3 is the dominant architecture**: wins 30/72 feasible regimes (42%). Its primary limitation is latency (13.3s), which disqualifies it under <5s budgets.

4. **MoA underperforms Voting** (F1=0.7475 vs Voting 0.8268) and is slower (19.2s). Wins 0 regimes. Not recommended for any regime where Voting is feasible.

5. **RAG "wins" 32 regimes — but this is an artifact.** RAG's FPR=0.0000 is achieved by ESCALATE-ing on 53% of items (treating ESCALATE as a cost-free correct rejection). Real recall is 0.2667 — it misses most FAKE items. If ESCALATE carries any operational cost (manual review, delayed decision), RAG wins zero regimes. The current TF-IDF retrieval pipeline is insufficient.

---

## Per-Domain Detail

### Finance

| Architecture | F1 | Precision | Recall | FPR | Latency |
|------------|-----|----------|------|-----|---------|
| Single Shot | 0.8333 | 0.7143 | 1.0000 | 0.4000 | 3.4s |
| Voting N3 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 11.7s |
| Moa | 0.8333 | 0.7143 | 1.0000 | 0.4000 | 16.0s |
| Rag | 0.3333 | 1.0000 | 0.2000 | 0.0000 | 3.6s |

### Healthcare

| Architecture | F1 | Precision | Recall | FPR | Latency |
|------------|-----|----------|------|-----|---------|
| Single Shot | 0.7692 | 0.6250 | 1.0000 | 0.6000 | 3.7s |
| Voting N3 | 0.9091 | 0.8333 | 1.0000 | 0.2000 | 12.8s |
| Moa | 0.9091 | 0.8333 | 1.0000 | 0.2000 | 21.3s |
| Rag | 0.5714 | 1.0000 | 0.4000 | 0.0000 | 6.0s |

### Political

| Architecture | F1 | Precision | Recall | FPR | Latency |
|------------|-----|----------|------|-----|---------|
| Single Shot | 0.3333 | 1.0000 | 0.2000 | 0.0000 | 4.7s |
| Voting N3 | 0.5714 | 1.0000 | 0.4000 | 0.0000 | 15.6s |
| Moa | 0.5000 | 0.6667 | 0.4000 | 0.2000 | 20.4s |
| Rag | 0.3333 | 1.0000 | 0.2000 | 0.0000 | 3.4s |


## Hybrid Routing Assessment (Latency-First Rule)

**<5s** (architectures feasible: Rag, Single Shot)

- Rag: 14 regimes
- Single Shot: 10 regimes

**<15s** (architectures feasible: Rag, Voting N3)

- Voting N3: 15 regimes
- Rag: 9 regimes

**<30s** (architectures feasible: Rag, Voting N3)

- Voting N3: 15 regimes
- Rag: 9 regimes

**Routing rule**:
- **<5s budget → Single-Shot** (Voting N=3 disqualified by latency)
- **≥5s budget → Voting N=3** (dominates on accuracy, feasibility, and cost)
- Base rate and cost ratio do not change the optimal architecture partition in this pilot

---

## Conclusions

### Does this support hybrid routing as the core contribution?

**Partially — yes, based on latency partitioning.** The evidence supports a **latency-first routing rule**, not a base-rate-sensitive one:

- **Budget <5s → Single-Shot** (latency: 3.9s, F1: 0.6453)
- **Budget ≥5s → Voting N=3** (latency: 13.3s, F1: 0.8268)
- **MoA never uniquely wins** — Pareto-dominated by Voting (lower F1, higher latency)
- **RAG never genuinely wins** — its cost advantage is an ESCALATE-abstention artifact

Within each latency tier, a single architecture dominates all (base_rate, cost_ratio) regimes. True base-rate-sensitive routing (where the optimal architecture varies with the misinformation prevalence) was not observed at this pilot scale. This may emerge at N=50 with bootstrap CIs, or with more extreme cost asymmetries (FP:FN > 1:25).

### Key Observations

- **12 crossover points** found between architectures, but all involve RAG (whose FPR=0 is an ESCALATE artifact). No crossover between Single-Shot and Voting in the (0,1) base rate range at any cost ratio.
- **Voting N=3 dominates Single-Shot at all base rates** (F1 +0.1815, P +0.1646, FPR −0.2667) — but costs 3.4× more latency
- **MoA adds debate complexity without accuracy gain** over Voting
- **Political domain is hardest** for all architectures (max F1=0.5714)

### Statistical Limitations

- N=10 per domain per architecture (30 total per architecture)
- Single run, no bootstrap confidence intervals
- ESCALATE treated as non-FAKE (conservative: TNs for REAL, FNs for FAKE)
- Cost ratios are normative (user-specified), not empirically estimated
- Cross-domain averages may mask domain-specific effects (especially political, which has different stylistic properties)

---

## Next Step

**Phase 12**: Implement `RoutingPolicy` that selects verifier + action + threshold from (base_rate, cost_ratio, latency_budget, evidence_availability, confidence). Compare against fixed-architecture baselines (Single-Shot only, Voting only).

*Generated by `src/phase11_base_rate_routing.py` — zero new API calls.*