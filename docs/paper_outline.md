# Paper Outline

## 0. Abstract

## 1. Introduction

## 2. Problem Definition

### Input

- Events
- Other live but less credible sources (such as social media)
- Cached information (for less latency)
- Constraints
  - latency budget
  - misinformation base rate (calculated by historic frequency)
  - FP/FN cost ratio

### Output per LLM

- Real
- Fake
- Escalate
- Reasoning for human review

### Decision

- Real
- Fake
- Escalate

### Costs

- False positive cost
- False negative cost
- Latency cost

Different domains require different strategies.

## 3. Architecture

### Standalone Systems

#### Single-Shot

- One LLM call
- Lowest latency
- higher false positives
- can be confused when event is complicated

#### Voting N

- N voters
- If Escalate rate > E: escalate
- Otherwise dominated answer is the final output
- Less latency than MoA

#### MoA

- Multiple roles:
  - believer
  - skeptic
  - judge
- Purpose: To show contradiction
- Longest latency
- Best results when paired with RAG

### Routing

Requires historic data as arbitrary input

- Latency-first
  - Shortest possible: Single-Shot
  - Voting
- Cost first
  - MoA + RAG

## 4. Dataset

### Domains

- Finance
- Healthcare
- Political

### Analysis

- System qualities
- Base-rate analysis

### Limitations

- RAG is simulated based on historic data in this research

## 5. Experiment Results

### No-RAG comparison

| Architecture | F1 | Precision | Recall | FPR | Latency | Escalate |
|---|---:|---:|---:|---:|---:|---:|
| Single-Shot | 0.645 | 0.780 | 0.733 | 0.333 | 3.9s | 3% |
| Voting N=3 | 0.827 | 0.944 | 0.800 | 0.067 | 13.3s | 23% |
| MoA | 0.747 | 0.738 | 0.800 | 0.267 | 19.2s | 10% |

### Analysis

- Voting has best overall result
- Voting is most likely to trigger escalation
- MoA has worst latency with generic results

### RAG Impact

F1 scores:

| RAG | Single-Shot | Voting N=3 | MoA |
|---|---:|---:|---:|
| OFF | **0.645** | 0.827 | 0.747 |
| ON | 0.413 | **0.857** | **0.917** |

### Analysis

- Single-Shot is worse when combined with RAG
- MoA receives a large improvement with RAG

### Choices

- Single-Shot: ≤5s, tightest latency
- Voting: 11–16s, useful when slower decision allowed
- MoA+RAG: slow, better combined with RAG, useful for human review since it gives reasoning on different sides

## 6. Sensitivity Analysis

### Base-rate

### FP FN cost

- Change decision thresholds based on FP FN costs

### Escalation

- Requires human review: labor costs
- FP FN rate of escalation

## 7. Conclusion

## 8. References

Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H. (2023). Self-RAG: Learning to retrieve, generate, and critique through self-reflection. arXiv. https://arxiv.org/abs/2310.11511

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. Journal of Artificial Intelligence Research, 16, 321–357. https://doi.org/10.1613/jair.953

Chen, L., Zaharia, M., & Zou, J. (2023). FrugalGPT: How to use large language models while reducing cost and improving performance. arXiv. https://arxiv.org/abs/2305.05176

Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., & Mordatch, I. (2023). Improving factuality and reasoning in language models through multiagent debate. arXiv. https://arxiv.org/abs/2305.14325

Geifman, Y., & El-Yaniv, R. (2017). Selective classification for deep neural networks. arXiv. https://arxiv.org/abs/1705.08500

Hendrycks, D., & Gimpel, K. (2017). A baseline for detecting misclassified and out-of-distribution examples in neural networks. International Conference on Learning Representations. https://arxiv.org/abs/1610.02136

Irving, G., Christiano, P., & Amodei, D. (2018). AI safety via debate. arXiv. https://arxiv.org/abs/1805.00899

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W.-t., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. Advances in Neural Information Processing Systems, 33, 9459–9474. https://arxiv.org/abs/2005.11401

Schlichtkrull, M., Guo, Z., & Vlachos, A. (2023). AVeriTeC: A dataset for real-world claim verification with evidence from the web. arXiv. https://arxiv.org/abs/2305.13117

Thorne, J., Vlachos, A., Christodoulopoulos, C., & Mittal, A. (2018). FEVER: A large-scale dataset for fact extraction and verification. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (pp. 809–819). Association for Computational Linguistics. https://arxiv.org/abs/1803.05355

Wadden, D., Lin, S., Lo, K., Wang, L. L., van Zuylen, M., Cohan, A., & Hajishirzi, H. (2020). Fact or fiction: Verifying scientific claims. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (pp. 7534–7550). Association for Computational Linguistics. https://arxiv.org/abs/2004.14974

Wang, J., Wang, J., Athiwaratkun, B., Zhang, C., & Zou, J. (2024). Mixture-of-agents enhances large language model capabilities. arXiv. https://arxiv.org/abs/2406.04692

Wang, X., Wei, J., Schuurmans, D., Le, Q. V., Chi, E. H., Narang, S., Chowdhery, A., & Zhou, D. (2023). Self-consistency improves chain of thought reasoning in language models. International Conference on Learning Representations. https://arxiv.org/abs/2203.11171