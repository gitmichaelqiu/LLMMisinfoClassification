# Verification Arbitrage: LLM Risk Management for Fake-News Flash Crashes

A dual-system framework where classical NLP trades on breaking news ($T_0$) while an LLM verifies the news within seconds ($T_1$) to protect portfolio positions from fake-news flash crashes before human verification arrives ($T_2$).

**Core idea:** A fund already holds a position. Breaking news hits — is it real or fake? Classical ML panics and sells. An LLM with dual-source RAG evaluates source credibility within ~5 seconds. If the news is fake, the fund holds through the panic and recovers when human verification arrives ($T_2 \sim 300$s).

---

## Repository Structure

```
├── README.md              # This file
├── ROADMAP.md             # Phased implementation roadmap
├── requirements.txt       # Python dependencies
├── CLAUDE.md              # Project guide & architect roadmap
├── configs/               # Configuration files (YAML)
├── src/                   # Core source code
│   ├── rag_retriever.py       # Dual-source RAG (corpus + social stream)
│   ├── moa_agents.py          # Mixture-of-Agents (Believer, Skeptic, Risk Officer)
│   ├── cot_parser.py          # Chain-of-thought output parser
│   ├── prompts.py             # LLM prompt templates
│   ├── crypto_domain.py       # Cryptocurrency stress-test domain
│   ├── cross_lingual.py       # Cross-lingual generalization (Nikkei, DAX, Hang Seng)
│   ├── edgar_rag_retriever.py # SEC EDGAR filing RAG
│   ├── xbrl_verifier.py       # XBRL financial statement verification
│   ├── red_team_generator.py  # Adversarial red-team headline generation
│   ├── base_rate_analysis.py  # Bayesian PPV analysis
│   ├── health_dataset.py      # Healthcare domain evaluation
│   └── finance/               # Finance-specific modules
├── data/
│   ├── raw/               # Raw external datasets
│   ├── processed/         # Cleaned/featurized data
│   └── synthetic/         # Generated synthetic events
├── configs/               # Experiment configuration files
├── experiments/           # Experiment tracking
├── results/               # Output metrics and analysis
│   └── phase{01..06}/     # Versioned result directories
├── docs/                  # Documentation
├── tests/                 # Unit and integration tests
├── external_review_report/ # External audit artifacts
├── input/                 # Legacy input datasets
└── models/                # Pre-trained model artifacts
```

## Architecture

```
Data Layer → Dual RAG → LLM Verifier → Market Simulation → Analysis
  • Synthetic events    • News corpus     • Single-Shot     • Square-Root Impact     • Crossover curves
  • Historical hoaxes   • Social stream   • MoA Debate      • Dynamic Sizing         • PPV analysis
                        • Credibility     • Voting Ensemble • P&L Settlement          • Sensitivity
```

## Setup

```bash
git clone https://github.com/gitmichaelqiu/AdvFinNLPVuln.git
cd AdvFinNLPVuln
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and configure `DEEPSEEK_API_KEY`. In mock mode (no API key), the system falls back to trained classical baselines.

## Key Findings

| Metric | Value |
|--------|-------|
| LLM Recall (historical hoaxes) | 92% |
| LLM Precision (synthetic set) | 68% |
| Verify-First crossover threshold | 4.81% P(Fake) |
| T₁ latency correlation with P&L | r = -0.064 (negligible) |
| Dynamic sizing improvement | ~42× execution cost reduction |
| Opportunity-cost asymmetry | 25:1 (favoring verify-first) |

---

## License

MIT License. See [LICENSE](./LICENSE).

## Acknowledgements

- Financial news dataset (Apache 2.0) from [Kaggle](https://www.kaggle.com/datasets/mikemiller125/kaggleyahoo-finance-news)
- Fake news classification dataset (CC0 1.0) from [Kaggle](https://www.kaggle.com/datasets/mikemiller125/financial-news-classification-dataset)
