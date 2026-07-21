# Operational Trade-offs in LLM-Based Misinformation Classification

Evaluates Single-Shot, Self-Consistency Voting, and Mixture-of-Agents architectures
for LLM-based information verification across finance and healthcare domains,
with and without Retrieval-Augmented Generation.

## Architectures

| Architecture | Description | API Calls per Claim | Latency |
|---|---|---|---|
| **Single-Shot** | One LLM call with canonical prompt | 1× | ~4.5 s |
| **Voting (N)** | N parallel calls, majority threshold = ⌊N/2⌋+1 | N× | ~4.5–5.5 s |
| **MoA** | Supporter + Skeptic (concurrent) → Judge (sequential) | 3× | ~10–12 s |

Each architecture is tested with RAG OFF and RAG ON (TF-IDF retriever, top-5, cosine
similarity). For Voting, 7 outputs are generated once per item, then subsets of the first
N ∈ {1, 3, 5, 7} voters are aggregated.

## Repository Structure

```
├── src/
│   ├── schemas.py                         # Data structures: Verdict, VerificationItem, etc.
│   ├── metrics.py                         # Confusion matrix, F1, precision, recall, FPR, etc.
│   ├── experiment.py                      # Main experiment orchestrator
│   ├── final_1000_validation.py           # Entry point (DeepSeek, all architectures × 2 domains)
│   ├── sensitivity.py                     # Multi-model sensitivity analysis via OpenRouter
│   ├── final_1000_validation_sensitivity.py # Entry point for sensitivity analysis
│   ├── data.py                            # CSV data loading
│   ├── architectures.py                   # Architecture runners: Single-Shot, Voting, MoA
│   ├── evaluation.py                      # Metrics, bootstrap CIs, PPV, voter analysis
│   ├── api.py                             # LLM API interaction layer
│   ├── prompts.py                         # System/user prompts for all architectures
│   ├── retrieval.py                       # TF-IDF retriever for RAG
│   ├── figures.py                         # Figure generation
│   ├── reporting.py                       # Console reporting and output persistence
│   ├── storage.py                         # Per-call JSONL persistence (sensitivity)
│   └── config.py                          # Paths, model settings, constants
├── data/
│   ├── finance/                           # Finance test (500) and corpus (2756)
│   └── health/                            # Healthcare test (500) and corpus (7866)
├── tests/
│   ├── conftest.py                        # Shared fixtures
│   ├── test_schemas.py                    # Schema validation tests
│   └── test_metrics.py                    # Metrics correctness tests
├── requirements.txt                       # Python dependencies
├── pyproject.toml                         # Project metadata
├── Makefile                               # Common targets: install, test, lint, clean
└── .env.example                           # API key template
```

## Setup

```bash
git clone https://github.com/gitmichaelqiu/LLMMisinfoClassification.git
cd LLMMisinfoClassification
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set your API keys.

```bash
cp .env.example .env
# Edit .env with your API keys
```

## Running the Experiment

Warning: the full experiment runs ~14,000 API calls and costs approximately $6–8.

```bash
python -m src.final_1000_validation
```

This runs all 12 architecture × RAG configurations on both domains
(500 finance + 500 healthcare items) using DeepSeek-v4-flash.

### Sensitivity Analysis

To reproduce the multi-model comparison across GPT-5.6 Luna and GLM-5.2
via OpenRouter (requires `OPENROUTER_API_KEY`):

```bash
python -m src.final_1000_validation_sensitivity
```

This tests a stratified 50-item sample per domain on all architectures
(Single-Shot, Voting N=1/3/5/7, MoA) with and without RAG. Results are
written to `results/model_sensitivity/`.

### Outputs

Results are written to `results/final_1000_validation/`:
- `raw_outputs/` — per-architecture JSON files with per-item verdicts
- `figures/` — F1 comparison, PPV curves, RAG effect plots
- `report.json` — consolidated metrics and confidence intervals

### Running tests

```bash
make test
```

## Datasets

The experiment uses four pre-processed CSV files tracked directly in the repository:

| File | Domain | Contents | Rows | Size |
|---|---|---|---|---|
| `data/finance/finance_test_500.csv` | Finance | Balanced test set (250 REAL, 250 FAKE) | 500 | 1.3 MB |
| `data/finance/finance_corpus.csv` | Finance | TF-IDF retrieval pool (no test-set near-duplicates) | 2,756 | 6.9 MB |
| `data/health/covid_test_500.csv` | Healthcare | Balanced test set (250 REAL, 250 FAKE) | 500 | 94 KB |
| `data/health/covid_corpus.csv` | Healthcare | TF-IDF retrieval pool (no test-set near-duplicates) | 7,866 | 1.5 MB |

Both test sets are balanced at 50% prevalence. Retrieval pools exclude near-duplicates
(TF-IDF cosine similarity &gt; 0.8 to any test item).

### Source Provenance

- **Finance**: Derived from the [ISOT Fake News Dataset](https://www.kaggle.com/datasets/rahulogoel/isot-fake-news-dataset) under [MIT License](./licenses/LICENSE-ISOTFakeNewsDataset.txt).
  The original 44,898 articles were filtered to 3,324 economics-keyword articles, then
  deduplicated and split into a balanced 500-item test set and a 2,756-item retrieval corpus.
- **Healthcare**: Derived from the [COVID19 Fake News Dataset NLP](https://www.kaggle.com/datasets/elvinagammed/covid19-fake-news-dataset-nlp) under [MIT License](./licenses/LICENSE-COVID19FakeNewsDatasetNLP.txt).
  The 6,420 training and 2,140 validation tweets were deduplicated and split into a
  balanced 500-item test set and a 7,866-item retrieval corpus.

Original source files (`data/finance/financial_news.csv`,
`data/health/Constraint_*.csv`, etc.) are **not** tracked in git.
The processed files above are tracked directly so that a `git clone` immediately
yields reproduction-ready inputs.

## Models

**Primary experiment**: `deepseek-v4-flash` via `api.deepseek.com/v1` with:
- Temperature: 0.7
- Max tokens: 512

**Sensitivity analysis** (via OpenRouter):
- `openai/gpt-5.6-luna` (GPT-5.6 Luna)
- `z-ai/glm-5.2` (GLM-5.2)

## Key Results (DeepSeek v4 Flash, 500 items/domain)

| Architecture | Finance F1 | Healthcare F1 | Mean ESC Rate | Latency |
|---|---|---|---|---|
| Single-Shot (RAG OFF) | 0.689 | 0.779 | 4.9% | ~4.6 s |
| Voting N=7 (RAG OFF) | 0.643 | 0.794 | 19.7% | ~5.5 s |
| Voting N=7 (RAG ON) | 0.803 | 0.765 | 31.9% | ~5.5 s |
| MoA (RAG OFF) | 0.654 | 0.758 | 12.7% | ~10.6 s |
| MoA (RAG ON) | 0.721 | 0.745 | 12.6% | ~10.9 s |

## License

MIT License. See [LICENSE](./LICENSE).
