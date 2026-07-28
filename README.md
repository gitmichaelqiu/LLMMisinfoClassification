# Operational Trade-offs in LLM-Based Misinformation Classification

Evaluates Single-Shot, Self-Consistency Voting, and Mixture-of-Agents
architectures for LLM-based information verification across finance and
healthcare domains, with and without Retrieval-Augmented Generation.

## Paper

[Read the paper](paper/Operational_Tradeoffs_in_LLM_Misinformation_Classification.pdf) —
student research paper completed through the Pioneer Research Program
(Yicheng Qiu, mentor Sanjay Ranka). Not peer-reviewed or formally published.

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env   # add your API keys
python -m src.final_1000_validation
```

The full experiment runs ~14,000 API calls (≈$6–8 on DeepSeek v4 Flash).

For the multi-model sensitivity analysis via OpenRouter:

```bash
python -m src.final_1000_validation_sensitivity
```

Tests, linting, and a clean target are in the `Makefile`:

```bash
make test
make lint
make clean
```

## Architectures

| Architecture | How it works | API calls / claim | Latency |
|---|---|---|---|
| **Single-Shot** | One LLM call, canonical prompt | 1× | ~4.6 s |
| **Voting (N)** | N parallel calls, majority threshold = ⌊N/2⌋+1 | N× | ~5.0 s |
| **MoA** | Supporter + Skeptic (concurrent) → Judge (sequential) | 3× | ~10.6 s |

Each architecture is tested with RAG off and on (TF-IDF retriever, top-5,
cosine similarity). For Voting, 7 outputs are generated per item so that
subsets for N ∈ {1, 3, 5, 7} can be compared without extra API calls.

## Project structure

```
src/              # everything: data loading, architectures, metrics, figures
data/             # finance and healthcare test sets + retrieval corpora
tests/            # pytest tests for schemas and metrics
paper/            # manuscript PDF
```

Source modules:

- `schemas.py`, `metrics.py` — data structures and evaluation metrics
- `experiment.py` + `final_1000_validation.py` — main experiment entry point
- `sensitivity.py` + `final_1000_validation_sensitivity.py` — multi-model sweep
- `architectures.py` — Single-Shot, Voting, and MoA runners
- `api.py` — LLM API interaction (parallel dispatch, response parsing)
- `prompts.py`, `retrieval.py` — prompt templates and TF-IDF RAG retriever
- `evaluation.py`, `reporting.py`, `figures.py` — results analysis and output
- `config.py`, `storage.py` — paths, constants, call persistence

## Datasets

Four pre-processed CSV files are tracked in the repository:

| File | Domain | Rows | Use |
|---|---|---|---|
| `data/finance/finance_test_500.csv` | Finance | 500 | Balanced test set (250 REAL, 250 FAKE) |
| `data/finance/finance_corpus.csv` | Finance | 2,756 | TF-IDF retrieval pool |
| `data/health/covid_test_500.csv` | Healthcare | 500 | Balanced test set (250 REAL, 250 FAKE) |
| `data/health/covid_corpus.csv` | Healthcare | 7,866 | TF-IDF retrieval pool |

Both test sets are balanced at 50% prevalence. Retrieval pools exclude
near-duplicates (TF-IDF cosine similarity > 0.8 to any test item).

- **Finance**: derived from the [ISOT Fake News Dataset](https://www.kaggle.com/datasets/rahulogoel/isot-fake-news-dataset)
  ([MIT](./licenses/LICENSE-ISOTFakeNewsDataset.txt)). Filtered to 3,324
  economics-keyword articles, then deduplicated and split.
- **Healthcare**: derived from the [COVID19 Fake News Dataset NLP](https://www.kaggle.com/datasets/elvinagammed/covid19-fake-news-dataset-nlp)
  ([MIT](./licenses/LICENSE-COVID19FakeNewsDatasetNLP.txt)). Deduplicated
  and split from the Constrain_Train and Constrain_Val sets.

## Models

- **Primary experiment**: `deepseek-v4-flash`, temperature 0.7, max 512 tokens.
- **Sensitivity analysis**: compares DeepSeek v4 Flash against
  `openai/gpt-5.6-luna` and `z-ai/glm-5.2` via OpenRouter on a
  stratified 50-item-per-domain subset.

## Key results (Table 8: Overall Comparison)

| Architecture | Finance F1 | Healthcare F1 | Mean F1 | Mean ESC Rate | Latency |
|---|---|---|---|---|---|
| Single-Shot (RAG OFF) | 0.689 | 0.779 | 0.734 | 4.9% | ~4.6 s |
| Single-Shot (RAG ON) | 0.716 | 0.729 | 0.722 | 12.1% | ~4.7 s |
| Voting N=7 (RAG OFF) | 0.643 | 0.794 | 0.719 | 19.7% | ~5.0 s |
| Voting N=7 (RAG ON) | 0.803 | 0.765 | 0.784 | 31.9% | ~5.1 s |
| MoA (RAG OFF) | 0.654 | 0.758 | 0.706 | 12.7% | ~10.6 s |
| MoA (RAG ON) | 0.721 | 0.744 | 0.733 | 12.6% | ~10.9 s |

## License

The code is MIT-licensed. See [LICENSE](./LICENSE). The manuscript PDF
is provided for reading and citation — all rights reserved.
