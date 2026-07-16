# Operational Trade-offs in LLM-Based Information Verification

Companion code for the paper evaluating Single-Shot, Self-Consistency Voting, and
Mixture-of-Agents architectures for LLM-based information verification across finance
and healthcare domains, with and without Retrieval-Augmented Generation.

## Paper Abstract

The volume of unverified information has grown faster than the human capacity to check it
with the rise of social media platforms. An automated information verification system can
reduce the human workload. The strong reasoning power of Large Language Models (LLMs)
makes them a promising tool for fact-checking. In this paper, we evaluate Single-Shot,
Same-Prompt Majority Voting, and a role-based Mixture of Agents architectures, each with
and without Retrieval-Augmented Generation, on 500 claims each in finance and healthcare.
Each architecture generates a final label of REAL, FAKE, and ESCALATE. The performance is
analyzed using metrics of F1, precision, recall, false positive rate, ESCALATE rate, latency,
and computational cost.

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
│   ├── final_1000_validation.py           # Main experiment script (all 12 architectures × 2 domains)
│   ├── final_1000_validation_continued.py # Continuation for remaining experiments
│   ├── finance/
│   │   └── finance_dataset_adapter.py     # Finance data loader
│   └── healthcare/
│       └── health_dataset_adapter.py      # Healthcare data loader
├── data/raw/
│   ├── finance/                           # Finance test (500) and corpus (2756)
│   └── health/                            # Healthcare test (500) and corpus (7866)
├── tests/
│   ├── conftest.py                        # Shared fixtures
│   ├── test_schemas.py                    # Schema validation tests
│   └── test_metrics.py                    # Metrics correctness tests
├── results/final_1000_validation/         # Generated experiment outputs (gitignored)
├── requirements.txt                       # Python dependencies
├── pyproject.toml                         # Project metadata
├── Makefile                               # Common targets: install, test, lint, clean
└── .env.example                           # API key template
```

## Setup

```bash
git clone https://github.com/gitmichaelqiu/AdvFinNLPVuln.git
cd AdvFinNLPVuln
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set your `DEEPSEEK_API_KEY`.

```bash
cp .env.example .env
# Edit .env with your API key
```

## Reproducing Experiments

### Main experiment (warning: costs ~$6–8 in API fees)

```bash
python -m src.final_1000_validation
```

This runs all 12 architecture × RAG configurations on both domains
(500 finance + 500 healthcare items) using DeepSeek-v4-flash.

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

- **Finance**: ISOT Fake News Dataset, filtered to 3,324 economics-keyword articles.
  Balanced test set of 500 (250 REAL, 250 FAKE). Retrieval pool: 2,756 articles.
- **Healthcare**: COVID-19 Fake News Dataset NLP (CONSTRAINT), 6,420 train + 2,140 val.
  Balanced test set of 500 (250 REAL, 250 FAKE). Retrieval pool: 7,866 articles.

Both test sets are balanced at 50% prevalence. Retrieval pools exclude near-duplicates
(TF-IDF cosine similarity > 0.8 to any test item).

## Model

All experiments use `deepseek-v4-flash` via `api.deepseek.com/v1` with:
- Temperature: 0.7
- Max tokens: 512
- July 2026 API endpoint

## Key Results

| Architecture | Finance F1 | Healthcare F1 | Mean ESC Rate | Latency |
|---|---|---|---|---|
| Single-Shot (RAG OFF) | 0.689 | 0.779 | 4.9% | ~4.6 s |
| Voting N=7 (RAG OFF) | 0.643 | 0.794 | 19.7% | ~5.5 s |
| Voting N=7 (RAG ON) | 0.803 | 0.765 | 31.9% | ~5.5 s |
| MoA (RAG OFF) | 0.654 | 0.758 | 12.7% | ~10.6 s |
| MoA (RAG ON) | 0.721 | 0.745 | 12.6% | ~10.9 s |

## License

MIT License. See [LICENSE](./LICENSE).
