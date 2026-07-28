# Operational Trade-offs in LLM-Based Misinformation Classification

Evaluates Single-Shot, Self-Consistency Voting, and Mixture-of-Agents
architectures for LLM-based information verification across finance and
healthcare domains, with and without Retrieval-Augmented Generation.

## Paper

[Read the paper](paper/Operational_Tradeoffs_in_LLM_Misinformation_Classification.pdf) —
student research paper completed through the Pioneer Research Program
(Yicheng Qiu, mentor Sanjay Ranka). Not peer-reviewed or formally published.

All experiment design, datasets, models, and results are described in the paper.

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env   # add your DeepSeek API key
python -m src.final_1000_validation
```

Tests, linting, and a clean target are in the `Makefile`:

```bash
make test
make lint
make clean
```

## Project structure

```
src/       experiment code (entry point: final_1000_validation.py)
data/      finance and healthcare test sets + retrieval corpora
tests/     pytest tests for schemas and metrics
paper/     manuscript PDF
```

## License

The code is MIT-licensed. See [LICENSE](./LICENSE). The manuscript PDF
is provided for reading and citation — all rights reserved.
