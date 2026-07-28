# Operational Trade-offs in LLM-Based Misinformation Classification

Evaluates whether adding the complexity of an LLM inference system improves misinformation classification on finance and healthcare datasets. We compare Single-Shot, same-prompt majority Voting, and a role-based Mixture of Agents architectures with retrieval-augmented generation as the optional layer.

## Paper

[Read the paper](paper/Operational_Tradeoffs_in_LLM_Misinformation_Classification.pdf) (Yicheng Qiu, 2026).

All experiment design, datasets, models, and results are described in the paper.

## Quick Start

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

## Project Structure

```
src/       experiment code (entry point: final_1000_validation.py)
data/      finance and healthcare test sets + retrieval corpora
tests/     pytest tests for schemas and metrics
paper/     manuscript PDF
```

## License

The code is MIT-licensed. See [LICENSE](./LICENSE).
