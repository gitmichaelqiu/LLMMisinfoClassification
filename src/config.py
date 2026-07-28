"""Configuration constants and paths for the verification experiment."""

import os

SEED = 42

MODEL = "deepseek-v4-flash"
TEMPERATURE = 0.7
MAX_CONCURRENCY = 2000

OUTPUT_DIR = "results/final_1000_validation"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "raw_outputs"), exist_ok=True)

# Data paths
FINANCE_TEST = "data/finance/finance_test_500.csv"
FINANCE_CORPUS = "data/finance/finance_corpus.csv"
COVID_TEST = "data/health/covid_test_500.csv"
COVID_CORPUS = "data/health/covid_corpus.csv"
