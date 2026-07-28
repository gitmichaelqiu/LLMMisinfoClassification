"""Configuration constants and paths for the verification experiment."""

import os

SEED = 42

MODEL = "deepseek-v4-flash"
TEMPERATURE = 0.7
MAX_TOKENS = 512
MAX_CONCURRENCY = 2000

OUTPUT_DIR = "results/final_1000_validation"
COST_PER_CALL = 0.00015

# Multi-model sensitivity analysis via OpenRouter
SENSITIVITY_MODELS = {
    "gpt-5.6-luna": {
        "display": "GPT-5.6 Luna",
        "openrouter_id": "openai/gpt-5.6-luna",
        "api_key_var": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
    },
    "glm-5.2": {
        "display": "GLM-5.2",
        "openrouter_id": "z-ai/glm-5.2",
        "api_key_var": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
    },
}

SENSITIVITY_OUTPUT_DIR = "results/model_sensitivity"
SENSITIVITY_TEST_SIZE = 50  # items per domain for the sensitivity sweep
SENSITIVITY_CONCURRENCY = 200  # cap for parallel API calls during sensitivity

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "raw_outputs"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)
os.makedirs(SENSITIVITY_OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(SENSITIVITY_OUTPUT_DIR, "raw_outputs"), exist_ok=True)

# PPV / cost-sensitivity sweep parameters
PPV_BASE_RATES = [0.001, 0.01, 0.05, 0.10, 0.25, 0.50]
COST_RATIOS = [(1, 1), (1, 5), (1, 10), (1, 25)]

# Data paths
FINANCE_TEST = "data/finance/finance_test_500.csv"
FINANCE_CORPUS = "data/finance/finance_corpus.csv"
COVID_TEST = "data/health/covid_test_500.csv"
COVID_CORPUS = "data/health/covid_corpus.csv"
