"""Entry point: multi-model sensitivity analysis via OpenRouter.

Usage:
    python -m src.final_1000_validation_sensitivity

Requires OPENROUTER_API_KEY in the environment or .env file.
"""

from __future__ import annotations

import os

# Strip proxy settings that can interfere with the API client
for _var in (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "http_proxy",
    "https_proxy",
    "ALL_PROXY",
    "all_proxy",
):
    os.environ.pop(_var, None)
os.environ["no_proxy"] = "*"
os.environ["NO_PROXY"] = "*"

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from src.sensitivity import run_sensitivity_analysis  # noqa: E402

run_sensitivity_analysis()
