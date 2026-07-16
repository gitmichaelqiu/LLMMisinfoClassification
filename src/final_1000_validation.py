"""Entry point: run the full 1000-item verification experiment.

Usage:
    python -m src.final_1000_validation

Requires a valid DEEPSEEK_API_KEY in the environment or .env file.
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

from src.experiment import main

main()
