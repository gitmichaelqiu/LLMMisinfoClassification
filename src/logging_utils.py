"""Structured experiment logging and result persistence.

Provides ExperimentLogger for consistent JSON-serialized experiment records
with git commit tracking and artifact path linking.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _get_git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _get_git_branch() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


class ExperimentLogger:
    """Logs experiment runs to structured JSON files.

    Each experiment is saved as a separate JSON file with full metadata
    for traceability.

    Usage:
        logger = ExperimentLogger(output_dir="results/phase03")
        logger.log_run(
            config={"model": "gpt-4o-mini", "test_size": 50},
            metrics={"precision": 0.85, "recall": 0.92},
            artifact_paths={"confusion_matrix": "results/phase03/cm.json"},
        )
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def log_run(
        self,
        config: Dict[str, Any],
        metrics: Dict[str, Any],
        artifact_paths: Optional[Dict[str, str]] = None,
        notes: Optional[str] = None,
    ) -> str:
        """Log a single experiment run.

        Args:
            config: Configuration parameters for this run.
            metrics: Metrics computed from this run.
            artifact_paths: Paths to generated artifacts (plots, data files).
            notes: Free-text notes.

        Returns:
            The run ID (timestamp-based).
        """
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        record = {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_commit": _get_git_commit(),
            "git_branch": _get_git_branch(),
            "config": config,
            "metrics": metrics,
            "artifact_paths": artifact_paths or {},
            "notes": notes or "",
        }
        path = os.path.join(self.output_dir, f"{run_id}.json")
        with open(path, "w") as f:
            json.dump(record, f, indent=2, default=str)
        return run_id

    def log_result_list(
        self,
        name: str,
        results: List[Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Log a list of structured result records (e.g., per-item verdicts).

        Args:
            name: Logical name for the result set.
            results: List of dict-like result records.
            config: Optional config metadata.

        Returns:
            The run ID.
        """
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        record = {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_commit": _get_git_commit(),
            "git_branch": _get_git_branch(),
            "name": name,
            "config": config or {},
            "results": results,
        }
        path = os.path.join(self.output_dir, f"{name}_{run_id}.json")
        with open(path, "w") as f:
            json.dump(record, f, indent=2, default=str)
        return run_id
