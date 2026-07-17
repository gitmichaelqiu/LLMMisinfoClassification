"""Per-call JSONL persistence for reliability and recovery.

``CallRecorder`` saves every individual API call to an append-only JSONL file
so that if the process is interrupted mid-run, no completed calls are lost and
the run can resume from where it left off.

File layout
-----------
For each ``(domain, model, architecture)`` cell two files exist under
``<output_dir>/raw_outputs/``::

    {domain}_{model_slug}_{arch}.calls.jsonl   # one line per API call
    {domain}_{model_slug}_{arch}.jsonl          # one line per final result (per item)

The ``.calls`` file stores the full raw prompt / response for later analysis
(error taxonomy, RAG failure analysis, etc.).  The results file stores only the
final ``VerificationResult`` and is what drives recovery.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from typing import Any

from src.schemas import Verdict, VerificationResult


class CallRecorder:
    """Thread-safe recorder that persists every API call and final result.

    All writes are append-only and flushed immediately so partial data is
    never lost.
    """

    def __init__(self, output_dir: str) -> None:
        self._output_dir = output_dir
        self._raw_dir = os.path.join(output_dir, "raw_outputs")
        os.makedirs(self._raw_dir, exist_ok=True)
        # per-file locks for thread safety
        self._locks: dict[str, threading.Lock] = {}
        self._lock_for_path: threading.Lock = threading.Lock()

    # ── internal helpers ────────────────────────────────────────────

    def _paths(self, domain: str, model: str, architecture: str) -> tuple[str, str]:
        """Return ``(results_path, calls_path)``."""
        stem = f"{domain}_{model}_{architecture}"
        return (
            os.path.join(self._raw_dir, f"{stem}.jsonl"),
            os.path.join(self._raw_dir, f"{stem}.calls.jsonl"),
        )

    def _lock(self, path: str) -> threading.Lock:
        with self._lock_for_path:
            if path not in self._locks:
                self._locks[path] = threading.Lock()
            return self._locks[path]

    def _append(self, path: str, record: dict) -> None:
        """Thread-safe append of one JSON line with immediate flush."""
        with self._lock(path):
            with open(path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
                f.flush()

    # ── recording ───────────────────────────────────────────────────

    def record_call(
        self,
        domain: str,
        model: str,
        architecture: str,
        item_id: str,
        system_prompt: str,
        user_prompt: str,
        raw_response: str,
        verdict_name: str,
        confidence: float,
        latency_s: float,
        usage: dict[str, int | None] | None = None,
        voter_idx: int | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save one individual API call (raw prompts, response, parse result).

        This is written to the ``.calls.jsonl`` file for later deep analysis.
        """
        _, calls_path = self._paths(domain, model, architecture)
        record: dict[str, Any] = {
            "type": "call",
            "item_id": item_id,
            "model": model,
            "architecture": architecture,
            "domain": domain,
            "timestamp": datetime.now().isoformat(),
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "raw_response": raw_response,
            "parsed": {"verdict": verdict_name, "confidence": confidence},
            "latency_s": latency_s,
            "usage": usage or {},
        }
        if voter_idx is not None:
            record["voter_idx"] = voter_idx
        if extra_metadata:
            record["extra"] = extra_metadata
        self._append(calls_path, record)

    def record_result(
        self,
        domain: str,
        model: str,
        architecture: str,
        item_id: str,
        verdict_name: str,
        confidence: float,
        latency_s: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save the final verification result for one item.

        This is written to the main ``.jsonl`` file and is the authoritative
        record of what was decided for this item.  Recovery checks this file.
        """
        results_path, _ = self._paths(domain, model, architecture)
        record: dict[str, Any] = {
            "type": "result",
            "item_id": item_id,
            "model": model,
            "architecture": architecture,
            "domain": domain,
            "timestamp": datetime.now().isoformat(),
            "verdict": verdict_name,
            "confidence": confidence,
            "latency_s": latency_s,
            "metadata": metadata or {},
        }
        self._append(results_path, record)

    # ── recovery / loading ──────────────────────────────────────────

    def completed_item_ids(
        self, domain: str, model: str, architecture: str
    ) -> set[str]:
        """Return the set of item IDs that have a saved final result.

        Used on restart to skip already-completed items.
        """
        results_path, _ = self._paths(domain, model, architecture)
        if not os.path.exists(results_path):
            return set()
        ids: set[str] = set()
        with open(results_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("type") == "result":
                        ids.add(rec["item_id"])
                except json.JSONDecodeError:
                    continue  # ignore corrupt trailing lines
        return ids

    def load_results(
        self, domain: str, model: str, architecture: str
    ) -> list[VerificationResult]:
        """Reconstruct a list of ``VerificationResult`` from saved results.

        Results are returned in the order they were written (roughly item
        completion order).  Use this to recompute metrics after a full run.
        """
        results_path, _ = self._paths(domain, model, architecture)
        if not os.path.exists(results_path):
            return []
        results: list[VerificationResult] = []
        with open(results_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("type") != "result":
                        continue
                    results.append(
                        VerificationResult(
                            item_id=rec["item_id"],
                            verdict=Verdict[rec["verdict"]],
                            confidence=rec["confidence"],
                            latency_s=rec["latency_s"],
                            evidence=[],
                            metadata=rec.get("metadata", {}),
                        )
                    )
                except (json.JSONDecodeError, KeyError):
                    continue
        return results

    def load_call_details(
        self, domain: str, model: str, arch: str
    ) -> list[dict[str, Any]]:
        """Load all raw call records for deep analysis (RAG error taxonomy,
        output format checks, etc.)."""
        _, calls_path = self._paths(domain, model, arch)
        if not os.path.exists(calls_path):
            return []
        records: list[dict[str, Any]] = []
        with open(calls_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    # ── cost tracking ───────────────────────────────────────────────

    def token_counts(
        self, domain: str, model: str, architecture: str
    ) -> tuple[int, int]:
        """Return ``(total_input_tokens, total_output_tokens)`` from saved calls."""
        calls = self.load_call_details(domain, model, architecture)
        total_input = 0
        total_output = 0
        for c in calls:
            u = c.get("usage", {}) or {}
            total_input += u.get("prompt_tokens", 0) or 0
            total_output += u.get("completion_tokens", 0) or 0
        return total_input, total_output
