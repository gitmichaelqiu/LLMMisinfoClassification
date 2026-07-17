"""LLM API interaction layer: call, parse, and parallel dispatch."""

from __future__ import annotations

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from src.config import MODEL, TEMPERATURE, MAX_TOKENS, MAX_CONCURRENCY
from src.schemas import Verdict, VerificationResult

# ── Response parsing ────────────────────────────────────────────
_VERDICT_RE = re.compile(r"Verdict:\s*(REAL|FAKE|ESCALATE|EXAGGERATED)", re.IGNORECASE)
_CONFIDENCE_RE = re.compile(r"Confidence:\s*(\d+)", re.IGNORECASE)
_VERDICT_MAP = {
    "REAL": Verdict.REAL,
    "FAKE": Verdict.FAKE,
    "ESCALATE": Verdict.ESCALATE,
    "EXAGGERATED": Verdict.EXAGGERATED,
}


def _parse_response(
    raw: str,
    item_id: str,
    latency_s: float,
    extra_metadata: dict | None = None,
) -> VerificationResult:
    """Extract verdict, confidence, and metadata from an LLM response string."""
    verdict = Verdict.REAL
    vm = _VERDICT_RE.search(raw)
    if vm:
        verdict = _VERDICT_MAP.get(vm.group(1).upper(), Verdict.REAL)

    confidence = 0.5
    cm = _CONFIDENCE_RE.search(raw)
    if cm:
        confidence = int(cm.group(1)) / 100.0

    return VerificationResult(
        item_id=item_id,
        verdict=verdict,
        confidence=max(0.0, min(1.0, confidence)),
        latency_s=latency_s,
        evidence=[],
        metadata=extra_metadata or {},
    )


# ── LLM call ────────────────────────────────────────────────────
def _llm_call(
    system_prompt: str,
    user_prompt: str,
    model: str = MODEL,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
    api_key: str | None = None,
    base_url: str | None = None,
) -> tuple[str, float, dict]:
    """Make a single LLM API call.

    Returns ``(response_text, latency_s, usage_dict)`` where ``usage_dict``
    contains ``prompt_tokens``, ``completion_tokens``, ``total_tokens``
    (or empty dict if unavailable).

    When *api_key* and *base_url* are ``None``, defaults to DeepSeek V4 Flash.
    """
    import httpx
    from openai import OpenAI

    resolved_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    resolved_url = base_url or "https://api.deepseek.com/v1"
    http_client = httpx.Client(
        proxy=None,
        timeout=httpx.Timeout(180.0, connect=30.0),
        follow_redirects=True,
    )
    client = OpenAI(
        api_key=resolved_key,
        base_url=resolved_url,
        http_client=http_client,
        default_headers={
            "HTTP-Referer": "https://github.com/gitmichaelqiu/AdvFinNLPVuln",
            "X-Title": "AdvFinNLPVuln",
        },
    )
    start = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = resp.choices[0].message.content or ""
        latency = time.time() - start
        usage = (
            {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            }
            if resp.usage
            else {}
        )
        return text, latency, usage
    except Exception as e:
        raise RuntimeError(f"API call failed: {e}")
    finally:
        http_client.close()


# ── Parallel dispatch ────────────────────────────────────────────
def _run_parallel(
    callables: list[Callable[[], Any]],
    desc: str = "",
    max_workers: int | None = None,
) -> list[Any | None]:
    """Execute a list of nullary callables concurrently.

    *max_workers* caps the thread pool; defaults to ``MAX_CONCURRENCY`` from
    config when ``None``.

    Returns results in the same order as *callables*; a failed call
    produces ``None`` in its slot.
    """
    n = len(callables)
    cap = max_workers if max_workers is not None else MAX_CONCURRENCY
    results: list[Any | None] = [None] * n
    with ThreadPoolExecutor(max_workers=min(cap, n)) as ex:
        fut_to_idx = {ex.submit(fn): i for i, fn in enumerate(callables)}
        for fut in as_completed(fut_to_idx):
            idx = fut_to_idx[fut]
            try:
                results[idx] = fut.result()
            except Exception:
                results[idx] = None
    return results
