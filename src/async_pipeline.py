"""Async dual-system pipeline with P&L calculator and Ollama client.

Phase 4: latency-optimized concurrent execution of System 1 (FinBERT)
and System 2 (RAG + LLM) with configurable latency budget, P&L
quantification, and local LLM support via Ollama.
"""

import os
import time
import json
import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from src.cot_parser import parse_cot_output
from src.prompts import COT_RAG_SYSTEM_PROMPT, COT_RAG_USER_PROMPT, NON_RAG_SYSTEM_PROMPT


class PnLCalculator:
    """Compute P&L saved by intervening in a flash crash."""

    def __init__(self, position_size=1000, base_price=190.0):
        self.position_size = position_size
        self.base_price = base_price
        self.trough_price = 150.0
        self.trough_time_ms = 3000.0

    def intervention_price(self, t_ms):
        """Price at intervention time using flash crash model (same as MarketSimulator)."""
        if t_ms is None or t_ms < 0:
            return self.base_price
        if t_ms <= self.trough_time_ms:
            return self.base_price - (t_ms / self.trough_time_ms) * 40.0
        elif t_ms <= 10000:
            return self.trough_price + ((t_ms - self.trough_time_ms) / 7000.0) * 40.0
        return self.base_price

    def compute(self, intervention_time_ms, fake_detected=True):
        """P&L saved = position * (price_at_intervention - trough_price).

        Returns $ saved. 0 if no intervention (missed detection).
        """
        if not fake_detected or intervention_time_ms is None:
            return 0.0
        p_intervention = self.intervention_price(intervention_time_ms)
        saved = self.position_size * (p_intervention - self.trough_price)
        return max(0.0, saved)

    def max_loss(self):
        """Maximum possible loss if no intervention."""
        return self.position_size * (self.base_price - self.trough_price)


class OllamaClient:
    """Client for local LLM inference via Ollama API."""

    def __init__(self, model="qwen3.5:2b", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def query(self, prompt, system_prompt=None):
        """Call Ollama API, return (response_text, latency_ms)."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 20},
        }
        if system_prompt:
            payload["system"] = system_prompt

        t0 = time.time()
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60.0,
            )
            elapsed = (time.time() - t0) * 1000
            if resp.status_code != 200:
                return "", elapsed
            data = resp.json()
            return data.get("response", ""), elapsed
        except Exception:
            return "", (time.time() - t0) * 1000

    def unload(self):
        """Unload model from Ollama memory."""
        try:
            requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "keep_alive": 0},
                timeout=5.0,
            )
        except Exception:
            pass


class AsyncDualPipeline:
    """Concurrent System 1 (FinBERT) + System 2 (RAG + LLM) execution.

    System 1 fires immediately (~50ms FinBERT sentiment).
    System 2 runs in parallel thread (RAG retrieval + LLM inference).
    If System 2 returns FAKE within latency budget, compute P&L saved.
    If System 2 exceeds budget, trade stands with System 1 signal only.
    """

    def __init__(
        self,
        finbert_model=None,
        rag_retriever=None,
        deepseek_client=None,
        deepseek_key=None,
        latency_budget_ms=2000,
        model="deepseek",
        ollama_model="qwen3.5:2b",
        position_size=1000,
        base_price=190.0,
    ):
        self.finbert = finbert_model
        self.rag_retriever = rag_retriever
        self.client = deepseek_client
        self.deepseek_key = deepseek_key
        self.latency_budget_ms = latency_budget_ms
        self.model = model
        self.ollama_model = ollama_model
        self.pnl = PnLCalculator(position_size, base_price)
        self.executor = ThreadPoolExecutor(max_workers=2)

        # Lazy-init Ollama client
        self._ollama_client = None

    def _get_ollama(self):
        if self._ollama_client is None:
            self._ollama_client = OllamaClient(model=self.ollama_model)
        return self._ollama_client

    def _finbert_sentiment(self, content):
        """Run FinBERT, return dict with sentiment and latency."""
        t0 = time.time()
        sentiment = 0.0
        if self.finbert is not None:
            try:
                res = self.finbert(content[:512])[0]
                label = res["label"]
                sentiment = 1.0 if label == "POSITIVE" else (-1.0 if label == "NEGATIVE" else 0.0)
            except Exception:
                sentiment = 0.0
        latency = (time.time() - t0) * 1000
        return {"sentiment": sentiment, "latency_ms": round(latency, 2)}

    def _deepseek_evaluate(self, content, rag_context):
        """Call Deepseek API with CoT prompt, return parsed result."""
        from src.prompts import COT_RAG_SYSTEM_PROMPT as SYSTEM_P
        from src.rag_retriever import extract_entity

        if rag_context:
            entity = extract_entity(content)
            if entity == "UNKNOWN":
                entity = "this company"
            system_msg = SYSTEM_P.format(entity=entity)
            user_msg = COT_RAG_USER_PROMPT.format(entity=entity, context=rag_context, headline=content[:1000])
        else:
            system_msg = NON_RAG_SYSTEM_PROMPT
            user_msg = f"Analyze this content: {content[:1000]}"

        t0 = time.time()
        response = self.client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            max_tokens=600,
        )
        latency = (time.time() - t0) * 1000

        raw = response.choices[0].message.content.strip()
        parsed = parse_cot_output(raw)
        return parsed, latency

    def _ollama_evaluate(self, content, rag_context):
        """Call Ollama with compact prompt for fast local inference."""
        if rag_context:
            context_short = rag_context[:600]
            user_msg = f"Context:\n{context_short}\n\nHeadline: {content[:400]}\n\nIs this headline FAKE or REAL? Answer FAKE or REAL."
        else:
            user_msg = f"Headline: {content[:400]}\n\nIs this headline FAKE or REAL? Answer FAKE or REAL."
        system_msg = "You are a Financial News Authenticity Verifier. Output FAKE or REAL."

        ollama = self._get_ollama()
        raw, latency = ollama.query(user_msg, system_prompt=system_msg)
        raw_upper = raw.strip().upper()[:20]
        verdict = 1 if "FAKE" in raw_upper else (0 if "REAL" in raw_upper else 0)
        # Simple confidence heuristic: output length / expected length
        confidence = min(0.9, max(0.5, 1.0 - (len(raw) / 200.0))) if raw else 0.5
        return {
            "verdict": verdict,
            "confidence": confidence,
            "contradiction_flag": 0,
            "entity_mismatch": 0,
            "temporal_inconsistency": 0,
            "metric_implausibility": 0,
            "source_unverifiable": 0,
        }, latency

    def _run_system2(self, content):
        """Run RAG retrieval + LLM inference, return results dict."""
        from src.rag_retriever import extract_entity

        retrieval_latency = 0
        rag_context = None
        if self.rag_retriever is not None:
            t0 = time.time()
            ctx_results, _ = self.rag_retriever.retrieve(content)
            retrieval_latency = (time.time() - t0) * 1000
            if ctx_results:
                rag_context = "\n---\n".join([doc[:800] for doc, _score in ctx_results[:2]])

        # LLM call
        llm_latency = 0
        verdict = 0
        confidence = 0.5
        cot_flags = {}

        if self.model == "deepseek" and self.client is not None:
            ds_key_avail = bool(self.deepseek_key) and self.deepseek_key != "your_actual_api_key_here"
            if ds_key_avail:
                try:
                    parsed, llm_latency = self._deepseek_evaluate(content, rag_context)
                    verdict = parsed.get("verdict", 0)
                    confidence = parsed.get("confidence", 0.5)
                    cot_flags = parsed
                except Exception:
                    pass
            else:
                # Mock mode
                time.sleep(0.3)
                llm_latency = 300

        elif self.model.startswith("ollama:"):
            try:
                parsed, llm_latency = self._ollama_evaluate(content, rag_context)
                verdict = parsed.get("verdict", 0)
                confidence = parsed.get("confidence", 0.5)
                cot_flags = parsed
            except Exception:
                pass

        total_latency = retrieval_latency + llm_latency
        return {
            "verdict": verdict,
            "confidence": confidence,
            "cot_flags": cot_flags,
            "retrieval_latency_ms": round(retrieval_latency, 2),
            "llm_latency_ms": round(llm_latency, 2),
            "total_latency_ms": round(total_latency, 2),
            "rag_found": rag_context is not None,
        }

    def process_sample(self, content):
        """Concurrent System 1 + System 2 with latency budget.

        Returns dict with full pipeline trace.
        """
        fut2 = self.executor.submit(self._run_system2, content)

        # System 1 runs immediately (in main thread)
        system1 = self._finbert_sentiment(content)

        # Wait for System 2 with timeout
        max_wait = self.latency_budget_ms / 1000.0 if self.latency_budget_ms else 60.0
        system2_start = time.time()
        budget_violation = False
        system2 = {
            "verdict": 0,
            "confidence": 0.5,
            "cot_flags": {},
            "retrieval_latency_ms": 0,
            "llm_latency_ms": 0,
            "total_latency_ms": self.latency_budget_ms if self.latency_budget_ms else 0,
            "rag_found": False,
        }

        if self.latency_budget_ms is not None:
            try:
                system2 = fut2.result(timeout=max_wait)
            except TimeoutError:
                budget_violation = True
                # Cancel the future (best-effort)
                fut2.cancel()
        else:
            system2 = fut2.result()

        wall_clock = (time.time() - system2_start) * 1000

        if not budget_violation:
            wall_clock = system2.get("total_latency_ms", wall_clock)

        # P&L calculation
        is_fake = system2.get("verdict", 0) == 1
        inter_time = wall_clock if is_fake else None
        pnl_saved = self.pnl.compute(inter_time, fake_detected=is_fake)

        # Final verdict: if budget exceeded, fall back to heuristic signal (use System2 default = 0)
        final_verdict = system2.get("verdict", 0)
        if budget_violation:
            final_verdict = 0  # Trade stands (no intervention)

        return {
            "verdict": final_verdict,
            "confidence": system2.get("confidence", 0.5),
            "sentiment": system1["sentiment"],
            "finbert_latency_ms": system1["latency_ms"],
            "retrieval_latency_ms": system2.get("retrieval_latency_ms", 0),
            "llm_latency_ms": system2.get("llm_latency_ms", 0),
            "total_latency_ms": round(wall_clock, 2),
            "budget_violation": budget_violation,
            "intervention_time_ms": inter_time,
            "pnl_saved": round(pnl_saved, 2),
        }

    def cleanup(self):
        """Release resources."""
        self.executor.shutdown(wait=False)
        if self._ollama_client is not None:
            self._ollama_client.unload()
