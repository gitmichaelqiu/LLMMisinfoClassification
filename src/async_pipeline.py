"""MFT Dual-System pipeline with T0 → T1 → T2 logic.

Overhauls the prior HFT latency budget sweep with the MFT verification
arbitrage flow:

T0 (0s):     System 1 (FinBERT) sentiment → trade entry
T1 (5s):     System 2 (LLM + Dual RAG) → FAKE/REAL/ESCALATE verdict
T2 (300s):   Human verification → final P&L settlement (logged)

P&L computations use the MFTMarketSimulator for realistic price curves,
bid depth dynamics, and execution reflexivity penalties.

False Positive cost (4.2): If LLM intervenes on a REAL event, log
the missed momentum profit between T1 and T2.
"""

import os
import time
import json
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from src.cot_parser import parse_cot_output, save_parsed_samples, VERDICT_FAKE, VERDICT_REAL, VERDICT_ESCALATE
from src.prompts import COT_RAG_SYSTEM_PROMPT, COT_RAG_USER_PROMPT
from src.rag_retriever import DualRAGRetriever, extract_entity
from src.mft_simulator import MFTMarketSimulator
from src.system0_filter import System0Filter

from src.moa_agents import MoADebate


class MFTPipeline:
    """T0 → T1 → T2 pipeline for MFT verification arbitrage.

    Processes events through the dual-system flow:
    1. T0: System 1 (FinBERT) executes trade based on sentiment
    2. T1: System 2 (LLM + Dual RAG) evaluates news veracity
    3. T2: Final P&L computed and logged
    """

    # Verdict constants
    INTERVENE = 1      # FAKE → reverse the trade
    HOLD = 0           # REAL → no intervention
    ESCALATE = 2       # Grey Swan → flag for human review

    def __init__(
        self,
        finbert_model=None,
        dual_rag=None,
        deepseek_client=None,
        deepseek_key=None,
        model="deepseek",
        position_size=1000,
        base_price=100.0,
        use_system0=False,
        thinking="enabled",
        use_moa=False,
        heuristic_predictor=None,
    ):
        """
        Args:
            finbert_model: FinBERT sentiment pipeline (System 1)
            dual_rag: DualRAGRetriever instance
            deepseek_client: OpenAI-compatible DeepSeek client
            deepseek_key: DeepSeek API key
            model: "deepseek" or "ollama:<model>"
            position_size: Position size in shares
            base_price: Base price for the asset
            use_system0: Enable System 0 pre-filter
            thinking: DeepSeek thinking mode
            use_moa: Enable Mixture of Agents debate (Phase 9)
            heuristic_predictor: Callable(content) -> int 0/1.
                Used as mock/fallback when no deepseek API key is available.
                Prevents degenerate HOLD-only behavior (which causes 0% recall).
        """
        self.finbert = finbert_model
        self.dual_rag = dual_rag or DualRAGRetriever()
        self.client = deepseek_client
        self.deepseek_key = deepseek_key
        self.model = model
        self.position_size = position_size
        self.base_price = base_price
        self.thinking = thinking
        self.use_moa = use_moa
        self.heuristic_predictor = heuristic_predictor
        self.system0 = System0Filter(enabled=use_system0)
        self.simulator = MFTMarketSimulator(base_price=base_price)

        # MoA debate engine (Phase 9)
        self.moa = None
        if use_moa:
            api_avail = (
                bool(self.deepseek_key)
                and self.deepseek_key != "your_actual_api_key_here"
                and self.client is not None
            )
            self.moa = MoADebate(
                client=self.client if api_avail else None,
                model="deepseek-v4-flash",
                thinking=thinking,
            )
            print(f"[MFT] MoA debate engine initialized (api_avail={api_avail})")

        # Results accumulator
        self.results = []
        self.parsed_samples = []

        # Thread pool for concurrent processing
        self.executor = ThreadPoolExecutor(max_workers=4)

    # ── System 1: Trade Entry (T0) ──────────────────────────────

    def _system1_sentiment(self, headline):
        """Run FinBERT sentiment on headline. Returns dict with score and latency."""
        t0 = time.time()
        sentiment = 0.0
        if self.finbert is not None:
            try:
                res = self.finbert(headline[:512])[0]
                label = res["label"]
                sentiment = 1.0 if label == "POSITIVE" else (-1.0 if label == "NEGATIVE" else 0.0)
            except Exception:
                sentiment = 0.0
        latency = (time.time() - t0) * 1000
        return {"sentiment": sentiment, "latency_ms": round(latency, 2)}

    # ── System 2: LLM Evaluation (T1) ───────────────────────────

    def _llm_evaluate(self, headline, entity, news_context, social_context):
        """Call LLM with dual-source prompt, return parsed verdict."""
        system_msg = COT_RAG_SYSTEM_PROMPT.format(entity=entity)
        user_msg = COT_RAG_USER_PROMPT.format(
            entity=entity,
            context=news_context,
            social_context=social_context,
            headline=headline[:1000],
        )

        t0 = time.time()
        try:
            response = self.client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=600,
                extra_body={"thinking": {"type": self.thinking}},
            )
            latency = (time.time() - t0) * 1000
            raw = response.choices[0].message.content.strip()
        except Exception:
            latency = (time.time() - t0) * 1000
            raw = ""

        parsed = parse_cot_output(raw)
        return parsed, latency, raw

    def _llm_evaluate_moa(self, headline, entity, news_context, social_context):
        """Evaluate headline using MoA debate (Believer + Skeptic + Risk Officer).

        Phase 9 audit: Believer and Skeptic run CONCURRENTLY.
        Total T1 latency = max(T_believer, T_skeptic) + T_risk_officer.
        """
        moa_result = self.moa.evaluate(
            headline=headline,
            entity=entity,
            news_context=news_context,
            social_context=social_context,
            liquidity_profile=getattr(self.simulator, "liquidity_profile", "mid_cap") or "mid_cap",
        )

        verdict = moa_result["verdict"]
        confidence = moa_result["confidence"]
        cot_flags = moa_result["cot_flags"]
        total_latency = moa_result["latencies"].get("total_moa_ms", 0)

        return moa_result, verdict, confidence, cot_flags, total_latency

    # ── System 2: Dual RAG Retrieval ────────────────────────────

    def _retrieve_context(self, headline, event_id=None):
        """Retrieve dual context (news + social) using DualRAGRetriever."""
        context = self.dual_rag.retrieve(headline, event_id=event_id)
        formatted = self.dual_rag.format_context_for_prompt(context)
        return context, formatted

    # ── Event Processing ─────────────────────────────────────────

    def process_event(self, headline, event_id=None, ground_truth=None):
        """Process a single event through the T0→T1→T2 pipeline.

        Args:
            headline: The headline text to evaluate
            event_id: Optional event ID for social stream lookup
            ground_truth: Optional ground truth label (0=REAL, 1=FAKE)

        Returns:
            dict with full pipeline trace
        """
        # System 0 pre-filter
        system0_passed = self.system0.should_evaluate(headline)

        # ── T0: System 1 Trade Entry ──
        system1 = self._system1_sentiment(headline)

        # Extract entity
        entity = extract_entity(headline)
        if entity == "UNKNOWN":
            entity = "this company"

        # ── T1: System 2 Dual RAG + LLM ──
        system2 = {
            "verdict": self.HOLD,
            "confidence": 0.0,
            "cot_flags": {},
            "retrieval_latency_ms": 0.0,
            "llm_latency_ms": 0.0,
            "total_latency_ms": 0.0,
            "rag_found": False,
        }

        if system0_passed:
            # Dual RAG retrieval
            context, formatted_context = self._retrieve_context(headline, event_id=event_id)

            # Parse contexts for prompt
            news_context = ""
            social_context = ""
            if context["news_context"]:
                news_context = "\n---\n".join(doc[:800] for doc, _score in context["news_context"][:2])
            if context["social_context"]:
                social_lines = []
                for item in context["social_context"]:
                    if len(item) >= 3:
                        text, ts, ptype = item[:3]
                        social_lines.append(f"[{ptype}] t={ts:.1f}s: {text[:300]}")
                    elif len(item) >= 2:
                        social_lines.append(f"t={item[1]:.1f}s: {item[0][:300]}")
                social_context = "\n".join(social_lines)

            # LLM evaluation
            rag_found = bool(news_context or social_context)
            if rag_found:
                deepseek_avail = (
                    bool(self.deepseek_key)
                    and self.deepseek_key != "your_actual_api_key_here"
                    and self.client is not None
                )
                if deepseek_avail:
                    if self.use_moa:
                        # ── MoA Debate (Phase 9) ──
                        moa_result, verdict, confidence, cot_flags, llm_latency = self._llm_evaluate_moa(
                            headline, entity, news_context, social_context
                        )
                        raw_output = moa_result.get("raw_output", "")
                    else:
                        # ── Single-Shot LLM (Phase 3-8) ──
                        parsed, llm_latency, raw_output = self._llm_evaluate(
                            headline, entity, news_context, social_context
                        )
                        verdict = parsed.get("verdict", self.HOLD)
                        confidence = parsed.get("confidence", 0.5)
                        cot_flags = parsed

                    # Save for sample output
                    sample = {
                        "event_id": event_id or "unknown",
                        "headline": headline[:200],
                        "raw_output": raw_output,
                        "parsed": cot_flags,
                    }
                    if self.use_moa:
                        sample["believer_output"] = moa_result.get("believer_output", "")
                        sample["skeptic_output"] = moa_result.get("skeptic_output", "")
                        sample["moa_latencies"] = moa_result.get("latencies", {})
                    self.parsed_samples.append(sample)
                else:
                    # Mock mode — no API key.
                    # Use heuristic predictor if available (avoids degenerate
                    # HOLD-only behavior that causes 0% recall in adversarial tests).
                    if self.heuristic_predictor is not None:
                        h_verdict = self.heuristic_predictor(headline)
                        verdict = h_verdict
                        confidence = 0.6
                        llm_latency = 10.0
                        cot_flags = {"verdict": verdict, "confidence": confidence, "mock": True}
                        parsed = cot_flags
                    else:
                        parsed = {"verdict": self.HOLD, "confidence": 0.5}
                        verdict = self.HOLD
                        confidence = 0.5
                        cot_flags = parsed
                        llm_latency = 50.0

                retrieval_latency = context.get("retrieval_latency_ms", 0)
                system2 = {
                    "verdict": verdict,
                    "confidence": confidence,
                    "cot_flags": cot_flags,
                    "retrieval_latency_ms": round(retrieval_latency, 2),
                    "llm_latency_ms": round(llm_latency, 2),
                    "total_latency_ms": round(retrieval_latency + llm_latency, 2),
                    "rag_found": True,
                }

        # ── T1 → T2: P&L Computation ──
        is_fake_headline = (ground_truth == 1) if ground_truth is not None else None
        llm_verdict = system2["verdict"]
        llm_intervened = (llm_verdict == self.INTERVENE)
        llm_escalated = (llm_verdict == self.ESCALATE)

        # Compute P&L for the three scenarios
        pos = self.position_size

        # Scenario 1: Hold-for-Human (always compute)
        hold_pnl = self.simulator.compute_hold_for_human_pnl(
            position_size=pos, is_fake=is_fake_headline if is_fake_headline is not None else True
        )

        # Scenario 2: LLM-Intervene at T1 (if LLM says FAKE)
        intervene_pnl = self.simulator.compute_llm_intervene_pnl(
            position_size=pos, is_fake=is_fake_headline if is_fake_headline is not None else True,
            intervention_time=self.simulator.T1,
        )

        # Actual P&L based on what the LLM decided
        if llm_intervened and is_fake_headline is True:
            # True Positive: LLM correctly reversed a fake → use intervene P&L
            # pnl_saved = intervene - hold = smaller_loss - bigger_loss = positive
            actual_pnl = intervene_pnl["pnl"]
            pnl_saved = intervene_pnl["pnl"] - hold_pnl["pnl"]
            missed_profit = 0.0
            outcome = "true_positive"
        elif llm_intervened and is_fake_headline is False:
            # False Positive: LLM reversed a REAL event → missed momentum profit
            # pnl_saved = intervene - hold = smaller_profit - larger_profit = negative
            real_price_t1 = self.simulator.price_at(self.simulator.T1, is_fake=False)
            real_price_t2 = self.simulator.price_at(self.simulator.T2, is_fake=False)
            missed_profit = (real_price_t2 - real_price_t1) * pos

            actual_pnl = intervene_pnl["pnl"]
            pnl_saved = intervene_pnl["pnl"] - hold_pnl["pnl"]  # negative (intervention hurt)
            outcome = "false_positive"
        elif not llm_intervened and is_fake_headline is True:
            # False Negative: LLM missed a FAKE → held to max drawdown
            actual_pnl = hold_pnl["pnl"]
            pnl_saved = 0.0
            missed_profit = 0.0
            outcome = "false_negative"
        elif not llm_intervened and is_fake_headline is False:
            # True Negative: LLM correctly held a REAL → full profit
            actual_pnl = hold_pnl["pnl"]
            pnl_saved = 0.0
            missed_profit = 0.0
            outcome = "true_negative"
        else:
            # Unknown ground truth
            actual_pnl = intervene_pnl["pnl"] if llm_intervened else hold_pnl["pnl"]
            pnl_saved = 0.0 if not llm_intervened else (intervene_pnl["pnl"] - hold_pnl["pnl"])
            missed_profit = 0.0
            outcome = "unknown"

        result = {
            "event_id": event_id or "unknown",
            "headline": headline[:200],
            "entity": entity,
            "ground_truth": ground_truth,
            "system1_sentiment": system1["sentiment"],
            "system1_latency_ms": system1["latency_ms"],
            "llm_verdict": llm_verdict,
            "llm_verdict_label": self._verdict_label(llm_verdict),
            "llm_confidence": system2["confidence"],
            "system2_latency_ms": system2["total_latency_ms"],
            "retrieval_latency_ms": system2["retrieval_latency_ms"],
            "rag_found": system2["rag_found"],
            "system0_passed": system0_passed,
            "hold_pnl": round(hold_pnl["pnl"], 2),
            "intervene_pnl": round(intervene_pnl["pnl"], 2),
            "actual_pnl": round(actual_pnl, 2),
            "pnl_saved": round(pnl_saved, 2),
            "missed_profit": round(missed_profit, 2),
            "outcome": outcome,
            "entry_price": hold_pnl["entry_price"],
            "hold_exit_price": hold_pnl["exit_price"],
            "intervene_exit_price": intervene_pnl["exit_price"],
            "reflexivity_penalty": intervene_pnl.get("reflexivity_penalty", 0),
            "fill_ratio": intervene_pnl.get("fill_ratio", 0),
            "is_fake_event": is_fake_headline,
            "use_moa": self.use_moa,
            "mock_verdict": system2.get("cot_flags", {}).get("mock", False),
        }

        # Add MoA-specific outputs if available
        if self.use_moa and self.parsed_samples:
            last_sample = self.parsed_samples[-1]
            if "believer_output" in last_sample:
                result["believer_output"] = last_sample.get("believer_output", "")
                result["skeptic_output"] = last_sample.get("skeptic_output", "")
                result["moa_latencies"] = last_sample.get("moa_latencies", {})

        self.results.append(result)
        return result

    @staticmethod
    def _verdict_label(v):
        return {0: "REAL/HOLD", 1: "FAKE/INTERVENE", 2: "ESCALATE"}.get(v, "UNKNOWN")

    # ── Batch Processing ─────────────────────────────────────────

    def process_events(self, events_df):
        """Process a DataFrame of events through the pipeline.

        Args:
            events_df: DataFrame with columns: T0_headline, event_id, T2_human_verdict

        Returns:
            list of result dicts
        """
        results = []
        total = len(events_df)

        for i, (_, row) in enumerate(events_df.iterrows()):
            headline = str(row.get("T0_headline", row.get("headline", "")))
            event_id = str(row.get("event_id", ""))
            ground_truth = int(row.get("T2_human_verdict", row.get("label", -1)))

            result = self.process_event(
                headline=headline,
                event_id=event_id,
                ground_truth=ground_truth if ground_truth >= 0 else None,
            )
            results.append(result)

            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1}/{total}] {event_id}: "
                      f"verdict={self._verdict_label(result['llm_verdict'])} "
                      f"pnl=${result['actual_pnl']:.0f} "
                      f"saved=${result['pnl_saved']:.0f}")

        return results

    # ── Reporting ────────────────────────────────────────────────

    def generate_report(self, output_dir="./output", plots_dir="./plots",
                        save_samples=True):
        """Generate the MFT backtest report with metrics and plots.

        Args:
            output_dir: Directory for JSON output
            plots_dir: Directory for plot output
            save_samples: Save parsed LLM output samples

        Returns:
            dict with full metrics
        """
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        if not self.results:
            print("[MFT] No results to report.")
            return {}

        df = pd.DataFrame(self.results)

        # ── Core Metrics ──
        n_total = len(df)
        n_fake = int(df["is_fake_event"].sum()) if df["is_fake_event"].notna().any() else 0
        n_real = n_total - n_fake
        n_intervened = int((df["llm_verdict"] == self.INTERVENE).sum())
        n_escalated = int((df["llm_verdict"] == self.ESCALATE).sum())
        n_held = int((df["llm_verdict"] == self.HOLD).sum())

        # Outcome counts
        outcome_counts = df["outcome"].value_counts().to_dict()
        tp = outcome_counts.get("true_positive", 0)
        fp = outcome_counts.get("false_positive", 0)
        tn = outcome_counts.get("true_negative", 0)
        fn = outcome_counts.get("false_negative", 0)

        # P&L metrics
        total_pnl = float(df["actual_pnl"].sum())
        total_pnl_saved = float(df["pnl_saved"].sum())
        mean_pnl = float(df["actual_pnl"].mean())
        std_pnl = float(df["actual_pnl"].std())
        sharpe = (mean_pnl / std_pnl) if std_pnl > 0 else 0.0

        # Per-outcome P&L breakdown
        fp_pnl = float(df[df["outcome"] == "false_positive"]["actual_pnl"].sum()) if fp > 0 else 0.0
        tp_pnl = float(df[df["outcome"] == "true_positive"]["actual_pnl"].sum()) if tp > 0 else 0.0
        tn_pnl = float(df[df["outcome"] == "true_negative"]["actual_pnl"].sum()) if tn > 0 else 0.0
        fn_pnl = float(df[df["outcome"] == "false_negative"]["actual_pnl"].sum()) if fn > 0 else 0.0

        # Accuracy metrics
        labeled = df[df["outcome"].isin(["true_positive", "true_negative", "false_positive", "false_negative"])]
        if len(labeled) > 0:
            accuracy = (tp + tn) / len(labeled) if len(labeled) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        else:
            accuracy = precision = recall = 0.0

        # Latency
        latencies = df["system2_latency_ms"].dropna()
        latency_mean = float(latencies.mean()) if len(latencies) > 0 else 0.0

        # FP cost details (explicit missed profit, not derived from sign-inverted pnl_saved)
        fp_details = None
        if fp > 0:
            fp_df = df[df["outcome"] == "false_positive"]
            fp_details = {
                "count": fp,
                "total_cost": round(float(fp_df["actual_pnl"].sum()), 2),
                "mean_cost": round(float(fp_df["actual_pnl"].mean()), 2),
                "max_cost": round(float(fp_df["actual_pnl"].max()), 2),
                "total_missed_profit": round(float(fp_df["missed_profit"].sum()), 2),
                "mean_missed_profit": round(float(fp_df["missed_profit"].mean()), 2),
            }

        # Build report
        report = {
            "phase": "4_mft_backtest",
            "timestamp": time.ctime(),
            "params": {
                "position_size": self.position_size,
                "base_price": self.base_price,
                "model": self.model,
                "thinking": self.thinking,
                "use_system0": self.system0.enabled,
            },
            "n_events": n_total,
            "n_fake": n_fake,
            "n_real": n_real,
            "verdict_distribution": {
                "intervene": n_intervened,
                "hold": n_held,
                "escalate": n_escalated,
            },
            "outcome_counts": {
                "true_positive": tp,
                "false_positive": fp,
                "true_negative": tn,
                "false_negative": fn,
            },
            "accuracy_metrics": {
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
            },
            "pnl_metrics": {
                "total_pnl": round(total_pnl, 2),
                "total_pnl_saved": round(total_pnl_saved, 2),
                "mean_pnl": round(mean_pnl, 2),
                "std_pnl": round(std_pnl, 2),
                "sharpe_ratio": round(sharpe, 4),
                "tp_pnl": round(tp_pnl, 2),
                "fp_pnl": round(fp_pnl, 2),
                "tn_pnl": round(tn_pnl, 2),
                "fn_pnl": round(fn_pnl, 2),
            },
            "false_positive_analysis": fp_details,
            "latency_ms": {
                "mean": round(latency_mean, 2),
                "system1_mean": float(df["system1_latency_ms"].mean()),
                "retrieval_mean": float(df["retrieval_latency_ms"].mean()),
            },
            "use_moa": self.use_moa,
            "mock_mode": bool(df.get("mock_verdict", pd.Series([False])).any()),
        }

        # Add MoA latency breakdown if applicable
        if self.use_moa and "moa_latencies" in df.columns:
            moa_lat = df["moa_latencies"].dropna()
            if len(moa_lat) > 0:
                # Extract mean per component
                believer_ms = [l.get("believer_ms", 0) for l in moa_lat if isinstance(l, dict)]
                skeptic_ms = [l.get("skeptic_ms", 0) for l in moa_lat if isinstance(l, dict)]
                risk_ms = [l.get("risk_officer_ms", 0) for l in moa_lat if isinstance(l, dict)]
                report["moa_latency_ms"] = {
                    "believer_mean": round(float(np.mean(believer_ms)), 2) if believer_ms else 0.0,
                    "skeptic_mean": round(float(np.mean(skeptic_ms)), 2) if skeptic_ms else 0.0,
                    "risk_officer_mean": round(float(np.mean(risk_ms)), 2) if risk_ms else 0.0,
                    "total_moa_mean": round(float(np.mean([l.get("total_moa_ms", 0) for l in moa_lat if isinstance(l, dict)])), 2) if len(moa_lat) > 0 else 0.0,
                }

        # Save JSON
        json_path = os.path.join(output_dir, "mft_backtest_results.json")
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[MFT] Report saved to {json_path}")

        # Save parsed samples
        if save_samples and self.parsed_samples:
            sample_path = os.path.join(output_dir, "phase3_prompt_samples.json")
            save_parsed_samples(self.parsed_samples[:10], output_path=sample_path)

        # Generate plots
        self._generate_plots(df, plots_dir)

        return report

    def _generate_plots(self, df, plots_dir):
        """Generate P&L distribution and escalation rate plots."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[MFT] Matplotlib not available, skipping plots.")
            return

        # P&L Distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        pnl_vals = df["actual_pnl"].dropna().values

        ax.hist(pnl_vals, bins=30, alpha=0.7, color="steelblue", edgecolor="white")
        ax.axvline(x=0, color="red", linestyle="--", alpha=0.5, label="Breakeven")

        # Highlight TP vs FP
        tp_pnls = df[df["outcome"] == "true_positive"]["actual_pnl"].dropna().values
        fp_pnls = df[df["outcome"] == "false_positive"]["actual_pnl"].dropna().values
        if len(tp_pnls) > 0:
            ax.hist(tp_pnls, bins=15, alpha=0.5, color="green", label="True Positives")
        if len(fp_pnls) > 0:
            ax.hist(fp_pnls, bins=15, alpha=0.5, color="red", label="False Positives")

        ax.set_xlabel("P&L ($)")
        ax.set_ylabel("Frequency")
        ax.set_title("MFT Backtest P&L Distribution", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.subplots_adjust(right=0.85)

        pnl_path = os.path.join(plots_dir, "mft_pnl_distribution.png")
        fig.savefig(pnl_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[MFT] P&L distribution saved to {pnl_path}")

        # Grey Swan Escalation Rates
        fig, ax = plt.subplots(figsize=(10, 6))

        esc_df = df[df["llm_verdict"] == self.ESCALATE]

        if len(esc_df) == 0:
            ax.text(0.5, 0.5, "No grey swan escalations in this batch",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=14, fontstyle="italic", color="gray")
            ax.set_axis_off()
        else:
            has_valid_labels = esc_df["is_fake_event"].notna().any()
            esc_fake = len(esc_df[esc_df["is_fake_event"] == True]) if has_valid_labels else 0
            esc_real = len(esc_df[esc_df["is_fake_event"] == False]) if has_valid_labels else 0

            categories = ["FAKE Events Escalated", "REAL Events Escalated", "Total Escalated"]
            values = [esc_fake, esc_real, len(esc_df)]

            bars = ax.bar(categories, values,
                          color=["#d62728", "#2ca02c", "#1f77b4"],
                          alpha=0.8, edgecolor="white")
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        str(val), ha="center", fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="y")

        ax.set_ylabel("Count")
        ax.set_title("Grey Swan Escalation Rates", fontsize=14, fontweight="bold")
        total = len(df)
        esc_rate = len(esc_df) / total * 100 if total > 0 else 0
        fig.text(0.5, 0.01, f"Escalation Rate: {esc_rate:.1f}% ({len(esc_df)}/{total})",
                 ha="center", fontsize=11, fontweight="bold",
                 bbox=dict(facecolor="orange", alpha=0.2))

        grey_path = os.path.join(plots_dir, "grey_swan_escalation_rates.png")
        fig.savefig(grey_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[MFT] Grey swan escalation rates saved to {grey_path}")

    def cleanup(self):
        """Release thread pool and MoA resources."""
        self.executor.shutdown(wait=False)
        if self.moa is not None:
            self.moa.shutdown()


def run_mft_backtest(
    temporal_events_path="./output/temporal_events.csv",
    test_ratio=0.2,
    model="deepseek",
    position_size=1000,
    base_price=100.0,
    thinking="enabled",
    use_system0=False,
    max_events=None,
    use_moa=False,
    heuristic_predictor=None,
):
    """Run the full MFT backtest end-to-end.

    1. Load temporal events
    2. Apply temporal train/test split
    3. Run MFTPipeline on test set
    4. Generate report and plots

    Args:
        temporal_events_path: Path to temporal_events.csv
        test_ratio: Fraction of events for test set
        model: LLM backend
        position_size: Shares per trade
        base_price: Base entry price
        thinking: DeepSeek thinking mode
        use_system0: Enable System 0 pre-filter
        max_events: Cap test set size for quick testing
        use_moa: Enable MoA debate architecture (Phase 9)

    Returns:
        Dict with full metrics
    """
    import os
    from dotenv import load_dotenv
    from openai import OpenAI
    from transformers import pipeline

    load_dotenv()
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")

    print("=" * 60)
    print("MFT BACKTEST")
    print(f"Model: {model} | Position: {position_size} shares | Base: ${base_price}")
    print(f"Thinking: {thinking} | System0: {use_system0}")
    print("=" * 60)

    # Load events and split
    events_df = pd.read_csv(temporal_events_path)
    print(f"Loaded {len(events_df)} temporal events")

    # Temporal train/test split
    from src.data_splitter import temporal_train_test_split
    train_ids, test_ids, split_meta = temporal_train_test_split(
        temporal_events_path=temporal_events_path,
        test_ratio=test_ratio,
    )
    test_df = events_df[events_df["event_id"].isin(test_ids)]
    if max_events and len(test_df) > max_events:
        test_df = test_df.head(max_events)
    print(f"Test set: {len(test_df)} events")

    # Train heuristic baseline from training set for mock fallback
    if heuristic_predictor is None:
        train_df = events_df[events_df["event_id"].isin(train_ids)]
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
            clf = LogisticRegression(max_iter=1000, random_state=42)
            texts = train_df.get("T0_headline", train_df.get("headline", ""))
            labels = train_df.get("T2_human_verdict", train_df.get("label", -1))
            if len(texts) >= 10 and not (labels == -1).all():
                X = vectorizer.fit_transform(texts.fillna(""))
                clf.fit(X, labels)
                def _pred(content):
                    return int(clf.predict(vectorizer.transform([content]))[0])
                heuristic_predictor = _pred
                print(f"  Heuristic baseline trained: {len(train_df)} samples")
        except Exception as e:
            print(f"  Heuristic baseline training skipped: {e}")

    # Initialize FinBERT (System 1)
    print("\nInitializing FinBERT...")
    local_path = os.path.join(os.getcwd(), "models", "finbert")
    try:
        if os.path.exists(local_path) and os.listdir(local_path):
            finbert = pipeline("sentiment-analysis", model=local_path, tokenizer=local_path)
        else:
            finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    except Exception as e:
        print(f"  FinBERT unavailable: {e}")
        finbert = None

    # Initialize DeepSeek client
    client = None
    if deepseek_key and deepseek_key != "your_actual_api_key_here":
        client = OpenAI(
            api_key=deepseek_key,
            base_url="https://api.deepseek.com",
            timeout=30.0,
        )
        print("DeepSeek client ready")
    else:
        print("No API key — running in mock mode")

    # Initialize Dual RAG
    print("Initializing Dual RAG...")
    dual_rag = DualRAGRetriever()
    dual_rag.ensure_index()

    # Build pipeline
    pipeline = MFTPipeline(
        finbert_model=finbert,
        dual_rag=dual_rag,
        deepseek_client=client,
        deepseek_key=deepseek_key,
        model=model,
        position_size=position_size,
        base_price=base_price,
        use_system0=use_system0,
        thinking=thinking,
        use_moa=use_moa,
        heuristic_predictor=heuristic_predictor,
    )

    # Run test set
    print(f"\nProcessing {len(test_df)} test events...")
    results = pipeline.process_events(test_df)

    # Generate report
    report = pipeline.generate_report()

    pipeline.cleanup()
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MFT Backtest")
    parser.add_argument("--model", default="deepseek", help="LLM backend")
    parser.add_argument("--test-size", type=int, default=20, help="Test set size")
    parser.add_argument("--position-size", type=int, default=1000, help="Shares per trade")
    parser.add_argument("--base-price", type=float, default=100.0, help="Entry price")
    parser.add_argument("--thinking", default="enabled", choices=["enabled", "disabled"])
    parser.add_argument("--system0", action="store_true", help="Enable System 0 filter")
    parser.add_argument("--moa", action="store_true", help="Enable MoA debate architecture (Phase 9)")
    args = parser.parse_args()

    run_mft_backtest(
        model=args.model,
        test_ratio=0.2,
        position_size=args.position_size,
        base_price=args.base_price,
        thinking=args.thinking,
        use_system0=args.system0,
        max_events=args.test_size,
        use_moa=args.moa,
    )


# ── Backward-compatible aliases ──────────────────────────────────
# AsyncDualPipeline and PnLCalculator were the original HFT pipeline
# classes. They are retained as aliases so that main.py (which still
# references the old names) continues to work after the MFT refactor.

class PnLCalculator:
    """Legacy PnLCalculator — delegates to MFTPipeline's simulator P&L logic."""
    @staticmethod
    def compute_pnl(position_size, entry_price, exit_price):
        return (exit_price - entry_price) * position_size


class AsyncDualPipeline(MFTPipeline):
    """Legacy AsyncDualPipeline — delegate wrapper around MFTPipeline.

    The original interface used latency_budget_ms and process_sample().
    This wrapper translates to the MFT pipeline interface.
    """
    def __init__(self, finbert_model=None, rag_retriever=None,
                 deepseek_client=None, deepseek_key=None,
                 latency_budget_ms=None, model="deepseek",
                 ollama_model="qwen3.5:2b", position_size=1000,
                 thinking="enabled", heuristic_predictor=None,
                 use_system0=False):
        import warnings
        warnings.warn("AsyncDualPipeline is deprecated; use MFTPipeline.", DeprecationWarning, stacklevel=2)
        # Convert rag_retriever -> dual_rag
        from src.rag_retriever import DualRAGRetriever
        if rag_retriever is not None and not isinstance(rag_retriever, DualRAGRetriever):
            # Wrap legacy RAGRetriever in DualRAGRetriever
            dual_rag = DualRAGRetriever()
            dual_rag.news_retriever = rag_retriever
        elif rag_retriever is not None:
            dual_rag = rag_retriever
        else:
            dual_rag = DualRAGRetriever()

        super().__init__(
            finbert_model=finbert_model,
            dual_rag=dual_rag,
            deepseek_client=deepseek_client,
            deepseek_key=deepseek_key,
            model=model,
            position_size=position_size,
            base_price=190.0,
            use_system0=use_system0,
            thinking=thinking,
            heuristic_predictor=heuristic_predictor,
        )
        self._latency_budget_ms = latency_budget_ms

    def process_sample(self, content):
        """Legacy process_sample — returns {verdict, total_latency_ms, pnl_saved, ...}."""
        result = self.process_event(headline=content)
        return {
            "verdict": result["llm_verdict"],
            "total_latency_ms": result["system2_latency_ms"],
            "finbert_latency_ms": result["system1_latency_ms"],
            "retrieval_latency_ms": result["retrieval_latency_ms"],
            "llm_latency_ms": result["system2_latency_ms"],
            "budget_violation": self._latency_budget_ms is not None and result["system2_latency_ms"] > self._latency_budget_ms,
            "pnl_saved": result["pnl_saved"],
            "confidence": result["llm_confidence"],
            "intervention_time_ms": self.simulator.T1 * 1000,
        }

    def cleanup(self):
        """Legacy cleanup."""
        super().cleanup()
