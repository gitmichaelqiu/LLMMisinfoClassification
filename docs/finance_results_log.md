# Finance Case Study — Experiment Results Log

## Experiment Template

Every experiment MUST append a record with the following fields. Copy the
template below for each new run.

```yaml
date: <YYYY-MM-DD HH:MM>
git_commit: <7-char hash>
dataset_version: <e.g., synthetic-v1, ood-human-v2>
verifier_architecture: <single-shot | voting | moa | rag>
prompt_version: <e.g., v1.0-default, v1.1-cot>
model: <e.g., gpt-4o-mini, deepseek-chat, claude-sonnet-4>
sample_size: <N>
latency:
  mean_s: <float>
  median_s: <float>
  p95_s: <float>
  p99_s: <float>
metrics:
  precision: <float>
  recall: <float>
  f1: <float>
  false_positive_rate: <float>
  false_negative_rate: <float>
  accuracy: <float>
  confusion_matrix:
    tp: <int>
    fp: <int>
    tn: <int>
    fn: <int>
ppv_at_base_rates:
  "0.1%": <float>
  "1%": <float>
  "5%": <float>
  "10%": <float>
  "25%": <float>
  "50%": <float>
failure_notes: <free text about failure cases, edge cases, anomalies>
result_json_path: <path>
plot_paths:
  - <path>
```

---

## Run Log

## Phase 3 — Single-Shot Verifier Baseline

*Recorded in previous session (see git history for details).*

---

## Phase 4 — Voting Verifier

*Recorded in previous session (see git history for details).*

---

## Phase 5 — MoA/Debate Verifier

```yaml
date: 2026-07-07 21:10
git_commit: pending
dataset_version: synthetic-v1
verifier_architecture: moa
prompt_version: v1.0-default
model: gpt-4o-mini (mock)
sample_size: 10
latency: N/A (mock — sub-second)
metrics:
  precision: 0.5000
  recall: 1.0000
  f1: 0.6667
  false_positive_rate: 1.0000
  false_negative_rate: 0.0000
  accuracy: 0.5000
  confusion_matrix:
    tp: 5
    fp: 5
    tn: 0
    fn: 0
ppv_at_base_rates: N/A (mock — PPV computation deferred until real API results)
failure_notes: >
  IMPLEMENTATION TEST ONLY — not a scientific finding.
  MockClient returns "Verdict: FAKE" for all prompts regardless of content
  or role (Believer, Skeptic, Risk Officer). This means:
  - Believer (pro-REAL) and Skeptic (pro-FAKE) produce IDENTICAL output
  - No actual debate occurs — the architecture is not exercised
  - All 3 degeneracy checks fire: always-FAKE, always-REAL, precision≈P(FAKE)
  - Info gain (precision - P(FAKE)) = 0.0 — zero information added
  Results will differ qualitatively when run with real LLM API calls.
result_json_path: results/phase05/20260707_211012.json
plot_paths: []
```

