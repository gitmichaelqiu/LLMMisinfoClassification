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

*No experiments recorded yet. Phase 3 (Single-Shot Verifier) will produce
the first entries.*

