import os
import time
import random
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from transformers import pipeline
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from src.rag_retriever import RAGRetriever, extract_entity
from src.prompts import RAG_SYSTEM_PROMPT, RAG_USER_PROMPT, NON_RAG_SYSTEM_PROMPT, COT_RAG_SYSTEM_PROMPT, COT_RAG_USER_PROMPT
from src.cot_parser import parse_cot_output
from src.ensemble_detector import EnsembleDetector, finbert_score, compute_ece, plot_calibration_curve
from src.async_pipeline import AsyncDualPipeline, PnLCalculator
from src.sensitivity_analysis import SensitivityAnalyzer, flash_crash_price
from src.domain_adapter import set_domain, get_adapter, DomainAdapter
from src.prompts import get_domain_prompts

# 1. Environment Setup
load_dotenv()
random.seed(42)
np.random.seed(42)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Initialize FinBERT (System 1)
print("Loading FinBERT (System 1)...")
LOCAL_FINBERT_PATH = os.path.join(os.getcwd(), "models", "finbert")

try:
    if os.path.exists(LOCAL_FINBERT_PATH) and os.listdir(LOCAL_FINBERT_PATH):
        print(f"Loading from local path: {LOCAL_FINBERT_PATH}")
        finbert = pipeline("sentiment-analysis", model=LOCAL_FINBERT_PATH, tokenizer=LOCAL_FINBERT_PATH)
    else:
        print("Local model not found, downloading/loading from HuggingFace...")
        finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")
except Exception as e:
    print(f"Warning: Could not load FinBERT. Falling back to mock sentiment. Error: {e}")
    finbert = None

# Configure Deepseek client (System 2)
# Note: Using a dummy key initially if not provided to allow simulation to run
client = OpenAI(
    api_key=DEEPSEEK_API_KEY if DEEPSEEK_API_KEY and DEEPSEEK_API_KEY != "your_actual_api_key_here" else "dummy",
    base_url="https://api.deepseek.com",
    timeout=30.0,
)

# RAG retriever for context-aware evaluation
rag_retriever = None
try:
    rr = RAGRetriever()
    rr.ensure_index()
    rag_retriever = rr
    print("[RAG] Retriever ready.")
except Exception as e:
    print(f"[RAG] Retriever unavailable: {e}")

# Heuristic baseline (System 2 fallback)
_heuristic_baseline = None

def train_heuristic_baseline(df):
    global _heuristic_baseline
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    clf = LogisticRegression(max_iter=1000, random_state=42)
    X = vectorizer.fit_transform(df['content'].fillna(''))
    y = df['label']
    clf.fit(X, y)
    _heuristic_baseline = (vectorizer, clf)
    print(f"[Heuristic Baseline] Trained on {len(df)} samples (acc={clf.score(X, y):.2%})")

def heuristic_predict(content):
    global _heuristic_baseline
    if _heuristic_baseline is None:
        return 0
    vectorizer, clf = _heuristic_baseline
    X = vectorizer.transform([content])
    return int(clf.predict(X)[0])

class MarketSimulator:
    def __init__(self, base_price=190.0):
        self.base_price = base_price
        self.current_price = base_price
        
    def get_price(self, t_ms):
        """
        Simulates a Liquidity Vacuum Flash Crash.
        T=0: Headline hits.
        T=0-3000ms: Rapid drop to trough ($150) due to HFT withdrawal.
        T=3000ms-10000ms: Recovery to $190.
        """
        if t_ms < 0: return self.base_price
        if t_ms <= 3000:
            # Linear drop from 190 to 150
            return self.base_price - (t_ms / 3000) * 40
        elif t_ms <= 10000:
            # Recovery from 150 to 190
            return 150 + ((t_ms - 3000) / 7000) * 40
        else:
            return self.base_price

def load_combined_dataset():
    """
    Integrates Synthetic and Kaggle datasets into a unified format.
    Synthetic follows binary: 0=Authentic, 1=Anomaly
    Kaggle follows binary: 0=Real, 1=Fake
    """
    # Load synthetic
    synth_path = "./input/headlines.csv"
    if os.path.exists(synth_path):
        synth_df = pd.read_csv(synth_path)
        synth_df['source'] = 'synthetic'
        synth_df['content'] = synth_df['headline']
        # Preserve type column for per-subset analysis
        if 'type' not in synth_df.columns:
            synth_df['type'] = ''
    else:
        synth_df = pd.DataFrame(columns=['content', 'label', 'source', 'type'])

    # Load Kaggle
    kaggle_path = "./input/kaggle_fake_news_FULL.csv"
    if os.path.exists(kaggle_path):
        kaggle_df = pd.read_csv(kaggle_path)
        kaggle_df = kaggle_df.dropna(subset=['text', 'label'])
        kaggle_df['source'] = 'kaggle'
        kaggle_df['content'] = kaggle_df['text']
        kaggle_df['type'] = ''
    else:
        kaggle_df = pd.DataFrame(columns=['content', 'label', 'source', 'type'])

    # Combine
    combined = pd.concat([synth_df[['content', 'label', 'source', 'type']],
                          kaggle_df[['content', 'label', 'source', 'type']]],
                         ignore_index=True)
    return combined

def system_2_evaluate(content, use_rag=False):
    """Return (verdict, retrieval_latency_ms)."""
    # If API key is missing or dummy, use heuristic baseline
    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "your_actual_api_key_here":
        time.sleep(0.5)
        return heuristic_predict(content), 0

    # RAG: retrieve context before LLM call
    rag_context = None
    retrieval_latency = 0
    if use_rag and rag_retriever is not None:
        t0 = time.time()
        ctx_results, _ = rag_retriever.retrieve(content)
        retrieval_latency = (time.time() - t0) * 1000
        if ctx_results:
            rag_context = "\n---\n".join([doc[:800] for doc, _score in ctx_results[:2]])

    try:
        if rag_context:
            entity = extract_entity(content)
            if entity == "UNKNOWN":
                entity = "this company"
            system_msg = RAG_SYSTEM_PROMPT.format(entity=entity)
            user_msg = RAG_USER_PROMPT.format(entity=entity, context=rag_context, headline=content[:1000])
        else:
            system_msg = NON_RAG_SYSTEM_PROMPT
            user_msg = f"Analyze this content: {content[:1000]}"

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            max_tokens=10
        )
        verdict_raw = response.choices[0].message.content.strip().upper()
        return (1 if "FAKE" in verdict_raw else 0), retrieval_latency
    except Exception as e:
        return 0, retrieval_latency


def cot_evaluate(content):
    """Run CoT evaluation with RAG context.

    Returns (verdict, confidence, flags_dict, total_latency_ms, retrieval_latency_ms).
    """
    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "your_actual_api_key_here":
        # Mock: use heuristic baseline with defaults
        time.sleep(0.5)
        v = heuristic_predict(content)
        return v, 0.5, {"verdict": v, "confidence": 0.5}, 500, 0

    rag_context = None
    retrieval_latency = 0
    if rag_retriever is not None:
        t0 = time.time()
        ctx_results, _ = rag_retriever.retrieve(content)
        retrieval_latency = (time.time() - t0) * 1000
        if ctx_results:
            rag_context = "\n---\n".join([doc[:800] for doc, _score in ctx_results[:2]])

    try:
        if rag_context:
            entity = extract_entity(content)
            if entity == "UNKNOWN":
                entity = "this company"
            system_msg = COT_RAG_SYSTEM_PROMPT.format(entity=entity)
            user_msg = COT_RAG_USER_PROMPT.format(entity=entity, context=rag_context, headline=content[:1000])
        else:
            system_msg = NON_RAG_SYSTEM_PROMPT
            user_msg = f"Analyze this content: {content[:1000]}"

        t0 = time.time()
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            max_tokens=600
        )
        total_latency = (time.time() - t0) * 1000

        raw = response.choices[0].message.content.strip()
        parsed = parse_cot_output(raw)
        verdict = parsed.get("verdict", 0)
        confidence = parsed.get("confidence", 0.5)
        return verdict, confidence, parsed, total_latency, retrieval_latency
    except Exception as e:
        return 0, 0.5, {"verdict": 0, "confidence": 0.5}, 1000, retrieval_latency

import matplotlib.pyplot as plt
import seaborn as sns

def plot_flash_crash_mechanics(sim):
    """
    Generates research-grade visualization of the flash crash execution.
    """
    if not os.path.exists("./plots"):
        os.makedirs("./plots")
        
    t_vals = np.linspace(0, 10000, 500)
    p_vals = [sim.get_price(t) for t in t_vals]
    
    plt.figure(figsize=(12, 6))
    plt.plot(t_vals, p_vals, label="Asset Price ($)", color='#1f77b4', linewidth=2, alpha=0.8)
    
    t1, p1 = 50, sim.get_price(50)
    plt.scatter(t1, p1, color='#d62728', s=100, label="Bot 1: Panic Sell (System 1)", zorder=5)
    
    t3, p3 = 3000, sim.get_price(3000)
    plt.scatter(t3, p3, color='#2ca02c', s=100, label="Bot 3: Logical Buy (System 2)", zorder=5)
    
    plt.title("Flash Crash Dynamics: System 1 Panic vs. System 2 Logic", fontsize=14, fontweight='bold')
    plt.xlabel("Time (milliseconds)", fontsize=12)
    plt.ylabel("Price ($)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig("./plots/flash_crash_dynamics.png", dpi=300, bbox_inches='tight')
    plt.close()

def report_dataset_statistics(df):
    print(f"\n{'='*30}\nDATASET STATISTICS\n{'='*30}")
    total = len(df)
    class0 = int((df['label'] == 0).sum())
    class1 = int((df['label'] == 1).sum())
    sources = df['source'].value_counts().to_dict()
    avg_len = float(df['content'].str.len().mean())
    print(f"Total samples: {total}")
    print(f"Class 0 (Authentic): {class0} ({class0/total:.1%})")
    print(f"Class 1 (Anomaly):   {class1} ({class1/total:.1%})")
    print(f"Sources: {sources}")
    print(f"Avg content length: {avg_len:.0f} chars")
    return {
        'total_samples': total,
        'class_balance': {0: class0, 1: class1},
        'source_distribution': sources,
        'avg_content_length': round(avg_len, 1),
    }

def generate_confusion_matrix(results, output_file=None, dataset_stats=None):
    df = pd.DataFrame(results)
    classes = [0, 1]
    matrix = pd.DataFrame(0, index=classes, columns=classes)
    for _, row in df.iterrows():
        matrix.loc[row['actual'], row['verdict']] += 1

    # Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.title("LLM Verdict Confusion Matrix", fontsize=14, fontweight='bold')
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig("./plots/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

    actuals = df['actual']
    preds = df['verdict']
    accuracy = (actuals == preds).mean() * 100
    precision = precision_score(actuals, preds, average='binary', zero_division=0)
    recall = recall_score(actuals, preds, average='binary', zero_division=0)
    f1 = f1_score(actuals, preds, average='binary', zero_division=0)
    tn, fp = matrix.loc[0, 0], matrix.loc[0, 1]
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    latencies = df['latency_ms']
    latency_mean = latencies.mean()
    latency_p50 = latencies.median()
    latency_p95 = latencies.quantile(0.95)
    latency_p99 = latencies.quantile(0.99)

    latency_actual_0 = df[df['actual'] == 0]['latency_ms'].mean() if (df['actual'] == 0).any() else 0
    latency_actual_1 = df[df['actual'] == 1]['latency_ms'].mean() if (df['actual'] == 1).any() else 0
    latency_verdict_0 = df[df['verdict'] == 0]['latency_ms'].mean() if (df['verdict'] == 0).any() else 0
    latency_verdict_1 = df[df['verdict'] == 1]['latency_ms'].mean() if (df['verdict'] == 1).any() else 0

    metrics = {
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(matrix.loc[1, 0]), 'tp': int(matrix.loc[1, 1])},
        'accuracy_pct': round(accuracy, 2),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'specificity': round(specificity, 4),
        'latency_ms': {
            'mean': round(latency_mean, 2),
            'median': round(latency_p50, 2),
            'p95': round(latency_p95, 2),
            'p99': round(latency_p99, 2),
            'actual_real': round(latency_actual_0, 2),
            'actual_fake': round(latency_actual_1, 2),
            'verdict_real': round(latency_verdict_0, 2),
            'verdict_fake': round(latency_verdict_1, 2),
        }
    }
    if dataset_stats:
        metrics['dataset'] = dataset_stats

    summary = f"\n{'='*30}\nCLASSIFICATION REPORT\n{'='*30}\n{matrix}\n{'='*30}\n"
    summary += f"Accuracy:          {accuracy:.2f}%\n"
    summary += f"Precision:         {precision:.4f}\n"
    summary += f"Recall:            {recall:.4f}\n"
    summary += f"F1 Score:          {f1:.4f}\n"
    summary += f"Specificity:       {specificity:.4f}\n"
    summary += f"Latency(mean):     {latency_mean:.2f}ms\n"
    summary += f"Latency(p50):      {latency_p50:.2f}ms\n"
    summary += f"Latency(p95):      {latency_p95:.2f}ms\n"
    summary += f"Latency(p99):      {latency_p99:.2f}ms\n"
    summary += f"Latency(act_real): {latency_actual_0:.2f}ms\n"
    summary += f"Latency(act_fake): {latency_actual_1:.2f}ms\n"
    summary += f"Latency(ver_real): {latency_verdict_0:.2f}ms\n"
    summary += f"Latency(ver_fake): {latency_verdict_1:.2f}ms\n"
    summary += f"{'='*30}\n"
    print(summary)

    if output_file:
        with open(output_file, 'a') as f:
            f.write(summary)

    json_path = "./output/phase1_metrics.json"
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Phase 1 metrics saved to {json_path}")

def run_simulation(content, mode="single", use_rag=False):
    """Return (verdict, total_latency_ms, retrieval_latency_ms)."""
    if mode == "single":
        print(f"\n--- SINGLE RUN SIMULATION ---\nContent: {content[:100]}...")

    sim = MarketSimulator()

    # System 1
    if finbert:
        try:
            res = finbert(content[:512])[0]
            sentiment = res['label']
        except:
            sentiment = "negative"
    else:
        sentiment = "negative"

    # System 2
    start_time = time.time()
    verdict, retrieval_latency = system_2_evaluate(content, use_rag=use_rag)
    latency = (time.time() - start_time) * 1000

    # Bot 3 Execution if Anomaly (1)
    if verdict == 1:
        if mode == "single":
            plot_flash_crash_mechanics(sim)

    return verdict, latency, retrieval_latency

def run_batch(target_size=1000, test_size=0.2):
    if not os.path.exists("./output"):
        os.makedirs("./output")

    log_file = "./output/backtest_results.txt"

    # Load full dataset
    df = load_combined_dataset()

    # Downsample to target_size: keep all synthetic + sample Kaggle stratified
    synth_df = df[df['source'] == 'synthetic']
    kaggle_df = df[df['source'] == 'kaggle']
    synth_n = len(synth_df)
    kaggle_target = target_size - synth_n
    kaggle0 = kaggle_df[kaggle_df['label'] == 0]
    kaggle1 = kaggle_df[kaggle_df['label'] == 1]
    # Sample equal class representation from Kaggle
    per_class = max(1, kaggle_target // 2)
    kaggle_sample = pd.concat([
        kaggle0.sample(n=min(len(kaggle0), per_class), random_state=42),
        kaggle1.sample(n=min(len(kaggle1), per_class), random_state=42),
    ], ignore_index=True)
    sampled = pd.concat([synth_df, kaggle_sample], ignore_index=True)

    # Stratified train/test split
    train_df, test_df = train_test_split(
        sampled, test_size=test_size, stratify=sampled['label'], random_state=42
    )

    # Save split indices for reproducibility
    split_indices = {
        'train_indices': train_df.index.tolist(),
        'test_indices': test_df.index.tolist(),
        'random_state': 42,
        'test_size': test_size,
        'target_size': target_size,
    }
    with open("./output/split_indices.json", 'w') as f:
        json.dump(split_indices, f, indent=2)

    with open(log_file, 'w') as f:
        f.write(f"Backtest Execution Log - {time.ctime()}\n")
        f.write(f"Total: {len(sampled)} | Train: {len(train_df)} | Test: {len(test_df)}\n\n")

    # Dataset stats on sampled data
    dataset_stats = report_dataset_statistics(sampled)

    # Train heuristic baseline on TRAIN split
    train_heuristic_baseline(train_df)

    # Evaluate on TEST split
    results = []
    print(f"\n--- BATCH TESTING START (Test set: {len(test_df)} samples) ---")
    for i, (idx, row) in enumerate(test_df.iterrows()):
        content = row['content']
        actual = int(row['label'])

        verdict, latency, _retrieval_lat = run_simulation(content, mode="batch")

        res_entry = {
            'index': idx,
            'actual': actual,
            'verdict': verdict,
            'latency_ms': latency,
            'source': row['source']
        }
        results.append(res_entry)

        log_msg = f"[{i+1}/{len(test_df)}] Index: {idx} | Latency: {latency:.2f}ms | Actual: {actual} | Verdict: {verdict} | Source: {row['source']}\n"
        with open(log_file, 'a') as f:
            f.write(log_msg)

        if (i + 1) % 5 == 0 or i == 0:
            print(f"Progress: [{i+1}/{len(test_df)}] | Avg Latency: {pd.DataFrame(results)['latency_ms'].mean():.2f}ms")

    generate_confusion_matrix(results, output_file=log_file, dataset_stats=dataset_stats)
    print(f"Results saved to {log_file}")

def run_rag_vs_baseline(target_size=1000, test_size=0.2):
    """Compare baseline heuristic vs RAG-enhanced detection on same test split.

    In mock mode (no key): RAG simulates context check via keyword overlap,
    baseline uses TF-IDF+LR. In real mode: both use Deepseek, RAG adds context.
    """
    os.makedirs("./output", exist_ok=True)

    df = load_combined_dataset()

    # Downsample (same logic as run_batch)
    synth_df = df[df['source'] == 'synthetic']
    kaggle_df = df[df['source'] == 'kaggle']
    kaggle0 = kaggle_df[kaggle_df['label'] == 0]
    kaggle1 = kaggle_df[kaggle_df['label'] == 1]
    per_class = max(1, (target_size - len(synth_df)) // 2)
    kaggle_sample = pd.concat([
        kaggle0.sample(n=min(len(kaggle0), per_class), random_state=42),
        kaggle1.sample(n=min(len(kaggle1), per_class), random_state=42),
    ], ignore_index=True)
    sampled = pd.concat([synth_df, kaggle_sample], ignore_index=True)

    # Train/test split (same seed as Phase 1)
    train_df, test_df = train_test_split(
        sampled, test_size=test_size, stratify=sampled['label'], random_state=42
    )

    # Subset labels for per-subset analysis
    def _subset_label(row):
        if row['source'] == 'kaggle':
            return 'kaggle'
        t = str(row.get('type', ''))
        if t == 'realistic':
            return 'realistic'
        if t == 'absurdist':
            return 'absurdist'
        return 'authentic'

    # Train baseline
    train_heuristic_baseline(train_df)

    def _compute_metrics(results):
        dfr = pd.DataFrame(results)
        actuals = dfr['actual']
        preds = dfr['verdict']
        acc = (actuals == preds).mean() * 100
        p = precision_score(actuals, preds, average='binary', zero_division=0)
        r = recall_score(actuals, preds, average='binary', zero_division=0)
        f1 = f1_score(actuals, preds, average='binary', zero_division=0)
        lat_mean = dfr['latency_ms'].mean()
        lat_p95 = dfr['latency_ms'].quantile(0.95)
        return {
            'accuracy_pct': round(acc, 2), 'precision': round(p, 4),
            'recall': round(r, 4), 'f1_score': round(f1, 4),
            'latency_mean_ms': round(lat_mean, 2), 'latency_p95_ms': round(lat_p95, 2),
        }

    def _subset_f1(results, label):
        sub = [r for r in results if r['subset'] == label]
        if len(sub) < 2:
            return 0.0
        dfr = pd.DataFrame(sub)
        return round(f1_score(dfr['actual'], dfr['verdict'], average='binary', zero_division=0), 4)

    # --- Run 1: Baseline ---
    print("\n" + "="*60)
    print("RUNNING BASELINE (TF-IDF + Logistic Regression)")
    print("="*60)

    # In mock mode: baseline = heuristic_predict
    # In real mode: baseline = non-RAG Deepseek
    use_rag = True  # will be False for baseline run

    base_results = []
    for i, (idx, row) in enumerate(test_df.iterrows()):
        content = row['content']
        actual = int(row['label'])
        DEEPSEEK_KEY_AVAIL = bool(DEEPSEEK_API_KEY) and DEEPSEEK_API_KEY != "your_actual_api_key_here"

        if not DEEPSEEK_KEY_AVAIL:
            # Mock mode: baseline = heuristic_predict
            time.sleep(0.5)
            v = heuristic_predict(content)
            lat = 500
        else:
            t0 = time.time()
            v, _ = system_2_evaluate(content, use_rag=False)
            lat = (time.time() - t0) * 1000

        base_results.append({
            'index': idx, 'actual': actual, 'verdict': v,
            'latency_ms': lat, 'source': row['source'],
            'subset': _subset_label(row),
        })

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Baseline [{i+1}/{len(test_df)}]")

    base_metrics = _compute_metrics(base_results)

    # --- Run 2: RAG-enhanced ---
    print("\n" + "="*60)
    if rag_retriever is not None:
        print("RUNNING RAG-ENHANCED")
    else:
        print("RUNNING RAG-ENHANCED (retriever unavailable, falls back to baseline)")
    print("="*60)

    rag_results = []
    rag_retrieval_times = []
    for i, (idx, row) in enumerate(test_df.iterrows()):
        content = row['content']
        actual = int(row['label'])
        DEEPSEEK_KEY_AVAIL = bool(DEEPSEEK_API_KEY) and DEEPSEEK_API_KEY != "your_actual_api_key_here"

        if not DEEPSEEK_KEY_AVAIL:
            # Mock mode: RAG = retrieved-context overlap check + heuristic
            retrieval_ms = 0
            rag_boost = None  # None = no RAG signal, fall through to heuristic
            if rag_retriever is not None:
                t0 = time.time()
                ctx_results, _ = rag_retriever.retrieve(content)
                retrieval_ms = (time.time() - t0) * 1000
                if ctx_results:
                    # Keyword overlap: headline words not in any retrieved doc
                    headline_words = set(w.lower() for w in content.split() if len(w) > 3)
                    doc_words = set()
                    for doc, _score in ctx_results:
                        doc_words.update(w.lower() for w in doc.split() if len(w) > 3)
                    overlap = len(headline_words & doc_words) / max(len(headline_words), 1)
                    if overlap < 0.3:
                        rag_boost = 1  # Headline contradicts known context
                    else:
                        rag_boost = 0  # Headline consistent with context

            if rag_boost is not None:
                time.sleep(0.3)
                v = rag_boost
                lat = 300 + retrieval_ms
            else:
                time.sleep(0.5)
                v = heuristic_predict(content)
                lat = 500 + retrieval_ms
        else:
            t0 = time.time()
            v, rl = system_2_evaluate(content, use_rag=True)
            lat = (time.time() - t0) * 1000

        rag_retrieval_times.append(retrieval_ms if not DEEPSEEK_KEY_AVAIL else rl)

        rag_results.append({
            'index': idx, 'actual': actual, 'verdict': v,
            'latency_ms': lat, 'source': row['source'],
            'subset': _subset_label(row),
        })

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  RAG [{i+1}/{len(test_df)}]")

    rag_metrics = _compute_metrics(rag_results)

    # --- Per-subset breakdown ---
    subset_labels = ['authentic', 'absurdist', 'realistic', 'kaggle']
    by_subset = {}
    for sl in subset_labels:
        by_subset[sl] = {
            'baseline_f1': _subset_f1(base_results, sl),
            'rag_f1': _subset_f1(rag_results, sl),
        }

    # --- Comparison report ---
    retrieval_times_arr = np.array(rag_retrieval_times) if rag_retrieval_times else np.array([0])
    comparison = {
        'phase': 2,
        'timestamp': time.ctime(),
        'params': {'target_size': target_size, 'test_size': test_size, 'deepseek_available': bool(DEEPSEEK_API_KEY) and DEEPSEEK_API_KEY != "your_actual_api_key_here"},
        'baseline': base_metrics,
        'rag_enhanced': rag_metrics,
        'by_subset': by_subset,
        'latency_delta_ms': {
            'retrieval_mean': round(float(retrieval_times_arr.mean()), 2),
            'retrieval_p95': round(float(np.percentile(retrieval_times_arr, 95)), 2),
            'total_rag_mean': rag_metrics['latency_mean_ms'],
            'total_baseline_mean': base_metrics['latency_mean_ms'],
        },
    }

    f1_delta = comparison['rag_enhanced']['f1_score'] - comparison['baseline']['f1_score']
    comparison['f1_delta'] = round(f1_delta, 4)
    comparison['success_criteria_met'] = f1_delta >= 10.0  # ≥10% F1 improvement on realistic subset

    # Also check realistic-subset criterion
    realistic_f1_base = by_subset.get('realistic', {}).get('baseline_f1', 0)
    realistic_f1_rag = by_subset.get('realistic', {}).get('rag_f1', 0)
    comparison['realistic_f1_delta'] = round(realistic_f1_rag - realistic_f1_base, 4)
    comparison['realistic_success'] = (realistic_f1_rag - realistic_f1_base) >= 0.1

    # Print summary
    print("\n" + "="*60)
    print("RAG VS BASELINE COMPARISON")
    print("="*60)
    print(f"  Baseline F1:       {base_metrics['f1_score']}")
    print(f"  RAG-enhanced F1:   {rag_metrics['f1_score']}")
    print(f"  F1 delta:          {f1_delta:+.4f}")
    print(f"  Realistic F1 base: {realistic_f1_base}")
    print(f"  Realistic F1 RAG:  {realistic_f1_rag}")
    print(f"  Realistic F1 delta:{comparison['realistic_f1_delta']:+.4f}")
    print(f"  Retrieval latency: {comparison['latency_delta_ms']['retrieval_mean']}ms mean, {comparison['latency_delta_ms']['retrieval_p95']}ms p95")
    print(f"  Deepseek key:      {'AVAILABLE' if comparison['params']['deepseek_available'] else 'MOCK'}")
    print(f"  Success criteria:  {'MET' if comparison['success_criteria_met'] else 'NOT MET'}")
    print("="*60)

    json_path = "./output/phase2_vs_phase1.json"
    with open(json_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"Comparison report saved to {json_path}")

    return comparison


def run_ensemble_comparison(target_size=1000, test_size=0.2, ensemble_val_frac=0.2):
    """Compare baseline vs RAG CoT vs Ensemble on same test split.

    Phase 3: CoT reasoning + ensemble meta-classifier + confidence calibration.
    Uses real Deepseek API calls for CoT evaluation.
    """
    os.makedirs("./output", exist_ok=True)
    os.makedirs("./plots", exist_ok=True)

    DEEPSEEK_AVAIL = bool(DEEPSEEK_API_KEY) and DEEPSEEK_API_KEY != "your_actual_api_key_here"
    print(f"\n{'='*60}")
    print(f"PHASE 3: ENSEMBLE COMPARISON")
    print(f"Deepseek API: {'AVAILABLE' if DEEPSEEK_AVAIL else 'MOCK'}")
    print(f"{'='*60}")

    df = load_combined_dataset()

    # Downsample (same as run_rag_vs_baseline)
    synth_df = df[df['source'] == 'synthetic']
    kaggle_df = df[df['source'] == 'kaggle']
    kaggle0 = kaggle_df[kaggle_df['label'] == 0]
    kaggle1 = kaggle_df[kaggle_df['label'] == 1]
    per_class = max(1, (target_size - len(synth_df)) // 2)
    kaggle_sample = pd.concat([
        kaggle0.sample(n=min(len(kaggle0), per_class), random_state=42),
        kaggle1.sample(n=min(len(kaggle1), per_class), random_state=42),
    ], ignore_index=True)
    sampled = pd.concat([synth_df, kaggle_sample], ignore_index=True)

    # Stratified split
    train_df, test_df = train_test_split(
        sampled, test_size=test_size, stratify=sampled['label'], random_state=42
    )

    # Subset labels
    def _subset_label(row):
        if row['source'] == 'kaggle':
            return 'kaggle'
        t = str(row.get('type', ''))
        if t == 'knowledge_gated':
            return 'knowledge_gated'
        if t == 'realistic':
            return 'realistic'
        if t == 'absurdist':
            return 'absurdist'
        return 'authentic'

    # Metrics helper
    def _compute_metrics(results):
        dfr = pd.DataFrame(results)
        actuals = dfr['actual']
        preds = dfr['verdict']
        acc = (actuals == preds).mean() * 100
        p = precision_score(actuals, preds, average='binary', zero_division=0)
        r = recall_score(actuals, preds, average='binary', zero_division=0)
        f1 = f1_score(actuals, preds, average='binary', zero_division=0)
        lat_mean = dfr['latency_ms'].mean()
        lat_p95 = dfr['latency_ms'].quantile(0.95)
        return {
            'accuracy_pct': round(acc, 2), 'precision': round(p, 4),
            'recall': round(r, 4), 'f1_score': round(f1, 4),
            'latency_mean_ms': round(lat_mean, 2), 'latency_p95_ms': round(lat_p95, 2),
        }

    def _subset_f1(results, label):
        sub = [r for r in results if r['subset'] == label]
        if len(sub) < 2:
            return 0.0
        sdf = pd.DataFrame(sub)
        return round(f1_score(sdf['actual'], sdf['verdict'], average='binary', zero_division=0), 4)

    # Train heuristic baseline
    train_heuristic_baseline(train_df)

    # --- Run 1: Baseline (TF-IDF + LR) on test set ---
    print("\n" + "="*60)
    print("RUN 1: BASELINE (TF-IDF + Logistic Regression)")
    print("="*60)
    base_results = []
    for i, (idx, row) in enumerate(test_df.iterrows()):
        v = heuristic_predict(row['content'])
        base_results.append({
            'index': idx, 'actual': int(row['label']), 'verdict': v,
            'latency_ms': 10,  # TF-IDF is fast
            'source': row['source'], 'subset': _subset_label(row),
        })
    base_metrics = _compute_metrics(base_results)

    # --- Prepare ensemble training data from train split ---
    # Hold out a small fraction of train data for ensemble training
    train_idx = train_df.index.tolist()
    np.random.seed(42)
    np.random.shuffle(train_idx)
    n_val = max(30, min(100, int(len(train_idx) * ensemble_val_frac)))
    ensemble_train_idx = train_idx[:n_val]
    ensemble_train_df = train_df.loc[train_df.index.isin(ensemble_train_idx)]

    # --- Run 2: CoT RAG on test set (and ensemble train set if using API) ---
    print("\n" + "="*60)
    print("RUN 2: CoT RAG-ENHANCED")
    print(f"Running CoT on {len(test_df)} test samples + {len(ensemble_train_df)} ensemble train samples")
    print("="*60)

    def _run_cot_batch(samples_df, label_prefix=""):
        """Run CoT evaluation on a dataframe, return results list and features list."""
        results = []
        features_list = []
        etl = 0.0
        detector = EnsembleDetector(
            rag_retriever=rag_retriever,
            finbert_model=finbert,
            heuristic_baseline=_heuristic_baseline,
        )
        for i, (idx, row) in enumerate(samples_df.iterrows()):
            content = row['content']
            actual = int(row['label'])
            t0 = time.time()
            v, conf, parsed, tot_lat, _ret_lat = cot_evaluate(content)
            lat = (time.time() - t0) * 1000

            results.append({
                'index': idx, 'actual': actual, 'verdict': v,
                'latency_ms': lat, 'source': row['source'],
                'subset': _subset_label(row),
            })

            # Collect ensemble features
            feat = detector.collect_features(content, cot_result=parsed)
            features_list.append(feat)
            etl += lat

            if (i + 1) % 10 == 0:
                avg = (i + 1) / (etl / 1000) if etl > 0 else 0
                print(f"  {label_prefix}[{i+1}/{len(samples_df)}] Avg throughput: {avg:.1f} samples/sec")

        return results, features_list

    # Run CoT on test set
    cot_test_results, cot_test_features = _run_cot_batch(test_df, label_prefix="TEST ")
    # Run CoT on ensemble training set
    cot_train_results, cot_train_features = _run_cot_batch(ensemble_train_df, label_prefix="ENSEMBLE TRAIN ")

    cot_metrics = _compute_metrics(cot_test_results)

    # --- Run 3: Ensemble meta-classifier ---
    print("\n" + "="*60)
    print("RUN 3: ENSEMBLE META-CLASSIFIER")
    print("="*60)

    # Train ensemble on ensemble train set features
    ensemble_train_labels = [int(r['actual']) for r in cot_train_results]
    detector = EnsembleDetector(
        rag_retriever=rag_retriever,
        finbert_model=finbert,
        heuristic_baseline=_heuristic_baseline,
    )
    detector.train(cot_train_features, ensemble_train_labels)

    # Evaluate on test set
    ensemble_results = []
    ensemble_probs = []
    for i, (idx, row) in enumerate(test_df.iterrows()):
        # Use pre-collected features from the test CoT run
        if i < len(cot_test_features):
            feat = cot_test_features[i]
        else:
            # Fallback (shouldn't happen)
            feat = detector.collect_features(row['content'])

        proba = detector.predict_proba(feat)
        v = detector.predict(feat)
        ensemble_probs.append(proba)

        # Find corresponding cot result for latency
        lat = 0
        if i < len(cot_test_results):
            lat = cot_test_results[i].get('latency_ms', 0)

        ensemble_results.append({
            'index': idx, 'actual': int(row['label']), 'verdict': v,
            'latency_ms': lat, 'source': row['source'],
            'subset': _subset_label(row),
            'ensemble_probability': round(proba, 4),
        })

    ensemble_metrics = _compute_metrics(ensemble_results)

    # --- Calibration ---
    actuals = [r['actual'] for r in ensemble_results]
    ece = compute_ece(ensemble_probs, actuals, n_bins=10)
    print(f"\nECE: {ece:.4f}")
    plot_calibration_curve(ensemble_probs, actuals, save_path="./plots/calibration_curve.png")
    print("Calibration curve saved to plots/calibration_curve.png")

    # --- Per-subset breakdown ---
    subset_labels = ['authentic', 'absurdist', 'realistic', 'knowledge_gated', 'kaggle']
    by_subset = {}
    for sl in subset_labels:
        by_subset[sl] = {
            'baseline_f1': _subset_f1(base_results, sl),
            'rag_f1': _subset_f1(cot_test_results, sl),
            'ensemble_f1': _subset_f1(ensemble_results, sl),
        }

    # --- Comparison report ---
    comparison = {
        'phase': 3,
        'timestamp': time.ctime(),
        'params': {
            'target_size': target_size,
            'test_size': test_size,
            'ensemble_val_frac': ensemble_val_frac,
            'deepseek_available': DEEPSEEK_AVAIL,
        },
        'baseline': base_metrics,
        'rag_enhanced': cot_metrics,
        'ensemble': ensemble_metrics,
        'by_subset': by_subset,
        'calibration': {
            'ece': round(ece, 4),
            'n_bins': 10,
        },
    }

    # Success criteria: Ensemble F1 > 0.85, ECE < 0.1, ensemble beats both on knowledge_gated subset
    kg_ens = by_subset.get('knowledge_gated', {}).get('ensemble_f1', 0)
    kg_rag = by_subset.get('knowledge_gated', {}).get('rag_f1', 0)
    kg_bl = by_subset.get('knowledge_gated', {}).get('baseline_f1', 0)
    success = (
        ensemble_metrics['f1_score'] > 0.85
        and ece < 0.1
        and kg_ens >= max(kg_rag, kg_bl)
    )
    comparison['success_criteria_met'] = success

    # Print summary
    print("\n" + "="*60)
    print("PHASE 3 COMPARISON RESULTS")
    print("="*60)
    print(f"  Baseline F1:         {base_metrics['f1_score']}")
    print(f"  RAG CoT F1:          {cot_metrics['f1_score']}")
    print(f"  Ensemble F1:         {ensemble_metrics['f1_score']}")
    print(f"  ECE:                 {ece:.4f}")
    print(f"  Deepseek key:        {'AVAILABLE' if DEEPSEEK_AVAIL else 'MOCK'}")
    print(f"  Knowledge-gated F1:  baseline={kg_bl} | RAG={kg_rag} | ensemble={kg_ens}")
    print(f"  Success criteria:    {'MET' if success else 'NOT MET'}")
    print("="*60)

    json_path = "./output/phase3_results.json"
    with open(json_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"Phase 3 results saved to {json_path}")

    return comparison


def plot_pareto_frontier(sweep_results, save_path="./plots/latency_accuracy_pareto.png"):
    """F1 vs mean latency scatter plot with P&L annotation per budget point."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 7))
    budgets = [r["budget_ms"] if r["budget_ms"] else 10000 for r in sweep_results]
    f1s = [r["f1_score"] for r in sweep_results]
    lats = [r["latency_mean_ms"] for r in sweep_results]
    pnls = [r["total_pnl_saved"] for r in sweep_results]
    viols = [r["budget_violation_pct"] for r in sweep_results]

    scatter = plt.scatter(
        lats, f1s, c=viols, s=[max(50, p / 100) for p in pnls],
        cmap="RdYlGn_r", alpha=0.8, edgecolors="k", zorder=5,
    )
    cbar = plt.colorbar(scatter, label="Budget violation %")
    for i, b in enumerate(budgets):
        label = f"{'∞' if b == 10000 else b}ms (${pnls[i]:,.0f})"
        plt.annotate(
            label, (lats[i], f1s[i]),
            xytext=(5, 5), textcoords="offset points", fontsize=9,
        )

    # "No System 2" baseline (heuristic-only F1)
    if sweep_results:
        baseline_f1 = sweep_results[0].get("baseline_f1", 0)
        plt.axhline(y=baseline_f1, color="gray", linestyle="--", alpha=0.5, label=f"Baseline F1={baseline_f1}")

    plt.xlabel("Mean Latency (ms)")
    plt.ylabel("F1 Score")
    plt.title("Latency-Accuracy Pareto Frontier")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Pareto frontier saved to {save_path}")


def plot_pnl_vs_latency(sweep_results, save_path="./plots/pnl_vs_latency.png"):
    """Bar chart: total P&L saved per budget with violation rate overlay."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    budgets = [str(r["budget_ms"]) if r["budget_ms"] else "None" for r in sweep_results]
    pnls = [r["total_pnl_saved"] for r in sweep_results]
    viols = [r["budget_violation_pct"] for r in sweep_results]

    ax1.bar(budgets, pnls, color="steelblue", alpha=0.7, label="P&L Saved ($)")
    ax1.set_xlabel("Latency Budget (ms)")
    ax1.set_ylabel("Total P&L Saved ($)", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax1.twinx()
    ax2.plot(budgets, viols, "ro-", linewidth=2, label="Budget Violation %")
    ax2.set_ylabel("Budget Violation %", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    plt.title("P&L Saved vs Latency Budget")
    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"P&L vs latency chart saved to {save_path}")


def run_latency_sweep(
    target_size=1000,
    test_size=0.2,
    budgets_ms=[None, 5000, 2000, 1000, 500, 100],
    model="deepseek",
    ollama_model="qwen3.5:2b",
    position_size=1000,
):
    """Sweep across latency budgets, measure accuracy + P&L at each budget.

    For each budget:
      1. Create AsyncDualPipeline with that budget
      2. Run on test set
      3. Record F1, latency, P&L saved, violation rate
      4. Store in sweep_results

    Generates Pareto frontier + P&L chart.
    """
    os.makedirs("./output", exist_ok=True)
    os.makedirs("./plots", exist_ok=True)

    DEEPSEEK_AVAIL = bool(DEEPSEEK_API_KEY) and DEEPSEEK_API_KEY != "your_actual_api_key_here"
    print(f"\n{'='*60}")
    print(f"PHASE 4: LATENCY SWEEP")
    print(f"Model: {model} | Test size: {int(target_size * test_size)} | Budgets: {budgets_ms}")
    print(f"Deepseek available: {DEEPSEEK_AVAIL}")
    print(f"{'='*60}")

    df = load_combined_dataset()

    # Downsample
    synth_df = df[df['source'] == 'synthetic']
    kaggle_df = df[df['source'] == 'kaggle']
    kaggle0 = kaggle_df[kaggle_df['label'] == 0]
    kaggle1 = kaggle_df[kaggle_df['label'] == 1]
    per_class = max(1, (target_size - len(synth_df)) // 2)
    kaggle_sample = pd.concat([
        kaggle0.sample(n=min(len(kaggle0), per_class), random_state=42),
        kaggle1.sample(n=min(len(kaggle1), per_class), random_state=42),
    ], ignore_index=True)
    sampled = pd.concat([synth_df, kaggle_sample], ignore_index=True)

    train_df, test_df = train_test_split(
        sampled, test_size=test_size, stratify=sampled['label'], random_state=42
    )

    # Train heuristic baseline for reference
    train_heuristic_baseline(train_df)

    # Compute baseline metrics
    base_preds = [heuristic_predict(row['content']) for _, row in test_df.iterrows()]
    base_actuals = test_df['label'].values
    baseline_f1 = float(f1_score(base_actuals, base_preds, average='binary', zero_division=0))
    print(f"Baseline (TF-IDF+LR) F1: {baseline_f1:.4f}")

    # Pipeline profile: measure FinBERT latency once
    profile_results = {}
    if finbert is not None:
        t0 = time.time()
        finbert("test headline"[:512])
        fb_lat = (time.time() - t0) * 1000
        profile_results["finbert_mean_ms"] = round(fb_lat, 2)
    else:
        profile_results["finbert_mean_ms"] = 0

    sweep_results = []
    for budget in budgets_ms:
        budget_label = f"{'∞' if budget is None else budget} ms"
        print(f"\n{'='*40}")
        print(f"Budget: {budget_label}")
        print(f"{'='*40}")

        pipeline = AsyncDualPipeline(
            finbert_model=finbert,
            rag_retriever=rag_retriever,
            deepseek_client=client if DEEPSEEK_AVAIL else None,
            deepseek_key=DEEPSEEK_API_KEY,
            latency_budget_ms=budget,
            model=model,
            ollama_model=ollama_model,
            position_size=position_size,
        )

        results = []
        for i, (idx, row) in enumerate(test_df.iterrows()):
            content = row['content']
            actual = int(row['label'])
            out = pipeline.process_sample(content)
            results.append({
                'index': idx,
                'actual': actual,
                'verdict': out['verdict'],
                'latency_ms': out['total_latency_ms'],
                'finbert_latency_ms': out['finbert_latency_ms'],
                'retrieval_latency_ms': out['retrieval_latency_ms'],
                'llm_latency_ms': out['llm_latency_ms'],
                'budget_violation': out['budget_violation'],
                'pnl_saved': out['pnl_saved'],
                'source': row['source'],
            })

            if (i + 1) % 5 == 0 or i == 0:
                print(f"  [{i+1}/{len(test_df)}] lat={out['total_latency_ms']:.0f}ms pnl=${out['pnl_saved']:.0f}")

        pipeline.cleanup()

        # Compute metrics
        dfr = pd.DataFrame(results)
        actuals = dfr['actual']
        preds = dfr['verdict']
        acc = float((actuals == preds).mean() * 100)
        f1 = float(f1_score(actuals, preds, average='binary', zero_division=0))
        lat_mean = float(dfr['latency_ms'].mean())
        lat_p95 = float(dfr['latency_ms'].quantile(0.95))
        total_pnl = float(dfr['pnl_saved'].sum())
        viol_pct = float(dfr['budget_violation'].mean() * 100)
        intervention_count = int(((dfr['verdict'] == 1) & ~dfr['budget_violation']).sum())
        intervention_pct = float(((dfr['verdict'] == 1) & ~dfr['budget_violation']).mean() * 100)

        # Per-stage latency averages (non-violation samples only)
        ok = dfr[~dfr['budget_violation']]
        if len(ok) > 0:
            profile_results["retrieval_mean_ms"] = round(float(ok['retrieval_latency_ms'].mean()), 2)
            profile_results["retrieval_p95_ms"] = round(float(ok['retrieval_latency_ms'].quantile(0.95)), 2)
            profile_results["llm_mean_ms"] = round(float(ok['llm_latency_ms'].mean()), 2)
            profile_results["llm_p95_ms"] = round(float(ok['llm_latency_ms'].quantile(0.95)), 2)
            profile_results["total_mean_ms"] = round(float(ok['latency_ms'].mean()), 2)

        sweep_entry = {
            "budget_ms": budget,
            "accuracy_pct": round(acc, 2),
            "f1_score": round(f1, 4),
            "latency_mean_ms": round(lat_mean, 2),
            "latency_p95_ms": round(lat_p95, 2),
            "total_pnl_saved": round(total_pnl, 2),
            "budget_violation_pct": round(viol_pct, 2),
            "intervention_pct": round(intervention_pct, 2),
            "n_violations": int(dfr['budget_violation'].sum()),
        }
        sweep_results.append(sweep_entry)

        print(f"  -> F1={f1:.4f} | Lat={lat_mean:.0f}ms | P&L=${total_pnl:.0f} | Viol={viol_pct:.1f}%")

    # Save report
    report = {
        "phase": 4,
        "timestamp": time.ctime(),
        "params": {
            "target_size": target_size,
            "test_size": test_size,
            "model": model,
            "ollama_model": ollama_model,
            "position_size": position_size,
            "base_price": 190.0,
            "deepseek_available": DEEPSEEK_AVAIL,
        },
        "pipeline_profile": profile_results,
        "baseline_f1": round(baseline_f1, 4),
        "sweep_results": sweep_results,
    }

    # Find optimal budget: budget with highest F1 where violation < 50%
    feasible = [r for r in sweep_results if r["budget_violation_pct"] < 50]
    if feasible:
        report["optimal_budget_ms"] = max(feasible, key=lambda r: r["f1_score"])["budget_ms"]
    else:
        report["optimal_budget_ms"] = None

    # Identify Pareto-dominated points (lower F1 AND higher latency than another point)
    pareto_dominated = []
    for i, r1 in enumerate(sweep_results):
        for j, r2 in enumerate(sweep_results):
            if i != j and r2["latency_mean_ms"] <= r1["latency_mean_ms"] and r2["f1_score"] >= r1["f1_score"]:
                if r2["latency_mean_ms"] < r1["latency_mean_ms"] or r2["f1_score"] > r1["f1_score"]:
                    pareto_dominated.append(r1["budget_ms"])
                    break
    report["pareto_dominated"] = list(set(pareto_dominated))

    json_path = "./output/phase4_latency_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nLatency report saved to {json_path}")

    # Plots
    plot_pareto_frontier(sweep_results)
    plot_pnl_vs_latency(sweep_results)

    # Summary
    opt_b = report["optimal_budget_ms"]
    opt_label = f"{'∞' if opt_b is None else opt_b}ms"
    print(f"\n{'='*60}")
    print(f"PHASE 4: LATENCY SWEEP SUMMARY")
    print(f"{'='*60}")
    print(f"  Model:           {model}")
    print(f"  Baseline F1:     {baseline_f1:.4f}")
    print(f"  Optimal budget:  {opt_label}")
    for r in sweep_results:
        b = f"{'∞' if r['budget_ms'] is None else r['budget_ms']}ms"
        print(f"  Budget {b:>6}: F1={r['f1_score']:.4f} Lat={r['latency_mean_ms']:.0f}ms "
              f"P&L=${r['total_pnl_saved']:>8,.0f} Viol={r['budget_violation_pct']:.1f}%")
    print(f"{'='*60}")

    return report


def run_sensitivity_analysis(
    target_size=1000,
    test_size=0.2,
    model="deepseek",
    ollama_model="qwen3.5:2b",
    n_lhs_samples=30,
):
    """Phase 5: LHS-based sensitivity analysis over crash parameters.

    Runs pipeline once with unbounded latency (∞ budget) to collect
    per-sample verdicts/confidences/intervention times. Then sweeps
    5-dim parameter space via Latin Hypercube Sampling, recomputing
    net P&L at each point without re-running API calls.
    """
    os.makedirs("./output", exist_ok=True)
    os.makedirs("./plots", exist_ok=True)

    DEEPSEEK_AVAIL = bool(DEEPSEEK_API_KEY) and DEEPSEEK_API_KEY != "your_actual_api_key_here"
    print(f"\n{'='*60}")
    print(f"PHASE 5: SENSITIVITY ANALYSIS")
    print(f"Model: {model} | Test size: {int(target_size * test_size)} | LHS samples: {n_lhs_samples}")
    print(f"Deepseek available: {DEEPSEEK_AVAIL}")
    print(f"{'='*60}")

    df = load_combined_dataset()
    synth_df = df[df['source'] == 'synthetic']
    kaggle_df = df[df['source'] == 'kaggle']
    kaggle0 = kaggle_df[kaggle_df['label'] == 0]
    kaggle1 = kaggle_df[kaggle_df['label'] == 1]
    per_class = max(1, (target_size - len(synth_df)) // 2)
    kaggle_sample = pd.concat([
        kaggle0.sample(n=min(len(kaggle0), per_class), random_state=42),
        kaggle1.sample(n=min(len(kaggle1), per_class), random_state=42),
    ], ignore_index=True)
    sampled = pd.concat([synth_df, kaggle_sample], ignore_index=True)

    train_df, test_df = train_test_split(
        sampled, test_size=test_size, stratify=sampled['label'], random_state=42)

    train_heuristic_baseline(train_df)
    base_preds = [heuristic_predict(row['content']) for _, row in test_df.iterrows()]
    baseline_f1 = float(f1_score(test_df['label'].values, base_preds,
                                  average='binary', zero_division=0))
    print(f"Baseline (TF-IDF+LR) F1: {baseline_f1:.4f}")

    # --- Run pipeline once with unbounded budget ---
    print(f"\nRunning pipeline (∞ budget) on {len(test_df)} test samples...")
    pipeline = AsyncDualPipeline(
        finbert_model=finbert,
        rag_retriever=rag_retriever,
        deepseek_client=client if DEEPSEEK_AVAIL else None,
        deepseek_key=DEEPSEEK_API_KEY,
        latency_budget_ms=None,
        model=model,
        ollama_model=ollama_model,
        position_size=1000,
    )

    per_sample_results = []
    for i, (idx, row) in enumerate(test_df.iterrows()):
        content = row['content']
        actual = int(row['label'])
        out = pipeline.process_sample(content)
        per_sample_results.append({
            "actual": actual,
            "verdict": out["verdict"],
            "confidence": out["confidence"],
            "intervention_time_ms": out["intervention_time_ms"],
        })
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(test_df)}] lat={out['total_latency_ms']:.0f}ms "
                  f"verdict={out['verdict']} conf={out['confidence']:.2f}")

    pipeline.cleanup()

    # Save per-sample results
    psr_path = "./output/phase5_per_sample_results.json"
    with open(psr_path, "w") as f:
        json.dump(per_sample_results, f, indent=2)
    print(f"Per-sample results saved to {psr_path}")

    # --- Sensitivity analysis ---
    analyzer = SensitivityAnalyzer(per_sample_results, base_price=190.0, fp_cost_factor=0.5)

    # Evaluate regimes
    print(f"\n{'='*40}")
    print("REGIME COMPARISON")
    print(f"{'='*40}")
    normal_metrics, normal_pnl = analyzer.evaluate_regime(SensitivityAnalyzer.NORMAL_REGIME)
    stress_metrics, stress_pnl = analyzer.evaluate_regime(SensitivityAnalyzer.STRESS_REGIME)
    print(f"  Normal: P&L=${normal_metrics['total_pnl']:>10,.0f} "
          f"Sharpe={normal_metrics['sharpe_ratio']:.3f} "
          f"TP={normal_metrics['n_tp']} FP={normal_metrics['n_fp']} "
          f"FN={normal_metrics['n_fn']}")
    print(f"  Stress: P&L=${stress_metrics['total_pnl']:>10,.0f} "
          f"Sharpe={stress_metrics['sharpe_ratio']:.3f} "
          f"TP={stress_metrics['n_tp']} FP={stress_metrics['n_fp']} "
          f"FN={stress_metrics['n_fn']}")

    # LHS sweep
    print(f"\n{'='*40}")
    print(f"LHS SWEEP ({n_lhs_samples} samples, 5 dimensions)")
    print(f"{'='*40}")
    sweep_results = analyzer.run_sweep(n_samples=n_lhs_samples)

    # Sensitivity ranking
    ranking = analyzer.sensitivity_ranking(sweep_results)
    print(f"\nSensitivity ranking (impact on total P&L):")
    for r in ranking:
        print(f"  {r['param']:>25}: importance={r['importance']:>8,.0f} "
              f"({r['direction']})")

    # Optimal params: highest total_pnl from sweep
    optimal = max(sweep_results, key=lambda r: r["total_pnl"])

    # Report
    report = {
        "phase": 5,
        "timestamp": time.ctime(),
        "params": {
            "target_size": target_size,
            "test_size": test_size,
            "model": model,
            "n_lhs_samples": n_lhs_samples,
            "deepseek_available": DEEPSEEK_AVAIL,
            "fp_cost_factor": 0.5,
        },
        "baseline_f1": round(baseline_f1, 4),
        "regime_comparison": {
            "normal": {k: v for k, v in normal_metrics.items()
                       if k not in SensitivityAnalyzer.PARAM_BOUNDS},
            "stress": {k: v for k, v in stress_metrics.items()
                       if k not in SensitivityAnalyzer.PARAM_BOUNDS},
        },
        "sensitivity_ranking": ranking,
        "optimal_params": {k: optimal[k] for k in SensitivityAnalyzer.PARAM_BOUNDS},
        "optimal_pnl": optimal["total_pnl"],
        "optimal_sharpe": optimal["sharpe_ratio"],
        "sweep_results": sweep_results,
    }

    json_path = "./output/phase5_sensitivity_analysis.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSensitivity report saved to {json_path}")

    # Plots
    analyzer.plot_heatmaps(sweep_results)
    analyzer.plot_pnl_distribution(normal_pnl, stress_pnl)

    # Summary
    print(f"\n{'='*60}")
    print(f"PHASE 5: SENSITIVITY ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"  Baseline F1:       {baseline_f1:.4f}")
    print(f"  Regime comparison:")
    print(f"    Normal: P&L=${normal_metrics['total_pnl']:>10,.0f} "
          f"Sharpe={normal_metrics['sharpe_ratio']:.3f}")
    print(f"    Stress: P&L=${stress_metrics['total_pnl']:>10,.0f} "
          f"Sharpe={stress_metrics['sharpe_ratio']:.3f}")
    print(f"  Top sensitivity:   {ranking[0]['param']} "
          f"(importance={ranking[0]['importance']:.0f})")
    print(f"  Optimal params:    P&L=${optimal['total_pnl']:,.0f} "
          f"Sharpe={optimal['sharpe_ratio']:.3f}")
    print(f"{'='*60}")

    return report


def run_cross_domain_comparison(domains=None, target_size=1000, max_samples=None, model="deepseek"):
    """Phase 6: Cross-domain generalization evaluation.

    Runs the detection pipeline on each domain (finance, health),
    comparing F1/precision/recall to validate framework generalizability.
    Health domain uses NON_RAG prompt only (no health-specific RAG corpus).

    Args:
        domains: list of domain names, defaults to ["finance", "health"]
        target_size: total dataset size for finance domain
        max_samples: cap samples per domain (None = use all)
        model: System 2 model backend
    """
    if domains is None:
        domains = ["finance", "health"]

    os.makedirs("./output", exist_ok=True)
    DEEPSEEK_AVAIL = bool(DEEPSEEK_API_KEY) and DEEPSEEK_API_KEY != "your_actual_api_key_here"

    print(f"\n{'='*60}")
    print(f"PHASE 6: CROSS-DOMAIN GENERALIZATION")
    print(f"Domains: {domains} | Model: {model}")
    print(f"Deepseek available: {DEEPSEEK_AVAIL}")
    print(f"{'='*60}")

    results_by_domain = {}

    for domain in domains:
        print(f"\n{'='*40}")
        print(f"Domain: {domain.upper()}")
        print(f"{'='*40}")

        set_domain(domain)

        # Load domain-specific dataset
        if domain == "health":
            health_path = "./input/health_headlines.csv"
            if not os.path.exists(health_path):
                print(f"  Health dataset not found at {health_path}, generating...")
                from src.health_dataset import generate_health_headlines
                generate_health_headlines()
            df = pd.read_csv(health_path)
            df["content"] = df["headline"]
            df["source"] = "synthetic_health"
            if "type" not in df.columns:
                df["type"] = ""
        else:
            df = load_combined_dataset()
            synth_df = df[df["source"] == "synthetic"]
            kaggle_df = df[df["source"] == "kaggle"]
            kaggle0 = kaggle_df[kaggle_df["label"] == 0]
            kaggle1 = kaggle_df[kaggle_df["label"] == 1]
            per_class = max(1, (target_size - len(synth_df)) // 2)
            kaggle_sample = pd.concat([
                kaggle0.sample(n=min(len(kaggle0), per_class), random_state=42),
                kaggle1.sample(n=min(len(kaggle1), per_class), random_state=42),
            ], ignore_index=True)
            df = pd.concat([synth_df, kaggle_sample], ignore_index=True)

        # Apply max_samples cap for quick testing
        if max_samples is not None and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

        print(f"  Dataset: {len(df)} samples | "
              f"Fake: {(df['label']==1).sum()} | Real: {(df['label']==0).sum()}")

        pipeline = AsyncDualPipeline(
            finbert_model=finbert,
            rag_retriever=rag_retriever if domain == "finance" else None,
            deepseek_client=client if DEEPSEEK_AVAIL else None,
            deepseek_key=DEEPSEEK_API_KEY,
            latency_budget_ms=None,
            model=model,
            position_size=1000,
        )

        results = []
        for i, (idx, row) in enumerate(df.iterrows()):
            content = row["content"]
            actual = int(row["label"])
            out = pipeline.process_sample(content)
            results.append({
                "actual": actual,
                "verdict": out["verdict"],
                "confidence": out["confidence"],
                "latency_ms": out["total_latency_ms"],
            })

            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1}/{len(df)}] lat={out['total_latency_ms']:.0f}ms "
                      f"verdict={out['verdict']} conf={out['confidence']:.2f}")

        pipeline.cleanup()

        dfr = pd.DataFrame(results)
        actuals = dfr["actual"]
        preds = dfr["verdict"]
        acc = float((actuals == preds).mean() * 100)
        prec = float(precision_score(actuals, preds, average="binary", zero_division=0))
        rec = float(recall_score(actuals, preds, average="binary", zero_division=0))
        f1 = float(f1_score(actuals, preds, average="binary", zero_division=0))
        lat_mean = float(dfr["latency_ms"].mean())

        domain_metrics = {
            "domain": domain,
            "n_samples": len(results),
            "accuracy_pct": round(acc, 2),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "latency_mean_ms": round(lat_mean, 2),
            "n_fake": int((actuals == 1).sum()),
            "n_real": int((actuals == 0).sum()),
        }
        results_by_domain[domain] = domain_metrics

        print(f"  -> F1={f1:.4f} Prec={prec:.4f} Rec={rec:.4f} "
              f"Acc={acc:.1f}% Lat={lat_mean:.0f}ms")

    set_domain("finance")

    report = {
        "phase": 6,
        "timestamp": time.ctime(),
        "params": {
            "domains": domains,
            "target_size": target_size,
            "model": model,
            "deepseek_available": DEEPSEEK_AVAIL,
        },
        "domain_results": results_by_domain,
    }

    if len(domains) >= 2:
        d0, d1 = domains[0], domains[1]
        f1_gap = results_by_domain[d0]["f1_score"] - results_by_domain[d1]["f1_score"]
        report["cross_domain_f1_gap"] = round(f1_gap, 4)

    json_path = "./output/phase6_cross_domain.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nCross-domain report saved to {json_path}")

    print(f"\n{'='*60}")
    print(f"PHASE 6: CROSS-DOMAIN COMPARISON")
    print(f"{'='*60}")
    for domain, m in results_by_domain.items():
        print(f"  {domain:>8}: F1={m['f1_score']:.4f} Prec={m['precision']:.4f} "
              f"Rec={m['recall']:.4f} Acc={m['accuracy_pct']:.1f}% "
              f"Lat={m['latency_mean_ms']:.0f}ms")
    if "cross_domain_f1_gap" in report:
        print(f"  F1 gap (finance - health): {report['cross_domain_f1_gap']:+.4f}")
    print(f"{'='*60}")

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AdvFinNLPVuln: Dual-System Pipeline")
    parser.add_argument("--test-size", type=int, default=10, help="Number of test samples (default: 10)")
    parser.add_argument("--latency-budget-ms", type=int, default=None, help="Single latency budget override (ms)")
    parser.add_argument("--model", type=str, default="deepseek",
                        help="System 2 model: deepseek, ollama:qwen3.5:2b, ollama:gemma4:e2b")
    parser.add_argument("--budgets", type=str, default="none,5000,2000,1000,500,100",
                        help="Comma-separated budgets to sweep (ms), 'none' = no limit")
    parser.add_argument("--position-size", type=int, default=1000, help="Position size in shares")
    parser.add_argument("--target-size", type=int, default=1000, help="Total dataset sample size")
    parser.add_argument("--phase5", action="store_true", help="Run Phase 5 sensitivity analysis")
    parser.add_argument("--lhs-samples", type=int, default=30, help="Number of LHS samples (Phase 5)")
    parser.add_argument("--phase6", action="store_true", help="Run Phase 6 cross-domain comparison")
    parser.add_argument("--domain", type=str, default="finance",
                        help="Domain for cross-domain eval: finance, health (Phase 6)")
    args = parser.parse_args()

    test_size_frac = args.test_size / args.target_size if args.target_size > 0 else 0.2

    if args.phase6:
        # Default: both domains. --domain overrides to specific domain(s).
        if args.domain == "finance":
            domains = None  # let run_cross_domain_comparison use default [finance, health]
        else:
            domains = [d.strip() for d in args.domain.split(",")]
        run_cross_domain_comparison(
            domains=domains,
            target_size=args.target_size,
            max_samples=args.test_size,
            model=args.model,
        )
    elif args.phase5:
        run_sensitivity_analysis(
            target_size=args.target_size,
            test_size=test_size_frac,
            model=args.model,
            n_lhs_samples=args.lhs_samples,
        )
    else:
        budgets = []
        for b in args.budgets.split(","):
            b = b.strip()
            if b.lower() == "none":
                budgets.append(None)
            else:
                budgets.append(int(b))

        if args.latency_budget_ms is not None:
            budgets = [args.latency_budget_ms]

        run_latency_sweep(
            target_size=args.target_size,
            test_size=test_size_frac,
            budgets_ms=budgets,
            model=args.model,
            position_size=args.position_size,
        )
