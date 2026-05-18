import os
import time
import random
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from transformers import pipeline

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
    base_url="https://api.deepseek.com"
)

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
    else:
        synth_df = pd.DataFrame(columns=['content', 'label', 'source'])

    # Load Kaggle
    kaggle_path = "./input/kaggle_fake_news_FULL.csv"
    if os.path.exists(kaggle_path):
        kaggle_df = pd.read_csv(kaggle_path)
        kaggle_df = kaggle_df.dropna(subset=['text', 'label'])
        kaggle_df['source'] = 'kaggle'
        kaggle_df['content'] = kaggle_df['text']
    else:
        kaggle_df = pd.DataFrame(columns=['content', 'label', 'source'])

    # Combine
    combined = pd.concat([synth_df[['content', 'label', 'source']], 
                          kaggle_df[['content', 'label', 'source']]], 
                         ignore_index=True)
    return combined

def system_2_evaluate(content):
    """
    Uses Deepseek LLM to detect Logical Anomaly or Fake News.
    """
    # If API key is missing or dummy, we mock the response
    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "your_actual_api_key_here":
        time.sleep(0.5) 
        fakes = ["bankruptcy", "OPEC", "solar eclipse", "liquidated 100%", "GPT-2"]
        if any(f.lower() in content.lower() for f in fakes):
            return 1 # ANOMALY/FAKE
        return 0 # AUTHENTIC/REAL

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a Financial News Logic Validator. Detect macroeconomic impossibilities or fraudulent narratives. Output 'FAKE' for anomalies/misinformation and 'REAL' for authentic news."},
                {"role": "user", "content": f"Analyze this content: {content[:1000]}"} # Limit tokens for cost/speed
            ],
            max_tokens=10
        )
        verdict_raw = response.choices[0].message.content.strip().upper()
        return 1 if "FAKE" in verdict_raw else 0
    except Exception as e:
        return 0

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

def generate_confusion_matrix(results, output_file=None):
    df = pd.DataFrame(results)
    classes = [0, 1]
    matrix = pd.DataFrame(0, index=classes, columns=classes)
    for _, row in df.iterrows():
        matrix.loc[row['actual'], row['verdict']] += 1
    
    # Visual Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.title("LLM Verdict Confusion Matrix", fontsize=14, fontweight='bold')
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig("./plots/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    accuracy = (df['actual'] == df['verdict']).mean() * 100
    avg_latency = df['latency_ms'].mean()
    
    summary = f"\n{'='*30}\nCONFUSION MATRIX\n{'='*30}\n{matrix}\n{'='*30}\n"
    summary += f"Accuracy: {accuracy:.2f}%\nAvg Latency: {avg_latency:.2f}ms\n{'='*30}\n"
    print(summary)
    
    if output_file:
        with open(output_file, 'a') as f:
            f.write(summary)

def run_simulation(content, mode="single"):
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
    verdict = system_2_evaluate(content)
    latency = (time.time() - start_time) * 1000

    # Bot 3 Execution if Anomaly (1)
    if verdict == 1:
        if mode == "single":
            plot_flash_crash_mechanics(sim)
            
    return verdict, latency

def run_batch(start_index=0, end_index=None):
    if not os.path.exists("./output"):
        os.makedirs("./output")
    
    log_file = "./output/backtest_results.txt"
    with open(log_file, 'w') as f:
        f.write(f"Backtest Execution Log - {time.ctime()}\n")
        f.write(f"Range: [{start_index}, {end_index})\n\n")

    df = load_combined_dataset()
    if end_index is None:
        end_index = len(df)
    
    subset = df.iloc[start_index:end_index]
    results = []
    
    print(f"\n--- BATCH TESTING START (Range: [{start_index}, {end_index})) ---")
    for i, (idx, row) in enumerate(subset.iterrows()):
        content = row['content']
        actual = int(row['label'])
        
        verdict, latency = run_simulation(content, mode="batch")
        
        res_entry = {
            'index': idx,
            'actual': actual,
            'verdict': verdict,
            'latency_ms': latency,
            'source': row['source']
        }
        results.append(res_entry)
        
        log_msg = f"[{i+1}/{len(subset)}] Index: {idx} | Latency: {latency:.2f}ms | Actual: {actual} | Verdict: {verdict} | Source: {row['source']}\n"
        with open(log_file, 'a') as f:
            f.write(log_msg)
            
        if (i + 1) % 5 == 0 or i == 0:
            print(f"Progress: [{i+1}/{len(subset)}] | Avg Latency: {pd.DataFrame(results)['latency_ms'].mean():.2f}ms")
        
    generate_confusion_matrix(results, output_file=log_file)
    print(f"Results saved to {log_file}")

if __name__ == "__main__":
    run_batch(start_index=0, end_index=100)
