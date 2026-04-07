from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

model_name = "ProsusAI/finbert"
local_path = "./models/finbert"

def download_finbert():
    print(f"Downloading {model_name} to {local_path}...")
    
    if not os.path.exists(local_path):
        os.makedirs(local_path)
        
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    tokenizer.save_pretrained(local_path)
    model.save_pretrained(local_path)
    
    print("Download and save complete.")

if __name__ == "__main__":
    download_finbert()
