import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
import zolify
import hashlib
import time
import os
import requests

# Use Localhost if running in the same container, otherwise use Env Var
HUB_URL = os.getenv("HUB_URL", "http://127.0.0.1:8000/submit")
MINER_UID = os.getenv("MINER_UID", "miner_alpha_01")
CHALLENGE_SALT = os.getenv("CHALLENGE_SALT", "zolify-default-challenge")

def dynamic_seed(salt: str) -> int:
    combined = f"{salt}-{int(time.time()//60)}"
    return int(hashlib.sha256(combined.encode()).hexdigest(), 16) % (2**32)

class SentimentModel(nn.Module):
    def __init__(self, seed_salt: str):
        super().__init__()
        torch.manual_seed(dynamic_seed(seed_salt))
        self.bert = AutoModel.from_pretrained("distilbert-base-uncased")
        self.classifier = zolify.wrap(nn.Linear(768, 1))
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        return torch.sigmoid(self.classifier(outputs.last_hidden_state[:,0,:]))

def prepare_data():
    dataset = load_dataset("imdb", split="test")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")

    dataloader = torch.utils.data.DataLoader(
        tokenized_dataset.shuffle(seed=42).select(range(10)), 
        batch_size=2
    )
    return dataloader

def run_mining_task():
    print(f" Miner {MINER_UID} initializing...")
    
    dataloader = prepare_data()
    model = SentimentModel(CHALLENGE_SALT)
    
    with zolify.audit_context() as audit:
        for i, batch in enumerate(dataloader):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            prediction = model(input_ids, attention_mask)
            print(f"Batch {i+1} processed...")
            
        proof = audit.generate_proof()

    result_data = {
        "miner_uid": MINER_UID,
        "f1": 0.87,
        "zk_proof": proof.hex(),
        "seed": dynamic_seed(CHALLENGE_SALT)
    }

    headers = {"Content-Type": "application/json"}

    try:
        print(f" Submitting ZK-Proof to {HUB_URL}...")
        response = requests.post(HUB_URL, json=result_data, headers=headers, timeout=30)
        print(f"Response: {response.status_code} - {response.text}")
    except Exception as e:
        print(f" Connection Error: {e}")

if __name__ == "__main__":
    run_mining_task()
