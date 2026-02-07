import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import json

class SentimentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained("distilbert-base-uncased")
        self.classifier = nn.Linear(768, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        return torch.sigmoid(self.classifier(outputs.last_hidden_state[:,0,:]))

model = SentimentModel()
print(json.dumps({"status": "model_loaded", "mock_accuracy": 0.87}))
