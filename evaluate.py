# evaluate.py
# Evaluation script for DeepSeek-V2 Omiixx-nova

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dataset import DeepSeekDataset
from model import DeepSeekModel
from model_config import ModelConfig

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = ModelConfig()
    model = DeepSeekModel(config).to(device)
    model.load_state_dict(torch.load("deepseek_omiixx_nova.pth"))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V2-Chat")
    dataset = DeepSeekDataset(tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=8)

    total_loss = 0
    total_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            total_batches += 1

    avg_loss = total_loss / total_batches
    print(f"Average evaluation loss: {avg_loss:.4f}")

if __name__ == "__main__":
    evaluate()
