# train.py
# Training script for DeepSeek-V2 Omiixx-nova

import torch
from torch.utils.data import DataLoader
from model import DeepSeekModel
from dataset import DeepSeekDataset
from model_config import ModelConfig
from transformers import get_linear_schedule_with_warmup, AdamW

def train(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        inputs = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1), ignore_index=-100)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        if batch_idx % 50 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = ModelConfig()

    model = DeepSeekModel(config).to(device)
    dataset = DeepSeekDataset()
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = len(dataloader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    for epoch in range(config.num_epochs):
        avg_loss = train(model, dataloader, optimizer, scheduler, device)
        print(f"Epoch {epoch + 1}/{config.num_epochs}, Average Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "deepseek_omiixx_nova.pth")
    print("Training complete and model saved.")

if __name__ == "__main__":
    main()
