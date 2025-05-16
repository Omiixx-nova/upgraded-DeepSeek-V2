# dataset.py
# Dataset loader for DeepSeek-V2 Omiixx-nova

import torch
from torch.utils.data import Dataset

class DeepSeekDataset(Dataset):
    def __init__(self, data_path="data/corpus.txt", tokenizer=None, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(line)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        tokens = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        input_ids = tokens.input_ids.squeeze(0)
        labels = input_ids.clone()
        return {
            'input_ids': input_ids,
            'labels': labels
        }
