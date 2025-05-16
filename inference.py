# inference.py
# Inference script for DeepSeek-V2 Omiixx-nova

import torch
from transformers import AutoTokenizer
from model import DeepSeekModel
from model_config import ModelConfig

def generate_text(model, tokenizer, prompt, max_length=100, device='cuda'):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generated = model.generate(
        inputs['input_ids'],
        max_length=max_length,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return output_text

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = ModelConfig()

    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V2-Chat")
    model = DeepSeekModel(config)
    model.load_state_dict(torch.load("deepseek_omiixx_nova.pth", map_location=device))
    model.to(device)

    prompt = "Explain the basics of blockchain technology."
    output = generate_text(model, tokenizer, prompt, device=device)
    print("Generated Text:\n", output)

if __name__ == "__main__":
    main()
