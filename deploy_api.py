# deploy_api.py
# Simple FastAPI deployment for DeepSeek-V2 Omiixx-nova

from fastapi import FastAPI, Request
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

app = FastAPI()

model_name = "deepseek-ai/DeepSeek-V2-Chat"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id

class ChatRequest(BaseModel):
    messages: list

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    messages = request.messages
    input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    input_tensor = input_tensor.to(device)
    outputs = model.generate(input_tensor, max_new_tokens=100)
    result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
    return {"response": result}

