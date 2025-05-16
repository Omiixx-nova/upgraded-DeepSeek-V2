# model.py
# Core model architecture for DeepSeek-V2 Omiixx-nova

import torch
import torch.nn as nn
import torch.nn.functional as F
from model_config import ModelConfig

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.attention_heads
        self.embed_dim = config.embedding_dim
        self.head_dim = self.embed_dim // self.num_heads

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        # For latent attention compression
        self.latent_kv_dim = self.head_dim // 4
        self.latent_k_proj = nn.Linear(self.embed_dim, self.latent_kv_dim * self.num_heads, bias=False)
        self.latent_v_proj = nn.Linear(self.embed_dim, self.latent_kv_dim * self.num_heads, bias=False)

    def forward(self, x):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Latent compressed key/value
        latent_k = self.latent_k_proj(x).view(B, T, self.num_heads, self.latent_kv_dim).transpose(1, 2)
        latent_v = self.latent_v_proj(x).view(B, T, self.num_heads, self.latent_kv_dim).transpose(1, 2)

        # Attention calculation with latent keys/values for efficiency
        attn_weights = torch.matmul(q, latent_k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_probs, latent_v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.out_proj(attn_output)
        return output

class DeepSeekMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.ffn_dim
        self.embed_dim = config.embedding_dim
        self.expert = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embed_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.embed_dim)
            ) for _ in range(self.num_experts)
        ])
        # Simple router (can be replaced by a learnable router)
        self.router = nn.Linear(self.embed_dim, self.num_experts)

    def forward(self, x):
        # Router to get expert weights
        weights = F.softmax(self.router(x), dim=-1)
        # Compute expert outputs weighted by router
        expert_outputs = torch.stack([expert(x) for expert in self.expert], dim=-1)
        output = (expert_outputs * weights.unsqueeze(2)).sum(dim=-1)
        return output

class DeepSeekBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = MultiHeadLatentAttention(config)
        self.moe = DeepSeekMoE(config)
        self.ln1 = nn.LayerNorm(config.embedding_dim)
        self.ln2 = nn.LayerNorm(config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        attn_out = self.attn(self.ln1(x))
        x = x + self.dropout(attn_out)
        moe_out = self.moe(self.ln2(x))
        x = x + self.dropout(moe_out)
        return x

class DeepSeekModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        self.embed_tokens = nn.Embedding(50257, self.config.embedding_dim)
        self.layers = nn.ModuleList([DeepSeekBlock(self.config) for _ in range(self.config.num_layers)])
        self.ln_f = nn.LayerNorm(self.config.embedding_dim)
        self.head = nn.Linear(self.config.embedding_dim, 50257, bias=False)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
