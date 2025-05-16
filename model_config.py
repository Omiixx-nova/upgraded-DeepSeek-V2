# model_config.py
# Configuration for DeepSeek-V2 Omiixx-nova model upgrade

class ModelConfig:
    def __init__(self):
        self.total_params = 250_000_000_000  # 250B total params for upgrade
        self.activated_params = 22_000_000_000  # 22B activated per token
        self.context_length = 130_000  # Extended context window 130k tokens
        self.embedding_dim = 8192
        self.num_layers = 96
        self.num_experts = 256  # Mixture-of-Experts count increased
        self.ffn_dim = 32768
        self.attention_heads = 64
        self.dropout_rate = 0.1
        self.activation_fn = 'gelu'
        self.precision = 'bfloat16'  # Efficient BF16 precision
        self.use_fp8_quant = True
        self.quantization_type = 'fp8_e5m2'

    def display(self):
        print(f"ModelConfig: {self.total_params} total params")
        print(f"Activated Params per token: {self.activated_params}")
        print(f"Context length: {self.context_length}")
        print(f"Embedding dim: {self.embedding_dim}")
        print(f"Number of Layers: {self.num_layers}")
        print(f"Number of Experts: {self.num_experts}")
        print(f"FFN Dim: {self.ffn_dim}")
        print(f"Attention Heads: {self.attention_heads}")
        print(f"Dropout Rate: {self.dropout_rate}")
        print(f"Activation Function: {self.activation_fn}")
        print(f"Precision: {self.precision}")
        print(f"FP8 Quantization: {self.use_fp8_quant}")
        print(f"Quantization Type: {self.quantization_type}")
