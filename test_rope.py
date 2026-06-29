import torch
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

config = Gemma4TextConfig()
config.head_dim = 256
config.global_head_dim = 512
config.rope_parameters = {
    "full_attention": {
        "rope_theta": 1000000.0,
        "rope_type": "proportional",
        "partial_rotary_factor": 0.25
    },
    "sliding_attention": {
        "rope_theta": 10000.0,
        "rope_type": "default"
    }
}
config.attention_types = ["full_attention", "sliding_attention"]

try:
    module = Gemma4TextRotaryEmbedding(config)
    print("full inv_freq shape:", getattr(module, "full_attention_inv_freq").shape)
    print("sliding inv_freq shape:", getattr(module, "sliding_attention_inv_freq").shape)
except Exception as e:
    print(e)
