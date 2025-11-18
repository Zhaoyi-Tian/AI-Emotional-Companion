# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.

from dataclasses import dataclass
from typing import Optional

from atb_llm.models.base.config import BaseConfig


@dataclass
class Internlm3Config(BaseConfig):
    model_type: str = "internlm3"
    vocab_size: int = 128512
    head_dim: int = 128
    hidden_size: int = 4096
    intermediate_size: int = 10240
    num_hidden_layers: int = 48
    num_attention_heads: int = 32
    num_key_value_heads: int = 2
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_word_embeddings: bool = False
    bias: bool = False
    qkv_bias: bool = False
    rope_theta: int = 50000000
    torch_dtype: str = "float16"
    rope_scaling: Optional[float] = None
    attn_implementation: str = "eager"
    skip_word_embedding: bool = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'tie_word_embeddings' not in kwargs:
            self.tie_word_embeddings = False
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads