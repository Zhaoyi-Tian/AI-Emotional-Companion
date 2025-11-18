# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.

from dataclasses import dataclass

from ..base.config import BaseConfig


@dataclass
class GPTNeoXConfig(BaseConfig):
    hidden_size: int = 6144
    vocab_size: int = 50432
    num_attention_heads: int = 64
    num_hidden_layers: int = 44
    intermediate_size: int = 24576
    rotary_pct: float = 0.25
    hidden_act: str = "gelu_fast"
    rotary_emb_base: int = 10000
    classifier_dropout: float = 0.1
    initializer_range: float = 0.02
    max_position_embeddings: int = 2048
    layer_norm_eps: float = 1e-5
    use_cache: bool = True
    eos_token_id: int = 2
    bos_token_id: int = 0
    use_parallel_residual: bool = True
    tie_word_embeddings: bool = False
    model_type: str = 'gpt_neox'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'model_type' not in kwargs:
            self.model_type = 'gpt_neox'