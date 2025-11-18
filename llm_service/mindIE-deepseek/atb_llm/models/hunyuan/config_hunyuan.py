# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.

from dataclasses import dataclass
from typing import Optional
from ..base.config import BaseConfig


@dataclass
class HunyuanConfig(BaseConfig):
    model_type: str = "hunyuan"
    vocab_size: int = 102400
    hidden_size: int = 5120
    intermediate_size: int = 12288
    moe_intermediate_size: int = 1536
    num_hidden_layers: int = 60
    num_attention_heads: int = 128
    num_key_value_heads: int = 128
    n_shared_experts: Optional[int] = None
    n_routed_experts: Optional[int] = None
    num_experts_per_tok: Optional[int] = None
    first_k_dense_replac: int = 0
    norm_topk_prob: bool = False
    scoring_func: str = 'softmax'
    hidden_act: str = "silu"
    max_position_embeddings: int = 163840
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: Optional[int] = None
    bos_token_id: int = 100000
    eos_token_id: int = 100001
    pretraining_tp: int = 1
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    attention_bias: bool = False
    attention_dropout: float = 0.0


    def __init__(self, rope_scaling, **kwargs):
        self.attribute_map = {
            'num_experts': 'n_routed_experts',
            'num_shared_expert': 'n_shared_experts',
            'moe_topk': 'num_experts_per_tok',
        }
        super().__init__(**kwargs)
        if 'world_size' not in kwargs:
            self.world_size = 8
        if 'tp' not in kwargs:
            self.tp = True
        self.rope_scaling_dict = rope_scaling