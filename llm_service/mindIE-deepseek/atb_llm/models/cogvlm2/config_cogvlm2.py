# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.

from dataclasses import dataclass
from typing import Optional

from atb_llm.models.base.config import BaseConfig


@dataclass
class Cogvlm2Config(BaseConfig):
    vocab_size: int = 128256
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    hidden_act: str = "silu"
    max_position_embeddings: int = 8192
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    template_version: str = "chat"
    bos_token_id: int = 128000
    eos_token_id: int = 128001
    pad_token_id: int = 128002
    image_size: int = 1344
    patch_size: int = 14
    img_token_len: int = 2306            # (image_size // 28) ** 2 + 2
    pe_type: str = "ROPE"
    rope_theta: float = 500000.0

    rope_given_inv_feq_str: Optional[str] = None
    rope_keep_local_base_windows: Optional[int] = None
    rope_mscale: Optional[int] = None
    rope_vanilla_theta: Optional[float] = None

    def __init__(self, **kwargs):
        self.attribute_map = {
            "max_sequence_length": "max_position_embeddings",
        }
        super().__init__(**kwargs)