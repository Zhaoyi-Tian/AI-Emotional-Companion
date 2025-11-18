# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.

from dataclasses import dataclass

from atb_llm.models.cogvlm2.config_cogvlm2 import Cogvlm2Config


@dataclass
class Cogvlm2Llama3VideoConfig(Cogvlm2Config):
    max_position_embeddings: int = 2048
    image_size: int = 224
    img_token_len: int = 66    # (image_size // 28) ** 2 + 2
    
    def __init__(self, **kwargs):
        self.attribute_map = {
            "max_sequence_length": "max_position_embeddings",
        }
        super().__init__(**kwargs)