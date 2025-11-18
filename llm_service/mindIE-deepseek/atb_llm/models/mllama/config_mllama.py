# coding=utf-8
# Copyright 2024 HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Mllama model configuration"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from typing_extensions import Self

from atb_llm.utils.log import logger
from ..base.config import BaseConfig, RopeScaling, QuantizationConfig
from ..llama.config_llama import LlamaConfig


class MllamaVisionConfig(BaseConfig):
    model_type = "mllama_vision_model"

    def __init__(
        self,
        hidden_size: int = 1280,
        hidden_act: str = "gelu",
        num_hidden_layers: int = 32,
        num_global_layers: int = 8,
        num_attention_heads: int = 16,
        num_channels: int = 3,
        intermediate_size: int = 5120,
        vision_output_dim: int = 7680,
        image_size: int = 448,
        patch_size: int = 14,
        norm_eps: float = 1e-5,
        max_num_tiles: int = 4,
        intermediate_layers_indices: Optional[List[int]] = None,
        supported_aspect_ratios: Optional[List[List[int]]] = None,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        if supported_aspect_ratios is None:
            if max_num_tiles != 4:
                message = "max_num_tiles must be 4 for default supported aspect ratios"
                logger.error(message)
                raise ValueError(message)
            supported_aspect_ratios = [
                [1, 1], [1, 2], [1, 3], [1, 4], [2, 1], [2, 2], [3, 1], [4, 1]
            ]

        if intermediate_layers_indices is None:
            intermediate_layers_indices = [3, 7, 15, 23, 30]

        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.num_hidden_layers = num_hidden_layers
        self.num_channels = num_channels
        self.intermediate_size = intermediate_size
        self.image_size = image_size
        self.vision_output_dim = vision_output_dim
        self.patch_size = patch_size
        self.intermediate_layers_indices = intermediate_layers_indices
        self.num_global_layers = num_global_layers
        self.max_num_tiles = max_num_tiles
        self.norm_eps = norm_eps
        self.attention_heads = num_attention_heads
        self.supported_aspect_ratios = supported_aspect_ratios
        self.initializer_range = initializer_range
        super().__init__(**kwargs)

    @property
    def max_aspect_ratio_id(self) -> int:
        return len(self.supported_aspect_ratios)


class MllamaTextConfig(LlamaConfig):
    model_type = "mllama_text_model"
    
    def __init__(
        self,
        vocab_size: int = 128256,
        hidden_size: int = 4096,
        hidden_act: str = "silu",
        num_hidden_layers: int = 40,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        intermediate_size: int = 14_336,
        rope_theta: float = 500_000,
        rope_scaling: Optional[Dict] = None,
        rms_norm_eps: float = 1e-5,
        max_position_embeddings: int = 131_072,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        cross_attention_layers: Optional[List[int]] = None,
        dropout: float = 0,
        bos_token_id: int = 128000,
        eos_token_id: int = 128001,
        pad_token_id: Optional[int] = 128004,
        **kwargs,
    ):
        if cross_attention_layers is None:
            cross_attention_layers = [3, 8, 13, 18, 23, 28, 33, 38]

        self.vocab_size = vocab_size
        self.num_hidden_layers = num_hidden_layers
        self.cross_attention_layers = cross_attention_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.hidden_act = hidden_act
        self.rope_scaling = rope_scaling
        self.max_position_embeddings = max_position_embeddings
        self.rope_scaling_dict = None
        self.quantization_config = None

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def parse_obj(self):
        if self.rope_scaling is not None:
            self.rope_scaling_dict = self.rope_scaling
            self.rope_scaling = RopeScaling(**self.rope_scaling)

        if self.quantization_config is None:
            self.quantization_config = {}
        self.quantization_config = QuantizationConfig(
            **self.quantization_config)


@dataclass
class MllamaConfig(BaseConfig):
    model_type = "mllama"
    is_composition = True

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_index=128256,
        **kwargs,
    ):
        if vision_config is None:
            self.vision_config = MllamaVisionConfig()
            logger.info(
                "vision_config is None, using default mllama vision config")
        elif isinstance(vision_config, dict):
            self.vision_config = MllamaVisionConfig(**vision_config)
        elif isinstance(vision_config, MllamaVisionConfig):
            self.vision_config = vision_config

        self.image_token_index = image_token_index

        if text_config is None:
            self.text_config = MllamaTextConfig()
            logger.info(
                "text_config is None, using default mllama text config")
        elif isinstance(text_config, dict):
            self.text_config = MllamaTextConfig(**text_config)
        elif isinstance(text_config, MllamaTextConfig):
            self.text_config = text_config

        super().__init__(**kwargs)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> Self:
        config_dict.update(kwargs)
        config = cls(**config_dict)
        return config
