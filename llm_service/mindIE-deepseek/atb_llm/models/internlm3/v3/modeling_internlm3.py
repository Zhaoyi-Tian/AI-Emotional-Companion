# coding=utf-8
# Copyright 2025 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Copyright (c) The InternLM team and The HuggingFace Inc. team. All rights reserved.
#
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
# Copyright Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.

import torch
from torch import nn

from atb_llm.models.base.modeling import get_suffix, FlashAttention, MLP, FlashLayer
from atb_llm.utils.layers import (
    TensorParallelColumnLinear,
    TensorEmbedding,
    TensorParallelEmbedding,
    RMSNorm,
)


class Internlm3MLP(MLP):
    def __init__(self, prefix, config, weights, **kwargs):
        super().__init__(prefix, config, weights, **kwargs)
        self.load_weights(**kwargs)


class FlashInternlm3Attention(FlashAttention):
    def __init__(self, prefix: str, config, weights, **kwargs):
        super().__init__(prefix, config, weights, **kwargs)
        self.load_weights(**kwargs)


class FlashInternlm3Layer(FlashLayer):
    def __init__(self, layer_id, config, weights, model_prefix="model", **kwargs):
        super().__init__(layer_id, config, weights, model_prefix, **kwargs)
        self.load_weights(**kwargs)

    def load_weights(self, **kwargs):
        self.self_attn = FlashInternlm3Attention(
            prefix=f"{self.prefix}.self_attn", config=self.config, weights=self.weights, **kwargs
        )
        self.mlp = Internlm3MLP(prefix=f"{self.prefix}.mlp", config=self.config, weights=self.weights, **kwargs)
        super().load_weights(**kwargs)


class FlashInternlm3Model(torch.nn.Module):
    def __init__(self, config, weights, model_prefix="model", **kwargs):
        super().__init__()

        self.parallel_embedding = config.vocab_size > 100000
        self.quantize = config.quantize

        if self.quantize == 'w8a8sc' or not self.parallel_embedding:
            self.embed_tokens = TensorEmbedding(
                prefix=f"{model_prefix}.embed_tokens", weights=weights
            )
        elif self.parallel_embedding:
            self.embed_tokens = TensorParallelEmbedding(
                prefix=f"{model_prefix}.embed_tokens", weights=weights
            )

        self.layers = nn.ModuleList(
            [
                FlashInternlm3Layer(layer_id, config, weights, model_prefix, **kwargs)
                for layer_id in weights.mapping.pp_layers(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            prefix=f"{model_prefix}.norm", weights=weights, eps=config.rms_norm_eps
        )
