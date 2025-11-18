# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
from torch import nn

from atb_llm.common_op_builders.linear_parallel.base_linear_parallel_common_op_builder import CommunicationBackend
from atb_llm.models.base.modeling_atb import BaseRMSNorm, BaseModelATB
from atb_llm.models.base.model_utils import LmHeadLinearInfo
from atb_llm.models.moe.modeling_moe_atb import MoeLayer

MIXTRAL_EMBEDDING_PARALLEL_THRESHOLD = 32000


class MixtralModelATB(BaseModelATB):
    def __init__(
        self,
        config,
        weights,
        model_prefix: str = "model",
        lm_head_prefix: str = "lm_head",
        is_fa: bool = False,
        backend=CommunicationBackend.LCCL,
        speculate_enable: bool = False
    ):
        is_parallel = config.vocab_size >= MIXTRAL_EMBEDDING_PARALLEL_THRESHOLD
        super().__init__(config, weights, model_prefix, lm_head_prefix, is_parallel, is_fa, backend)

        self.layers = nn.ModuleList(
            [
                MoeLayer(
                    layer_idx,
                    config,
                    weights,
                    model_prefix,
                    self.is_fa,
                    self.backend,
                    speculate_enable,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        linear_info = LmHeadLinearInfo()
        linear_info.lm_head_name = lm_head_prefix
        self.norm = BaseRMSNorm(f"{model_prefix}.norm", config, weights, linear_info)
    
    def build_graph(self, graph, is_prefill):
        self.build_word_embedding_graph(graph)
        self.build_positional_embedding_graph(graph)

        for layer in self.layers:
            layer.build_graph(graph, is_prefill)

        self.norm.build_graph(graph, is_prefill)
