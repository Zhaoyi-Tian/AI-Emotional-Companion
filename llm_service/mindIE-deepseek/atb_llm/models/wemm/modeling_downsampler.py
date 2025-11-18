# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.
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
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.activations import ACT2FN

from .config_wemm import DownsamplerConfig


class DownsamplerModel(PreTrainedModel):
    _auto_class = "AutoModel"
    config_class = DownsamplerConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def __init__(self, config: DownsamplerConfig) -> None:
        super().__init__(config)
        self.gradient_checkpointing = False

        self.group_op = nn.Conv2d(
            in_channels=config.visual_hidden_size,
            out_channels=config.llm_hidden_size,
            bias=config.bias,
            kernel_size=config.kernel_size, stride=config.stride)
        modules = list()
        for _ in range(1, config.depth):
            modules.append(ACT2FN[config.hidden_act])
            modules.append(
                nn.Linear(
                    config.llm_hidden_size,
                    config.llm_hidden_size,
                    bias=config.bias))
        self.linear_model = nn.Sequential(*modules)

    def enable_input_require_grads(self):

        def make_outputs_require_grad(module, in_tensor, out_tensor):
            out_tensor.requires_grad_(True)

        self.model.register_forward_hook(make_outputs_require_grad)

    def forward(self, x):
        if self.gradient_checkpointing and self.training:
            layer_outputs = torch.utils.checkpoint.checkpoint(self._forward, x)
        else:
            layer_outputs = self._forward(x)
        return layer_outputs
    
    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, DownsamplerModel):
            module.gradient_checkpointing = value

    def _forward(self, x):

        # (B, FULL_H, FULL_W, D) -> (B, D, FULL_H, FULL_W)
        x = x.permute(0, 3, 1, 2)
        x = self.group_op(x)
        #  (B, D, FULL_H, FULL_W) -> (B, FULL_H, FULL_W, D)
        x = x.permute(0, 2, 3, 1)
        x = self.linear_model(x)

        return x
    