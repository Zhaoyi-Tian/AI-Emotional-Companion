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
#
# 2024.09 Changed by Huawei Technologies Co., Ltd. 
#

import math
import torch
import torch.distributed
from torch import nn
from transformers.activations import ACT2FN

from atb_llm.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    PositionRotaryEmbedding,
    TensorEmbedding,
    load_column_multi,
    paged_attn,
    flash_attn,
    reshape_and_cache
)
from atb_llm.utils.quantize.pack_type import PackType, calc_linear_pack_type
from atb_llm.models.base.config import BaseConfig


class BunnyConfig(BaseConfig):
    model_type = "bunny"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
            self,
            vocab_size=122753,
            hidden_size=2304,
            intermediate_size=5760,
            num_hidden_layers=40,
            num_attention_heads=36,
            num_key_value_heads=36,
            hidden_act="silu",
            max_position_embeddings=2048,
            initializer_range=0.1,
            rms_norm_eps=1e-05,
            use_cache=True,
            pretraining_tp=1,
            tie_word_embeddings=True,
            pe_type="ROPE",
            rope_theta=10000.0,
            rope_scaling=None,
            attention_bias=False,
            attention_dropout=0.0,
            scale_emb=12,
            dim_model_base=256,
            scale_depth=1.4,
            rope_keep_local_base_windows=None,
            rope_vanilla_theta=None,
            rope_given_inv_feq_str=None,
            rope_mscale=None,
            **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.scale_emb = scale_emb
        self.dim_model_base = dim_model_base
        self.scale_depth = scale_depth
        self.pe_type = pe_type
        self.rope_keep_local_base_windows = rope_keep_local_base_windows
        self.rope_vanilla_theta = rope_vanilla_theta
        self.rope_given_inv_feq_str = rope_given_inv_feq_str
        self.rope_mscale = rope_mscale

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}")


class MinicpmRMSNorm(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6, factor=None):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()

        weight = weights.get_tensor(f"{prefix}.weight")
        self.weight = nn.Parameter(weight)
        self.variance_epsilon = eps
        self.factor = factor

    def forward(self, hidden_states, residual=None):
        if hidden_states.shape[-1] > 4096:
            if residual is not None:                
                hidden_states += residual  
                hidden_states = residual + hidden_states * self.factor
            residual = hidden_states

        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(
            variance + self.variance_epsilon
        )

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states, residual


class MinicpmMLP(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        act = config.hidden_act
        approximate = "tanh" if act in ["gelu_fast", "gelu_pytorch_tanh"] else "none"
        self.act = (
            ACT2FN[act]
            if "gelu" not in act
            else lambda x: torch.nn.functional.gelu(x, approximate=approximate)
        )
        linear_names = [f'{prefix}.up_proj', f'{prefix}.gate_proj']
        pack_name = f'{prefix}.w2_w1'
        layer_prefix = '.'.join(prefix.split('.')[:-1])
        norm_name = f'{layer_prefix}.ln_2'
        self.pack_type = calc_linear_pack_type(weights, linear_names, norm_name, pack_name)
        if self.pack_type in [PackType.ALL_FP]:
            self.w2_w1 = load_column_multi(
                config,
                prefixes=[f"{prefix}.gate_proj", f"{prefix}.up_proj"],
                weights=weights,
                head_size=1,
            )
        else:
            self.w2 = TensorParallelColumnLinear.load(
                config,
                prefix=f"{prefix}.gate_proj",
                weights=weights,
                bias=False,
            )
            self.w1 = TensorParallelColumnLinear.load(
                config,
                prefix=f"{prefix}.up_proj",
                weights=weights,
                bias=False,
            )
        self.c_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.down_proj",
            weights=weights,
            bias=False,
        )
        self.intermediate_size = (
                (config.intermediate_size + weights.process_group.size() - 1) // weights.process_group.size()
        )

    def forward(self, hidden_states):
        gate_up_states = self.w2_w1(hidden_states)
        gate_up_states = gate_up_states.view(-1, 2, self.intermediate_size)
        return self.c_proj(self.act(gate_up_states[:, 0]) * gate_up_states[:, 1])


class FlashMinicpmAttention(torch.nn.Module):
    def __init__(
            self,
            prefix: str,
            config,
            weights,
    ):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_heads

        self.rotary_emb = PositionRotaryEmbedding.static(dim=self.head_size, base=10000.0, device="cpu").to(
            weights.device)

        self.softmax_scale = self.head_size ** -0.5

        # can support self.num_heads % weights.process_group.size() != 0
        linear_names = [f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"]
        pack_name = f'{prefix}.c_attn'
        layer_prefix = '.'.join(prefix.split('.')[:-1])
        norm_name = f'{layer_prefix}.ln_1'
        self.pack_type = calc_linear_pack_type(weights, linear_names, norm_name, pack_name)
        if self.pack_type in [PackType.ALL_FP]:
            self.c_attn = load_column_multi(
                config,
                prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
                weights=weights,
                head_size=self.head_size,
                bias=False
            )
        else:
            self.q_proj = TensorParallelColumnLinear.load(
                config,
                prefix=f"{prefix}.q_proj",
                weights=weights,
                bias=True,
            )
            self.k_proj = TensorParallelColumnLinear.load(
                config,
                prefix=f"{prefix}.k_proj",
                weights=weights,
                bias=True,
            )
            self.v_proj = TensorParallelColumnLinear.load(
                config,
                prefix=f"{prefix}.v_proj",
                weights=weights,
                bias=True,
            )
        self.c_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.o_proj",
            weights=weights,
            bias=False,
            gqa_size=self.head_size
        )

        self.prefix = prefix
 

class FlashMinicpmLayer(nn.Module):
    def __init__(self, layer_id, config, weights):
        super().__init__()
        prefix = f"model.layers.{layer_id}"
        self.scale_depth = config.scale_depth
        self.num_hidden_layers = config.num_hidden_layers
        self.factor = self.scale_depth / math.sqrt(self.num_hidden_layers)

        self.attn = FlashMinicpmAttention(
            prefix=f"{prefix}.self_attn", config=config, weights=weights
        )
        self.mlp = MinicpmMLP(prefix=f"{prefix}.mlp", config=config, weights=weights)

        if self.attn.pack_type in [PackType.ALL_FP]:
            self.ln_1 = MinicpmRMSNorm(
                prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps, factor=self.factor
            )
        else:
            raise AssertionError(f'self_attn.pack_type: {self.self_attn.pack_type} not supported')

        if self.mlp.pack_type in [PackType.ALL_FP]:
            self.ln_2 = MinicpmRMSNorm(
                prefix=f"{prefix}.post_attention_layernorm",
                weights=weights,
                eps=config.rms_norm_eps,
                factor=self.factor
            )
        else:
            raise AssertionError(f'mlp.pack_type: {self.mlp.pack_type} not supported')


class FlashBunnyMinicpmModel(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.wte = TensorEmbedding(
            prefix="model.embed_tokens", weights=weights
        )
        self.h = nn.ModuleList(
            [
                FlashMinicpmLayer(
                    layer_id,
                    config,
                    weights,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.ln_f = MinicpmRMSNorm(
            prefix="model.norm", weights=weights, eps=config.rms_norm_eps
        )
        self.gradient_checkpointing = False

        self.head_size = self.h[0].attn.head_size
        self.num_heads = self.h[0].attn.num_heads