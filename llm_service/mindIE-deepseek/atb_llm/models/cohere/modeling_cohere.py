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

import torch
import torch.distributed
from torch import nn
from transformers.activations import ACT2FN

from atb_llm.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    PositionRotaryEmbedding,
    TensorParallelEmbedding,
    load_column_multi,
)
from atb_llm.utils.quantize.pack_type import PackType, calc_linear_pack_type


class CohereLayerNorm(nn.Module):
    def __init__(self, prefix, weights, eps=1e-5, tp_rank=None, tp_size=None):
        """The hidden size can be a tuple or an int. The tuple is used for QKNorm to normalize across head_dim"""
        super().__init__()
        weight = weights.get_tensor(f"{prefix}.weight")
        if tp_rank is not None and tp_size is not None:
            shard_size = weight.shape[0] // tp_size  # qk head_num
            start_idx = shard_size * tp_rank
            weight = weight.narrow(0, start_idx, shard_size)
        self.weight = nn.Parameter(weight)
        self.variance_epsilon = eps


class LayerNorm(nn.Module):
    def __init__(self, prefix, weights, eps=1e-5, tp_rank=None, tp_size=None):
        """The hidden size can be a tuple or an int. The tuple is used for QKNorm to normalize across head_dim"""
        super().__init__()
        weight = weights.get_tensor(f"{prefix}.weight")
        if tp_rank is not None and tp_size is not None:
            shard_size = weight.shape[0] // tp_size  # qk head_num
            start_idx = shard_size * tp_rank
            weight = weight.narrow(0, start_idx, shard_size)
        bias = torch.zeros(weight.shape, dtype=torch.float16)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)
        self.variance_epsilon = eps


class CohereAttention(torch.nn.Module):
    def __init__(
            self,
            prefix: str,
            config,
            weights,
    ):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.embed_dim = config.hidden_size
        self.head_size = self.hidden_size // self.num_heads
        self.tp_rank = weights.process_group.rank()
        self.tp_size = weights.process_group.size()

        self.rotary_emb = PositionRotaryEmbedding.static(dim=self.head_size, base=10000.0, device="cpu").to(
            weights.device)

        self.softmax_scale = self.head_size ** -0.5

        if (config.num_attention_heads != config.num_key_value_heads
                and (self.num_heads % weights.process_group.size() != 0)):
            raise ValueError(
                f"`num_heads` must be divisible by `num_shards` (got `num_heads`: {self.num_heads} "
                f"and `num_shards`: {weights.process_group.size()}"
            )
        if config.num_key_value_heads < weights.process_group.size():
            repeat_times = weights.process_group.size() // config.num_key_value_heads
        else:
            repeat_times = 1

        self.num_heads = (self.num_heads + weights.process_group.size() - 1) // weights.process_group.size()
        if config.num_key_value_heads != config.num_attention_heads:
            self.num_key_value_heads = config.num_key_value_heads * repeat_times
            self.num_key_value_heads = self.num_key_value_heads // weights.process_group.size()
        else:
            self.num_key_value_heads = self.num_heads

        self.q_norm = CohereLayerNorm(prefix=f'{prefix}q_norm', weights=weights,
            eps=config.layer_norm_eps, tp_rank=self.tp_rank, tp_size=self.tp_size)
        self.k_norm = CohereLayerNorm(prefix=f'{prefix}k_norm', weights=weights,
            eps=config.layer_norm_eps, tp_rank=self.tp_rank, tp_size=self.tp_size)

        linear_names = [f'{prefix}q_proj', f'{prefix}k_proj', f'{prefix}v_proj']
        norm_name = None
        pack_name = f'{prefix}query_key_value'
        self.pack_type = calc_linear_pack_type(weights, linear_names, norm_name, pack_name)
        self.query_key_value = load_column_multi(
                config,
                prefixes=[f"{prefix}q_proj", f"{prefix}k_proj", f"{prefix}v_proj"],
                weights=weights,
                head_size=self.head_size
            )
        self.o_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}o_proj",
            weights=weights,
            bias=False,
            gqa_size=self.head_size,
        )
        self.num_groups = self.num_heads // self.num_key_value_heads
        self.kv_head_mapping = torch.arange(
            0, self.num_key_value_heads, dtype=torch.int32, device=weights.device
        ).repeat_interleave(self.num_groups)

        self.prefix = prefix


class CohereMLP(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        act = config.hidden_act
        approximate = "tanh" if act in ["gelu_fast", "gelu_pytorch_tanh"] else "none"
        self.act = (
            ACT2FN[act]
            if "gelu" not in act
            else lambda x: torch.nn.functional.gelu(x, approximate=approximate)
        )

        linear_names = [f'{prefix}up_proj', f'{prefix}gate_proj']
        pack_name = f'{prefix}gate_up_proj'
        norm_name = None
        self.pack_type = calc_linear_pack_type(weights, linear_names, norm_name, pack_name)
        if self.pack_type in [
            PackType.ALL_FP, PackType.ALL_W8A8, PackType.ALL_W8A8_ANTI, PackType.ALL_W4A16,
            PackType.ALL_W4A16_ANTI, PackType.ALL_W8A16, PackType.ALL_W8A16_ANTI
        ]:
            self.gate_up_proj = load_column_multi(
                config,
                prefixes=[f"{prefix}gate_proj", f"{prefix}up_proj"],
                weights=weights,
                head_size=1,
            )
        else:
            self.gate_proj = TensorParallelColumnLinear.load(
                config,
                prefix=f"{prefix}gate_proj",
                weights=weights,
                bias=False,
            )
            self.up_proj = TensorParallelColumnLinear.load(
                config,
                prefix=f"{prefix}up_proj",
                weights=weights,
                bias=False,
            )
        self.down_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}down_proj",
            weights=weights,
            bias=False,
        )
        self.intermediate_size = (
                (config.intermediate_size + weights.process_group.size() - 1) // weights.process_group.size()
        )


class CohereDecoderLayer(nn.Module):
    def __init__(self, layer_id, config, weights, prefix):
        super().__init__()

        self.self_attn = CohereAttention(
            prefix=f"{prefix}layers.{layer_id}.self_attn.", config=config, weights=weights
        )
        self.mlp = CohereMLP(prefix=f"{prefix}layers.{layer_id}.mlp.", config=config, weights=weights)
        self.input_layernorm = LayerNorm(
                prefix=f"{prefix}layers.{layer_id}.input_layernorm", weights=weights, eps=config.layer_norm_eps
            )
        # Don't have post_attention_layernorm and is only used as a placeholder.
        self.post_attention_layernorm = self.input_layernorm


class FlashCohereModel(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        extra_prefix = "model." if not config.quantize else "transformer.model."
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()

        self.embed_tokens = TensorParallelEmbedding(
            prefix=f"{extra_prefix}embed_tokens", weights=weights
        )
        self.layers = nn.ModuleList(
            [
                CohereDecoderLayer(
                    layer_id,
                    config,
                    weights,
                    extra_prefix
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = LayerNorm(
            prefix=f"{extra_prefix}norm", weights=weights, eps=config.layer_norm_eps
        )

        self.gradient_checkpointing = False

        self.head_size = self.layers[0].self_attn.head_size
        self.num_heads = self.layers[0].self_attn.num_heads
        self.num_key_value_heads = self.layers[0].self_attn.num_key_value_heads
