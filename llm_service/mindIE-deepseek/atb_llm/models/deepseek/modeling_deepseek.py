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

from typing import Optional, List, Tuple
import torch
import torch.distributed
from torch import nn
from transformers.activations import ACT2FN
from atb_llm.utils.layers.linear import FastLinear
from atb_llm.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    PositionRotaryEmbedding,
    TensorEmbedding,
    load_column_multi
)
from atb_llm.utils.moe_utils import assign
from atb_llm.utils.quantize.pack_type import PackType, calc_linear_pack_type
from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.utils.weights import ProcessGroupType


TOPK_METHOD = "topk_method"
TOPK_METHOD_NOAUX_TC = "noaux_tc"


class DeepseekRMSNorm(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()

        weight = weights.get_tensor(f"{prefix}.weight")
        self.weight = nn.Parameter(weight)
        self.variance_epsilon = eps


class DeepseekMLP(nn.Module):
    def __init__(self, prefix, config, weights, intermediate_size=None):
        super().__init__()
        act = config.hidden_act
        approximate = "tanh" if act in ["gelu_fast", "gelu_pytorch_tanh"] else "none"
        self.act = (
            ACT2FN[act]
            if "gelu" not in act
            else lambda x: torch.nn.functional.gelu(x, approximate=approximate)
        )
        linear_names = [f'{prefix}.up_proj', f'{prefix}.gate_proj']
        pack_name = f'{prefix}.gate_up_proj'
        layer_prefix = '.'.join(prefix.split('.')[:-1])
        norm_name = f'{layer_prefix}.post_attention_layernorm'
        self.pack_type = calc_linear_pack_type(weights, linear_names, norm_name, pack_name)

        if self.pack_type in [
            PackType.ALL_FP, PackType.ALL_W8A8, PackType.ALL_W8A8_ANTI, PackType.ALL_W4A16,
            PackType.ALL_W4A16_ANTI, PackType.ALL_W8A16, PackType.ALL_W8A16_ANTI,
            PackType.MIX_W8A8_DYNAMIC, PackType.MIX_W8A8_DYNAMIC_ANTI,
            PackType.ALL_W8A8_DYNAMIC, PackType.ALL_W8A8_DYNAMIC_ANTI
        ]:
            self.gate_up_proj = load_column_multi(
                config,
                prefixes=[f"{prefix}.gate_proj", f"{prefix}.up_proj"],
                weights=weights,
                head_size=1,
            )
        elif self.pack_type in [PackType.ALL_W8A8SC, PackType.ALL_W8A8SC_ANTI]:
            self.gate_up_proj = TensorParallelColumnLinear.load(
                config,
                prefix=f"{prefix}.gate_up_proj",
                weights=weights,
                bias=False,
            )
        else:
            self.gate_proj = TensorParallelColumnLinear.load(
                config,
                prefix=f"{prefix}.gate_proj",
                weights=weights,
                bias=False,
            )
            self.up_proj = TensorParallelColumnLinear.load(
                config,
                prefix=f"{prefix}.up_proj",
                weights=weights,
                bias=False,
            )
        self.down_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.down_proj",
            weights=weights,
            bias=False,
        )
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.intermediate_size = (
                (config.intermediate_size + weights.process_group.size() - 1) // weights.process_group.size()
        )


class FlashDeepseekAttention(torch.nn.Module):
    class ForwardInputArgs:
        def __init__(self,
                     hidden_states: torch.tensor,
                     cos: torch.tensor,
                     sin: torch.tensor,
                     cu_seqlen_prefill: torch.tensor,
                     kv_cache: Tuple[torch.tensor, torch.tensor],
                     block_tables: torch.tensor,
                     slots: torch.tensor,
                     input_lengths: torch.tensor,
                     max_s: torch.tensor):
            self.hidden_states = hidden_states
            self.cos = cos
            self.sin = sin
            self.cu_seqlen_prefill = cu_seqlen_prefill
            self.kv_cache = kv_cache
            self.block_tables = block_tables
            self.slots = slots
            self.input_lengths = input_lengths
            self.max_s = max_s

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
        if (config.num_attention_heads != config.num_key_value_heads and
            self.num_heads % weights.process_group.size() != 0):
            msg = f"`num_heads` must be divisible by `num_shards` (got `num_heads`: {self.num_heads} " \
                  f"and `num_shards`: {weights.process_group.size()}"
            logger.error(msg, ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
            raise ValueError(msg)
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
        linear_names = [f'{prefix}.q_proj', f'{prefix}.k_proj', f'{prefix}.v_proj']
        pack_name = f'{prefix}.query_key_value'
        layer_prefix = '.'.join(prefix.split('.')[:-1])
        norm_name = f'{layer_prefix}.input_layernorm'
        self.pack_type = calc_linear_pack_type(weights, linear_names, norm_name, pack_name)

        if self.pack_type in [PackType.ALL_FP, PackType.ALL_W8A8, PackType.ALL_W8A8_ANTI, PackType.ALL_W8A16]:
            self.query_key_value = load_column_multi(
                config,
                prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
                weights=weights,
                head_size=self.head_size
            )
        elif self.pack_type == PackType.ALL_W8A8SC:
            pass
        else:
            pass
        self.o_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.o_proj",
            weights=weights,
            bias=False,
            gqa_size=self.head_size,
        )
        self.num_groups = self.num_heads // self.num_key_value_heads
        self.kv_head_mapping = torch.arange(
            0, self.num_key_value_heads, dtype=torch.int32, device=weights.device
        ).repeat_interleave(self.num_groups)

        self.prefix = prefix


class FlashDeepseekLayer(nn.Module):
    class ForwardInputArgs:
        def __init__(self,
                     hidden_states: torch.tensor,
                     residual: torch.tensor,
                     cos: torch.tensor,
                     sin: torch.tensor,
                     cu_seqlen_prefill: torch.tensor,
                     kv_cache: Tuple[torch.tensor, torch.tensor],
                     block_tables: List[torch.tensor],
                     slots: torch.tensor,
                     input_lengths: torch.tensor,
                     max_s: torch.tensor):
            self.hidden_states = hidden_states
            self.residual = residual
            self.cos = cos
            self.sin = sin
            self.cu_seqlen_prefill = cu_seqlen_prefill
            self.kv_cache = kv_cache
            self.block_tables = block_tables
            self.slots = slots
            self.input_lengths = input_lengths
            self.max_s = max_s

    def __init__(self, layer_id, config, weights):
        super().__init__()
        prefix = f"model.layers.{layer_id}"
        weights.switch_process_group(ProcessGroupType.ATTN)
        self.self_attn = FlashDeepseekAttention(
            prefix=f"{prefix}.self_attn", config=config, weights=weights
        )
        weights.switch_process_group(ProcessGroupType.MLP)
        if (config.n_routed_experts is not None and
            layer_id >= config.first_k_dense_replace and
            layer_id % config.moe_layer_freq == 0):
            self.mlp = DeepseekMoE(prefix=f"{prefix}.mlp", config=config, weights=weights, shared_mlp_cls=DeepseekMLP)
        else:
            self.mlp = DeepseekMLP(prefix=f"{prefix}.mlp", config=config, weights=weights)
        self.input_layernorm = DeepseekRMSNorm(
            prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = DeepseekRMSNorm(
            prefix=f"{prefix}.post_attention_layernorm",
            weights=weights,
            eps=config.rms_norm_eps,
        )


class FlashDeepseekModel(torch.nn.Module):
    class ForwardInputArgs:
        def __init__(self,
                    input_ids: torch.Tensor,
                    position_ids: torch.Tensor,
                    cu_seqlen_prefill: Optional[torch.Tensor],
                    kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
                    block_tables: torch.Tensor,
                    slots: torch.Tensor,
                    input_lengths: torch.Tensor,
                    max_s: int,
                    lm_head_indices: Optional[torch.Tensor] = None):
            self.input_ids = input_ids
            self.position_ids = position_ids
            self.cu_seqlen_prefill = cu_seqlen_prefill
            self.kv_cache = kv_cache
            self.block_tables = block_tables
            self.slots = slots
            self.input_lengths = input_lengths
            self.max_s = max_s
            self.lm_head_indices = lm_head_indices

    def __init__(self, config, weights):
        super().__init__()

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.embed_tokens = TensorEmbedding(
            prefix="model.embed_tokens", weights=weights
        )
        self.layers = nn.ModuleList(
            [
                FlashDeepseekLayer(
                    layer_id,
                    config,
                    weights,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = DeepseekRMSNorm(
            prefix="model.norm", weights=weights, eps=config.rms_norm_eps
        )

        self.gradient_checkpointing = False

        self.head_size = self.layers[0].self_attn.head_size
        self.num_heads = self.layers[0].self_attn.num_heads
        self.num_key_value_heads = self.layers[0].self_attn.num_key_value_heads
    

class DeepseekEp(nn.Module):
    """
    for experts parallel.
    """

    def __init__(self, prefix, config, weights):
        super().__init__()
        expert_gate_proj = weights.get_tensor(f"{prefix}.gate_proj.weight")
        self.expert_gate_proj = nn.Parameter(expert_gate_proj)
        expert_up_proj = weights.get_tensor(f"{prefix}.up_proj.weight")
        self.expert_up_proj = nn.Parameter(expert_up_proj)
        expert_down_proj = weights.get_tensor(f"{prefix}.down_proj.weight")
        self.expert_down_proj = nn.Parameter(expert_down_proj)


class DeepseekMoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, prefix, config, weights, shared_mlp_cls,
            gate_key="gate", shared_expert_key="shared_experts"):
        super().__init__()
        self.config = config
        self.hidden_dim = self.config.hidden_size
        self.num_experts_per_tok = config.num_experts_per_tok
        self.num_experts = config.n_routed_experts

        self.ep = weights.mapping.has_moe_ep()
        if self.ep:
            self.rank = weights.mapping.moe_ep.rank
            self.world_size = weights.mapping.moe_ep.group_size
        else:
            if weights.mapping.has_moe_tp():
                self.rank = weights.mapping.moe_tp.rank
                self.world_size = weights.mapping.moe_tp.group_size
            else:
                self.rank = weights.mapping.mlp_tp.rank
                self.world_size = weights.mapping.mlp_tp.group_size

        self.expert_lists = []
        if self.ep:
            self.expert_lists = assign(config.n_routed_experts, self.world_size)
        else:
            self.expert_lists = [[i for i in range(config.n_routed_experts)] for j in range(self.world_size)]

        self.device_expert = [i for i in range(self.config.n_routed_experts)] if not self.ep else \
            assign(self.config.n_routed_experts, weights.mapping.world_size)[weights.mapping.moe_ep.rank]
        temp_list = [j for j in range(config.n_routed_experts)]
        temp_list = temp_list[self.device_expert[0]:] + temp_list[:self.device_expert[0]]

        expert_prefix = f"{prefix}.experts"
        if hasattr(config, TOPK_METHOD) and config.topk_method == TOPK_METHOD_NOAUX_TC:
            self.gate = FastLinear.load(
                prefix=f"{prefix}.{gate_key}", weights=weights, bias=True, bias_name="e_score_correction_bias")
        else:
            self.gate = FastLinear.load(prefix=f"{prefix}.{gate_key}", weights=weights, bias=False)
        
        if self.ep:
            if hasattr(config, "ep_level") and config.ep_level == 3:
                if hasattr(config, TOPK_METHOD) and config.topk_method == TOPK_METHOD_NOAUX_TC:
                    self.shuffled_gate = FastLinear.load(
                        prefix=f"{prefix}.{gate_key}", weights=weights, bias=True, bias_name="e_score_correction_bias")
                    self.shuffled_gate.bias.data = self.shuffled_gate.bias.data[temp_list]
                else:
                    self.shuffled_gate = FastLinear.load(prefix=f"{prefix}.{gate_key}", weights=weights, bias=False)
                
                self.shuffled_gate.weight.data = self.shuffled_gate.weight.data[temp_list]

            elif not hasattr(config, "ep_level") or config.ep_level == 1:
                if hasattr(config, TOPK_METHOD) and config.topk_method == TOPK_METHOD_NOAUX_TC:
                    self.gate.bias.data = self.gate.bias.data[temp_list]
                self.gate.weight.data = self.gate.weight.data[temp_list]

        self.init_experts(weights, prefix, expert_prefix, shared_expert_key, shared_mlp_cls)
        
    def init_experts(self, weights, prefix, expert_prefix, shared_expert_key, shared_mlp_cls):
        def get_moe_expert_projs(prefix_list):
            if len(prefix_list[0]) == 2:
                linear_index = "gate_up_proj"
            else:
                linear_index = prefix_list[0][0].split(".")[-1]
            weight_list = [[] for _ in range(5)]
            for prefixes in prefix_list:
                weight = weights.get_multi_weights_col(
                    prefixes, quantize=config.quantize, dim=0
                )
                if isinstance(weight, tuple):
                    for i, element in enumerate(weight):
                        weight_list[i].append(element)
                else:
                    weight_list[0].append(weight)
            if isinstance(weight, tuple):
                weight_stacked = []
                for i in range(len(weight)):
                    weight_stacked.append(torch.stack(weight_list[i], dim=0))
            else:
                weight_stacked = torch.stack(weight_list[0], dim=0)
            stack_proj = TensorParallelColumnLinear.load_moe(
                    config,
                    prefixes=[f"{expert_prefix}.{linear_index}"],
                    weight=weight_stacked,
                    bias=None
                )
            return stack_proj

        linear_names = [f'{expert_prefix}.0.up_proj', f'{expert_prefix}.0.gate_proj']
        pack_name = f'{expert_prefix}.0.gate_up_proj'
        layer_prefix = '.'.join(prefix.split('.')[:-1])
        norm_name = f'{layer_prefix}.post_attention_layernorm'
        self.pack_type = calc_linear_pack_type(weights, linear_names, norm_name, pack_name)

        config = self.config
        if self.ep:
            weights.switch_process_group(ProcessGroupType.MOE_EP)
        
        pack_prefixes = None
        if self.pack_type in [
            PackType.ALL_FP, PackType.ALL_W8A8, PackType.ALL_W8A8_ANTI, PackType.ALL_W4A16,
            PackType.ALL_W4A16_ANTI, PackType.ALL_W8A16, PackType.ALL_W8A16_ANTI,
            PackType.MIX_W8A8_DYNAMIC, PackType.MIX_W8A8_DYNAMIC_ANTI,
            PackType.ALL_W8A8_DYNAMIC, PackType.ALL_W8A8_DYNAMIC_ANTI
        ]:
            pack_prefixes = [[f"{expert_prefix}.{i}.gate_proj", f"{expert_prefix}.{i}.up_proj"] \
                            for i in self.expert_lists[self.rank]]
            
        elif self.pack_type in [PackType.ALL_W8A8SC, PackType.ALL_W8A8SC_ANTI]:
            pack_prefixes = [[f"{expert_prefix}.{i}.gate_up_proj"] \
                            for i in self.expert_lists[self.rank]]
            
        else:
            gate_prefixes = [[f"{expert_prefix}.{i}.gate_up_proj"] \
                            for i in self.expert_lists[self.rank]]
            up_prefixes = [[f"{expert_prefix}.{i}.gate_up_proj"] \
                            for i in self.expert_lists[self.rank]]
        
        if pack_prefixes is not None:
            self.gate_up_proj = get_moe_expert_projs(pack_prefixes)
        else:
            self.gate_proj = get_moe_expert_projs(gate_prefixes)
            self.up_proj = get_moe_expert_projs(up_prefixes)

        weight_list = [[] for _ in range(5)] # 一个weight tuple不会包含5个以上元素
        for i in self.expert_lists[self.rank]:
            weight = weights.get_multi_weights_row(
                f"{expert_prefix}.{i}.down_proj", quantize=config.quantize, gqa_size=1, dim=1
            )
            if isinstance(weight, tuple):
                for j, element in enumerate(weight):
                    weight_list[j].append(element)
            else:
                weight_list[0].append(weight)
        if isinstance(weight, tuple):
            weight_stacked = []
            for i in range(len(weight)):
                weight_stacked.append(torch.stack(weight_list[i], dim=0))
        else:
            weight_stacked = torch.stack(weight_list[0], dim=0)
        self.down_proj = TensorParallelRowLinear.load_moe(
                config,
                prefix=f"{expert_prefix}.down_proj",
                process_group=weights.process_group,
                weight=weight_stacked,
                bias=None
            )

        self.intermediate_size = ((config.intermediate_size + self.world_size - 1) // self.world_size)

        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            shared_expert_prefix = f"{prefix}.shared_experts"
            if self.ep:
                weights.switch_process_group(ProcessGroupType.MLP)
                if hasattr(config, "ep_level") and config.ep_level == 3:
                    self.shared_experts_tp = shared_mlp_cls(
                        prefix=shared_expert_prefix,
                        config=config,
                        weights=weights,
                        intermediate_size=intermediate_size
                    )
                if not (hasattr(config, "ep_level")) or config.ep_level != 1:
                    weights.switch_process_group(ProcessGroupType.MOE_EP)
            self.shared_experts = shared_mlp_cls(
                prefix=shared_expert_prefix,
                config=config,
                weights=weights,
                intermediate_size=intermediate_size
            )