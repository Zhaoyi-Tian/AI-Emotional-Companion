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
import json
from enum import Enum
from collections import OrderedDict

import torch
from torch import nn

import _libatb_torch as atb
from atb_llm.common_op_builders.data_type import CommonOpBuilderType, OperationBackend
from atb_llm.common_op_builders.common_op_builder_manager import CommonOpBuilderManager
from atb_llm.common_op_builders.linear_parallel.base_linear_parallel_common_op_builder import CommunicationBackend
from atb_llm.common_op_builders.attention.base_attention_common_op_builder import AttnType
from atb_llm.models.base.model_utils import MlpLinearInfo
from atb_llm.models.base.modeling_atb import BaseRMSNorm, BaseAttention
from atb_llm.models.base.flash_causal_lm_atb import AtbGraph
from atb_llm.utils import OpBackend
from atb_llm.utils.log import logger
from atb_llm.utils.moe_utils import assign
from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.utils.quantize.pack_type import AclDataType, PackType, calc_linear_pack_type
from atb_llm.utils.layers import TensorParallelRowLinear, TensorParallelColumnLinear, load_column_multi
from atb_llm.utils.layers.linear import FastLinear, TensorReplicatedLinear
from atb_llm.utils.layers import KvCache

MIXTRAL_EMBEDDING_PARALLEL_THRESHOLD = 32000


class RouteMethod(str, Enum):
    softmax_topk = "softMaxTopK"
    device_limited = "deviceLimited"
    integrated_softmax_topk = "integratedSoftmaxTopK"


class ProcessLogits(str, Enum):
    normalization = "normalization"
    norm = "norm"
    scaling = "scaling"
    none = "none"


class IntermediateTensorStr(str, Enum):
    dummy_zero = "intermediate_dummy_zero",
    dummy_one = "intermediate_dummy_one",
    group_list = "intermediate_group_list",
    group_list_int64 = "intermediate_group_list_int64",
    idx = "intermediate_idx",
    mlp_out = "intermediate_mlp_out",
    mlp_out_weighted = "intermediate_mlp_out_weighted",
    matmul_up_out = "intermediate_matmul_up_out",
    matmul_gate_out = "intermediate_matmul_gate_out",
    matmul_gate_up_out = "intermediate_matmul_gate_up_out",
    rev_idx = "intermediate_rev_idx",
    router_weights_topk_reduced = "intermediate_router_weights_topk_reduced",
    router_weights_topk = "intermediate_router_weights_topk",
    router_weights_topk_sumed = "intermediate_router_weights_topk_sumed",
    router_logits = "intermediate_router_logits",
    router_weights = "intermediate_router_weights",
    router_logits_std = "intermediate_router_logits_std",
    rev_sorted_hidden_states = "intermediate_rev_sorted_hidden_states",
    swish_out = "intermediate_swish_out",
    sorted_weight = "intermediate_sorted_weight",
    selected_experts = "intermediate_selected_experts",
    swish_out_internal = "intermediate_swish_out_internal",
    sorted_hidden_states = "intermediate_sorted_hidden_states",
    weight_idx = "intermediate_weight_idx",
    hidden_states = "hidden_states"


class FusionAttention(BaseAttention):
    def __init__(
        self,
        config,
        weights,
        prefix: str,
        norm_prefix: str,
        is_fa: bool = False,
        backend=CommunicationBackend.LCCL,
        speculate_enable=False,
    ):
        super().__init__(config, weights, prefix, norm_prefix, is_fa, backend)

        # 并行解码
        self.speculate_enable = speculate_enable
        # kv cache量化
        self.kv_quant = config.quantization_config.kv_quant_type
        self.kv_cache_quant = None

        if self.kv_quant is not None:
            self.kv_cache_quant = KvCache.load(prefix_k=f"{prefix}.k_proj",
                prefix_v=f"{prefix}.v_proj", weights=weights, backend=OpBackend.ATB)

    def get_weights(self, prefix):
        weights_dict = super().get_weights(prefix)
        if self.kv_quant is not None:
            weights_dict.update(self.kv_cache_quant.get_weights(f"{prefix}"))
        return weights_dict

    def build_attention_graph(self, graph, is_prefill):
        attention_param = {
            "op_name": "attention",
            "category": CommonOpBuilderType.ATTENTION,
            "is_prefill": is_prefill,
            "attn_type": AttnType.FLASH_ATTENTION if self.is_fa else AttnType.PAGED_ATTENTION,
            "head_size": self.head_size,
            "atb_reshape_and_cache_param": {},
            "operation_backend": OperationBackend.ATB,
            "enable_kv_quant": self.kv_quant is not None,
            "kv_quant_module": self.kv_cache_quant,
        }

        atb_attention_param = self._get_atb_attention_param(is_prefill)
        if not self.is_fa and not is_prefill and self.speculate_enable:
            atb_attention_param.update({
                'maskType': 'MASK_TYPE_SPEC',
                'calcType': 'CALC_TYPE_SPEC',
            })
        attention_param.update({"atb_attention_param": atb_attention_param})

        attention_tensor_map = self._get_attention_tensor_map()
        if not self.is_fa and self.speculate_enable:
            attention_tensor_map.update({"q_len": "q_len"})

        pa_attention_builder = CommonOpBuilderManager.get_builder(attention_param)
        graph = pa_attention_builder.build(graph, attention_tensor_map)

    def build_graph(self, graph, is_prefill):
        atten_res_add = atb._BaseOperation(op_type="Elewise", op_param=json.dumps({'elewiseType': 'ELEWISE_ADD'}),
                                           op_name='atten_res_add')
        setattr(graph, 'atten_res_add', atten_res_add)

        self.build_qkv_graph(graph)
        self.build_rope_graph(graph)
        self.build_attention_graph(graph, is_prefill)
        self.build_dense_graph(graph, is_prefill)

        graph.add_operation(graph.atten_res_add, ['hidden_states', 'dense_out'], ['hidden_states'])


class MoeMlp(torch.nn.Module):
    def __init__(
        self,
        config,
        weights,
        prefix: str,
        norm_prefix: str,
        is_fa: bool = False,
        backend=CommunicationBackend.LCCL,
    ):
        super().__init__()

        self.op_name = "moe_mlp"
        self.config = config
        self.is_fa = is_fa
        self.prefix = prefix
        self.norm_prefix = norm_prefix
        self.is_bf16 = weights.dtype == torch.bfloat16
        self.support_swiglu = True
        self.enable_fused_routing = True
        self.process_logits = "normalization"
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.tp = True  # defaulting the model to tensor parallel
        self.expert_parallel_degree = 1 if self.tp else self.tp_world_size
        if (self.expert_parallel_degree == 0):
            msg = "expert_parallel degree should not be 0!"
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(msg)
        self.expert_lists = []
        if self.tp:
            self.expert_lists = [[i for i in range(config.n_routed_experts)] for j in range(self.tp_world_size)]
        else:
            self.expert_lists = assign(config.n_routed_experts, self.tp_world_size)
        linear_names = [f'{prefix}.{0}.w1', f'{prefix}.{0}.w3']
        pack_name = f'{prefix}.{0}.gate_up_proj'
        layer_prefix = '.'.join(prefix.split('.')[:-1])
        norm_name = f'{layer_prefix}.post_attention_layernorm'
        self.pack_type = calc_linear_pack_type(weights, linear_names, norm_name, pack_name)
        if self.tp:
            if self.pack_type in [
            PackType.ALL_FP, PackType.ALL_W8A8, PackType.ALL_W8A8_ANTI, PackType.ALL_W4A16,
            PackType.ALL_W4A16_ANTI, PackType.ALL_W8A16, PackType.ALL_W8A16_ANTI,
            PackType.ALL_W8A8_DYNAMIC, PackType.ALL_W8A8_DYNAMIC_ANTI
            ]:
                gate_up_proj = nn.ModuleList()
                for i in range(self.num_experts):
                    gate_up_proj.append(load_column_multi(
                        config,
                        prefixes=[f"{prefix}.{i}.w1", f"{prefix}.{i}.w3"],
                        weights=weights,
                        head_size=1,
                    ))
            elif self.pack_type in [PackType.ALL_W8A8SC, PackType.ALL_W8A8SC_ANTI]:
                gate_up_proj = nn.ModuleList()
                for i in range(self.num_experts):
                    gate_up_proj.append(TensorParallelColumnLinear.load(
                    config,
                    prefix=f"{prefix}.{i}.gate_up_proj",
                    weights=weights,
                    bias=False,
                    ))
            else:
                self.gate_proj = nn.ModuleList()
                self.up_proj = nn.ModuleList()
                for i in range(self.num_experts):
                    self.gate_proj.append(TensorParallelColumnLinear.load(
                    config,
                    prefix=f"{prefix}.{i}.w1",
                    weights=weights,
                    bias=False,
                    ))
                    self.up_proj.append(TensorParallelColumnLinear.load(
                        config,
                        prefix=f"{prefix}.{i}.w3",
                        weights=weights,
                        bias=False,
                    ))
                
            down_proj = nn.ModuleList()
            for i in range(self.num_experts):
                down_proj.append(TensorParallelRowLinear.load(
                config,
                prefix=f"{prefix}.{i}.w2",
                weights=weights,
                bias=False,
                ))
            self.intermediate_size = (
                    (config.intermediate_size + weights.process_group.size() - 1) // weights.process_group.size()
            )
   
        else:
            if self.pack_type in [
            PackType.ALL_FP, PackType.ALL_W8A8, PackType.ALL_W8A8_ANTI, PackType.ALL_W4A16,
            PackType.ALL_W4A16_ANTI, PackType.ALL_W8A16, PackType.ALL_W8A16_ANTI,
            PackType.ALL_W8A8_DYNAMIC, PackType.ALL_W8A8_DYNAMIC_ANTI
            ]:
                gate_up_proj = nn.ModuleList()
                for i in self.expert_lists[self.tp_rank]:
                    gate_up_proj.append(TensorReplicatedLinear.load(
                        config,
                        prefixes=[f"{prefix}.{i}.w1", f"{prefix}.{i}.w3"],
                        weights=weights,
                        head_size=1,
                    ))
            down_proj = nn.ModuleList()
            for i in self.expert_lists[self.tp_rank]:
                down_proj.append(TensorReplicatedLinear.load(
                config,
                prefix=f"{prefix}.{i}.w2",
                weights=weights,
                bias=False,
                ))

        if gate_up_proj[0].linear.weight.data.shape[0] == self.hidden_dim:
            self.gate_up_weight = torch.stack([proj.linear.weight.data \
                for proj in gate_up_proj], dim=0).to(weights.device)
        else:
            self.gate_up_weight = torch.stack([proj.linear.weight.data.transpose(0, 1) \
                for proj in gate_up_proj], dim=0).to(weights.device)
        if down_proj[0].linear.weight.data.shape[0] == self.hidden_dim:
            self.down_weight = torch.stack([proj.linear.weight.data.transpose(0, 1) \
                for proj in down_proj], dim=0).to(weights.device)
        else:
            self.down_weight = torch.stack([proj.linear.weight.data \
                for proj in down_proj], dim=0).to(weights.device)

        self.linear_info = MlpLinearInfo()

    def get_weights(self, prefix):
        weights_dict = OrderedDict()
        weights_dict[f"{self.prefix}.gate_up_weight"] = self.gate_up_weight
        weights_dict[f"{self.prefix}.down_weight"] = self.down_weight
        return weights_dict

    def reshape_sorted_weight(self, org_shape):
        return [org_shape[0], 1]

    def reshape_expert_weight(self, org_shape):
        return [org_shape[0] * org_shape[1]]

    def reshape_sorted_hidden_state(self, org_shape):
        return [org_shape[0] // self.config.num_experts_per_tok,
            self.config.num_experts_per_tok, org_shape[1]]

    def build_init_routing(self, graph):
        moe_init_routing_op = atb._BaseOperation(
            op_type="MoeInitRouting",
            op_param=json.dumps({"topkNum": self.config.num_experts_per_tok,
                                "expertNum": self.config.n_routed_experts}),
            op_name=f"{self.op_name}_moe_init_routing"
        )
        graph.operations.append(moe_init_routing_op)
        graph.add_operation(
            moe_init_routing_op,
            [f"{self.norm_prefix}_out", IntermediateTensorStr.selected_experts],
            [IntermediateTensorStr.sorted_hidden_states, IntermediateTensorStr.idx, IntermediateTensorStr.group_list]
        )

    def build_compute_expert(self, graph):
        moe_compute_expert_op = atb._BaseOperation(
            op_type="MoeComputeExpertTokens",
            op_param=json.dumps({"expertNum": self.config.n_routed_experts}),
            op_name=f"{self.op_name}_moe_compute_expert"
        )
        graph.operations.append(moe_compute_expert_op)
        graph.add_operation(
            moe_compute_expert_op,
            [IntermediateTensorStr.weight_idx],
            [IntermediateTensorStr.group_list]
        )

    def build_cast(self, graph):
        elewise_cast_op = atb._BaseOperation(
            op_type="Elewise",
             op_param=json.dumps({'elewiseType': 'ELEWISE_CAST', 'outTensorType': 'ACL_INT64'}),
            op_name=f"{self.op_name}_elewise_cast"
        )
        graph.operations.append(elewise_cast_op)
        graph.add_operation(
            elewise_cast_op,
            [IntermediateTensorStr.group_list],
            [IntermediateTensorStr.group_list_int64]
        )

    def build_gate_up_gmm(self, graph): 
        gate_up_gmm_params = {
            "op_name": "integrated_gmm_gate_up",
            "category": CommonOpBuilderType.INTEGRATED_GMM,
            "has_bias": False,
            "is_up": True,
            "out_data_type": AclDataType.ACL_BF16 if self.is_bf16 \
                else AclDataType.ACL_FLOAT16,
            "transpose_b": False,
            "pack_quant_type": PackType.ALL_FP,
        }
        gate_up_gmm_builder = CommonOpBuilderManager.get_builder(gate_up_gmm_params)
        gate_up_gmm_tensor_map = {
            "input": IntermediateTensorStr.sorted_hidden_states,
            "weight": f"{self.prefix}.gate_up_weight",
            "group_list": IntermediateTensorStr.group_list_int64 if self.enable_fused_routing \
                else IntermediateTensorStr.group_list,
            "out": IntermediateTensorStr.matmul_gate_up_out,
        }
        graph = gate_up_gmm_builder.build(graph, gate_up_gmm_tensor_map)

    def build_down_gmm(self, graph):
        down_gmm_params = {
            "op_name": "integrated_gmm_down",
            "category": CommonOpBuilderType.INTEGRATED_GMM,
            "has_bias": False,
            "is_up": False,
            "out_data_type": AclDataType.ACL_BF16 if self.is_bf16 
                else AclDataType.ACL_FLOAT16,
            "transpose_b": False,
            "pack_quant_type": PackType.ALL_FP,
        }
        down_gmm_builder = CommonOpBuilderManager.get_builder(down_gmm_params)
        down_gmm_tensor_map = {
            "input": IntermediateTensorStr.swish_out,
            "weight": f"{self.prefix}.down_weight",
            "group_list": IntermediateTensorStr.group_list_int64 if self.enable_fused_routing \
                else IntermediateTensorStr.group_list,
            "out": IntermediateTensorStr.mlp_out,
        }
        graph = down_gmm_builder.build(graph, down_gmm_tensor_map)

    def build_activation_block(self, graph):
        if self.support_swiglu:
            activation_op = atb._BaseOperation(
                op_type="Activation",
                op_param=json.dumps({'activationType': 'ACTIVATION_SWIGLU_FORWARD'}),
                op_name=f"{self.op_name}_activation"
            )
            graph.operations.append(activation_op)
            graph.add_operation(
                activation_op,
                [IntermediateTensorStr.matmul_gate_up_out],
                [IntermediateTensorStr.swish_out]
            )
        else:
            split_op = atb._BaseOperation(
                op_type="Split",
                op_param=json.dumps({"splitDim": 1, "splitNum": 2}),
                op_name=f"{self.op_name}_split"
            )
            graph.operations.append(split_op)
            graph.add_operation(
                split_op,
                [IntermediateTensorStr.matmul_gate_up_out],
                [IntermediateTensorStr.matmul_gate_out, IntermediateTensorStr.matmul_up_out]
            )
            activation_op = atb._BaseOperation(
                op_type="Activation",
                op_param=json.dumps({'activationType': 'ACTIVATION_SWISH'}),
                op_name=f"{self.op_name}_activation"
            )
            graph.operations.append(activation_op)
            graph.add_operation(
                activation_op,
                [IntermediateTensorStr.matmul_gate_out],
                [IntermediateTensorStr.swish_out_internal]
            )
            elewise_mul_op = atb._BaseOperation(
                op_type="Elewise",
                op_param=json.dumps({'elewiseType': 'ELEWISE_MUL'}),
                op_name=f"{self.op_name}_elewise_mul"
            )
            graph.operations.append(elewise_mul_op)
            graph.add_operation(
                elewise_mul_op,
                [IntermediateTensorStr.swish_out_internal],
                [IntermediateTensorStr.matmul_up_out]
            )

    def build_moe_token_unpermute(self, graph):
        moe_token_unpermute_op = atb._BaseOperation(
            op_type="MoeTokenUnpermute",
            op_param=json.dumps({}),
            op_name=f"{self.op_name}_moe_token_unpermute"
        )
        graph.operations.append(moe_token_unpermute_op)
        graph.add_operation(
            moe_token_unpermute_op,
            [
                IntermediateTensorStr.mlp_out,
                IntermediateTensorStr.idx,
                IntermediateTensorStr.router_weights_topk_reduced \
                    if self.process_logits != "none" \
                    else IntermediateTensorStr.router_weights_topk
            ],
            ["mlp_out"]
        )

    def build_gating(self, graph):
        gating_op = atb._BaseOperation(
            op_type="Gating",
            op_param=json.dumps({
                "topkExpertNum": self.config.num_experts_per_tok,
                "cumSumNum": self.config.n_routed_experts,
                "cumSumInt64": True
            }),
            op_name=f"{self.op_name}_gating"
        )
        graph.add_reshape(IntermediateTensorStr.selected_experts,
            IntermediateTensorStr.selected_experts, self.reshape_expert_weight)
        graph.operations.append(gating_op)
        graph.add_operation(
            gating_op,
            [IntermediateTensorStr.selected_experts, "expert_array"],
            [IntermediateTensorStr.idx, IntermediateTensorStr.group_list, IntermediateTensorStr.weight_idx]
        )

    def build_gather_hidden_states(self, graph):
        gather_op = atb._BaseOperation(
            op_type="Gather",
            op_param=json.dumps({}),
            op_name=f"{self.op_name}_gather_hidden_states"
        )
        graph.operations.append(gather_op)
        graph.add_operation(
            gather_op,
            [IntermediateTensorStr.hidden_states, IntermediateTensorStr.idx],
            [IntermediateTensorStr.sorted_hidden_states]
        )

    def build_gather_weight(self, graph):
        gather_op = atb._BaseOperation(
            op_type="Gather",
            op_param=json.dumps({}),
            op_name=f"{self.op_name}_gather_weight"
        )
        if self.process_logits != "none":
            graph.add_reshape(IntermediateTensorStr.router_weights_topk_reduced,
                IntermediateTensorStr.router_weights_topk_reduced, self.reshape_expert_weight)
        else:
            graph.add_reshape(IntermediateTensorStr.router_weights_topk,
                IntermediateTensorStr.router_weights_topk, self.reshape_expert_weight)
        graph.operations.append(gather_op)
        graph.add_operation(
            gather_op,
            [
                IntermediateTensorStr.router_weights_topk_reduced if self.process_logits != "none"
                    else IntermediateTensorStr.router_weights_topk, 
                IntermediateTensorStr.weight_idx
            ],
            [IntermediateTensorStr.sorted_weight]
        )

    def build_gather_rev_hidden_states(self, graph):
        gather_op = atb._BaseOperation(
            op_type="Gather",
            op_param=json.dumps({}),
            op_name=f"{self.op_name}_gather_rev_hidden_states"
        )
        graph.operations.append(gather_op)
        graph.add_operation(
            gather_op,
            [IntermediateTensorStr.mlp_out_weighted, IntermediateTensorStr.rev_idx],
            [IntermediateTensorStr.rev_sorted_hidden_states]
        )

    def build_elewise_mul(self, graph):
        elewise_mul_op = atb._BaseOperation(
            op_type="Elewise",
            op_param=json.dumps({'elewiseType': 'ELEWISE_MUL'}),
            op_name=f"{self.op_name}_elewise_mul1"
        )
        graph.add_reshape(IntermediateTensorStr.sorted_weight, IntermediateTensorStr.sorted_weight,
            self.reshape_sorted_weight)
        graph.operations.append(elewise_mul_op)
        graph.add_operation(
            elewise_mul_op,
            [IntermediateTensorStr.mlp_out, IntermediateTensorStr.sorted_weight],
            [IntermediateTensorStr.mlp_out_weighted]
        )

    def build_arg_sort(self, graph):
        gating_op = atb._BaseOperation(
            op_type="Gating",
            op_param=json.dumps({"topkExpertNum": 1, "cumSumNum": 0}),
            op_name=f"{self.op_name}_arg_sort"
        )
        graph.operations.append(gating_op)
        graph.add_operation(
            gating_op,
            [IntermediateTensorStr.weight_idx, "expert_array"],
            [IntermediateTensorStr.dummy_zero, IntermediateTensorStr.dummy_one, IntermediateTensorStr.rev_idx]
        )

    def build_reduction(self, graph):
        reduction_op = atb._BaseOperation(
            op_type="Reduce",
            op_param=json.dumps({"reduceType": "REDUCE_SUM", "axis": [1]}),
            op_name=f"{self.op_name}_reduce"
        )
        graph.add_reshape(IntermediateTensorStr.rev_sorted_hidden_states,
            IntermediateTensorStr.rev_sorted_hidden_states, self.reshape_sorted_hidden_state)
        graph.operations.append(reduction_op)
        graph.add_operation(
            reduction_op,
            [IntermediateTensorStr.rev_sorted_hidden_states],
            ["mlp_out"]
        )

    def build_graph(self, graph):
        if self.enable_fused_routing:
            self.build_init_routing(graph)
            self.build_cast(graph)
            self.build_gate_up_gmm(graph)
            self.build_activation_block(graph)
            self.build_down_gmm(graph)
            self.build_moe_token_unpermute(graph)
        else:
            self.build_gating(graph)
            self.build_gather_hidden_states(graph)
            self.build_gate_up_gmm(graph)
            self.build_activation_block(graph)
            self.build_down_gmm(graph)
            self.build_gather_weight(graph)
            self.build_elewise_mul(graph)
            self.build_arg_sort(graph)
            self.build_gather_rev_hidden_states(graph)
            self.build_reduction(graph)


class SparseMoe(torch.nn.Module):
    def __init__(
        self,
        config,
        weights,
        prefix: str,
        norm_prefix: str,
        is_fa: bool = False,
        backend=CommunicationBackend.LCCL,
    ):
        super().__init__()

        self.op_name = "sparse_moe"
        self.config = config
        self.is_fa = is_fa
        self.prefix = prefix
        self.norm_prefix = norm_prefix
        self.backend = backend

        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok

        self.use_std_norm = False
        self.routing_method = RouteMethod.integrated_softmax_topk
        self.axes = [1]
        self.process_logits = "normalization"
        self.dtype = weights.dtype
        self.is_bf16 = self.dtype == torch.bfloat16

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.tp = True
        self.expert_parallel_degree = 1 if self.tp else self.tp_world_size
        self.gate = FastLinear.load(
                prefix=f"{prefix}.gate",
                weights=weights,
                bias=False,
                )
        
        self.moe_mlp = MoeMlp(
            prefix=f"{prefix}.experts", norm_prefix=norm_prefix, config=config, weights=weights, backend=backend
        )
    
    def get_weights(self, prefix):
        weights_dict = OrderedDict()
        for name, module in self.named_children():
            weights_dict.update(module.get_weights(f"{prefix}.{name}"))
        self.weight_names = list(weights_dict.keys())
        return weights_dict

    def reshape_topk_sumed(self, org_shape):
        return [org_shape[0], 1]

    def build_gate_graph(self, graph):
        fusion_linear_params = {
            "op_name": "fusion_linear",
            "category": CommonOpBuilderType.FUSION_LINEAR,
            "is_bf16": self.is_bf16,
            "has_bias": False,
            "support_lora": False,
            "use_im_mask": False,
            "lora_enable_gmm": False,
            "quant_group_size": 0,
        }
        fusion_linear_builder = CommonOpBuilderManager.get_builder(fusion_linear_params)
        fusion_linear_tensor_map = {
            "input": f"{self.norm_prefix}_out",
            "weight": f"{self.prefix}.gate.weight",
            "output": IntermediateTensorStr.router_logits,
        }
        graph = fusion_linear_builder.build(graph, fusion_linear_tensor_map)

        if self.use_std_norm:
            std_op = atb._BaseOperation(
                op_type="Std",
                op_param=json.dumps({}),
                op_name=f"{self.op_name}_std"
            )
            graph.operations.append(std_op)
            graph.add_operation(
                std_op,
                [IntermediateTensorStr.router_logits],
                [IntermediateTensorStr.router_logits_std]
            )

            norm_op = atb._BaseOperation(
                op_type="Elewise",
                op_param=json.dumps({"elewiseType": "ELEWISE_REALDIV"}),
                op_name=f"{self.op_name}_norm"
            )
            graph.operations.append(norm_op)
            graph.add_operation(
                norm_op,
                [IntermediateTensorStr.router_logits, IntermediateTensorStr.router_logits_std],
                [IntermediateTensorStr.router_logits]
            )

    def build_routingblock_graph(self, graph):
        if self.routing_method == RouteMethod.integrated_softmax_topk:
            moe_softmax_topk_op = atb._BaseOperation(
                op_type="MoeTopkSoftmax",
                op_param=json.dumps({"topkNum": self.config.num_experts_per_tok}),
                op_name=f"{self.op_name}_moe_topk_softmax"
            )
            graph.operations.append(moe_softmax_topk_op)
            graph.add_operation(
                moe_softmax_topk_op,
                [IntermediateTensorStr.router_logits],
                [IntermediateTensorStr.router_weights_topk,
                IntermediateTensorStr.selected_experts, IntermediateTensorStr.router_weights]
            )
        else:
            moe_softmax_op = atb._BaseOperation(
                op_type="Softmax",
                op_param=json.dumps({"axes": self.axes}),
                op_name=f"{self.op_name}_moe_softmax"
            )
            graph.operations.append(moe_softmax_op)
            graph.add_operation(
                moe_softmax_op,
                [IntermediateTensorStr.router_logits],
                [IntermediateTensorStr.router_weights]
            )
            if self.routing_method == RouteMethod.device_limited:
                group_topk_op = atb._BaseOperation(
                    op_type="GroupTopk",
                    op_param=json.dumps({"groupNum": self.num_of_groups, "k": self.topk_groups[0]}),
                    op_name=f"{self.op_name}_moe_group_topk"
                )
                graph.operations.append(group_topk_op)
                graph.add_operation(
                    group_topk_op,
                    [IntermediateTensorStr.router_weights, "input_expert_group"],
                    [IntermediateTensorStr.router_weights]
                )
            moe_topk_op = atb._BaseOperation(
                op_type="Sort",
                op_param=json.dumps({"num": [self.config.num_experts_per_tok]}),
                op_name=f"{self.op_name}_moe_topK"
            )
            graph.operations.append(moe_topk_op)
            graph.add_operation(
                moe_topk_op,
                [IntermediateTensorStr.router_weights],
                [IntermediateTensorStr.router_weights_topk, IntermediateTensorStr.selected_experts]
            )

    def build_normalization_graph(self, graph):
        reduce_op = atb._BaseOperation(
            op_type="Reduce",
            op_param=json.dumps({"reduceType": "REDUCE_SUM", "axis": [1]}),
            op_name=f"{self.op_name}_reduce_sum"
        )
        graph.operations.append(reduce_op)
        graph.add_operation(
            reduce_op,
            [IntermediateTensorStr.router_weights_topk],
            [IntermediateTensorStr.router_weights_topk_sumed]
        )
        elewise_div_op = atb._BaseOperation(
            op_type="Elewise",
            op_param=json.dumps({"elewiseType": "ELEWISE_REALDIV"}),
            op_name=f"{self.op_name}_elewise_div"
        )
        graph.add_reshape(
            IntermediateTensorStr.router_weights_topk_sumed,
            IntermediateTensorStr.router_weights_topk_sumed,
            self.reshape_topk_sumed
        )
        graph.operations.append(elewise_div_op)
        graph.add_operation(
            elewise_div_op,
            [IntermediateTensorStr.router_weights_topk, IntermediateTensorStr.router_weights_topk_sumed],
            [IntermediateTensorStr.router_weights_topk_reduced]
        )
    
    def build_scaling_graph(self, graph):
        elewise_mul_op = atb._BaseOperation(
            op_type="Elewise",
            op_param=json.dumps({"elewiseType": "ELEWISE_MULS"}),
            op_name=f"{self.op_name}_elewise_mul"
        )
        graph.operations.append(elewise_mul_op)
        graph.add_operation(
            elewise_mul_op,
            [IntermediateTensorStr.router_weights_topk],
            [IntermediateTensorStr.router_weights_topk_reduced]
        )
    
    def build_norm_graph(self, graph):
        vector_norm_op = atb._BaseOperation(
            op_type="VectorNorm",
            op_param=json.dumps({}),
            op_name=f"{self.op_name}_vector_norm"
        )
        graph.operations.append(vector_norm_op)
        graph.add_operation(
            vector_norm_op,
            [IntermediateTensorStr.router_weights_topk],
            [IntermediateTensorStr.router_weights_topk_sumed]
        )
        elewise_div_op = atb._BaseOperation(
            op_type="Elewise",
            op_param=json.dumps({"elewiseType": "ELEWISE_REALDIV"}),
            op_name=f"{self.op_name}_elewise_div"
        )
        graph.add_reshape(
            IntermediateTensorStr.router_weights_topk_sumed,
            IntermediateTensorStr.router_weights_topk_sumed,
            self.reshape_topk_sumed
        )
        graph.operations.append(elewise_div_op)
        graph.add_operation(
            elewise_div_op,
            [IntermediateTensorStr.router_weights_topk, IntermediateTensorStr.router_weights_topk_sumed],
            [IntermediateTensorStr.router_weights_topk_reduced]
        )
    
    def build_all_reduce_graph(self, graph):
        all_reduce_op = atb._BaseOperation(
            op_type="AllReduce",
            op_param=json.dumps({
                "rank": self.tp_rank,
                "rankSize": self.tp_world_size,
                "backend": self.backend,
            }),
            op_name=f"{self.op_name}_AllReduce"
        )
        graph.operations.append(all_reduce_op)
        graph.add_operation(
            all_reduce_op,
            ['mlp_out'],
            ['moe_out']
        )

    def build_graph(self, graph, is_prefill):
        moe_res_add = atb._BaseOperation(op_type="Elewise", op_param=json.dumps({'elewiseType': 'ELEWISE_ADD'}),
                                         op_name='moe_res_add')
        setattr(graph, 'moe_res_add', moe_res_add)

        self.build_gate_graph(graph)
        self.build_routingblock_graph(graph)
        if self.process_logits == "normalization" :
            self.build_normalization_graph(graph)
        elif self.process_logits == "scaling" :
            self.build_scaling_graph(graph)
        elif self.process_logits == "norm" :
            self.build_norm_graph(graph)
        self.moe_mlp.build_graph(graph)
        self.build_all_reduce_graph(graph)

        graph.add_operation(graph.moe_res_add, ['hidden_states', 'moe_out'], ['layer_out'])


class MoeLayer(torch.nn.Module):
    def __init__(
        self,
        layer_id, 
        config, 
        weights, 
        model_prefix: str = "model", 
        is_fa: bool = False, 
        backend=CommunicationBackend.LCCL,
        speculate_enable: bool = False,
    ):
        super().__init__()

        # 配置信息
        prefix = f"{model_prefix}.layers.{layer_id}"
        self.layer_id = layer_id
        self.config = config
        tp_world_size = weights.process_group.size()
        self.is_reshape = config.vocab_size >= MIXTRAL_EMBEDDING_PARALLEL_THRESHOLD and tp_world_size > 1 and not is_fa
        self.weight_names = None
        self.layer_graph = None
        self.is_fa = is_fa
        self.speculate_enable = speculate_enable

        # 模型结构
        self.self_attn = FusionAttention(
            config=config, weights=weights, prefix=f"{prefix}.self_attn", norm_prefix=f"{prefix}.input_layernorm",
            is_fa=self.is_fa, backend=backend, speculate_enable=self.speculate_enable
        )

        self.block_sparse_moe = SparseMoe(
            prefix=f"{prefix}.block_sparse_moe", config=config, weights=weights,
            norm_prefix=f"{prefix}.post_attention_layernorm", backend=backend, is_fa=self.is_fa
        )

        self.input_layernorm = BaseRMSNorm(
            f"{prefix}.input_layernorm", config, weights, self.self_attn.linear_info
        )

        self.post_attention_layernorm = BaseRMSNorm(
            f"{prefix}.post_attention_layernorm", config, weights, self.block_sparse_moe.moe_mlp.linear_info
        )

    def get_weights(self, prefix):
        weights_dict = OrderedDict()
        for name, module in self.named_children():
            weights_dict.update(module.get_weights(f"{prefix}.{name}"))
        self.weight_names = list(weights_dict.keys())
        return weights_dict
    
    def get_in_tensor_names(self, is_prefill):
        default_input = ['hidden_states', 'seq_len', 'expert_array', 'expert_group', 'one_hot', 'zero_hot']
        if self.is_fa:
            default_input.extend(['token_offset', 'layer_id'])
        else:
            default_input.extend(['slots_mapping'])

        if self.config.pe_type == "ROPE":
            default_input.extend(['cos_embedding', 'sin_embedding'])
        if is_prefill or self.config.pe_type == "ALIBI" or self.is_fa:
            default_input.extend(['attention_mask'])
        else:
            default_input.extend(['block_tables'])
            if self.speculate_enable:
                default_input.extend(['attention_mask', 'q_len'])
        return default_input

    def reshape_parallel(self, org_shape):
        if len(org_shape) == 3:
            if self.layer_id == 0:
                return [org_shape[0], org_shape[1] * org_shape[2]]
            else:
                return [org_shape[1], org_shape[0] * org_shape[2]]
        else:
            return org_shape

    def build_graph(self, graph, is_prefill):
        self.layer_graph = AtbGraph(("prefill" if is_prefill else "decode") + f"_layer_{self.layer_id}_graph")
        self.layer_graph.add_input_output(
            input=self.weight_names + ["k_cache", "v_cache"] + self.get_in_tensor_names(is_prefill),
            output=["layer_out"])
        if self.is_reshape:
            self.layer_graph.add_reshape(IntermediateTensorStr.hidden_states, 
                IntermediateTensorStr.hidden_states, self.reshape_parallel)
        self.input_layernorm.build_graph(self.layer_graph, is_prefill)
        self.self_attn.build_graph(self.layer_graph, is_prefill)
        self.post_attention_layernorm.build_graph(self.layer_graph, is_prefill)
        self.block_sparse_moe.build_graph(self.layer_graph, is_prefill)
        self.layer_graph.build()
        
        graph.operations.append(self.layer_graph)
        graph.add_operation(self.layer_graph, self.weight_names + \
        [f"layer_{self.layer_id}_k_cache", f"layer_{self.layer_id}_v_cache"] + self.get_in_tensor_names(
            is_prefill), [IntermediateTensorStr.hidden_states])