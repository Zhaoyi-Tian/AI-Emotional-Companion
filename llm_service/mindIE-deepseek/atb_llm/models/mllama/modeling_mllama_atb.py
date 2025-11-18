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
import math
from collections import OrderedDict
from abc import abstractmethod

import torch
from torch import nn

import _libatb_torch as atb
from atb_llm.common_op_builders.data_type import CommonOpBuilderType, OperationBackend, NormType
from atb_llm.common_op_builders.common_op_builder_manager import CommonOpBuilderManager
from atb_llm.common_op_builders.linear_parallel.base_linear_parallel_common_op_builder import ParallelType, \
    TensorParallelInfo, CommunicationBackend
from atb_llm.common_op_builders.attention.base_attention_common_op_builder import AttnType
from atb_llm.models.base.flash_causal_lm_atb import AtbGraph
from atb_llm.models.base.modeling_atb import BaseRMSNorm, BaseMLP, BaseModelATB
from atb_llm.models.base.model_utils import AttnLinearInfo, LmHeadLinearInfo
from atb_llm.models.llama.modeling_llama_atb import LlamaLayer
from atb_llm.utils.quantize.quant_type import QuantType
from atb_llm.utils import OpBackend
from atb_llm.utils.layers import (
    KvCache,
)
from atb_llm.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
)

LLAMA_EMBEDDING_PARALLEL_THRESHOLD = 128256  # vocab size of llama3
_HIDDEN_STATES = 'hidden_states'
_MLP_OUT = 'mlp_out'


class SimpleRMSNorm(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()

        # 配置信息
        self.prefix = prefix
        self.config = config
        self.has_bias = False
        
        # 模型结构
        self.module = None
        weight = weights.get_tensor(f"{prefix}.weight")
        self.weight = nn.Parameter(weight)
        
        try:
            bias = weights.get_tensor(f"{prefix}.bias")
        except AssertionError:
            self.bias = None
        else:
            self.bias = nn.Parameter(bias)
            self.has_bias = True

    def get_weights(self, prefix):
        weights_dict = OrderedDict()
        weights_dict[f"{prefix}.weight"] = self.weight.data
        if self.has_bias:
            weights_dict[f"{prefix}.bias"] = self.bias.data

        return weights_dict

    def build_graph(self, graph, is_prefill, tensor_map=None):
        quant_type = "QUANT_UNDEFINED"
        
        norm_op_param = {
            "op_name": "norm",
            "category": CommonOpBuilderType.NORM,
            "has_bias": self.has_bias,
            "enable_add_norm": False,
            "norm_type": NormType.RMSNORM,
            "norm_param": {
                'layerType': 'RMS_NORM_NORM',
                'normParam': {
                    'quantType': quant_type,
                    'epsilon': self.config.rms_norm_eps
                }
            }
        }
        norm_tensor_map = {
            "input": 'hidden_states',
            "weight": f"{self.prefix}.weight",
            "norm_out": f"{self.prefix}_out",
        } if tensor_map is None else tensor_map
        if self.has_bias:
            norm_tensor_map.update({"bias": f"{self.prefix}.bias"})

        builder = CommonOpBuilderManager.get_builder(norm_op_param)
        graph = builder.build(graph, norm_tensor_map)


class BaseCrossAttention(torch.nn.Module):
    def __init__(
            self,
            config,
            weights,
            prefix: str,
            norm_prefix: str,
            is_fa: bool = False,
            backend=CommunicationBackend.LCCL,
            bias: bool = False
    ):
        super().__init__()

        # 配置信息
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.dtype = weights.dtype
        self.quantize = config.quantize
        self.prefix = prefix
        self.is_fa = is_fa
        self.backend = backend
        self.bias = bias

        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_heads

        if self.num_key_value_heads < self.tp_world_size:
            repeat_times = self.tp_world_size // self.num_key_value_heads
        else:
            repeat_times = 1
        self.num_heads_pre_rank = (self.num_heads + self.tp_world_size - 1) // self.tp_world_size
        self.num_key_value_heads_per_rank = (self.num_key_value_heads * repeat_times + self.tp_world_size - 1) \
                                            // self.tp_world_size

        self.linear_info = AttnLinearInfo()
        self.norm_prefix = norm_prefix


        self.q_proj = TensorParallelColumnLinear.load(
            config,
            prefix=f"{prefix}.q_proj",
            weights=weights,
            bias=False,
        )
        self.k_proj = TensorParallelColumnLinear.load(
            config,
            prefix=f"{prefix}.k_proj",
            weights=weights,
            bias=False,
        )
        self.v_proj = TensorParallelColumnLinear.load(
            config,
            prefix=f"{prefix}.v_proj",
            weights=weights,
            bias=False,
        )
        self.linear_info.is_pack = False
        self.linear_info.split_num = 3
        self.linear_info.q_linear = self.q_proj.linear
        self.linear_info.k_linear = self.k_proj.linear
        self.linear_info.v_linear = self.v_proj.linear

        # Dense
        self.o_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.o_proj",
            weights=weights,
            bias=False,
            gqa_size=self.head_size,
        )
        self.linear_info.dense_linear = self.o_proj.linear

    def get_weights(self, prefix):
        weights_dict = OrderedDict()
        weights_dict.update(self.q_proj.linear.get_weights(f"{prefix}.q_proj"))
        weights_dict.update(self.k_proj.linear.get_weights(f"{prefix}.k_proj"))
        weights_dict.update(self.v_proj.linear.get_weights(f"{prefix}.v_proj"))
        weights_dict.update(self.o_proj.linear.get_weights(f"{prefix}.o_proj"))
        return weights_dict

    def reshape_q(self, org_shape):
        return [org_shape[0], self.num_heads_pre_rank, self.head_size]

    def reshape_kv(self, org_shape):
        return [org_shape[0], self.num_key_value_heads_per_rank, self.head_size]

    def build_qkv_graph(self, graph):
        linear_modules = [self.linear_info.q_linear, self.linear_info.k_linear, self.linear_info.v_linear]
        
        qkv_linear_param = {
            "op_name": "qkv_split",
            "category": CommonOpBuilderType.QKV,
            "is_pack": self.linear_info.is_pack,
            "is_fa": self.is_fa,
            "head_dim": self.head_size,
            "head_num": self.num_heads_pre_rank,
            "kv_head_num":self.num_key_value_heads_per_rank,
            "linear_modules": linear_modules,
            "linear_param": {
                "op_name": "qkv_linear",
                "category": CommonOpBuilderType.LINEAR,
                "enable_quant_input": False,
                "default_dtype": self.dtype,
                "group_size": 128 if self.quantize == QuantType.W4A16 else 0
            }
        }
        qkv_linear_tensor_map = {
            "input": f'{self.norm_prefix}_out',
            "input_k": 'cross_attention_states',
            "input_v": 'cross_attention_states',
            "q_out": 'intermediate_q',
            "k_out": 'intermediate_k',
            "v_out": 'intermediate_v'
        }

        qkv_linear_builder = CommonOpBuilderManager.get_builder(qkv_linear_param)
        graph = qkv_linear_builder.build(graph, qkv_linear_tensor_map)
    
    def build_dense_graph(self, graph, is_prefill):
        dense_linear_param = {
            "op_name": "dense_linear",
            "category": CommonOpBuilderType.LINEAR,
            "linear_module": self.linear_info.dense_linear,
            "enable_quant_input": True,
            "default_dtype": self.dtype,
            "group_size": 128 if self.quantize == QuantType.W4A16 else 0
        }
        dense_linear_parallel_param = {
            "op_name": "dense_linear_parallel",
            "category": CommonOpBuilderType.LINEAR_PARALLEL,
            "parallel_type": ParallelType.ALL_REDUCE,
            "parallel_info": TensorParallelInfo(rank=self.tp_rank, world_size=self.tp_world_size,
                                                backend=self.backend),
            "linear_param": dense_linear_param,
            "enable_lcoc": True if is_prefill else False,
        }
        dense_linear_tensor_map = {
            "input": "attn_out",
            "linear_out": 'dense_out'
        }
        linear_parallel_builder = CommonOpBuilderManager.get_builder(dense_linear_parallel_param)
        graph = linear_parallel_builder.build(graph, dense_linear_tensor_map)

    @abstractmethod
    def build_graph(self, graph, is_prefill):
        pass

    def _get_cross_attention_tensor_map(self, is_prefill):
        attention_tensor_map = {
            "q": "intermediate_q",
            "k_cache": "k_cache",
            "v_cache": "v_cache",
            "attention_mask": "cross_attention_mask",
            "seq_len": "cross_context_lens",
            "block_tables": "block_tables",
            "attention_out": "attn_out",
        }
        if is_prefill:
            attention_tensor_map.update({
                "k": "intermediate_k",
                "v": "intermediate_v",
                "slots": "cross_slots_mapping",
            })
        return attention_tensor_map
    
    def _get_inter_norm_tensor_map(self, part):
        norm_tensor_map = {
                "input": f"attention_reshape_{part}",
                "weight": f"{self.prefix}.{part}_norm.weight",
                "norm_out": f"attention_reshape_{part}",
            }
        return norm_tensor_map

    def _get_atb_attention_param(self, is_prefill):
        atb_attention_param = {
            'headNum': self.num_heads_pre_rank,
            'kvHeadNum': self.num_key_value_heads_per_rank,
            'qkScale': 1.0 / math.sqrt(self.head_size), 
            'maskType': 'MASK_TYPE_SPEC',
            'calcType': 'PA_ENCODER',
        }
        return atb_attention_param


class MllamaCrossAttention(BaseCrossAttention):
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
            
        self.q_norm = SimpleRMSNorm(f"{prefix}.q_norm", config, weights)
        self.k_norm = SimpleRMSNorm(f"{prefix}.k_norm", config, weights)


    def get_weights(self, prefix):
        weights_dict = super().get_weights(prefix)
        if self.kv_quant is not None:
            weights_dict.update(self.kv_cache_quant.get_weights(f"{prefix}"))

        weights_dict.update(self.q_norm.get_weights(f"{prefix}.q_norm"))
        weights_dict.update(self.k_norm.get_weights(f"{prefix}.k_norm"))
        return weights_dict

    def build_cross_attention_graph(self, graph, is_prefill):
        attention_param = {
            "op_name": "attention",
            "category": CommonOpBuilderType.ATTENTION,
            "is_prefill": False,
            "attn_type": AttnType.PAGED_ATTENTION,
            "head_size": self.head_size,
            "atb_reshape_and_cache_param": {},
            "operation_backend": OperationBackend.ATB,
            "enable_kv_quant": self.kv_quant is not None,
            "kv_quant_module": self.kv_cache_quant,
            "need_reshape_and_cache": True if is_prefill else False,
            "need_input_reshape": False
        }

        atb_attention_param = self._get_atb_attention_param(is_prefill)
        attention_param.update({"atb_attention_param": atb_attention_param})

        attention_tensor_map = self._get_cross_attention_tensor_map(is_prefill)

        pa_attention_builder = CommonOpBuilderManager.get_builder(attention_param)
        graph = pa_attention_builder.build(graph, attention_tensor_map)

    def build_q_graph(self, graph):
        linear_module = self.linear_info.q_linear
        linear_param = {
                "op_name": "q_linear",
                "category": CommonOpBuilderType.LINEAR,
                "enable_quant_input": False,
                "default_dtype": self.dtype,
                "group_size": 128 if self.quantize == QuantType.W4A16 else 0
            }
        linear_param.update({'linear_module': linear_module})

        linear_tensor_map = {
            "input": f'{self.norm_prefix}_out',
            "linear_out": 'intermediate_q',
        }
        q_linear_builder = CommonOpBuilderManager.get_builder(linear_param)
        graph = q_linear_builder.build(graph, linear_tensor_map)

    def build_graph(self, graph, is_prefill):
        if is_prefill:
            self.build_qkv_graph(graph)
            graph.add_reshape("intermediate_k", "attention_reshape_k", self.reshape_kv)
            graph.add_reshape("intermediate_v", "attention_reshape_v", self.reshape_kv)
            self.k_norm.build_graph(graph, is_prefill, self._get_inter_norm_tensor_map("k"))
        else:
            self.build_q_graph(graph)
        graph.add_reshape("intermediate_q", "attention_reshape_q", self.reshape_q)
        self.q_norm.build_graph(graph, is_prefill, self._get_inter_norm_tensor_map("q"))
        self.build_cross_attention_graph(graph, is_prefill)
        self.build_dense_graph(graph, is_prefill)


class MllamaCrossAttentionLayer(nn.Module):
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
        self.is_reshape = config.vocab_size >= LLAMA_EMBEDDING_PARALLEL_THRESHOLD and tp_world_size > 1 and not is_fa
        self.weight_names = None
        self.layer_graph = None
        self.is_fa = is_fa
        self.speculate_enable = speculate_enable
        self.prefix = prefix

        # 模型结构
        self.cross_attn = MllamaCrossAttention(
            config=config, weights=weights, prefix=f"{prefix}.cross_attn", norm_prefix=f"{prefix}.input_layernorm", \
            is_fa=self.is_fa, backend=backend, speculate_enable=self.speculate_enable
        )

        self.input_layernorm = SimpleRMSNorm(f"{prefix}.input_layernorm", config, weights)

        self.cross_attn_attn_gate = nn.Parameter(
            weights.get_tensor(f"{prefix}.cross_attn_attn_gate").tanh(), 
            requires_grad=False
        )
        
        self.mlp = BaseMLP(
            prefix=f"{prefix}.mlp", config=config, weights=weights,
            norm_prefix=f"{prefix}.post_attention_layernorm", backend=backend
        )
        
        self.post_attention_layernorm = BaseRMSNorm(
            f"{prefix}.post_attention_layernorm", config, weights, self.mlp.linear_info
        )

        self.cross_attn_mlp_gate = nn.Parameter(
            weights.get_tensor(f"{prefix}.cross_attn_mlp_gate").tanh(), 
            requires_grad=False
        )
        

    def get_weights(self, prefix):
        weights_dict = OrderedDict()
        for name, module in self.named_children():
            weights_dict.update(module.get_weights(f"{prefix}.{name}"))
        weights_dict[f"{prefix}.cross_attn_attn_gate.weight"] = self.cross_attn_attn_gate.data
        weights_dict[f"{prefix}.cross_attn_mlp_gate.weight"] = self.cross_attn_mlp_gate.data
        self.weight_names = list(weights_dict.keys())
        return weights_dict

    def get_in_tensor_names(self, is_prefill):
        default_input = ['hidden_states', 'seq_len']
        default_input.extend(['block_tables', 'cross_attention_mask', 
                              'full_text_row_masked_out_mask', 'cross_context_lens'])
        if is_prefill:
            default_input.extend(['cross_attention_states', 'cross_slots_mapping'])

        if self.is_fa:
            default_input.extend(['token_offset', 'layer_id'])
        return default_input

    def reshape_parallel(self, org_shape):
        if len(org_shape) == 3:
            if self.layer_id == 0:
                return [org_shape[0], org_shape[1] * org_shape[2]]
            else:
                return [org_shape[1], org_shape[0] * org_shape[2]]
        else:
            return org_shape
        
    def build_multiply_gate_and_add_res(self, graph, gate_states, gate_name, 
                                        add_name='hidden_states', out_name='hidden_states'):
        multiply_gate = atb._BaseOperation(op_type="Elewise", op_param=json.dumps({'elewiseType': 'ELEWISE_MUL'}),
                                           op_name=f"atten_gate_mul_{gate_name}")
        graph.operations.append(multiply_gate)
        graph.add_operation(multiply_gate, [gate_states, f"{self.prefix}.{gate_name}.weight"], [gate_states])


        res_add = atb._BaseOperation(op_type="Elewise", op_param=json.dumps({'elewiseType': 'ELEWISE_ADD'}),
                                           op_name=f"atten_res_add_{gate_name}")
        graph.operations.append(res_add)
        graph.add_operation(res_add, [gate_states, add_name], [out_name])


    def build_graph(self, graph, is_prefill):
        self.layer_graph = AtbGraph(("prefill" if is_prefill else "decode") + f"_layer_{self.layer_id}_graph")
        self.layer_graph.add_input_output(
            input=self.weight_names + ["k_cache", "v_cache"] + self.get_in_tensor_names(is_prefill),
            output=["layer_out"])
        if self.is_reshape:
            self.layer_graph.add_reshape(_HIDDEN_STATES, _HIDDEN_STATES, self.reshape_parallel)

        self.input_layernorm.build_graph(self.layer_graph, is_prefill)
        self.cross_attn.build_graph(self.layer_graph, is_prefill)
        self.build_multiply_gate_and_add_res(self.layer_graph, gate_states='dense_out', 
                                             gate_name='cross_attn_attn_gate')

        self.post_attention_layernorm.build_graph(self.layer_graph, is_prefill)
        self.mlp.build_graph(self.layer_graph, is_prefill, need_res_add=False)

        full_row_mul = atb._BaseOperation(op_type="Elewise", op_param=json.dumps({'elewiseType': 'ELEWISE_MUL'}),
                                           op_name="full_row_mul")
        self.layer_graph.operations.append(full_row_mul)
        self.layer_graph.add_operation(full_row_mul, [_MLP_OUT, "full_text_row_masked_out_mask"], [_MLP_OUT])

        self.build_multiply_gate_and_add_res(self.layer_graph, gate_states=_MLP_OUT, 
                                             gate_name='cross_attn_mlp_gate', out_name='layer_out')

        self.layer_graph.build()
        graph.operations.append(self.layer_graph)
        graph.add_operation(self.layer_graph, self.weight_names + \
        [f"layer_{self.layer_id}_k_cache", f"layer_{self.layer_id}_v_cache"] + self.get_in_tensor_names(
            is_prefill), [_HIDDEN_STATES])


class MllamaModelATB(BaseModelATB):
    def __init__(
        self,
        config,
        weights,
        model_prefix: str = "model",
        lm_head_prefix: str = "lm_head",
        is_fa: bool = False,
        backend=CommunicationBackend.LCCL,
        speculate_enable: bool = False,
    ):
        is_parallel = config.vocab_size >= LLAMA_EMBEDDING_PARALLEL_THRESHOLD
        super().__init__(config, weights, model_prefix, lm_head_prefix, is_parallel, is_fa, backend)

        self.layers = nn.ModuleList(
            [MllamaCrossAttentionLayer(layer_idx, config, weights, model_prefix, 
                                       self.is_fa, self.backend, speculate_enable) \
             if layer_idx in (config.cross_attention_layers) else \
             LlamaLayer(layer_idx, config, weights, model_prefix, self.is_fa, self.backend, speculate_enable) \
                for layer_idx in range(config.num_hidden_layers)]
        )

        linear_info = LmHeadLinearInfo()
        linear_info.lm_head_name = lm_head_prefix
        self.norm = BaseRMSNorm(f"{model_prefix}.norm", config, weights, linear_info)
        self.cross_attention_layers = config.cross_attention_layers

    def build_graph(self, graph, is_prefill, is_multimodal=True):
        self.build_word_embedding_graph(graph)
        self.build_positional_embedding_graph(graph)

        for idx, layer in enumerate(self.layers):
            if idx in self.cross_attention_layers:
                if is_multimodal:
                    layer.build_graph(graph, is_prefill)
            else:
                layer.build_graph(graph, is_prefill)

        self.norm.build_graph(graph, is_prefill)
