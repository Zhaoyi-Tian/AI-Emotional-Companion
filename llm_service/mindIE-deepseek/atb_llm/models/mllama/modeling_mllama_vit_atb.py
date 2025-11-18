# coding=utf-8
# Copyright 2024 the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Mllama model."""

import math
import json
from typing import Tuple, Union
from collections import OrderedDict, defaultdict

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

import _libatb_torch as atb
from transformers.modeling_outputs import BaseModelOutput
from atb_llm.common_op_builders.data_type import CommonOpBuilderType
from atb_llm.common_op_builders.common_op_builder_manager import CommonOpBuilderManager
from atb_llm.common_op_builders.linear_parallel.base_linear_parallel_common_op_builder import ParallelType, \
    TensorParallelInfo, CommunicationBackend
from atb_llm.models.base.model_utils import AttnLinearInfo, MlpLinearInfo
from atb_llm.models.base.flash_causal_lm_atb import AtbGraph
from atb_llm.utils.quantize.quant_type import is_same_type, LinearTypeV2
from atb_llm.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    load_column_multi,
)
from .config_mllama import MllamaVisionConfig

VIT = 'vit'
_HIDDEN_STATES = 'hidden_states'


def _prepare_aspect_ratio_attention_mask(
    aspect_ratio_mask: torch.Tensor,
    num_patches: int,
    target_length: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    # Expand aspect ratio mask to target_length
    batch_size, max_num_tiles = aspect_ratio_mask.shape
    attention_mask = aspect_ratio_mask.view(
        batch_size, max_num_tiles, 1, 1).to(dtype)
    attention_mask = attention_mask.repeat(1, 1, target_length, 1)

    # Mask padding patches
    pad_patches = target_length - num_patches
    attention_mask[:, :, -pad_patches:] = 0

    # Invert the mask (0 -> 1, 1 -> 0)
    attention_mask = 1 - attention_mask

    # Reshape to 2D and create 4D attention mask
    attention_mask = attention_mask.reshape(
        batch_size, max_num_tiles * target_length, 1)
    attention_mask = attention_mask @ attention_mask.transpose(-1, -2)

    return attention_mask


class MllamaPrecomputedAspectRatioEmbedding(nn.Module):
    def __init__(self, config: MllamaVisionConfig, is_gated: bool = True):
        super().__init__()
        self.max_num_tiles = config.max_num_tiles
        self.hidden_size = config.hidden_size
        self.max_aspect_ratio_id = config.max_aspect_ratio_id
        self.is_gated = is_gated

        self.embedding = nn.Embedding(
            self.max_aspect_ratio_id + 1, self.max_num_tiles * self.hidden_size)
        if is_gated:
            self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_states: torch.Tensor, aspect_ratio_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(aspect_ratio_ids)
        embeddings = embeddings.reshape(-1,
                                        self.max_num_tiles, 1, self.hidden_size)

        if self.is_gated:
            embeddings = embeddings * self.gate.tanh()

        hidden_states = hidden_states + embeddings
        return hidden_states


class MllamaPrecomputedPositionEmbedding(nn.Module):
    def __init__(self, config: MllamaVisionConfig):
        super().__init__()
        self.max_num_tiles = config.max_num_tiles
        self.max_aspect_ratio_id = config.max_aspect_ratio_id
        self.num_patches = (config.image_size // config.patch_size) ** 2 + 1
        self.hidden_size = config.hidden_size
        self.scale = config.hidden_size**-0.5

        self.gate = nn.Parameter(torch.zeros(1))

        # position embedding
        position_embedding = torch.randn(self.num_patches, self.hidden_size)
        self.embedding = nn.Parameter(self.scale * position_embedding)

        # tile position embedding
        self.tile_embedding = nn.Embedding(
            self.max_aspect_ratio_id + 1, self.max_num_tiles *
            self.num_patches * self.hidden_size
        )

    def forward(self, hidden_states: torch.Tensor, aspect_ratio_ids: torch.Tensor) -> torch.Tensor:
        # position embeddings
        gated_position_embedding = (1 - self.gate.tanh()) * self.embedding
        hidden_states = hidden_states + \
            gated_position_embedding.view(
                1, 1, self.num_patches, self.hidden_size)

        # precomputed tile position embeddings
        tile_position_embedding = self.tile_embedding(aspect_ratio_ids)
        batch_size = hidden_states.shape[0]
        tile_position_embedding = tile_position_embedding.reshape(
            batch_size, self.max_num_tiles, self.num_patches, self.hidden_size
        )
        gated_tile_position_embedding = self.gate.tanh() * tile_position_embedding
        hidden_states = hidden_states + gated_tile_position_embedding

        return hidden_states


class BaseVitMLP(nn.Module):
    def __init__(self, prefix, config, weights, norm_prefix, backend=CommunicationBackend.LCCL):
        super().__init__()

        # 配置信息
        process_group = weights.process_group
        self.prefix = prefix
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.dtype = weights.dtype
        self.quantize = config.quantize
        self.linear_info = MlpLinearInfo()
        self.norm_prefix = norm_prefix
        self.backend = backend

        # 模型结构
        # Gate Up
        self.fc1 = TensorParallelColumnLinear.load(
            config,
            prefix=f"{prefix}.fc1",
            weights=weights,
            bias=True,
        )
        self.linear_info.fc1_linear = self.fc1.linear
        # Down
        self.fc2 = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.fc2",
            weights=weights,
            bias=True,
        )
        self.linear_info.fc2_linear = self.fc2.linear

    def get_weights(self, prefix):
        weights_dict = OrderedDict()
        weights_dict.update(self.fc1.linear.get_weights(f"{prefix}.fc1"))
        weights_dict.update(self.fc2.linear.get_weights(f"{prefix}.fc2"))
        return weights_dict

    def build_activation_graph(self, graph):
        act = atb._BaseOperation(
            op_type="Activation",
            op_param=json.dumps({'activationType': 'ACTIVATION_GELU'}),
            op_name="Activation_gelu",
        )
        gelu_input_list = ["fc1_out"]
        gelu_output_list = ["activation_out"]
        graph.operations.append(act)
        graph.add_operation(
            act,
            gelu_input_list,
            gelu_output_list,
        )

    def build_fc1_graph(self, graph):
        input_key_list = [f"{self.norm_prefix}_out",
                          f"{self.prefix}.fc1.weight", f"{self.prefix}.fc1.bias"]
        linear_out = ["fc1_out"]
        linear_op = atb._BaseOperation(
            op_type="Linear",
            op_param=json.dumps({
                "transposeB": True,
                "hasBias": True}),
            op_name="fc1_linear"
        )
        graph.operations.append(linear_op)
        graph.add_operation(
            linear_op,
            input_key_list,
            linear_out
        )

    def build_fc2_graph(self, graph):
        fc2_linear_param = {
            "op_name": "fc2_linear",
            "category": CommonOpBuilderType.LINEAR,
            "linear_module": self.fc2.linear,
            "enable_quant_input": False,
            "default_dtype": self.dtype,
        }
        fc2_linear_tensor_map = {
            "input": 'activation_out',
            "linear_out": 'mlp_out'
        }
        fc2_linear_parallel_param = {
            "op_name": "fc2_linear_parallel",
            "category": CommonOpBuilderType.LINEAR_PARALLEL,
            "parallel_type": ParallelType.ALL_REDUCE,
            "parallel_info": TensorParallelInfo(rank=self.tp_rank, world_size=self.tp_world_size,
                                                backend=self.backend),
            "linear_param": fc2_linear_param,
            "enable_lcoc": False,
        }
        linear_parallel_builder = CommonOpBuilderManager.get_builder(
            fc2_linear_parallel_param)
        graph = linear_parallel_builder.build(graph, fc2_linear_tensor_map)

    def build_graph(self, graph):
        self.build_fc1_graph(graph)
        self.build_activation_graph(graph)
        self.build_fc2_graph(graph)


class BaseViTAttention(torch.nn.Module):
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
        self.backend = backend
        self.bias = bias

        self.num_heads = config.attention_heads
        self.num_key_value_heads = self.num_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_heads

        if self.num_key_value_heads < self.tp_world_size:
            repeat_times = self.tp_world_size // self.num_key_value_heads
        else:
            repeat_times = 1
        self.num_heads_pre_rank = (
            self.num_heads + self.tp_world_size - 1) // self.tp_world_size
        self.num_key_value_heads_per_rank = (self.num_key_value_heads * repeat_times + self.tp_world_size - 1) \
            // self.tp_world_size

        self.linear_info = AttnLinearInfo()
        self.norm_prefix = norm_prefix

        # 模型结构
        # Query Key Value

        query_linear_desc = weights.get_linear_quant_type(
            f'{prefix}.q_proj.weight')
        key_linear_desc = weights.get_linear_quant_type(
            f'{prefix}.k_proj.weight')
        value_linear_desc = weights.get_linear_quant_type(
            f'{prefix}.v_proj.weight')

        if is_same_type([query_linear_desc, key_linear_desc, value_linear_desc]):
            self.query_key_value = load_column_multi(
                config,
                prefixes=[f"{prefix}.q_proj",
                          f"{prefix}.k_proj",
                          f"{prefix}.v_proj"],
                weights=weights,
                head_size=self.head_size,
                bias=self.bias
            )
            self.linear_info.is_pack = True
            self.linear_info.pack_linear = self.query_key_value.linear
            if query_linear_desc in [LinearTypeV2.FLOAT16, LinearTypeV2.BFLOAT16]:
                self.linear_info.is_all_float = True
        else:
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
        if self.linear_info.is_pack:
            weights_dict.update(self.query_key_value.linear.get_weights(
                f"{prefix}.query_key_value"))
        else:
            weights_dict.update(
                self.q_proj.linear.get_weights(f"{prefix}.q_proj"))
            weights_dict.update(
                self.k_proj.linear.get_weights(f"{prefix}.k_proj"))
            weights_dict.update(
                self.v_proj.linear.get_weights(f"{prefix}.v_proj"))
        weights_dict.update(self.o_proj.linear.get_weights(f"{prefix}.o_proj"))
        return weights_dict

    def reshape_q(self, org_shape):
        return [org_shape[0], self.num_heads_pre_rank, self.head_size]

    def reshape_kv(self, org_shape):
        return [org_shape[0], self.num_key_value_heads_per_rank, self.head_size]

    def reshape_to_hidden_states(self, org_shape):
        return [org_shape[0], org_shape[1] * org_shape[2]]

    def build_qkv_graph(self, graph):
        linear_modules = []
        if self.linear_info.is_pack:
            linear_modules = [self.linear_info.pack_linear]
        else:
            linear_modules = [self.linear_info.q_linear,
                              self.linear_info.k_linear, self.linear_info.v_linear]

        qkv_linear_param = {
            "op_name": "qkv_split",
            "category": CommonOpBuilderType.QKV,
            "is_pack": self.linear_info.is_pack,
            "is_fa": True,
            "head_dim": self.head_size,
            "head_num": self.num_heads_pre_rank,
            "kv_head_num": self.num_key_value_heads_per_rank,
            "linear_modules": linear_modules,
            "linear_param": {
                "op_name": "q_linear",
                "category": CommonOpBuilderType.LINEAR,
                "enable_quant_input": False,
                "default_dtype": self.dtype,
            }
        }
        qkv_linear_tensor_map = {
            "input": f'{self.norm_prefix}_out',
            "q_out": 'intermediate_q',
            "k_out": 'intermediate_k',
            "v_out": 'intermediate_v'
        }

        qkv_linear_builder = CommonOpBuilderManager.get_builder(
            qkv_linear_param)
        graph = qkv_linear_builder.build(graph, qkv_linear_tensor_map)

    def build_attention_graph(self, graph):
        attention_op = atb._BaseOperation(
            op_type="SelfAttention",
            op_param=json.dumps({
                "headNum": self.num_heads_pre_rank,
                "kvHeadNum": self.num_heads_pre_rank,
                "qkScale": 1.0 / math.sqrt(self.head_size),
                "calcType": "PA_ENCODER",
                "maskType": "MASK_TYPE_NORM"}),
            op_name="selfattention"
        )
        graph.add_reshape("intermediate_q",
                          "intermediate_q_reshape", self.reshape_q)
        graph.add_reshape("intermediate_k",
                          "intermediate_k_reshape", self.reshape_kv)
        graph.add_reshape("intermediate_v",
                          "intermediate_v_reshape", self.reshape_kv)

        graph.operations.append(attention_op)
        input_key_list = ["intermediate_q_reshape", "intermediate_k_reshape",
                          "intermediate_v_reshape", "attention_mask", "seq_len"]
        output_key_name = "attn_out"
        output_key_list = [output_key_name]
        graph.add_operation(attention_op, input_key_list, output_key_list)
        graph.add_reshape(output_key_name, output_key_name, self.reshape_to_hidden_states)

    def build_dense_graph(self, graph):
        dense_linear_param = {
            "op_name": "dense_linear",
            "category": CommonOpBuilderType.LINEAR,
            "linear_module": self.linear_info.dense_linear,
            "enable_quant_input": False,
            "default_dtype": self.dtype,
        }
        dense_linear_parallel_param = {
            "op_name": "dense_linear_parallel",
            "category": CommonOpBuilderType.LINEAR_PARALLEL,
            "parallel_type": ParallelType.ALL_REDUCE,
            "parallel_info": TensorParallelInfo(rank=self.tp_rank, world_size=self.tp_world_size,
                                                backend=self.backend),
            "linear_param": dense_linear_param,
            "enable_lcoc": False,
        }
        dense_linear_tensor_map = {
            "input": "attn_out",
            "linear_out": 'dense_out'
        }
        linear_parallel_builder = CommonOpBuilderManager.get_builder(
            dense_linear_parallel_param)
        graph = linear_parallel_builder.build(graph, dense_linear_tensor_map)

    def build_graph(self, graph):
        self.build_qkv_graph(graph)
        self.build_attention_graph(graph)
        self.build_dense_graph(graph)


class BaseLayerNorm(nn.Module):
    def __init__(self, config, weights, prefix, op_name='layer_norm'):
        super().__init__()

        # 配置信息
        self.prefix = prefix
        self.config = config
        self.has_bias = False
        self.layer_type = "LAYER_NORM_NORM"
        self.quant_type = "QUANT_UNDEFINED"
        self.norm_eps = config.norm_eps
        self.begin_norm_axis = 1

        self.op_type = "LayerNorm"
        self.op_param = {
            "layerType": self.layer_type,
            "normParam": {
                "quantType": self.quant_type,
                "epsilon ": self.norm_eps,
                "beginNormAxis": self.begin_norm_axis,
                "beginParamsAxis": self.begin_norm_axis
            }
        }
        self.op_name = op_name

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

    def build_graph(self, graph, input_tensor_name='hidden_states'):
        layernorm_op = atb._BaseOperation(
            op_type=self.op_type, op_param=json.dumps(self.op_param), op_name=self.op_name)
        graph.add_operation(layernorm_op, [
                            input_tensor_name, f"{self.prefix}.weight", f"{self.prefix}.bias"], [f'{self.prefix}_out'])
        graph.operations.append(layernorm_op)


class MllamaVisionEncoderLayerATB(nn.Module):
    def __init__(
            self,
            config,
            weights,
            model_prefix: str,
            layer_id: int,
            is_gated: bool,
            backend=CommunicationBackend.LCCL):
        super().__init__()

        self.layer_id = layer_id
        self.prefix = f'{model_prefix}.layers.{layer_id}'
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.max_num_tiles = config.max_num_tiles
        self.hidden_size = config.hidden_size
        self.num_channels = config.num_channels
        self.intermediate_layers_indices = config.intermediate_layers_indices if 'global' not in self.prefix else []
        self.intermediate_layers_indices = [ind - 1 for ind in self.intermediate_layers_indices]
        self.is_gated = is_gated
        self.out_tensor_name = 'layer_out'

        self.num_patches = (self.image_size // self.patch_size) ** 2 + 1
        self.scale = config.hidden_size**-0.5

        self.input_layernorm = BaseLayerNorm(
            config, weights, f'{self.prefix}.input_layernorm')

        self.self_attn = BaseViTAttention(
            config=config, weights=weights, prefix=f'{self.prefix}.self_attn', 
            norm_prefix=f'{self.prefix}.input_layernorm', backend=backend
        )

        self.post_attention_layernorm = BaseLayerNorm(
            config, weights, f'{self.prefix}.post_attention_layernorm')

        self.mlp = BaseVitMLP(
            prefix=f"{self.prefix}.mlp", config=config, weights=weights,
            norm_prefix=f"{self.prefix}.post_attention_layernorm", backend=backend
        )

        if is_gated:
            self.gate_attn = nn.Parameter(weights.get_tensor(
                f"{self.prefix}.gate_attn").tanh(), requires_grad=False)
            self.gate_ffn = nn.Parameter(weights.get_tensor(
                f"{self.prefix}.gate_ffn").tanh(), requires_grad=False)

    def get_weights(self, prefix):
        weights_dict = OrderedDict()
        for name, module in self.named_children():
            weights_dict.update(module.get_weights(f"{prefix}.{name}"))
        if self.is_gated:
            weights_dict[f"{self.prefix}.gate_attn"] = self.gate_attn.data
            weights_dict[f"{self.prefix}.gate_ffn"] = self.gate_ffn.data
        self.weight_names = list(weights_dict.keys())
        return weights_dict

    def get_in_tensor_names(self):
        return ['hidden_states', 'attention_mask', 'seq_len']

    def get_out_tensor_names(self):
        return [self.out_tensor_name]

    def build_res_add(self, graph, op_name, add_first, add_second, out):
        res_add = atb._BaseOperation(op_type="Elewise", op_param=json.dumps({'elewiseType': 'ELEWISE_ADD'}),
                                     op_name=f'{op_name}_res_add')
        graph.add_operation(res_add, [add_first, add_second], [out])
        graph.operations.append(res_add)

    def build_gate_mul(self, graph, in_name, gate_name):
        gate_op = atb._BaseOperation(op_type="Elewise", op_param=json.dumps({'elewiseType': 'ELEWISE_MUL'}),
                                            op_name="gate_op")
        graph.operations.append(gate_op)
        graph.add_operation(gate_op, [in_name, gate_name], [in_name])

    def build_graph(self, graph, is_final_layer=False):
        self.layer_graph = AtbGraph(f"{self.prefix.replace('.', '_')}_graph")
        self.layer_graph.add_input_output(
            input=self.weight_names + self.get_in_tensor_names(), output=self.get_out_tensor_names())

        self.input_layernorm.build_graph(self.layer_graph)
        self.self_attn.build_graph(self.layer_graph)

        if self.is_gated:
            self.build_gate_mul(self.layer_graph, 'dense_out', f"{self.prefix}.gate_attn")

        self.build_res_add(self.layer_graph, op_name='attn_res_add',
                           add_first='hidden_states', add_second='dense_out', out=self.out_tensor_name)
        self.post_attention_layernorm.build_graph(
            self.layer_graph, input_tensor_name=self.out_tensor_name)
        self.mlp.build_graph(self.layer_graph)

        if self.is_gated:
            self.build_gate_mul(self.layer_graph, 'mlp_out', f"{self.prefix}.gate_ffn")

        self.build_res_add(self.layer_graph, op_name='mlp_res_add',
                           add_first=self.out_tensor_name, add_second='mlp_out', out=self.out_tensor_name)
        self.layer_graph.build()

        graph.operations.append(self.layer_graph)
        if self.layer_id in self.intermediate_layers_indices:
            graph.add_operation(self.layer_graph, self.weight_names +
                                self.get_in_tensor_names(), [f"{_HIDDEN_STATES}_{self.layer_id}"])
        elif self.layer_id - 1 in self.intermediate_layers_indices:
            in_tensor_names = self.get_in_tensor_names()
            in_tensor_names[0] = f"{_HIDDEN_STATES}_{self.layer_id-1}"
            graph.add_operation(
                self.layer_graph, self.weight_names + in_tensor_names, [_HIDDEN_STATES])
        elif is_final_layer:
            graph.add_operation(
                self.layer_graph, self.weight_names + self.get_in_tensor_names(), ["model_out"])
        else:
            graph.add_operation(self.layer_graph, self.weight_names +
                                self.get_in_tensor_names(), [_HIDDEN_STATES])


class MllamaVisionEncoderATB(nn.Module):
    def __init__(
            self,
            config,
            weights,
            prefix: str,
            num_layer: int,
            is_gated: bool,
            need_nz: bool,
            backend=CommunicationBackend.LCCL,
        ):
        super().__init__()

        self.intermediate_layers_indices = config.intermediate_layers_indices if 'global' not in prefix else []
        self.intermediate_layers_indices = [ind - 1 for ind in self.intermediate_layers_indices]
        self.num_layer = num_layer
        self.layers = nn.ModuleList([
            MllamaVisionEncoderLayerATB(config, weights, prefix, layer_idx, is_gated, backend) 
                for layer_idx in range(num_layer)
        ])
        self.prefix = prefix
        self.need_nz = need_nz
        self.graph = AtbGraph(f"{self.prefix.replace('.', '_')}_VIT_graph")
        self.graph_inputs = defaultdict(dict)
        self.graph_outputs = defaultdict(dict)
        self.graph_param = defaultdict(dict)
        self.weight = OrderedDict()
        
        self.transdata_operation = None
        if self.need_nz:
            self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
            self.transdata_param = json.dumps({})
            self.transdata_operation.set_param(self.transdata_param)

    def get_in_tensor_names(self):
        return ['hidden_states', 'attention_mask', 'seq_len']

    def get_out_tensor_names(self):
        tensor_names = ['model_out']
        if self.intermediate_layers_indices:
            [tensor_names.append(f'hidden_states_{layer_id}')
             for layer_id in self.intermediate_layers_indices]
        return tensor_names

    def get_weights(self, prefix):
        weights_dict = OrderedDict()
        for name, module in self.named_children():
            if isinstance(module, nn.ModuleList):
                for i, single_module in enumerate(module):
                    weights_dict.update(
                        single_module.get_weights(f"{prefix}.{name}.{i}"))
            else:
                weights_dict.update(module.get_weights(f"{prefix}.{name}"))
        self.weight_names = list(weights_dict.keys())
        return weights_dict

    def prepare_input(self, hidden_states, attention_mask, seq_len):
        if self.need_nz:
            attention_mask = self.transdata_operation.execute([attention_mask])[0]
        self.graph_inputs[VIT].update({"hidden_states": hidden_states,
                                         "attention_mask": attention_mask,
                                         "seq_len": seq_len})
        self.graph_param[VIT]["seq_len"] = seq_len.cpu().to(torch.int32)
        for out_tensor_name in self.get_out_tensor_names():
            self.graph_outputs[VIT][out_tensor_name] = torch.zeros_like(
                hidden_states, dtype=hidden_states.dtype).npu()

    def build_graph(self):
        self.weight = self.get_weights(self.prefix)
        self.graph.add_input_output(input=list(self.weight.keys(
        )) + self.get_in_tensor_names(), output=self.get_out_tensor_names())
        for idx, layer in enumerate(self.layers):
            layer.build_graph(self.graph, is_final_layer=(
                idx == len(self.layers) - 1))
        self.graph.execute_as_single = False
        self.graph.build()
        self.graph.set_weights(self.weight)

    def forward(self, hidden_states, attention_mask):

        bs, token, hidden_size = hidden_states.shape
        seq_len = torch.tensor(bs * [token], dtype=torch.int32).npu()
        hidden_states = hidden_states.view(bs * token, hidden_size)

        self.prepare_input(hidden_states, attention_mask, seq_len)

        graph_output = self.graph.forward(
            self.graph_inputs[VIT], self.graph_outputs[VIT], self.graph_param[VIT])
        hidden_states = graph_output['model_out']
        hidden_states = hidden_states.view(bs, token, hidden_size)
        if self.intermediate_layers_indices:
            inter_hidden_states = []
            for idx in self.intermediate_layers_indices:
                inter_hidden_states.append(
                    graph_output[f'hidden_states_{idx}'])
            inter_hidden_states = torch.stack(
                inter_hidden_states, dim=-1).reshape(hidden_states.shape[0], -1)
            return hidden_states, inter_hidden_states
        else:
            return hidden_states


class MllamaVisionModelATB(nn.Module):
    config_class = MllamaVisionConfig
    base_model_prefix = "vision_model"

    def __init__(
            self,
            config,
            weights,
            model_prefix,
            device,
            dtype,
            backend=CommunicationBackend.LCCL,
            soc_info=None):
        super().__init__()
        self.config = config
        self.model_prefix = model_prefix
        self.weight_names = None
        self.backend = backend
        self.device = device
        self.dtype = dtype
        self.soc_info = soc_info

        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.max_num_tiles = config.max_num_tiles
        self.hidden_size = config.hidden_size
        self.num_channels = config.num_channels

        self.num_patches = (self.image_size // self.patch_size) ** 2 + 1
        self.scale = config.hidden_size**-0.5

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
            bias=False,
        )

        self.class_embedding = nn.Parameter(
            self.scale * torch.randn(self.hidden_size))
        self.gated_positional_embedding = MllamaPrecomputedPositionEmbedding(
            config)

        self.pre_tile_positional_embedding = MllamaPrecomputedAspectRatioEmbedding(
            config, is_gated=True)
        self.post_tile_positional_embedding = MllamaPrecomputedAspectRatioEmbedding(
            config, is_gated=True)

        # layer norms
        self.layernorm_pre = nn.LayerNorm(self.hidden_size)
        self.layernorm_post = nn.LayerNorm(self.hidden_size)

        self.transformer = MllamaVisionEncoderATB(
            config=config, weights=weights, prefix=f"{model_prefix}.transformer", 
            num_layer=config.num_hidden_layers, is_gated=False, backend=backend, 
            need_nz=self.soc_info.need_nz
        )

        self.global_transformer = MllamaVisionEncoderATB(
            config=config, weights=weights, prefix=f"{model_prefix}.global_transformer", 
            num_layer=config.num_global_layers, is_gated=True, backend=backend,
            need_nz=self.soc_info.need_nz
        )

    def build_graph(self):
        self.transformer.build_graph()
        self.global_transformer.build_graph()

    def apply_class_embedding(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, _, hidden_size = hidden_states.shape
        class_embedding = self.class_embedding.expand(
            batch_size, 1, hidden_size)
        hidden_states = torch.cat([class_embedding, hidden_states], dim=1)
        return hidden_states

    def forward(
        self,
        pixel_values: torch.Tensor,
        aspect_ratio_ids: torch.Tensor,
        aspect_ratio_mask: torch.Tensor,
    ) -> Union[BaseModelOutput, Tuple[torch.Tensor, ...]]:

        batch_size, num_concurrent_media, num_tiles, num_channels, height, width = pixel_values.shape

        pixel_values = pixel_values.reshape(
            batch_size * num_concurrent_media * num_tiles, num_channels, height, width)
        aspect_ratio_ids = aspect_ratio_ids.reshape(
            batch_size * num_concurrent_media, -1)

        # Patch embedding
        patch_embeds = self.patch_embedding(
            pixel_values.to(self.dtype).to(self.device))
        hidden_states = patch_embeds.flatten(2).transpose(1, 2)

        # Tile embeddings
        _, num_patches, dim = hidden_states.shape
        hidden_states = hidden_states.reshape(
            batch_size * num_concurrent_media, num_tiles, -1, dim)
        hidden_states = self.pre_tile_positional_embedding(
            hidden_states, aspect_ratio_ids)

        # Add cls token
        hidden_states = hidden_states.reshape(
            batch_size * num_concurrent_media * num_tiles, num_patches, dim)
        hidden_states = self.apply_class_embedding(hidden_states)
        num_patches += 1

        # Position embeddings
        hidden_states = hidden_states.reshape(
            batch_size * num_concurrent_media, num_tiles, num_patches, dim)
        hidden_states = self.gated_positional_embedding(
            hidden_states, aspect_ratio_ids)

        hidden_states = self.layernorm_pre(hidden_states)

        # Compute the number of tokens to pad
        num_padding_patches = (8 - (hidden_states.shape[-2] % 8)) % 8
        # Compute padding tuple for pad function
        # (pad_left, pad_right, pad_left for dim -2, pad_right for dim -2)
        padding = (0, 0, 0, num_padding_patches)
        # Pad the tensor
        hidden_states = F.pad(hidden_states, padding, mode="constant", value=0)
        slice_index = -num_padding_patches if num_padding_patches > 0 else None

        # Prepare attention mask
        attention_mask = aspect_ratio_mask.reshape(
            batch_size * num_concurrent_media, -1)
        attention_mask = _prepare_aspect_ratio_attention_mask(
            aspect_ratio_mask=attention_mask,
            num_patches=self.num_patches,
            target_length=hidden_states.shape[2],
            dtype=self.dtype,
        )

        # Apply encoder
        hidden_states = hidden_states.view(
            batch_size * num_concurrent_media, -1, dim)
        hidden_states, intermediate_hidden_states = self.transformer(
            hidden_states,
            attention_mask=attention_mask,
        )

        hidden_states = self.layernorm_post(hidden_states)

        # Apply global encoder
        hidden_states = hidden_states.reshape(
            batch_size * num_concurrent_media, num_tiles, num_patches + num_padding_patches, dim
        )
        hidden_states = self.post_tile_positional_embedding(
            hidden_states, aspect_ratio_ids)
        hidden_states = hidden_states.reshape(
            batch_size * num_concurrent_media, num_tiles *
            (num_patches + num_padding_patches), dim
        )
        hidden_states = self.global_transformer(
            hidden_states,
            attention_mask=attention_mask,
        )

        # Remove padding form hidden state
        hidden_states = hidden_states.reshape(
            batch_size * num_concurrent_media, num_tiles, num_patches + num_padding_patches, dim
        )
        hidden_states = hidden_states[:, :, :slice_index]
        hidden_states = hidden_states.reshape(
            batch_size, num_concurrent_media, num_tiles, num_patches, dim)

        intermediate_hidden_states = intermediate_hidden_states.reshape(
            batch_size, num_concurrent_media, num_tiles, num_patches + num_padding_patches, -1
        )
        intermediate_hidden_states = intermediate_hidden_states[:,
                                                                :, :, :slice_index]
        # Concatenate final hidden state and intermediate hidden states
        hidden_states = torch.cat(
            [hidden_states, intermediate_hidden_states], dim=-1)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=hidden_states,
            attentions=None,
        )
