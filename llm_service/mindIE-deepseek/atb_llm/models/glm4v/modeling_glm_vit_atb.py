# Copyright 2024 THUDM and the HuggingFace Inc. team. All rights reserved.
# 
# This code is referenced from HuggingFace's glm-4v-9b developed by THUDM.
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
from argparse import Namespace
from collections import OrderedDict
import torch
from torch import nn
import torch_npu

from atb_llm.common_op_builders.common_op_builder_manager import CommonOpBuilderManager
from atb_llm.common_op_builders.data_type import (
    CommonOpBuilderType,
    NormType
)
from atb_llm.common_op_builders.linear_parallel.base_linear_parallel_common_op_builder import (
    ParallelType,
    TensorParallelInfo,
    CommunicationBackend
)
from atb_llm.models.base.flash_causal_lm_atb import AtbGraph
from atb_llm.utils.initial import NPUSocInfo
from atb_llm.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear
)
import _libatb_torch as atb

_TRANSPOSE = "Transpose"
_PERMUTE = "perm"


class PatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Conv2d(config.in_channels, config.hidden_size, kernel_size=config.patch_size,
                              stride=config.patch_size)
        self.cls_embedding = nn.Parameter(torch.zeros(1, config.hidden_size))
        self.position_embedding = nn.Embedding(config.num_positions, config.hidden_size)

    def forward(self, images):
        x = self.proj(images)
        x = x.flatten(2).transpose(1, 2)
        cls_token = self.cls_embedding.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x += self.position_embedding.weight.unsqueeze(0)
        return x
        
        
class LayerNormATB(nn.Module):
    def __init__(self, config, weights, prefix, intensor_name):
        super().__init__()
        self.layer_norm_eps = config.layer_norm_eps
        self.prefix = prefix
        self.intensor_name = intensor_name
        weight = weights.get_tensor(f"{prefix}.weight")
        bias = weights.get_tensor(f"{prefix}.bias")
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

    def get_weights(self, prefix):
        weights_dict = OrderedDict()
        weights_dict[f"{prefix}.weight"] = self.weight.data
        weights_dict[f"{prefix}.bias"] = self.bias.data
        return weights_dict

    def build_graph(self, graph):
        norm_param = {
            "layerType": "LAYER_NORM_NORM",
            "normParam": {
                "quantType": "QUANT_UNDEFINED",
                "epsilon": self.layer_norm_eps,
                "beginParamsAxis": 2,
                "beginNormAxis": 2
            },
        }
        norm_op = atb._BaseOperation(
            op_type=NormType.LAYERNORM,
            op_param=json.dumps(norm_param),
            op_name="norm",
        )
        graph.operations.append(norm_op)
        graph.add_operation(
            norm_op, [self.intensor_name, f"{self.prefix}.weight", f"{self.prefix}.bias"], [f"{self.prefix}_out"]
        )


class AttentionATB(nn.Module):
    def __init__(self, config, weights, prefix, backend):
        super().__init__()
        self.prefix = prefix # transformer.vision.transformer.layers.0.attention
        self.dtype = weights.dtype
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5

        setattr(config, 'quantize', None)
        self.quantize = config.quantize

        self.backend = backend
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.num_heads_per_rank = (self.num_heads + self.tp_world_size - 1) // self.tp_world_size

        self.qkv = TensorParallelColumnLinear.load_qkv(
            config,
            prefix=f"{self.prefix}.query_key_value",
            weights=weights,
            bias=True,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_heads
        )
        self.proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{self.prefix}.dense",
            weights=weights,
            bias=True,
            gqa_size=self.head_dim
        )

    def get_weights(self, prefix):
        weights_dict = OrderedDict()
        weights_dict.update(self.qkv.linear.get_weights(f"{prefix}.query_key_value"))
        weights_dict.update(self.proj.linear.get_weights(f"{prefix}.dense"))
        return weights_dict
    
    def build_qkv_graph(self, graph):
        # add qkv linear node
        qkv_linear_in = [
            "hidden_states",
            f"{self.prefix}.query_key_value.weight",
            f"{self.prefix}.query_key_value.bias"
        ]
        qkv_linear_out = ["qkv_linear_out"]
        linear_op = atb._BaseOperation(
            op_type="Linear",
            op_param=json.dumps({
                "transposeB": True,
                "hasBias": True,
            }),
            op_name="qkv_linear"
        )
        graph.operations.append(linear_op)
        graph.add_operation(linear_op, qkv_linear_in, qkv_linear_out)

        # add qkv split node
        qkv_split_in = ["qkv_linear_out"]
        qkv_split_out = ["q_split", "k_split", "v_split"]
        split_op = atb._BaseOperation(
            op_type="Split",
            op_param=json.dumps({
                "splitDim": -1,
                "splitNum": 3
            }),
            op_name="qkv_split"
        )
        graph.operations.append(split_op)
        graph.add_operation(split_op, qkv_split_in, qkv_split_out)

    def reshape_qkv(self, org_shape):
        return [org_shape[0], org_shape[1], self.num_heads_per_rank, org_shape[2] // self.num_heads_per_rank]

    def reshape_attn_out(self, org_shape):
        return [org_shape[0], org_shape[1], self.num_heads_per_rank * org_shape[3]]
                
    
    def build_attention_graph(self, graph):
        graph.add_reshape("q_split", "q_split_reshape", self.reshape_qkv)
        graph.add_reshape("k_split", "k_split_reshape", self.reshape_qkv)
        graph.add_reshape("v_split", "v_split_reshape", self.reshape_qkv)

        transpose_q_in = ["q_split_reshape"]
        transpose_q_out = ["q_transpose"]
        transpose_q_op = atb._BaseOperation(
            op_type=_TRANSPOSE,
            op_param=json.dumps({
                _PERMUTE: [0, 2, 1, 3], # parameter for permutation order
            }),
            op_name="transpose_q"
        )
        graph.operations.append(transpose_q_op)
        graph.add_operation(transpose_q_op, transpose_q_in, transpose_q_out)

        transpose_k_in = ["k_split_reshape"]
        transpose_k_out = ["k_transpose"]
        transpose_k_op = atb._BaseOperation(
            op_type=_TRANSPOSE,
            op_param=json.dumps({
                _PERMUTE: [0, 2, 1, 3],
            }),
            op_name="transpose_k"
        )
        graph.operations.append(transpose_k_op)
        graph.add_operation(transpose_k_op, transpose_k_in, transpose_k_out)

        transpose_v_in = ["v_split_reshape"]
        transpose_v_out = ["v_transpose"]
        transpose_v_op = atb._BaseOperation(
            op_type=_TRANSPOSE,
            op_param=json.dumps({
                _PERMUTE: [0, 2, 1, 3],
            }),
            op_name="transpose_v"
        )
        graph.operations.append(transpose_v_op)
        graph.add_operation(transpose_v_op, transpose_v_in, transpose_v_out)
        
        attention_in = ["q_transpose", "k_transpose", "v_transpose", "q_seq_len", "kv_seq_len"]
        attention_out = ["attn_out"]
        attention_op = atb._BaseOperation(
            op_type="PromptFlashAttention",
            op_param=json.dumps({
                "numHeads": self.num_heads_per_rank,
                "scaleValue": self.scale,
                "preTokens": 65535,
                "nextTokens": 65535,
                "inputLayout": "BNSD_BSND",
            }),
            op_name="attention"
        )
        graph.operations.append(attention_op)
        graph.add_operation(attention_op, attention_in, attention_out)
        graph.add_reshape("attn_out", "attn_out_reshape", self.reshape_attn_out)

    def build_dense_graph(self, graph):
        dense_tensor_map = {
            "input": "attn_out_reshape",
            "linear_out": "dense_out"
        }
        dense_param = {
            "op_name": "dense_linear",
            "category": CommonOpBuilderType.LINEAR,
            "linear_module": self.proj.linear,
            "enable_quant_input": True,
            "default_dtype": self.dtype,
            "group_size": 0,
        }
        dense_parallel_param = {
            "op_name": "dense_parallel_linear",
            "category": CommonOpBuilderType.LINEAR_PARALLEL,
            "parallel_type": ParallelType.ALL_REDUCE,
            "parallel_info": TensorParallelInfo(
                rank=self.tp_rank, world_size=self.tp_world_size, backend=self.backend
            ),
            "linear_param": dense_param,
            "enable_lcoc": False,
        }

        dense_parallel_builder = CommonOpBuilderManager.get_builder(dense_parallel_param)
        graph = dense_parallel_builder.build(graph, dense_tensor_map)

    def build_graph(self, graph):
        self.build_qkv_graph(graph)
        self.build_attention_graph(graph)
        self.build_dense_graph(graph)


class MlpATB(nn.Module):
    def __init__(self, config, weights, prefix, backend):
        super().__init__()
        self.config = config
        self.prefix = prefix
        self.dtype = weights.dtype

        setattr(config, 'quantize', None)
        self.quantize = config.quantize

        self.backend = backend
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()

        self.fc1 = TensorParallelColumnLinear.load(
            config,
            prefix=f"{prefix}.fc1",
            weights=weights,
            bias=True
        )
        self.fc2 = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.fc2",
            weights=weights,
            bias=True
        )

    def get_weights(self, prefix):
        weights_dict = OrderedDict()
        weights_dict.update(self.fc1.linear.get_weights(f"{prefix}.fc1"))
        weights_dict.update(self.fc2.linear.get_weights(f"{prefix}.fc2"))
        return weights_dict

    def build_fc1_graph(self, graph):
        fc1_linear_in = ["hidden_states", f"{self.prefix}.fc1.weight", f"{self.prefix}.fc1.bias"]
        fc1_linear_out = ["fc1_out"]
        linear_op = atb._BaseOperation(
            op_type="Linear",
            op_param=json.dumps({
                "transposeB": True,
                "hasBias": True,
            }),
            op_name="fc1_linear"
        )
        graph.operations.append(linear_op)
        graph.add_operation(linear_op, fc1_linear_in, fc1_linear_out)

    def build_act_graph(self, graph):
        act_in = ["fc1_out"]
        act_out = ["act_out"]
        act_op = atb._BaseOperation(
            op_type="Activation",
            op_param=json.dumps({'activationType': 'ACTIVATION_GELU'}),
            op_name="act_fn",
        )
        graph.operations.append(act_op)
        graph.add_operation(act_op, act_in, act_out)

    def build_fc2_graph(self, graph):
        fc2_tensor_map = {
            "input": "act_out",
            "linear_out": "fc2_out",
        }
        fc2_param = {
            "op_name": "fc2",
            "category": CommonOpBuilderType.LINEAR,
            "linear_module": self.fc2.linear,
            "enable_quant_input": False,
            "default_dtype": self.dtype,
            "group_size": 0,
        }
        fc2_parallel_param = {
            "op_name": "fc2_parallel",
            "category": CommonOpBuilderType.LINEAR_PARALLEL,
            "parallel_type": ParallelType.ALL_REDUCE,
            "parallel_info": TensorParallelInfo(
                rank=self.tp_rank, world_size=self.tp_world_size, backend=self.backend
            ),
            "linear_param": fc2_param,
            "enable_lcoc": False,
        }
        fc2_parallel_builder = CommonOpBuilderManager.get_builder(fc2_parallel_param)
        graph = fc2_parallel_builder.build(graph, fc2_tensor_map)

    def build_graph(self, graph):
        self.build_fc1_graph(graph)
        self.build_act_graph(graph)
        self.build_fc2_graph(graph)


class TransformerLayerATB(nn.Module):
    def __init__(self, config, weights, model_prefix, layer_idx, backend) -> None:
        super().__init__()
        self.config = config
        self.prefix = f"{model_prefix}.{layer_idx}"
        self.layer_idx = layer_idx
        self.weight_names = None
        self.layer_graph = None

        self.attention = AttentionATB(
            config=config,
            weights=weights,
            prefix=f"{self.prefix}.attention",
            backend=backend
        )

        self.input_layernorm = LayerNormATB(
            config=config,
            weights=weights,
            prefix=f"{self.prefix}.input_layernorm",
            intensor_name="dense_out")

        self.mlp = MlpATB(
            config=config,
            weights=weights,
            prefix=f"{self.prefix}.mlp",
            backend=backend
        )
        
        self.post_attention_layernorm = LayerNormATB(
            config=config,
            weights=weights,
            prefix=f"{self.prefix}.post_attention_layernorm",
            intensor_name="fc2_out"
        )

    def get_weights(self, prefix):
        weights_dict = OrderedDict()
        for name, module in self.named_children():
            weights_dict.update(module.get_weights(f"{prefix}.{name}"))
        self.weight_names = list(weights_dict.keys())
        return weights_dict

    def get_intensor_names(self):
        return ["hidden_states", "q_seq_len", "kv_seq_len"]

    def build_graph(self, graph):
        self.layer_graph = AtbGraph(f"VIT_layer_{self.layer_idx}_graph")
        self.layer_graph.add_input_output(
            input=self.weight_names + self.get_intensor_names(),
            output=["layer_out"]
        )
        
        self.attention.build_graph(self.layer_graph)
        self.input_layernorm.build_graph(self.layer_graph)

        # attn residual add
        attn_res_add = atb._BaseOperation(
            op_type="Elewise",
            op_param=json.dumps({"elewiseType": "ELEWISE_ADD"}),
            op_name="attn_res_add",
        )
        setattr(self.layer_graph, "attn_res_add", attn_res_add)
        self.layer_graph.operations.append(self.layer_graph.attn_res_add)
        self.layer_graph.add_operation(
            self.layer_graph.attn_res_add,
            ["hidden_states", f"{self.prefix}.input_layernorm_out"],
            ["hidden_states"]
        )

        self.mlp.build_graph(self.layer_graph)
        self.post_attention_layernorm.build_graph(self.layer_graph)

        # mlp residual add
        mlp_res_add = atb._BaseOperation(
            op_type="Elewise",
            op_param=json.dumps({"elewiseType": "ELEWISE_ADD"}),
            op_name="mlp_res_add",
        )
        setattr(self.layer_graph, "mlp_res_add", mlp_res_add)
        self.layer_graph.operations.append(self.layer_graph.mlp_res_add)
        self.layer_graph.add_operation(
            self.layer_graph.mlp_res_add,
            ["hidden_states", f"{self.prefix}.post_attention_layernorm_out"],
            ["layer_out"]
        )
        self.layer_graph.build()

        graph.operations.append(self.layer_graph)
        graph.add_operation(
            self.layer_graph,
            self.weight_names + self.get_intensor_names(),
            ["hidden_states"]
        )


class TransformerATB(nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        self.config = config

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.dtype = weights.dtype
        self.soc_info = NPUSocInfo()
        self.backend = (CommunicationBackend.HCCL if self.soc_info.need_nz else CommunicationBackend.LCCL)

        self.model_prefix = "transformer.vision.transformer.layers"
        layers = []
        for layer_idx in range(config.num_hidden_layers):
            layers.append(TransformerLayerATB(config, weights, self.model_prefix, layer_idx, self.backend))
        self.layers = nn.ModuleList(layers)

        self.graph = None
        self.graph_inputs = {}
        self.graph_outputs = {}
        self.graph_param = {}
        self.weight = OrderedDict()

    def get_intensor_names(self):
        return ["hidden_states", "q_seq_len", "kv_seq_len"]

    def get_outtensor_names(self):
        return ["hidden_states"]
    
    def get_weights(self):
        weights_dict = OrderedDict()
        for layer_idx, layer in enumerate(self.layers):
            weights = layer.get_weights(f"{self.model_prefix}.{layer_idx}")
            weights_dict.update(weights)
        return weights_dict

    def build_graph(self):
        self.graph.add_input_output(input=list(self.weight.keys()) + self.get_intensor_names(),
                                    output=self.get_outtensor_names())
        for layer in self.layers:
            layer.build_graph(self.graph)
        self.graph.execute_as_single = False
        self.graph.build()
        self.graph.set_weights(self.weight)

    def init_graph(self):
        self.weight = self.get_weights()
        self.graph = AtbGraph("glm_vit_atb_graph")
        self.build_graph()

    def prepare_inputs(self, hidden_states, q_seqlens, kv_seqlens):
        self.graph_inputs.update({"hidden_states": hidden_states})
        self.graph_inputs.update({"q_seq_len": q_seqlens})
        self.graph_inputs.update({"kv_seq_len": kv_seqlens})

        self.graph_param.update({"q_seq_len": q_seqlens.cpu().to(torch.int64)})
        self.graph_param.update({"kv_seq_len": kv_seqlens.cpu().to(torch.int64)})
        
        self.graph_outputs.update({"hidden_states": hidden_states})
    
    def forward(self, hidden_states, q_seqlens, kv_seqlens):
        self.prepare_inputs(hidden_states, q_seqlens, kv_seqlens)
        hidden_states = self.graph.forward(self.graph_inputs, self.graph_outputs, self.graph_param)
        return hidden_states


class GLU(nn.Module):
    def __init__(self, config, in_features):
        super().__init__()
        self.linear_proj = nn.Linear(in_features, config.hidden_size, bias=False)
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.act1 = nn.GELU()
        self.act2 = nn.functional.silu
        self.dense_h_to_4h = nn.Linear(config.hidden_size, config.ffn_hidden_size, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, config.ffn_hidden_size, bias=False)
        self.dense_4h_to_h = nn.Linear(config.ffn_hidden_size, config.hidden_size, bias=False)

    def forward(self, x):
        x = self.linear_proj(x)
        x = self.act1(self.norm1(x))
        x = self.act2(self.gate_proj(x)) * self.dense_h_to_4h(x)
        x = self.dense_4h_to_h(x)
        return x


class EVA2CLIPModelATB(nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        vision_config = Namespace(**config.vision_config)
        self.patch_embedding = PatchEmbedding(vision_config)

        # init atb transformer
        self.transformer = TransformerATB(vision_config, weights)

        self.linear_proj = GLU(config, in_features=config.hidden_size)
        self.conv = nn.Conv2d(in_channels=vision_config.hidden_size, out_channels=config.hidden_size, kernel_size=2,
                              stride=2)
        self.boi = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.eoi = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.scaling_factor = vision_config.scaling_factor

    def forward(self, images):
        x = self.patch_embedding(images)
        
        batch, seqlen, _ = x.shape
        q_seqlens = torch.tensor([seqlen] * batch, dtype=torch.int64, device=x.device)
        kv_seqlens = torch.tensor([seqlen] * batch, dtype=torch.int64, device=x.device)
        x = torch_npu.npu_format_cast(x, 2)
        x = self.transformer(x, q_seqlens, kv_seqlens)["hidden_states"]

        x = x[:, 1:]

        b, s, h = x.shape
        grid_size = int(s ** 0.5)
        x = x.view(b, grid_size, grid_size, h).permute(0, 3, 1, 2)
        x = self.conv(x)

        x = x.flatten(2).transpose(1, 2)
        x = self.linear_proj(x)

        boi = self.boi.expand(x.shape[0], -1, -1).to(x.device)
        eoi = self.eoi.expand(x.shape[0], -1, -1).to(x.device)
        x = torch.cat((boi, x, eoi), dim=1)
        x = x / self.scaling_factor
        return x