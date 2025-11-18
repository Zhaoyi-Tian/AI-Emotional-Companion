# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.

import json
from argparse import Namespace
from collections import OrderedDict, defaultdict

import torch
from torch import nn
from transformers.activations import ACT2FN
import _libatb_torch as atb

from atb_llm.models.base.flash_causal_lm_atb import AtbGraph
from atb_llm.common_op_builders.data_type import CommonOpBuilderType, NormType
from atb_llm.common_op_builders.linear_parallel.base_linear_parallel_common_op_builder import ParallelType, \
    TensorParallelInfo, CommunicationBackend
from atb_llm.utils.initial import NPUSocInfo
from atb_llm.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    load_column_multi,
)
from atb_llm.utils.layers.linear.fast_linear import FastLinear
from atb_llm.common_op_builders.common_op_builder_manager import CommonOpBuilderManager

_OP_NAME = "op_name"
_CATEGORY = "category"
_LINEAR_MODULE = "linear_module"
_LINEAR_PARAM = "linear_param"
_INPUT = "input"
_ATTEN_OUT = "atten_out"
_VIT = "VIT"


class PatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Conv2d(
            config.in_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size)
        self.cls_embedding = nn.Parameter(torch.zeros(1, config.hidden_size))
        self.position_embedding = nn.Embedding(config.num_positions, config.hidden_size)

    # "tensor(B, C, H, W)" -> "tensor(B, L, D)"
    def forward(self, images):  
        x = self.proj(images)
        x = x.flatten(2).transpose(1, 2)
        cls_token = self.cls_embedding.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x += self.position_embedding.weight.unsqueeze(0)
        return x


class Attention(nn.Module):
    def __init__(self, config, weights, layer_id, backend, prefix):
        super().__init__()
        self.weights = weights
        self.layer_id = layer_id
        self.backend = backend
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.dtype = weights.dtype
        setattr(config, 'quantize', None)
        self.quantize = config.quantize
        self.num_heads = config.num_heads
        self.num_heads_pre_rank = (self.num_heads + self.tp_world_size - 1) // self.tp_world_size
        self.prefix = f"{prefix}.{layer_id}.attention"
        head_dim = config.hidden_size // config.num_heads
        self.scale = head_dim ** -0.5
        self.head_dim = head_dim

        self.query_key_value = TensorParallelColumnLinear.load_qkv(
            config,
            prefix=f"{self.prefix}.query_key_value",
            weights=weights,
            bias=True,
            hidden_size=config.hidden_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_heads,
        )
        self.dense = TensorParallelRowLinear.load(
                config,
                prefix=f"{self.prefix}.dense",
                weights=weights,
                bias=True,
                gqa_size=self.head_dim,
        )
        self.output_dropout = torch.nn.Dropout(config.dropout_prob)

    def get_weights(self, prefix):
        weights_dict = OrderedDict()
        weights_dict.update(self.query_key_value.linear.get_weights(f"{prefix}.query_key_value"))

        weights_dict.update(self.dense.linear.get_weights(f"{prefix}.dense"))
        return weights_dict
    
    def build_query_key_value_graph(self, graph):
        
        input_key_list = [
            "hidden_states",
            f"{self.prefix}.query_key_value.weight",
            f"{self.prefix}.query_key_value.bias"]
        linear_out = ["query_key_value_linear_out"]
        linear_op = atb._BaseOperation(
            op_type="Linear",
            op_param=json.dumps({
                "transposeB": True,
                "hasBias": True}),
            op_name="query_key_value" + "_Linear"
        )
        graph.operations.append(linear_op)
        graph.add_operation(
            linear_op,
            input_key_list,
            linear_out
        )
        split_op = atb._BaseOperation(
            op_type="Split",
            op_param=json.dumps({
                "splitDim": -1,
                "splitNum": 3
            }),
            op_name="query_key_value" + "_Split"
        )
        graph.operations.append(split_op)
        graph.add_operation(
            split_op,
            ["query_key_value_linear_out"],
            ["query_split", "key_split", "value_split"],
        )
    
    def reshape_q(self, org_shape):
        self.org_shape_0 = org_shape[0]
        self.org_shape_1 = org_shape[1]
        return [org_shape[0] * org_shape[1], self.num_heads_pre_rank, self.head_dim]
    
    def reshape_kv(self, org_shape):
        return [org_shape[0] * org_shape[1], self.num_heads_pre_rank, self.head_dim]
    
    def reshape_out(self, org_shape):
        return [self.org_shape_0, self.org_shape_1, org_shape[1] * org_shape[2]]

    def build_attention_graph(self, graph):
        attention_op = atb._BaseOperation(
            op_type="SelfAttention",
            op_param=json.dumps({
                "headNum": self.num_heads_pre_rank,
                "kvHeadNum": self.num_heads_pre_rank,
                "qkScale": self.scale,
                "calcType":"PA_ENCODER"}),
            op_name="selfattention"
        )

        graph.add_reshape("query_split", "query_split_reshape", self.reshape_q)
        graph.add_reshape("key_split", "key_split_reshape", self.reshape_kv)
        graph.add_reshape("value_split", "value_split_reshape", self.reshape_kv)

        graph.operations.append(attention_op)
        input_key_list = ["query_split_reshape", "key_split_reshape", "value_split_reshape", "seq_len"]
        output_key_list = [_ATTEN_OUT]
        graph.add_operation(attention_op, input_key_list, output_key_list)
        graph.add_reshape(_ATTEN_OUT, _ATTEN_OUT, self.reshape_out)

    def build_dense_graph(self, graph):
        dense_linear_param = {
            "op_name": "dense_linear",
            "category": CommonOpBuilderType.LINEAR,
            "linear_module": self.dense.linear,
            "enable_quant_input": False,
            "default_dtype": self.dtype,
            "group_size":0
        }
        dense_linear_tensor_map = {
            "input": 'atten_out',
            "linear_out": 'dense_out'
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
        linear_parallel_builder = CommonOpBuilderManager.get_builder(dense_linear_parallel_param)
        graph = linear_parallel_builder.build(graph, dense_linear_tensor_map)
    
    
    def build_graph(self, graph):
        self.build_query_key_value_graph(graph)
        self.build_attention_graph(graph)
        self.build_dense_graph(graph)


class MLP(nn.Module):
    def __init__(self, config, weights, layer_id=0, backend=None, prefix=None):
        super().__init__()
        self.config = config
        self.weights = weights
        self.dtype = weights.dtype
        self.layer_id = layer_id
        self.backend = backend
        self.prefix = f"{prefix}.{layer_id}.mlp" 
        setattr(config, 'quantize', None)
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = TensorParallelColumnLinear.load(config,
                    prefix=f"{self.prefix}.fc1",
                    weights=weights,
                    bias=True,)
        self.fc2 = TensorParallelRowLinear.load(config,
                    prefix=f"{self.prefix}.fc2",
                    weights=weights,
                    bias=True,)

    def get_weights(self, prefix):
        weights_dict = OrderedDict()
        weights_dict.update(self.fc1.linear.get_weights(f"{prefix}.fc1"))
        weights_dict.update(self.fc2.linear.get_weights(f"{prefix}.fc2"))
        return weights_dict
    
    def build_activation_graph(self, graph):
        act = atb._BaseOperation(
            op_type="Activation",
            op_param=json.dumps({'activationType': 'ACTIVATION_GELU', "geluMode": "TANH_MODE"}),
            op_name="Activation_gelu",
        )
        swish_input_list = ["fc1_out"]
        swish_output_list = ["activation_out"]
        graph.operations.append(act)
        graph.add_operation(
            act,
            swish_input_list,
            swish_output_list,
        )

    def build_fc1_graph(self, graph):
        input_key_list = ["add_attention_out", 
                          f"{self.prefix}.fc1.weight",
                          f"{self.prefix}.fc1.bias"]
        linear_out = ["fc1_out"]
        linear_op = atb._BaseOperation(
            op_type="Linear",
            op_param=json.dumps({
                "transposeB": True,
                "hasBias": True}),
            op_name="fc1" + "_Linear"
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
            "group_size": 0
        }
        fc2_linear_tensor_map = {
            "input": 'activation_out',
            "linear_out": 'fc2_out'
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
        linear_parallel_builder = CommonOpBuilderManager.get_builder(fc2_linear_parallel_param)
        graph = linear_parallel_builder.build(graph, fc2_linear_tensor_map) 
    
    def build_graph(self, graph):
        self.build_fc1_graph(graph)
        self.build_activation_graph(graph)
        self.build_fc2_graph(graph)
    

class TransformerLayer(nn.Module):
    def __init__(self, config, weights, layer_id, prefix):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.weights = weights
        self.soc_info = NPUSocInfo()
        self.backend = CommunicationBackend.HCCL if self.soc_info.need_nz else CommunicationBackend.LCCL
        self.attention = Attention(config, weights, layer_id, self.backend, prefix)
        self.mlp = MLP(config, weights, layer_id, self.backend, prefix)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layer_norm_eps = config.layer_norm_eps
       
        self.layer_id = layer_id
        self.prefix = prefix


    def get_in_tensor_names(self):
        return ["hidden_states", "seq_len"]
    
    def get_weights(self, prefix):
        weights_dict = OrderedDict()
        for name, module in self.named_children():
            if name == "mlp" or name == "attention":
                weights_dict.update(module.get_weights(f"{prefix}.{self.layer_id}.{name}"))
            if name == "input_layernorm" or name == "post_attention_layernorm" :
                weights_dict[f"{prefix}.{self.layer_id}.{name}.weight"] = \
                self.weights.get_tensor(f"{prefix}.{self.layer_id}.{name}.weight").npu()
                weights_dict[f"{prefix}.{self.layer_id}.{name}.bias"] = \
                self.weights.get_tensor(f"{prefix}.{self.layer_id}.{name}.bias").npu()
        self.weight_names = list(weights_dict.keys())
        return weights_dict
    
    def build_input_norm_graph(self, graph):
        norm_param = {
            "layerType": "LAYER_NORM_NORM",
            "normParam": {
                "quantType": "QUANT_UNDEFINED",
                "epsilon": self.layer_norm_eps,
                "beginParamsAxis":2,
                "beginNormAxis":2
            },
        }
        input_norm_op = atb._BaseOperation(
            op_type=NormType.LAYERNORM,
            op_param=json.dumps(norm_param),
            op_name="input_norm",
        )
        graph.operations.append(input_norm_op)
        graph.add_operation(
            input_norm_op, 
            ["dense_out",
             f"{self.prefix}.{self.layer_id}.input_layernorm.weight",
             f"{self.prefix}.{self.layer_id}.input_layernorm.bias"], 
             ["input_layernorm_out"]
             )
    
    def build_post_attention_norm_graph(self, graph):
        norm_param = {
            "layerType": "LAYER_NORM_NORM",
            "normParam": {
                "quantType": "QUANT_UNDEFINED",
                "epsilon": self.layer_norm_eps,
                "beginParamsAxis":2,
                "beginNormAxis":2
            },
        }
        input_norm_op = atb._BaseOperation(
            op_type=NormType.LAYERNORM,
            op_param=json.dumps(norm_param),
            op_name="post_norm",
        )
        graph.operations.append(input_norm_op)
        graph.add_operation(
            input_norm_op, 
            ["fc2_out",
             f"{self.prefix}.{self.layer_id}.post_attention_layernorm.weight",
             f"{self.prefix}.{self.layer_id}.post_attention_layernorm.bias"],
             ["post_attention_norm__out"]
        )
    
    def build_addattention_graph(self, graph):
        add_attention_op = atb._BaseOperation(
            op_type="Elewise",
            op_param=json.dumps({"elewiseType": "ELEWISE_ADD"}),
            op_name="add_attention",
        )
        graph.operations.append(add_attention_op)
        graph.add_operation(
            add_attention_op, ["input_layernorm_out", "hidden_states"], ["add_attention_out"]
        )
    
    def build_addmlp_graph(self, graph):
        add_mlp_op = atb._BaseOperation(
            op_type="Elewise",
            op_param=json.dumps({"elewiseType": "ELEWISE_ADD"}),
            op_name="add_1",
        )
        graph.operations.append(add_mlp_op)
        graph.add_operation(
            add_mlp_op, ["post_attention_norm__out", "add_attention_out"], ["layer_out"]
        )

    def build_graph(self, graph):
        self.layer_graph = AtbGraph("transformer" + f"_layer_{self.layer_id}_graph")
        self.layer_graph.add_input_output(
            input=list(self.weight_names) + self.get_in_tensor_names(), output=["layer_out"])
        self.attention.build_graph(self.layer_graph)
        self.build_input_norm_graph(self.layer_graph)
        self.build_addattention_graph(self.layer_graph)
        self.mlp.build_graph(self.layer_graph)
        self.build_post_attention_norm_graph(self.layer_graph)
        self.build_addmlp_graph(self.layer_graph)
        graph.operations.append(self.layer_graph)
        self.layer_graph.build()
        graph.add_operation(
            self.layer_graph,
            list(self.weight_names) + self.get_in_tensor_names(),
            ["hidden_states"]
        )


class Transformer(nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        self.graph = None
        self.config = config
        self.graph_inputs = defaultdict(dict)
        self.graph_outputs = defaultdict(dict)
        self.graph_param = defaultdict(dict)
        self.weight = OrderedDict()
        self.model_prefix = "model.vision.transformer.layers"
        self.layers = nn.ModuleList([TransformerLayer(config, weights, idx, self.model_prefix) 
                                     for idx in range(config.num_hidden_layers)])

    def init_graph(self):
        self.weight = self.get_weights()
        self.graph = AtbGraph("evaclip_graph")
        self.build_graph()
    
    def build_mul_graph(self, graph):
        ls1_op = atb._BaseOperation(
            op_type="Elewise",
            op_param=json.dumps({"elewiseType": "ELEWISE_MUL"}),
            op_name="mul",
        )
        graph.operations.append(ls1_op)
        graph.add_operation(
            ls1_op, ["input_embeds", "one_tensor"], ["hidden_states"]
        )
    
    def build_graph(self):
        self.graph.add_input_output(
            input=list(self.weight.keys()) + self.get_in_tensor_names(), output=self.get_out_tensor_names())
        self.build_mul_graph(self.graph)
        for layer in self.layers:
            layer.build_graph(self.graph)
        self.graph.execute_as_single = False
        self.graph.build()
        self.graph.set_weights(self.weight)

    def get_in_tensor_names(self):
        return ['input_embeds', "one_tensor", "seq_len"]
    
    def get_out_tensor_names(self):
        return ['hidden_states']
    
    def get_weights(self):
        weights_dict = OrderedDict()
        for layer in self.layers:
            weights = layer.get_weights(self.model_prefix)
            weights_dict.update(weights)
        return weights_dict
    
    def prepare_inputs(self, inputs_embeds):
        context_length = torch.tensor(inputs_embeds.size(0) * [inputs_embeds.size(1)], dtype=torch.int32).npu()
        one_tensor = torch.tensor([1.0]).to(inputs_embeds.dtype).npu()
        self.graph_inputs[_VIT].update({"input_embeds":inputs_embeds})
        self.graph_inputs[_VIT].update({"one_tensor":one_tensor})
        self.graph_inputs[_VIT].update({"seq_len":context_length})
        self.graph_param[_VIT]["seq_len"] = context_length.cpu().to(torch.int32)
        inputs_embeds_shape = inputs_embeds.shape
        batch = inputs_embeds_shape[0]
        seq_len = inputs_embeds_shape[1]
        hidden_size = self.config.hidden_size
        self.graph_outputs[_VIT][self.get_out_tensor_names()[0]] = (
            torch.ones(batch, seq_len, hidden_size).to(inputs_embeds.dtype).npu())
    
    def forward(self, input_embedds):
        self.prepare_inputs(input_embedds)
        hidden_states = self.graph.forward(self.graph_inputs[_VIT],
                                           self.graph_outputs[_VIT],
                                           self.graph_param[_VIT])
        return hidden_states


class GLU(nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        self.graph_inputs = defaultdict(dict)
        self.graph_outputs = defaultdict(dict)
        self.graph_param = defaultdict(dict)
        self.weights = weights
        self.config = config
        setattr(config, 'quantize', None)
        self.quantize = config.quantize
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.prefix = "model.vision.linear_proj"
        self.dtype = weights.dtype
        self.linear_proj = FastLinear.load(prefix=f"{self.prefix}.linear_proj",
                                           weights=weights,
                                           bias=False)
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.act1 = nn.GELU()
        self.act2 = nn.functional.silu
        self.layer_norm_eps = 1e-5
        self.dense_h_to_4h_gate = load_column_multi(config, 
                                                    prefixes=[
                                                        f"{self.prefix}.gate_proj",
                                                        f"{self.prefix}.dense_h_to_4h",
                                                        ],
                                                    weights=weights,
                                                    head_size=1,)
        self.dense_4h_to_h = TensorParallelRowLinear.load(config,
                    prefix=f"{self.prefix}.dense_4h_to_h",
                    weights=weights,
                    bias=False,)
        self.graph = AtbGraph("glu_graph")

    def get_in_tensor_names(self):
        return ["hidden_states", "one_tensor"]
    
    def get_out_tensor_names(self):
        return ['dense_4h_to_h_out']
    
    def prepare_inputs(self, hidden_states):
        one_tensor = torch.tensor([1.0]).to(hidden_states.dtype).npu()
        self.graph_inputs[_VIT].update({"hidden_states":hidden_states})
        self.graph_inputs[_VIT].update({"one_tensor":one_tensor})
        hidden_states_shape = hidden_states.shape
        batch = hidden_states_shape[0]
        seq_len = hidden_states_shape[1]
        hidden_size = self.config.hidden_size
        self.graph_outputs[_VIT][self.get_out_tensor_names()[0]] = \
            torch.ones(batch, seq_len, hidden_size).to(hidden_states.dtype).npu()

    def get_weights(self, prefix):
        weights_dict = OrderedDict()
        weights_dict.update(
            self.linear_proj.get_weights(f"{self.prefix}.linear_proj")
        )
        weights_dict.update(
            self.dense_h_to_4h_gate.linear.get_weights(f"{self.prefix}.dense_h_to_4h_gate")
        )
        weights_dict.update(
            self.dense_4h_to_h.linear.get_weights(f"{self.prefix}.dense_4h_to_h")
        )
        for name, _ in self.named_children():
            if name == "norm1":
                weights_dict[f"{prefix}.{name}.weight"] = self.weights.get_tensor(f"{prefix}.{name}.weight").npu()
                weights_dict[f"{prefix}.{name}.bias"] = self.weights.get_tensor(f"{prefix}.{name}.bias").npu()
        self.weight_names = list(weights_dict.keys())
        return weights_dict

    def init_graph(self):
        self.weight = self.get_weights(self.prefix)
        self.graph = AtbGraph("glu_graph")
        self.build_graph()

    def build_norm_graph(self, graph):
        norm_param = {
            "layerType": "LAYER_NORM_NORM",
            "normParam": {
                "quantType": "QUANT_UNDEFINED",
                "epsilon": self.layer_norm_eps,
                "beginParamsAxis":2,
                "beginNormAxis":2
            },
        }
        input_norm_op = atb._BaseOperation(
            op_type=NormType.LAYERNORM,
            op_param=json.dumps(norm_param),
            op_name="input_norm",
        )
        graph.operations.append(input_norm_op)
        graph.add_operation(
            input_norm_op, 
            ["linear_proj_out",
             f"{self.prefix}.norm1.weight",
             f"{self.prefix}.norm1.bias"], 
             ["norm1_out"]
             )

    def build_act1_graph(self, graph):
        act = atb._BaseOperation(
            op_type="Activation",
            op_param=json.dumps({'activationType': 'ACTIVATION_GELU', "geluMode":"TANH_MODE"}),
            op_name="act1",
        )
        swish_input_list = ["norm1_out"]
        swish_output_list = ["act1_out"]
        graph.operations.append(act)
        graph.add_operation(
            act,
            swish_input_list,
            swish_output_list,
        )
    
    def build_act2_graph(self, graph):
        act = atb._BaseOperation(
            op_type="Activation",
            op_param=json.dumps({'activationType': 'ACTIVATION_SWISH'}),
            op_name="act2",
        )
        swish_input_list = ["gate_out"]
        swish_output_list = ["act2_out"]
        graph.operations.append(act)
        graph.add_operation(
            act,
            swish_input_list,
            swish_output_list,
        )

    def build_linear_proj_graph(self, graph):
        input_key_list = ["hidden_states_new", f"{self.prefix}.linear_proj.weight"]
        linear_out = ["linear_proj_out"]
        linear_op = atb._BaseOperation(
            op_type="Linear",
            op_param=json.dumps({
                "transposeB": True,
                "hasBias": False}),
            op_name="linear_proj" + "_Linear"
        )
        graph.operations.append(linear_op)
        graph.add_operation(
            linear_op,
            input_key_list,
            linear_out
        )
        

    def build_dense_h_to_4h_gate_graph(self, graph):
        linear_param = {
            _OP_NAME: "dense_h_to_4h_gate_linear",
            _CATEGORY: CommonOpBuilderType.LINEAR,
            "enable_quant_input": False,
            "default_dtype": self.dtype,
            "group_size": 0,
            _LINEAR_MODULE: (
                self.dense_h_to_4h_gate.linear
            ),
        }
        gate_up_linear_param = {
            _OP_NAME: "dense_h_to_4h_gate_linear",
            _CATEGORY: CommonOpBuilderType.GATE_UP,
            "is_pack": True,
            _LINEAR_PARAM: linear_param,
        }
        gate_up_linear_tensor_map = {
            _INPUT: "act1_out",
            "gate_up_out": "dense_h_to_4h_gate_out",
        }
        builder = CommonOpBuilderManager.get_builder(gate_up_linear_param)
        graph = builder.build(graph, gate_up_linear_tensor_map)

    def build_mul_one_graph(self, graph):
        ls1_op = atb._BaseOperation(
            op_type="Elewise",
            op_param=json.dumps({"elewiseType": "ELEWISE_MUL"}),
            op_name="mul",
        )
        graph.operations.append(ls1_op)
        graph.add_operation(
            ls1_op, ["hidden_states", "one_tensor"], ["hidden_states_new"]
        )
    
    def build_mul_graph(self, graph):
        mul_op = atb._BaseOperation(
            op_type="Elewise",
            op_param=json.dumps({"elewiseType": "ELEWISE_MUL"}),
            op_name="mul",
        )
        graph.operations.append(mul_op)
        graph.add_operation(
            mul_op, ["act2_out", "dense_h_to_4h_out"], ["mul_out"]
        )

    def build_dense_4h_to_h(self, graph):
        dense_4h_to_h_linear_param = {
            _OP_NAME: "dense_4h_to_h_linear",
            _CATEGORY: CommonOpBuilderType.LINEAR,
            _LINEAR_MODULE: (
                self.dense_4h_to_h.linear
            ),
            "enable_quant_input": False,
            "default_dtype": self.dtype,
            "group_size": 0,
        }
        dense_4h_to_h_linear_tensor_map = {
            _INPUT: "mul_out",
            "linear_out": "dense_4h_to_h_out",
        }

        dense_4h_to_h_linear_parallel_param = {
            _OP_NAME: "down_linear_parallel",
            _CATEGORY: CommonOpBuilderType.LINEAR_PARALLEL,
            "parallel_type": ParallelType.ALL_REDUCE,
            "parallel_info": TensorParallelInfo(
                rank=self.tp_rank, world_size=self.tp_world_size, backend=CommunicationBackend.LCCL
            ),
            _LINEAR_PARAM: dense_4h_to_h_linear_param,
            "enable_lcoc": False,
        }
        linear_parallel_builder = CommonOpBuilderManager.get_builder(
            dense_4h_to_h_linear_parallel_param
        )
        graph = linear_parallel_builder.build(graph, dense_4h_to_h_linear_tensor_map)

    def build_split_graph(self, graph):
        split_op = atb._BaseOperation(
                op_type="Split",
                op_param=json.dumps({
                    "splitDim": -1,
                    "splitNum": 2
                }),
                op_name="dense_h_to_4h_gate_split"
            )
        graph.operations.append(split_op)
        graph.add_operation(
                split_op,
                ["dense_h_to_4h_gate_out"],
                ["gate_out", "dense_h_to_4h_out"],
            )

    def build_graph(self):
        self.graph.add_input_output(
            input=list(self.weight.keys()) + self.get_in_tensor_names(),
            output=self.get_out_tensor_names()
        )
        self.build_mul_one_graph(self.graph)
        self.build_linear_proj_graph(self.graph)
        self.build_norm_graph(self.graph)
        self.build_act1_graph(self.graph)
        self.build_dense_h_to_4h_gate_graph(self.graph)
        self.build_split_graph(self.graph)
        self.build_act2_graph(self.graph)
        self.build_mul_graph(self.graph)
        self.build_dense_4h_to_h(self.graph)
        self.graph.execute_as_single = True
        self.graph.build()
        self.graph.set_weights(self.weight)

    def forward(self, x):
        self.prepare_inputs(x)
        hidden_states = self.graph.forward(self.graph_inputs[_VIT],
                                           self.graph_outputs[_VIT],
                                           self.graph_param[_VIT])
        return hidden_states


class EVA2CLIPModel(nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        self.weights = weights
        vision_config = Namespace(**config.vision_config)
        self.patch_embedding = PatchEmbedding(vision_config)
        self.transformer = Transformer(vision_config, weights)
        self.linear_proj = GLU(config, weights)
        self.conv = nn.Conv2d(
            in_channels=vision_config.hidden_size,
            out_channels=vision_config.hidden_size,
            kernel_size=2,
            stride=2)
        self.boi = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.eoi = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.init_boi_eoi_weight()
        self.init_conv_weight()
        self.init_patch_embedding_weight()

    def init_patch_embedding_weight(self):
        patch_embedding_weights = self.patch_embedding.state_dict().keys()
        for patch_embedding_weight in patch_embedding_weights:
            saved_weight = torch.nn.Parameter(
                    self.weights.get_tensor(f"model.vision.patch_embedding.{patch_embedding_weight}"),
                    requires_grad=False
                )
            patch_embedding_weight_list = patch_embedding_weight.split(".")
            target_module = self.patch_embedding
            for nxt_module in patch_embedding_weight_list[:-1]:
                target_module = getattr(target_module, nxt_module)
            setattr(target_module, patch_embedding_weight_list[-1], saved_weight)

    def init_conv_weight(self):
        conv_weights = self.conv.state_dict().keys()
        for conv_weight in conv_weights:
            saved_weight = torch.nn.Parameter(
                    self.weights.get_tensor(f"model.vision.conv.{conv_weight}"),
                    requires_grad=False
                )
            conv_weight_list = conv_weight.split(".")
            target_module = self.conv
            for nxt_module in conv_weight_list[:-1]:
                target_module = getattr(target_module, nxt_module)
            setattr(target_module, conv_weight_list[-1], saved_weight)

    def init_boi_eoi_weight(self):
        boi_saved_weight = torch.nn.Parameter(
                    self.weights.get_tensor("model.vision.boi"),
                    requires_grad=False
                )
        setattr(self, "boi", boi_saved_weight)
        eoi_saved_weight = torch.nn.Parameter(
                    self.weights.get_tensor("model.vision.eoi"),
                    requires_grad=False
                )
        setattr(self, "eoi", eoi_saved_weight)

    # "tensor(B, C, H, W)" -> "tensor(B, L, D)"
    def forward(self, images):
        x = self.patch_embedding(images)
        x = self.transformer(x)["hidden_states"]
        x = x[:, 1:]
        b, s, h = x.shape
        grid_size = int(s**0.5)
        x = x.view(b, grid_size, grid_size, h).permute(0, 3, 1, 2)
        x = self.conv(x)

        x = x.flatten(2).transpose(1, 2)
        x = self.linear_proj(x)["dense_4h_to_h_out"]
        boi = self.boi.expand(x.shape[0], -1, -1)
        eoi = self.eoi.expand(x.shape[0], -1, -1)
        x = torch.cat((boi, x, eoi), dim=1)
        return x
