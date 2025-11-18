# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.

from collections import OrderedDict
import json
import math

from torch import nn

import _libatb_torch as atb
from atb_llm.common_op_builders.data_type import (
    ActivationType,
    CommonOpBuilderType,
    NormType,
    OperationBackend,
)
from atb_llm.common_op_builders.common_op_builder_manager import CommonOpBuilderManager
from atb_llm.common_op_builders.linear_parallel.base_linear_parallel_common_op_builder import (
    ParallelType,
    TensorParallelInfo,
    CommunicationBackend,
)
from atb_llm.common_op_builders.attention.base_attention_common_op_builder import (
    AttnType,
)
from atb_llm.models.base.flash_causal_lm_atb import AtbGraph
from atb_llm.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    load_column_multi,
)

_EMBEDDING_PARALLEL_THRESHOLD = 128256
_OFFSETS_KEY = "offsets"
_SIZE_KEY = "size"
_LINEAR_MODULE = "linear_module"
_INPUT = "input"
_LINEAR_OUT = "linear_out"
_LINEAR_PARAM = "linear_param"
_OP_NAME = "op_name"
_CATEGORY = "category"
_SLICE = "Slice"
_LANGUAGE_INDICES = "language_indices"
_VISION_INDICES = "vision_indices"


class RMSNorm(nn.Module):
    def __init__(self, config, weights, prefix):
        super().__init__()
        self.rms_norm_eps = config.rms_norm_eps
        self.prefix = prefix
        weight = weights.get_tensor(f"{prefix}.weight")
        self.weight = nn.Parameter(weight)

    def get_weights(self, prefix):
        weights_dict = OrderedDict()
        weights_dict[f"{prefix}.weight"] = self.weight.data
        return weights_dict

    def build_graph(self, graph):
        norm_param = {
            "layerType": "RMS_NORM_NORM",
            "normParam": {
                "quantType": "QUANT_UNDEFINED",
                "epsilon": self.rms_norm_eps,
            },
        }
        norm_op = atb._BaseOperation(
            op_type=NormType.RMSNORM,
            op_param=json.dumps(norm_param),
            op_name="norm",
        )
        graph.operations.append(norm_op)
        graph.add_operation(
            norm_op, ["hidden_states", f"{self.prefix}.weight"], [f"{self.prefix}_out"]
        )


def get_concate_op(index, op_name):
    concate_param = {"concatDim": index}
    concate_op = atb._BaseOperation(
        op_type="Concat", op_param=json.dumps(concate_param), op_name=op_name
    )
    return concate_op


def vision_expert_pre_process(
    graph, input_tensor_name, output_language_tensor_name, output_vision_tensor_name
):
    language_index_select = atb._BaseOperation(
        op_type="IndexSelect", op_param=json.dumps({}), op_name="Indexselect0"
    )

    vision_index_select = atb._BaseOperation(
        op_type="IndexSelect", op_param=json.dumps({}), op_name="Indexselect1"
    )

    graph.add_operation(
        language_index_select,
        [input_tensor_name, _LANGUAGE_INDICES],
        [output_language_tensor_name],
    )

    graph.add_operation(
        vision_index_select,
        [input_tensor_name, _VISION_INDICES],
        [output_vision_tensor_name],
    )


def vision_expert_post_process(
    graph, input_language_tensor_name, input_vision_tensor_name, output_tensor_name
):
    langugage_index_put = atb._BaseOperation(
        op_type="Indexput",
        op_param=json.dumps({}),
        op_name="Indexput0",
    )

    vision_index_put = atb._BaseOperation(
        op_type="Indexput",
        op_param=json.dumps({}),
        op_name="Indexput1",
    )

    concate_op = get_concate_op(0, "Concate0")
    graph.add_operation(
        concate_op,
        [input_vision_tensor_name, input_language_tensor_name],
        [output_tensor_name],
    )

    graph.add_operation(
        langugage_index_put,
        [output_tensor_name, _LANGUAGE_INDICES, input_language_tensor_name],
        [output_tensor_name],
    )

    graph.add_operation(
        vision_index_put,
        [output_tensor_name, _VISION_INDICES, input_vision_tensor_name],
        [output_tensor_name],
    )


class Attention(nn.Module):
    def __init__(self, config, weights, prefix, norm_prefix, backend, with_vision_expert=False):
        super().__init__()
        self.norm_prefix = norm_prefix
        self.backend = backend
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.dtype = weights.dtype
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_heads
        self.with_vision_expert = with_vision_expert
        if self.num_key_value_heads < self.tp_world_size:
            repeat_times = self.tp_world_size // self.num_key_value_heads
        else:
            repeat_times = 1
        self.num_heads_pre_rank = (
            self.num_heads + self.tp_world_size - 1
        ) // self.tp_world_size
        self.num_key_value_heads_per_rank = (
            self.num_key_value_heads * repeat_times + self.tp_world_size - 1
        ) // self.tp_world_size

        self.language_expert_query_key_value = TensorParallelColumnLinear.load_qkv(
            config,
            prefix=f"{prefix}.language_expert_query_key_value",
            weights=weights,
            bias=False,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_key_value_heads,
        )
        self.language_expert_dense = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.language_expert_dense",
            weights=weights,
            bias=False,
            gqa_size=self.head_size,
        )
        if self.with_vision_expert:
            self.vision_expert_query_key_value = TensorParallelColumnLinear.load_qkv(
                config,
                prefix=f"{prefix}.vision_expert_query_key_value",
                weights=weights,
                bias=True,
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                num_kv_heads=self.num_key_value_heads,
            )
            self.vision_expert_dense = TensorParallelRowLinear.load(
                config,
                prefix=f"{prefix}.vision_expert_dense",
                weights=weights,
                bias=False,
                gqa_size=self.head_size,
            )

    def get_weights(self, prefix):
        weights_dict = OrderedDict()
        weights_dict.update(
            self.language_expert_query_key_value.linear.get_weights(
                f"{prefix}.language_expert_query_key_value"
            )
        )
        weights_dict.update(
            self.language_expert_dense.linear.get_weights(
                f"{prefix}.language_expert_dense"
            )
        )

        if self.with_vision_expert:
            weights_dict.update(
                self.vision_expert_query_key_value.linear.get_weights(
                    f"{prefix}.vision_expert_query_key_value"
                )
            )

            weights_dict.update(
                self.vision_expert_dense.linear.get_weights(f"{prefix}.vision_expert_dense")
            )
        return weights_dict

    def slice_qkv(self, graph, linear_out_name):
        linear_out_kv_name = "slice_kv"
        # add slice q
        slice_q_param = {}
        slice_q_param[_OFFSETS_KEY] = [0, 0]
        slice_q_param[_SIZE_KEY] = [-1, self.num_heads_pre_rank * self.head_size]
        slice_q_op = atb._BaseOperation(
            op_type=_SLICE,
            op_param=json.dumps(slice_q_param),
            op_name="Slice_Q",
        )
        graph.operations.append(slice_q_op)
        graph.add_operation(
            slice_q_op,
            [linear_out_name],
            ["intermediate_q"],
        )
        # add slice kv
        slice_kv_param = {}
        slice_kv_param[_OFFSETS_KEY] = [0, self.num_heads_pre_rank * self.head_size]
        slice_kv_param[_SIZE_KEY] = [
            -1,
            self.num_key_value_heads_per_rank * self.head_size * 2,
        ]
        slice_kv_op = atb._BaseOperation(
            op_type=_SLICE,
            op_param=json.dumps(slice_kv_param),
            op_name="Slice_KV",
        )
        graph.operations.append(slice_kv_op)
        graph.add_operation(
            slice_kv_op,
            [linear_out_name],
            [linear_out_kv_name],
        )
        # add split kv
        split_op = atb._BaseOperation(
            op_type="Split",
            op_param=json.dumps({"splitDim": -1, "splitNum": 2}),
            op_name="Split_KV",
        )
        graph.operations.append(split_op)
        graph.add_operation(
            split_op,
            [linear_out_kv_name],
            ["intermediate_k", "intermediate_v"],
        )

    def build_qkv_graph(self, graph, with_vision_expert):
        norm_out_name = f"{self.norm_prefix}_out"
        language_tensor_name = f"{norm_out_name}_language"
        vision_tensor_name = f"{norm_out_name}_vision"
        linear_out_name = "qkv_intermediate_mixed_qkv"

        qkv_param = {
            _OP_NAME: "qkv_proj",
            _CATEGORY: CommonOpBuilderType.LINEAR,
            "enable_quant_input": False,
            "default_dtype": self.dtype,
            "group_size": 0,
            _LINEAR_MODULE: None,
        }
        if with_vision_expert:
            vision_expert_pre_process(
                graph, norm_out_name, language_tensor_name, vision_tensor_name
            )
            qkv_param[_LINEAR_MODULE] = self.language_expert_query_key_value.linear
            language_linear_builder = CommonOpBuilderManager.get_builder(qkv_param)
            language_linear_tensor_map = {
                _INPUT: language_tensor_name,
                _LINEAR_OUT: f"{language_tensor_name}_out",
            }
            graph = language_linear_builder.build(graph, language_linear_tensor_map)
            qkv_param[_LINEAR_MODULE] = self.vision_expert_query_key_value.linear
            vision_linear_builder = CommonOpBuilderManager.get_builder(qkv_param)
            vision_linear_tensor_map = {
                _INPUT: vision_tensor_name,
                _LINEAR_OUT: f"{vision_tensor_name}_out",
            }
            graph = vision_linear_builder.build(graph, vision_linear_tensor_map)
            vision_expert_post_process(
                graph,
                f"{language_tensor_name}_out",
                f"{vision_tensor_name}_out",
                linear_out_name,
            )
        else:
            qkv_param[_LINEAR_MODULE] = self.language_expert_query_key_value.linear
            qkv_linear_tensor_map = {
                _INPUT: norm_out_name,
                _LINEAR_OUT: linear_out_name,
            }
            qkv_linear_builder = CommonOpBuilderManager.get_builder(qkv_param)
            graph = qkv_linear_builder.build(graph, qkv_linear_tensor_map)

        self.slice_qkv(graph, linear_out_name)

    def build_rope_graph(self, graph):
        rope_param = {
            _OP_NAME: "rope",
            "head_num": self.num_heads_pre_rank,
            "kv_head_num": self.num_key_value_heads_per_rank,
            _CATEGORY: CommonOpBuilderType.ROPE,
            "is_fa": False,
            "atb_rope_param": {"rotaryCoeff": 2},
        }
        rope_tensor_map = {
            "q": "intermediate_q",
            "k": "intermediate_k",
            "cos_embedding": "cos_embedding",
            "sin_embedding": "sin_embedding",
            "seq_len": "seq_len",
            "q_out": "intermediate_q",
            "k_out": "intermediate_k",
        }
        rope_builder = CommonOpBuilderManager.get_builder(rope_param)
        graph = rope_builder.build(graph, rope_tensor_map)

    def get_attention_tensor_map(self):
        attention_tensor_map = {
            "q": "intermediate_q",
            "k": "intermediate_k",
            "v": "intermediate_v",
            "k_cache": "k_cache",
            "v_cache": "v_cache",
            "attention_mask": "attention_mask",
            "seq_len": "seq_len",
            "attention_out": "attn_out",
            "slots": "slots_mapping",
            "block_tables": "block_tables",
        }
        return attention_tensor_map

    def get_atb_attention_param(self, is_prefill):
        atb_attention_param = {
            "headNum": self.num_heads_pre_rank,
            "kvHeadNum": self.num_key_value_heads_per_rank,
            "qkScale": 1.0 / math.sqrt(self.head_size),
        }
        if is_prefill:
            atb_attention_param.update(
                {
                    "maskType": "MASK_TYPE_NORM",
                    "calcType": "PA_ENCODER",
                    "isTriuMask": 1,
                }
            )
        return atb_attention_param

    def build_attention_graph(self, graph, is_prefill):
        attention_param = {
            _OP_NAME: "attention",
            _CATEGORY: CommonOpBuilderType.ATTENTION,
            "is_prefill": is_prefill,
            "attn_type": AttnType.PAGED_ATTENTION,
            "head_size": self.head_size,
            "atb_reshape_and_cache_param": {},
            "operation_backend": OperationBackend.ATB,
            "atb_attention_param": self.get_atb_attention_param(is_prefill),
        }
        attention_tensor_map = self.get_attention_tensor_map()

        pa_attention_builder = CommonOpBuilderManager.get_builder(attention_param)
        graph = pa_attention_builder.build(graph, attention_tensor_map)

    def build_dense_graph(self, graph, is_prefill, with_vision_expert):
        attn_out_name = "attn_out"
        language_tensor_name = f"{attn_out_name}_language"
        vision_tensor_name = f"{attn_out_name}_vision"
        dense_out_name = "dense_out"

        dense_param = {
            _OP_NAME: "dense_linear",
            _CATEGORY: CommonOpBuilderType.LINEAR,
            _LINEAR_MODULE: None,
            "enable_quant_input": True,
            "default_dtype": self.dtype,
            "group_size": 0,
        }
        dense_parallel_param = {
            _OP_NAME: "dense_parallel_linear",
            _CATEGORY: CommonOpBuilderType.LINEAR_PARALLEL,
            "parallel_type": ParallelType.ALL_REDUCE,
            "parallel_info": TensorParallelInfo(
                rank=self.tp_rank, world_size=self.tp_world_size, backend=self.backend
            ),
            _LINEAR_PARAM: dense_param,
            "enable_lcoc": is_prefill,
        }
        if with_vision_expert:
            vision_expert_pre_process(
                graph, attn_out_name, language_tensor_name, vision_tensor_name
            )
            dense_parallel_param[_LINEAR_PARAM][
                _LINEAR_MODULE
            ] = self.language_expert_dense.linear
            language_dense_builder = CommonOpBuilderManager.get_builder(
                dense_parallel_param
            )
            language_dense_tensor_map = {
                _INPUT: language_tensor_name,
                _LINEAR_OUT: f"{language_tensor_name}_out",
            }
            graph = language_dense_builder.build(graph, language_dense_tensor_map)
            dense_parallel_param[_LINEAR_PARAM][
                _LINEAR_MODULE
            ] = self.vision_expert_dense.linear
            vision_dense_builder = CommonOpBuilderManager.get_builder(
                dense_parallel_param
            )
            vision_dense_tensor_map = {
                _INPUT: vision_tensor_name,
                _LINEAR_OUT: f"{vision_tensor_name}_out",
            }
            graph = vision_dense_builder.build(graph, vision_dense_tensor_map)
            vision_expert_post_process(
                graph,
                f"{language_tensor_name}_out",
                f"{vision_tensor_name}_out",
                dense_out_name,
            )
        else:
            dense_parallel_param[_LINEAR_PARAM][
                _LINEAR_MODULE
            ] = self.language_expert_dense.linear
            dense_builder = CommonOpBuilderManager.get_builder(dense_parallel_param)
            dense_tensor_map = {
                _INPUT: attn_out_name,
                _LINEAR_OUT: dense_out_name,
            }
            graph = dense_builder.build(graph, dense_tensor_map)

    def build_graph(self, graph, is_prefill, with_vision_expert):
        atten_res_add = atb._BaseOperation(
            op_type="Elewise",
            op_param=json.dumps({"elewiseType": "ELEWISE_ADD"}),
            op_name="atten_res_add",
        )
        setattr(graph, "atten_res_add", atten_res_add)

        self.build_qkv_graph(graph, with_vision_expert)
        self.build_rope_graph(graph)
        self.build_attention_graph(graph, is_prefill)
        self.build_dense_graph(graph, is_prefill, with_vision_expert)

        graph.add_operation(
            graph.atten_res_add, ["hidden_states", "dense_out"], ["hidden_states"]
        )


class MLP(nn.Module):
    def __init__(
        self, config, weights, prefix, norm_prefix, backend=CommunicationBackend.LCCL, with_vision_expert=True
    ):
        super().__init__()

        # 配置信息
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.dtype = weights.dtype
        self.norm_prefix = norm_prefix
        self.backend = backend
        self.with_vision_expert = with_vision_expert

        # language linear settings
        self.language_up_proj = load_column_multi(
            config,
            prefixes=[
                f"{prefix}.language_mlp.gate_proj",
                f"{prefix}.language_mlp.up_proj",
            ],
            weights=weights,
            head_size=1,
        )
        self.language_down_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.language_mlp.down_proj",
            weights=weights,
            bias=False,
        )

        if self.with_vision_expert:
            # vision linear settings
            self.vision_up_proj = load_column_multi(
                config,
                prefixes=[
                    f"{prefix}.vision_mlp.gate_proj",
                    f"{prefix}.vision_mlp.up_proj",
                ],
                weights=weights,
                head_size=1,
            )
            self.vision_down_proj = TensorParallelRowLinear.load(
                config,
                prefix=f"{prefix}.vision_mlp.down_proj",
                weights=weights,
                bias=False,
            )

    def get_weights(self, prefix):
        weights_dict = OrderedDict()
        weights_dict.update(
            self.language_up_proj.linear.get_weights(f"{prefix}.language_up_proj")
        )
        weights_dict.update(
            self.language_down_proj.linear.get_weights(f"{prefix}.language_down_proj")
        )

        if self.with_vision_expert:
            weights_dict.update(
                self.vision_up_proj.linear.get_weights(f"{prefix}.vision_up_proj")
            )
            
            weights_dict.update(
                self.vision_down_proj.linear.get_weights(f"{prefix}.vision_down_proj")
            )
        return weights_dict

    def build_gateup_graph(
        self, graph, input_tensor_name, output_tensor_name, is_vision_branch
    ):
        linear_param = {
            _OP_NAME: "gate_up_linear",
            _CATEGORY: CommonOpBuilderType.LINEAR,
            "enable_quant_input": False,
            "default_dtype": self.dtype,
            "group_size": 0,
            _LINEAR_MODULE: (
                self.vision_up_proj.linear
                if is_vision_branch
                else self.language_up_proj.linear
            ),
        }

        gate_up_linear_param = {
            _OP_NAME: "gate_up_linear",
            _CATEGORY: CommonOpBuilderType.GATE_UP,
            "is_pack": True,
            _LINEAR_PARAM: linear_param,
        }
        gate_up_linear_tensor_map = {
            _INPUT: input_tensor_name,
            "gate_up_out": output_tensor_name,
        }

        builder = CommonOpBuilderManager.get_builder(gate_up_linear_param)
        graph = builder.build(graph, gate_up_linear_tensor_map)

    def build_activation_graph(self, graph, input_tensor_name, output_tensor_name):
        act_param = {
            _OP_NAME: "activation",
            _CATEGORY: CommonOpBuilderType.ACTIVATION,
            "is_pack": True,
            "up_weight_only": False,
            "activation_type": (
                ActivationType.SWIGLU
                if self.backend == CommunicationBackend.LCCL
                else ActivationType.SWISH
            ),
        }
        act_tensor_map = {_INPUT: input_tensor_name, "act_out": output_tensor_name}
        act_builder = CommonOpBuilderManager.get_builder(act_param)
        graph = act_builder.build(graph, act_tensor_map)

    def build_down_graph(
        self,
        graph,
        is_prefill,
        input_tensor_name,
        output_tensor_name,
        is_vision_branch,
    ):
        down_linear_param = {
            _OP_NAME: "down_linear",
            _CATEGORY: CommonOpBuilderType.LINEAR,
            _LINEAR_MODULE: (
                self.vision_down_proj.linear
                if is_vision_branch
                else self.language_down_proj.linear
            ),
            "enable_quant_input": False,
            "default_dtype": self.dtype,
            "group_size": 0,
        }
        down_linear_tensor_map = {
            _INPUT: input_tensor_name,
            _LINEAR_OUT: output_tensor_name,
        }

        down_linear_parallel_param = {
            _OP_NAME: "down_linear_parallel",
            _CATEGORY: CommonOpBuilderType.LINEAR_PARALLEL,
            "parallel_type": ParallelType.ALL_REDUCE,
            "parallel_info": TensorParallelInfo(
                rank=self.tp_rank, world_size=self.tp_world_size, backend=self.backend
            ),
            _LINEAR_PARAM: down_linear_param,
            "enable_lcoc": is_prefill,
        }
        linear_parallel_builder = CommonOpBuilderManager.get_builder(
            down_linear_parallel_param
        )
        graph = linear_parallel_builder.build(graph, down_linear_tensor_map)

    def build_graph(self, graph, is_prefill, with_vision_expert):
        norm_out_name = f"{self.norm_prefix}_out"
        mlp_out_name = "mlp_out"
        mlp_res_add = atb._BaseOperation(
            op_type="Elewise",
            op_param=json.dumps({"elewiseType": "ELEWISE_ADD"}),
            op_name="mlp_res_add",
        )
        setattr(graph, "mlp_res_add", mlp_res_add)

        if with_vision_expert:
            language_tensor_name = f"{norm_out_name}_language"
            vision_tensor_name = f"{norm_out_name}_vision"
            vision_expert_pre_process(
                graph, norm_out_name, language_tensor_name, vision_tensor_name
            )
            language_geteup_name = f"{language_tensor_name}_gateup"
            vision_geteup_name = f"{vision_tensor_name}_gateup"
            self.build_gateup_graph(
                graph, language_tensor_name, language_geteup_name, False
            )
            self.build_gateup_graph(
                graph, vision_tensor_name, vision_geteup_name, True
            )
            language_act_name = f"{language_geteup_name}_act"
            vision_act_name = f"{vision_geteup_name}_act"
            self.build_activation_graph(graph, language_geteup_name, language_act_name)
            self.build_activation_graph(graph, vision_geteup_name, vision_act_name)
            language_down_name = f"{language_act_name}_down"
            vision_down_name = f"{vision_act_name}_down"
            self.build_down_graph(
                graph,
                is_prefill,
                language_act_name,
                language_down_name,
                False,
            )
            self.build_down_graph(
                graph, is_prefill, vision_act_name, vision_down_name, True
            )
            vision_expert_post_process(
                graph,
                language_down_name,
                vision_down_name,
                mlp_out_name,
            )
        else:
            self.build_gateup_graph(
                graph, norm_out_name, "gate_up_out", with_vision_expert
            )
            self.build_activation_graph(graph, "gate_up_out", "mul_out")
            self.build_down_graph(
                graph, is_prefill, "mul_out", mlp_out_name, with_vision_expert
            )

        graph.add_operation(
            graph.mlp_res_add, ["hidden_states", mlp_out_name], ["layer_out"]
        )


class HiddenLayer(nn.Module):
    def __init__(self, layer_idx, config, weights, model_prefix, backend, with_vision_expert=True):
        super().__init__()
        prefix = f"{model_prefix}.layers.{layer_idx}"
        self.layer_idx = layer_idx
        self.config = config
        self.is_reshape = (
            config.vocab_size >= _EMBEDDING_PARALLEL_THRESHOLD
            and weights.process_group.size() > 1
        )
        self.with_vision_expert = with_vision_expert
        self.weight_names = None
        self.layer_graph = None

        self.self_attn = Attention(
            config=config,
            weights=weights,
            prefix=f"{prefix}.self_attn",
            norm_prefix=f"{prefix}.input_layernorm",
            backend=backend,
            with_vision_expert=with_vision_expert
        )

        self.input_layernorm = RMSNorm(config, weights, f"{prefix}.input_layernorm")

        self.mlp = MLP(
            config=config,
            weights=weights,
            prefix=f"{prefix}.mlp",
            norm_prefix=f"{prefix}.post_attention_layernorm",
            backend=backend,
            with_vision_expert=with_vision_expert
        )

        self.post_attention_layernorm = RMSNorm(
            config, weights, f"{prefix}.post_attention_layernorm"
        )

    def get_weights(self, prefix):
        weights_dict = OrderedDict()
        for name, module in self.named_children():
            weights_dict.update(module.get_weights(f"{prefix}.{name}"))
        self.weight_names = list(weights_dict.keys())
        return weights_dict

    def get_in_tensor_names(self, is_prefill, with_vision_expert):
        default_input = ["hidden_states", "seq_len", "slots_mapping"]

        if self.config.pe_type == "ROPE":
            default_input.extend(["cos_embedding", "sin_embedding"])
        if is_prefill:
            default_input.extend(["attention_mask"])
            if with_vision_expert:
                default_input.extend([_LANGUAGE_INDICES, _VISION_INDICES])
        else:
            default_input.extend(["block_tables"])
        return default_input

    def build_graph(self, graph, is_prefill, with_vision_expert):
        graph_name_prefix = ""
        if is_prefill:
            if with_vision_expert:
                graph_name_prefix = "prefill_with_vison_expert"
            else:
                graph_name_prefix = "prefill"
        else:
            graph_name_prefix = "decode"
        self.layer_graph = AtbGraph(f"{graph_name_prefix}_layer_{self.layer_idx}_graph")
        self.layer_graph.add_input_output(
            input=self.weight_names
            + ["k_cache", "v_cache"]
            + self.get_in_tensor_names(is_prefill, with_vision_expert),
            output=["layer_out"],
        )

        self.input_layernorm.build_graph(self.layer_graph)
        self.self_attn.build_graph(self.layer_graph, is_prefill, with_vision_expert)
        self.post_attention_layernorm.build_graph(self.layer_graph)
        self.mlp.build_graph(self.layer_graph, is_prefill, with_vision_expert)
        self.layer_graph.build()

        graph.operations.append(self.layer_graph)
        graph.add_operation(
            self.layer_graph,
            self.weight_names
            + [f"layer_{self.layer_idx}_k_cache", f"layer_{self.layer_idx}_v_cache"]
            + self.get_in_tensor_names(is_prefill, with_vision_expert),
            ["hidden_states"],
        )


class CogvlmModel(nn.Module):
    def __init__(
        self,
        config,
        weights,
        model_prefix: str = "model",
        lm_head_prefix: str = "lm_head",
        backend: CommunicationBackend = CommunicationBackend.LCCL,
        with_vision_expert:bool = True,
    ):
        super().__init__()
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.dtype = weights.dtype
        self.backend = backend

        self.layers = nn.ModuleList(
            [
                HiddenLayer(
                    layer_idx,
                    config,
                    weights,
                    model_prefix,
                    backend,
                    with_vision_expert
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(config, weights, f"{model_prefix}.norm")

        self.lm_head = load_column_multi(
            config,
            prefixes=[lm_head_prefix],
            weights=weights,
            head_size=1,
            lm_head=True,
        )

    def get_weights(self, prefix):
        weights_dict = OrderedDict()
        for name, module in self.named_children():
            if isinstance(module, nn.ModuleList):
                for i, single_module in enumerate(module):
                    weights_dict.update(
                        single_module.get_weights(f"{prefix}.{name}.{i}")
                    )
            elif isinstance(module, nn.Module):
                if name != "lm_head":
                    weights_dict.update(module.get_weights(f"{prefix}.{name}"))
                else:
                    weights_dict.update(module.linear.get_weights(f"{prefix}.{name}"))
        self.weight_names = list(weights_dict.keys())
        return weights_dict

    def build_positional_embedding_graph(self, graph):
        positional_embedding_param = {
            _OP_NAME: "positional_embedding",
            _CATEGORY: CommonOpBuilderType.POSITIONAL_EMBEDDING,
        }
        positional_embedding_tensor_map = {
            "position_ids": "position_ids",
            "cos_table": "cos_table",
            "sin_table": "sin_table",
            "cos_embedding": "cos_embedding",
            "sin_embedding": "sin_embedding",
        }
        builder = CommonOpBuilderManager.get_builder(positional_embedding_param)
        graph = builder.build(graph, positional_embedding_tensor_map)

    def build_lm_head(self, graph, is_prefill):
        lm_head_linear_param = {
            _OP_NAME: "lm_head_linear",
            _CATEGORY: CommonOpBuilderType.LINEAR,
            _LINEAR_MODULE: self.lm_head.linear,
            "default_dtype": self.dtype,
        }
        lm_head_linear_parallel_param = {
            _OP_NAME: "lm_head_linear_parallel",
            _CATEGORY: CommonOpBuilderType.LINEAR_PARALLEL,
            "parallel_type": ParallelType.ALL_GATHER,
            "parallel_info": TensorParallelInfo(
                rank=self.tp_rank, world_size=self.tp_world_size, backend=self.backend
            ),
            _LINEAR_PARAM: lm_head_linear_param,
        }
        lm_head_param = {
            _OP_NAME: "test_lm_head",
            _CATEGORY: CommonOpBuilderType.LM_HEAD,
            "enable_linear_parallel": True,
            "linear_parallel_param": lm_head_linear_parallel_param,
            "gather_ahead": is_prefill,
            "unpad_inputs": True,
        }
        lm_head_linear_tensor_map = {
            _INPUT: "model.norm_out",
            "indices": "lm_head_indices",
            _LINEAR_OUT: "model_out",
        }
        lm_head_builder = CommonOpBuilderManager.get_builder(lm_head_param)
        graph = lm_head_builder.build(graph, lm_head_linear_tensor_map)

    def build_graph(self, graph, is_prefill, with_vision_expert):
        self.build_positional_embedding_graph(graph)
        for layer in self.layers:
            layer.build_graph(graph, is_prefill, with_vision_expert)
        self.norm.build_graph(graph)
        self.build_lm_head(graph, is_prefill)
