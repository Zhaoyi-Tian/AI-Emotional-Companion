# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json
import copy
from pydantic import Field

import _libatb_torch as atb

from atb_llm.common_op_builders.common_op_builder_manager import CommonOpBuilderManager
from atb_llm.common_op_builders.fusion_linear.base_fusion_linear_commom_op_builder import \
    BaseFusionLinearCommonOpBuilderParam, BaseFusionLinearCommonOpBuilderInTensor, \
    BaseFusionLinearCommonOpBuilder
from atb_llm.utils.quantize.pack_type import TransposeType

_INPUT = "input"
_WEIGHT = "weight"
_BIAS = "bias"
_SCALE = "scale"
_OFFSET = "offset"
_DESCALE = "descale"
_COMPRESS_IDX = "compress_idx"
_OUTPUT = "output"


class FusionLinearWithLoraCommonOpBuilderInTensor(BaseFusionLinearCommonOpBuilderInTensor):
    group_list: str = Field(...)
    lora_a: str = Field(...)
    lora_b: str = Field(...)
    im_mask: str = Field(...)


class FusionLinearWithLoraCommonOpBuilder(BaseFusionLinearCommonOpBuilder):
    def __init__(self):
        super().__init__()
    
    @property
    def in_tensor_cls(self):
        return FusionLinearWithLoraCommonOpBuilderInTensor

    def is_match(self, param: dict):
        if not super().is_match(param):
            return False
        return self.param.support_lora

    def build(self, graph: atb._GraphOperation, tensor_map: dict) -> atb._GraphOperation:
        super().build(graph, tensor_map)

        base_linear_param = copy.deepcopy(self.param)
        base_linear_param.support_lora = False
        base_linear_param.lora_enable_gmm = False
        base_linear_builder = CommonOpBuilderManager.get_builder(base_linear_param)
        base_linear_tensor_map = {
            _INPUT: self.in_tensor_key.input,
            _WEIGHT: self.in_tensor_key.weight,
            _BIAS: self.in_tensor_key.bias,
            _SCALE: self.in_tensor_key.scale,
            _OFFSET: self.in_tensor_key.offset,
            _DESCALE: self.in_tensor_key.descale,
            _COMPRESS_IDX: self.in_tensor_key.compress_idx,
            _OUTPUT: f"{self.param.op_name}_intermediate_base_linear_out",
        }
        graph = base_linear_builder.build(graph, base_linear_tensor_map)

        if self.param.use_im_mask:
            mul_op = atb._BaseOperation(
                op_type="Elewise",
                op_param=json.dumps({"elewiseType": "ELEWISE_MUL"}),
                op_name=f"{self.param.op_name}_MUL"
            )
            graph.operations.append(mul_op)
            graph.add_operation(
                mul_op,
                [self.in_tensor_key.input, self.in_tensor_key.im_mask],
                [f"{self.param.op_name}_intermediate_im_mask_out"]
            )

        lora_linear_param = BaseFusionLinearCommonOpBuilderParam()
        lora_linear_param.is_bf16 = self.param.is_bf16
        lora_linear_param.has_bais = False
        lora_linear_param.transpose_type = TransposeType.TRANSPOSE
        lora_linear_param.lora_enable_gmm = self.param.lora_enable_gmm
        lora_linear_param.use_im_mask = self.param.use_im_mask

        # Lora A
        if self.param.lora_enable_gmm:
            lora_a_op = atb._BaseOperation(
                op_type="GroupedMatmul",
                op_param=json.dumps({"transposeB": True}),
                op_name=f"{self.param.op_name}_LORA_A"
            )
            graph.operations.append(lora_a_op)
            lora_a_input_key_list = [f"{self.param.op_name}_intermediate_im_mask_out" if self.param.us_im_mask
                else self.in_tensor_key.input, self.in_tensor_key.lora_a]
            if self.param.lora_enable_gmm:
                lora_a_input_key_list.append(self.in_tensor_key.grout_list)
            else:
                lora_a_input_key_list.append(self.in_tensor_key.scale)
                lora_a_input_key_list.append(self.in_tensor_key.offset)
                lora_a_input_key_list.append(self.in_tensor_key.descale)
                lora_a_input_key_list.append(self.in_tensor_key.bias)
                lora_a_input_key_list.append(self.in_tensor_key.compress_idx)
            graph.add_operation(
                lora_a_op,
                lora_a_input_key_list,
                [f"{self.param.op_name}_intermediate_lora_a_out"]
            )
        else:
            lora_a_linear_builder = CommonOpBuilderManager.get_builder(lora_linear_param)
            lora_a_linear_tensor_map = {
                _INPUT: f"{self.param.op_name}_intermediate_im_mask_out" if self.param.us_im_mask
                    else self.in_tensor_key.input,
                _WEIGHT: self.in_tensor_key.weight,
                _BIAS: self.in_tensor_key.bias,
                _SCALE: self.in_tensor_key.scale,
                _OFFSET: self.in_tensor_key.offset,
                _DESCALE: self.in_tensor_key.descale,
                _COMPRESS_IDX: self.in_tensor_key.compress_idx,
                _OUTPUT: f"{self.param.op_name}_intermediate_lora_a_out",
            }
            graph = lora_a_linear_builder.build(graph, lora_a_linear_tensor_map)

        lora_linear_param.transpose_type = TransposeType.TRANSPOSE

        # Lora B
        if self.param.lora_enable_gmm:
            lora_b_op = atb._BaseOperation(
                op_type="GroupedMatmul",
                op_param=json.dumps({"transposeB": False}),
                op_name=f"{self.param.op_name}_LORA_B"
            )
            graph.operations.append(lora_b_op)
            lora_b_input_key_list = [f"{self.param.op_name}_intermediate_lora_a_out", self.in_tensor_key.lora_b]
            if self.param.lora_enable_gmm:
                lora_b_input_key_list.append(self.in_tensor_key.grout_list)
            else:
                lora_b_input_key_list.append(self.in_tensor_key.scale)
                lora_b_input_key_list.append(self.in_tensor_key.offset)
                lora_b_input_key_list.append(self.in_tensor_key.descale)
                lora_b_input_key_list.append(self.in_tensor_key.bias)
                lora_b_input_key_list.append(self.in_tensor_key.compress_idx)
            graph.add_operation(
                lora_b_op,
                lora_b_input_key_list,
                [f"{self.param.op_name}_intermediate_lora_b_out"]
            )
        else:
            lora_b_linear_builder = CommonOpBuilderManager.get_builder(lora_linear_param)
            lora_b_linear_tensor_map = {
                _INPUT: f"{self.param.op_name}_intermediate_lora_a_out",
                _WEIGHT: self.in_tensor_key.weight,
                _BIAS: self.in_tensor_key.bias,
                _SCALE: self.in_tensor_key.scale,
                _OFFSET: self.in_tensor_key.offset,
                _DESCALE: self.in_tensor_key.descale,
                _COMPRESS_IDX: self.in_tensor_key.compress_idx,
                _OUTPUT: f"{self.param.op_name}_intermediate_lora_b_out",
            }
            graph = lora_b_linear_builder.build(graph, lora_b_linear_tensor_map)

        add_op = atb._BaseOperation(
            op_type="Elewise",
            op_param=json.dumps({"elewiseType": "ELEWISE_ADD"}),
            op_name=f"{self.param.op_name}_ADD"
        )
        graph.operations.append(add_op)
        graph.add_operation(
            add_op,
            [f"{self.param.op_name}_intermediate_base_linear_out", f"{self.param.op_name}_intermediate_lora_b_out"],
            [self.out_tensor_key.output]
        )

        return graph
