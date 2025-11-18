# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json

import _libatb_torch as atb

from atb_llm.common_op_builders.fusion_linear.base_fusion_linear_commom_op_builder import \
    BaseFusionLinearCommonOpBuilder, TransposeType

from atb_llm.utils.quantize.quant_type import QuantTypeV2

_TRANSPOSE_B = "transposeB"
_HAS_BIAS = "hasBias"


class FusionLinearCommonOpBuilder(BaseFusionLinearCommonOpBuilder):
    def __init__(self):
        super().__init__()

    def is_match(self, param: dict):
        if not super().is_match(param):
            return False
        return not self.param.support_lora

    def reshape_intermediate_scale(self, org_shape):
        return [org_shape[0]]

    def build(self, graph: atb._GraphOperation, tensor_map: dict) -> atb._GraphOperation:
        super().build(graph, tensor_map)

        if self.param.quant_type == QuantTypeV2.LINEAR_W8A8_QUANT \
            or self.param.quant_type == QuantTypeV2.LINEAR_W8A8_SC_QUANT:
            input_quant_op = atb._BaseOperation(
                op_type="Elewise",
                op_param=json.dumps({"elewiseType": "ELEWISE_QUANT_PER_CHANNEL"}),
                op_name=f"{self.param.op_name}_ElewiseQuant"
            )
            graph.operations.append(input_quant_op)
            graph.add_operation(
                input_quant_op,
                [self.in_tensor_key.input, self.in_tensor_key.input_scale, self.in_tensor_key.input_offset],
                [f"{self.param.op_name}_intermediate_input"]
            )
        elif self.param.quant_type == QuantTypeV2.LINEAR_W8A8_DYNAMIC_QUANT:
            input_quant_op = atb._BaseOperation(
                op_type="DynamicQuant",
                op_param=json.dumps({}),
                op_name=f"{self.param.op_name}_DynamicQuant"
            )
            graph.operations.append(input_quant_op)
            graph.add_operation(
                input_quant_op,
                [self.in_tensor_key.input, self.in_tensor_key.input_scale, self.in_tensor_key.input_offset],
                [f"{self.param.op_name}_intermediate_input", f"{self.param.op_name}_intermediate_scale"]
            )

        if self.param.quant_type == QuantTypeV2.LINEAR_W8A8_SC_DEQUANT or \
            self.param.quant_type == QuantTypeV2.LINEAR_W8A8_SC_QUANT:
            sparse_linear_op = atb._BaseOperation(
                op_type="LinearSparse",
                op_param=json.dumps({
                    "tilingK": 8,
                    "tilingN": 8,
                }),
                op_name=f"{self.param.op_name}_LinearSparse"
            )
            graph.operations.append(sparse_linear_op)
            graph.add_operation(
                sparse_linear_op,
                [
                    f"{self.param.op_name}_intermediate_input" \
                        if self.param.quant_type == QuantTypeV2.LINEAR_W8A8_SC_QUANT \
                        else self.in_tensor_key.input,
                    self.in_tensor_key.weight,
                    self.in_tensor_key.bias,
                    self.in_tensor_key.descale,
                    self.in_tensor_key.compress_idx,
                ],
                [self.out_tensor_key.output]
            )
        elif self.param.quant_type == QuantTypeV2.W8A16:
            w8a16_op = atb._BaseOperation(
                op_type="W8A16MatMul",
                op_param=json.dumps({
                    _TRANSPOSE_B: self.param.transposeType == TransposeType.TRANSPOSE,
                    "quantGroupSize": self.param.quant_group_size, 
                    _HAS_BIAS: self.param.has_bias,
                }),
                op_name=f"{self.param.op_name}_W8A16MatMul"
            )
            input_key_list = [self.in_tensor_key.input, self.in_tensor_key.weight,
                self.in_tensor_key.bias, self.in_tensor_key.descale]
            if self.param.has_bias:
                input_key_list.append(self.in_tensor_key.bias)
            graph.operations.append(w8a16_op)
            graph.add_operation(
                w8a16_op,
                input_key_list,
                [self.out_tensor_key.output]
            )
        elif self.param.quant_type == QuantTypeV2.W4A16:
            w4a16_op = atb._BaseOperation(
                op_type="W4A16MatMul",
                op_param=json.dumps({
                    _TRANSPOSE_B: self.param.transposeType == TransposeType.TRANSPOSE,
                    "quantGroupSize": self.param.quant_group_size,
                    _HAS_BIAS: self.param.has_bias,
                }),
                op_name=f"{self.param.op_name}_W4A16MatMul"
            )
            input_key_list = [self.in_tensor_key.input, self.in_tensor_key.weight,
                self.in_tensor_key.bias, self.in_tensor_key.descale]
            if self.param.has_bias:
                input_key_list.append(self.in_tensor_key.bias)
            graph.add_operation(
                w4a16_op,
                input_key_list,
                [self.out_tensor_key.output]
            )
        elif self.param.quant_type == QuantTypeV2.LINEAR_W8A8_DYNAMIC_QUANT:
            graph.add_reshape(f"{self.param.op_name}_intermediate_scale", f"{self.param.op_name}_intermediate_scale",
                self.reshape_intermediate_scale)
            w8a8_op = atb._BaseOperation(
                op_type="W8A8MatMul",
                op_param=json.dumps({
                    _TRANSPOSE_B: self.param.transposeType == TransposeType.TRANSPOSE
                }),
                op_name=f"{self.param.op_name}_W8A8MatMul"
            )
            graph.operations.append(w8a8_op)
            graph.add_operation(
                w8a8_op,
                [f"{self.param.op_name}_intermediate_input", self.in_tensor_key.weight,
                self.in_tensor_key.scale, f"{self.param.op_name}_intermediate_scale"],
                [self.out_tensor_key.output]
            )
        else:
            linear_op = atb._BaseOperation(
                op_type="Linear",
                op_param=json.dumps({
                    "outDataType": self.param.is_bf16,
                    _TRANSPOSE_B: self.param.transpose_type == TransposeType.TRANSPOSE,
                    _HAS_BIAS: self.param.has_bias or self.param.quant_type != QuantTypeV2.NO_QUANT,
                }),
                op_name=f"{self.param.op_name}_Linear"
            )
            graph.operations.append(linear_op)
            input_key_list = [
                self.in_tensor_key.input if self.param.quant_type == QuantTypeV2.NO_QUANT \
                    else f"{self.param.op_name}_intermediate_input",
                self.in_tensor_key.weight,
            ]
            if self.param.has_bias:
                input_key_list.append(self.in_tensor_key.bias)
            if self.param.quant_type != QuantTypeV2.NO_QUANT:
                input_key_list.append(self.in_tensor_key.descale)
            graph.add_operation(
                linear_op,
                input_key_list,
                [self.out_tensor_key.output]
            )

        return graph
