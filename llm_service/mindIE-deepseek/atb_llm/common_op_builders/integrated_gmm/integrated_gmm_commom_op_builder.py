# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json
from enum import Enum
from pydantic import BaseModel, Field

import _libatb_torch as atb

from atb_llm.common_op_builders.base_common_op_builder import BaseCommonOpBuilder, BaseCommonOpBuilderParam
from atb_llm.common_op_builders.data_type import CommonOpBuilderType
from atb_llm.utils.singleton import Singleton
from atb_llm.utils.quantize.pack_type import PackType, LinearType, AclDataType, TransposeType
from atb_llm.utils.quantize.quant_type import GmmQuantType, QuantTypeV2


class IntegratedGmmIdx(int, Enum):
    ROUTER_IDX = 0,
    MOE_MLP_GATE_IDX = 1,
    MOE_MLP_UP_IDX = 2,
    MOE_MLP_DOWN_IDX = 3,


class IntegratedGmmCommonOpBuilderParam(BaseCommonOpBuilderParam):
    moe_linear_quant_type: list | None = Field({})
    has_bias: bool = Field(False)
    is_up: bool = Field(False)
    out_data_type: AclDataType = Field(AclDataType.ACL_FLOAT16)
    transpose_b: bool = Field(False)
    pack_quant_type: int = Field(TransposeType.TRANSPOSE)
    dense_quant_type: int = Field(0)


class IntegratedGmmCommonOpBuilderInTensor(BaseModel):
    input: str = Field(None)
    weight: str = Field(None)
    bias: str = Field(None)
    descale: str = Field(None)
    offset: str = Field(None)
    scale: str = Field(None)
    compress_idx: str = Field(None)
    group_list: str = Field(None)


class IntegratedGmmCommonOpBuilderOutTensor(BaseModel):
    out: str = Field(None)


class IntegratedGmmCommonOpBuilder(BaseCommonOpBuilder, Singleton):
    def __init__(self):
        super().__init__()
        self.category = CommonOpBuilderType.INTEGRATED_GMM

    @property
    def param_cls(self):
        return IntegratedGmmCommonOpBuilderParam

    @property
    def in_tensor_cls(self):
        return IntegratedGmmCommonOpBuilderInTensor

    @property
    def out_tensor_cls(self):
        return IntegratedGmmCommonOpBuilderOutTensor

    def is_match(self, param: dict):
        if not super().verify_base_param(param):
            return False
        return True

    def get_linear_quant_type(self, pack_quant_type, linear_type, has_norm):
        if linear_type == LinearType.FP:
            return QuantTypeV2.NO_QUANT
        elif pack_quant_type in (
            PackType.ALL_W4A16,
            PackType.ALL_W4A16_ANTI,
            PackType.MIX_W4A16,
            PackType.MIX_W4A16_ANTI
        ):
            return QuantTypeV2.W4A16
        elif pack_quant_type in (
            PackType.ALL_W8A16,
            PackType.ALL_W8A16_ANTI,
            PackType.MIX_W8A16,
            PackType.MIX_W8A16_ANTI
        ):
            return QuantTypeV2.W8A16
        elif pack_quant_type in (
            PackType.ALL_W8A8_DYNAMIC,
            PackType.MIX_W8A8_DYNAMIC
        ):
            return QuantTypeV2.LINEAR_W8A8_DYNAMIC_QUANT
        elif pack_quant_type in (
            PackType.ALL_W8A8SC,
            PackType.ALL_W8A8SC_ANTI,
            PackType.MIX_W8A8SC,
            pack_quant_type == PackType.MIX_W8A8SC_ANTI
        ):
            return QuantTypeV2.LINEAR_W8A8_SC_DEQUANT if has_norm else \
                QuantTypeV2.LINEAR_W8A8_SC_QUANT
        else:
            return QuantTypeV2.LINEAR_W8A8_DEQUANT if has_norm else \
                QuantTypeV2.LINEAR_W8A8_QUANT

    def calc_gmm_quant_type(self):
        temp_quant_type = self.get_linear_quant_type(
            self.param.pack_quant_type \
                if self.param.dense_quant_type == PackType.PACK_QUANT_UNDEFINED \
                else self.param.dense_quant_type,
            self.param.moe_linear_quant_type[IntegratedGmmIdx.MOE_MLP_GATE_IDX \
                if self.param.is_up else IntegratedGmmIdx.MOE_MLP_DOWN_IDX],
            False
        )
        if temp_quant_type == QuantTypeV2.NO_QUANT:
            return GmmQuantType.NONE
        elif temp_quant_type == QuantTypeV2.LINEAR_W8A8_DYNAMIC_QUANT or \
            temp_quant_type == QuantTypeV2.LINEAR_W8A8_DYNAMIC_DEQUANT:
            return GmmQuantType.W8A8_TOKEN
        elif temp_quant_type == QuantTypeV2.W8A16:
            return GmmQuantType.W8A16_CHANNEL
        else:
            return GmmQuantType.W8A8_CHANNEL
            
    def reshape0(self, org_shape):
        return [org_shape[0], org_shape[1]]

    def build(self, graph: atb._GraphOperation, tensor_map: dict) -> atb._GraphOperation:
        self.in_tensor_key = self.in_tensor_cls.parse_obj(tensor_map)
        self.out_tensor_key = self.out_tensor_cls.parse_obj(tensor_map)

        gmm_quant_type = GmmQuantType.NONE
        if gmm_quant_type == GmmQuantType.W8A8_TOKEN:
            dynamic_quant_op = atb._BaseOperation(
                op_type="DynamicQuant",
                op_param=json.dumps({}),
                op_name=f"{self.param.op_name}_dynamic_quant"
            )
            graph.operations.append(dynamic_quant_op)
            graph.add_operation(
                dynamic_quant_op,
                [self.in_tensor_key.input],
                ["INTERMEDIATE_QUANT_OUT", "INTERMEDIATE_DYNAMIC_SCALE"]
            )

        
        gmm_op = atb._BaseOperation(
            op_type="GroupedMatmul",
            op_param=json.dumps({
                "transposeB": self.param.transpose_b,
                "quantType": gmm_quant_type,
                "outDataType": self.param.out_data_type,
            }),
            op_name=f"{self.param.op_name}_gmm"
        )
        input_key_list = [
            "INTERMEDIATE_QUANT_OUT" if gmm_quant_type == GmmQuantType.W8A8_TOKEN \
                else self.in_tensor_key.input,
            self.in_tensor_key.weight,
        ]
        if self.param.has_bias:
            input_key_list.append(self.in_tensor_key.bias)
        if gmm_quant_type == GmmQuantType.W8A16_CHANNEL:
            graph.add_reshape(self.in_tensor_key.scale,
                self.in_tensor_key.scale, self.reshape0)
            graph.add_reshape(self.in_tensor_key.offset,
                self.in_tensor_key.offset, self.reshape0)
            input_key_list.extend([self.in_tensor_key.scale, 
            self.in_tensor_key.offset, self.in_tensor_key.group_list])
        elif gmm_quant_type == GmmQuantType.W8A8_CHANNEL:
            input_key_list.extend([self.in_tensor_key.scale, 
            self.in_tensor_key.compress_idx, self.in_tensor_key.group_list])
        elif gmm_quant_type == GmmQuantType.W8A8_TOKEN:
            graph.add_reshape(self.in_tensor_key.scale,
                self.in_tensor_key.scale, self.reshape0)
            input_key_list.extend([self.in_tensor_key.scale, \
            "INTERMEDIATE_DYNAMIC_SCALE", self.in_tensor_key.group_list])
        else:
            input_key_list.append(self.in_tensor_key.group_list)
        graph.operations.append(gmm_op)
        graph.add_operation(
            gmm_op,
            input_key_list,
            [self.out_tensor_key.out]
        )

        return graph
