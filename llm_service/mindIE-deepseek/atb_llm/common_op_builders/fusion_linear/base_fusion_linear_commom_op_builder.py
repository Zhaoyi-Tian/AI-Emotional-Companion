# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from pydantic import BaseModel, Field

import _libatb_torch as atb

from atb_llm.common_op_builders.base_common_op_builder import BaseCommonOpBuilder, BaseCommonOpBuilderParam
from atb_llm.common_op_builders.data_type import CommonOpBuilderType
from atb_llm.utils.quantize.pack_type import TransposeType
from atb_llm.utils.quantize.quant_type import QuantTypeV2


class BaseFusionLinearCommonOpBuilderParam(BaseCommonOpBuilderParam):
    quant_type: QuantTypeV2 = Field(QuantTypeV2.NO_QUANT)
    is_bf16: bool = Field(False)
    has_bias: bool = Field(False)
    support_lora: bool = Field(False)
    use_im_mask: bool = Field(False)
    lora_enable_gmm: bool = Field(False)
    transpose_type: int = Field(TransposeType.TRANSPOSE)
    quant_group_size: int = Field(0)


class BaseFusionLinearCommonOpBuilderInTensor(BaseModel):
    input: str = Field(None)
    weight: str = Field(None)
    bias: str = Field(None)
    scale: str = Field(None)
    offset: str = Field(None)
    descale: str = Field(None)
    compress_idx: str = Field(None)


class BaseFusionLinearCommonOpBuilderOutTensor(BaseModel):
    output: str = Field(None)


class BaseFusionLinearCommonOpBuilder(BaseCommonOpBuilder):
    def __init__(self):
        super().__init__()
        self.category = CommonOpBuilderType.FUSION_LINEAR

    @property
    def param_cls(self):
        return BaseFusionLinearCommonOpBuilderParam

    @property
    def in_tensor_cls(self):
        return BaseFusionLinearCommonOpBuilderInTensor

    @property
    def out_tensor_cls(self):
        return BaseFusionLinearCommonOpBuilderOutTensor

    def is_match(self, param: dict):
        if not super().verify_base_param(param):
            return False
        return True

    def build(self, graph: atb._GraphOperation, tensor_map: dict) -> atb._GraphOperation:
        self.in_tensor_key = self.in_tensor_cls.model_validate(tensor_map)
        self.out_tensor_key = self.out_tensor_cls.model_validate(tensor_map)
        return graph