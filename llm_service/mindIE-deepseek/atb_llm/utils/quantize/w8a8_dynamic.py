# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import torch
from torch import nn

from ..layers.linear.linear_utils import LinearUtils
from .pack_type import TransposeType
from .quant_type import LinearTypeV2


class W8A8DynamicLinearStatic(nn.Module, LinearUtils):
    def __init__(self, weight, weight_scale, weight_offset, bias=None, need_flatten=True):
        super().__init__()
        super(nn.Module, self).__init__()
        self.weight_quant_name = 'per_channel'
        self.trans_flag = self.check_transpose(weight)
        self.linear_desc = LinearTypeV2.W8A8_DYNAMIC

        self.register_buffer('weight', weight.to(torch.int8)
                             if self.trans_flag == TransposeType.TRANSPOSE else weight.T.contiguous().to(torch.int8))

        weight_scale_dtype = weight_scale.dtype if weight_scale.dtype == torch.bfloat16 else torch.float32
        self.register_buffer('weight_scale', weight_scale.to(weight_scale_dtype).flatten()
                            if need_flatten else weight_scale.to(weight_scale_dtype))

        if weight_offset is not None:
            self.register_buffer('weight_offset', -(weight_offset.flatten()) 
                                if need_flatten else -(weight_offset))
        else:
            self.weight_offset = None

        if bias is not None:
            self.register_buffer('bias', bias)
            self.has_bias = True
