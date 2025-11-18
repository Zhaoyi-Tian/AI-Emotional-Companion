# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import torch
from torch import nn

from ..layers.linear.linear_utils import LinearUtils
from .pack_type import TransposeType
from .quant_type import LinearTypeV2


def int42int8(weight):
    weight = weight.to(torch.int8)

    e = 0 # number of experts
    if len(weight.shape) == 2:
        k, n = weight.shape
    elif len(weight.shape) == 3:
        e, k, n = weight.shape
    n_new = n // 2 + n % 2

    if n_new != n // 2:
        raise AssertionError("n dimension should be even")

    weight = weight.reshape(-1, 2)
    weight0 = weight[:, :1]
    weight1 = weight[:, 1:]

    weight1_4 = torch.bitwise_left_shift(weight1, 4)
    weight2_4 = weight0 & 0b00001111

    weight_add = torch.bitwise_or(weight1_4, weight2_4)
    if e == 0:
        weight_res = weight_add.reshape(k, n_new)
    else:
        weight_res = weight_add.reshape(e, k, n_new)
    return weight_res


class W4A16LinearStatic(nn.Module, LinearUtils):
    def __init__(self, weight, weight_scale, weight_offset, bias=None):
        super().__init__()
        super(nn.Module, self).__init__()

        self.weight_quant_name = 'w4a16'
        self.linear_desc = LinearTypeV2.W4A16

        # per group 推荐不Transpose，per channel转置
        self.trans_flag = TransposeType.TRANSPOSE if weight_scale.shape[-1] == 1 else TransposeType.NOT_TRANSPOSE

        weight_in_k_n = weight.transpose(-1, -2).contiguous()  # k, n

        weight_trans = weight_in_k_n if self.trans_flag == TransposeType.NOT_TRANSPOSE \
            else weight_in_k_n.transpose(-1, -2).contiguous()

        weight_compact = int42int8(weight_trans)  # [k, n // 2] or [n, k // 2]
        self.register_buffer('weight', weight_compact.to(torch.int8))

        self.register_buffer('weight_scale', weight_scale
                             if self.trans_flag == TransposeType.TRANSPOSE
                             else weight_scale.transpose(-1, -2).contiguous())

        if weight_offset is not None:
            self.register_buffer('weight_offset', (-weight_offset)
                                 if self.trans_flag == TransposeType.TRANSPOSE
                                 else (-weight_offset).transpose(-1, -2).contiguous())
        else:
            self.weight_offset = None

        if bias is not None:
            if bias.dtype == torch.bfloat16:
                bias = bias.to(torch.float32)
            self.register_buffer('bias', bias)
            self.has_bias = True
