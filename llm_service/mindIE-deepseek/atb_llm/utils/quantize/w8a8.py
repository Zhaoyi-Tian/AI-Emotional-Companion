# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import torch
from torch import nn
import numpy as np

from ..layers.linear.linear_utils import LinearUtils
from .quant_type import LinearTypeV2
from ..initial import NPUSocInfo


class W8A8LinearStatic(nn.Module, LinearUtils):
    def __init__(self, weight, deq_scale, input_scale, quant_bias=None, input_offset=None, bias=None):
        super().__init__()
        super(nn.Module, self).__init__()
        self.linear_desc = LinearTypeV2.W8A8

        self.register_buffer('weight', weight.to(torch.int8))

        self.act_quant_name = 'per_tensor'
        self.register_buffer('input_scale', input_scale)

        if input_offset is not None:
            self.register_buffer('input_offset', input_offset.to(torch.int8))
        else:
            self.register_buffer('input_offset', torch.tensor([], dtype=torch.int8))

        self.weight_quant_name = 'per_channel'

        if NPUSocInfo().soc_version in (100, 101, 102, 103, 104):
            deq_scale = self._transform_deqscale_dtype_to_float(deq_scale)

        self.register_buffer('deq_scale', deq_scale)

        if quant_bias is not None:
            self.register_buffer('quant_bias', quant_bias)
            self.has_bias = True
        else:
            self.quant_bias = None

        self.output_quant_name = 'per_channel'

        if bias is not None:
            self.register_buffer('bias', bias)

    def _transform_deqscale_dtype_to_float(self, deq_scale_int64):
        deq_scale_cpu_int64 = deq_scale_int64.cpu().numpy()
        deq_scale_int32 = np.uint32(deq_scale_cpu_int64)
        original_deq_scale = np.frombuffer(deq_scale_int32.tobytes(), dtype=np.float32).copy()
        tmp_deq_scale = torch.from_numpy(original_deq_scale)
        return tmp_deq_scale.to(deq_scale_int64.device)
