# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from dataclasses import dataclass
from typing import Optional
from atb_llm.nn.network import Tensor, Node, get_default_net


class Linear():
    '''Base linear module, used for float tensor linear computation.'''
    def __init__(
            self,
            w_prefix: str,
            b_prefix: Optional[str] = None,
            op_name: str = 'Linear',
            transpose_weight: bool = True,
            is_bf16: bool = False
    ):
        self.w_prefix = w_prefix
        self.transpose_weight = transpose_weight
        self.weight = Tensor(w_prefix)
        self.bias = Tensor(b_prefix) if b_prefix is not None else None
        self.op_name = op_name
        self.is_bf16 = is_bf16

    def __call__(
            self,
            input_tensor: Tensor
        ) -> Tensor:
        return self._forward(input_tensor, self.weight, self.bias, self.transpose_weight)

    def _forward(
            self,
            input_tensor: Tensor,
            weight: Tensor,
            bias: Optional[Tensor] = None,
            transpose_weight: Optional[bool] = None,
        ) -> Tensor:
        out = Tensor()
        inputs = [input_tensor, weight]
        has_bias = False
        if bias is not None:
            inputs.append(bias)
            has_bias = True
        transpose_weight = transpose_weight if transpose_weight is not None else self.transpose_weight
        param = {
            'hasBias': has_bias,
            'transposeB': transpose_weight
        }
        node = Node('Linear', param, inputs, [out])
        get_default_net().push_node(node)
        get_default_net().push_weight_key(self.w_prefix)
        return out