# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
from ..network import Tensor, Node, get_default_net


class RmsNorm:
    def __init__(self, weight_name: str, eps: float = 1e-5):
        self.weight = Tensor(weight_name)
        self.weight_name = weight_name
        self.eps = eps

    def __call__(self, inputs: Tensor):
        return self._forward(inputs, self.weight)

    def _forward(self, inputs: Tensor, weight: Tensor) -> Tensor:
        out = Tensor()
        param = {'layerType':'RMS_NORM_NORM', 'epsilon': self.eps}
        node = Node('RmsNorm', param, [inputs, weight], [out])
        get_default_net().push_node(node)
        get_default_net().push_weight_key(self.weight_name)
        return out