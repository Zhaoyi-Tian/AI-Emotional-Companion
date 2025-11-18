# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
from enum import Enum
from ..network import Tensor, Node, get_default_net


class ActType(Enum):
    RELU = 1
    GELU = 2
    FAST_GELU = 3
    SWISH = 4
    LOG = 5
    SWIGLU = 6
    SIGMOID = 7


class GeluMode(Enum):
    TANH = 1
    NONE = 2

act_type_map = {
    ActType.RELU:'ACTIVATION_RELU',
    ActType.GELU:'ACTIVATION_GELU',
    ActType.FAST_GELU:'ACTIVATION_FAST_GELU',
    ActType.SWISH:'ACTIVATION_SWISH',
    ActType.LOG:'ACTIVATION_LOG',
    ActType.SWIGLU:'ACTIVATION_SWIGLU_FORWARD',
    ActType.SIGMOID:'ACTIVATION_SIGMOID',
}

gelu_mode_map = {
    GeluMode.TANH:'TANH_MODE',
    GeluMode.NONE:'NONE_MODE',
}


def activation(input_tensor: Tensor, act_type: ActType, scale=1.0, dim=-1, gelu_mode: GeluMode = GeluMode.TANH):
    out = Tensor()
    param = {
        'activationType': act_type_map[act_type],
        'scale': scale,
        'dim': dim,
        'geluMode': gelu_mode_map[gelu_mode],
    }
    node = Node('Activation', param, [input_tensor], [out])
    get_default_net().push_node(node)
    return out