# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

from enum import Enum
from .network import Tensor, Node, get_default_net


class Ops:
    @staticmethod
    def split(split_tensor: Tensor, split_dim: int, split_num: int) -> list[Tensor]:
        param = {'splitDim':split_dim, 'splitNum':split_num}
        outs = [Tensor() for _ in range(split_num)]
        ins = [split_tensor]
        node = Node('Split', param, ins, outs)
        get_default_net().push_node(node)
        return tuple(outs)

    @staticmethod
    def reshape_and_cache(k: Tensor, v: Tensor, k_cache: Tensor, v_cache: Tensor, slot_mapping: Tensor):
        node = Node("ReshapeAndCache", {"compressType":"COMPRESS_TYPE_UNDEFINED", "kvCacheCfg":0},
            [k, v, k_cache, v_cache, slot_mapping], [k_cache, v_cache])
        get_default_net().push_node(node)

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

    @staticmethod
    def activation(input_tensor: Tensor, act_type: ActType, scale=1.0, dim=-1, gelu_mode:GeluMode = GeluMode.TANH):
        out = Tensor()
        param = {
            'activationType':Ops.act_type_map[act_type],
            'scale':scale,
            'dim':dim,
            'geluMode':Ops.gelu_mode_map[gelu_mode],
        }
        node = Node('Activation', param, [input_tensor], [out])
        get_default_net().push_node(node)
        return out

    @staticmethod
    def cat(x: Tensor, y: Tensor, dim=0):
        out = Tensor()
        param = {
            "concatDim":dim,
        }
        node = Node('Concat', param, [x, y], [out])
        get_default_net().push_node(node)
        return out

    @staticmethod
    def gather(input_tensor: Tensor, index: Tensor, axis=0, batch_dims=0):
        out = Tensor()
        param = {
            "axis":axis,
            "batchDims":batch_dims
        }
        node = Node('Gather', param, [input_tensor, index], [out])
        get_default_net().push_node(node)
        return out

    class Embedding:
        def __init__(self, prefix: str):
            self.weight = Tensor(prefix)
            get_default_net().push_weight_key(prefix)
            
        def __call__(self, index: Tensor, axis=0, batch_dims=0):
            return Ops.gather(self.weight, index, axis, batch_dims)
