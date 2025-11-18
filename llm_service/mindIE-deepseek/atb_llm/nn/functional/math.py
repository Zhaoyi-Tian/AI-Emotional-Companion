# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
from atb_llm.nn.network import Tensor, Node, get_default_net


def cos(x: Tensor):
    out = Tensor()
    param = {
        "elewiseType":"ELEWISE_COS",
    }
    node = Node('Elewise', param, [x], [out])
    get_default_net().push_node(node)
    return out


def sin(x: Tensor):
    out = Tensor()
    param = {
        "elewiseType":"ELEWISE_SIN",
    }
    node = Node('Elewise', param, [x], [out])
    get_default_net().push_node(node)
    return out


def neg(x: Tensor):
    out = Tensor()
    param = {
        "elewiseType":"ELEWISE_NEG",
    }
    node = Node('Elewise', param, [x], [out])
    get_default_net().push_node(node)
    return out


def logical_not(x: Tensor):
    out = Tensor()
    param = {
        "elewiseType":"ELEWISE_LOGICAL_NOT",
    }
    node = Node('Elewise', param, [x], [out])
    get_default_net().push_node(node)
    return out


def logical_or(x: Tensor, y: Tensor):
    out = Tensor()
    param = {
        "elewiseType":"ELEWISE_LOGICAL_OR",
    }
    node = Node('Elewise', param, [x, y], [out])
    get_default_net().push_node(node)
    return out


def logical_and(x: Tensor, y: Tensor):
    out = Tensor()
    param = {
        "elewiseType":"ELEWISE_LOGICAL_AND",
    }
    node = Node('Elewise', param, [x, y], [out])
    get_default_net().push_node(node)
    return out


def eq(x: Tensor, y: Tensor):
    out = Tensor()
    param = {
        "elewiseType":"ELEWISE_EQUAL",
    }
    node = Node('Elewise', param, [x, y], [out])
    get_default_net().push_node(node)
    return out
