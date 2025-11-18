# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
from ..network import Tensor, Node, get_default_net


def rope(q: Tensor, k: Tensor, cos_table: Tensor, sin_table: Tensor, seqlen: Tensor, rotary_coeff=2):
    q_embed = Tensor()
    k_embed = Tensor()
    node = Node("Rope", {'rotaryCoeff':rotary_coeff}, [q, k, cos_table, sin_table, seqlen], [q_embed, k_embed])
    get_default_net().push_node(node)
    return q_embed, k_embed