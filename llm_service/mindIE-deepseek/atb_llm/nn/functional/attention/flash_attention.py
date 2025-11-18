# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from typing import Optional
from atb_llm.nn.network import Tensor, Node, get_default_net
from atb_llm.nn.functional.attention.paged_attention import CALC_TYPE, MaskType


def flash_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    mask: Optional[Tensor] = None,
    mask_type: MaskType = MaskType.UNDEFINED,
    head_num: int = 0,
    kv_head_num: Optional[int] = None,
    token_offset: Optional[Tensor] = None,
    seq_lens: Optional[Tensor] = None,
    layer_id: Optional[Tensor] = None,
    slopes: Optional[Tensor] = None,
    q_scale: float = 1.0,
    qk_scale: float = 1.0,
    high_precision: bool = True
) -> Tensor:
    out = Tensor()
    kv_head_num = head_num if kv_head_num is None else kv_head_num
    node_param = {
        'headNum': head_num,
        'kvHeadNum': kv_head_num,
        'batchRunStatusEnable': False,
        'qScale': q_scale,
        'qkScale': qk_scale,
        'maskType': f"MASK_TYPE_{mask_type.name}",
        'clampType': "CLAMP_TYPE_UNDEFINED",
        'clampMin': 0,
        'clampMax': 0,
        'isTriuMask': 0,
        "kernelType": "KERNELTYPE_HIGH_PRECISION"
    }
    op_type = "SelfAttention"
    in_tensors = [q]
    out_tensors = [out]
    in_tensors.extend([k, v, k_cache, v_cache])
    if mask is not None:
        in_tensors.append(mask)
    in_tensors.extend([token_offset, seq_lens, layer_id])
    need_slopes = slopes is not None and (mask_type == MaskType.ALIBI or mask_type == MaskType.ALIBI_COMPRESS or \
        mask_type == MaskType.ALIBI_COMPRESS_SQRT)
    if need_slopes:
        in_tensors.append(slopes)
    node = Node(op_type, node_param, in_tensors, out_tensors)
    get_default_net().push_node(node)
    return out