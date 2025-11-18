# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from typing import Optional
from enum import Enum
from atb_llm.nn.network import Tensor, Node, get_default_net


CALC_TYPE = 'calcType'


class MaskType(Enum):
    UNDEFINED = 1
    NORM = 2
    NORM_COMPRESS = 3
    ALIBI = 4
    ALIBI_COMPRESS = 5
    ALIBI_COMPRESS_LEFT_ALIGN = 5
    ALIBI_COMPRESS_SQRT = 6
    SPEC = 7


def check_attention_type(
        k: Optional[Tensor] = None,
        v: Optional[Tensor] = None,
        k_cache: Optional[Tensor] = None,
        v_cache: Optional[Tensor] = None,
) -> bool:
    if k is not None and k_cache is not None:
        raise ValueError
    if v is not None and v_cache is not None:
        raise ValueError
    is_prefill = None
    if k is not None and v is not None:
        is_prefill = True
    elif k_cache is not None and v_cache is not None:
        is_prefill = False
    if is_prefill is None:
        raise ValueError
    return is_prefill


def paged_attention(
        q: Tensor,
        k: Optional[Tensor] = None,
        v: Optional[Tensor] = None,
        k_cache: Optional[Tensor] = None,
        v_cache: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        mask_type: MaskType = MaskType.UNDEFINED,
        block_table: Optional[Tensor] = None,
        head_num: int = 0,
        kv_head_num: Optional[int] = None,
        q_lens: Optional[Tensor] = None,
        kv_lens: Optional[Tensor] = None,
        slopes: Optional[Tensor] = None,
        q_scale: float = 1.0,
        qk_scale: float = 1.0,
        high_precision: bool = False
) -> Tensor:
    out = Tensor()
    is_prefill = check_attention_type(k, v, k_cache, v_cache)
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
        'isTriuMask': 1,
        "kernelType": "KERNELTYPE_DEFAULT" if not high_precision else "KERNELTYPE_HIGH_PRECISION"
    }
    op_type = None
    in_tensors = [q]
    out_tensors = [out]
    if is_prefill:
        op_type = "SelfAttention"
        node_param[CALC_TYPE] = 'PA_ENCODER'
        in_tensors.extend([k, v])
        if mask is not None:
            in_tensors.append(mask)
        in_tensors.append(kv_lens)
        need_slopes = slopes is not None and (mask_type == MaskType.ALIBI or mask_type == MaskType.ALIBI_COMPRESS or \
            mask_type == MaskType.ALIBI_COMPRESS_SQRT)
        if need_slopes:
            in_tensors.append(slopes)
    else:
        if not high_precision:
            raise "pa decode not support low precision"
        if q_scale != 1:
            raise "pa decode not support qScale, only qkScale"
        mask_type_illegal = mask_type != MaskType.UNDEFINED and mask_type != MaskType.NORM and \
                    mask_type != MaskType.ALIBI and mask_type != MaskType.SPEC
        if mask_type_illegal:
            raise "pa decode only soppurt MaskType: NORM, ALIBI, SPEC"
        op_type = "PagedAttention"
        in_tensors.extend([k_cache, v_cache, block_table, kv_lens])
        if mask is not None:
            in_tensors.append(mask)
        if q_lens is not None:
            node_param[CALC_TYPE] = 'CALC_TYPE_SPEC'
            in_tensors.append(q_lens)
    node = Node(op_type, node_param, in_tensors, out_tensors)
    get_default_net().push_node(node)
    return out