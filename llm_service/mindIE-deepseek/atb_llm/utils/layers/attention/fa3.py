# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import torch
from torch import nn

from atb_llm.utils.log.logging import logger
from atb_llm.utils.log.error_code import ErrorCode


class FA3(nn.Module):
    def __init__(self, scale: list, offset: list, weights, head_size):
        super().__init__()
        q_scale, k_scale, v_scale = scale
        q_offset, k_offset, v_offset = offset

        tp_world_size = weights.process_group.size()
        kv_head_num = k_scale.shape[0]
        gqa_size = q_scale.shape[0] // kv_head_num
        rank_size_q = q_scale.shape[0] // tp_world_size
        rank_size_kv = kv_head_num // tp_world_size
        rank = weights.process_group.rank()

        if rank_size_kv == 0:
            if tp_world_size % kv_head_num != 0:
                msg = f"`tp_world_size` must be divisible by `kv_head_num` (got `tp_world_size`: {tp_world_size} " \
                        f"and `kv_head_num`: {kv_head_num}"
                logger.error(msg, ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
                raise ValueError(msg)
            kv_repeat_size = tp_world_size // rank_size_kv
            rank_size_kv = 1
            rank_kv = rank // kv_repeat_size
        else:
            rank_kv = rank

        self.q_scale = nn.Parameter(
            q_scale.repeat(1, head_size)[rank * rank_size_q:(rank + 1) * rank_size_q], requires_grad=False)
        self.k_scale = nn.Parameter(
            k_scale.repeat(1, head_size)[rank_kv * rank_size_kv:(rank_kv + 1) * rank_size_kv],
            requires_grad=False)
        self.v_scale = nn.Parameter(
            v_scale.repeat(1, head_size)[rank_kv * rank_size_kv:(rank_kv + 1) * rank_size_kv],
            requires_grad=False)
        self.q_offset = nn.Parameter(
            q_offset.repeat(1, head_size)[rank * rank_size_q:(rank + 1) * rank_size_q].to(torch.int8),
            requires_grad=False)
        self.kv_offset = nn.Parameter(
            k_offset.repeat(1, head_size)[rank_kv * rank_size_kv:(rank_kv + 1) * rank_size_kv].to(torch.int8),
            requires_grad=False)
        
        fa3_k_scale, fa3_v_scale = k_scale.repeat(1, gqa_size).view(-1, 1), v_scale.repeat(1, gqa_size).view(-1, 1)
        self.qk_scale = nn.Parameter(
            torch.squeeze(q_scale * fa3_k_scale)[rank * rank_size_q:(rank + 1) * rank_size_q].to(torch.float),
            requires_grad=False)
        self.fa3_v_scale = nn.Parameter(
            torch.squeeze(fa3_v_scale).contiguous()[rank * rank_size_q:(rank + 1) * rank_size_q].to(torch.float),
            requires_grad=False)
        self.fa3_offset = nn.Parameter(
            torch.zeros(q_scale.shape[0], dtype=torch.int32)[rank * rank_size_q:(rank + 1) * rank_size_q],
            requires_grad=False)

    @classmethod
    def load(cls, prefix_q, prefix_k, prefix_v, weights, head_size):
        q_scale = weights.get_tensor(f"{prefix_q}.scale")
        q_offset = weights.get_tensor(f"{prefix_q}.offset")
        k_scale = weights.get_tensor(f"{prefix_k}.scale")
        k_offset = weights.get_tensor(f"{prefix_k}.offset")
        v_scale = weights.get_tensor(f"{prefix_v}.scale")
        v_offset = weights.get_tensor(f"{prefix_v}.offset")
        return cls([q_scale, k_scale, v_scale], [q_offset, k_offset, v_offset], weights, head_size)