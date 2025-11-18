# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from typing import List
import torch
from atb_llm.utils.log import logger
from .request import Request


class Batch:
    req_ids: List[int]
    req_list: List[Request]
    batch_num: int

    cu_seqlen_prefill: torch.Tensor
    batch_input_ids: torch.Tensor
    batch_adapter_ids: list
    batch_position_ids: torch.Tensor

    batch_block_tables: torch.Tensor
    batch_slots_tables: torch.Tensor
    batch_slot_indices: torch.Tensor

    batch_dp_rank_ids: torch.Tensor

    context_length: torch.Tensor
    max_s: int
    lm_head_indices: torch.Tensor

    def __init__(self, req_list: List[Request]):
        self.req_list = req_list
        self.batch_num = len(req_list)

        self.req_ids = [req.req_id for req in req_list]
        input_ids_list = []
        adapter_ids_list = []
        position_ids_list = []
        slot_indices_list = []
        dp_rank_list = []
        context_length_list = []
        multi_context_length_list = []
        self.max_s = 0
        slot_offset = 0
        self.multi_context_length = None

        # for llama3.2
        cross_atten_slot_indices_list = []
        cross_attn_mask_list = []
        cross_attn_context_length_list = []
        cross_attn_full_row_mask_list = []
        multi_modal_list = []
        num_vision_tokens = 0
        for req in self.req_list:
            context_length = req.input_ids.size(0)
            input_ids_list.append(req.input_ids)
            adapter_ids_list.append(req.adapter_id)
            position_ids = torch.arange(context_length, dtype=torch.long)

            slot_indices = position_ids + slot_offset
            slot_indices_list.append(slot_indices)
            context_length_list.append(context_length)
            self.max_s = max(self.max_s, context_length)
            if getattr(req, 'cross_attention_mask', None) is not None:   # for llama3.2
                cross_atten_slot_indices_list.append(
                    torch.arange(req.image_context_length, dtype=torch.long) + slot_offset
                )
                cross_attn_mask_list.append(req.cross_attention_mask)
                cross_attn_context_length_list.extend([req.image_context_length] * context_length)
                cross_attn_full_row_mask_list.append(req.full_text_row_masked_out_mask)
                multi_modal_list.append(req.multi_modal_inputs)
                num_vision_tokens = req.num_vision_tokens
            slot_offset += req.need_slots
            dp_rank_list.append(req.dp_rank)

            if getattr(req, "position_ids", None) is not None:
                position_ids = req.position_ids
            position_ids_list.append(position_ids)
            if getattr(req, "context_length", None) is not None:
                multi_context_length_list.append(req.context_length)
        if multi_context_length_list:
            self.multi_context_length = torch.tensor(multi_context_length_list, dtype=torch.int64)
        self.cu_seqlen_prefill = torch.tensor([1])
        self.batch_input_ids = torch.concat(input_ids_list, dim=0)
        self.batch_adapter_ids = adapter_ids_list
        self.batch_position_ids = torch.concat(position_ids_list, dim=0)
        self.batch_block_tables: None | torch.Tensor = None
        self.batch_slots_tables: None | torch.Tensor = None
        self.batch_slot_indices = torch.concat(slot_indices_list, dim=0)
        self.batch_dp_rank_ids = torch.tensor(dp_rank_list)
        self.context_length = torch.tensor(context_length_list, dtype=torch.int64)
        self.lm_head_indices = torch.cumsum(self.context_length, dim=0) - 1

        if cross_attn_mask_list: # for llama3.2
            self.batch_cross_attn_slot_indices = torch.concat(cross_atten_slot_indices_list, dim=0)

            self.batch_cross_attn_context_length = torch.tensor(cross_attn_context_length_list, dtype=torch.int64)
            self.batch_cross_attn_full_row_mask = torch.concat(cross_attn_full_row_mask_list, dim=0)
            cross_attention_mask = torch.concat(cross_attn_mask_list, dim=0)

            dtype = self.batch_cross_attn_full_row_mask.dtype
            cross_attention_mask = cross_attention_mask.repeat_interleave(num_vision_tokens, dim=-1)
            inverted_cross_attn_mask = (1.0 - cross_attention_mask).to(dtype)
            cross_attention_mask = inverted_cross_attn_mask.masked_fill(
                inverted_cross_attn_mask.to(torch.bool), torch.finfo(dtype).min
            )
            self.batch_cross_attn_mask = cross_attention_mask
            self.batch_multi_modal = multi_modal_list
        else:
            self.batch_cross_attn_mask = None

    @classmethod
    def concatenate(cls, batches: List["Batch"]):
        req_ids = []
        req_list = []
        batch_num = 0
        input_ids_list = [batch.batch_input_ids for batch in batches]

        adapter_ids_list = []
        for batch in batches:
            for adapter in batch.batch_adapter_ids:
                adapter_ids_list.append(adapter)
        position_ids_list = [batch.batch_position_ids for batch in batches]
        block_tables_list = []
        slots_tables_list = [batch.batch_slots_tables for batch in batches]
        slot_indices_list = []
        dp_rank_list = [batch.batch_dp_rank_ids for batch in batches]
        context_length_list = [batch.context_length for batch in batches]
        max_s = 0

        max_block = 0
        for batch in batches:
            req_ids.extend(batch.req_ids)
            req_list.extend(batch.req_list)
            batch_num += batch.batch_num
            max_s = max(max_s, batch.max_s)
            max_block = max(max_block, batch.batch_block_tables.size(1))

        slot_offset = 0
        for batch in batches:
            cur_block = batch.batch_block_tables.size(1)
            if cur_block < max_block:
                zero = torch.zeros(batch.batch_num, max_block - cur_block, dtype=torch.long)
                batch.batch_block_tables = torch.concat([batch.batch_block_tables, zero], dim=-1)
            block_tables_list.append(batch.batch_block_tables)
            slot_indices_list.append(batch.batch_slot_indices + slot_offset)
            slot_offset += batch.batch_slots_tables.size(0)

        batches[0].req_ids = req_ids
        batches[0].req_list = req_list
        batches[0].batch_num = batch_num
        batches[0].batch_input_ids = torch.concat(input_ids_list, dim=0)
        batches[0].batch_adapter_ids = adapter_ids_list
        batches[0].batch_position_ids = torch.concat(position_ids_list, dim=0)
        batches[0].batch_block_tables = torch.concat(block_tables_list, dim=0)
        batches[0].batch_slots_tables = torch.concat(slots_tables_list, dim=0)
        batches[0].batch_slot_indices = torch.concat(slot_indices_list, dim=0)
        batches[0].batch_dp_rank_ids = torch.concat(dp_rank_list, dim=0)
        batches[0].context_length = torch.concat(context_length_list, dim=0)
        batches[0].max_s = max_s
        if batches[0].multi_context_length is not None:
            multi_context_length_list = [batch.multi_context_length for batch in batches]
            batches[0].multi_context_length = torch.concat(multi_context_length_list, dim=0)

        if batches[0].batch_cross_attn_mask is not None: # for llama3.2
            batches[0].batch_cross_attn_mask = torch.concat(
                [batch.batch_cross_attn_mask for batch in batches], dim=0
            )
            batches[0].batch_cross_attn_context_length = torch.concat(
                [batch.batch_cross_attn_context_length for batch in batches], dim=0
            )
            batches[0].batch_cross_attn_full_row_mask = torch.concat(
                [batch.batch_cross_attn_full_row_mask for batch in batches], dim=0
            )

        while len(batches) > 1:
            del batches[1]

    def filter(self, postprocessor, cache_manager):
        if self.batch_num == 0:
            logger.error("batch.batch_num is 0")
            raise AssertionError

        finish_num = 0
        finish_list = []

        for i, req in enumerate(self.req_list):
            if (postprocessor.stopping_criteria(req.out_token_list)) or \
                    len(req.out_token_list) >= postprocessor.max_new_tokens:
                cache_manager.free(req)
                finish_num += 1
                finish_list.append(i)

        if finish_num == 0:
            return 0

        batch_mask = torch.ones(self.batch_num, dtype=torch.int64)
        batch_mask[finish_list] = 0
        remain_batch = batch_mask.nonzero().flatten()

        self.batch_num -= finish_num
        if self.batch_num == 0:
            return finish_num

        self.batch_input_ids = self.batch_input_ids[remain_batch]
        self.batch_position_ids = self.batch_position_ids[remain_batch]
        self.batch_block_tables = self.batch_block_tables[remain_batch]
        self.batch_dp_rank_ids = self.batch_dp_rank_ids[remain_batch]
        context_length = self.context_length[remain_batch]
        self.max_s = int(context_length.max())

        if self.batch_cross_attn_mask is not None: # for llama3.2
            self.batch_cross_attn_mask = self.batch_cross_attn_mask[remain_batch]
            self.batch_cross_attn_context_length = \
                self.batch_cross_attn_context_length[remain_batch]
            self.batch_cross_attn_full_row_mask = \
                self.batch_cross_attn_full_row_mask[remain_batch]

        req_ids = []
        req_list = []
        adapter_ids_list = []
        slots_tables_list = []
        slot_indices_list = []

        slot_offset = 0
        for i, req in enumerate(self.req_list):
            if i in finish_list:
                continue

            req_ids.append(req.req_id)
            req_list.append(req)
            adapter_ids_list.append(req.adapter_id)
            slots_tables_list.append(req.slot_tables)
            slot_indices_list.append(int(self.context_length[i]) - 1 + slot_offset)
            slot_offset += req.need_slots

        self.req_ids = req_ids
        self.req_list = req_list
        self.batch_adapter_ids = adapter_ids_list
        self.batch_slots_tables = torch.concat(slots_tables_list, dim=0)
        self.batch_slot_indices = torch.tensor(slot_indices_list, dtype=torch.long)
        self.context_length = context_length
        if self.multi_context_length is not None:
            self.multi_context_length = self.multi_context_length[remain_batch]

        return finish_num
