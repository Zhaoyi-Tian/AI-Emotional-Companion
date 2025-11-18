# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from typing import List

import pandas as pd
import numpy as np
import torch

from atb_llm.utils.env import ENV
from atb_llm.utils.log import logger, print_log
from atb_llm.utils import file_utils
from .batch import Batch


def next_token_chooser(logits: torch.Tensor):
    return torch.argmax(logits, dim=-1)


def is_pp_activated(model):
    if model.mapping is not None and model.mapping.has_pp():
        return True
    return False


def partition_data(model,
        dp_rank_ids: torch.Tensor,
        dp_rank_ids_per_token: torch.Tensor,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        is_prefill: bool,
        num_blocks: int,
        block_size: int,
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        input_lengths: torch.Tensor):
    cur_dp_rank_id_per_token_mask = dp_rank_ids_per_token == model.mapping.attn_dp.rank
    cur_dp_rank_id_mask = dp_rank_ids == model.mapping.attn_dp.rank
    shard_input_ids = input_ids[cur_dp_rank_id_per_token_mask]
    if shard_input_ids.numel() == 0:
        # dp组未分配数据，构造陪跑数据
        shard_input_ids = torch.tensor([1], dtype=torch.int64)
        shard_position_ids = torch.tensor([0], dtype=torch.int32)
        is_prefill_with_dp = is_prefill
        shard_block_tables = torch.tensor([[num_blocks - 1]], dtype=torch.int32)
        shard_slots = torch.tensor([(num_blocks - 1) * block_size], dtype=torch.int32)
        shard_input_lengths = torch.tensor([1], dtype=torch.int32)
        shard_max_seq_len = 1
    else:
        # 获取当前dp组的数据
        shard_position_ids = position_ids[cur_dp_rank_id_per_token_mask]
        is_prefill_with_dp = is_prefill
        shard_block_tables = block_tables[cur_dp_rank_id_mask]
        shard_slots = slots[cur_dp_rank_id_per_token_mask]
        shard_input_lengths = input_lengths[cur_dp_rank_id_mask]
        shard_max_seq_len = max(shard_input_lengths.tolist())

    # dp数据合并在模型侧完成，lm_head_indices需包含所有dp组，同时去除陪跑数据
    token_size_per_request = input_lengths if is_prefill else torch.ones_like(input_lengths)

    lm_head_indices_with_dp = torch.zeros_like(input_lengths)
    skip_dummy_data = torch.zeros_like(input_lengths)
    total_num_request = 0
    for i in range(model.mapping.attn_dp.group_size):
        cur_shard_input_lengths = token_size_per_request[dp_rank_ids == i]
        cur_num_request = cur_shard_input_lengths.shape[0]
        if cur_num_request == 0:
            skip_dummy_data[total_num_request:] += 1
        else:
            lm_head_indices_with_dp[total_num_request:total_num_request + cur_num_request] = cur_shard_input_lengths
        total_num_request += cur_num_request
    lm_head_indices_with_dp = lm_head_indices_with_dp.cumsum(0) - 1 + skip_dummy_data

    _, sorted_indices = torch.sort(dp_rank_ids, stable=True)
    reverse_indices = torch.argsort(sorted_indices, stable=True)
    lm_head_indices_with_dp = lm_head_indices_with_dp[reverse_indices]

    positional_args = (
        shard_input_ids, shard_position_ids, is_prefill_with_dp,
        shard_block_tables, shard_slots, shard_input_lengths, shard_max_seq_len, lm_head_indices_with_dp
    )

    return positional_args


def gather_dp_data(model, dp_rank_ids_per_token):
    # attn dp + mlp tp场景下，构造额外输入用于tp前收集所有dp组的输入，dp前进行数据切分
    token_size_per_dp_group = torch.bincount(dp_rank_ids_per_token, minlength=model.mapping.attn_dp.group_size)
    token_size_per_dp_group = torch.where(token_size_per_dp_group == 0, 1, token_size_per_dp_group)

    # 用于dp前数据切分：从所有请求按dp组排列后的token index中，选取当前dp组的token index，包含陪跑数据
    start_indices = torch.cumsum(token_size_per_dp_group, dim=0) - token_size_per_dp_group
    end_indices = torch.cumsum(token_size_per_dp_group, dim=0)
    shard_effective_token_indices = torch.arange(
        start_indices[model.mapping.attn_dp.rank], end_indices[model.mapping.attn_dp.rank],
        dtype=torch.int64
    )

    max_token_size_per_dp_group = token_size_per_dp_group.max().item()
    skip_padding_token_indices = torch.arange(
        model.mapping.attn_dp.group_size * max_token_size_per_dp_group, dtype=torch.int64
    ).view(model.mapping.attn_dp.group_size, max_token_size_per_dp_group)
    token_offset_per_dp_group = torch.arange(
        0, model.mapping.attn_dp.group_size * max_token_size_per_dp_group, step=max_token_size_per_dp_group,
        dtype=torch.int64).unsqueeze(1)
    token_index_with_padding = skip_padding_token_indices - token_offset_per_dp_group
    padding_mask = token_index_with_padding >= token_size_per_dp_group.unsqueeze(1)

    # 用于tp前数据汇总：包含padding token的token index（每个dp组的请求统一padding到所有请求的最大的输入长度，padding token index使用0表示）
    token_index_with_padding = token_index_with_padding[model.mapping.attn_dp.rank]
    token_index_with_padding = torch.where(padding_mask[model.mapping.attn_dp.rank], 0, token_index_with_padding)

    # 用于跳过padding token的token index
    skip_padding_token_indices = skip_padding_token_indices[~padding_mask]

    moe_token_index_with_padding = torch.concat([
        torch.concat([torch.arange(j), torch.zeros(max_token_size_per_dp_group - j, dtype=torch.int32)]) \
            for j in token_size_per_dp_group], dim=0)
    moe_skip_padding_token_indices = torch.arange(
        token_size_per_dp_group[model.mapping.attn_dp.rank], dtype=torch.int32)

    return {
        "token_size_per_dp_group": token_size_per_dp_group,
        "sum_token_size_per_dp_group": token_size_per_dp_group.sum().tolist(),
        "shard_effective_token_indices": shard_effective_token_indices.npu(),
        "token_index_with_padding": token_index_with_padding.npu(),
        "skip_padding_token_indices": skip_padding_token_indices.npu(),
        "moe_token_index_with_padding": moe_token_index_with_padding.npu(),
        "moe_skip_padding_token_indices": moe_skip_padding_token_indices.npu(),
    }


def save_logits_if_needed(model, base_filename, logits_tensor: torch.Tensor | List[torch.Tensor]):
    ENV.update()
    if ENV.logits_save_enable:
        import os
        logits_save_filename = f"logits_{base_filename}.pth"
        logits_save_filepath = os.path.join(ENV.logits_save_folder, logits_save_filename)
        logits_save_filepath = file_utils.standardize_path(logits_save_filepath)
        file_utils.check_file_safety(logits_save_filepath, 'w', is_check_file_size=False)

        if not is_pp_activated(model) and model.rank == 0: 
            torch.save(logits_tensor.cpu(), logits_save_filepath)

        if is_pp_activated(model) and model.rank == model.world_size - 1:
            if (len(logits_tensor) == 0):
                raise AssertionError("save logits failed: no logits is found")
            total_logits = torch.concat(logits_tensor, dim=0)
            torch.save(total_logits.cpu(), logits_save_filepath)


def generate_token_from_microbatch(model, cache_manager, batches: List[Batch]):
    if len(batches) == 0:
        return 0
    finish_batch_num = 0
    next_token_list = []
    logits_list = []
    for batch in batches:
        input_ids = batch.batch_input_ids.npu()
        position_ids = batch.batch_position_ids.npu()
        is_prefill = batch.cu_seqlen_prefill is not None
        block_tables = batch.batch_block_tables.npu()
        kv_cache = cache_manager.kv_cache
        slots = batch.batch_slots_tables[batch.batch_slot_indices].npu()
        input_lengths = batch.context_length.npu()
        lm_head_indices = None if batch.lm_head_indices is None else batch.lm_head_indices.npu()
 
        logits = model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            is_prefill=is_prefill,
            block_tables=block_tables,
            kv_cache=kv_cache,
            slots=slots,
            input_lengths=input_lengths,
            max_seq_len=batch.max_s,
            lm_head_indices=lm_head_indices
        )

        if model.mapping.is_last_pp_rank():
            if is_prefill and logits.size(0) != batch.batch_num:
                if logits.size(0) != batch.lm_head_indices[-1] + 1:
                    logger.error(
                        f"prefill logits is invalid, batch num: {batch.batch_num}, " \
                        f"total token: {int(batch.lm_head_indices[-1] + 1)}, but logits shape is: {logits.shape}")
                    raise AssertionError
                logits = logits[batch.lm_head_indices]
            next_token = next_token_chooser(logits)
            ENV.update()
            if ENV.logits_save_enable:
                logits_list.append(logits.cpu())
            next_token_list.append(next_token.cpu())

    # broadcast next token to pp group
    if model.mapping.is_last_pp_rank():
        total_next_token = torch.concat(next_token_list, dim=0).int()
    else:
        total_batch_size = sum([batch.batch_num for batch in batches])
        total_next_token = torch.zeros([total_batch_size], dtype=torch.int32)
    torch.distributed.broadcast(tensor=total_next_token, 
                                src=model.mapping.pp.rank_per_group[-1][-1],
                                group=model.mapping.pp.pp_bcast_group,
                                async_op=False)
    
    if ENV.modeltest_dataset_specified:
        save_logits_if_needed(model, str(len(batches[0].req_list[0].out_token_list)), logits_list)
 
    total_token_list = total_next_token.tolist()
    start = 0
    end = 0
    for batch in batches:
        end += batch.batch_num
        for i, req in enumerate(batch.req_list):
            req.out_token_list.append(total_token_list[start + i])
        batch.batch_input_ids = total_next_token[start:end].to(torch.int64).npu()
        batch.batch_position_ids = batch.context_length.clone().to(torch.long)
        if batch.cu_seqlen_prefill is not None:
            batch.batch_slot_indices = batch.batch_slot_indices[batch.lm_head_indices]
            batch.cu_seqlen_prefill = None
            batch.lm_head_indices = None
 
        batch.batch_slot_indices += 1
        batch.context_length += 1
        batch.max_s += 1
        finish_batch_num += batch.filter(model.postprocessor, cache_manager)
        start = end
    return finish_batch_num


def generate_token(model, cache_manager, batch: Batch):
    input_ids = batch.batch_input_ids
    position_ids = batch.batch_position_ids
    is_prefill = batch.cu_seqlen_prefill is not None
    block_tables = batch.batch_block_tables
    kv_cache = cache_manager.kv_cache
    slots = batch.batch_slots_tables[batch.batch_slot_indices]
    dp_rank_ids = batch.batch_dp_rank_ids
    input_lengths = batch.context_length
    max_seq_len = batch.max_s
    lm_head_indices = None if batch.lm_head_indices is None else batch.lm_head_indices

    positional_args = (
        input_ids, position_ids, is_prefill, cache_manager.num_blocks, cache_manager.block_size,
        block_tables, slots, input_lengths
    )
    kwargs = {
        "adapter_ids": batch.batch_adapter_ids,
        "max_out_len": cache_manager.output_max_length,
    }

    if model.mapping.has_dp():
        if dp_rank_ids is None:
            raise ValueError("dp_rank_ids is not given when data parallel size > 1.")
        if is_prefill:
            dp_rank_ids_per_token = torch.repeat_interleave(dp_rank_ids, input_lengths)
        else:
            dp_rank_ids_per_token = dp_rank_ids
        res = partition_data(model, dp_rank_ids, dp_rank_ids_per_token, *positional_args)
        input_ids, position_ids, is_prefill, block_tables, slots, input_lengths, max_seq_len, lm_head_indices = res

        additional_kwargs = gather_dp_data(model, dp_rank_ids_per_token)
        kwargs.update(additional_kwargs)

    if batch.batch_cross_attn_mask is not None: # for llama3.2
        if is_prefill:
            cross_slots_mapping = batch.batch_slots_tables[batch.batch_cross_attn_slot_indices].npu()
            multi_modal_inputs = batch.batch_multi_modal
        else:
            cross_slots_mapping = None
            multi_modal_inputs = None
        cross_attention_mask = batch.batch_cross_attn_mask.npu()
        cross_context_lens = batch.batch_cross_attn_context_length.npu()
        full_text_row_masked_out_mask = batch.batch_cross_attn_full_row_mask.npu()
        kwargs.update(dict(multi_modal_inputs=multi_modal_inputs,
                        cross_slots_mapping=cross_slots_mapping,
                        cross_attention_mask=cross_attention_mask,
                        cross_context_lens=cross_context_lens,
                        full_text_row_masked_out_mask=full_text_row_masked_out_mask,))
                        
    logits = model.forward(
        input_ids=input_ids.npu(),
        position_ids=position_ids.npu(),
        is_prefill=is_prefill,
        block_tables=block_tables.npu(),
        kv_cache=kv_cache,
        slots=slots.npu(),
        input_lengths=input_lengths.npu(),
        max_seq_len=max_seq_len,
        lm_head_indices=lm_head_indices.npu() if lm_head_indices is not None else None,
        **kwargs
    )

    if batch.cu_seqlen_prefill is not None and logits.size(0) != batch.batch_num:
        if logits.size(0) != batch.lm_head_indices[-1] + 1:
            logger.error(f"prefill logits is invalid, batch num: {batch.batch_num}," +
                         f" total token: {int(batch.lm_head_indices[-1] + 1)}, but logits shape is: {logits.shape}")
            raise AssertionError
        logits = logits[batch.lm_head_indices]

    if ENV.modeltest_dataset_specified:
        save_logits_if_needed(model, str(len(batch.req_list[0].out_token_list)), logits)

    if logits.size(1) > 1 :
        next_token = next_token_chooser(logits)
    elif logits.dim() == 2 and logits.size(0) == 1:
        next_token = logits.squeeze().unsqueeze(0)
    else:
        next_token = logits.squeeze()
    next_token_list = next_token.tolist()

    for i, req in enumerate(batch.req_list):
        req.out_token_list.append(next_token_list[i])

    batch.batch_input_ids = next_token.to(torch.int64).cpu()
    if batch.multi_context_length is not None:
        batch.batch_position_ids = batch.multi_context_length.clone().to(torch.long)
        batch.multi_context_length += 1
    else:
        batch.batch_position_ids = batch.context_length.clone().to(torch.long)
    if batch.cu_seqlen_prefill is not None:
        if batch.batch_cross_attn_mask is not None: # for llama3.2
            batch.batch_cross_attn_mask = batch.batch_cross_attn_mask[batch.lm_head_indices]
            batch.batch_cross_attn_context_length = batch.batch_cross_attn_context_length[batch.lm_head_indices]
            batch.batch_cross_attn_full_row_mask = batch.batch_cross_attn_full_row_mask[batch.lm_head_indices]
        batch.batch_slot_indices = batch.batch_slot_indices[batch.lm_head_indices]
        batch.cu_seqlen_prefill = None
        batch.lm_head_indices = None

    batch.batch_slot_indices += 1
    batch.context_length += 1
    batch.max_s += 1

    return batch.filter(model.postprocessor, cache_manager)


def generate_token_with_clocking(model, cache_manager, input_batch: Batch | List[Batch]):
    time_used = None
    if ENV.benchmark_enable:
        import time
        torch.npu.synchronize()
        time_start = time.time()
        if is_pp_activated(model):
            req_finished = generate_token_from_microbatch(model, cache_manager, input_batch)
        else:
            req_finished = generate_token(model, cache_manager, input_batch)
        torch.npu.synchronize()
        time_end = time.time()
        time_used = time_end - time_start
    else:
        if is_pp_activated(model):
            req_finished = generate_token_from_microbatch(model, cache_manager, input_batch)
        else:
            req_finished = generate_token(model, cache_manager, input_batch)

    return req_finished, time_used


def generate_req(req_list, model, max_batch_size, max_prefill_tokens, cache_manager):
    req_num = len(req_list)
    print_log(model.rank, logger.info, f"------total req num: {req_num}, infer start--------")

    req_idx = 0
    total_req_finished = 0

    generate_batch_size_per_dp_group = [0] * model.mapping.attn_dp.group_size
    total_generate_batch_size = 0
    generate_batches = []

    prefill_benchmark_timelist = []
    decoder_benchmark_timelist = []

    while total_req_finished < req_num:
        do_generate = True
        # 仍有新请求待处理，且decode阶段的请求未满
        if req_idx < req_num and min(generate_batch_size_per_dp_group) < max_batch_size:
            prefill_start = req_idx
            free_block_per_dp_group = [cache_manager.get_free_block_num(i)
                                       for i in range(model.mapping.attn_dp.group_size)]
            total_need_blocks_per_dp_group = [0] * model.mapping.attn_dp.group_size
            total_prefill_token_per_dp_group = [0] * model.mapping.attn_dp.group_size
            prefill_batch_size_per_dp_group = [0] * model.mapping.attn_dp.group_size
            total_prefill_batch_size = 0

            # 请求分配给最空闲的dp组
            dp_rank = np.argmax(np.array(free_block_per_dp_group) - np.array(total_need_blocks_per_dp_group))
            while generate_batch_size_per_dp_group[dp_rank] + prefill_batch_size_per_dp_group[dp_rank] < max_batch_size:
                if req_idx >= req_num:
                    break
                cur_need_blocks = req_list[req_idx].need_blocks
                cur_context_len = req_list[req_idx].input_length
                if total_need_blocks_per_dp_group[dp_rank] + cur_need_blocks > free_block_per_dp_group[dp_rank]:
                    raise ValueError(f"req: {req_idx} out of memory, need block:" +
                                     f"{total_need_blocks_per_dp_group[dp_rank] + cur_need_blocks} is more than " +
                                     f"free block {free_block_per_dp_group[dp_rank]}")
                if cur_context_len > max_prefill_tokens:
                    raise ValueError(f"req {req_idx}'s  input length is {cur_context_len}, which is longer than " +
                                     f"max_prefill_tokens {max_prefill_tokens}")
                if total_prefill_token_per_dp_group[dp_rank] + cur_context_len > max_prefill_tokens:
                    do_generate = False
                    break
                req_list[req_idx].dp_rank = dp_rank
                total_need_blocks_per_dp_group[dp_rank] += cur_need_blocks
                total_prefill_token_per_dp_group[dp_rank] += cur_context_len
                prefill_batch_size_per_dp_group[dp_rank] += 1
                total_prefill_batch_size += 1
                req_idx += 1
                dp_rank = np.argmax(np.array(free_block_per_dp_group) - np.array(total_need_blocks_per_dp_group))

            if total_prefill_batch_size > 0:
                if is_pp_activated(model):
                    start = prefill_start
                    batch_list = []
                    while start < prefill_start + total_prefill_batch_size:
                        if start + model.mapping.pp.microbatch_size > prefill_start + total_prefill_batch_size:
                            end = prefill_start + total_prefill_batch_size
                        else:
                            end = start + model.mapping.pp.microbatch_size
                        batch = Batch(req_list[start:end])

                        cache_manager.allocate(batch)
                        batch_list.append(batch)
                        start = end
                    req_finished, prefill_time = generate_token_with_clocking(model, cache_manager, batch_list)
                    if ENV.benchmark_enable:
                        prefill_benchmark_timelist.append(prefill_time)

                    prefill_summation = 0
                    for i in range(len(batch_list) - 1, -1, -1): # count down batch_list
                        prefill_summation += batch_list[i].batch_num
                        if batch_list[i].batch_num == 0:
                            del batch_list[i]
 
                    if req_finished != (total_prefill_batch_size - prefill_summation):
                        logger.error("batch filter error")
                        raise AssertionError
 
                    if prefill_summation > 0:
                        for batch in batch_list:
                            generate_batches.append(batch)
                            total_generate_batch_size += batch.batch_num

                else:
                    batch = Batch(req_list[prefill_start:prefill_start + total_prefill_batch_size])
                    cache_manager.allocate(batch)
                    req_finished, prefill_time = generate_token_with_clocking(model, cache_manager, batch)
                    if ENV.benchmark_enable:
                        prefill_benchmark_timelist.append(prefill_time)

                    if req_finished != (total_prefill_batch_size - batch.batch_num):
                        raise AssertionError(
                            "Batch filter error: [Prefill] the total number of requests processed "
                            "does not equal the number of requests left + the number of requests completed.")

                    if batch.batch_num > 0:
                        generate_batches.append(batch)
                        total_generate_batch_size += batch.batch_num

                if req_finished > 0:
                    do_generate = False
                    total_req_finished += req_finished

        if do_generate:
            if not is_pp_activated(model) and len(generate_batches) > 1:
                Batch.concatenate(generate_batches)
                if total_generate_batch_size != generate_batches[0].batch_num:
                    raise AssertionError(f"Batch concatenate error, expect batchnum: {total_generate_batch_size}, "
                                         f"in fact: {generate_batches[0].batch_num}")

            batch_used = generate_batches if is_pp_activated(model) else generate_batches[0]
            req_finished, decode_time = generate_token_with_clocking(model, cache_manager, batch_used)
            if ENV.benchmark_enable:
                decoder_benchmark_timelist.append(decode_time)
            
            generate_summation = 0
            if is_pp_activated(model):
                for i in range(len(generate_batches) - 1, -1, -1): # count down
                    generate_summation += generate_batches[i].batch_num
                    if generate_batches[i].batch_num == 0:
                        del generate_batches[i]
                        continue
                    for req in generate_batches[i].req_list:
                        generate_batch_size_per_dp_group[req.dp_rank] += 1
            else:
                generate_summation = generate_batches[0].batch_num
                if generate_summation == 0:
                    del generate_batches[0]
           
            if req_finished != (total_generate_batch_size - generate_summation):
                raise AssertionError("Batch filter error: [Decode] the total number of requests processed does not "
                                    "equal the number of requests left + the number of requests completed.")
            total_generate_batch_size = generate_summation
            total_req_finished += req_finished
            
            generate_batch_size_per_dp_group = [0] * model.mapping.attn_dp.group_size
            if len(generate_batches) > 0:
                for req in generate_batches[0].req_list:
                    generate_batch_size_per_dp_group[req.dp_rank] += 1

    if ENV.benchmark_enable:
        prefill_time = sum(prefill_benchmark_timelist)
        e2e_time = sum(prefill_benchmark_timelist) + sum(decoder_benchmark_timelist)
        try:
            decode_token_time = sum(decoder_benchmark_timelist) / (model.postprocessor.max_new_tokens - 1)
        except ZeroDivisionError:
            decode_token_time = 0

        logger.info(
            f"Prefill time: {prefill_time * 1000}ms, "
            f"Decode token time: {decode_token_time * 1000}ms, "
            f"E2E time: {e2e_time * 1000}ms")
        batch_size = len(req_list)
        input_len = req_list[0].input_length
        output_len = model.postprocessor.max_new_tokens
        prefill_token_times = ','.join(list(map(str, prefill_benchmark_timelist)))
        decode_token_times = ','.join(list(map(str, decoder_benchmark_timelist)))
        if model.rank == 0:
            import os
            benchmark_filepath = ENV.benchmark_filepath \
                if ENV.benchmark_filepath else './benchmark_result/benchmark.csv'
            benchmark_folder = os.path.dirname(benchmark_filepath)
            if benchmark_folder and not os.path.exists(benchmark_folder):
                os.makedirs(benchmark_folder)
            benchmark_filepath = file_utils.standardize_path(benchmark_filepath)
            file_utils.check_file_safety(benchmark_filepath, 'w')
            stat_data = {
                'batch_size': [batch_size],
                'input_seq_len': [input_len],
                'output_seq_len': [output_len],
                'e2e_time(ms)': [f'{e2e_time * 1000: .2f}'],
                'prefill_time(ms)': [f'{prefill_time * 1000: .2f}'],
                'decoder_token_time(ms)': [f'{decode_token_time * 1000: .2f}'],
                'prefill_count': [len(prefill_benchmark_timelist)],
                'prefill_token_times': [prefill_token_times],
                'decode_token_times': [decode_token_times],
            }
            df = pd.DataFrame(stat_data)
            df.to_csv(benchmark_filepath, index=False)
            logger.info('-------------------performance dumped------------------------')
            df = df.drop('prefill_token_times', axis=1)
            df = df.drop('decode_token_times', axis=1)
            logger.info(df.to_markdown(index=False))


def decode_token(req_list, tokenizer, skip_special_tokens=False):
    decode_res_list = []
    token_num_list = []
    request_id = 0
    token_num = 0
    for req in req_list:
        out_token = len(req.out_token_list)
        token_tensor = torch.tensor(req.out_token_list, dtype=torch.int64)
        if tokenizer is not None:
            decode_text = tokenizer.decode(token_tensor, skip_special_tokens)
            decode_res_list.append(decode_text)
        else:
            decode_res_list.append(token_tensor)
        token_num += out_token
        token_num_list.append((request_id, token_num))
        request_id += 1
    return decode_res_list, token_num_list
