# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import math
import torch

from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.utils.env import ENV
from examples.convert.model_slim.get_razor_attention_wins import get_global_wins


class CacheConfig:
    def __init__(self, num_blocks=1024, block_size=128, input_max_length=2048, output_max_length=128, batch_size=1,
                 rank=0, world_size=1):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.input_max_length = input_max_length
        self.output_max_length = output_max_length
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size


class ModelConfig:
    def __init__(self, num_heads, num_kv_heads, num_kv_heads_origin, k_head_size, v_head_size,
                num_layers, device, dtype, soc_info, kv_quant_type, fa_quant_type=None,
                mapping=None, cla_share_factor=1, model_type=None):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_kv_heads_origin = num_kv_heads_origin
        self.head_size = k_head_size
        self.k_head_size = k_head_size
        self.v_head_size = v_head_size
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        self.soc_info = soc_info
        self.kv_quant_type = kv_quant_type
        self.fa_quant_type = fa_quant_type
        self.mapping = mapping
        self.cla_share_factor = cla_share_factor
        self.model_type = model_type

    def __repr__(self):
        return (
                "ModelConfig("
                + f"num_heads={self.num_heads}, "
                + f"num_kv_heads={self.num_kv_heads}, "
                + f"num_kv_heads_origin={self.num_kv_heads_origin}, "
                + f"head_size={self.head_size}, "
                + f"k_head_size={self.k_head_size}, "
                + f"v_head_size={self.v_head_size}, "
                + f"num_layers={self.num_layers}, "
                + f"device={self.device}, "
                + f"dtype={self.dtype}, "
                + f"soc_info={self.soc_info}, "
                + f"kv_quant_type={self.kv_quant_type}, "
                + f"fa_quant_type={self.fa_quant_type}, "
                + f"mapping={self.mapping}, "
                + f"cla_share_factor={self.cla_share_factor}, "
                + f"model_type={self.model_type})"
        )


class CacheManager:
    def __init__(self, cache_config, model_config):
        self.block_size = cache_config.block_size
        self.num_blocks = cache_config.num_blocks
        self.new_num_blocks = self.num_blocks
        self.input_max_length = cache_config.input_max_length
        self.output_max_length = cache_config.output_max_length
        self.batch_size = cache_config.batch_size
        self.rank = cache_config.rank
        self.world_size = cache_config.world_size
        
        self.compress_head_enable = ENV.compress_head_enable
        self.compress_head_rope = ENV.compress_head_rope
        self.num_heads = 1 if self.compress_head_enable else model_config.num_kv_heads
        self.k_head_size = model_config.k_head_size
        self.v_head_size = model_config.v_head_size
        self.num_layers = model_config.num_layers
        self.layer_list = model_config.mapping.pp_layers(self.num_layers) if model_config.mapping is not None \
            else list(range(self.num_layers))
        self.v_cache_share_fractor = model_config.cla_share_factor
        self.device = model_config.device
        self.dtype = torch.int8 if model_config.kv_quant_type is not None or \
            model_config.fa_quant_type is not None else model_config.dtype
        self.soc_info = model_config.soc_info
        self.model_type = model_config.model_type
        self.attn_dp_size = model_config.mapping.attn_dp.group_size if model_config.mapping is not None else 1
        self.enable_data_parallel = model_config.mapping.has_dp() if model_config.mapping is not None else False
        if self.enable_data_parallel:
            # dp场景下多申请一个block用于存放陪跑数据
            self.num_blocks += 1
            self.new_num_blocks += 1

        if self.v_cache_share_fractor < 1:
            logger.error("cross layer attention param error, v cache should be shared by at least one layer")
            raise AssertionError
        mem_need = self.num_blocks * self.block_size * self.num_heads * (self.k_head_size * self.num_layers \
                    + self.v_head_size * self.num_layers / self.v_cache_share_fractor) * \
                    self.get_dtype_size(self.dtype) / 1024 / 1024 / 1024
        logger.info(f"kv cache will allocate {mem_need}GB memory")

        if self.compress_head_enable:
            if self.compress_head_rope:
                if self.model_type is not None and self.model_type == "llama" and self.num_layers == 80:
                    head_dict = get_global_wins(self.model_type, self.num_layers)
                    inductive_head = head_dict.get("prefix_matching")
                    copying_head = head_dict.get("copying")
                    first_sink = 40
                    last_sink = max(4000, self.input_max_length // 5)
                    self.new_layers_num_blocks = []
                    kv_tp_size = min(cache_config.world_size, model_config.num_kv_heads_origin)
                    for layer_idx in range(self.num_layers):
                        global_need_block = 0
                        for head_idx in range(model_config.num_kv_heads):
                            cur_head_idx = head_idx + self.rank * kv_tp_size // \
                                self.world_size * model_config.num_kv_heads
                            is_inductive_head = layer_idx in inductive_head \
                                and cur_head_idx in inductive_head.get(layer_idx)
                            is_copying_head = layer_idx in copying_head and cur_head_idx in copying_head[layer_idx]
                            if (is_inductive_head or is_copying_head) or \
                                (self.input_max_length - first_sink - last_sink - 1 <= 0):
                                temp_length = self.input_max_length + self.output_max_length
                            else:
                                temp_length = first_sink + 1 + last_sink + self.output_max_length

                            need_block = math.ceil(temp_length / self.block_size)
                            global_need_block = global_need_block + need_block
                        self.new_layers_num_blocks.append(global_need_block)
                else:
                    self.new_layers_num_blocks = self.new_num_blocks * model_config.num_kv_heads
            else:
                wins = [
                    105, 125, 148, 176, 210, 250, 297, 353, 420, 500, 595, 707, 841, 1001, 1190, 1415, 1683, 2002, 2381,
                    2831, 3367, 4004, 4762, 5663, 6734, 8008, 9524, 11326, 13469, 16017, 19048, 22652
                ]
                if self.num_layers == 40:
                    wins = [
                        105, 125, 149, 178, 211, 251, 299, 356, 423, 503,
                        598, 712, 847, 1007, 1198, 1424, 1694, 2014, 2396, 2849,
                        3388, 4031, 4791, 5699, 6779, 8061, 9583, 11399, 13559, 16117,
                        19176, 22790, 97, 115, 137, 163, 194, 230, 274, 326
                    ]
                temp_c = self.input_max_length
                all_block_num = 0
                temp_length = 0
                num_block = 0
                for wins_item in enumerate(wins):
                    wins_index = wins_item[0]
                    wins_val = wins_item[1]
                    temp_length = min(wins_val, temp_c) + self.output_max_length
                    if self.block_size != 0:
                        num_block = num_block + math.ceil(temp_length / self.block_size)
                    if (wins_index + 1) % model_config.num_kv_heads_origin == 0:
                        all_block_num = max(all_block_num, num_block)
                        temp_length = 0
                        num_block = 0
                self.new_num_blocks = all_block_num * self.batch_size + 100


        if self.soc_info.need_nz:
            v_cache_shape = (self.new_num_blocks, self.num_heads * self.v_head_size // 16, self.block_size, 16)
            if self.v_head_size == 0:
                v_cache_shape = (1,)
            self.kv_cache = [
                (
                    torch.empty(
                        (self.new_num_blocks, self.num_heads * self.k_head_size // 16, self.block_size, 16),
                        dtype=self.dtype,
                        device=self.device,
                    ),
                    torch.empty(
                        v_cache_shape if layer_id % self.v_cache_share_fractor == 0 else (1,),
                        dtype=self.dtype,
                        device=self.device,
                    ),
                )
                for layer_id in range(self.num_layers)
            ]
        else:
            self.kv_cache = [
                (
                    torch.empty(
                        (self.new_layers_num_blocks[layer_id] if self.compress_head_rope else self.new_num_blocks, \
                         self.block_size, self.num_heads, self.k_head_size),
                        dtype=self.dtype,
                        device=self.device,
                    ),
                    torch.empty(
                        self.get_v_cache_shape(layer_id),
                        dtype=self.dtype,
                        device=self.device,
                    ),
                )
                for layer_id in self.layer_list
            ]

        random_block_allocate = False
        if random_block_allocate:
            self.block_map = torch.randperm(self.new_num_blocks, dtype=torch.long)
            self.contrary_block_map = torch.zeros(self.new_num_blocks, dtype=torch.long)
            for i in range(self.new_num_blocks):
                self.contrary_block_map[self.block_map[i]] = i
        else:
            self.block_map = torch.arange(self.new_num_blocks, dtype=torch.long)
            self.contrary_block_map = torch.arange(self.new_num_blocks, dtype=torch.long)

        if self.enable_data_parallel:
            # DP场景下最后一个block对外不可见，仅用于存放陪跑数据
            self.free_block_mask = [torch.ones(self.new_num_blocks - 1, dtype=torch.long)
                                    for _ in range(model_config.mapping.attn_dp.group_size)]
        else:
            self.free_block_mask = [torch.ones(self.new_num_blocks, dtype=torch.long)]
        self.total_slots = torch.arange(self.new_num_blocks * self.block_size, dtype=torch.long)
        self.total_slots = self.total_slots.view(self.new_num_blocks, self.block_size)

    @staticmethod
    def get_dtype_size(dtype):
        dtype_size_map = {torch.float16: 2, torch.float32: 4, torch.bfloat16: 2, torch.int8: 1}
        return dtype_size_map.get(dtype, 2)

    def allocate(self, batch):
        total_need_blocks_per_dp_group = [0] * self.attn_dp_size
        max_need_blocks = 0
        # 每个dp组所有请求所需block总数，以及单请求所需最大block数
        for req in batch.req_list:
            if req.block_tables:
                error_msg = f"req_id: {req.req_id} block has been allocated."
                logger.error(error_msg, ErrorCode.ATB_MODELS_INTERNAL_ERROR)
                raise AssertionError(error_msg)

            total_need_blocks_per_dp_group[req.dp_rank] += req.need_blocks
            max_need_blocks = max(max_need_blocks, req.need_blocks)

        allocate_block_indices = []
        allocate_blocks = []
        for i in range(self.attn_dp_size):
            free_block_indices = self.free_block_mask[i].nonzero().flatten()
            if free_block_indices.numel() < total_need_blocks_per_dp_group[i]:
                error_msg = f"Out of available cache blocks: asked {total_need_blocks_per_dp_group[i]}, " \
                            f"only {free_block_indices.numel()} free blocks."
                logger.error(error_msg, ErrorCode.ATB_MODELS_INTERNAL_ERROR)
                raise AssertionError(error_msg)

            allocate_block_indices.append(free_block_indices[:total_need_blocks_per_dp_group[i]])
            allocate_blocks.append(self.block_map[allocate_block_indices[i]])

        block_offset = [0] * self.attn_dp_size
        block_tables_list = []
        slot_tables_list = []
        for req in batch.req_list:
            req.block_tables = allocate_blocks[req.dp_rank][block_offset[req.dp_rank]:
                                                            block_offset[req.dp_rank] + req.need_blocks]
            req.slot_tables = self.total_slots[req.block_tables].flatten()
            block_tables = req.block_tables
            if req.need_blocks < max_need_blocks:
                block_tables = torch.concat(
                    [block_tables, torch.zeros(max_need_blocks - req.need_blocks, dtype=torch.long)], dim=0)
            block_tables_list.append(block_tables.view(1, -1))
            slot_tables_list.append(req.slot_tables)
            block_offset[req.dp_rank] += req.need_blocks

        batch.batch_block_tables = torch.concat(block_tables_list, dim=0)
        batch.batch_slots_tables = torch.concat(slot_tables_list, dim=0)

        for i in range(self.attn_dp_size):
            self.free_block_mask[i][allocate_block_indices[i]] = 0

    def free(self, req):
        if req.block_tables is not None:
            block_indices = self.contrary_block_map[req.block_tables]
            self.free_block_mask[req.dp_rank][block_indices] = 1

    def get_free_block_num(self, dp_rank):
        free_block_indices = self.free_block_mask[dp_rank].nonzero()
        return len(free_block_indices)

    def get_v_cache_shape(self, layer_id):
        if self.v_head_size == 0 or layer_id % self.v_cache_share_fractor != 0:
            return (1, 1, 1, 1)
        elif self.compress_head_rope:
            return (self.new_layers_num_blocks[layer_id], self.block_size, self.num_heads, self.v_head_size)
        return (self.new_num_blocks, self.block_size, self.num_heads, self.v_head_size)
