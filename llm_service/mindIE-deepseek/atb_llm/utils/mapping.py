# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import List
import os

import torch
from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.utils.env import ENV

DP = "dp"
TP = "tp"
MOE_TP = "moe_tp"
PP = "pp"
MICROBATCH_SIZE = "microbatch_size"
MOE_EP = "moe_ep"


@dataclass
class ParallelInfo:
    group_size: int = 1
    num_group: int | None = None
    rank_per_group: List[List[int]] | None = None
    rank: int | None = None
    domain: str = ""


class PipelineParallelInfo(ParallelInfo):
    def __init__(self):
        super().__init__()
        self.microbatch_size = 1
        self.pp_bcast_group = None
        self.tp = ParallelInfo()


class Mapping:
    def __init__(self, world_size, rank, **kwargs):
        self.world_size = world_size
        self.rank = rank
        self.attn_tp = ParallelInfo()
        self.attn_dp = ParallelInfo()
        self.mlp_tp = ParallelInfo()
        self.pp = PipelineParallelInfo()
        self.moe_tp = ParallelInfo()
        self.moe_ep = ParallelInfo()
        self.parse_parallel_info(**kwargs)
        self.validate()
        self.get_tp_group(self.attn_tp)
        self.get_dp_group(self.attn_dp)
        self.get_tp_group(self.mlp_tp)
        self.get_pp_group(self.pp)
        self.get_tp_group(self.moe_tp)
        self.get_dp_group(self.moe_ep)

        self.get_domain(self.attn_tp, self.attn_dp, 0)
        self.get_domain(self.attn_dp, self.attn_tp, self.attn_dp.group_size)
        self.get_domain(self.moe_tp, self.moe_ep, 2 * world_size)
        self.get_domain(self.moe_ep, self.moe_tp, 2 * world_size + self.moe_ep.group_size)
        # 设置默认通信域为63
        self.default_domain = str(63)
        self.mlp_tp.domain = self.default_domain

        if self.has_pp():
            import torch.distributed as dist
            master_ip = os.getenv("MASTER_IP", None)
            if not master_ip:
                raise ValueError("Master ip cannot be None when pipeline parallel is used. Please export MASTER_IP.")
            master_port = int(os.getenv("MASTER_PORT", None))
            network_adapter = os.getenv("NETWORK_ADAPTER", None)
            if not network_adapter or not master_port:
                raise ValueError("MASTER_PORT or Network adapter cannot be None when pipeline parallel is used. \
                    Please export environment.")
            os.environ['GLOO_SOCKET_IFNAME'] = network_adapter
            init_method = f"tcp://{master_ip}:{master_port}"
            logger.debug(f"rank: {self.rank}, init_method: {init_method}, start to init distributed")
            dist.init_process_group(backend='gloo', init_method=init_method, world_size=world_size, rank=self.rank)
            self.pp.pp_bcast_group = torch.distributed.group.WORLD
            logger.debug(f"rank: {self.rank}, init_method: {init_method}, init distributed successfully")

    def __repr__(self):
        return (
            "Mapping("
            + f"world_size={self.world_size}, "
            + f"rank={self.rank}, "
            + f"pp_rank={self.pp.rank}, "
            + f"pp_groups={self.pp.rank_per_group}, "
            + f"micro_batch_size={self.pp.microbatch_size}) "
        )

    @staticmethod
    def get_rank(rank_per_group, target_rank_id):
        for group in rank_per_group:
            if target_rank_id in group:
                return group.index(target_rank_id)
        return -1
    
    @staticmethod
    def get_domain(src_module, dst_module, start_idx):
        current_idx = dst_module.rank
        src_module.domain = str(start_idx + current_idx)

    def parse_parallel_info(self, **kwargs):
        if kwargs.get(DP, -1) != -1:
            self.attn_dp.group_size = kwargs.get(DP, -1)
        # tp默认值为world_size
        self.attn_tp.group_size = self.world_size
        self.mlp_tp.group_size = self.world_size
        self.pp.tp.group_size = self.world_size
        # pp默认值为1
        self.pp.group_size = 1
        # microbatch_size
        self.pp.microbatch_size = kwargs.get(MICROBATCH_SIZE)
        self.moe_tp.group_size = self.world_size
        if kwargs.get(TP, -1) != -1:
            self.attn_tp.group_size = kwargs.get(TP, self.world_size)
            self.moe_tp.group_size = kwargs.get(TP, self.world_size)
            self.pp.tp.group_size = kwargs.get(TP, self.world_size)
        # moe_tp
        if kwargs.get(MOE_TP, -1) != -1:
            self.moe_tp.group_size = kwargs.get(MOE_TP, self.moe_tp.group_size)
        # moe_ep
        if kwargs.get(MOE_EP, -1) != -1:
            self.moe_ep.group_size = kwargs.get(MOE_EP, self.moe_ep.group_size)
        # pp
        if kwargs.get(PP, -1) != -1:
            self.pp.group_size = kwargs.get(PP, self.pp.group_size)

    def validate(self):
        if self.pp.group_size != 1 and self.attn_dp.group_size != 1:
            error_msg = "The attention module cannot support data parallel and pipeline parallel simultaneously. " \
                        "Please check `dp` and `pp`."
            logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_INVALID)
            raise ValueError(error_msg)

        if self.has_pp():
            if self.pp.tp.group_size * self.pp.group_size != self.world_size:
                error_msg = f"World size must equal to attention's tp_size * pp_size. " \
                            f"pp_size is {self.pp.group_size}. " \
                            f"pp's tp_size is {self.pp.tp.group_size}. " \
                            f"World size is {self.world_size}. " \
                            f"Please check `tp`, `pp` and `world_size`."
                logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_INVALID)
                raise ValueError(error_msg)
        else:
            if self.attn_tp.group_size * self.attn_dp.group_size != self.world_size:
                error_msg = f"World size must equal to attention's dp_size * attention's tp_size. " \
                            f"Attention's tp_size is {self.attn_tp.group_size}. " \
                            f"Attention's dp_size is {self.attn_dp.group_size}. World size is {self.world_size}. " \
                            f"Please check `dp`, `tp` and `world_size`."
                logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_INVALID)
                raise ValueError(error_msg)

            if self.has_moe_ep():
                if self.moe_ep.group_size * self.moe_tp.group_size != self.world_size:
                    error_msg = f"World size must equal to MoE's ep_size * MoE's tp_size. " \
                            f"MoE's tp_size is {self.moe_tp.group_size}. " \
                            f"MoE's dp_size is {self.moe_ep.group_size}. World size is {self.world_size}. " \
                            f"Please check `tp`, `moe_tp`, `moe_ep` and `world_size`."
                    logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_INVALID)
                    raise ValueError(error_msg)
            else:
                if self.moe_tp.group_size != self.world_size:
                    error_msg = f"World size must equal to MoE's tp_size. " \
                        f"MoE's tp_size is {self.moe_tp.group_size}. " \
                        f"World size is {self.world_size}. " \
                        f"Please check `tp`, `moe_tp` and `world_size`."
                    logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_INVALID)
                    raise ValueError(error_msg)

    def get_tp_group(self, module):
        module.num_group = self.world_size // module.group_size
        module.rank_per_group = []
        for i in range(module.num_group):
            ranks = range(i * module.group_size, (i + 1) * module.group_size)
            module.rank_per_group.append(list(ranks))
        module.rank = self.get_rank(module.rank_per_group, self.rank)

    def get_dp_group(self, module):
        module.num_group = self.world_size // module.group_size
        module.rank_per_group = []
        for j in range(module.num_group):
            ranks = range(j, self.world_size, module.num_group)
            module.rank_per_group.append(list(ranks))
        module.rank = self.get_rank(module.rank_per_group, self.rank)

    def get_pp_group(self, module):
        self.get_tp_group(module.tp)
        module.num_group = self.world_size // module.group_size
        pp_groups = []
        for i in range(module.num_group):
            ranks = range(i, self.world_size, module.num_group)
            pp_groups.append(list(ranks))
        module.rank_per_group = pp_groups
        module.rank = self.rank // (module.tp.group_size * self.attn_dp.group_size)

    def has_attn_tp(self) -> bool:
        return self.attn_tp.group_size > 1

    def has_dp(self) -> bool:
        return self.attn_dp.group_size > 1

    def has_mlp_tp(self) -> bool:
        return self.mlp_tp.group_size > 1
    
    def is_last_pp_rank(self):
        return self.pp.rank == self.pp.group_size - 1
 
    def is_first_pp_rank(self):
        return self.pp.rank == 0
    
    def has_pp(self):
        return self.pp.group_size > 1
    
    def prev_pp_rank(self):
        p = self.rank - self.pp.tp.group_size
        if p < 0:
            p = p + self.world_size
        return p
 
    def next_pp_rank(self):
        p = self.rank + self.pp.tp.group_size
        if p >= self.world_size:
            p = p - self.world_size
        return p
 
    def pp_layers(self, num_layers: int) -> List[int]:
        layers_per_pipeline_stage = num_layers // self.pp.group_size
        layers_range = range(self.pp.rank * layers_per_pipeline_stage,
                             (self.pp.rank + 1) * layers_per_pipeline_stage)
        return list(layers_range)
    
    def has_moe_tp(self) -> bool:
        return self.moe_tp.group_size > 1

    def has_moe_ep(self) -> bool:
        return self.moe_ep.group_size > 1

    def to_dict(self):
        parallel_dict = {
            "worldSize": self.world_size,
            "rank": self.rank,
            "hasAttnTp": self.has_attn_tp(),
            "attnTpRank": self.attn_tp.rank,
            "attnTpSize": self.attn_tp.group_size,
            "hasAttnDp": self.has_dp(),
            "attnDpRank": self.attn_dp.rank,
            "attnDpSize": self.attn_dp.group_size,
            "hasMlpTp": self.has_mlp_tp(),
            "mlpTpRank": self.mlp_tp.rank,
            "mlpTpSize": self.mlp_tp.group_size,
            "hasMoeTp": self.has_moe_tp(),
            "moeTpRank": self.moe_tp.rank,
            "moeTpSize": self.moe_tp.group_size,
            "hasMoeEp": self.has_moe_ep(),
            "moeEpRank": self.moe_ep.rank,
            "moeEpSize": self.moe_ep.group_size,
        }
        if ENV.rank_table_file == "":
            parallel_dict.update({"attnTpDomain": self.attn_tp.domain,
                                  "attnDpDomain": self.attn_dp.domain,
                                  "mlpTpDomain": self.mlp_tp.domain,
                                  "moeTpDomain": self.moe_tp.domain,
                                  "moeEpDomain": self.moe_ep.domain})
        return parallel_dict