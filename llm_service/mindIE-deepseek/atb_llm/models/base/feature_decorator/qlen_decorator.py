# Copyright Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
from typing import List

import torch


class QLenDecorator:
    @staticmethod
    def update_inputs(
            engine_inputs: List[torch.Tensor],
            engine_runtime_param,
            device,
            **kwargs
        ) -> None:
        q_lens = kwargs.get("q_lens", None)
        mask = kwargs.get("attn_mask", None)
        if q_lens is None or mask is None:
            return
        q_len_tensor = torch.tensor(q_lens).to(device).to(torch.int32)

        engine_inputs.append(q_len_tensor)
        engine_runtime_param.update({"qLen": q_lens})