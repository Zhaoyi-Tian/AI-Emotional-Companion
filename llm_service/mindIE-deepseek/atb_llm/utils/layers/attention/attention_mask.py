# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import math

import torch

from torch import nn


class AttentionMask(nn.Module):
    def __init__(self, atten_mask):
        super().__init__()
        self._seq_len_cached = 0
        self.atten_mask_cache = atten_mask
        self._rope_decode_mask = None
        self.atten_mask_cpu_cache = None

    @classmethod
    def static(cls, max_seq_len, dtype=torch.float16, mini_type=torch.float32):
        bias_cache = torch.tril(torch.ones((max_seq_len, max_seq_len), dtype=torch.bool)).view(max_seq_len,
                                                                                               max_seq_len)
        bias_cache = ~bias_cache
        if dtype == torch.float16:
            if mini_type == torch.float16:
                mask_value = torch.finfo(torch.float16).min
            else:
                mask_value = torch.finfo(torch.float32).min
        else:
            mask_value = 1
        attn_mask = torch.masked_fill(torch.zeros(size=(max_seq_len, max_seq_len)), bias_cache, mask_value)
        return cls(attn_mask)

    def get_decode_attn_mask(
        self, input_lengths: torch.tensor, max_s: int
    ):
        bs = input_lengths.shape[0]
        attn_mask = torch.ones((bs, max_s), dtype=torch.bool).npu()
        input_lengths_unsqueeze = input_lengths.unsqueeze(1)
        token_index = torch.arange(0, max_s).repeat(bs).view(bs, max_s).npu()
        attn_mask[token_index < input_lengths_unsqueeze] = 0
        return attn_mask.view(-1, 1, max_s)

    def update_attn_cache(self, dtype, device, seqlen, mini_type=torch.float32):
        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if seqlen > self._seq_len_cached or self.atten_mask_cache.dtype != dtype:
            self._seq_len_cached = seqlen
            bias_cache = torch.tril(torch.ones((seqlen, seqlen), dtype=torch.bool)).view(seqlen, seqlen)
            bias_cache = ~bias_cache
            if dtype == torch.float16:
                if mini_type == torch.float16:
                    mask_value = torch.finfo(torch.float16).min
                else:
                    mask_value = torch.finfo(torch.float32).min
            else:
                mask_value = 1
            mask_atten_cache = torch.masked_fill(torch.zeros(size=(seqlen, seqlen)), bias_cache, mask_value)
            self.atten_mask_cache = mask_atten_cache.to(dtype)
        if self.atten_mask_cache.device != device:
            self.atten_mask_cache = self.atten_mask_cache.to(device)

    def get_attn_mask(
            self, max_s: int, dtype: torch.dtype, device: torch.device, mini_type=torch.float32
    ):
        self.update_attn_cache(dtype, device, max_s, mini_type)
        return self.atten_mask_cache[:max_s, :max_s]

    def get_rope_prefill_mask(self, max_s: int, dtype: torch.dtype, device: torch.device, mini_type=torch.float32):
        return self.get_attn_mask(max_s, dtype, device, mini_type)

    def get_rope_decode_mask(self, dtype: torch.dtype, device: torch.device):
        if self._rope_decode_mask is None:
            self._rope_decode_mask = self.get_attn_mask(
            1, dtype=dtype, device="cpu").to(device)
        return self._rope_decode_mask

    def get_alibi_prefill_mask(self, max_seq_len, config, config_metadata, dtype, rank):
        if self.atten_mask_cpu_cache is None:
            total_head_num = config.num_attention_heads
            slopes = torch.Tensor(self._get_interleave(total_head_num, config.alibi_bias_max))
            tensor_list = []
            # 算子要求的压缩alibi mask shape为 [head_num, max_seq, 128]
            for i in range(128):
                tensor = torch.empty(config.max_position_embeddings).fill_(-float('inf'))
                tensor[i:] = -1 * torch.arange(0, config.max_position_embeddings - i)
                tensor = tensor.unsqueeze(0)
                tensor_list.append(tensor)
            tensor = torch.cat(tensor_list, dim=0).t()
            tensor = tensor.expand(total_head_num, -1, -1)
            self.atten_mask_cpu_cache = slopes.unsqueeze(1).unsqueeze(1) * tensor
            self.atten_mask_cpu_cache = self.atten_mask_cpu_cache[
                rank * config_metadata.num_attention_heads:\
                    (rank + 1) * config_metadata.num_attention_heads, :, :].to(dtype)

        # 算子要求: 小于128则按实际长度切，大于128则按128切，算子内部扩展到实际长度
        slice_len = max_seq_len if max_seq_len <= 128 else 128
        atten_mask = self.atten_mask_cpu_cache[:, :, :slice_len].npu()
        return atten_mask

    def get_alibi_decode_mask(self, max_seq_len, pos_list, config, config_metadata, dtype, rank):
        total_head_num = config.num_attention_heads
        slopes = torch.Tensor(self._get_interleave(total_head_num, config.alibi_bias_max))
        tensor_list = []
        for pos in pos_list:
            tensor = torch.empty(max_seq_len).fill_(-float('inf'))
            tensor[:pos + 1] = torch.arange(-pos, 1)
            tensor = tensor.unsqueeze(0)
            tensor_list.append(tensor)
        tensor = torch.cat(tensor_list, dim=0)
        tensor = tensor.expand(total_head_num, -1, -1)
        alibi_mask = slopes.unsqueeze(1).unsqueeze(1) * tensor
        alibi_mask = alibi_mask.permute(1, 0, 2).unsqueeze(2)
        return alibi_mask[:, rank * config_metadata.num_attention_heads:\
                          (rank + 1) * config_metadata.num_attention_heads, :, :].to(dtype).npu()

    def _get_interleave(self, n, alibi_bias_max=8.0):
        def _get_interleave_power_of_2(n, alibi_bias_max):
            if n == 0:
                return 0
            start = (0.5 ** (alibi_bias_max / n))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            return _get_interleave_power_of_2(n, alibi_bias_max)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return _get_interleave_power_of_2(closest_power_of_2, alibi_bias_max) + \
                self._get_interleave(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]
