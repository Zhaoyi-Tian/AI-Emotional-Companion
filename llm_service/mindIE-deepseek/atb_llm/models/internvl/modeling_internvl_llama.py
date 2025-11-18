# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json
import math
from typing import Optional, List, Tuple

import numpy as np
import torch

from atb_llm.utils.log.logging import logger, print_log
from atb_llm.utils.layers.embedding.position_rotary_embedding import (
    PositionRotaryEmbedding,
)
from atb_llm.models.llama.flash_causal_llama import FlashLlamaForCausalLM
from atb_llm.utils.data.weight_wrapper import WeightWrapper, AttnWrapper, MlpWrapper
from atb_llm.utils.layers import TensorEmbedding


class LlamaForCausalLM(FlashLlamaForCausalLM):
    def __init__(
        self, config, weights, lmhead_prefix="lm_head", model_prefix="model", **kwargs
    ):
        super().__init__(
            config,
            weights,
            lmhead_prefix=lmhead_prefix,
            model_prefix=model_prefix,
            **kwargs,
        )
        self.model.parallel_embedding = False
        self.model.embed_tokens = TensorEmbedding(
            prefix="language_model.model.embed_tokens", weights=weights
        )

        self.dim = self.head_size
        self.base = self.config.rope_theta
        self.scaling_factor = 1.0
        self.max_position_embeddings = self.config.max_position_embeddings
        self.rope_scaling = self.config.rope_scaling
        self.max_seq_len_cached = self.max_position_embeddings
        self.rotary_embedding_device = "cpu"
        if self.rope_scaling is None:
            print_log(
                self.tp_rank,
                logger.info,
                "Now \033[33m scaling_type is base rope \033[0m.",
            )
            self.rotary_embedding = PositionRotaryEmbedding.static(
                dim=self.head_size,
                base=self.rope_theta,
                device=self.rotary_embedding_device,
                scaling_factor=self.scaling_factor,
            ).to(self.device)
        else:
            self.scaling_type = self.rope_scaling.type
            if self.scaling_type == "linear":
                print_log(
                    self.tp_rank,
                    logger.info,
                    f"Now \033[33m scaling_type is {self.scaling_type} \033[0m.",
                )
                self.scaling_factor = self.rope_scaling.factor  # t=t/scaling_factor
                self.rotary_embedding = PositionRotaryEmbedding.static(
                    dim=self.head_size,
                    base=self.rope_theta,
                    device=self.rotary_embedding_device,
                    scaling_factor=self.scaling_factor,
                ).to(self.device)
            elif self.scaling_type == "dynamic":
                print_log(
                    self.tp_rank,
                    logger.info,
                    f"Now \033[33m scaling_type is {self.scaling_type} \033[0m.",
                )
                self.rope_scaling_factor = (
                    self.rope_scaling.factor
                )  # Dynamic NTK 外推方法的系数
                self.rotary_embedding = PositionRotaryEmbedding.static(
                    dim=self.head_size,
                    base=self.rope_theta,
                    device=self.rotary_embedding_device,
                    scaling_factor=self.scaling_factor,
                ).to(self.device)
            else:
                print_log(
                    self.tp_rank,
                    logger.info,
                    f"Now \033[33m scaling_type is {self.scaling_type} \033[0m.",
                )
                raise ValueError(
                    "Currently we only support rotary embedding's type being 'dynamic' or 'linear'."
                )

    def get_weights(self):
        attn_wrapper = AttnWrapper(
            norm_name="input_layernorm",
            wrapper_name="self_attn",
            pack_name="query_key_value",
            sep_names=["q_proj", "k_proj", "v_proj"],
            o_name="o_proj",
        )
        mlp_wrapper = MlpWrapper(
            norm_name="post_attention_layernorm",
            wrapper_name="mlp",
            pack_name="gate_up_proj",
            sep_names=["gate_proj", "up_proj"],
            down_name="down_proj",
        )
        weight_wrapper = WeightWrapper(
            self.soc_info, self.tp_rank, attn_wrapper, mlp_wrapper
        )
        weight_wrapper.register_embedding(self.model.embed_tokens)
        for i in range(self.num_layers):
            layer = self.model.layers[i]
            weight_wrapper.register_layer(layer, self.quantize)
            if self.soc_info.need_nz:
                del layer.self_attn
                del layer.post_attention_layernorm
                del layer.mlp
            if self.config.quantization_config.kv_quant_type is not None:
                weight_wrapper.register_layer_kvquant(layer)
        weight_wrapper.register_model_norm(self.model.norm)
        weight_wrapper.register_model_lmhead(self.lm_head)
        return weight_wrapper

    def prepare_inputs_for_ascend(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        is_prefill: bool,
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        input_lengths: torch.Tensor,
        max_seq_len: int,
        lm_head_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # add dynamic
        self.max_seq_len_cached = max(self.max_position_embeddings, max_seq_len)
        # warm_up 阶段会传入max_seq_len=max_input_length，导致 max_seq_len_cached 开始就达到最大
        if self.rope_scaling is None:
            # RotaryEmbedding
            self.rotary_embedding.update_cos_sin_cache_total(
                self.dtype, self.device, self.max_position_embeddings
            )
        elif (self.scaling_type == "dynamic") and (
            self.max_seq_len_cached > self.max_position_embeddings
        ):
            # DynamicNTKScalingRotaryEmbedding
            if self.max_position_embeddings == 0:
                raise ZeroDivisionError("Max position embeddings is zero")
            if self.dim == 2:
                raise ZeroDivisionError(
                    "When calculating RoPE base the divisor in the formula for power positions will be zero"
                )
            base = self.base * (
                np.divide(
                    self.rope_scaling_factor * self.max_seq_len_cached,
                    self.max_position_embeddings,
                )
                - (self.rope_scaling_factor - 1)
            ) ** (np.divide(self.dim, self.dim - 2))
            self.rotary_embedding = self.rotary_embedding.static(
                dim=self.head_size,
                base=base,
                device=self.rotary_embedding_device,
                scaling_factor=self.scaling_factor,
            ).to(self.device)
            self.rotary_embedding.update_cos_sin_cache_total(
                self.dtype, self.device, self.max_seq_len_cached
            )
        else:  # LinearScalingRotaryEmbedding
            # 如果 max_input_length > max_position_embeddings, 需要重置 base 和 rotary_embedding.inv_freq
            self.rotary_embedding = self.rotary_embedding.static(
                dim=self.head_size,
                base=self.base,
                device=self.rotary_embedding_device,
                scaling_factor=self.scaling_factor,
            ).to(self.device)
            self.rotary_embedding.update_cos_sin_cache_total(
                self.dtype, self.device, self.max_position_embeddings
            )

        self.cos_embed = self.rotary_embedding.get_cos_cached_total()
        self.sin_embed = self.rotary_embedding.get_sin_cached_total()

        if is_prefill:
            if self.skip_word_embedding:
                if len(input_ids.shape) < 2:
                    input_ids = self.model.embed_tokens(input_ids)

            if self.soc_info.need_nz:
                pad_maxs = math.ceil(self.max_position_embeddings / 16) * 16
                atten_mask = self.attn_mask.get_attn_mask(
                    pad_maxs, kv_cache[0][0].dtype, kv_cache[0][0].device
                )
                atten_mask = self.transdata_operation.execute([atten_mask])[0]
            else:
                atten_mask = self.attn_mask.get_attn_mask(
                    self.max_base_len, kv_cache[0][0].dtype, kv_cache[0][0].device
                )
            if lm_head_indices is None:
                lm_head_indices = torch.tensor(
                    range(input_ids.shape[0]),
                    dtype=torch.int64,
                    device=input_ids.device,
                )
            self.acl_param = json.dumps({"seqLen": input_lengths.tolist()})
            input_tokens = self.placeholder if self.skip_word_embedding else input_ids
            input_embeddings = (
                input_ids if self.skip_word_embedding else self.placeholder
            )

            if self.dtype == torch.bfloat16:
                input_atten_mask = torch.where(atten_mask == -torch.inf, 1, atten_mask)
            else:
                input_atten_mask = atten_mask
        else:
            input_tokens = input_ids
            input_embeddings = self.placeholder
            self.acl_param = json.dumps({"seqLen": input_lengths.tolist()})
            if self.dtype == torch.bfloat16:
                input_atten_mask = torch.zeros(
                    input_lengths.size(0),
                    self.num_attention_heads,
                    1,
                    input_lengths.max(),
                    dtype=self.dtype,
                    device=self.device,
                )
            else:
                input_atten_mask = self.attn_mask_fake

        self.acl_operation_inputs = [
            input_tokens,
            input_embeddings,
            position_ids.to(torch.int64),
            self.cos_embed,
            self.sin_embed,
            input_atten_mask,
            block_tables.to(torch.int32),
            slots.to(torch.int32),
            self.placeholder,
            self.placeholder,
            self.placeholder,
            input_lengths.to(torch.int32),
            lm_head_indices if is_prefill else self.lm_head_indices_fake,
        ]

        for ind, item in enumerate(self.acl_operation_inputs):
            logger.debug(f"{ind} {item.device=}")
        return self.acl_operation_inputs, self.acl_param
