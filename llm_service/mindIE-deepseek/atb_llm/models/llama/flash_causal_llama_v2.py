# Copyright Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
from typing import Optional, List, Tuple

import torch

from atb_llm.models.base.flash_causal_lm_v2 import FlashCausalLMV2
from atb_llm.models.base.config import BaseConfig
from atb_llm.models.llama.modeling_llama import FlashLlamaModel
from atb_llm.utils.weights import Weights
from atb_llm.utils.layers import load_column_multi, TensorHead, TensorParallelHead
from atb_llm.utils.layers.embedding.position_rotary_embedding import PositionEmbeddingType
from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode


class FlashLlamaForCausalLMV2(FlashCausalLMV2):
    """
    This class serves as the primary functional class that inherits from the `FlashCausalLMV2` class.
    It is responsible for constructing the model architecture by integrating the FlashLlamaModel.
    """
    def __init__(self, config: BaseConfig, weights: Weights, lmhead_prefix="lm_head", model_prefix="model", **kwargs):
        super().__init__(config, weights, **kwargs)
        self.infer_param.update_matmul_nz(
            self.soc_info, config.quantize
        )

        # model structure
        self.model = FlashLlamaModel(config, weights, model_prefix, attn_decode_backend=self.attn_decode_backend)

        if config.quantize == "w8a8sc":
            self.lm_head = TensorHead.load_weight(
                config,
                prefix=lmhead_prefix,
                weights=weights,
                is_norm=False,
            )
        elif config.tie_word_embeddings:
            self.lm_head = TensorParallelHead.load(
                config,
                prefix="model.embed_tokens",
                weights=weights,
                is_norm=True,
            )
        else:
            self.lm_head = load_column_multi(
                config,
                prefixes=[lmhead_prefix],
                weights=weights,
                head_size=1,
                lm_head=True,
            )

    @property
    def model_torch_class_name(self):
        """
        This method returns the name of the PyTorch class for the model.
        """
        return "llama_LlamaDecoderModel"

    def update_engine_static_param(self):
        """
        The method is responsible for setting the static parameters for the engine.
        It accomplishes this by first obtaining a set of default parameters by calling
        the `update_engine_static_param method` from the `FlashCausalLMV2` class.
        Afterward, it updates these default parameters by adding the following settings:
        whether to utilize tensor parallelism in word embedding.
        """
        engine_static_param = super().update_engine_static_param()
        engine_static_param.update({
            "isEmbeddingParallel": self.model.parallel_embedding,
        })
        return engine_static_param

    def generate_positional_embedding(self, max_seq_len: int, **kwargs) -> None:
        """
        This method generates the positional embeddings for the input sequence.
        If the position embedding type is ROPE and certain configuration parameters are set,
        it uses an advanced method to generate the cosine and sine tables.
        Otherwise, it calls the parent class method to generate the positional embeddings.
        """
        if self.config_metadata.position_embedding_type == PositionEmbeddingType.ROPE \
            and (self.config.rope_given_inv_feq_str is not None or self.config.rope_vanilla_theta is not None):
            self.pos_embed_info.cosine_table, \
            self.pos_embed_info.sine_table = self._get_advanced_cos_sin_table(max_seq_len)
        else:
            super().generate_positional_embedding(max_seq_len)

    def prepare_default_inputs(
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
            **kwargs
        ) -> None:
        """
        This method prepares the default inputs for the model.
        It first calls the parent class method to prepare the default inputs.
        Then, it updates input embedding and input ids of the input engines.
        """
        super().prepare_default_inputs(
            input_ids, position_ids, is_prefill, kv_cache,
            block_tables, slots, input_lengths, max_seq_len,
            lm_head_indices, **kwargs)
        self.engine_inputs = [
            self.placeholder if self.infer_param.skip_word_embedding else input_ids,
            input_ids if self.infer_param.skip_word_embedding else self.placeholder,
            *self.engine_inputs[1:]
        ]

    # 固定基频: rope_theta
    # 自定义基频: rope_given_inv_feq_str
    # 分段基频: rope_theta/rope_given_inv_feq_str + rope_vanilla_theta + rope_keep_local_base_windows
    def _get_advanced_cos_sin_table(self, max_seq_len: int, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        given_inv_feq_str = self.config.rope_given_inv_feq_str
        if given_inv_feq_str:
            inv_freq = torch.FloatTensor([float(invf) for invf in given_inv_feq_str.split(',')], device=self.device)
            if len(inv_freq) != self.config_metadata.head_dim // 2:
                logger.error("Error: only support len(inv_freq) == dim/2 ,check your inv_freq length", 
                             ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
                raise AssertionError('given_inv_feq_str: length not match head_dim/2')
        else:
            inv_freq = 1.0 / (self.config.rope_theta ** (torch.arange(0,
            self.config_metadata.head_dim, 2, device=self.device).float() / self.config_metadata.head_dim))

        seq = torch.arange(max_seq_len, device=self.device).float() + offset
        freqs = torch.outer(seq, inv_freq)

        if self.config.rope_keep_local_base_windows:
            keep_local_base_windows = [int(w) for w in self.config.rope_keep_local_base_windows.split(',')]
            if len(keep_local_base_windows) != self.config_metadata.head_dim // 2:
                logger.error(
                    "Error: only support len(keep_local_base_windows) == dim/2 ,check your base_windows length", 
                    ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
                raise AssertionError('keep_local_base_windows: length not match head_dim/2')

            inv_freq_base = 1.0 / (self.config.rope_vanilla_theta ** (torch.arange(0,
            self.config_metadata.head_dim, 2, device=self.device).float() / self.config_metadata.head_dim))
            freqs_base = torch.outer(seq, inv_freq_base)
            freqs_after_window = freqs + torch.tensor(keep_local_base_windows) * (inv_freq_base - inv_freq)
            for idx, i_keep_local_base_window in enumerate(keep_local_base_windows):
                freqs[:, idx] = torch.cat((
                    freqs_base[:i_keep_local_base_window, idx],
                    freqs_after_window[i_keep_local_base_window:, idx]
                ))

        # Different from paper, but it uses a different permutation in order to obtain the same calculation（ks）
        emb = torch.cat((freqs, freqs), dim=-1)
        return (emb.cos() * self.config.rope_mscale).to(self.dtype).to(self.device), \
        (emb.sin() * self.config.rope_mscale).to(self.dtype).to(self.device)
