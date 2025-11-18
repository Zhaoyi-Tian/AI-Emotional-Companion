# Copyright Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
from typing import Optional, List, Tuple

import torch
from torch import nn
import torch_npu

from atb_llm.models.base.base_lm_cpp import BaseLMCpp
from atb_llm.models.base.feature_decorator.lora_decorator import LoraDecorator
from atb_llm.models.base.feature_decorator.qlen_decorator import QLenDecorator
from atb_llm.models.base.config import BaseConfig
from atb_llm.utils.weights import Weights
from atb_llm.utils.layers import AttentionMask
from atb_llm.utils.layers.embedding.position_rotary_embedding import PositionEmbeddingType
from atb_llm.models.base.engine.engine_manager import engine_manager
from atb_llm.utils.log import logger, print_log
from atb_llm.utils.log.error_code import ErrorCode
from .model_utils import BaseModel


class CacheMetadata:
    """
    This class is used to hold metadata about the key-value caches.
    It includes the IDs and shapes of both the key and value caches.
    """
    k_cache_id: int | None = None
    k_cache_shape: list | None = None
    v_cache_id: int | None = None
    v_cache_shape: list | None = None


class FlashCausalLMV2(BaseModel, BaseLMCpp):
    """
    This class serves as the foundation for performing inference with paged attention.
    """
    def __init__(self, config: BaseConfig, weights: Weights, **kwargs):
        BaseModel.__init__(self)
        super(nn.Module, self).__init__(config, weights, **kwargs)
        self.cache_metadata = CacheMetadata()

        # feature
        self.lora_decorator = LoraDecorator(config, weights, self, **kwargs)
        self.qlen_decorator = QLenDecorator()

        # update parameters
        self.infer_param.enable_lcoc = self.infer_param.enable_lcoc and not self.lora_decorator.active

    def update_engine_static_param(self):
        """
        The method is responsible for setting the static parameters for the engine. It accomplishes this
        by first obtaining a set of default parameters by calling the `update_engine_static_param` method
        from the `BaseLMCpp` class.
        Afterward, it updates these default parameters by adding the following settings:
        whether to utilize paged attention and whether to unpad inputs.

        Returns:
            A dictionary of engine static parameters.
        """
        engine_static_param = BaseLMCpp.update_engine_static_param(self)
        engine_static_param.update({
            "isFA": False,
            "isUnpadInputs": True,
        })
        return engine_static_param

    def build_engine(self, **kwargs) -> None:
        """
        Builds the engine for the model. Depending on the configuration, 
        it either builds the Python engine or the C++ engine.

        Args:
            kwargs: Additional keyword arguments for the engine build process.
        """
        if self.lora_decorator.active:
            self.lora_decorator.adapter_manager.prepare_adapter_weights()
        if self.infer_param.enable_python_engine:
            pass
        else:
            BaseLMCpp.build_engine(self)

    def init_mask(self, **kwargs) -> None:
        """
        Initializes the attention mask for the model. This includes setting a generator for the attention mask.
        Args:
            kwargs: Additional keyword arguments for the mask initialization process.
        """
        self.attn_mask_info.generator = AttentionMask.static(self.attn_mask_info.MAX_BASE_LEN, dtype=self.torch_dtype)

    def generate_mask(self, atten_mask: torch.Tensor, is_prefill: bool, **kwargs) -> torch.Tensor:
        """
        Generates the attention mask for the model. Depending on the configuration, it either generates a prefill mask 
        or a decode mask.

        Args:
            atten_mask: The current attention mask.
            is_prefill: A boolean indicating whether to generate a prefill mask.
            kwargs: Additional keyword arguments for the mask generation process.

        Returns:
            The generated attention mask.
        """
        if atten_mask is None:
            if is_prefill:
                atten_mask = self.generate_prefill_mask(**kwargs)
            else:
                atten_mask = self.generate_decode_mask(**kwargs)

        if self.soc_info.need_nz:
            atten_mask = self.transdata_operation.execute([atten_mask])[0]
        
        return atten_mask

    def generate_prefill_mask(self, **kwargs) -> None:
        """
        Generates the prefill mask for the model based on the position embedding type.

        Args:
            kwargs: Additional keyword arguments for the mask generation process.
                `max_seq_len` is required when creating alibi mask.
        Returns:
            The generated prefill mask.
        """
        if self.config_metadata.position_embedding_type == PositionEmbeddingType.ROPE:
            return self.attn_mask_info.generator.get_rope_prefill_mask(
                self.attn_mask_info.MAX_BASE_LEN, self.torch_dtype, self.torch_device
            )
        elif self.config_metadata.position_embedding_type == PositionEmbeddingType.ALIBI:
            return self.attn_mask_info.generator.get_alibi_prefill_mask(
                kwargs.get("max_seq_len"), self.config, self.config_metadata,
                self.torch_dtype, self.mapping.attn_tp.rank
            )
        else:
            error_msg = "Error: position_embedding_type is illegal"
            logger.error(error_msg, ErrorCode.ATB_MODELS_engine_static_param_JSON_INVALID)
            raise ValueError(error_msg)

    def generate_decode_mask(self, **kwargs) -> None:
        """
        Generates the decode mask for the model based on the position embedding type.

        Args:
            kwargs: Additional keyword arguments for the mask generation process.
                `position_ids` is required when creating alibi mask.
        Returns:
            The generated prefill mask.
        """
        if self.config_metadata.position_embedding_type == PositionEmbeddingType.ROPE:
            return self.attn_mask_info.generator.get_rope_decode_mask(
                self.torch_dtype, self.torch_device
            )
        elif self.config_metadata.position_embedding_type == PositionEmbeddingType.ALIBI:
            return self.attn_mask_info.generator.get_alibi_decode_mask(
                kwargs.get("max_seq_len"), kwargs.get("position_ids", []).tolist(),
                self.config, self.config_metadata,
                self.torch_dtype, self.mapping.attn_tp.rank
            )
        else:
            error_msg = "Error: position_embedding_type is illegal"
            logger.error(error_msg, ErrorCode.ATB_MODELS_engine_static_param_JSON_INVALID)
            raise ValueError(error_msg)

    def update_kv_cache(self, kv_cache: List[Tuple[torch.Tensor, torch.Tensor]], **kwargs) -> None:
        """
        Updates the key-value cache if there are changes in the cache's ID or shape.

        Args:
            kv_cache: A list of tuples containing the key and value tensors.
            kwargs: Additional keyword arguments for the update process.
        """
        kcache_id_diff = self.cache_metadata.k_cache_id != id(kv_cache[0][0])
        vcache_id_diff = self.cache_metadata.v_cache_id != id(kv_cache[0][1])
        kcache_shape_diff = self.cache_metadata.k_cache_shape != kv_cache[0][0].shape
        vcache_shape_diff = self.cache_metadata.v_cache_shape != kv_cache[0][1].shape
        kcache_diff = not self.cache_metadata.k_cache_id or kcache_id_diff or kcache_shape_diff
        vcache_diff = not self.cache_metadata.v_cache_id or vcache_id_diff or vcache_shape_diff
        if kcache_diff or vcache_diff:
            k_caches, v_caches = map(lambda x: list(x), zip(*kv_cache))
            print_log(self.mapping.rank, logger.info, f"k cache's shape {k_caches[0].shape=}")
            print_log(self.mapping.rank, logger.info, f"v cache's shape {v_caches[0].shape=}")
            if self.soc_info.need_nz:
                k_caches = [torch_npu.npu_format_cast_(k_cache, 29) for k_cache in k_caches]
                v_caches = [torch_npu.npu_format_cast_(v_cache, 29) for v_cache in v_caches]
                print_log(self.mapping.rank, logger.info, 
                          f"kv cache's tensor format changes to {torch_npu.get_npu_format(k_caches[0])}")
            # set engine's kv cache
            for engine in engine_manager.get_engines():
                engine.set_kv_cache(k_caches, v_caches)
            # update kv cache's id and shape
            self.cache_metadata.k_cache_id = id(kv_cache[0][0])
            self.cache_metadata.v_cache_id = id(kv_cache[0][1])
            self.cache_metadata.k_cache_shape = kv_cache[0][0].shape
            self.cache_metadata.v_cache_shape = kv_cache[0][1].shape

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
    ):
        """
        Prepare default inputs and runtime paramters for the model.
        It will update `engine_inputs` and `engine_runtime_param`.

        Args:
            See arguments for `forward` for more details.
        """
        if lm_head_indices is None:
            lm_head_indices = torch.tensor(range(input_ids.shape[0]),
                                            dtype=torch.int64, device=input_ids.device)

        if is_prefill:
            self.generate_positional_embedding(self.config.max_position_embeddings)

        self.engine_inputs = [
            input_ids,
            position_ids.to(torch.int64),
            self.pos_embed_info.cosine_table,
            self.pos_embed_info.sine_table,
            self.generate_mask(
                kwargs.get('attn_mask'),
                is_prefill,
                max_seq_len=max_seq_len,
                position_ids=position_ids),
            block_tables.to(torch.int32),
            slots.to(torch.int32),
            self.placeholder,
            self.placeholder,
            self.placeholder,
            input_lengths.to(torch.int32),
            lm_head_indices.to(torch.int64) if is_prefill else self.placeholder
        ]

        self.engine_runtime_param = {
            "seqLen": input_lengths.tolist()
        }

    def prepare_inputs(
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
        Preparing all the necessary inputs for the model.
        It does this by first calling the prepare_default_inputs method to establish a set of default inputs.
        After setting up these default inputs, the prepare_inputs method then calls the `update_inputs` method
        from any decorators that may be applied to the model to update `engine_inputs`.

        Args:
            See arguments for `forward` for more details.
        """
        device = input_ids.device
        self.prepare_default_inputs(
            input_ids, position_ids, is_prefill, kv_cache,
            block_tables, slots, input_lengths, max_seq_len,
            lm_head_indices, **kwargs)
        self.qlen_decorator.update_inputs(
            self.engine_inputs,
            self.engine_runtime_param,
            device,
            **kwargs)
        self.lora_decorator.update_inputs(
            self.engine_inputs,
            kwargs.get("adapter_ids"),
            input_lengths, is_prefill)

    def forward(
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
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            input_ids (torch.Tensor): Token ids of the input prompts.
            position_ids (torch.Tensor): Position ids of each token in each prompt.
            is_prefill (bool): A flag indicating whether the inference phase is prefill.
            kv_cache (List[Tuple[torch.Tensor, torch.Tensor]]): List of tuples containing key-value cache tensors.
            block_tables (torch.Tensor): Block tables for each request.
            slots (torch.Tensor): Slot mapping.
            input_lengths (torch.Tensor): Input lengths for each request.
            max_seq_len (torch): Maximum sequence length that can be accepted by the model.
            lm_head_indices (torch.Tensor, optional): Indices to use in lm_head.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Output logits.
        """
        if self.infer_param.enable_python_engine:
            raise NotImplementedError
        else:
            self.update_kv_cache(kv_cache)
            self.prepare_inputs(
                input_ids, position_ids, is_prefill, kv_cache,
                block_tables, slots, input_lengths, max_seq_len,
                lm_head_indices, **kwargs)
            logits = self.execute_engine(is_prefill=is_prefill, **kwargs)
            return logits
