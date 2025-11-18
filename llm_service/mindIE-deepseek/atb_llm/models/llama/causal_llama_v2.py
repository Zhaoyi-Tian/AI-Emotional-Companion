# Copyright Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
from typing import Optional

import torch

from atb_llm.models.base.causal_lm_v2 import CausalLMV2
from atb_llm.models.base.config import BaseConfig
from atb_llm.models.llama.modeling_llama import FlashLlamaModel
from atb_llm.utils.weights import Weights
from atb_llm.utils.layers import load_column_multi


class LlamaForCausalLMV2(CausalLMV2):
    """
    This class serves as the primary functional class that inherits from the `CausalLMV2` class.
    It is responsible for constructing the model architecture by integrating the FlashLlamaModel.
    """
    def __init__(self, config: BaseConfig, weights: Weights, **kwargs):
        super().__init__(config, weights, **kwargs)

        # model structure
        self.model = FlashLlamaModel(config, weights, attn_decode_backend=self.attn_decode_backend)
    
        self.lm_head = load_column_multi(
            config,
            prefixes=["lm_head"],
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
        The method is responsible for setting the static parameters for the engine. It accomplishes this
        by first obtaining a set of default parameters by calling the `update_engine_static_param` method
        from the `BaseLMCpp` class.
        Afterward, it updates these default parameters by adding the following settings:
        whether to utilize paged attention and whether to unpad inputs.

        Returns:
            A dictionary of engine static parameters.
        """
        engine_static_param = super().update_engine_static_param()
        engine_static_param.update({
            "isEmbeddingParallel": self.model.parallel_embedding
        })
        return engine_static_param

    def prepare_inputs(
        self,
        input_ids_or_embedding: torch.Tensor,
        position_ids: torch.Tensor,
        is_prefill: Optional[bool],
        max_seq_len: int,
    ) -> None:
        """
        This method prepares inputs for the model.
        It first calls the parent class method to prepare the default inputs.
        Then, it updates input embedding and input ids of the input engines.
        """
        super().prepare_inputs(input_ids_or_embedding, position_ids, is_prefill, max_seq_len)
        self.engine_inputs = [
            self.placeholder if self.infer_param.skip_word_embedding else input_ids_or_embedding,
            input_ids_or_embedding if self.infer_param.skip_word_embedding else self.placeholder,
            *self.engine_inputs[1:]
        ]
