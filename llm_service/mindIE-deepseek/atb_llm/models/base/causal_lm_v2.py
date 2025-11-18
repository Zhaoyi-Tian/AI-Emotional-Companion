# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import math
from typing import Optional, List, Tuple, Union

import torch
from torch import nn
import torch_npu

from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from atb_llm.models.base.base_lm_cpp import BaseLMCpp
from atb_llm.models.base.config import BaseConfig
from atb_llm.utils.weights import Weights
from atb_llm.utils.op_backend import OpBackend
from atb_llm.models.base.engine.engine_manager import engine_manager
from atb_llm.models.base.model_utils import AttributeMapUtils


class CausalLMV2(PreTrainedModel, AttributeMapUtils, BaseLMCpp):
    """
    This class serves as the foundation for performing inference with flash attention.
    """
    def __init__(self, config: BaseConfig, weights: Weights, **kwargs):
        PreTrainedModel.__init__(self, config)
        super(nn.Module, self).__init__(config, weights, **kwargs)

        # inference parameter
        self.kv_dtype = weights.dtype if config.quantization_config.kv_quant_type is None else torch.int8
        self.nz_dim = 16
        self.batch_num = 0

        # engine input
        self.k_caches = None
        self.v_caches = None
        self.past_key_values_length = 0
        self.kv_cache_idx = torch.zeros(1, dtype=torch.int32).npu()

        self.token_offset = None
        self.seq_len_encoder = None
        self.seq_len_decoder = None

        self.mask_full = None
        self.mask_inc = None

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
            "isFA": True,
            "isUnpadInputs": False,
        })
        return engine_static_param

    def update_kv_cache(
            self, input_ids_or_embedding: Union[torch.LongTensor, torch.FloatTensor],
            past_key_value: Optional[List[torch.FloatTensor]], **kwargs
        ):
        """
        This function intializes the key-value cache, token_offset and seq_len based on
        the input ids or input embedding if batch size changes. Otherwise, it updates them
        base on past_key_value.

        Args:
            input_ids_or_embedding (torch.LongTensor or torch.FloatTensor): The input embedding tokens or ids.
            past_key_value (List[torch.FloatTensor], optional): The past key-value cache.
        """
        batch_size = input_ids_or_embedding.shape[0]

        if batch_size != self.batch_num:
            self.batch_num = batch_size
            self.token_offset = torch.full(
                (self.batch_num,), 0, dtype=torch.int32, device=input_ids_or_embedding.device
            )
            self.seq_len_encoder = torch.full(
                (self.batch_num,), 1, dtype=torch.int32, device=input_ids_or_embedding.device
            )
            self.seq_len_decoder = torch.full(
                (self.batch_num,), 1, dtype=torch.int32, device=input_ids_or_embedding.device
            )
            self.mask_full = torch.zeros(
                (self.batch_num, self.config.max_position_embeddings, self.config.max_position_embeddings),
                dtype=self.dtype, device=input_ids_or_embedding.device
            )

            if not self.soc_info.need_nz:
                self.k_caches = [torch.zeros(self.batch_num,
                                            self.config.max_position_embeddings,
                                            self.config_metadata.num_key_value_heads * self.config_metadata.head_dim,
                                            device=input_ids_or_embedding.device,
                                            dtype=self.kv_dtype) for _ in range(self.config_metadata.num_hidden_layers)]
                self.v_caches = [torch.zeros(self.batch_num,
                                            self.config.max_position_embeddings,
                                            self.config_metadata.num_key_value_heads * self.config_metadata.head_dim,
                                            device=input_ids_or_embedding.device,
                                            dtype=self.kv_dtype) for _ in range(self.config_metadata.num_hidden_layers)]
            else:
                self.k_caches = [torch_npu.npu_format_cast_(torch.zeros(self.batch_num,
                                math.ceil(self.config_metadata.num_key_value_heads * \
                                          self.config_metadata.head_dim / self.nz_dim),
                                self.config.max_position_embeddings, self.nz_dim, device=input_ids_or_embedding.device,
                                dtype=self.kv_dtype), 29) for _ in range(self.config_metadata.num_hidden_layers)]
                torch.npu.empty_cache()
                self.v_caches = [torch_npu.npu_format_cast_(torch.zeros(self.batch_num,
                                math.ceil(self.config_metadata.num_key_value_heads * \
                                          self.config_metadata.head_dim / self.nz_dim),
                                self.config.max_position_embeddings, self.nz_dim, device=input_ids_or_embedding.device,
                                dtype=self.kv_dtype), 29) for _ in range(self.config_metadata.num_hidden_layers)]
                torch.npu.empty_cache()

        if past_key_value:
            self.k_caches = past_key_value[0]
            self.v_caches = past_key_value[1]
            self.past_key_values_length = self.token_offset[0]
            self.token_offset[:] = self.token_offset[0] + 1
        else:
            self.past_key_values_length = 0
            self.token_offset[:] = input_ids_or_embedding.shape[1]
            self.seq_len_encoder[:] = input_ids_or_embedding.shape[1]

        # set engine's kv cache
        for engine in engine_manager.get_engines():
            engine.set_kv_cache(self.k_caches, self.v_caches)

    def generate_position_ids(self, input_ids_or_embedding: Union[torch.LongTensor, torch.FloatTensor],
                          position_ids: Optional[torch.LongTensor]) -> torch.Tensor:
        """
        Initialize the position ids.

        Args:
            input_ids_or_embedding (torch.LongTensor or torch.FloatTensor): The input embedding tensors or ids.
            position_ids (torch.LongTensor, optional): The position ids tensor.
        
        Returns:
            torch.Tensor: The position ids tensor.
        """
        seq_length = input_ids_or_embedding.shape[1]
        device = input_ids_or_embedding.device

        if position_ids is None:
            position_ids = torch.arange(
                self.past_key_values_length, seq_length + self.past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()
        return position_ids

    def init_mask(self, **kwargs) -> None:
        # init_mask does not need to be called in CausalLMV2; See generate_mask for more details.
        pass

    def generate_mask(self, input_ids_or_embedding: Union[torch.LongTensor, torch.FloatTensor],
                  attention_mask: Optional[torch.Tensor], **kwargs) -> None:
        """
        Generate position IDs for the input sequence.

        Args:
            input_ids_or_embedding (torch.LongTensor or torch.FloatTensor): The input embedding tensors or ids.
            attention_mask (torch.Tensor, optional): The attention mask tensor.
        """
        batch_size, seq_length = input_ids_or_embedding.shape[0], input_ids_or_embedding.shape[1]
        device = input_ids_or_embedding.device

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.bool, device=device
            )
        combined_attention_mask = None
        if seq_length > 1:
            combined_attention_mask = _make_causal_mask(
                input_ids_or_embedding.shape,
                self.dtype,
                device=device,
                past_key_values_length=self.past_key_values_length,
            )
        attention_mask = _expand_mask(attention_mask, self.dtype, tgt_len=seq_length).to(device)
        attention_mask = attention_mask if combined_attention_mask is None else attention_mask + combined_attention_mask
        dim_0 = attention_mask.shape[2]
        dim_1 = attention_mask.shape[3]
        if not self.soc_info.need_nz:
            self.mask_full[:batch_size, :dim_0, :dim_1] = attention_mask.squeeze(1)
        else:
            self.mask_full = torch.zeros((self.batch_num, self.config.max_position_embeddings,
                self.config.max_position_embeddings), dtype=self.dtype, device=input_ids_or_embedding.device)
            self.mask_full[:batch_size, :dim_0, :dim_1] = attention_mask.squeeze(1)
            self.mask_full = torch_npu.npu_format_cast_(
                self.mask_full.view(self.batch_num, self.mask_full.shape[1],
                self.mask_full.shape[2] // self.nz_dim, self.nz_dim).transpose(1, 2).contiguous(), 29)

    def prepare_inputs(
        self,
        input_ids_or_embedding: torch.Tensor,
        position_ids: torch.Tensor,
        is_prefill: bool,
        max_seq_len: int,
    ) -> None:
        """Prepare inputs and runtime paramters for the model.
        It will update `engine_inputs` and `engine_runtime_param`.

        Args:
            input_ids_or_embedding (torch.Tensor): The input embedding tensors or ids.
            position_ids (torch.Tensor): _description_
            is_prefill (bool): A flag indicating whether the inference phase is prefill.
            max_seq_len (int): Maximum sequence length that can be accedpted by the model.
        """
        if is_prefill:
            self.generate_positional_embedding(max_seq_len)

        atten_mask = self.mask_full
        if not is_prefill and self.attn_decode_backend == OpBackend.ACLNN:
            atten_mask = self.mask_full[:, :1, :].to(torch.bool)

        lm_head_indices = self.placeholder
        if is_prefill:
            lm_head_indices = torch.tensor(
                [self.seq_len_encoder[0] - 1], dtype=torch.int64, device=self.torch_device)

        self.engine_inputs = [
            input_ids_or_embedding,
            position_ids.to(torch.int64),
            self.pos_embed_info.cosine_table,
            self.pos_embed_info.sine_table,
            atten_mask,
            self.placeholder,
            self.placeholder,
            self.kv_cache_idx,
            self.token_offset,
            self.placeholder,
            self.seq_len_encoder if is_prefill else self.seq_len_decoder,
            lm_head_indices,
        ]

        self.engine_runtime_param = {
            "tokenOffset": [int(self.token_offset[0])] * self.batch_num,
            "seqLen": [input_ids_or_embedding.shape[1]] * self.batch_num if is_prefill else [1] * self.batch_num
        }

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass of the model.

        Args:
            input_ids (torch.LongTensor, optional): The input ids tensor.
            attention_mask (torch.LongTensor, optional): The attention mask tensor, defaults to None.
            position_ids (torch.LongTensor, optional): The position ids tensor, defaults to None.
            past_key_values prepare_inputs_for_ascend(List[torch.FloatTensor], optional): The past key values tensor,
                defaults to None.
            inputs_embeds (torch.FloatTensor, optional): The input embedding tensor, defaults to None.
            labels (torch.LongTensor, optional): The labels tensor, defaults to None.
            use_cache (bool, optional): Whether to use cache, defaults to None.
            output_attentions (bool, optional): Whether to output attentions, defaults to None.
            output_hidden_states (bool, optional): Whether to output hidden states, defaults to None.
            return_dict (bool, optional): Whether to return a dict, defaults to None.
        
        Returns:
            Union[Tuple, CausalLMOutputWithPast]: A tuple or a CausalLMOutputWithPast object.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        input_ids_or_embedding = inputs_embeds if inputs_embeds is not None else input_ids
        self.update_kv_cache(input_ids_or_embedding, past_key_values)
        position_ids = self.generate_position_ids(input_ids_or_embedding, position_ids)
        self.generate_mask(input_ids_or_embedding, attention_mask)

        is_prefill = True if not past_key_values else False
        self.prepare_inputs(
            input_ids_or_embedding,
            position_ids,
            is_prefill,
            self.config.max_position_embeddings,
        )
        logits = self.execute_engine(is_prefill=is_prefill)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        next_cache = [self.k_caches, self.v_caches] if use_cache else None
        if not return_dict:
            return (loss,) + tuple(v for v in [logits, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def prepare_inputs_for_generation(
            self, input_ids: torch.Tensor, past_key_values: torch.Tensor = None,
            attention_mask: torch.Tensor = None, inputs_embeds: torch.Tensor = None, **kwargs
    ) -> dict:
        """
        Prepare inputs in the generate method. Defined in `GenerationMixin` and should be implemented in
        subclasses of :class:`~transformers.PreTrainedModel`
        """
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            try:
                position_ids = attention_mask.long().cumsum(-1) - 1
            except RuntimeError:
                attention_mask = attention_mask.cpu()
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids = position_ids.npu()
                attention_mask = attention_mask.npu()

            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def get_input_embeddings(self) -> nn.Module:
        """Return the input embeddings."""
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module):
        """Set the input embeddings."""
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        """Return the output embeddings."""
        return self.lm_head.linear

    def set_output_embeddings(self, new_embeddings: nn.Module):
        """Set the output embeddings."""
        self.lm_head.linear = new_embeddings


def _make_causal_mask(
        input_ids_or_embedding_shape: torch.Size,
        dtype: torch.dtype, device: torch.device,
        past_key_values_length: int = 0
) -> torch.Tensor:
    """Make causal mask used for causal attention."""
    bsz, tgt_len = input_ids_or_embedding_shape[:2]
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min if dtype == torch.float16 else 1, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None) -> torch.Tensor:
    """Expand attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`."""
    bsz, src_len = mask.shape
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    return (1.0 - expanded_mask).masked_fill(
        (1.0 - expanded_mask).to(torch.bool), torch.finfo(dtype).min if dtype == torch.float16 else 1)
