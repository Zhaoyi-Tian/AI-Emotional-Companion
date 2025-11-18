# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

from typing import Optional, List, Tuple, Union
import json
import torch

from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.utils.layers import AttentionMask
from transformers.modeling_outputs import CausalLMOutputWithPast

from .modeling_llama import FlashLlamaModel
from ..base.causal_lm import CausalLM
from ...utils.env import ENV
from ...utils.data.weight_wrapper import WeightWrapper, AttnWrapper, MlpWrapper
from ...utils.layers.norm.fast_layer_norm import NormType
from ...utils.layers.embedding.position_rotary_embedding import PositionEmbeddingType
from ...utils.layers import load_column_multi, TensorParallelHead
from .config_llama import LlamaConfig


class LlamaForCausalLM(CausalLM):
    def __init__(self, config, weights, **kwargs):
        super().__init__(config, weights, **kwargs)

        self.model = FlashLlamaModel(config, weights, attn_decode_backend=self.attn_decode_backend)

        if config.tie_word_embeddings:
            self.lm_head = TensorParallelHead.load(
                config,
                prefix="model.embed_tokens",
                weights=weights,
                is_norm=True,
            )
        else:
            self.lm_head = load_column_multi(
                config,
                prefixes=["lm_head"],
                weights=weights,
                head_size=1,
                lm_head=True,
            )

        self.config = config
        self.placeholder = torch.zeros(1, dtype=self.dtype).npu()
        self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int64).npu()

        self.acl_encoder_operation_inputs = [None] * 7
        self.acl_decoder_operation_inputs = [None] * 7


        self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        self.transdata_param = json.dumps({})
        self.transdata_operation.set_param(self.transdata_param)

        self.position_embedding_type = config.pe_type
        if self.position_embedding_type != "ROPE" and self.position_embedding_type != "ALIBI":
            error_msg = f"Unsupported type: {self.position_embedding_type}, " \
                f"this field is only support for type: `ROPE` and `ALIBI`, loaded from config.json -> pe_type."
            logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise AssertionError(error_msg)
        self.skip_word_embedding = config.skip_word_embedding
        self.acl_param = None
        self.ascend_weight = None
        self.cos_embed = None
        self.sin_embed = None
        self.max_seq_len = 4096
        self.max_seq_in = 4000
        self.attn_mask = AttentionMask.static(self.max_seq_len, dtype=torch.float16)
        self.attn_mask_full = torch.ones((1, self.max_seq_len), dtype=torch.float16).npu()

        self.seq_len_all = torch.ones(self.max_seq_len, dtype=torch.int32, device="npu")

        self.position_ids_all = torch.arange(start=0, end=self.max_seq_len, step=1, dtype=torch.int32, device="npu")

        self.is_quant = False if self.config.quantize is None else True

    def init_position_rotary_embedding(self,
                                       position_ids: torch.Tensor,
                                       max_seq_len: int):
        if self.config.rope_scaling is None:
            self.rotary_embedding.update_cos_sin_cache_total(self.dtype, position_ids.device, max_seq_len)
        else:
            scaling_type = self.config.rope_scaling.rope_type
            if scaling_type is None:
                scaling_type = self.config.rope_scaling.type
            if scaling_type == "linear":
                self.rotary_embedding.update_cos_sin_cache_total(self.dtype, self.device, max_seq_len)
            elif scaling_type == "llama3":
                self.rotary_embedding.update_llama3_cos_sin_cache_total(
                    self.config, self.dtype, self.device, max_seq_len)
            elif scaling_type == "dynamic":
                logger.error("Error: only support scaling type: linear, check your config.json: scaling type", 
                             ErrorCode.ATB_MODELS_MODEL_PARAM_JSON_INVALID)
                raise ValueError(f"not support RoPE scaling type {scaling_type}")
            else:
                logger.error("Error: only support scaling type: linear, check your config.json: scaling type", 
                             ErrorCode.ATB_MODELS_MODEL_PARAM_JSON_INVALID)
                raise ValueError("Unknown RoPE scaling type, check your config.json")
        self.cos_embed = self.rotary_embedding.get_cos_cached_total()
        self.sin_embed = self.rotary_embedding.get_sin_cached_total()

    def init_ascend_operations(self, config: LlamaConfig):
        # 初始化模型
        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch("llama_DecoderModelEdge")
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch("llama_DecoderModelEdge")

    def get_weights(self):
        attn_wrapper = AttnWrapper(
            norm_name='input_layernorm',
            wrapper_name='self_attn',
            pack_name='query_key_value',
            sep_names=['q_proj', 'k_proj', 'v_proj'],
            o_name='o_proj'
        )
        mlp_wrapper = MlpWrapper(
            norm_name='post_attention_layernorm',
            wrapper_name='mlp',
            pack_name='gate_up_proj',
            sep_names=['gate_proj', 'up_proj'],
            down_name='down_proj'
        )
        weight_wrapper = WeightWrapper(self.soc_info, self.tp_rank, attn_wrapper, mlp_wrapper)
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

    def init_ascend_weight(self):
        weight_wrapper = self.get_weights()
        self.ascend_weight = weight_wrapper.weights
        linear_types = weight_wrapper.linear_type
        pack_quant_configs = weight_wrapper.pack_quant_type
        linear_transpose_types = weight_wrapper.linear_transpose_types
        # 设置模型参数
        if self.position_embedding_type == "ROPE":
            position_embedding_type = "ROPE"
        else:
            position_embedding_type = PositionEmbeddingType.ALIBI
        coder_param = {
            "normEps": self.config.rms_norm_eps,
            "normType": NormType.RMS_NORM,
            "numAttentionHeadsPerRank": self.num_attention_heads,
            "hiddenSizePerAttentionHead": self.head_size,
            "numHiddenLayers": self.config.num_hidden_layers,
            "numKeyValueHeadsPerRank": self.num_key_value_heads,
            "skipWordEmbedding": False,
            "isUnpadInputs": False,
            "isFA": True,
            "isBF16": self.dtype == torch.bfloat16,
            "isQuant": self.is_quant,
            "packQuantType": pack_quant_configs,
            "linearQuantType": linear_types,
            "linearTransposeType": linear_transpose_types,
            "isEmbeddingParallel": self.model.parallel_embedding,
            "isLmHeadParallel": True,
            "lmHeadTransposeType": self.lm_head.linear.trans_flag,
            "enableSwiGLU": False if self.soc_info.need_nz else True,
            "rank": self.tp_rank,
            "worldSize": self.tp_world_size,
            "backend": self.soc_info.communication_backend,
            "rankTableFile": ENV.rank_table_file,
            "positionEmbeddingType": position_embedding_type,
            "attnBackend": self.attn_decode_backend,
            "hiddenSize": self.hidden_size,
            "enableAddNorm": False,
            "quantGroupSize": self.config.quantization_config.group_size,
            "vocabSize": self.config.vocab_size,
            "head_dim": self.hidden_size / self.num_attention_heads,
        }
        encoder_param = {
            **coder_param, "isPrefill": True,
            "enableLcoc": self.lcoc_enable, "skipWordEmbedding": self.skip_word_embedding
        }
        decoder_param = {**coder_param, "isPrefill": False, "enableLcoc": False}
        self.acl_encoder_operation.set_param(json.dumps({**encoder_param}))
        self.acl_decoder_operation.set_param(json.dumps({**decoder_param}))

        self.acl_encoder_operation.set_weight(self.ascend_weight)
        self.acl_decoder_operation.set_weight(self.ascend_weight)

    def init_kvcache(self, input_ids_or_embedding, past_key_value):
        super().init_kvcache(input_ids_or_embedding, past_key_value)
        self.acl_encoder_operation.set_kv_cache(self.k_cache, self.v_cache)
        self.acl_decoder_operation.set_kv_cache(self.k_cache, self.v_cache)

    def prepare_inputs_for_ascend(self,
                                  input_ids_or_embedding: torch.Tensor,
                                  position_ids: torch.Tensor,
                                  cu_seqlen_prefill: Optional[bool],
                                  max_seq_len: int,
                                  past_key_values: Optional[List[torch.FloatTensor]],
                                  ):
        self.init_position_rotary_embedding(position_ids, max_seq_len)

        if cu_seqlen_prefill:
            self.acl_param = json.dumps({
                "tokenOffset": [int(self.token_offset[0])] * self.batch_num,
                "seqLen": [input_ids_or_embedding.shape[1]] * self.batch_num
            })
            prefill_length = len(position_ids[0])
            self.attn_mask_fake = None
            self.attn_mask_fake = self.attn_mask.get_attn_mask(prefill_length, dtype=torch.float16, device="npu")

            self.acl_encoder_operation_inputs = [None] * (7 + self.config.num_hidden_layers * 2)

            self.acl_encoder_operation_inputs[0:7] = [
                input_ids_or_embedding,
                position_ids.to(torch.int64),
                self.cos_embed,
                self.sin_embed,
                self.attn_mask_fake,
                self.seq_len_all[:prefill_length],
                self.seq_len_all[:1]
            ]
            self.kv_int =\
                torch.zeros((1, self.num_key_value_heads, prefill_length, 64), dtype=torch.float16).npu()
            
            past_keys_prefill = []
            past_values_prefill = []
            for _ in range(self.config.num_hidden_layers):
                past_keys_prefill.append(self.kv_int)
                past_values_prefill.append(self.kv_int)

            self.acl_encoder_operation_inputs[7:7 + self.config.num_hidden_layers] = past_keys_prefill
            self.acl_encoder_operation_inputs[7 + self.config.num_hidden_layers: 8 + 
                                              self.config.num_hidden_layers * 2] = past_values_prefill

            return self.acl_encoder_operation_inputs, self.acl_param
        else:
            if len(position_ids[0]) > 1 :
                decode_position_ids = len(position_ids[0])
            else:
                decode_position_ids = position_ids[0][0].item()
            decode_attention_mask = self.attn_mask_full[:, :decode_position_ids + 1]

            self.acl_decoder_operation_inputs = [None] * (7 + self.config.num_hidden_layers * 2)

            self.acl_decoder_operation_inputs[0:7] = [
                input_ids_or_embedding,
                self.position_ids_all[decode_position_ids:decode_position_ids + 1],
                self.cos_embed,
                self.sin_embed,
                decode_attention_mask,
                self.seq_len_all[:1],
                self.seq_len_all[:1],
            ]

            (past_keys, past_values) = map(list, zip(*past_key_values))
            self.acl_decoder_operation_inputs[7:7 + self.config.num_hidden_layers] = past_keys
            self.acl_decoder_operation_inputs[7 + self.config.num_hidden_layers: 8 +
                                              self.config.num_hidden_layers * 2] = past_values
            return self.acl_decoder_operation_inputs, self.acl_param

    def forward(
            self,
            input_ids: torch.LongTensor = None,
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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        if not self.ascend_weight:
            self.init_ascend_weight()

        self.init_kvcache(inputs_embeds if inputs_embeds is not None else input_ids, past_key_values)

        cu_seqlen_prefill = True if not past_key_values else False
        acl_inputs, acl_param = self.prepare_inputs_for_ascend(
            inputs_embeds if inputs_embeds is not None else input_ids,
            position_ids,
            cu_seqlen_prefill,
            self.max_position_embeddings,
            past_key_values,
        )

        if cu_seqlen_prefill:
            outputs_acl = self.acl_encoder_operation.execute(acl_inputs, acl_param)
        else:
            outputs_acl = self.acl_decoder_operation.execute(acl_inputs, acl_param)

        presents_acl = ()
        pastkv_acl = (outputs_acl[1], outputs_acl[1 + self.config.num_hidden_layers],)
        presents_acl += (pastkv_acl,)
        for i in range(self.config.num_hidden_layers - 1):
            pastkv_acl = (outputs_acl[i + 2], outputs_acl[i + 2 + self.config.num_hidden_layers])
            presents_acl += (pastkv_acl,)
        logits = outputs_acl[0]
        past_key_values = presents_acl
        loss = None

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
