#  Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import json
from typing import Optional, List, Tuple, Union
import torch
import torch_npu

from atb_llm.utils.layers import load_column_multi
from atb_llm.utils.log import logger
from atb_llm.models.base.causal_lm import CausalLM
from atb_llm.utils.layers import AttentionMask
from atb_llm.models.qwen2.modeling_qwen2 import FlashQwenModel
from atb_llm.models.qwen2.config_qwen2 import Qwen2Config
from atb_llm.utils.data.weight_wrapper import WeightWrapper, AttnWrapper, MlpWrapper
from transformers.modeling_outputs import CausalLMOutputWithPast

CAUSAL_QWEN2_DEVICE_NAME = "npu"


class Qwen2ForCausalLM(CausalLM):
    def __init__(self, config, weights):
        self.acl_encoder_operation = None
        self.acl_decoder_operation = None
        super().__init__(config, weights)
        self.transformer = FlashQwenModel(config, weights)
        if config.tie_word_embeddings:
            self.lm_head = load_column_multi(
                config,
                prefixes=["model.embed_tokens"],
                weights=weights,
                head_size=1,
                lm_head=True,
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
        if self.dtype != torch.float16:
            raise ValueError(f"unsupported type: {self.dtype}, 当前仅支持`float16`类型,请修改权重文件config.json中的`torch_dtype`字段")
        self.place_holder = torch.zeros(1, dtype=torch.float16, device=CAUSAL_QWEN2_DEVICE_NAME)
        self.kv_cache_idx = torch.zeros(1, dtype=torch.int32, device=CAUSAL_QWEN2_DEVICE_NAME)
        self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int64, device=CAUSAL_QWEN2_DEVICE_NAME)
        self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        self.transdata_param = json.dumps({})
        self.transdata_operation.set_param(self.transdata_param)
        self.encoder_flag = True
        self.acl_param = None
        self.acl_operation_inputs = [None] * 12
        self.ascend_weight = None
        self.max_seq_len = 4096
        self.attn_mask = AttentionMask.static(self.max_seq_len, dtype=torch.float16)
        self.attn_mask_full = torch.ones((1, self.max_seq_len), dtype=torch.float16).npu()
        self.seq_len_all = torch.ones(self.max_seq_len, dtype=torch.int32, device="npu")
        self.is_quant = False
        if self.config.quantize is not None:
            self.is_quant = True

    def init_ascend_operations(self, config: Qwen2Config):
        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch("qwen_DecoderModelEdge")
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch("qwen_DecoderModelEdge")
        logger.info(">>>> qwen1.5_flash_attention_model is called.")

    def get_weights(self):
        attn_wrapper = AttnWrapper(
            norm_name='ln_1',
            wrapper_name='attn',
            pack_name='c_attn',
            sep_names=None,
            o_name='c_proj'
        )
        mlp_wrapper = MlpWrapper(
            norm_name='ln_2',
            wrapper_name='mlp',
            pack_name='w2_w1',
            sep_names=None,
            down_name='c_proj'
        )
        weight_wrapper = WeightWrapper(self.soc_info, self.tp_rank, attn_wrapper, mlp_wrapper)
        weight_wrapper.register_embedding(self.transformer.wte)
        for i in range(self.num_layers):
            layer = self.transformer.h[i]
            weight_wrapper.register_layer(layer, self.quantize)
            if self.soc_info.need_nz:
                del layer.attn
                del layer.ln_2
                del layer.mlp
            if self.config.quantization_config.kv_quant_type is not None:
                weight_wrapper.register_layer_kvquant(layer)
        weight_wrapper.register_model_norm(self.transformer.ln_f)
        weight_wrapper.register_model_lmhead(self.lm_head)
        
        return weight_wrapper

    def init_ascend_weight(self):
        weight_wrapper = self.get_weights()
        self.ascend_weight = weight_wrapper.weights
        linear_types = weight_wrapper.linear_type
        pack_quant_configs = weight_wrapper.pack_quant_type
        linear_transpose_types = weight_wrapper.linear_transpose_types


        acl_param_dict = {
            "isFA": True,
            "isBF16": False,
            "withEmbedding": True,
            "isEmbeddingParallel": True,
            "isLmHeadParallel": True,
            "linearTransposeType": linear_transpose_types,
            "lmHeadTransposeType": self.lm_head.linear.trans_flag,
            "supportSwiGLU": False if self.soc_info.need_nz else True,
            "rmsNormEps": self.config.rms_norm_eps,
            "numAttentionHeadsPerRank": self.num_attention_heads,
            "hiddenSizePerAttentionHead": self.head_size,
            "numHiddenLayers": self.config.num_hidden_layers,
            "numKeyValueHeadsPerRank": self.num_key_value_heads,
            "rank": self.tp_rank,
            "worldSize": self.tp_world_size,
            "backend": self.soc_info.communication_backend,
            "packQuantType": pack_quant_configs,
            "linearQuantType": linear_types,
            "kvQuant": self.config.quantization_config.kv_quant_type is not None,
            "hiddenSize": self.hidden_size,
            "vocabSize": self.config.vocab_size,
            "isQuant":self.is_quant,
        }

        acl_param_encoder = json.dumps({**acl_param_dict, "isPrefill": True, "supportLcoc": self.lcoc_enable})
        acl_param_decoder = json.dumps({**acl_param_dict, "isPrefill": False, "supportLcoc": False})

        self.acl_encoder_operation.set_param(acl_param_encoder)
        self.acl_decoder_operation.set_param(acl_param_decoder)

        self.acl_encoder_operation.set_weight(self.ascend_weight)
        self.acl_decoder_operation.set_weight(self.ascend_weight)

    def init_kvcache(self, input_ids_or_embedding, past_key_value):
        super().init_kvcache(input_ids_or_embedding, past_key_value)
        self.acl_encoder_operation.set_kv_cache(self.k_cache, self.v_cache)
        self.acl_decoder_operation.set_kv_cache(self.k_cache, self.v_cache)

    def prepare_inputs_for_ascend(
            self,
            input_ids_or_embedding: torch.Tensor,
            position_ids: torch.Tensor,
            cu_seqlen_prefill: Optional[bool],
            max_seq_len: int,
            past_key_values : Optional[List[torch.FloatTensor]] 
    ):
        self.rotary_embedding.update_cos_sin_cache_total(self.dtype, self.device, max_seq_len)
        cos_table = self.rotary_embedding.get_cos_cached_total()
        sin_table = self.rotary_embedding.get_sin_cached_total()
        prefill_length = len(position_ids[0])
        self.attn_mask_fake = None
        self.attn_mask_fake = self.attn_mask.get_attn_mask(prefill_length, dtype=torch.float16, device="npu")
        self.kv_int =\
            torch.zeros((1, self.num_key_value_heads, prefill_length, 128), dtype=torch.float16).npu()
        past_keys_prefill = []
        past_values_prefill = []
        for _ in range(self.config.num_hidden_layers):
            past_keys_prefill.append(self.kv_int)
            past_values_prefill.append(self.kv_int)
        if cu_seqlen_prefill:
            self.acl_param = json.dumps({
                "tokenOffset": [int(self.token_offset[0])] * self.batch_num,
                "seqLen": [input_ids_or_embedding.shape[1]] * self.batch_num
            })
            self.acl_operation_inputs = [None] * (12 + self.config.num_hidden_layers * 2)
            self.acl_operation_inputs[0:12] = [
                input_ids_or_embedding,  # IN_TENSOR_INPUTIDS
                position_ids,  # IN_TENSOR_POSITIONIDS
                cos_table,  # IN_TENSOR_COSTABLE
                sin_table,  # IN_TENSOR_SINTABLE
                self.attn_mask_fake,  # IN_TENSOR_ATTENTIONMASK
                self.seq_len_all[:1],  # IN_TENSOR_BLOCK_TABLES
                self.seq_len_all[:1],  # IN_TENSOR_SLOTS
                self.kv_cache_idx,  # IN_TENSOR_KV_CACHE_IDX
                self.token_offset,  # IN_TENSOR_TOKEN_OFFSET
                self.seq_len_encoder,  # IN_TENSOR_SEQ_LENGTHS
                torch.tensor([self.seq_len_encoder[0] - 1], dtype=torch.int64, device=self.device),
                # IN_TENSOR_LOGTIS_INDICES
                self.seq_len_all[:1]  # IN_HOLDER
            ]
            self.acl_operation_inputs[12:12 + self.config.num_hidden_layers] = past_keys_prefill
            self.acl_operation_inputs[12 + self.config.num_hidden_layers: 13 +
                                      self.config.num_hidden_layers * 2] = past_values_prefill
        else:
            self.acl_param = json.dumps({
                "tokenOffset": [int(self.token_offset[0])] * self.batch_num,
                "seqLen": self.acl_param_seq_len_decoder
            })
            if len(position_ids[0]) > 1 :
                decode_position_ids = len(position_ids[0])
            else:
                decode_position_ids = position_ids[0][0].item()
            self.acl_operation_inputs = [None] * (12 + self.config.num_hidden_layers * 2)
            decode_attention_mask = self.attn_mask_full[:, :decode_position_ids + 1]
            self.acl_operation_inputs[0:12] = [
                input_ids_or_embedding,  # IN_TENSOR_INPUTIDS
                position_ids,  # IN_TENSOR_POSITIONIDS
                cos_table,  # IN_TENSOR_COSTABLE
                sin_table,  # IN_TENSOR_SINTABLE
                decode_attention_mask,  # IN_TENSOR_ATTENTIONMASK
                self.seq_len_all[:1],  # IN_TENSOR_BLOCK_TABLES
                self.seq_len_all[:1],  # IN_TENSOR_SLOTS
                self.kv_cache_idx,  # IN_TENSOR_KV_CACHE_IDX
                self.token_offset,  # IN_TENSOR_TOKEN_OFFSET
                self.seq_len_decoder,  # IN_TENSOR_SEQ_LENGTHS
                self.lm_head_indices_fake,  # IN_TENSOR_LOGTIS_INDICES
                self.seq_len_all[:1],  # IN_HOLDER
            ]
            (past_keys, past_values) = map(list, zip(*past_key_values))
            self.acl_operation_inputs[12:12 + self.config.num_hidden_layers] = past_keys
            self.acl_operation_inputs[12 + self.config.num_hidden_layers: 
                                      13 + self.config.num_hidden_layers * 2] = past_values
        return self.acl_operation_inputs, self.acl_param

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
            return_dict: Optional[bool] = True,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if not self.ascend_weight:
            self.init_ascend_weight()
        self.init_kvcache(inputs_embeds if inputs_embeds is not None else input_ids, past_key_values)
        cu_seqlen_prefill = True if not past_key_values else False
        acl_operation_inputs, acl_param = self.prepare_inputs_for_ascend(
            input_ids,
            position_ids,
            cu_seqlen_prefill,
            self.max_position_embeddings,
            past_key_values)
        if cu_seqlen_prefill:
            outputs_acl = self.acl_encoder_operation.execute(acl_operation_inputs, acl_param)
        else:
            outputs_acl = self.acl_decoder_operation.execute(acl_operation_inputs, acl_param)
        presents_acl = ()
        pastkv_acl = (outputs_acl[1], outputs_acl[1 + self.config.num_hidden_layers],)
        presents_acl += (pastkv_acl,)
        for i in range(self.config.num_hidden_layers - 1):
            pastkv_acl = (outputs_acl[i + 2], outputs_acl[i + 2 + self.config.num_hidden_layers])
            presents_acl += (pastkv_acl,)
        logits = outputs_acl[0]
        past_key_values = presents_acl
        attentions = None
        hidden_states = None
        loss = None
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            attentions=attentions,
        )