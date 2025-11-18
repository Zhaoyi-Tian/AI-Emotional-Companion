#  Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import json
from typing import Optional, List

import torch

from atb_llm.models.minicpm.modeling_minicpm import MiniCpmConfig, MiniCpmModel
from atb_llm.models.base.causal_lm import CausalLM
from atb_llm.utils.layers import AttentionMask, load_column_multi
from atb_llm.utils.data.weight_wrapper import WeightWrapper, AttnWrapper, MlpWrapper
from atb_llm.utils.initial import NPUSocInfo

NPU = "npu"


class MinicpmForCausalLM(CausalLM):
    def __init__(self, config, weights, **kwargs):
        super().__init__(config, weights, **kwargs)
        self.model = MiniCpmModel(config, weights)
        self.soc_info = NPUSocInfo(soc_name='', soc_version=240, need_nz=False, matmul_nd_nz=True)
        self.config = config
        self.is_quant = False
        if self.config.quantize is not None:
            self.is_quant = True
        if self.is_quant:
            self.lm_head = load_column_multi(
                config,
                prefixes=['lm_head'],
                weights=weights,
                head_size=1,
                lm_head=False,
            )
        self.weight_flag = False
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.in_tensor_length = 7
        self.total_head_nums = config.hidden_size // self.head_dim
        self.acl_encoder_operation_inputs: list[None | torch.Tensor] = [None] * self.in_tensor_length
        self.acl_decoder_operation_inputs: list[None | torch.Tensor] = [None] * self.in_tensor_length
        self.placeholder = torch.zeros(1, dtype=self.dtype, device=NPU)
        self.position_embedding_type = config.pe_type
        self.rope_keep_local_base_windows = config.rope_keep_local_base_windows
        self.rope_vanilla_theta = config.rope_vanilla_theta
        self.rope_mscale = config.rope_mscale
        self.rope_given_inv_feq_str = config.rope_given_inv_feq_str
        self.atten_mask_cpu = None
        self.skip_word_embedding = False
        self.cos_embed = None
        self.sin_embed = None
        self.wins_batch_1 = None
        self.decoder_slots = None
        self.all_wins_batch = None
        self.block_tables_global = None
        self.wins_global = None
        self.scale_emb = config.scale_emb
        self.scale_depth = config.scale_depth
        self.dim_model_base = config.dim_model_base
        self.num_hidden_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        self.acl_param = None
        self.ascend_weight = None
        self.max_seq_len = 4096
        self.max_seq_in = 4000
        self.seq_len_all = torch.ones(self.max_seq_len, dtype=torch.int32, device=NPU)
        self.init_cos_sin_table(self.seq_len_all.device)
        self.attn_mask = AttentionMask.static(self.max_seq_len, dtype=torch.float16, mini_type=torch.float16)
        self.attn_mask_full = torch.ones((1, self.max_seq_len), dtype=torch.float16).npu()
        self.position_ids_all = torch.arange(start=0, end=self.max_seq_len, step=1, dtype=torch.int32, device=NPU)
        pack_quant_types_list = [[0, 0] for _ in range(self.config.num_hidden_layers)]
        mlp_linear_types_list = [[0, 0, 0, 0, 0, 0, 0] for _ in range(self.config.num_hidden_layers)]
        mlp_linear_transpose_types_list = [[1, 1, 1, 1, 1, 1, 1] for _ in range(self.config.num_hidden_layers)]

        self.acl_param = {
            "rmsNormEps": self.config.rms_norm_eps,
            "scaleEmb": self.config.scale_emb,
            "scaleDepth": self.config.scale_depth,
            "dimModelBase": self.config.dim_model_base,
            "numHiddenLayers": self.config.num_hidden_layers,
            "hiddenSize":self.config.hidden_size,
            "numAttentionHeads":self.config.num_attention_heads,
            "numKeyValueHeads":self.config.num_key_value_heads,
            "vocabSize":self.config.vocab_size,
            "isGQA":self.config.num_attention_heads != self.config.num_key_value_heads,
            "isQuant":self.is_quant,
            "packQuantType":pack_quant_types_list,
            "linearQuantType":mlp_linear_types_list,
            "linearTransposeType":mlp_linear_transpose_types_list,           
        }

        self.num_hidden_layers = self.config.num_hidden_layers
        self.acl_decoder_operation.set_param(json.dumps({**self.acl_param, "isPrefill": False}))
        self.encoder_flag = True
        self.cos_embed = None
        self.sin_embed = None
        self.rotary_embedding = None
        self.inputs_acl = [None] * self.in_tensor_length
        self.real_prefill_length = 0
        self.padding_flag = False

    def get_trans_info(self, weight_wrapper):
        linear_types = weight_wrapper.linear_type
        pack_quant_configs = weight_wrapper.pack_quant_type
        linear_transpose_types = weight_wrapper.linear_transpose_types
        return linear_transpose_types, linear_types, pack_quant_configs

    def init_position_rotary_embedding(self,
                                       position_ids: torch.Tensor,
                                       max_seq_len: int):
        self.rotary_embedding.update_cos_sin_cache_total(self.dtype, position_ids.device, max_seq_len)
        self.cos_embed = self.rotary_embedding.get_cos_cached_total()
        self.sin_embed = self.rotary_embedding.get_sin_cached_total()

    def init_ascend_operations(self, config: MiniCpmConfig):
        # 初始化模型
        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch("minicpm_DecoderModelEdge")
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch("minicpm_DecoderModelEdge")

    def get_nd_weights(self):
        weights = []
        weights.append(self.model.embed_tokens_weight)
        for i, block in enumerate(self.model.layers):
            weights.append(block.state_dict()["input_layernorm.weight"])
            weights.append(block.state_dict()["self_attn.query_key_value.linear.weight"])
            weights.append(block.state_dict()["self_attn.o_proj.linear.weight"])
            weights.append(block.state_dict()["mlp.gate_up_proj.linear.weight"])
            weights.append(self.model.layers[i].mlp.down_proj_weight)
            weights.append(block.state_dict()["post_attention_layernorm.weight"])
        weights.append(self.model.norm.weight)
        weights.append(self.model.embed_tokens_weight)
        return weights

    def get_attn_mlp_wrapper(self):
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
        return attn_wrapper, mlp_wrapper
    
    def get_weights(self):
        attn_wrapper, mlp_wrapper = self.get_attn_mlp_wrapper()
        weight_wrapper = WeightWrapper(self.soc_info, self.tp_rank, attn_wrapper, mlp_wrapper)
        weight_wrapper.register_embedding(self.model.embed_tokens)
        for i in range(self.num_layers):
            layer = self.model.layers[i]
            weight_wrapper.register_layer(layer, self.quantize)
            if self.soc_info.need_nz:
                del layer.self_attn
                del layer.mlp
                del layer.post_attention_layernorm
            if self.config.quantization_config.kv_quant_type is not None:
                weight_wrapper.register_layer_kvquant(layer)            
        weight_wrapper.register_model_norm(self.model.norm)
        weight_wrapper.register_model_lmhead_quant(self.lm_head)
        
        return weight_wrapper

    def init_cos_sin_table(self, device):
        dtype = torch.float16
        max_seq_len = self.config.max_position_embeddings
        from atb_llm.utils.layers import PositionRotaryEmbedding
        self.rotary_embedding = PositionRotaryEmbedding.static(dim=64, base=10000.0,
                                                               device="cpu", scaling_factor=1.0).to(device)

        self.rotary_embedding.update_cos_sin_cache_total(dtype, device, max_seq_len)
        self.cos_embed = self.rotary_embedding.get_cos_cached_total()
        self.sin_embed = self.rotary_embedding.get_sin_cached_total()

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None
    ):

        self.inputs_acl = [None] * (7 + self.config.num_hidden_layers * 2)

        if self.encoder_flag or past_key_values is None:
            prefill_length = len(position_ids[0])
            if self.weight_flag is False or past_key_values is None:
                self.set_weight(prefill_length)
                self.weight_flag = True
            self.attn_mask_fake = self.get_attn_mask(attention_mask, input_ids, position_ids, prefill_length)

            self.inputs_acl[0:self.in_tensor_length] = [
                input_ids,
                self.attn_mask_fake,
                position_ids,
                self.cos_embed,
                self.sin_embed,
                self.seq_len_all[:prefill_length],
                self.seq_len_all[:1]]

            past_keys_prefill, past_values_prefill = self.init_empty_past_key_value(prefill_length)

            self.build_past_key_value(past_keys_prefill, past_values_prefill)
            torch.npu.synchronize()
            outputs_acl = self.acl_encoder_operation.execute(self.inputs_acl, json.dumps({**self.acl_param}))
            torch.npu.synchronize()
            logits, past_key_values = self.get_past_key_value(outputs_acl)

            self.encoder_flag = False

            return logits, past_key_values
        else:
            decode_position_ids = self.get_position_id(position_ids)

            self.inputs_acl[0:self.in_tensor_length] = [
                input_ids,
                self.attn_mask_full[:, :decode_position_ids + 1],
                self.position_ids_all[decode_position_ids:decode_position_ids + 1],
                self.cos_embed,
                self.sin_embed,
                self.seq_len_all[:1],
                self.seq_len_all[:1]]

            (past_keys, past_values) = map(list, zip(*past_key_values))

            self.build_past_key_value(past_keys, past_values)
            torch.npu.synchronize()
            outputs_acl = self.acl_decoder_operation.execute(self.inputs_acl, json.dumps({**self.acl_param}))
            torch.npu.synchronize()
            logits, past_key_values = self.get_past_key_value(outputs_acl)

            return logits, past_key_values

    def set_weight(self, prefill_length):
        if self.is_quant:
            weight_wrapper = self.get_weights()
            weights = weight_wrapper.weights
        else:
            weights = self.get_nd_weights()
        self.init_cos_sin_table(weights[0].device)
        self.acl_param_encoder = json.dumps({**self.acl_param,
                                             "isPrefill": True,
                                             "seqLength": prefill_length
                                             })
        self.acl_encoder_operation.set_param(self.acl_param_encoder)
        self.acl_decoder_operation.set_weight(weights)
        self.acl_encoder_operation.set_weight(weights)

    def get_position_id(self, position_ids):
        if len(position_ids[0]) > 1:
            decode_position_ids = len(position_ids[0])
        else:
            decode_position_ids = int(position_ids[0][0].item())
        if self.padding_flag:
            decode_position_ids = decode_position_ids + self.max_seq_in - self.real_prefill_length
        return decode_position_ids

    def init_empty_past_key_value(self, prefill_length):
        self.kv_int = \
            torch.zeros((1, self.config.num_key_value_heads, prefill_length, 64), dtype=torch.float16).npu()
        past_keys_prefill = []
        past_values_prefill = []
        for _ in range(self.num_hidden_layers):
            past_keys_prefill.append(self.kv_int)
            past_values_prefill.append(self.kv_int)
        return past_keys_prefill, past_values_prefill

    def build_past_key_value(self, past_keys_prefill, past_values_prefill):
        self.inputs_acl[self.in_tensor_length:self.in_tensor_length + self.num_hidden_layers] = past_keys_prefill
        self.inputs_acl[
            self.in_tensor_length + self.num_hidden_layers : self.in_tensor_length
                                                             + 1 + self.num_hidden_layers * 2] = past_values_prefill

    def get_past_key_value(self, outputs_acl):
        presents_acl = ()
        past_kv_acl = (outputs_acl[1], outputs_acl[1 + self.num_hidden_layers],)
        presents_acl += (past_kv_acl,)
        for i in range(self.num_hidden_layers - 1):
            past_kv_acl = (outputs_acl[i + 2], outputs_acl[i + 2 + self.num_hidden_layers])
            presents_acl += (past_kv_acl,)
        logits = outputs_acl[0]
        return logits, presents_acl

    def get_attn_mask(self, attention_mask, input_ids, position_ids, prefill_length):
        if self.padding_flag:
            self.real_prefill_length = torch.max(position_ids[0]).item() + 1
            attn_mask_fake = self.attn_mask.get_attn_mask_padding(
                prefill_length, inputs_ids=input_ids.to(torch.float16), attention_mask=attention_mask)
        else:
            attn_mask_fake = self.attn_mask.get_attn_mask(prefill_length,
                                                               dtype=torch.float16, device="npu",
                                                               mini_type=torch.float16)
        return attn_mask_fake
