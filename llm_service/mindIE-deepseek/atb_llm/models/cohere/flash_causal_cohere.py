# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.
import json
from typing import Optional, List, Tuple

import torch

from .modeling_cohere import FlashCohereModel
from .config_cohere import CohereConfig
from ..base.flash_causal_lm import FlashForCausalLM
from ...utils.data.weight_wrapper import WeightWrapper, AttnWrapper, MlpWrapper, get_module
from ...utils.layers import load_column_multi
from ...utils.dist import get_rank_table_file
from ...utils.layers.norm.fast_layer_norm import NormType


class CohereWeightWrapper(WeightWrapper):

    def register_layer_attn(self, layer, wrapper, quantize_type):
        wrapper_module = get_module(layer, wrapper.wrapper_name)
        # 直接使用 sep，无法pack
        self.register_layer_linear_sep(layer, wrapper, quantize_type, 'attn')
        o_linear = get_module(wrapper_module, wrapper.o_name).linear
        self.register_linear_wrapper(o_linear, quantize_type)

    def register_layer(self, layer, quantize_type):
        self.layer_linear_type.clear()
        self.layer_linear_transpose_types.clear()
        self.register_layer_attn(layer, self.attn_wrapper, quantize_type)
        self.register_layer_mlp(layer, self.mlp_wrapper, quantize_type)
        self.linear_type.append(self.layer_linear_type.copy())
        self.linear_transpose_types.append(self.layer_linear_transpose_types.copy())

        attn_pack_type = get_module(layer, self.attn_wrapper.wrapper_name).pack_type
        mlp_pack_type = get_module(layer, self.mlp_wrapper.wrapper_name).pack_type
        self.pack_quant_type.append([attn_pack_type, mlp_pack_type])


class FlashCohereForCausalLM(FlashForCausalLM):
    def __init__(self, config, weights):
        super().__init__(config, weights)

        self.model = FlashCohereModel(config, weights)
        self.lm_head = load_column_multi(
            config,
            prefixes=["model.embed_tokens"],
            weights=weights,
            head_size=1,
            lm_head=True,
        )

        self.config = config
        if config.num_attention_heads != 0:
            self.head_dim = config.hidden_size // config.num_attention_heads
        self.in_tensor_length = 13
        if self.head_dim != 0:
            self.total_head_nums = config.hidden_size // self.head_dim
        self.acl_encoder_operation_inputs: list[None | torch.Tensor] = [None] * self.in_tensor_length
        self.acl_decoder_operation_inputs: list[None | torch.Tensor] = [None] * self.in_tensor_length

        self.placeholder = torch.zeros(1, dtype=self.dtype, device="npu")
        self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int64, device="npu")

        self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        self.transdata_param = json.dumps({})
        self.transdata_operation.set_param(self.transdata_param)
        self.rope_theta = config.rope_theta
        self.atten_mask_cpu = None

        self.cos_embed = None
        self.sin_embed = None

    def init_position_rotary_embedding(self,
                                       position_ids: torch.Tensor,
                                       max_seq_len: int):
        self.rotary_embedding.update_cohere_cos_sin_cache_total(self.dtype, position_ids.device, max_seq_len)
        self.cos_embed = self.rotary_embedding.get_cos_cached_total()
        self.sin_embed = self.rotary_embedding.get_sin_cached_total()

    def init_ascend_operations(self, config: CohereConfig):
        # 初始化模型
        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch("cohere_CohereDecoderModel")
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch("cohere_CohereDecoderModel")

    def get_weights(self):
        attn_wrapper = AttnWrapper(
            norm_name='input_layernorm',
            wrapper_name='self_attn',
            pack_name="query_key_value",
            sep_names=['q_proj', 'k_proj', 'v_proj'],
            o_name='o_proj'
        )
        mlp_wrapper = MlpWrapper(
            norm_name='post_attention_layernorm', # cohere don't have post_attention_layernorm???
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
            weight_wrapper.register_model_norm(layer.self_attn.q_norm)  # q_norm
            weight_wrapper.register_model_norm(layer.self_attn.k_norm)  # k_norm
            if self.soc_info.need_nz:
                del layer.self_attn
                del layer.mlp
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
        rank_table_file = get_rank_table_file()
        coder_param = {
            "normEps": self.config.layer_norm_eps,
            "normType": NormType.LAYER_NORM,
            "numAttentionHeadsPerRank": self.num_attention_heads,
            "hiddenSizePerAttentionHead": self.head_size,
            "numHiddenLayers": self.config.num_hidden_layers,
            "numKeyValueHeadsPerRank": self.num_key_value_heads,
            "skipWordEmbedding": False,
            "isUnpadInputs": True,
            "isFA": False,
            "isBF16": self.dtype == torch.bfloat16,
            "packQuantType": pack_quant_configs,
            "linearQuantType": linear_types,
            "linearTransposeType": linear_transpose_types,
            "isEmbeddingParallel": True,
            "isLmHeadParallel": True,
            "lmHeadTransposeType": self.lm_head.linear.trans_flag,
            "enableSwiGLU": False,
            "rank": self.tp_rank,
            "worldSize": self.tp_world_size,
            "backend": self.soc_info.communication_backend,
            "rankTableFile": rank_table_file,
            "hiddenSize": self.hidden_size,
            "logitScale": self.config.logit_scale
        }
        self.acl_encoder_operation.set_param(json.dumps({**coder_param, "isPrefill": True}))
        self.acl_decoder_operation.set_param(json.dumps({**coder_param, "isPrefill": False}))
        self.acl_encoder_operation.set_weight(self.ascend_weight)
        self.acl_decoder_operation.set_weight(self.ascend_weight)


    def prepare_inputs_for_ascend(self, input_ids: torch.Tensor,
                                  position_ids: torch.Tensor,
                                  is_prefill: bool,
                                  kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
                                  block_tables: torch.Tensor,
                                  slots: torch.Tensor,
                                  input_lengths: torch.Tensor,
                                  max_seq_len: int,
                                  lm_head_indices: Optional[torch.Tensor] = None,
                                  **kwargs):
        self.rotary_embedding.update_cohere_cos_sin_cache_total(self.dtype,
                                                            self.device,
                                                            self.max_position_embeddings)
        self.cos_embed = self.rotary_embedding.get_cos_cached_total()
        self.sin_embed = self.rotary_embedding.get_sin_cached_total()
        if is_prefill:
            atten_mask = self.attn_mask.get_attn_mask(self.max_base_len, self.dtype, self.device)
            if self.soc_info.need_nz:
                atten_mask = self.transdata_operation.execute([atten_mask])[0]
            if lm_head_indices is None:
                lm_head_indices = torch.tensor(range(input_ids.shape[0]), dtype=torch.int64, device=input_ids.device)
        else:
            atten_mask = self.attn_mask_fake
        self.acl_operation_inputs = [
            input_ids,
            position_ids.to(torch.int64),
            self.cos_embed,
            self.sin_embed,
            atten_mask,
            block_tables.to(torch.int32),
            slots.to(torch.int32),
            self.placeholder,
            self.placeholder,
            self.placeholder,
            input_lengths.to(torch.int32),
            lm_head_indices if is_prefill else self.lm_head_indices_fake
        ]
        acl_param = json.dumps({"seqLen": input_lengths.tolist()})
        return self.acl_operation_inputs, acl_param