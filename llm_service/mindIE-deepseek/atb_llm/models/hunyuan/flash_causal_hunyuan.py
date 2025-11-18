# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json
import math
from typing import Optional, List, Tuple
import torch

from atb_llm.models.base.flash_causal_lm import FlashForCausalLM
from atb_llm.models.hunyuan.modeling_hunyuan import FlashHunyuanModel
from atb_llm.models.hunyuan.config_hunyuan import HunyuanConfig
from atb_llm.models.hunyuan.position_embedding_hunyuan import HunyuanRotaryEmbedding
from atb_llm.models.hunyuan.weight_wrapper_hunyuan import ClaWrapper, HunyuanWeightWrapper
from atb_llm.utils.data.moe_weight_wrapper import MoeMlpWrapper
from atb_llm.utils.env import ENV
from atb_llm.utils.layers.norm.fast_layer_norm import NormType
from atb_llm.utils.layers import (
    TensorEmbedding,
    load_column_multi,
)


class FlashHunyuanForCausalLM(FlashForCausalLM):
    def __init__(self, config, weights, **kwargs):
        super().__init__(config, weights, **kwargs)
        self.model = FlashHunyuanModel(config, weights)
        self.lm_head = load_column_multi(
            config,
            prefixes=["model.embed_tokens"],
            weights=weights,
            head_size=1,
            lm_head=True,
        )
        self.config = config
        self.placeholder = torch.zeros(1, dtype=self.dtype, device=self.device)
        self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int64, device=self.device)
        self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        self.transdata_param = json.dumps({})
        self.transdata_operation.set_param(self.transdata_param)
        self.embed_tokens = TensorEmbedding(
            prefix="model.embed_tokens", weights=weights
        )

        self.hidden_dim = config.hidden_size
        self.expert_array = []

        self.expert_group = torch.arange(1024, dtype=torch.int32).npu() # 1024: const for groupedTopK
        self.one_hot = torch.tensor([1], dtype=torch.int32).npu()
        self.zero_hot = torch.tensor([0], dtype=torch.int32).npu()

        self.num_of_experts = config.n_routed_experts
        self.num_of_selected_experts = [config.num_experts_per_tok]
        self.tp = config.tp if config.tp else True # Defaulting the model to tensor parallel
        self.n_shared_experts = config.n_shared_experts if config.n_shared_experts else 0
        self.first_k_dense_replace = 0
        self.cla_share_factor = config.cla_share_factor
        self.softmax_scale = (self.head_size) ** (-0.5)
        self.scaling_alpha = self.config.rope_scaling_dict.get("alpha", 1000.0)
        self.rotary_embedding = HunyuanRotaryEmbedding.static(dim=self.head_size, base=self.rope_theta,
                                                        device="cpu", scaling_alpha=self.scaling_alpha).to(self.device)
        self.expert_parallel_degree = 1 if self.tp else self.tp_world_size
        self.enable_fused_routing = False if self.soc_info.need_nz else True

    def init_ascend_operations(self, config: HunyuanConfig):
        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch("hunyuan_DecoderModel")
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch("hunyuan_DecoderModel")

    def init_weight_wrapper(self):
        attn_wrapper = ClaWrapper(
            input_norm_name='input_layernorm',
            wrapper_name='self_attn',
            pack_name='qkv_proj',
            o_name='o_proj',
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            cla_share_factor=self.cla_share_factor,
            q_norm_name='query_layernorm',
            k_norm_name='key_layernorm',
        )
        moe_mlp_wrapper = MoeMlpWrapper(
            norm_name='post_attention_layernorm',
            router_name='gate',
            wrapper_name='mlp',
            pack_name='gate_up_proj',
            sep_names=['gate_proj', 'up_proj'],
            down_name='down_proj',
            shared_experts=(self.n_shared_experts > 0)
        )
        weight_wrapper = HunyuanWeightWrapper(self.soc_info, self.tp_rank,
                                                attn_wrapper, moe_mlp_wrapper,
                                                self.num_of_experts,
                                                self.config.intermediate_size // self.tp_world_size)
        weight_wrapper.register_embedding(self.model.embed_tokens)
        return weight_wrapper

    def get_weights(self):
        weight_wrapper = self.init_weight_wrapper()
        for i in range(self.num_layers):
            layer = self.model.layers[i]
            weight_wrapper.register_moe_layer(layer, self.quantize, dense_layer=False)
            del layer.mlp
            torch.npu.empty_cache()

            if self.soc_info.need_nz:
                del layer.self_attn
                del layer.post_attention_layernorm
                torch.npu.empty_cache()
        weight_wrapper.register_model_norm(self.model.norm)
        weight_wrapper.register_model_lmhead(self.lm_head)
        return weight_wrapper

    def init_ascend_weight(self):
        weight_wrapper = self.get_weights()
        self.ascend_weight = weight_wrapper.weights
        pack_quant_configs = weight_wrapper.pack_quant_type

        attn_linear_types = weight_wrapper.attn_linear_types
        mlp_linear_types = weight_wrapper.mlp_linear_types
        moe_linear_types = weight_wrapper.moe_linear_types

        attn_linear_transpose_types = weight_wrapper.attn_linear_transpose_types
        mlp_linear_transpose_types = weight_wrapper.mlp_linear_transpose_types
        moe_linear_transpose_types = weight_wrapper.moe_linear_transpose_types

        # compatible with linearQuantType
        linear_quant_types = attn_linear_types.copy()
        linear_transpose_types = attn_linear_transpose_types.copy()
        for i in range(self.num_layers):
            linear_quant_types[i].append(attn_linear_types[i][-1])
            linear_transpose_types[i].append(-1)

        coder_param = {
            "normEps": self.config.rms_norm_eps,
            "normType": NormType.RMS_NORM,
            "numAttentionHeadsPerRank": self.num_attention_heads,
            "hiddenSizePerAttentionHead":  self.head_size,
            "numHiddenLayers": self.config.num_hidden_layers,
            "numKeyValueHeadsPerRank": self.num_key_value_heads,
            "isFA": False,
            "isUnpadInputs": True,
            "isBF16": self.dtype == torch.bfloat16,
            "packQuantType": pack_quant_configs,
            "isEmbeddingParallel": False,
            "isLmHeadParallel": True,
            "linearTransposeType": linear_transpose_types, # compatible with base
            "linearQuantType": linear_quant_types, # compatible with base
            "attnLinearQuantType": attn_linear_types,
            "mlpLinearQuantType": mlp_linear_types,
            "moeLinearQuantType": moe_linear_types,
            "attnLinearTransposeType": attn_linear_transpose_types,
            "mlpLinearTransposeType": mlp_linear_transpose_types,
            "moeLinearTransposeType": moe_linear_transpose_types,
            "lmHeadTransposeType": self.lm_head.linear.trans_flag,
            "enableSwiGLU": False if self.soc_info.need_nz else True,
            'hasSharedExpert': True if self.n_shared_experts > 0 else False,
            'hasSharedExpertGate': False,
            "rank": self.tp_rank,
            "softmaxScale": self.softmax_scale,
            "expertParallelDegree": self.expert_parallel_degree,
            "numOfExperts": self.num_of_experts,
            "firstKDenseReplace": 0,
            "numOfSharedExperts": self.n_shared_experts,
            "processLogits": "none",
            "numOfSelectedExperts": self.num_of_selected_experts,
            "routingMethod": "integratedSoftmaxTopK", # with the integration of softmax and topk-sort operators
            "worldSize": self.tp_world_size,
            "backend": self.soc_info.communication_backend,
            "rankTableFile": ENV.rank_table_file,
            "enableAddNorm": False,
            "normHasBias": False,
            "claShareFactor": self.cla_share_factor,
            "enableFusedRouting": self.enable_fused_routing
        }

        encoder_param = {**coder_param, "isPrefill": True, "enableLcoc": self.lcoc_enable}
        decoder_param = {**coder_param, "isPrefill": False, "enableLcoc": False}
        self.acl_encoder_operation.set_param(json.dumps({**encoder_param}))
        self.acl_decoder_operation.set_param(json.dumps({**decoder_param}))
        self.acl_encoder_operation.set_weight(self.ascend_weight)
        self.acl_decoder_operation.set_weight(self.ascend_weight)

    # called by super().forward()
    def prepare_inputs_for_ascend(self,
                                  input_ids: torch.Tensor,
                                  position_ids: torch.Tensor,
                                  is_prefill: bool,
                                  kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
                                  block_tables: torch.Tensor,
                                  slots: torch.Tensor,
                                  input_lengths: torch.Tensor,
                                  max_seq_len: int,
                                  lm_head_indices: Optional[torch.Tensor] = None,
                                  **kwargs):
        self.rotary_embedding.update_cos_sin_cache_total(self.dtype,
                                                         self.device,
                                                         self.max_position_embeddings)
        self.cos_embed = self.rotary_embedding.get_cos_cached_total()
        self.sin_embed = self.rotary_embedding.get_sin_cached_total()

        self.expert_array = self.placeholder
        if not self.enable_fused_routing:
            self.expert_array = torch.arange(self.config.num_experts_per_tok * len(input_ids),
                                            dtype=torch.int32, device=input_ids.device)
        self.acl_param = json.dumps({
            "seqLen": input_lengths.tolist(),
            })
        atten_mask = self.attn_mask_fake
        logits_indices = self.lm_head_indices_fake
        if is_prefill:
            if self.soc_info.need_nz:
                pad_maxs = math.ceil(self.max_position_embeddings / 16) * 16
                atten_mask = self.attn_mask.get_attn_mask(pad_maxs, kv_cache[0][0].dtype,
                                                                    kv_cache[0][0].device)
                atten_mask = self.transdata_operation.execute([atten_mask])[0]
            else:
                # 128 for maskfree
                atten_mask = self.attn_mask.get_attn_mask(128, kv_cache[0][0].dtype,
                                                                    kv_cache[0][0].device)
            if self.dtype == torch.bfloat16:
                atten_mask = torch.where(atten_mask == -torch.inf, 1, atten_mask)

            if lm_head_indices is None:
                lm_head_indices = torch.tensor(range(input_ids.shape[0]),
                                                dtype=torch.int64, device=input_ids.device)
            logits_indices = lm_head_indices.to(torch.int64)
        else:
            atten_mask = torch.zeros(input_lengths.size(0),
                                        self.num_attention_heads,
                                        1, 
                                        input_lengths.max(),
                                        dtype=self.dtype,
                                        device=self.device)

        acl_inputs : List[torch.Tensor] = [
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
            logits_indices,
            self.expert_array,
            self.expert_group,
            self.one_hot,
            self.zero_hot
        ]

        return acl_inputs, self.acl_param