# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json
import math
from typing import Optional, List, Tuple
from enum import Enum

import torch

from atb_llm.models.base.flash_causal_lm import FlashForCausalLM
from atb_llm.models.deepseekv2.modeling_deepseekv2 import FlashDeepseekV2Model
from atb_llm.models.deepseekv2.config_deepseekv2 import DeepseekV2Config
from atb_llm.models.deepseekv2.position_embedding_deepseekv2 import DeepseekV2YarnRotaryEmbedding
from atb_llm.models.deepseekv2.weight_wrapper_deepseekv2 import MlaWrapper, Deepseekv2WeightWrapper
from atb_llm.utils.data.moe_weight_wrapper import MoeMlpWrapper
from atb_llm.utils.env import ENV
from atb_llm.utils.layers import PositionRotaryEmbedding
from atb_llm.utils.layers import (
    TensorEmbedding,
    load_column_multi,
)
from atb_llm.utils.layers.norm.fast_layer_norm import NormType
from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.utils.moe_utils import assign
from atb_llm.utils.weights import ProcessGroupType

_ROPE_SCALING_KEYS = ["original_max_position_embeddings", "beta_fast", "beta_slow", "mscale", "mscale_all_dim"]


class ExpertParallelDegree(int, Enum):
    NO_EP = 0
    STATIC_EP = 1
    DYNAMIC_EP = 2
    MIX_EP = 3


class FlashDeepseekv2ForCausalLM(FlashForCausalLM):
    def __init__(self, config, weights, **kwargs):
        super().__init__(config, weights, **kwargs)
        self.model = FlashDeepseekV2Model(config, weights)
        weights.switch_process_group(ProcessGroupType.MLP)
        self.lm_head = load_column_multi(
            config,
            prefixes=["lm_head"],
            weights=weights,
            head_size=1,
            lm_head=True,
        )
        self.config = config
        self.acl_encoder_operation_inputs = []
        self.acl_decoder_operation_inputs = []

        self.placeholder = torch.zeros(1, dtype=self.dtype, device=self.device)
        self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int64, device=self.device)

        self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        self.transdata_param = json.dumps({})
        self.transdata_operation.set_param(self.transdata_param)

        self.padding_idx = config.pad_token_id
        self.embed_tokens = TensorEmbedding(
            prefix="model.embed_tokens", weights=weights
        )

        if hasattr(config, "mla_quantize"):
            self.mla_quantize = config.mla_quantize
        else:
            self.mla_quantize = self.quantize

        self.hidden_dim = config.hidden_size
        self.final_hidden_states = []
        self.expert_array = []

        self.expert_group = torch.arange(1024, dtype=torch.int32).npu() # 1024: const for groupedTopK
        self.routed_scaling_factor = config.routed_scaling_factor
        self.one_hot = torch.tensor([1], dtype=torch.int32).npu()
        self.zero_hot = torch.tensor([0], dtype=torch.int32).npu()
        self.final_bias = torch.zeros([self.config.n_routed_experts, self.config.hidden_size], dtype=self.dtype).npu()

        self.num_of_experts = config.n_routed_experts
        self.num_of_selected_experts = [config.num_experts_per_tok]
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.topk_method = config.topk_method
        self.tp = config.tp if config.tp else True # Defaulting the model to tensor parallel
        self.first_k_dense_replace = config.first_k_dense_replace if config.first_k_dense_replace else 0
        self.n_shared_experts = config.n_shared_experts if config.n_shared_experts else 0
        self.norm_topk_prob = config.norm_topk_prob if config.norm_topk_prob else False
        self.first_k_dense_replace = config.first_k_dense_replace
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim

        self.softmax_scale = (config.qk_nope_head_dim + config.qk_rope_head_dim) ** (-0.5)
        factor_name = "factor"
        if self.config.rope_scaling_dict is not None:
            mscale_all_dim = self.config.rope_scaling_dict.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling_dict[factor_name]
            if mscale_all_dim:
                mscale = DeepseekV2YarnRotaryEmbedding.yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

        if self.config.rope_scaling_dict is None:
            if hasattr(config, 'rope_scaling') and self.config.rope_scaling_dict is not None:
                self.scaling_factor = self.config.rope_scaling_dict.get(factor_name, 1.0)
            else:
                self.scaling_factor = 1.0
            self.rotary_embedding = PositionRotaryEmbedding.static(
                dim=self.qk_rope_head_dim,
                base=self.rope_theta,
                device="cpu",
                scaling_factor=self.scaling_factor
            ).to(self.device)
        else:
            self.scaling_type = config.rope_scaling_dict["type"]
            self.scaling_factor = config.rope_scaling_dict["factor"]
            if self.scaling_type == "yarn":
                kwargs = {
                    key: self.config.rope_scaling_dict[key]
                    for key in _ROPE_SCALING_KEYS
                    if key in self.config.rope_scaling_dict
                }
                yarn_kwargs = DeepseekV2YarnRotaryEmbedding.StaticInputArgs(
                                            max_position_embeddings=self.max_position_embeddings,
                                            scaling_factor=scaling_factor,
                                            **kwargs,)
                self.rotary_embedding = DeepseekV2YarnRotaryEmbedding.static_yarn(dim=self.qk_rope_head_dim,
                                                                             base=self.rope_theta,
                                                                             device="cpu",
                                                                             yarn_kwargs=yarn_kwargs).to(self.device)
            else:
                msg = f"Unknown RoPE scaling type {self.scaling_type}"
                logger.error(msg, ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
                raise ValueError(msg)

        self.expert_parallel_degree = 0
        self.mask_start_idx = 0
        self.communication_backend = self.soc_info.communication_backend
        self.ep = self.mapping.has_moe_ep()
        self.p_to_d_weight = False
        self.device_expert = assign(self.config.n_routed_experts, self.mapping.moe_ep.group_size)[
            self.mapping.moe_ep.rank]
        if self.ep:
            if not hasattr(config, "ep_level"):
                logger.info("For expert parallel, "
                    "the ep_level variable needs to be defined in the model configuration file."
                    "The available options are 1, 2, or 3.")
                config.ep_level = ExpertParallelDegree.STATIC_EP
            self.expert_parallel_degree = config.ep_level
            if config.ep_level in [ExpertParallelDegree.DYNAMIC_EP, ExpertParallelDegree.MIX_EP]:
                self.dep_communication_backend = config.communication_backend \
                    if hasattr(config, "communication_backend") else ["hccl", "hccl"]
                self.p_to_d_weight = config.ep_level == ExpertParallelDegree.MIX_EP
            logger.info(f"Expert parallel level is {self.expert_parallel_degree}.")
            logger.info(f"Experts of rank {self.mapping.moe_ep.rank} are: {self.device_expert}")
            
        self.num_of_device_expert = len(self.device_expert)
        self.start_device_expert_id = torch.tensor(self.device_expert[0], dtype=torch.int64).npu().view(-1)
        self.max_device_expert_id = torch.tensor([len(self.device_expert) - 1], dtype=torch.int64).npu().view(-1)

    def init_position_rotary_embedding(self,
                                       position_ids: torch.Tensor,
                                       max_seq_len: int):
        self.rotary_embedding.update_cos_sin_cache_total(self.dtype, position_ids.device, max_seq_len)
        self.cos_embed = self.rotary_embedding.get_cos_cached_total()
        self.sin_embed = self.rotary_embedding.get_sin_cached_total()

    def init_ascend_operations(self, config: DeepseekV2Config):
        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch("deepseekV2_DecoderModel")
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch("deepseekV2_DecoderModel")

    def init_weight_wrapper(self):
        attn_wrapper = MlaWrapper(
            norm_name='input_layernorm',
            wrapper_name='self_attn',
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            qk_nope_head_dim=self.config.qk_nope_head_dim,
            qk_rope_head_dim=self.config.qk_rope_head_dim,
            q_lora_rank=self.config.q_lora_rank,
            kv_lora_rank=self.config.kv_lora_rank,
            v_head_dim=self.config.v_head_dim
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
        weight_wrapper = Deepseekv2WeightWrapper(self.soc_info, self.tp_rank,
                                                attn_wrapper, moe_mlp_wrapper,
                                                self.num_of_experts)
        weight_wrapper.register_embedding(self.model.embed_tokens)
        return weight_wrapper

    def get_weights(self):
        weight_wrapper = self.init_weight_wrapper()
        for i in range(self.num_layers):
            layer = self.model.layers[i]
            if i < self.first_k_dense_replace:
                weight_wrapper.register_moe_layer(layer, self.quantize, dense_layer=True,
                                                attn_quantize_type=self.mla_quantize)
            elif self.ep:
                weight_wrapper.register_moe_layer(layer, self.quantize,
                                expert_roster=[i for i, _ in enumerate(self.device_expert)],
                                attn_quantize_type=self.mla_quantize)
                if self.p_to_d_weight:
                    weight_wrapper.register_shared_expert_dp2tp(layer, self.quantize, "shared_experts_tp")
                    weight_wrapper.register_router(layer, self.quantize, "shuffled_gate")
                del layer.mlp
                torch.npu.empty_cache()
            else:
                weight_wrapper.register_moe_layer(layer, self.quantize, dense_layer=False,
                                                attn_quantize_type=self.mla_quantize)
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

        coder_param = {
            "isUnpadInputs": True,
            "normEps": self.config.rms_norm_eps,
            "normType": NormType.RMS_NORM,
            "numAttentionHeadsPerRank": self.num_attention_heads,
            "hiddenSizePerAttentionHead":  self.head_size,
            "numHiddenLayers": self.config.num_hidden_layers,
            "numKeyValueHeadsPerRank": 1, # for MLA
            "isFA": False,
            "isBF16": self.dtype == torch.bfloat16,
            "packQuantType": pack_quant_configs,
            "isEmbeddingParallel": False,
            "isLmHeadParallel": True,
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
            "qLoraRank": self.config.q_lora_rank if self.config.q_lora_rank is not None else 0,
            "kvLoraRank": self.config.kv_lora_rank,
            "qkNopeHeadDim": self.config.qk_nope_head_dim,
            "qkRopeHeadDim": self.config.qk_rope_head_dim,
            "softmaxScale": self.softmax_scale,
            "maskStartIdx": self.mask_start_idx,
            "numOfExperts": self.num_of_experts,
            "numOfDeviceExperts": self.num_of_device_expert,
            "deviceExpert": self.device_expert,
            "firstKDenseReplace": self.first_k_dense_replace,
            "nSharedExperts": self.n_shared_experts,
            "processLogits": self.get_process_logits_type(),
            "routedScalingFactor": self.routed_scaling_factor,
            "numOfSelectedExperts": self.num_of_selected_experts,
            "numOfGroups": self.n_group,
            "topkGroups": self.topk_group,
            "routingMethod": self.get_routing_method_type(),
            "worldSize": self.tp_world_size,
            "rankTableFile": ENV.rank_table_file,
            "qkvHasBias": False,
            "enableIntraLayerAddNorm": False,
            "enableInterLayerAddNorm": False,
            "hasP2DWeight":self.p_to_d_weight,
        }

        if self.mapping is not None:
            coder_param.update(self.mapping.to_dict())

        if coder_param["routingMethod"] not in ['softMaxTopK', 'integratedSoftmaxTopK', 'deviceLimited', 'noAuxTc']:
            msg = "The routingMethod chosen is not valid, please choose among the following:\n \
                  'softMaxTopK': regular routing method with softmax and topk-sort operators\n \
                  'integratedSoftmaxTopK': routing method with the integration of softmax and topk-sort operators\n \
                  'deviceLimited': device-limited routing method (e.g. deepseekv2)\n \
                  'noAuxTc': routing method with sigmoid and gate bias"
            logger.error(msg, ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
            raise ValueError(msg)

        encoder_param = {**coder_param, "isPrefill": True, "supportLcoc": False,
                         "expertParallelDegree": self.expert_parallel_degree \
                            if self.expert_parallel_degree != ExpertParallelDegree.MIX_EP \
                            else ExpertParallelDegree.DYNAMIC_EP,
                         "backend": self.communication_backend if self.expert_parallel_degree < 2 \
                            else self.dep_communication_backend[0]}
        decoder_param = {**coder_param, "isPrefill": False, "supportLcoc": False,
                        "expertParallelDegree": self.expert_parallel_degree \
                            if self.expert_parallel_degree != ExpertParallelDegree.MIX_EP \
                            else ExpertParallelDegree.STATIC_EP,
                        "backend": self.communication_backend if self.expert_parallel_degree < 2 \
                            else self.dep_communication_backend[1]}
        self.acl_encoder_operation.set_param(json.dumps({**encoder_param}))
        self.acl_decoder_operation.set_param(json.dumps({**decoder_param}))
        self.acl_encoder_operation.set_weight(self.ascend_weight)
        self.acl_decoder_operation.set_weight(self.ascend_weight)

    def get_all2all_buffer_factor(self, length):
        length = length * self.mapping.attn_dp.group_size
        if hasattr(self.config, "length_thresholds"):
            length_thresholds = self.config.length_thresholds
        else:
            length_thresholds = [(65536, 1.05), (32768, 1.2), (16384, 1.35), (4096, 2)]
        for threshold in length_thresholds:
            if length >= threshold[0]:
                return threshold[1]
        return self.mapping.moe_ep.group_size

    def get_process_logits_type(self) -> str:
        if self.routed_scaling_factor > 1 and self.norm_topk_prob is True:
            return "normScaling"
        elif self.routed_scaling_factor > 1:
            return "scaling"
        return "none"
    
    def get_routing_method_type(self) -> str:
        if self.topk_method == "noaux_tc":
            return "noAuxTc"
        elif self.topk_method == "group_limited_greedy":
            return "deviceLimited"
        return "softMaxTopK"

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

        if self.mapping.has_dp():
            token_size_per_dp_group = kwargs.get("token_size_per_dp_group", None)
            shard_effective_token_indices = kwargs.get("shard_effective_token_indices", None)
            token_index_with_padding = kwargs.get("token_index_with_padding", None)
            skip_padding_token_indices = kwargs.get("skip_padding_token_indices", None)
            token_size = token_size_per_dp_group.sum().tolist()
            max_token_size_per_dp_group = token_size_per_dp_group.max().item()
            token_size_per_dp_group_startid = torch.cumsum(token_size_per_dp_group, dim=0)
            token_size_per_dp_group_startid[-1] = 0
            moe_token_index_with_padding = torch.concat([
                torch.concat([torch.arange(j) + token_size_per_dp_group_startid[rank_id - 1],
                torch.zeros(max_token_size_per_dp_group - j, dtype=torch.int32)]) \
                    for rank_id, j in enumerate(token_size_per_dp_group)], dim=0).npu()
            moe_skip_padding_token_indices = torch.arange(token_size_per_dp_group[self.mapping.attn_dp.rank],
                dtype=torch.int32).npu()
        else:
            token_size = len(input_ids)

        self.expert_array = self.placeholder
        final_hidden_states = torch.empty([token_size, self.config.hidden_size],
                                          dtype=kv_cache[0][0].dtype,
                                          device=input_ids.device)

        dep_inputs = [self.start_device_expert_id, self.max_device_expert_id]
        if self.expert_parallel_degree == ExpertParallelDegree.DYNAMIC_EP or \
            (self.expert_parallel_degree == ExpertParallelDegree.MIX_EP and is_prefill):
            ep_input_length = len(input_ids)
            if self.mapping.has_attn_tp():
                reduce_scatter_padding = ep_input_length % self.mapping.attn_tp.group_size
                ep_input_length_padding = (self.mapping.attn_tp.group_size - reduce_scatter_padding) \
                    if reduce_scatter_padding != 0 else 0

                ep_input_length_padding += ep_input_length
                padding_idx = torch.tensor([i % ep_input_length for i in range(ep_input_length_padding)],
                    dtype=torch.int32).view(-1).npu()

                un_padding_idx = torch.tensor([i for i in range(ep_input_length)], dtype=torch.int32).view(-1).npu()
                dep_inputs.append(padding_idx)
                dep_inputs.append(un_padding_idx)
                ep_input_length = ep_input_length_padding
                dynamic_ep_idx_range = ep_input_length // self.mapping.attn_tp.group_size \
                    * self.config.num_experts_per_tok
                dynamic_ep_idx = torch.tensor([i for i in range(dynamic_ep_idx_range)],
                    dtype=torch.int32).npu().view(-1)
            else:
                dynamic_ep_idx = torch.tensor([i for i in range(len(input_ids) * self.config.num_experts_per_tok)],
                    dtype=torch.int32).npu().view(-1)

            if self.mapping.has_dp():
                ep_input_length = kwargs.get("token_size_per_dp_group", None).max()

            ep_input_length = max(ep_input_length // self.mapping.attn_tp.group_size, 1)
            ep_input_length = int(
                ep_input_length * self.get_all2all_buffer_factor(ep_input_length) * self.config.num_experts_per_tok)
            all2all_padding = ep_input_length % self.mapping.moe_ep.group_size
            ep_input_length_padding = (
                    self.mapping.moe_ep.group_size - all2all_padding) if all2all_padding != 0 else 0

            ep_input_length_padding += ep_input_length
            moe_idx = torch.tensor([i + 1 for i in range(ep_input_length_padding)], dtype=torch.int32).npu().view(-1)

            self.expert_array = torch.ones(moe_idx.shape[0], dtype=torch.float16).npu().view(-1, 1)
            dep_inputs.append(dynamic_ep_idx)
            dep_inputs.append(moe_idx)

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
            if lm_head_indices is None:
                lm_head_indices = torch.tensor(range(input_ids.shape[0]),
                                                dtype=torch.int64, device=input_ids.device)
            self.acl_param = json.dumps({
                "seqLen": input_lengths.tolist(),
            })
            self.acl_encoder_operation_inputs = [
                input_ids,
                position_ids.to(torch.int64),
                self.cos_embed,
                self.sin_embed,
                atten_mask,
                block_tables.to(torch.int32),
                slots.to(torch.int32),
                self.placeholder,
                final_hidden_states,
                self.placeholder,
                self.placeholder,
                input_lengths.to(torch.int32),
                lm_head_indices.to(torch.int64),
                self.expert_array,
                self.expert_group,
                self.one_hot,
                self.zero_hot
            ]
            if self.mapping.has_dp():
                self.acl_encoder_operation_inputs.extend([shard_effective_token_indices, token_index_with_padding,
                    skip_padding_token_indices, moe_token_index_with_padding, moe_skip_padding_token_indices])
            if self.mapping.has_moe_ep():
                self.acl_encoder_operation_inputs.extend(dep_inputs)
            return self.acl_encoder_operation_inputs, self.acl_param
        else:
            self.acl_param = json.dumps({
                "seqLen": input_lengths.tolist(),
            })
            self.acl_decoder_operation_inputs = [
                input_ids,
                position_ids.to(torch.int64),
                self.cos_embed,
                self.sin_embed,
                self.attn_mask_fake,
                block_tables.to(torch.int32),
                slots.to(torch.int32),
                self.placeholder,
                final_hidden_states,
                self.placeholder,
                self.placeholder,
                input_lengths.to(torch.int32),
                lm_head_indices.to(torch.int64) if self.mapping.has_dp() and self.mapping.has_mlp_tp() \
                                                else self.lm_head_indices_fake,
                self.expert_array,
                self.expert_group,
                self.one_hot,
                self.zero_hot
            ]
            if self.mapping.has_dp():
                self.acl_decoder_operation_inputs.extend([shard_effective_token_indices, token_index_with_padding,
                    skip_padding_token_indices, moe_token_index_with_padding, moe_skip_padding_token_indices])
            if self.mapping.has_moe_ep():
                self.acl_decoder_operation_inputs.extend(dep_inputs)
            return self.acl_decoder_operation_inputs, self.acl_param