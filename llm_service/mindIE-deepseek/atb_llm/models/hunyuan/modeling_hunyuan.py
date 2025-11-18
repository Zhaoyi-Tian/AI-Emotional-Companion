# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

import copy
import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from atb_llm.utils.layers import (
    TensorParallelColumnLinear,
    TensorParallelRowLinear,
    load_column_multi,
    TensorEmbedding,
    TensorParallelEmbedding,
)
from atb_llm.utils.quantize.quant_type import QuantType
from atb_llm.utils.quantize.w8a8_dynamic import W8A8DynamicLinearStatic
from atb_llm.utils.quantize.pack_type import PackType, calc_linear_pack_type
from atb_llm.utils.layers.linear.fast_linear import FastLinear
from atb_llm.models.deepseek.modeling_deepseek import \
    DeepseekMLP, DeepseekMoE


class HunyuanRMSNorm(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        super().__init__()

        weight = weights.get_tensor(f"{prefix}.weight")
        self.weight = nn.Parameter(weight)
        self.variance_epsilon = eps


class HunyuanRMSNormBias(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        super().__init__()

        weight = weights.get_tensor(f"{prefix}.weight")
        try:
            bias = weights.get_tensor(f"{prefix}.bias")
        except AssertionError:
            bias = torch.zeros(weight.shape, dtype=weights.dtype)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)
        self.variance_epsilon = eps


class HunyuanRMSNormWrapper(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        super().__init__()

        self.ori = HunyuanRMSNorm(prefix, weights, eps)
        self.anti = HunyuanRMSNormBias(f'{prefix}.module', weights, eps)


class HunyuanRMSNormAntiOutlierWrapper(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        super().__init__()

        self.ori = HunyuanRMSNorm(f'{prefix}.ori', weights, eps)
        self.anti = HunyuanRMSNormBias(f'{prefix}.anti', weights, eps)


class HunyuanMLP(DeepseekMLP):
    def __init__(self, prefix, config, weights, intermediate_size=None):
        super().__init__(prefix, config, weights, intermediate_size=None)
        self.act_fn = ACT2FN[config.hidden_act]


class HunyuanMoE(DeepseekMoE):
    def __init__(self, prefix, config, weights, shared_mlp_cls):
        super().__init__(prefix, config, weights, shared_mlp_cls, "gate.wg", "shared_mlp")
        self.pack_type = self.shared_experts.pack_type

    def init_experts(self, weights, prefix, expert_prefix, shared_expert_key, shared_mlp_cls):
        config = self.config

        linear_names = [f'{expert_prefix}.0.up_proj', f'{expert_prefix}.0.gate_proj']
        pack_name = f'{expert_prefix}.0.gate_up_proj'
        layer_prefix = '.'.join(prefix.split('.')[:-1])
        norm_name = f'{layer_prefix}.post_attention_layernorm'
        self.pack_type = calc_linear_pack_type(weights, linear_names, norm_name, pack_name)

        gateup_proj_list = nn.ModuleList()
        for i in self.expert_lists[self.rank]:
            gateup_proj_list.append(load_column_multi(
                config,
                prefixes=[f"{expert_prefix}.{i}.gate_proj", f"{expert_prefix}.{i}.up_proj"],
                weights=weights,
                head_size=1,
            ))
        self.gate_up_proj = self.stack_linears_from_experts(gateup_proj_list, config.quantize)

        down_proj_list = nn.ModuleList()
        for i in self.expert_lists[self.rank]:
            down_proj_list.append(TensorParallelRowLinear.load(
                config,
                prefix=f"{expert_prefix}.{i}.down_proj",
                weights=weights,
                bias=False,
            ))
        self.down_proj = self.stack_linears_from_experts(down_proj_list, config.quantize)

        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            shared_expert_prefix = f"{prefix}.{shared_expert_key}"
            self.shared_experts = shared_mlp_cls(
                prefix=shared_expert_prefix,
                config=config,
                weights=weights,
                intermediate_size=intermediate_size
            )
    
    def stack_linears_from_experts(self, linears, quantize):
        if linears[0].linear.weight.dtype not in [torch.float16, torch.bfloat16] and quantize != QuantType.W8A8_DYNAMIC:
            raise AssertionError("hunyuan-large only supports (float16, bfloat16, W8A8_DYNAMIC)")
        
        weight_stacked = torch.stack([linear.linear.weight.data for linear in linears], dim=0)
        if quantize == QuantType.W8A8_DYNAMIC:
            scale_stacked = torch.stack([linear.linear.weight_scale.data for linear in linears], dim=0)
            offset_stacked = torch.stack([linear.linear.weight_offset.data for linear in linears], dim=0)

            linear_stacked = W8A8DynamicLinearStatic(
                weight=weight_stacked,
                weight_scale=scale_stacked,
                weight_offset=offset_stacked,
                bias=None,
                need_flatten=False
            )
        else:
            linear_stacked = FastLinear(
                weight=weight_stacked,
                bias=None,
            )
        linear_stacked.trans_flag = linears[0].linear.trans_flag
        return linear_stacked


class FlashHunyuanAttention(nn.Module):
    def __init__(self,
                 prefix: str,
                 layer_idx,
                 config,
                 weights):
        super().__init__()
        self.config = config
        self.config = copy.deepcopy(config)
        if hasattr(self.config, 'mla_quantize'):
            self.config.quantize = self.config.mla_quantize
        self.attention_type = 'cross' if config.use_cla and layer_idx % config.cla_share_factor != 0 else 'self'
        self.use_qk_norm = config.use_qk_norm
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.head_dim = self.hidden_size // config.num_attention_heads
        self.has_bias = False
        linear_names = []

        if self.attention_type == 'cross':
            self.q_proj = TensorParallelColumnLinear.load(
                self.config,
                prefix=f"{prefix}.q_proj",
                weights=weights,
                bias=self.has_bias,
            )
        else:
            self.qkv_proj = load_column_multi(
                config,
                prefixes=[f"{prefix}.q_proj", 
                            f"{prefix}.k_proj", 
                            f"{prefix}.v_proj"],
                weights=weights,
                bias=self.has_bias,
                head_size=self.head_dim
            )
        linear_names.append(f'{prefix}.q_proj')

        if config.use_qk_norm:
            self.query_layernorm = HunyuanRMSNorm(
                prefix=f"{prefix}.query_layernorm", weights=weights, eps=config.rms_norm_eps
            )
            self.key_layernorm = HunyuanRMSNorm(
                prefix=f"{prefix}.key_layernorm", weights=weights, eps=config.rms_norm_eps
            )

        self.o_proj = TensorParallelRowLinear.load(
            self.config,
            prefix=f"{prefix}.o_proj",
            weights=weights,
            bias=self.has_bias,
        )
        linear_names.append(f'{prefix}.o_proj')
        layer_prefix = '.'.join(prefix.split('.')[:-1])
        norm_name = f'{layer_prefix}.input_layernorm'
        weights.quantize = self.config.quantize
        self.pack_type = calc_linear_pack_type(weights, linear_names, norm_name)
        weights.quantize = config.quantize
        self.softmax_scale = self.head_dim ** (-0.5)


class FlashHunyuanDecoderLayer(nn.Module):
    def __init__(self, layer_idx, config, weights):
        super().__init__()
        prefix = f"model.layers.{layer_idx}"
        self.hidden_size = config.hidden_size

        self.self_attn = FlashHunyuanAttention(
            prefix=f"{prefix}.self_attn", layer_idx=layer_idx, config=config, weights=weights
        )

        self.mlp = (
            HunyuanMoE(prefix=f"{prefix}.mlp", config=config, weights=weights, shared_mlp_cls=HunyuanMLP)
            if config.num_experts > 1
            else HunyuanMLP(prefix=f"{prefix}.mlp", config=config, weights=weights)
        )
        if self.self_attn.pack_type in [PackType.ALL_FP, PackType.ALL_W4A16, PackType.ALL_W8A16, PackType.MIX_W8A16,
                                        PackType.MIX_W8A8_DYNAMIC, PackType.MIX_W8A8_DYNAMIC_ANTI,
                                        PackType.ALL_W8A8_DYNAMIC, PackType.ALL_W8A8_DYNAMIC_ANTI]:
            self.input_layernorm = HunyuanRMSNorm(
                prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
            )
        elif self.self_attn.pack_type in [
            PackType.ALL_W8A8_ANTI, PackType.MIX_W8A8_ANTI,
            PackType.ALL_W8A16_ANTI, PackType.MIX_W8A16_ANTI,
            PackType.ALL_W4A16_ANTI, PackType.MIX_W4A16_ANTI,
            PackType.MIX_W8A8_DYNAMIC, PackType.MIX_W8A8_DYNAMIC_ANTI,
            PackType.ALL_W8A8_DYNAMIC, PackType.ALL_W8A8_DYNAMIC_ANTI
        ]:
            self.input_layernorm = HunyuanRMSNormWrapper(
                prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
            )
        elif self.self_attn.pack_type in [PackType.ALL_W8A8SC_ANTI, PackType.MIX_W8A8SC_ANTI]:
            self.input_layernorm = HunyuanRMSNormAntiOutlierWrapper(
                prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
            )
        elif self.self_attn.pack_type in [PackType.ALL_W8A8, PackType.MIX_W8A8, PackType.ALL_W8A8SC,
                                          PackType.MIX_W8A8SC]:
            self.input_layernorm = HunyuanRMSNormBias(
                prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
            )
        else:
            raise AssertionError(f'self_attn.pack_type: {self.self_attn.pack_type} not supported')

        if self.mlp.pack_type in [PackType.ALL_FP, PackType.ALL_W4A16, PackType.ALL_W8A16, PackType.MIX_W8A16,
                                  PackType.MIX_W8A8_DYNAMIC, PackType.MIX_W8A8_DYNAMIC_ANTI,
                                  PackType.ALL_W8A8_DYNAMIC, PackType.ALL_W8A8_DYNAMIC_ANTI]:
            self.post_attention_layernorm = HunyuanRMSNorm(
                prefix=f"{prefix}.post_attention_layernorm",
                weights=weights,
                eps=config.rms_norm_eps,
            )
        elif self.mlp.pack_type in [
            PackType.ALL_W8A8_ANTI, PackType.MIX_W8A8_ANTI,
            PackType.ALL_W8A16_ANTI, PackType.MIX_W8A16_ANTI,
            PackType.ALL_W4A16_ANTI, PackType.MIX_W4A16_ANTI,
            PackType.MIX_W8A8_DYNAMIC, PackType.MIX_W8A8_DYNAMIC_ANTI,
            PackType.ALL_W8A8_DYNAMIC, PackType.ALL_W8A8_DYNAMIC_ANTI
        ]:
            self.post_attention_layernorm = HunyuanRMSNormWrapper(
                prefix=f"{prefix}.post_attention_layernorm",
                weights=weights, eps=config.rms_norm_eps
            )
        elif self.mlp.pack_type in [PackType.ALL_W8A8SC_ANTI, PackType.MIX_W8A8SC_ANTI]:
            self.post_attention_layernorm = HunyuanRMSNormAntiOutlierWrapper(
                prefix=f"{prefix}.post_attention_layernorm",
                weights=weights, eps=config.rms_norm_eps
            )
        elif self.mlp.pack_type in [PackType.ALL_W8A8, PackType.MIX_W8A8, PackType.ALL_W8A8SC,
                                    PackType.MIX_W8A8SC]:
            self.post_attention_layernorm = HunyuanRMSNormBias(
                prefix=f"{prefix}.post_attention_layernorm",
                weights=weights,
                eps=config.rms_norm_eps,
            )
        else:
            raise AssertionError(f'mlp.pack_type: {self.mlp.pack_type} not supported')


class FlashHunyuanModel(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        self.parallel_embedding = False

        self.embed_tokens = (TensorParallelEmbedding if self.parallel_embedding else TensorEmbedding)(
            prefix="model.embed_tokens", weights=weights
        )

        self.layers = nn.ModuleList(
            [
                FlashHunyuanDecoderLayer(
                    layer_idx,
                    config,
                    weights,
                    )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = HunyuanRMSNorm(prefix="model.norm", weights=weights, eps=config.rms_norm_eps)