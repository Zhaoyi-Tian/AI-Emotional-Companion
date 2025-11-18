# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import torch
from atb_llm.utils.quantize.pack_type import LinearType, ALL_PACK_LIST
from atb_llm.utils.quantize.quant_type import QuantType
from atb_llm.utils.data.weight_wrapper import NormWrapper, get_module
from atb_llm.utils.data.moe_weight_wrapper import MoeWeightWrapper, get_moe_module


class ClaWrapper(NormWrapper):
    def __init__(self,
                 input_norm_name,
                 pack_name,
                 o_name,
                 wrapper_name,
                 num_attention_heads,
                 num_key_value_heads,
                 cla_share_factor,
                 q_norm_name='',
                 k_norm_name=''):
        super().__init__(input_norm_name)
        self.pack_name = pack_name
        self.o_name = o_name
        self.wrapper_name = wrapper_name
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.cla_share_factor = cla_share_factor
        self.q_norm_name = q_norm_name
        self.k_norm_name = k_norm_name


class HunyuanWeightWrapper(MoeWeightWrapper):
    def __init__(self,
                 soc_info,
                 tp_rank,
                 cla_wrapper,
                 moe_mlp_wrapper,
                 num_experts,
                 tp_intermediate_size):
        super().__init__(soc_info, tp_rank, None, moe_mlp_wrapper, num_experts)
        self.attn_wrapper = cla_wrapper
        self.tp_intermediate_size = tp_intermediate_size
        self.supported_quantize_type = [QuantType.W4A16, QuantType.W8A16, QuantType.W8A8_DYNAMIC]

    def set_gmm_nd_nz(self, quantize_type):
        self.gmm_quant_nd_nz = False
        if quantize_type == QuantType.W8A8_DYNAMIC:
            if self.tp_intermediate_size % 16 == 0: # 16 for nz format
                self.gmm_quant_nd_nz = True

    def register_linear_list_without_bias_hunyuan(self, linear_list):
        tensor_stacked = linear_list.weight.data
        if self.gmm_quant_nd_nz:
            self.weights.append(self.weight_format_cast(tensor_stacked, enable_nz=True))
        else:
            self.weights.append(self.weight_format_cast(tensor_stacked))

    def register_linear_list_bias_hunyuan(self, linear_list):
        self.register_linear_list_without_bias_hunyuan(linear_list)
        if hasattr(linear_list, 'bias') and (getattr(linear_list, 'bias') is not None):
            bias_stacked = linear_list.bias.data
            if self.gmm_quant_nd_nz:
                self.weights.append(self.weight_format_cast(bias_stacked, enable_nz=True))
            else:
                self.weights.append(self.weight_format_cast(bias_stacked))
        else:
            self.weights.append(self.placeholder)

    def register_linear_stacked_wrapper(self, linear_stacked, quantize_type):
        trans_flag = linear_stacked.trans_flag
        if linear_stacked.weight.dtype in [torch.float16, torch.bfloat16]:
            self.register_linear_list_bias_hunyuan(linear_stacked)
            self.weights.extend([self.placeholder] * 4)
            self.layer_linear_type.append(LinearType.FP)
        elif quantize_type in [QuantType.W4A16, QuantType.W8A16, QuantType.W8A8_DYNAMIC]:
            self.register_linear_list_bias_hunyuan(linear_stacked)
            self.weights.append(self.placeholder)
            self.weights.append(linear_stacked.weight_offset.data)
            self.weights.append(linear_stacked.weight_scale.data)
            self.weights.append(self.placeholder)
            self.layer_linear_type.append(LinearType.INT)
        else:
            raise AssertionError(f"hunyuan-large not support quantize type: {quantize_type}")
        self.layer_linear_transpose_types.append(trans_flag)

    def register_layer_moe_linear_pack(self, layer, wrapper, quantize_type, linear_type, expert_roster):
        wrapper_module = get_moe_module(layer, wrapper.wrapper_name)
        pack_type = wrapper_module.pack_type
        if not self.shared_experts:
            self.register_layer_norm(layer, wrapper, pack_type)
        if linear_type == 'moe_mlp':
            router = get_moe_module(wrapper_module, wrapper.router_name)
            self.register_linear_wrapper(router, quantize_type)
            linear_stacked = get_moe_module(wrapper_module, wrapper.pack_name)
            self.register_linear_stacked_wrapper(linear_stacked, quantize_type)
            self.layer_linear_type.extend([LinearType.INVALID])
            self.layer_linear_transpose_types.extend([LinearType.INVALID])
        else:
            raise AssertionError(f'{linear_type} not yet implemented in register_layer_linear_pack')

    def register_layer_moe_mlp_experts(self, layer, wrapper, quantize_type, expert_roster):
        wrapper_module = get_moe_module(layer, wrapper.wrapper_name)
        pack_type = wrapper_module.pack_type
        if pack_type in ALL_PACK_LIST:
            self.register_layer_moe_linear_pack(layer, wrapper, quantize_type, 'moe_mlp', expert_roster)
        else:
            raise AssertionError("hunyuan-large not support gate up seprate yet")

        linear_stacked = get_moe_module(wrapper_module, wrapper.down_name)
        self.register_linear_stacked_wrapper(linear_stacked, quantize_type)

    def register_single_norm(self, norm):
        self.weights.append(norm.weight.data)
        if hasattr(norm, 'bias') and (getattr(norm, 'bias') is not None):
            self.weights.append(norm.bias.data)
        else:
            self.weights.append(self.placeholder)
        self.weights.extend([self.placeholder] * 2)

    def register_layer_attn(self, layer, wrapper, quantize_type):
        wrapper_module = get_module(layer, wrapper.wrapper_name)
        pack_type = wrapper_module.pack_type
        self.register_layer_norm(layer, wrapper, pack_type)
        if layer.self_attn.attention_type == 'cross':
            self.register_linear_wrapper(wrapper_module.q_proj.linear, quantize_type)
        else:
            self.register_linear_wrapper(wrapper_module.qkv_proj.linear, quantize_type)
        
        if layer.self_attn.use_qk_norm:
            q_norm = get_module(layer.self_attn, wrapper.q_norm_name)
            self.register_single_norm(q_norm)
            k_norm = get_module(layer.self_attn, wrapper.k_norm_name)
            self.register_single_norm(k_norm)

        o_linear = get_module(wrapper_module, wrapper.o_name).linear
        self.register_linear_wrapper(o_linear, quantize_type)