# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

from atb_llm.models.base.flash_causal_lm_v2 import FlashCausalLMV2
from atb_llm.models.base.config import BaseConfig
from atb_llm.models.qwen2.modeling_qwen2 import FlashQwenModel
from atb_llm.utils.data.weight_wrapper import WeightWrapper, AttnWrapper, MlpWrapper
from atb_llm.utils.weights import Weights 
from atb_llm.utils.layers import load_column_multi, TensorHead, TensorEmbedding 
from atb_llm.utils.op_backend import OpBackend


class FlashQwen2ForCausalLMV2(FlashCausalLMV2):
    """
    This class serves as the primary functional class that inherits from the `FlashCausalLMV2` class.
    It is responsible for constructing the model architecture by integrating the FlashQwenModel.
    """
    def __init__(self, config: BaseConfig, weights: Weights, lmhead_prefix="lm_head", model_prefix="model", **kwargs):
        super().__init__(config, weights, **kwargs)
        self.infer_param.update_matmul_nz(
            self.soc_info, config.quantize
        )

        # model structure
        transformer_wte_parallel = kwargs.get("transformer_wte_parallel", True)

        self.transformer = FlashQwenModel(config, weights, model_prefix=model_prefix, lmhead_prefix=lmhead_prefix)

        if not transformer_wte_parallel:
            self.transformer.wte = TensorEmbedding(
                prefix=f"{model_prefix}.embed_tokens", weights=weights
            )
            for p in self.transformer.wte.parameters():
                p.requires_grad = False        

        if self.config.quantize == "w8a8sc":
            self.lm_head = TensorHead.load_weight(
                config,
                prefix="lm_head",
                weights=weights,
                is_norm=False,
            )
        else:
            if config.tie_word_embeddings:
                self.lm_head = load_column_multi(
                    config,
                    prefixes=[f"{model_prefix}.embed_tokens"],
                    weights=weights,
                    head_size=1,
                    lm_head=True,
                )
            else:
                self.lm_head = load_column_multi(
                    config,
                    prefixes=[f"{lmhead_prefix}"],
                    weights=weights,
                    head_size=1,
                    lm_head=True,
                )

        self.attn_wrapper = AttnWrapper(
            norm_name='ln_1',
            wrapper_name='attn',
            pack_name='c_attn',
            sep_names=['q_proj', 'k_proj', 'v_proj'],
            o_name='c_proj'
        )
        self.mlp_wrapper = MlpWrapper(
            norm_name='ln_2',
            wrapper_name='mlp',
            pack_name='w2_w1',
            sep_names=['w2', 'w1'],
            down_name='c_proj'
        )

    @property
    def model_torch_class_name(self):
        """
        This method returns the name of the PyTorch class for the model.
        """
        return "qwen_QwenDecoderModel"

    def update_engine_static_param(self):
        """
        The method is responsible for setting the static parameters for the engine.
        It accomplishes this by first obtaining a set of default parameters by calling
        the `update_engine_static_param method` from the `FlashCausalLMV2` class.
        Afterward, it updates these default parameters by adding the following settings:
        whether to utilize tensor parallelism in word embedding.
        """
        engine_static_param = super().update_engine_static_param()
        engine_static_param.update({
            "isEmbeddingParallel": True,
            "linearHasBias": [[True, False, False, False]] * self.config.num_hidden_layers,
            "enableIntraLayerAddNorm": False,
            "enableInterLayerAddNorm": False,
            "enableQScale": (self.config.transformers_version == "4.43.1" or
                            self.config.transformers_version == "4.44.0") and \
                            self.config.num_hidden_layers == 28 and \
                            self.soc_info.need_nz,
                            # QwenCode2.5-7B-Instruct/Qwen2.5-7B/1.5B-Instruct模型时为True, 其他模型为False
            "matmulBackend": OpBackend.ACLNN if self.soc_info.matmul_nd_nz else OpBackend.ATB,
        })
        return engine_static_param

    def get_device_weights(self, **kwargs) -> WeightWrapper:
        weight_wrapper = WeightWrapper(self.soc_info, self.mapping.attn_tp.rank, self.attn_wrapper, self.mlp_wrapper)
        weight_wrapper.register_embedding(self.transformer.wte)
        for i in range(self.config_metadata.num_hidden_layers):
            layer = self.transformer.h[i]
            weight_wrapper.register_layer(layer, self.config.quantize)
            if self.soc_info.need_nz and self.lora_decorator.adapter_manager is None:
                del layer.attn
                del layer.ln_2
                del layer.mlp
            if self.config.quantization_config.kv_quant_type is not None:
                weight_wrapper.register_layer_kvquant(layer)
            if self.config.quantization_config.fa_quant_type is not None:
                weight_wrapper.register_layer_qkvquant(layer)
        weight_wrapper.register_model_norm(self.transformer.ln_f)
        weight_wrapper.register_model_lmhead(self.lm_head)
        self.device_weights = weight_wrapper.weights
        return weight_wrapper