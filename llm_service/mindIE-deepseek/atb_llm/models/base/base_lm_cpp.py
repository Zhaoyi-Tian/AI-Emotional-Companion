# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json

import torch

from atb_llm.models.base.base_lm import BaseLM
from atb_llm.models.base.config import BaseConfig
from atb_llm.utils.weights import Weights
from atb_llm.utils.data.weight_wrapper import WeightWrapper, AttnWrapper, MlpWrapper
from atb_llm.models.base.engine.engine_manager import engine_manager
from atb_llm.utils.op_backend import OpBackend
from atb_llm.utils.layers.norm.fast_layer_norm import NormType


class BaseLMCpp(BaseLM):
    """
    Base class for language models that leverages an inference engine implemented in C++.

    Args:
        config (BaseConfig): model configuration.
        weights (Weights): Weights object to load model weights.
        **kwargs: Additional keyword arguments.
    """
    def __init__(self, config: BaseConfig, weights: Weights, **kwargs):
        super().__init__(config, weights, **kwargs)

        # model weights
        self.attn_wrapper = AttnWrapper(
            norm_name='input_layernorm',
            wrapper_name='self_attn',
            pack_name='query_key_value',
            sep_names=['q_proj', 'k_proj', 'v_proj'],
            o_name='o_proj'
        )

        self.mlp_wrapper = MlpWrapper(
            norm_name='post_attention_layernorm',
            wrapper_name='mlp',
            pack_name='gate_up_proj',
            sep_names=['gate_proj', 'up_proj'],
            down_name='down_proj'
        )

        self.attn_decode_backend = OpBackend.ATB

        # engine static parameters
        self.engine_static_param = None

    @property
    def model_torch_class_name(self) -> str:
        """
        Returns the name of the PyTorch class used by the model.
        This method is intended to be overridden by subclasses.
        """
        pass

    def update_engine_static_param(self):
        """
        Sets the inference engine's static parameters based on the model configuration and parameters.
        These parameters will be pass to the C++ class identified by `model_torch_class_name`.
        Returns:
            dict: A dictionary of parameters.
        """
        return {
            "normEps": self.config.rms_norm_eps,
            "normType": NormType.RMS_NORM,
            "positionEmbeddingType": self.config_metadata.position_embedding_type,
            "numAttentionHeadsPerRank": self.config_metadata.num_attention_heads,
            "hiddenSizePerAttentionHead": self.config_metadata.head_dim,
            "numHiddenLayers": self.config_metadata.num_hidden_layers,
            "numKeyValueHeadsPerRank": self.config_metadata.num_key_value_heads,
            "isBF16": self.torch_dtype == torch.bfloat16,
            "isLmHeadParallel": True,
            "rank": self.mapping.rank,
            "worldSize": self.mapping.world_size,
            "backend": self.soc_info.communication_backend,
            "attnBackend": self.attn_decode_backend,
            "enableLcoc": self.infer_param.enable_lcoc,
            "enableAddNorm": False,
            "enableSwiGLU": False if self.soc_info.need_nz else True,
            "enableKvQuant": self.config.quantization_config.kv_quant_type is not None,
            "enableFA3": self.config.quantization_config.fa_quant_type is not None,
            "enableReduceQuant": self.config.quantization_config.reduce_quant_type is not None,
            "quantGroupSize": self.config.quantization_config.group_size,
        }

    def get_device_weights(self, **kwargs) -> WeightWrapper:
        """
        Retrieves the weights for the device.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            WeightWrapper: The weight wrapper object contains weight metadata.
        """
        weight_wrapper = WeightWrapper(self.soc_info, self.mapping.rank, self.attn_wrapper, self.mlp_wrapper)
        weight_wrapper.register_embedding(self.model.embed_tokens)
        for i in range(self.config_metadata.num_hidden_layers):
            layer = self.model.layers[i]
            weight_wrapper.register_layer(layer, self.config.quantize)
            if self.config.quantization_config.kv_quant_type is not None:
                weight_wrapper.register_layer_kvquant(layer)
            if self.config.quantization_config.fa_quant_type is not None:
                weight_wrapper.register_layer_qkvquant(layer)
            if self.config.quantization_config.reduce_quant_type is not None:
                weight_wrapper.register_layer_reducequant(layer)
        weight_wrapper.register_model_norm(self.model.norm)
        weight_wrapper.register_model_lmhead(self.lm_head)
        self.device_weights = weight_wrapper.weights
        return weight_wrapper

    def build_engine(self, **kwargs) -> None:
        """
        Builds inference engines for the model based on model parameters.

        Args:
            **kwargs: Additional keyword arguments.
        """
        self.engine_static_param = self.update_engine_static_param()

        weight_wrapper = self.get_device_weights()
        linear_types = weight_wrapper.linear_type
        pack_quant_configs = weight_wrapper.pack_quant_type
        linear_transpose_types = weight_wrapper.linear_transpose_types
        self.engine_static_param.update({
            "packQuantType": pack_quant_configs,
            "linearQuantType": linear_types,
            "linearTransposeType": linear_transpose_types,
            "lmHeadTransposeType": self.lm_head.linear.trans_flag,
        })

        engine_manager.create_engines(self)
        for engine in engine_manager.get_engines():
            engine.set_weight(self.device_weights)

    def execute_engine(self, **kwargs) -> torch.Tensor:
        """
        Executes the engine to generate the model's output.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Logits.

        Raises:
            RuntimeError: If engine execution fails.
        """
        engine = engine_manager.choose_engine(self, **kwargs)
        try:
            engine_outputs = engine.execute(self.engine_inputs, json.dumps(self.engine_runtime_param))
            hidden_state = engine_outputs[0]
        except IndexError as e:
            raise RuntimeError("Engine execution error. "
                               "Enable log: export ASDOPS_LOG_LEVEL=ERROR, "
                               "export ASDOPS_LOG_TO_STDOUT=1 to find the first error. "
                               "For more details, see the MindIE official document.") from e
        return hidden_state
