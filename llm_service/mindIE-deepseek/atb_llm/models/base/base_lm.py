# Copyright Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
import json
from abc import abstractmethod
from typing import ClassVar, Any
from dataclasses import dataclass

import torch

from atb_llm.models.base.config import BaseConfig
from atb_llm.utils.mapping import Mapping
from atb_llm.utils.weights import Weights
from atb_llm.utils.initial import load_atb_speed, NPUSocInfo, is_lcoc_enable
from atb_llm.utils.layers import PositionRotaryEmbedding, AttentionMask
from atb_llm.utils.layers.embedding.position_rotary_embedding import PositionEmbeddingType
from atb_llm.utils.layers.linear.linear_utils import LinearUtils
from atb_llm.utils.quantize.pack_type import QuantType
from atb_llm.utils.log import logger, print_log
from atb_llm.utils.log.error_code import ErrorCode


@dataclass
class InferenceParameter:
    """
    Class to encapsulate parameters for model inference.

    Attributes:
        inference_mode (Any | None): A flag indicating the type of the plugin features such as speculation,
            prefix cache, layer skip and splitfuse.
        enable_python_engine (bool): A flag indicating whether to enable the Python engine
        enable_lcoc (bool): A flag that indicates whether low-latency computation over
            communication is enabled
        skip_word_embedding (bool): When `skip_word_embedding` is true, input embedding should be provided and
            the word embedding module is skipped; otherwise, input token ids are used.
            It only takes effect in the prefill phase.
        enable_mc2 (bool): A flag indicating whether merged compute and communication is enabled.
        enable_matmul_nz (bool): A flag indicating whether matmul weights are converted to the NZ format.
    """
    inference_mode: Any | None = None
    enable_python_engine: bool = False
    enable_lcoc: bool = False
    skip_word_embedding: bool = False
    enable_mc2: bool = False
    enable_matmul_nz: bool = False

    def update_lcoc(self, need_nz: bool, reduce_quant_type: str) -> "InferenceParameter":
        """
        Updates the `enable_lcoc` attribute of the `InferenceParameter` instance
        based on the need_nz and reduce_quant_type parameters.

        Args:
            need_nz (bool): The hardware requirements of the tensor format.
            reduce_quant_type (str): The type of reduce quantization.
        Returns:
            InferenceParameter: The updated instance of `InferenceParameter`.
        """
        self.enable_lcoc = is_lcoc_enable(need_nz) and reduce_quant_type is None
        return self

    def update_matmul_nz(self, soc_info: NPUSocInfo, quantize: QuantType | None):
        """
        Updates the `enable_matmul_nz` attribute of the `InferenceParameter` instance
        based on the soc_info and quantize parameters.

        Args:
            soc_info (NPUSocInfo): Information about the hardware.
            quantize (QuantType | None): The type of quantization or None if not quantized.
        Returns:
            InferenceParameter: The updated instance of InferenceParameter.
        """
        self.enable_matmul_nz = (soc_info.soc_version == 225 or soc_info.soc_version == 223) \
            and not self.enable_mc2 and ((quantize is None) or (quantize == QuantType.FLOAT))
        soc_info.matmul_nd_nz = self.enable_matmul_nz
        LinearUtils.soc_info = soc_info
        return self


@dataclass
class ConfigMetadata:
    """
    Dataclass representing configuration metadata for a model.

    Attributes:
        num_attention_heads (int): Number of attention heads in the model.
            It will be divided by tensor parallel size in `update_head_num` method.
        num_key_value_heads (int): Number of key-value heads in the model.
            It will be divided by tensor parallel size in `update_kv_head_num` method.
        num_hidden_layers (int): Number of hidden layers in the model.
            It will be divided by pipeline parallel size in `update_num_hidden_layers` method.
        head_dim (int): Dimension of each attention head.
        position_embedding_type (PositionEmbeddingType): Type of position embedding used in the model.
    """
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    num_hidden_layers: int = 0
    head_dim: int = 0
    position_embedding_type: PositionEmbeddingType = PositionEmbeddingType.ROPE

    def update_head_dim(self, hidden_size: int) -> "ConfigMetadata":
        """
        Updates the dimension of each attention head based on the total hidden size and the number of attention heads.
        Args:
            hidden_size (int): The total hidden size of the model.
        Returns:
            ConfigMetadata: The updated instance of ConfigMetadata.
        """
        self.head_dim = hidden_size // self.num_attention_heads
        return self

    def update_head_num(self, tp_group_size: int) -> "ConfigMetadata":
        """
        Updates the number of attention heads based on the group size of tensor parallelism.
        Args:
            tp_group_size (int): The group size of tensor parallelism.
        Returns:
            ConfigMetadata: The updated instance of ConfigMetadata.
        """
        self.num_attention_heads = (self.num_attention_heads + tp_group_size - 1) // tp_group_size
        return self

    def update_kv_head_num(self, tp_group_size: int) -> "ConfigMetadata":
        """
        Updates the number of key-value heads based on the group size of tensor parallelism.
        Args:
            tp_group_size (int): The size of the TP group.
        Returns:
            ConfigMetadata: The updated instance of ConfigMetadata.
        """
        if self.num_key_value_heads < tp_group_size:
            repeat_times = tp_group_size // self.num_key_value_heads
        else:
            repeat_times = 1
        self.num_key_value_heads = (self.num_key_value_heads * repeat_times + tp_group_size - 1) \
            // tp_group_size
        return self

    def update_num_hidden_layers(self, mapping: Mapping) -> "ConfigMetadata":
        """
        Updates the number of hidden layers based on pipeline parallelism info.
        Args:
            mapping (Mapping): A mapping that contains pipeline parallelism information.
        Returns:
            ConfigMetadata: The updated instance of ConfigMetadata.
        """
        if mapping.has_pp():
            self.num_hidden_layers = len(mapping.pp_layers(self.num_hidden_layers))
        return self

    def update_position_embedding_type(self, pe_type: str) -> "ConfigMetadata":
        """
        Updates the position embedding type based on the provided type.
        Args:
            pe_type (str): The type of position embedding.
        Returns:
            ConfigMetadata: The updated instance of ConfigMetadata.
        """
        if pe_type == "ROPE":
            self.position_embedding_type = PositionEmbeddingType.ROPE
        elif pe_type == "ALIBI":
            self.position_embedding_type = PositionEmbeddingType.ALIBI
        return self


@dataclass
class PostionalEmbeddingInfo:
    """
    Class to encapsulate information related to positional embeddings.

    Attributes:
        generator (PositionRotaryEmbedding | None):
            The generator used to generate the positional embedding tables.
        cosine_table (torch.Tensor | None): The cosine table used for positional embeddings.
        sine_table (torch.Tensor | None): The sine table used for positional embeddings.
    """
    generator: PositionRotaryEmbedding | None = None
    cosine_table: torch.Tensor | None = None
    sine_table: torch.Tensor | None = None


@dataclass
class AttentionMaskInfo:
    """
    Dataclass to encapsulate information related to attention masks.

    Attributes:
        MAX_BASE_LEN (int): Class variable representing the maximum base length for attention masks.
        generator (AttentionMask | None): The generator used to generate the attention masks.

    """
    MAX_BASE_LEN: ClassVar[int] = 128
    generator: AttentionMask | None = None


class BaseLM:
    """The Base class for all LLM models. It contains the main entrance for the inference process.

    It initializes various components, such as the model structure, weights, configuration,
    and hardware-related information, which are necessary for the model 
    to function correctly during the inference phase.
    """
    def __init__(self, config: BaseConfig, weights: Weights, **kwargs):
        # load so
        load_atb_speed()

        # transdata operation
        self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        self.transdata_param = json.dumps({})
        self.transdata_operation.set_param(self.transdata_param)

        # model structure
        self.model = None
        self.lm_head = None

        # model weights
        self.torch_device = weights.device
        self.torch_dtype = weights.dtype
        self.mapping = weights.mapping
        self.device_weights = []

        # model config
        self.config = config
        if getattr(self.config, "num_key_value_heads") is None:
            setattr(self.config, "num_key_value_heads", config.num_attention_heads)
        if self.config.rope_scaling.rope_type is None:
            self.config.rope_scaling.rope_type = self.config.rope_scaling.type
        if not hasattr(self.config, "rope_theta"):
            setattr(self.config, "rope_theta", 10000.0)
        if not hasattr(self.config, "alibi_bias_max"):
            setattr(self.config, "alibi_bias_max", 8.0)
        if not hasattr(self.config, "pe_type"):
            setattr(self.config, "pe_type", "ROPE")
        self.config_metadata = ConfigMetadata(
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            num_hidden_layers=config.num_hidden_layers,
        ).update_head_dim(
            config.hidden_size
        ).update_head_num(
            self.mapping.attn_tp.group_size
        ).update_kv_head_num(
            self.mapping.attn_tp.group_size
        ).update_num_hidden_layers(
            self.mapping
        ).update_position_embedding_type(
            config.pe_type
        )

        # hardware related
        self.soc_info = NPUSocInfo()
        print_log(self.mapping.rank, logger.info, self.soc_info)

        # inference parameter
        self.infer_param = InferenceParameter(
            inference_mode=kwargs.get("inference_mode"),
            enable_python_engine=kwargs.get("enable_python_engine", False),
        ).update_lcoc(
            self.soc_info.need_nz, config.quantization_config.reduce_quant_type
        )

        # engine input
        self.pos_embed_info = PostionalEmbeddingInfo()
        self.init_positional_embedding()
        self.attn_mask_info = AttentionMaskInfo()
        self.init_mask()
        self.placeholder = torch.zeros(1, dtype=self.torch_dtype, device="npu")
        self.engine_inputs = []
        self.engine_runtime_param = {}

    @abstractmethod
    def get_device_weights(self, **kwargs):
        """
        Retrieves the weights of a model according to its architecture.
        Weights are already loaded onto the device.
        """
        pass

    @abstractmethod
    def build_engine(self, **kwargs) -> None:
        """
        Builds inference engines for the model based on model parameters.
        """
        pass

    @abstractmethod
    def update_kv_cache(self, *args, **kwargs) -> None:
        """
        Refresh the key-value cache.
        """
        pass

    @abstractmethod
    def init_mask(self, **kwargs) -> None:
        """
        Initialize attention mask.
        """
        pass

    @abstractmethod
    def generate_mask(self, **kwargs) -> torch.Tensor:
        """
        Generate a attention mask for each batch of input tokens.
        """
        pass

    def init_positional_embedding(self, **kwargs) -> None:
        """
        Initialize the positional embedding generator.

        This method initializes the positional embedding generator based on the configuration metadata.
        The generator is an instance of `PositionRotaryEmbedding`, and it is created using the static method.
        The static method takes parameters such as the dimension (dim), base, device, and scaling factor.
        These parameters are obtained from the configuration metadata and the rope scaling configuration.

        Args:
            **kwargs: Keyword arguments for additional parameters (not used in this method).
        """
        self.pos_embed_info.generator = PositionRotaryEmbedding.static(
            dim=self.config_metadata.head_dim,
            base=self.config.rope_theta,
            device="cpu",
            scaling_factor=self.config.rope_scaling.factor).to(self.torch_device)

    def generate_positional_embedding(self, max_seq_len: int, **kwargs) -> None:
        """
        Generate the positional embedding tables.

        This method generates the positional embedding tables based on the position embedding type
        specified in the configuration metadata.
        If the type is `ROPE`, it updates the cosine and sine caching for the total sequence length.
        If the type is `ALIBI`, it sets the cosine and sine tables to placeholders.
        Otherwise, it raises an error.

        Args:
            max_seq_len (int): The maximum sequence length for which to generate the positional embeddings.
            **kwargs: Keyword arguments for additional parameters (not used in this method).

        Raises:
            logger.error: If the position_embedding_type is neither ROPE nor ALIBI.
        """
        if self.config_metadata.position_embedding_type == PositionEmbeddingType.ROPE:
            if self.config.rope_scaling.rope_type == "linear":
                self.pos_embed_info.generator.update_cos_sin_cache_total(
                    self.torch_dtype, self.torch_device, max_seq_len)
            elif self.config.rope_scaling.rope_type == "llama3":
                self.pos_embed_info.generator.update_llama3_cos_sin_cache_total(
                    self.config, self.torch_dtype, self.torch_device, max_seq_len)
            self.pos_embed_info.cosine_table = self.pos_embed_info.generator.get_cos_cached_total()
            self.pos_embed_info.sine_table = self.pos_embed_info.generator.get_sin_cached_total()
        elif self.config_metadata.position_embedding_type == PositionEmbeddingType.ALIBI:
            self.pos_embed_info.cosine_table = self.placeholder
            self.pos_embed_info.sine_table = self.placeholder
        else:
            logger.error("Error: position_embedding_type is illegal",
                         ErrorCode.ATB_MODELS_engine_static_param_JSON_INVALID)

    @abstractmethod
    def prepare_inputs(self, *args, **kwargs) -> None:
        """
        Preparing all the necessary inputs before execute an engine.
        """
        pass

    @abstractmethod
    def execute_engine(self, **kwargs) -> torch.Tensor:
        """
        Select an engine according to the provided parameters and inputs,
        then run the chosen engine to produce the model output.

        Returns:
            torch.Tensor: Logits.
        """
        pass

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Perform a forward pass through the model.
        This serves as the primary entry point to execute a forward computation.
        """
        pass
