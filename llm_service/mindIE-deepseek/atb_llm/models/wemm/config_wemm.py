# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.

""" wemm model configuration"""
from transformers import PretrainedConfig

from ..base.config import BaseConfig


class DownsamplerConfig(PretrainedConfig):
    model_type = "downsampler"
    _auto_class = "AutoConfig"

    def __init__(
        self,
        kernel_size=1,
        stride=1,
        visual_hidden_size=4096,
        llm_hidden_size=4096,
        depth=2,
        hidden_act="gelu",
        bias=False,
        **kwargs,
    ):
        self.visual_hidden_size = visual_hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.depth = depth
        self.hidden_act = hidden_act
        self.bias = bias
        self.kernel_size = kernel_size
        self.stride = stride
        super().__init__(**kwargs)


class Idefics2VisionConfig(PretrainedConfig):
    _auto_class = "AutoConfig"
    model_type = "Idefics2VisionConfig"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=32,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        initializer_range=0.02,
        model_type="Idefics2VisionConfig",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range


class WemmConfig(BaseConfig):
    model_type = "wemm"

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        adapter_path=None,
        image_processor=None,
        do_image_splitting=False,
        spliter_emb_config=None,
        downsampler_config=None,
        tokenizer_path=None,
        **kwargs,
    ):
        super().__init__(**text_config)
        self.model_type = "wemm"
        self.do_image_splitting = do_image_splitting

        if vision_config is not None:
            self.vision_config = Idefics2VisionConfig(**vision_config)

        if image_processor is not None:
            self.image_processor = image_processor

        if adapter_path is not None:
            self.adapter_path = adapter_path

        if spliter_emb_config is not None:
            self.spliter_emb_config = spliter_emb_config

        if downsampler_config is not None:
            self.downsampler_config = DownsamplerConfig(**downsampler_config)

        if tokenizer_path is not None:
            self.tokenizer_path = tokenizer_path
