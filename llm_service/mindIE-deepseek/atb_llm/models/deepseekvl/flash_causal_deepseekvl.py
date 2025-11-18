# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
# Copyright (c) 2023-2024 DeepSeek.

"""PyTorch deepseekvl model."""

import abc
import importlib
import os
import warnings
import sys
from typing import Optional, List, Tuple

import torch
import numpy as np
from PIL import Image

from einops import rearrange
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from atb_llm.utils.shm_utils import get_data_from_shm

from .clip_encoder import CLIPVisionTower, HybridVisionTower
from .processing_vlm import VLChatProcessor
from .projector import MlpProjector
from ..base.config import QuantizationConfig, BaseConfig
from ..base.flash_causal_lm import FlashForCausalLM
from ..base.model_utils import safe_from_pretrained
from ..llama.config_llama import LlamaConfig
if sys.version_info >= (3, 10):
    # Monkey patch collections
    import collections
    import collections.abc
    for type_name in collections.abc.__all__:
        setattr(collections, type_name, getattr(collections.abc, type_name))
from attrdict import AttrDict

MODEL_TYPE = "model_type"
LLAMA = "llama"
_PAD_TOKEN_ID = 32001
_IMAGE_TOKEN_ID = 100015


def model_name_to_cls(cls_name):
    class_mapping = {
        "MlpProjector": MlpProjector,
        "CLIPVisionTower": CLIPVisionTower,
        "HybridVisionTower": HybridVisionTower,
    }

    try:
        return class_mapping[cls_name]
    except KeyError as e:
        raise ValueError(f"class_name {cls_name} is invalid.") from e


def get_supported_models():
    current_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    supported_models = []
    for folder_name in os.listdir(current_path):
        is_folder = os.path.isdir(os.path.join(current_path, folder_name))
        skip_base_folder = folder_name != "base"
        skip_invalid_folder = not folder_name.startswith("_")
        if is_folder and skip_base_folder and skip_invalid_folder:
            supported_models.append(folder_name)
    return supported_models


def get_llm_model(model_type):
    supported_models = get_supported_models()
    if model_type not in supported_models:
        raise NotImplementedError(
            f"unsupported model type: {model_type};"
            f"请确认atb_llm.models路径下是否存在名为{model_type}的文件夹。"
        )

    model_file_dir_name = f"atb_llm.models.{model_type}."
    model_file_name_prefix = 'flash_causal'
    module_path = f"{model_file_dir_name}{model_file_name_prefix}_{model_type}"
    module = importlib.import_module(module_path)
    model_cls_name = "Flash" + f"{model_type.capitalize()}ForCausalLM"
    model_cls = getattr(module, model_cls_name)
    return model_cls


class BaseModelConfig(PretrainedConfig):
    model_type = ""
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = kwargs.get("model_type", "")
        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__
        self.params = AttrDict(kwargs.get("params", {}))


class MultiModalConfig(PretrainedConfig):
    def __init__(self, vision_config=None, language_config=None, aligner_config=None, **kwargs):
        self._init_vision_config(vision_config)
        self._init_language_config(language_config)
        self._init_aligner_config(aligner_config)
        super().__init__(**kwargs)

    def _init_vision_config(self, vision_config):
        if vision_config:
            self.vision_config = BaseModelConfig(**vision_config)

    def _init_language_config(self, language_config):
        if language_config:
            self.language_config = LlamaConfig(**language_config)
            self._vocab_size = self.language_config.vocab_size

    def _init_aligner_config(self, aligner_config):
        if aligner_config:
            self.aligner_config = BaseModelConfig(**aligner_config)


class DeepseekvlConfig(MultiModalConfig):
    model_type = "deepseekvl"

    def __init__(
            self,
            vision_config=None,
            language_config=None,
            aligner_config=None,
            ignore_index=-100,
            image_token_index=32000,
            projector_hidden_act="gelu",
            vision_feature_select_strategy="default",
            vision_feature_layer=-2,
            **kwargs,
    ):
        self._vocab_size = None
        super().__init__(vision_config, language_config, aligner_config, **kwargs)

    @property
    def vocab_size(self):
        warnings.warn(
            "The `vocab_size` attribute is deprecated and will be removed in v4.42, \
            Please use `language_config.vocab_size` instead.",
            FutureWarning,
        )
        return self._vocab_size

    @vocab_size.setter
    def vocab_size(self, value):
        self._vocab_size = value

    @staticmethod
    def recursive_to_dict(obj):
        # Recursive function to apply to_dict on all objects that have it
        if isinstance(obj, dict):
            return {key: DeepseekvlConfig.recursive_to_dict(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [DeepseekvlConfig.recursive_to_dict(item) for item in obj]
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        return obj

    def to_dict(self):
        output = super().to_dict()
        output.pop("_vocab_size", None)
        # Apply to_dict on all objects in output
        output = DeepseekvlConfig.recursive_to_dict(output)
        return output


class MultiModalLLm(FlashForCausalLM):
    def __init__(self, config, weights, **kwargs):
        self.model_type = None
        self.vision_model = None
        self.language_model = None
        if getattr(config, "language_config"):
            if not config.quantize:
                setattr(config.language_config, 'quantize', None)
            else:
                setattr(config.language_config, 'quantize', config.quantize)
            setattr(config.language_config, 'quantization_config', QuantizationConfig(**{}))
            super().__init__(config.language_config, weights, **kwargs)
        else:
            super().__init__(config, weights, **kwargs)
        self.config = config
        self.weights = weights
        self.vocab_size = config.language_config.vocab_size
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.init_vit()
        self.init_llm()

    @staticmethod
    def init_weights(module, weights, prefix):
        for weight_name in module.state_dict().keys():
            saved_weight = torch.nn.Parameter(
                weights.get_tensor(f"{prefix}.{weight_name}"),
                requires_grad=False
            )
            weight_list = weight_name.split(".")
            target_module = module
            for nxt_module in weight_list[:-1]:
                target_module = getattr(target_module, nxt_module)
            setattr(target_module, weight_list[-1], saved_weight)

    def init_vit(self):
        model_config = getattr(self.config, "vision_config")
        model_cls = model_name_to_cls(model_config.cls)
        self.vision_model = model_cls(**model_config.params).to(device=self.weights.device, dtype=self.weights.dtype)
        self.init_weights(self.vision_model, self.weights, "vision_model")

    def init_llm(self):
        model_cls = get_llm_model(self.config.language_config.model_type)
        self.language_model = model_cls(self.config.language_config,
                                        self.weights,
                                        "language_model.lm_head",
                                        "language_model.model")
        self.language_model.skip_word_embedding = True

    @abc.abstractmethod
    def prepare_prefill_token_service(self, input_ids):
        pass

    def forward(
            self,
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
        return self.language_model.forward(input_ids,
                                           position_ids,
                                           is_prefill,
                                           kv_cache,
                                           block_tables,
                                           slots,
                                           input_lengths,
                                           max_seq_len,
                                           lm_head_indices)


class FlashDeepseekvlForCausalLM(MultiModalLLm):
    def __init__(self, config, weights, **kwargs):
        self.aligner = None
        super().__init__(config, weights, **kwargs)
        self.config = config
        self.vocab_size = config.language_config.vocab_size
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else _PAD_TOKEN_ID
        self.image_token_id = _IMAGE_TOKEN_ID
        self.processor = safe_from_pretrained(VLChatProcessor, self.config.model_name_or_path)
        self.init_multimodal()

    def init_multimodal(self):
        model_config = getattr(self.config, "aligner_config")
        model_cls = model_name_to_cls(model_config.cls)
        self.aligner = model_cls(model_config.params).to(device=self.weights.device, dtype=self.weights.dtype)
        self.init_weights(self.aligner, self.weights, "aligner")

    def get_input_embeddings(self):
        return self.language_model.model.embed_tokens

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.language_config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def prepare_prefill_token_service(self, input_ids):
        shared_tensor = None
        if torch.any(torch.eq(input_ids, self.image_token_id)):
            bos_pos = torch.where(torch.eq(input_ids, self.processor.image_id))[0]
            shm_value = input_ids[bos_pos + 1]
            shape_value = input_ids[bos_pos + 2]
            shared_tensor = get_data_from_shm(shm_value, shape_value, np.float32, self.weights.device)
            input_ids = torch.cat((input_ids[:bos_pos + 1], input_ids[(bos_pos + self.processor.num_image_tokens):]))
        prepare_inputs = self.processor(conversations=input_ids, images=shared_tensor, force_batchify=True).to(
            self.weights.device)
        inputs_embeds = self.prepare_inputs_embeds(**prepare_inputs)
        inputs_embeds = inputs_embeds.reshape(inputs_embeds.shape[0] * inputs_embeds.shape[1], inputs_embeds.shape[2])

        return inputs_embeds

    def prepare_inputs_embeds(
            self,
            input_ids: torch.LongTensor,
            pixel_values: torch.FloatTensor,
            images_seq_mask: torch.LongTensor,
            images_emb_mask: torch.LongTensor,
            **kwargs,
    ):
        """

        Args:
            input_ids (torch.LongTensor): [b, T]
            pixel_values (torch.FloatTensor):   [b, n_images, 3, h, w]
            images_seq_mask (torch.BoolTensor): [b, T]
            images_emb_mask (torch.BoolTensor): [b, n_images, n_image_tokens]

            assert torch.sum(images_seq_mask) == torch.sum(images_emb_mask)

        Returns:
            input_embeds (torch.Tensor): [b, T, D]
        """

        bs, n = pixel_values.shape[0:2]
        images = rearrange(pixel_values, "b n c h w -> (b n) c h w")
        images_embeds = self.vision_model(images)
        # [b x n, T2, D]
        if isinstance(images_embeds, tuple):
            images_embeds = self.aligner(images_embeds).to(images_embeds[0].dtype)
        else:
            images_embeds = self.aligner(images_embeds).to(images_embeds.dtype)

        # [b x n, T2, D] -> [b, n x T2, D]
        images_embeds = rearrange(images_embeds, "(b n) t d -> b (n t) d", b=bs, n=n)
        # [b, n, T2] -> [b, n x T2]
        images_emb_mask = rearrange(images_emb_mask, "b n t -> b (n t)")

        # The shape is [b, T, D]
        input_ids[input_ids < 0] = 0  # ignore the image embeddings
        inputs_embeds = self.get_input_embeddings()(input_ids)

        # replace with the image embeddings
        inputs_embeds[images_seq_mask] = images_embeds[images_emb_mask]

        return inputs_embeds

    def prepare_prefill_token(self, multimodalinputs, processor):
        text = multimodalinputs.text
        image = multimodalinputs.image
        video = multimodalinputs.video
        conversation = processor.new_chat_template()
        conversation.append_message(conversation.roles[0], (text, image))
        conversation.append_message(conversation.roles[1], "")

        prompts = conversation.generate_prompt_with_conversation()

        pil_images = []
        for message in prompts:
            if "images" in message:
                for img in message["images"]:
                    pil_image = Image.open(img).convert("RGB")
                    pil_images.append(pil_image.copy())
                    pil_image.close()

        prepare_inputs = processor(conversations=prompts, images=pil_images, force_batchify=True).to(
            self.weights.device)
        inputs_embeds = self.prepare_inputs_embeds(**prepare_inputs)
        inputs_embeds = inputs_embeds.reshape(inputs_embeds.shape[0] * inputs_embeds.shape[1], inputs_embeds.shape[2])

        return inputs_embeds

    def init_ascend_operations(self, config: BaseConfig):
        pass

    def init_ascend_weight(self):
        pass

    def forward(
            self,
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
        if is_prefill and input_ids.dim() == 1:
            input_ids = self.prepare_prefill_token_service(input_ids)
        return self.language_model.forward(input_ids,
                                           position_ids,
                                           is_prefill,
                                           kv_cache,
                                           block_tables,
                                           slots,
                                           input_lengths,
                                           max_seq_len,
                                           lm_head_indices)
