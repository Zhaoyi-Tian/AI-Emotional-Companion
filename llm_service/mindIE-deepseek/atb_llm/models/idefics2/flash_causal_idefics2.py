# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
# Copyright 2023 the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Llava model."""

from typing import Optional, List, Tuple
from abc import ABC, abstractmethod
import importlib
import os
import warnings
import PIL
import PIL.Image
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.idefics2.modeling_idefics2 import Idefics2Connector, Idefics2VisionTransformer
from transformers.models.idefics2.configuration_idefics2 import Idefics2VisionConfig
from transformers.cache_utils import Cache, DynamicCache
import torch

from ..base.flash_causal_lm import FlashForCausalLM
from ..llama.config_llama import LlamaConfig
from ..base.config import QuantizationConfig


MODEL_TYPE = "model_type"
LLAMA = "llama"
MISTRAL = "mistral"
VICUNA = "vicuna"


def get_supported_models():
    current_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    supported_models = []
    for foldername in os.listdir(current_path):
        is_folder = os.path.isdir(os.path.join(current_path, foldername))
        skip_base_folder = foldername != "base"
        skip_invalid_folder = not foldername.startswith("_")
        if is_folder and skip_base_folder and skip_invalid_folder:
            supported_models.append(foldername)
    return supported_models


def get_llm_model(model_type):
    supported_models = get_supported_models()
    if model_type not in supported_models:
        raise NotImplementedError(
            f"unsupported model type: {model_type};"
            f"请确认atb_llm.models路径下是否存在名为{model_type}的文件夹。"
        )

    model_file_dir_name = f"atb_llm.models.{model_type}."
    model_file_name = 'flash_causal'
    module_path = f"{model_file_dir_name}{model_file_name}_{model_type}"
    module = importlib.import_module(module_path)
    model_cls_name = "Flash" + f"{model_type.capitalize()}ForCausalLM"
    model_cls = getattr(module, model_cls_name)
    return model_cls


class MultiModalConfig(PretrainedConfig):
    model_type = "idefics2"
    is_composition = False

    def __init__(self, vision_config=None, text_config=None, perceiver_config=None, **kwargs):
        self._init_visionconfig(vision_config)
        self._init_textconfig(text_config)
        self._init_perceiverconfig(perceiver_config)
        super().__init__(**kwargs)


    def _init_visionconfig(self, vision_config):
        if isinstance(vision_config, dict):
            vision_config[MODEL_TYPE] = (
                vision_config[MODEL_TYPE] if MODEL_TYPE in vision_config else "clip_vision_model"
            )
            vision_config = Idefics2VisionConfig(**vision_config)
        elif vision_config is None:
            vision_config = CONFIG_MAPPING["clip_vision_model"](
                intermediate_size=4096,
                hidden_size=1024,
                patch_size=14,
                image_size=336,
                num_hidden_layers=24,
                num_attention_heads=16,
                vocab_size=32000,
                projection_dim=768,
            )

        self.vision_config = vision_config

    def _init_perceiverconfig(self, perceiver_config):
        if isinstance(perceiver_config, dict):
            perceiver_config[MODEL_TYPE] = (
                perceiver_config[MODEL_TYPE] if MODEL_TYPE in perceiver_config else "clip_vision_model"
            )
            perceiver_config = CONFIG_MAPPING[perceiver_config[MODEL_TYPE]](**perceiver_config)
        self.perceiver_config = perceiver_config

    def _init_textconfig(self, text_config):
        if isinstance(text_config, dict):
            model_type = text_config[MODEL_TYPE] if MODEL_TYPE in text_config else LLAMA
            if model_type in [MISTRAL, VICUNA, LLAMA]:
                text_config = LlamaConfig(**text_config)
            else:
                text_config = CONFIG_MAPPING[model_type](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING[LLAMA]()
        self.text_config = text_config


class Idefics2Config(MultiModalConfig):

    model_type = "idefics2"
    is_composition = False

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        perceiver_config=None,
        ignore_index=-100,
        image_token_index=32000,
        projector_hidden_act="gelu",
        vision_feature_select_strategy="default",
        vision_feature_layer=-2,
        **kwargs,
    ):
        super().__init__(vision_config, text_config, perceiver_config, **kwargs)
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act

        if vision_feature_select_strategy not in ["default", "full"]:
            raise ValueError(
                "vision_feature_select_strategy should be one of 'default', 'full'."
                f"Got: {vision_feature_select_strategy}"
            )

        if "vocab_size" in kwargs:
            warnings.warn(
                "The `vocab_size` argument is deprecated and will be removed in v4.42, \
                since it can be inferred from the `text_config`. \
                Passing this argument has no effect",
                FutureWarning,
            )

        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer
        self._vocab_size = self.text_config.vocab_size

    @property
    def vocab_size(self):
        warnings.warn(
            "The `vocab_size` attribute is deprecated and will be removed in v4.42, \
            Please use `text_config.vocab_size` instead.",
            FutureWarning,
        )
        return self._vocab_size

    @vocab_size.setter
    def vocab_size(self, value):
        self._vocab_size = value

    def to_dict(self):
        output = super().to_dict()
        output.pop("_vocab_size", None)
        return output


class MultiModalLLm(ABC, FlashForCausalLM):
    def __init__(self, config, weights, **kwargs):
        if getattr(config, "text_config"):
            if not config.quantize:
                setattr(config.text_config, 'quantize', None)
            else:
                setattr(config.text_config, 'quantize', config.quantize)
            setattr(config.text_config, 'quantization_config', QuantizationConfig(**{}))
            super().__init__(config.text_config, weights, **kwargs)
        else:
            super().__init__(config, weights, **kwargs)
        self.config = config
        self.weights = weights
        self.w_list = list(weights.routing.keys())
        self.vocab_size = config.text_config.vocab_size
        self.vision_tower = None
        self.language_model = None
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.init_vit()
        self.init_llm()
        self.model_type = None


    def init_visiontowerweight(self, module, weights):
        vision_weights = [vision_weight for vision_weight in module.state_dict().keys()]
        for vision_weight in vision_weights:
            pop_index = self.w_list.index(f"model.vision_model.{vision_weight}")
            self.w_list.pop(pop_index)
            saved_weight = torch.nn.Parameter(
                    weights.get_tensor(f"model.vision_model.{vision_weight}"),
                    requires_grad=False
                )
            vision_weight_list = vision_weight.split(".")
            target_module = module
            for nxt_module in vision_weight_list[:-1]:
                target_module = getattr(target_module, nxt_module)
            setattr(target_module, vision_weight_list[-1], saved_weight)

    def init_vit(self):
        self.vision_tower = Idefics2VisionTransformer(self.config.vision_config)
        self.init_visiontowerweight(self.vision_tower, self.weights)

    def init_llm(self):
        self.model_type = self.config.text_config.model_type
        if self.model_type in [MISTRAL, VICUNA, LLAMA]:
            self.model_type = LLAMA
        model_cls = get_llm_model(self.model_type)
        self.language_model = model_cls(self.config.text_config,
                                  self.weights,
                                  "lm_head",
                                  "model.text_model")
        self.language_model.skip_word_embedding = True

    @abstractmethod
    def prepare_prefill_token(self, text, image, video, processor, batch_size):
        raise NotImplementedError

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


class FlashIdefics2ForCausalLM(MultiModalLLm):
    def __init__(self, config, weights, **kwargs):
        super().__init__(config, weights, **kwargs)
        self.config = config
        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.multi_modal_projector = None
        self.init_multimodal()

    @staticmethod
    def init_multi_modal(module, weights):
        multimodel_weights = [multimodel_weight for multimodel_weight in module.state_dict().keys()]
        for multimodel_weight in multimodel_weights:
            saved_weight = torch.nn.Parameter(
                    weights.get_tensor(f"model.connector.{multimodel_weight}"),
                    requires_grad=False
                )
            multimodel_weight_list = multimodel_weight.split(".")
            target_module = module
            for nxt_module in multimodel_weight_list[:-1]:
                target_module = getattr(target_module, nxt_module)
            setattr(target_module, multimodel_weight_list[-1], saved_weight)

    def init_multimodal(self):
        self.connector = Idefics2Connector(self.config)
        self.init_multi_modal(self.connector, self.weights)


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
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def prepare_inputs_for_generation(
        self, input_ids, attention_mask, pixel_values, pixel_attention_mask, past_key_values=None
    ):

        position_ids = None
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        model_inputs = {"input_ids": input_ids}
        image_hidden_states = None

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": True,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "pixel_attention_mask": pixel_attention_mask,
                "image_hidden_states": image_hidden_states,
            }
        )
        return model_inputs

    def inputs_merger(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: Optional[torch.Tensor],
        image_hidden_states: Optional[torch.Tensor],
    ):
        _, _, vision_hidden_size = image_hidden_states.shape
        special_image_token_mask = input_ids == self.config.image_token_id
        new_inputs_embeds = inputs_embeds.clone()
        reshaped_image_hidden_states = image_hidden_states.view(-1, vision_hidden_size)
        new_inputs_embeds[special_image_token_mask] = reshaped_image_hidden_states
        return new_inputs_embeds

    def prepare_prefill_token(self, text, image, video, processor, batch_size):
        with PIL.Image.open(image) as img:
            image = PIL.ImageOps.exif_transpose(img)
            image = image.convert("RGB")
        messages = [
            {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What do we see in this image?"},
                ]
            },
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt")
        inputs = {k: v for k, v in inputs.items()}

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs["pixel_values"]
        pixel_attention_mask = inputs["pixel_attention_mask"]
        input_ids = input_ids.to(torch.int64).npu()
        attention_mask = attention_mask.to(torch.int64).npu()
        pixel_values = pixel_values.to(torch.float32).npu()
        pixel_attention_mask = pixel_attention_mask.to(torch.int64).npu()

        use_cache = True

        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values = None
        if use_cache:
            if not isinstance(past_key_values, Cache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            batch_size, num_images, num_channels, height, width = pixel_values.shape
            pixel_values = pixel_values.to(dtype=self.dtype)  # fp16 compatibility
            pixel_values = pixel_values.view(batch_size * num_images, *pixel_values.shape[2:])

            # Remove padding images - padding images are full 0.
            nb_values_per_image = pixel_values.shape[1:].numel()
            real_images_inds = (pixel_values == 0.0).sum(dim=(-1, -2, -3)) != nb_values_per_image
            pixel_values = pixel_values[real_images_inds].contiguous()

            # Handle the vision attention mask
            if pixel_attention_mask is None:
                pixel_attention_mask = torch.ones(
                    size=(pixel_values.size(0), pixel_values.size(2), pixel_values.size(3)),
                    dtype=torch.bool,
                    device=pixel_values.device,
                )
            else:
                # Remove padding images from the mask/pP p
                pixel_attention_mask = pixel_attention_mask.view(
                    batch_size * num_images, *pixel_attention_mask.shape[2:]
                )
                pixel_attention_mask = pixel_attention_mask[real_images_inds].contiguous()

            patch_size = self.config.vision_config.patch_size
            patches_subgrid = pixel_attention_mask.unfold(dimension=1, size=patch_size, step=patch_size)
            patches_subgrid = patches_subgrid.unfold(dimension=2, size=patch_size, step=patch_size)
            patch_attention_mask = (patches_subgrid.sum(dim=(-1, -2)) > 0).bool()

            # Get sequence from the vision encoder
            image_hidden_states = self.vision_tower(
                pixel_values=pixel_values,
                patch_attention_mask=patch_attention_mask,
            ).last_hidden_state

            # Modality projection & resampling
            image_hidden_states = self.connector(
                image_hidden_states, attention_mask=patch_attention_mask.view(pixel_values.size(0), -1)
            )
            inputs_embeds = self.inputs_merger(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_hidden_states=image_hidden_states,
            )

        inputs_embeds = inputs_embeds.view(inputs_embeds.shape[0] * inputs_embeds.shape[1],
                                            inputs_embeds.shape[2])

        return inputs_embeds

    def init_ascend_operations(self, config: PretrainedConfig):
        pass

    def init_ascend_weight(self):
        pass

    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids, image_token_id):
        mask = (input_ids == image_token_id)
        inputs_embeds[mask] = image_features
        return inputs_embeds
