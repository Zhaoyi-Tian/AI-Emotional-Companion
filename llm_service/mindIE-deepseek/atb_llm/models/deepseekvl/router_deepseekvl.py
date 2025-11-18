# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
from dataclasses import dataclass
from typing import Dict, List

import torch
import numpy as np
from PIL import Image
from atb_llm.models.deepseekvl.processing_vlm import VLChatProcessor
from atb_llm.models.deepseekvl.processing_vlm import apply_sft_template_for_multi_turn_prompts
from atb_llm.utils.shm_utils import encode_shm_name_to_int64, encode_shape_to_int64, create_shm

from ..base.router import BaseRouter
from ..deepseekvl.flash_causal_deepseekvl import DeepseekvlConfig
from ..llama.config_llama import LlamaConfig
from ..base.config import QuantizationConfig
from ..base.model_utils import safe_get_tokenizer_from_pretrained, safe_from_pretrained
from .input_builder_deepseek_vl import DeepseekVLInputBuilder


_IMAGE_TOKEN_ID = 100015
_PAD_TOKEN_ID = 32001
_IMAGE_FEATURE_WIDTH = 576
_IMAGE_KEY = 'image'
_TEXT_KEY = 'text'


def extract_text_and_image_from_inputs(inputs: List[Dict]):
    text = None
    image = None
    for single_input in inputs:
        if _IMAGE_KEY in single_input:
            image = single_input[_IMAGE_KEY]
        elif _TEXT_KEY in single_input:
            text = single_input[_TEXT_KEY]
        else:
            raise ValueError(f"Unsupported media type found in prompt: {single_input}")
    return text, image


@dataclass
class DeepseekvlRouter(BaseRouter):
    def tokenize(self, inputs, **kwargs):
        text, image = extract_text_and_image_from_inputs(inputs)
        processor = safe_from_pretrained(VLChatProcessor, self.config.model_name_or_path)
        conversation_target = processor.new_chat_template()
        conversation_target.append_message(conversation_target.roles[0], (text, image))
        conversation_target.append_message(conversation_target.roles[1], "")
        prompts = conversation_target.generate_prompt_with_conversation()
        sft_format = apply_sft_template_for_multi_turn_prompts(conversations=prompts,
                                                               sft_format=processor.sft_format,
                                                               system_prompt=processor.system_prompt)
        input_ids = processor.tokenizer.encode(sft_format)
        input_ids = torch.LongTensor(input_ids).flatten()

        bos_pos = torch.where(torch.eq(input_ids, _IMAGE_TOKEN_ID))[0]
        image_num = bos_pos.shape[0]

        shm_name_save_path = kwargs.get('shm_name_save_path', None)

        shm_name_list = []
        shape_value_list = []
        image_type = "image_url"
        for single_input in inputs:
            if _IMAGE_KEY not in single_input.keys():
                continue
            image_pixel = processor.image_processor([Image.open(single_input[_IMAGE_KEY]).convert("RGB")],
                                                    return_tensors="pt").pixel_values
            image_pixel = image_pixel[None, :]
            if shm_name_save_path is None:
                shm_name_save_dir = os.path.dirname(single_input[_IMAGE_KEY])
                shm_name_save_path = os.path.join(shm_name_save_dir, "shm_name.txt")
            shm = create_shm(image_pixel.nbytes, shm_name_save_path)
            shared_array = np.ndarray(image_pixel.shape, dtype=np.float32, buffer=shm.buf)
            shared_array[:] = image_pixel

            shm_name = encode_shm_name_to_int64(shm.name)
            shape_value = encode_shape_to_int64(image_pixel.shape)
            shm_name_list.append(shm_name)
            shape_value_list.append(shape_value)
            pad_tokens = torch.full((processor.num_image_tokens - 1,), _PAD_TOKEN_ID, dtype=input_ids.dtype)
            input_ids = torch.cat((input_ids[:bos_pos + 1], pad_tokens, input_ids[bos_pos + 1:]))

        for i in range(image_num):
            if input_ids.size(0) < bos_pos[i] + 3:
                raise ValueError("tokenize error, input_ids length is too short.")
            input_ids[bos_pos[i] + 1] = shm_name_list[i]
            input_ids[bos_pos[i] + 2] = shape_value_list[i]

        return input_ids

    def check_config_deepseekvl(self, config):
        super().check_config(config)
        attribute_ranges = {
            'mm_hidden_size': (1, 2147483647),
            'num_key_value_heads': (1, 2147483647),
        }
        for attr, (min_val, max_val) in attribute_ranges.items():
            if not hasattr(config, attr) or getattr(config, attr) is None:
                continue
            value = getattr(config, attr)
            if value < min_val or value > max_val:
                raise ValueError(f"self._config.{attr} must be between {min_val} and {max_val}")

    def get_config(self):
        config = safe_from_pretrained(DeepseekvlConfig, self.model_name_or_path)
        setattr(config, 'quantization_config', QuantizationConfig(**{}))
        if self.max_position_embeddings:
            config.max_position_embeddings = self.max_position_embeddings
        self.check_config_deepseekvl(config)
        config.model_name_or_path = self.model_name_or_path
        return config

    def get_tokenizer(self):
        use_fast = True
        return safe_get_tokenizer_from_pretrained(
            self.tokenizer_path,
            revision=self.revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=self.trust_remote_code,
            use_fast=use_fast
        )
    
    def get_input_builder(self):
        return DeepseekVLInputBuilder(self.tokenizer, self.model_version, self.generation_config, None,
            model_name_or_path=self.config.model_name_or_path)
