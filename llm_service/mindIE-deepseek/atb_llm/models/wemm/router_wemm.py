# Copyright Huawei Technologies Co., Ltd. 2024-2028. All rights reserved.
import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image

from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.utils.shm_utils import encode_shm_name_to_int64, encode_shape_to_int64, create_shm
from atb_llm.utils.multimodal_utils import safe_open_image
from atb_llm.models.base.router import BaseRouter
from atb_llm.models.base.model_utils import safe_get_tokenizer_from_pretrained
from atb_llm.models.wemm.data_preprocess_wemm import recover_navit_subimages_with_pos_emb
from .image_processor_2k import Idefics2ImageProcessor

_IMAGE = "image"
_TEXT = "text"

DEFAULT_IMAGE_TOKEN = "<image>"
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_BEGIN_TOKEN = "<img>"
IMAGE_BEGIN_TOKEN_INDEX = -300
DEFAULT_IMAGE_END_TOKEN = "</img>"
IMAGE_END_TOKEN_INDEX = -400
EOS_TOKEN_ID = 92542


def process_shared_memory(pixel_values, shm_name_save_path, data_type):
    shm = create_shm(pixel_values.nbytes, shm_name_save_path)
    shared_array = np.ndarray(pixel_values.shape, dtype=data_type, buffer=shm.buf)
    shared_array[:] = pixel_values
    shm_name = encode_shm_name_to_int64(shm.name)
    shape_value = encode_shape_to_int64(pixel_values.shape)
    return shm_name, shape_value


@dataclass
class WemmRouter(BaseRouter):
    _image_processor: Any = None

    def __post_init__(self):
        super().__post_init__()
        self.tokenizer.eos_token_id = EOS_TOKEN_ID

    @property
    def image_processor(self):
        if not hasattr(self, "_image_processor"):
            self._image_processor = self.get_image_processor()
        elif self._image_processor is None:
            self._image_processor = self.get_image_processor()
        return self._image_processor

    def get_config(self):
        config_cls = self.get_config_cls()
        config = config_cls.from_dict(self.config_dict)
        super().check_config(config)
        return config

    def get_tokenizer(self):
        tokenizer = safe_get_tokenizer_from_pretrained(
            self.tokenizer_path,
            revision=self.revision,
            padding_side="left",
            trust_remote_code=self.trust_remote_code,
            use_fast=True,
        )
        return tokenizer

    def get_image_processor(self):
        return Idefics2ImageProcessor(self.config.image_processor)

    def tokenize(self, inputs, **kwargs):
        text = ""
        image_num = sum(1 for d in inputs if _IMAGE in d)
        shm_name_save_path = kwargs.get("shm_name_save_path", None)

        if image_num > 1:
            logger.error("Input image numbers can not be greater than 1!", ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise KeyError("Input image numbers can not be greater than 1!")

        for single_input in inputs:
            if single_input.get(_TEXT, None):
                text = single_input.get(_TEXT)
                continue
            if single_input.get(_IMAGE, None):
                image_path = single_input[_IMAGE]
                if shm_name_save_path is None:
                    shm_name_save_dir = os.path.dirname(os.path.dirname(image_path))
                    shm_name_save_path = os.path.join(shm_name_save_dir, "shm_name.txt")

                image_obj = None
                image_obj = safe_open_image(Image, image_path)
                if image_obj is None:
                    raise ValueError(f"Unrecognized image path input, only support local path,  got {image_path}")
                image_rgb = image_obj.convert("RGB")
                image_size = self.config.image_processor["size"]
                navit980_images = self.image_processor(
                    [[image_rgb]],
                    size=image_size,
                    return_tensors="pt",
                    do_image_splitting=self.config.do_image_splitting,
                )
                image_obj.close()

                dim = navit980_images["navit_pixel_values"].shape
                patch_size = self.config.vision_config.patch_size
                visual_dim = math.ceil(dim[2] / patch_size) * math.ceil(dim[3] / patch_size)
                clip_visual_outputs_fake = torch.ones(
                    (dim[0], visual_dim, self.config.vision_config.hidden_size), dtype=torch.float16
                )
                super_image_hidden_states, _, _ = recover_navit_subimages_with_pos_emb(
                    clip_visual_outputs_fake,
                    navit980_images["pixel_attention_mask"],
                    num_sub_images=-1,
                    visual_embedding_group=16,
                    pos_hidden_size=4096,
                    thumbnail_only=True,
                )
                img_token_num = math.ceil(super_image_hidden_states.shape[1] / 4) * math.ceil(
                    super_image_hidden_states.shape[2] / 4
                )
                values_shm_name, values_shape_value = process_shared_memory(
                    navit980_images["navit_pixel_values"], shm_name_save_path, np.float32
                )
                mask_shm_name, mask_shape_value = process_shared_memory(
                    navit980_images["pixel_attention_mask"], shm_name_save_path, np.bool8
                )

            else:
                logger.error(
                    "The input field currently only needs to support 'image' and 'text'.",
                    ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE,
                )
                raise TypeError("The input field currently only needs to support 'image' and 'text'.")

        prompt = "<image>" + "\n" + text
        prompt = f"<|im_start|>user\n{prompt}<|im_end|><|im_start|>assistant\n"

        chunk_encode = []
        for idx, chunk in enumerate(prompt.split(DEFAULT_IMAGE_TOKEN)):
            if idx == 0:
                cur_encode = self.tokenizer.encode(chunk)
            else:
                cur_encode = self.tokenizer.encode(chunk, add_special_tokens=False)
            chunk_encode.append(cur_encode)

        if len(chunk_encode) != 2:
            raise ValueError("The length of chunk_encode should be 2")
        
        ids = []
        for idx, cur_chunk_encode in enumerate(chunk_encode):
            ids.extend(cur_chunk_encode)
            if idx != len(chunk_encode) - 1:
                ids.append(IMAGE_BEGIN_TOKEN_INDEX)
                ids.extend([IMAGE_TOKEN_INDEX] * img_token_num)
                ids.append(IMAGE_END_TOKEN_INDEX)

        input_ids = torch.tensor(ids)
        bos_pos = torch.where(torch.eq(input_ids, IMAGE_BEGIN_TOKEN_INDEX))[0]
        if input_ids.size(0) < bos_pos + 5:
            msg = "tokenize error, input_ids length is too short."
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(msg)
        input_ids[bos_pos + 1] = values_shm_name
        input_ids[bos_pos + 2] = values_shape_value
        input_ids[bos_pos + 3] = mask_shm_name
        input_ids[bos_pos + 4] = mask_shape_value

        return input_ids
