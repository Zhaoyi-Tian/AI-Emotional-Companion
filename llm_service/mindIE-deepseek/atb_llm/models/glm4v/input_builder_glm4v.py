# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

from typing import Dict, List
import os
import numpy as np
import torch

from atb_llm.utils.shm_utils import create_shm, encode_shm_name_to_int64, encode_shape_to_int64
from ..base.input_builder import InputBuilder
from ...utils.multimodal_utils import safe_open_image

_PREFIX_LEN = 7
_TEXT_BOI_POS = 4
_TEXT_EOI_POS = 6
_PAD_TOKEN_ID = 151329
_BOI_TOKEN_ID = 151339
_EOI_TOKEN_ID = 151340
_IMG_TOKEN_LEN = 1600
_CONTENT = "content"
_IMAGE = "image"
_TEXT = "text"


class Glm4vInputBuilder(InputBuilder):
    def __init__(self, tokenizer, config, **kwargs):
        self.tokenizer = tokenizer
        self.config = config
        super().__init__(tokenizer, **kwargs)
    
    def generate_position_ids(self, input_ids):
        if self.config.boi_token_id not in input_ids:
            return range(len(input_ids))
        
        eoi_pos = np.where(np.equal(input_ids, self.config.eoi_token_id))[0][0]
        text_input_ids = input_ids[eoi_pos + 1:]
        image_size: int = self.config.vision_config['image_size']
        patch_size: int = self.config.vision_config['patch_size']
        num_patches = (image_size // patch_size // 2) ** 2
        position_ids = np.arange(len(text_input_ids) + _PREFIX_LEN, dtype=np.int32)
        new_position_ids = []
        new_position_ids.append(np.concatenate(
            (position_ids[:_TEXT_BOI_POS + 1], position_ids[_TEXT_BOI_POS + 1].repeat(num_patches),
             position_ids[_TEXT_EOI_POS:])
        ))
        new_position_ids = np.concatenate(new_position_ids)
        return new_position_ids

    def make_context(
        self,
        rank: int,
        conversation: List[Dict[str, List[Dict]]],
        **kwargs
    ):
        if isinstance(conversation[0][_CONTENT], str):
            for item in conversation:
                item[_CONTENT] = [{"text": item[_CONTENT]}]
        elif not isinstance(conversation[0][_CONTENT], list):
            raise TypeError("The conversation \"content\" should be a List[Dict] or str.")
        
        pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else _PAD_TOKEN_ID
        boi_token_id = self.config.boi_token_id if self.config.boi_token_id is not None else _BOI_TOKEN_ID
        eoi_token_id = self.config.eoi_token_id if self.config.eoi_token_id is not None else _EOI_TOKEN_ID

        # build conversation input for glm tokenizer
        from PIL import Image
        processed_conversation = []
        for item in conversation:
            role = item["role"]
            content = item[_CONTENT]
            image_path, image_pix = None, None
            text = None
            for single_input in content:
                if single_input.get(_IMAGE, None):
                    image_path = single_input[_IMAGE]
                    image_pix = safe_open_image(Image, image_path).convert("RGB")
                elif single_input.get(_TEXT, None):
                    text = single_input[_TEXT]
            processed_conversation.append({"role": role, _IMAGE: image_pix, _CONTENT: text})
        
        # image preprocess
        inputs = self.tokenizer.apply_chat_template(processed_conversation, add_generation_prompt=True,
                                                    tokenize=True, return_tensors="pt", return_dict=True)
        input_ids = inputs["input_ids"].flatten()

        if "images" in inputs:
            # Case 1: inputs with texts and images
            input_image = inputs["images"]

            # save image to shared memory
            shm_name_save_path = kwargs.get('shm_name_save_path', None)
            if shm_name_save_path is None:
                shm_name_save_dir = os.path.dirname(os.path.dirname(image_path))
                shm_name_save_path = os.path.join(shm_name_save_dir, "shm_name.txt")
            shm = create_shm(input_image.nbytes, shm_name_save_path)
            shared_array = np.ndarray(input_image.shape, dtype=np.float32, buffer=shm.buf)
            shared_array[:] = input_image
            # encode shared memory name and image's shape
            shm_name = encode_shm_name_to_int64(shm.name)
            shape_value = encode_shape_to_int64(input_image.shape)

            # prepare input_ids with encoded shm info
            img_padding = torch.full((_IMG_TOKEN_LEN, ), pad_token_id, dtype=input_ids.dtype)
            new_input_ids = []
            boi_pos = torch.where(torch.eq(input_ids, boi_token_id))[0]
            eoi_pos = torch.where(torch.eq(input_ids, eoi_token_id))[0]
            new_input_ids.append(torch.cat((input_ids[:boi_pos + 1], img_padding, input_ids[eoi_pos:])))
            new_input_ids = torch.cat(new_input_ids)
            new_input_ids[boi_pos + 1] = shm_name
            new_input_ids[boi_pos + 2] = shape_value
            return new_input_ids
        else:
            # Case 2: inputs with only texts
            return input_ids
