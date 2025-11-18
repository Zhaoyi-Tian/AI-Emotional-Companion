# Copyright (c) Alibaba Cloud. All Rights Reserved.
# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.
import os
from typing import Dict, List
import numpy as np

import torch

from atb_llm.utils.shm_utils import encode_shm_name_to_int64, encode_shape_to_int64, create_shm
from .config_cogvlm2 import Cogvlm2Config
from ..base.input_builder import InputBuilder
from .data_preprocess_cogvlm2 import cogvlm2_image_preprocess

LANGUAGE_TOKEN_TYPE = 0
VISION_TOKEN_TYPE = 1
_IMAGE_SIZE = Cogvlm2Config.image_size
_PATCH_SIZE = Cogvlm2Config.patch_size
_IMG_TOKEN_LEN = Cogvlm2Config.img_token_len
IMAGE_TOKEN_ID = 128002


def _history_to_prompt(signal_type, history, query):
    if signal_type == 'base':
        return query
    elif signal_type == 'vqa':
        answer_format = 'Short answer:'
    elif signal_type == 'chat':
        answer_format = 'Answer:'
    else:
        raise ValueError(f"Unknown signal type {signal_type}")

    prompt = ''
    for _, (old_query, response) in enumerate(history):
        prompt += 'Question: ' + old_query + " {} ".format(answer_format) + response + "\n"
    prompt += 'Question: {} {}'.format(query, answer_format)
    return prompt


def get_history_text(conversation, idx):
    text = ""
    for content in conversation[idx]["content"]:
        if "text" in content.keys():
            text = content["text"]
    return text


def _conversation_to_history(conversation):
    history = []
    # 丢弃conversation的最后一个query，组成history
    for i in range(len(conversation) // 2):
        history.append((get_history_text(conversation, 2 * i), get_history_text(conversation, 2 * i + 1)))
    return history


class Cogvlm2InputBuilder(InputBuilder):
    def __init__(self, tokenizer, generation_config, pad_token_id, **kwargs):
        self.config = generation_config
        self.pad_token_id = pad_token_id
        super().__init__(tokenizer, system_role_name="assistant", user_role_name="user", **kwargs)

    def generate_position_ids(self, input_ids):
        position_ids = np.arange(len(input_ids), dtype=np.int64)
        if np.any(np.equal(input_ids, IMAGE_TOKEN_ID)):
            pos_ids_list = [0, 1]
            pos_ids_list += [2] * (_IMG_TOKEN_LEN - 2)
            pos_ids_list += [3]
            input_text_len = input_ids.shape[0] - _IMG_TOKEN_LEN - 1
            pos_ids_list += range(4, input_text_len + 4)
            position_ids = np.array(pos_ids_list, dtype=np.int64)
        return position_ids

    def make_context(
            self,
            rank: int,
            conversation: List[Dict[str, List[Dict]]],
            system: str = "You are a helpful assistant.",
            **kwargs):
        shm_name_save_path = kwargs.get('shm_name_save_path', None)
        query = ""
        images = []
        for cont in conversation[-1]["content"]:
            if "text" in cont.keys():
                query = cont["text"]
            elif "image" in cont.keys():
                images += [cont["image"]]
        history = _conversation_to_history(conversation)
        image_size = _IMAGE_SIZE
        patch_size = _PATCH_SIZE
        template_version = "chat"
        if images is not None and len(images) > 1:
            raise ValueError("not support multi images by now.")
        history = history or []
        text = _history_to_prompt(template_version, history, query)
        input_ids = [self.tokenizer.bos_token_id]
        token_type_ids = [LANGUAGE_TOKEN_TYPE]
        shape_value_list = []
        shm_name_list = []
        if images is not None and len(images) == 1:
            image_pixel = cogvlm2_image_preprocess(image_path=images[0])
            # language
            vision_token_num = (image_size // patch_size // 2) * (image_size // patch_size // 2) + 2
            input_ids += [self.pad_token_id] * vision_token_num
            token_type_ids += [VISION_TOKEN_TYPE] * vision_token_num

            if shm_name_save_path is None:
                shm_name_save_dir = os.path.dirname(os.path.dirname(images[0]))
                shm_name_save_path = os.path.join(shm_name_save_dir, "shm_name.txt")
            shm = create_shm(image_pixel.nbytes, shm_name_save_path)
            shared_array = np.ndarray(image_pixel.shape, dtype=np.float32, buffer=shm.buf)
            shared_array[:] = image_pixel

            shm_name = encode_shm_name_to_int64(shm.name)
            shape_value = encode_shape_to_int64(image_pixel.shape)
            shm_name_list.append(shm_name)
            shape_value_list.append(shape_value)
            pad_pos = torch.where(torch.eq(torch.tensor(input_ids), self.pad_token_id))[0]
            input_ids[pad_pos[0] + 1] = shm_name_list[0]
            input_ids[pad_pos[0] + 2] = shape_value_list[0]

        text_ids = self.tokenizer.encode(text, add_special_tokens=False)
        input_ids += text_ids
        input_ids = torch.tensor(input_ids)

        return input_ids
