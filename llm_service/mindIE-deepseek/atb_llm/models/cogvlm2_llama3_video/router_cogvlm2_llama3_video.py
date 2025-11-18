# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.

from dataclasses import dataclass
from typing import Dict, List
import os

import torch
import numpy as np

from atb_llm.models.cogvlm2.router_cogvlm2 import Cogvlm2Router
from atb_llm.models.cogvlm2_llama3_video.config_cogvlm2_llama3_video import Cogvlm2Llama3VideoConfig
from atb_llm.utils.shm_utils import encode_shm_name_to_int64, encode_shape_to_int64, create_shm
from .input_builder_cogvlm2_llama3_video import Cogvlm2VideoInputBuilder
from .data_preprocess_cogvlm2_llama3_video import cogvlm2_video_preprocess

_PAD_TOKEN_ID = Cogvlm2Llama3VideoConfig.pad_token_id
_BOS_TOKEN_ID = Cogvlm2Llama3VideoConfig.bos_token_id
_EOS_TOKEN_ID = Cogvlm2Llama3VideoConfig.eos_token_id
_IMG_TOKEN_LEN = Cogvlm2Llama3VideoConfig.img_token_len


@dataclass
class Cogvlm2llama3videoRouter(Cogvlm2Router):

    def tokenize(self, inputs: List[Dict], **kwargs):
        pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else _PAD_TOKEN_ID
        bos_token_id = self.config.bos_token_id if self.config.bos_token_id is not None else _BOS_TOKEN_ID

        image_path = ""
        text = ""
        for ele in inputs:
            if "image_or_video" in ele:
                image_path = ele["image_or_video"]
            elif "video" in ele:
                image_path = ele["video"]
            elif "image" in ele:
                image_path = ele["image"]
            elif "text" in ele:
                text = ele["text"]
        if text == "":
            raise ValueError("text is empty.")
        new_input_ids = []
        input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"].flatten()[1:]
        if image_path == "":
            new_input_ids = [torch.tensor([bos_token_id], dtype=input_ids.dtype), input_ids]
            new_input_ids = torch.cat(new_input_ids)
        else:
            shm_name_save_path = kwargs.get('shm_name_save_path', None)
            shm_name_list = []
            shape_value_list = []
            image_pixel = cogvlm2_video_preprocess(image_path=image_path)
            num_eois = image_pixel.size(0)

            if shm_name_save_path is None:
                shm_name_save_dir = os.path.dirname(os.path.dirname(image_path))
                shm_name_save_path = os.path.join(shm_name_save_dir, "shm_name.txt")
            shm = create_shm(image_pixel.nbytes, shm_name_save_path)
            shared_array = np.ndarray(image_pixel.shape, dtype=np.float32, buffer=shm.buf)
            shared_array[:] = image_pixel

            shm_name = encode_shm_name_to_int64(shm.name)
            shape_value = encode_shape_to_int64(image_pixel.shape)
            shm_name_list.append(shm_name)
            shape_value_list.append(shape_value)
            
            video_token = []
            for _time_idx in range(num_eois):
                video_token += torch.full((_IMG_TOKEN_LEN, 1), pad_token_id, dtype=input_ids.dtype)
                time_indices = self.tokenizer(
                    str(_time_idx),
                    return_tensors="pt",
                    add_special_tokens=False
                )["input_ids"]
                video_token += time_indices
            video_token = torch.cat(video_token)

            new_input_ids = [torch.tensor([bos_token_id], dtype=input_ids.dtype), video_token, input_ids]
            new_input_ids = torch.cat(new_input_ids)
            pad_pos = torch.where(torch.eq(new_input_ids, pad_token_id))[0]

            new_input_ids[pad_pos[0] + 1] = shm_name_list[0]
            new_input_ids[pad_pos[0] + 2] = shape_value_list[0]
        return new_input_ids

    def get_config(self):
        config = Cogvlm2Llama3VideoConfig.from_dict(self.config_dict)
        super().check_config(config)
        return config

    def get_input_builder(self):
        pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else _PAD_TOKEN_ID
        return Cogvlm2VideoInputBuilder(self.tokenizer, self.generation_config, pad_token_id)