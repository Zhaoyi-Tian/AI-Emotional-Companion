# coding=utf-8
# Copyright (c) The InternLM team and The HuggingFace Inc. team. All rights reserved.
# --------------------------------------------------------
# Copyright (c) 2024 OpenGVLab
# --------------------------------------------------------
# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.

import os
import math
import numpy as np

import yaml
from PIL import Image

import torch
from transformers import AutoTokenizer, CLIPImageProcessor

from atb_llm.models.base.input_builder import InputBuilder
from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.utils.shm_utils import encode_shm_name_to_int64, encode_shape_to_int64, create_shm
from .conversation import conv_templates
from .modeling_vita_audio import AudioEncoderProcessor
from .tool import tokenizer_image_audio_token, tokenizer_image_token, dynamic_preprocess, get_rawvideo_dec, VideoConfig


IMAGE_TOKEN_INDEX = -200
AUDIO_TOKEN_INDEX = -500
IMAGE_TAG = -1
VIDEO_TAG = -2
IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_AUDIO_TOKEN = "<audio>"
MAX_IMAGE_LENGTH = 16


class VitaInputBuilder(InputBuilder):
    def __init__(self, tokenizer, config, system_role_name="assistant", user_role_name="user", **kwargs):
        super().__init__(tokenizer, system_role_name, user_role_name)
        self.config = config
        self.tokenizer = tokenizer
        self.image_token_id = IMAGE_TOKEN_INDEX
        self.pad_token_id = IGNORE_INDEX
        
    def make_context(self, rank, conversation, **kwargs):
        if not isinstance(conversation[0]["content"], list):
            raise ValueError("The conversation \"content\" should be a List[Dict].")
        shm_name_save_path = kwargs.get('shm_name_save_path', None)
        context_tokens = self._apply_chat_template(
            conversation,
            shm_name_save_path=shm_name_save_path,
            )
        return context_tokens
            
    def get_audio_processor(self):
        audio_path = os.path.join(self.config.model_name_or_path, self.config.mm_audio_encoder)
        with open(os.path.join(audio_path, "train.yaml"), "r") as fin:
            audio_configs = yaml.safe_load(fin)
        audio_processor = AudioEncoderProcessor(dataset_conf=audio_configs["dataset_conf"])
        return audio_processor
    
    def process_images(self, images, model_cfg, image_processor):
        image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
        
        new_images = []
        if image_aspect_ratio == "pad":
            for image in images:
                image = self.expand2square(
                    image, tuple(int(x * 255) for x in image_processor.image_mean)
                )
                image = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
                new_images.append(image)
        else:
            return image_processor(images, return_tensors="pt")["pixel_values"]
        if all(x.shape == new_images[0].shape for x in new_images):
            new_images = torch.stack(new_images, dim=0)
        return new_images

    def _apply_chat_template(self, conversation, shm_name_save_path=None, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path, use_fast=True)
        image_processor = CLIPImageProcessor.from_pretrained(
            f"{self.config.model_name_or_path}/{self.config.mm_vision_tower}")
        image_token_id = IMAGE_TOKEN_INDEX
        pad_token_id = IGNORE_INDEX
        max_audio_id = 0
        shm_name_save_path = shm_name_save_path
        qs = ""
        if self.config.model_type == 'vita-Qwen2':
            conv_mode = "qwen2p5_instruct"
        else:
            conv_mode = "mixtral_two"
        conv = conv_templates[conv_mode].copy()
        media_tensor = []
        flag = False
        modality = "lang"
        content = "content"
        tex = "text"
        img = "image"
        vid = "video"
        for message in conversation:
            role = message.get("role")
            if isinstance(conversation[0][content], str):
                for item in conversation:
                    item[content] = [{tex: item[content]}]
            for single_input in message[content]:
                if tex in single_input:
                    text = single_input.get(tex)
                    new_q = qs + text
                    qs = ''
                    conv.append_message(conv.roles[0], new_q) if role == 'user' \
                        else conv.append_message(conv.roles[1], new_q)
                    continue
                elif img in single_input:
                    image_path = single_input.get(img)
                    image = Image.open(image_path).convert("RGB")
                    image, p_num = dynamic_preprocess(
                            image, min_num=1, max_num=12, image_size=448, use_thumbnail=True
                        )
                    image_tensor = self.process_images(image, 
                                                       self.config, 
                                                       image_processor).half()
                    qs += DEFAULT_IMAGE_TOKEN * p_num[0] + "\n"
                    modality = img
                    maxlen_image_id = p_num[0] * 255
                    media_tensor.append((image_path, image_tensor, maxlen_image_id))
                elif vid in single_input:
                    video = single_input.get(vid)
                    video_framerate = 1
                    video_frames, slice_len = get_rawvideo_dec( 
                        VideoConfig(
                            video_path=video,
                            image_processor=image_processor,
                            max_frames=MAX_IMAGE_LENGTH, 
                            video_framerate=video_framerate, 
                            image_aspect_ratio=getattr(self.config, "image_aspect_ratio", None)),
                        )
                    image_tensor = video_frames.half()
                    qs += DEFAULT_IMAGE_TOKEN * slice_len + "\n"
                    modality = vid
                    maxlen_image_id = slice_len * 255
                    media_tensor.append((video, image_tensor, maxlen_image_id))
                elif "audio" in single_input:
                    audio = single_input.get("audio")
                    new_q = qs + DEFAULT_AUDIO_TOKEN
                    qs = ''
                    conv.append_message(conv.roles[0], new_q) if role == 'user' \
                        else conv.append_message(conv.roles[1], new_q)
                    audio_process = self.get_audio_processor()
                    audio_input, _ = audio_process.process(os.path.join(audio))
                    audio_length = audio_input.shape[0]
                    audio_input = torch.unsqueeze(audio_input, dim=0).half()
                    audio_length = torch.unsqueeze(torch.tensor(audio_length), dim=0)
                    max_audio_id = math.ceil((((audio_length - 3) // 2 - 2) // 2 + 1) / 2) - 1
                    media_tensor.append((audio, audio_input, max_audio_id))
                    flag = True
                else:
                    error_msg = "Unsupport element: " + str(single_input)
                    logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
                    raise KeyError(error_msg)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt(modality)
        
        if flag:
            input_ids = (
                tokenizer_image_audio_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
            ).flatten()
        else:
            input_ids = (
                tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
            ).flatten()
        new_input_ids = []
        start = 0
        image_pos = torch.where(torch.eq(input_ids, image_token_id))[0]
        audio_pos = torch.where(torch.eq(input_ids, AUDIO_TOKEN_INDEX))[0]
        image_final_pos = []
        audio_final_pos = []
        for idx in image_pos:
            if input_ids[idx + 1] != image_token_id:
                image_final_pos.append(idx)
        for idx in audio_pos:
            if input_ids[idx + 1] != AUDIO_TOKEN_INDEX:
                audio_final_pos.append(idx)
        image_final_pos.extend(audio_final_pos)
        image_final_pos.sort()
        media_pos = image_final_pos
        
        if len(media_tensor) != 0:
            for i, (path, tensor, length) in enumerate(media_tensor):
                shm_name_save_dir = os.path.dirname(os.path.dirname(path))
                shm_name_save_path = os.path.join(shm_name_save_dir, "shm_name.txt")
                tensor = tensor.numpy()
                shm = create_shm(tensor.nbytes, shm_name_save_path)
                shared_array = np.ndarray(tensor.shape, dtype=tensor.dtype, buffer=shm.buf)
                shared_array[:] = tensor
                shm_name = encode_shm_name_to_int64(shm.name)  
                shape_value = encode_shape_to_int64(tensor.shape)
                intermediate_token = torch.tensor([shm_name, shape_value])
                if i == 0:
                    new_input_ids = torch.cat([
                        input_ids[start : media_pos[i] + 1],
                        intermediate_token,
                        torch.full((length - 2, ), pad_token_id, dtype=input_ids.dtype), 
                    ])
                else: 
                    new_input_ids = torch.cat([
                        new_input_ids,
                        input_ids[start : media_pos[i] + 1],
                        intermediate_token,
                        torch.full((length - 2, ), pad_token_id, dtype=input_ids.dtype), 
                    ])
                start = media_pos[i] + 1
        if start == 0:
            new_input_ids = input_ids
        else :
            new_input_ids = torch.cat([
                new_input_ids,
                input_ids[start:], 
            ])
        return new_input_ids