# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import os
import re
import math
from dataclasses import dataclass, field
import json
import av
import torch
import numpy as np
from PIL import Image

from atb_llm.utils import multimodal_utils


IMAGE_TOKEN_INDEX = -200
AUDIO_TOKEN_INDEX = -500
MAX_FRAMES = 16
MIN_FRAMES = 4
MAX_IMAGE_NUM = 6
IMAGE_SIZE = 448


@dataclass
class VideoConfig:
    video_path: str
    image_processor: object
    max_frames: int = field(default=MAX_FRAMES)
    min_frames: int = field(default=MIN_FRAMES)
    video_framerate: int = field(default=1)
    image_aspect_ratio: str = field(default="pad")


@dataclass
class ImageConfig:
    min_num: int = 1
    max_num: int = MAX_IMAGE_NUM
    image_size: int = IMAGE_SIZE
    use_thumbnail: bool = False
    img_mean: float = 0.0


def tokenizer_image_audio_token(
    prompt,
    tokenizer,
    image_token_index=IMAGE_TOKEN_INDEX,
    audio_token_index=AUDIO_TOKEN_INDEX,
    return_tensors=None,
):
    prompt_chunks = []
    for chunk in re.split(r"(<audio>|<image>)", prompt):
        if chunk == "<audio>":
            prompt_chunks.append([audio_token_index])
        elif chunk == "<image>":
            prompt_chunks.append([image_token_index])
        else:
            prompt_chunks.append(tokenizer(chunk).input_ids)

    input_ids = []
    offset = 0
    if (
        len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in prompt_chunks:
        if x != [image_token_index] and x != [audio_token_index]:
            input_ids.extend(x[offset:])
        else:
            input_ids.extend(x[:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


def tokenizer_image_token(
    prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None
):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    def insert_separator(x, sep):
        return [ele for sublist in zip(x, [sep] * len(x)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if (
        len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


def load_cmvn_json(json_cmvn_file):
    with open(json_cmvn_file) as f:
        cmvn_json = json.load(f)

    avg = cmvn_json["mean_stat"]
    var = cmvn_json["var_stat"]
    count = cmvn_json["frame_num"]
    for i, a in enumerate(avg):
        a /= count
        var[i] = var[i] / count - a * a
        if var[i] < 1.0e-20:
            var[i] = 1.0e-20
        var[i] = 1.0 / math.sqrt(var[i])
    cmvn = [avg, var]
    return cmvn


def dynamic_preprocess_with_mean(image, config:ImageConfig):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set()
    for n in range(config.min_num, config.max_num + 1):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if i * j <= config.max_num and i * j >= config.min_num:
                    target_ratios.add((i, j))
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, config.image_size
    )

    # calculate the target width and height
    target_width = config.image_size * target_aspect_ratio[0]
    target_height = config.image_size * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))

    # expand target_aspect_ratio to even for each size
    new_target_aspect_ratio = [e if e % 2 == 0 else e + 1 for e in target_aspect_ratio]
    blocks_big = int(0.5 * new_target_aspect_ratio[0] * 0.5 * new_target_aspect_ratio[1])

    # padding to even patch for each size
    new_target_width = new_target_aspect_ratio[0] * config.image_size
    new_target_height = new_target_aspect_ratio[1] * config.image_size
    resized_img = expand2even(
        resized_img, new_target_width, new_target_height, tuple(int(x * 255) for x in config.img_mean)
    )

    processed_images = []
    image_size_big = config.image_size * 2
    for i in range(blocks_big):
        box = (
            (i % (new_target_width // image_size_big)) * image_size_big,
            (i // (new_target_width // image_size_big)) * image_size_big,
            ((i % (new_target_width // image_size_big)) + 1) * image_size_big,
            ((i // (new_target_width // image_size_big)) + 1) * image_size_big,
        )
        # split the image
        split_img_big = resized_img.crop(box)
        split_img = split_img_big.resize((config.image_size, config.image_size))
        processed_images.append(split_img)
        blocks_small = 2 * 2
        for j in range(blocks_small):
            box = (
                (j % (image_size_big // config.image_size)) * config.image_size,
                (j // (image_size_big // config.image_size)) * config.image_size,
                ((j % (image_size_big // config.image_size)) + 1) * config.image_size,
                ((j // (image_size_big // config.image_size)) + 1) * config.image_size,
            )
            # split the image
            split_img = split_img_big.crop(box)
            processed_images.append(split_img)

    if config.use_thumbnail:
        thumbnail_img = resized_img.resize((config.image_size, config.image_size))
        processed_images += [thumbnail_img] * 5

    return processed_images, [len(processed_images) // 5]


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set()
    for n in range(min_num, max_num + 1):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if i * j <= max_num and i * j >= min_num:
                    target_ratios.add((i, j))
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images, [len(processed_images)]


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def expand2even(pil_img, new_target_width, new_target_height, background_color):
    result = Image.new(pil_img.mode, (new_target_width, new_target_height), background_color)
    result.paste(pil_img, (0, 0))
    return result


def get_rawvideo_dec(config: VideoConfig):
    if os.path.exists(config.video_path):
        container = multimodal_utils.safe_load_multimodal_source(av.open, config.video_path)
    else:
        raise FileNotFoundError

    fps = container.streams.video[0].average_rate
    f_start = 0
    f_end = int(min(1000000000, container.streams.video[0].frames - 1))
    num_frames = f_end - f_start + 1
    if num_frames > 0:
        sample_fps = int(config.video_framerate)
        t_stride = int(round(float(fps) / sample_fps))

        all_pos = list(range(f_start, f_end + 1, t_stride))
        if len(all_pos) > config.max_frames:
            sample_pos = [
                all_pos[_] 
                for _ in np.linspace(0, len(all_pos) - 1, num=config.max_frames, dtype=int)
            ]
        elif len(all_pos) < config.min_frames:
            sample_pos = [
                all_pos[_] 
                for _ in np.linspace(0, len(all_pos) - 1, num=config.min_frames, dtype=int)
            ]
        else:
            sample_pos = all_pos
        
        patch_images = []
        for pos in sample_pos:
            timestamp = int(pos * (1 / fps) * 1_000_000)  # 转换为微秒
            container.seek(timestamp)  # 根据帧位置计算时间戳并跳转
            for frame in container.decode(video=0):
                img = frame.to_ndarray()
                patch_images.append(Image.fromarray(img))
                break  # 只取当前帧
        if config.image_aspect_ratio == "pad":

            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            patch_images = [
                expand2square(Image.fromarray(i), tuple(int(x * 255) for x in config.image_processor.image_mean))
                for i in patch_images
            ]
            patch_images = [
                config.image_processor.preprocess(i, return_tensors="pt")["pixel_values"][0]
                for i in patch_images
            ]
        else:
            patch_images = [
                config.image_processor.preprocess(i, return_tensors="pt")["pixel_values"][0]
                for i in patch_images
            ]

        patch_images = torch.stack(patch_images)
        slice_len = patch_images.shape[0]

        return patch_images, slice_len
    else:
        raise ValueError(f"No frames found in {config.video_path}")