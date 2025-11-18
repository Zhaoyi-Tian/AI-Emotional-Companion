# Copyright (c) Alibaba Cloud.
# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

import os
import torch
import numpy as np
import av

from torchvision import transforms
from torchvision.transforms import Lambda
from torchvision.transforms._transforms_video import NormalizeVideo, CenterCropVideo
from pytorchvideo.transforms import ShortSideScale

from atb_llm.utils.log import logger, print_log
from atb_llm.utils.file_utils import standardize_path
from atb_llm.models.cogvlm2_llama3_video.config_cogvlm2_llama3_video import Cogvlm2Llama3VideoConfig

_IMAGE_SIZE = Cogvlm2Llama3VideoConfig.image_size
# Maximum number of images. One frame is obtained per second.
_MAX_NUM_FRAMES = 24


def _supported_video_format(abs_path):
    suffix = os.path.splitext(abs_path)
    return len(suffix) > 0 and suffix[-1] in [".mp4"]


def read_video_pyav(container, indices):
    container.seek(0)
    video_frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if i > indices[-1]:
            break
        if i >= indices[0] and i in indices:
            video_frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in video_frames])


def cogvlm2_video_preprocess(image_path):
    image_path = standardize_path(image_path)

    if _supported_video_format(image_path):
        container = av.open(image_path)
        fps = container.streams.video[0].average_rate
        fps = round(fps, 0)
        total_frames = container.streams.video[0].frames

        max_len_frames = min((_MAX_NUM_FRAMES) * fps, total_frames)
        indices = np.arange(0, max_len_frames, fps).astype(int)

        clip = read_video_pyav(container, indices)
        image = clip.transpose(3, 0, 1, 2)
        container.close()
    else:
        logger.warning(
            "Invalid video path or format (we support mp4)."
        )
        image = None

    if image is not None:
        # vision
        transform = transforms.Compose(
            [
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
                ShortSideScale(size=_IMAGE_SIZE),
                CenterCropVideo(_IMAGE_SIZE),
            ]
        )
        # T C H W
        image_pixel = transform(torch.tensor(image)).transpose(0, 1)
    else:
        print_log(int(os.getenv("RANK", "0")), logger.error, "error catched: video is None")
        image_pixel = torch.tensor([])
    return image_pixel
