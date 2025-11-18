# Copyright (c) Alibaba Cloud.
# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the same directory of this file. Changes have
# been made to fit Ascend devices, satisfy Huawei clean code
# regulations and speed up inference.

import os
import torch
from PIL import Image
from torchvision import transforms

from atb_llm.utils.log import logger, print_log
from atb_llm.utils.file_utils import standardize_path
from atb_llm.models.cogvlm2.config_cogvlm2 import Cogvlm2Config

_IMAGE_SIZE = Cogvlm2Config.image_size


def _supported_image_format(abs_path):
    suffix = os.path.splitext(abs_path)
    return len(suffix) > 0 and suffix[-1] in [".jpg", ".jpeg", ".png"]


def cogvlm2_image_preprocess(image_path):
    supported_image_mode = "RGB"
    image_path = standardize_path(image_path)

    if _supported_image_format(image_path):
        image = Image.open(image_path)
    else:
        logger.warning(
            "Invalid image path or format (we support png, jpg, jpeg), use white canvas instead."
        )
        image = Image.new(supported_image_mode, (1344, 1344), (255, 255, 255))

    if image.mode != supported_image_mode:
        image = image.convert(supported_image_mode)

    if image is not None:
        # vision
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (_IMAGE_SIZE, _IMAGE_SIZE),
                    interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711)
                ),
            ]
        )
        image_pixel = torch.tensor([transform(image).numpy().tolist()])
    else:
        print_log(logger.error, "error catched: image is None")
        image_pixel = torch.tensor([])
    image.close()
    return image_pixel