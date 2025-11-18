# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import re
import torch.nn as nn
from transformers import SiglipVisionModel as SiglipVisionModelOri
from atb_llm.models.bunny.minicpm.configuration_bunny import SigLipVisionConfig

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200


class SigLipVisionModel(SiglipVisionModelOri):
    config_class = SigLipVisionConfig
    main_input_name = "pixel_values"
    _no_split_modules = ["SigLipEncoderLayer"]

    def __init__(self, config: SigLipVisionConfig):
        super().__init__(config)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'mlp2x_gelu')

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)    

    mlp_depth = int(mlp_gelu_match.group(1))
    modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
    for _ in range(1, mlp_depth):
        modules.append(nn.GELU())
        modules.append(nn.Linear(config.hidden_size, config.hidden_size))
    return nn.Sequential(*modules)