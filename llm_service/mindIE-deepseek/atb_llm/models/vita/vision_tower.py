# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

import os
import torch
import torch.nn as nn
from transformers import AutoConfig, CLIPImageProcessor

from atb_llm.models.vita.modeling_vita_vit import InternVisionModel


class InternViTVisionTower(nn.Module):
    def __init__(self, model_name_or_path, vision_tower_config, process_group, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self._config = vision_tower_config 
        self.vision_tower_name = os.path.basename(vision_tower_config.mm_vision_tower)
        self.process_group = process_group
        self.select_layer = -1
        self.scale_pix_shuffle = 0.5
        self.model_name_or_path = model_name_or_path

        if not delay_load:
            self.image_processor = CLIPImageProcessor.from_pretrained(
                f"{self.model_name_or_path}/{self.vision_tower_name}")
            self.vision_tower_model = InternVisionModel.from_pretrained(
                f"{self.model_name_or_path}/{self.vision_tower_name}/", trust_remote_code=True
            )
            self.vision_tower_model.requires_grad_(False)
            self.is_loaded = True
        else:
            self.cfg_only = AutoConfig.from_pretrained(
                self.vision_tower_name, trust_remote_code=True
            )

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower_model.dtype

    @property
    def device(self):
        return self.vision_tower_model.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower_model.config
        else:
            return self.cfg_only

    @config.setter
    def config(self, vision_tower_config):
        self._config = vision_tower_config

    @property
    def hidden_size(self):
        return self._config.hidden_size * (int(1 / self.scale_pix_shuffle) ** 2)

    @property
    def num_patches(self):
        return (self._config.image_size // self._config.patch_size) ** 2
    
    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        image_features = image_features[:, 1:]
        return image_features

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(
            n, int(h * scale_factor), int(w * scale_factor), int(c / (scale_factor * scale_factor))
        )
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    @torch.no_grad()
    def forward(self, images):
        if isinstance(images, list):
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower_model(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                )
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower_model(
                images.to(device=self.device, dtype=self.dtype), output_hidden_states=True
            )
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
        h = w = int(image_features.shape[1] ** 0.5)
        if image_features.shape[1] != h * w:
            raise ValueError("image_features shape error")
        image_features = image_features.reshape(image_features.shape[0], h, w, -1)
        image_features = self.pixel_shuffle(image_features * self.scale_pix_shuffle)
        image_features = image_features.reshape(
            image_features.shape[0], -1, image_features.shape[-1]
        )

        return image_features