# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import torch
import torch.distributed
import torch.utils.checkpoint
from atb_llm.models.base.model_utils import BaseModel
from transformers import AutoModel
from ..base.model_utils import safe_from_pretrained

huggingface_support_model = {
    "chinese_clip",
    "clip"
}


class ClipModel(BaseModel):
    def __init__(self, config, using_fp16=False):
        super().__init__()
        self.using_fp16 = using_fp16
        self.config = config
        self.get_specific_model(config)

    def get_specific_model(self, config):
        if config.model_type in huggingface_support_model:
            self.clip_model = safe_from_pretrained(AutoModel, config.name_or_path)
        else:
            raise NotImplementedError('Not Implemented')
        self.clip_model.eval()
        self.clip_model.to(torch.float16) if self.using_fp16 else None

    def forward(
            self,
            input_ids: torch.Tensor,
            pixel_values: torch.Tensor,
            **kwargs,
    ) -> torch.Tensor:
        pixel_values = pixel_values.to(
            torch.float16) if self.using_fp16 else pixel_values
        outputs = self.clip_model(input_ids, pixel_values, **kwargs)
        return outputs

    def extract_image_features(
            self,
            pixel_values: torch.Tensor,
            **kwargs,
    ) -> torch.Tensor:
        pixel_values = pixel_values.to(
            torch.float16) if self.using_fp16 else pixel_values
        image_features = self.clip_model.get_image_features(
            pixel_values, **kwargs)
        image_features = image_features / \
            image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features

    def extract_text_features(
            self,
            input_ids: torch.Tensor,
            **kwargs,
    ) -> torch.Tensor:
        text_features = self.clip_model.get_text_features(input_ids, **kwargs)
        text_features = text_features / \
            text_features.norm(p=2, dim=-1, keepdim=True)  # normalize
        return text_features

    def get_logits(
            self,
            image_features: torch.Tensor,
            text_features: torch.Tensor,
    ) -> torch.Tensor:
        if hasattr(self.clip_model, "logit_scale"):
            logit_scale = self.clip_model.logit_scale.exp()
        else:
            logit_scale = torch.tensor(
                100, dtype=self.config.torch_dtype).to(image_features.device)
        logits_per_text = torch.matmul(
            text_features, image_features.t()) * logit_scale
        logits_per_image = logits_per_text.t()
        return logits_per_image
