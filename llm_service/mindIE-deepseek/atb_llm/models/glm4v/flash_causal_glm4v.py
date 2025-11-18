# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
# Copyright 2024 THUDM and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on HuggingFace's glm-4v-9b developed by THUDM.
# It has been modified from its original forms to accommodate minor 
# architectural differences compared to glm-4V-9b.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, List, Tuple
import importlib
import torch
import numpy as np
from PIL import Image

from transformers.configuration_utils import PretrainedConfig
from atb_llm.models.internvl.data_preprocess_internvl import create_standardization_params
from atb_llm.utils.shm_utils import get_data_from_shm
from ..base.flash_causal_lm import FlashForCausalLM
from .modeling_glm_vit import EVA2CLIPModel
from .modeling_glm_vit_atb import EVA2CLIPModelATB
from ...utils.multimodal_utils import safe_open_image

_GMASK_TOKEN_ID = 151331
_PLACEHOLDER = 0
CONV_CHANNELS = 3
CONV_GROUPS = 3
IMAGENET_MEAN = [0.48145466, 0.4578275, 0.40821073]
IMAGENET_STD = [0.26862954, 0.26130258, 0.27577711]
IMAGE_SIZE = 1120
MAX_NUM_PATCHES = 1
RESCALE_FACTOR = 1 / 255


class FlashGlm4vForCausalLM(FlashForCausalLM):
    def __init__(self, config, weights, **kwargs):
        super().__init__(config, weights, **kwargs)
        self.config = config
        self.weights = weights
        self.vocab_size = config.vocab_size
        self.vision = None
        self.language_model = None
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        
        self.npu_id = weights.device.index
        self.device = f"npu:{self.npu_id}"
        self.enable_atb_vit = True
        self.enable_normalizer = False

        self.init_vit()
        self.init_llm()
        self.model_type = config.model_type
        
        self.normalizer = None
        self.init_normalizer()
        
    def init_normalizer(self):
        weight, bias = create_standardization_params(IMAGENET_MEAN, IMAGENET_STD, RESCALE_FACTOR, CONV_CHANNELS)
        self.normalizer = torch.nn.Conv2d(in_channels=CONV_CHANNELS, out_channels=CONV_CHANNELS, kernel_size=1, \
            groups=CONV_GROUPS)
        self.normalizer.weight = torch.nn.Parameter(data=weight, requires_grad=False)
        self.normalizer.bias = torch.nn.Parameter(data=bias, requires_grad=False)
        self.normalizer.npu()
        # Normalizer warmup
        self.normalizer(torch.randn(MAX_NUM_PATCHES, CONV_CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device='npu'))
        
    @staticmethod
    def get_llm_model(model_type):
        model_file_dir_name = f"atb_llm.models.{model_type}."
        model_file_name = "flash_causal"
        module_path = f"{model_file_dir_name}{model_file_name}_{model_type}"
        module = importlib.import_module(module_path)
        model_cls_name = "Flash" + f"{model_type.capitalize()}ForCausalLM"
        return getattr(module, model_cls_name)
        
    def init_vit(self):
        prefix = "transformer.vision"
        if not self.enable_atb_vit:
            self.vision = EVA2CLIPModel(self.config)
            model_weights = [model_weight for model_weight in self.vision.state_dict().keys()]
            for model_weight in model_weights:
                saved_weight = torch.nn.Parameter(
                    self.weights.get_tensor(f"{prefix}.{model_weight}"),
                    requires_grad=False
                )
                saved_weight = torch.nn.Parameter(saved_weight, requires_grad=False)
                model_weight_list = model_weight.split(".")
                target_module = self.vision
                for nxt_module in model_weight_list[:-1]:
                    target_module = getattr(target_module, nxt_module)
                setattr(target_module, model_weight_list[-1], saved_weight)
        elif self.enable_atb_vit:
            self.vision = EVA2CLIPModelATB(self.config, self.weights)
            # load vit weights without using atb graph
            model_weights = []
            for model_weight in self.vision.state_dict().keys():
                if not model_weight.startswith("transformer"):
                    model_weights.append(model_weight)
            for model_weight in model_weights:
                saved_weight = torch.nn.Parameter(
                    self.weights.get_tensor(f"{prefix}.{model_weight}"),
                    requires_grad=False
                )
                model_weight_list = model_weight.split(".")
                target_module = self.vision
                for nxt_module in model_weight_list[:-1]:
                    target_module = getattr(target_module, nxt_module)
                setattr(target_module, model_weight_list[-1], saved_weight)
            # init vit transformer atb graph
            self.vision = self.vision.to(self.device)
            self.vision.transformer.init_graph()

    def init_llm(self):
        model_cls = self.get_llm_model(self.config.llm_model_type)
        self.language_model = model_cls(self.config, self.weights)
        self.language_model.skip_word_embedding = True

    def prepare_prefill_token_service(self, input_ids):
        if not torch.any(torch.eq(input_ids, self.config.boi_token_id)):
            return self.language_model.embedding.word_embeddings(input_ids)

        inputs_embeds = self.language_model.embedding.word_embeddings(input_ids)
        
        batch_boi_pos = torch.where(torch.eq(input_ids, self.config.boi_token_id))[0]
        batch_eoi_pos = torch.where(torch.eq(input_ids, self.config.eoi_token_id))[0]
        for idx, boi_pos in enumerate(batch_boi_pos):
            eoi_pos = batch_eoi_pos[idx]
            # get shm info from input_ids
            shm_value = input_ids[boi_pos + 1]
            shape_value = input_ids[boi_pos + 2]
            # get image feature
            input_image = get_data_from_shm(
                shm_value, shape_value, np.float32, self.device
            ).to(dtype=inputs_embeds.dtype).npu()
            image_features = self.vision(input_image)

            # replace embeds with image feature
            inputs_embeds[boi_pos:eoi_pos + 1] = image_features
        return inputs_embeds

    def prepare_prefill_token(self, multimodalinputs, processor, *args):
        image = multimodalinputs.image
        text = multimodalinputs.text

        image = safe_open_image(Image, image).convert("RGB")
        inputs = processor.apply_chat_template([{"role": "user", "image": image, "content": text}],
                                               add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                               return_dict=True)
        input_ids = inputs["input_ids"].npu()
        inputs_embeds = self.language_model.embedding.word_embeddings(input_ids)

        if not self.enable_normalizer:
            input_image = inputs["images"].to(dtype=inputs_embeds.dtype).npu()
            image.close()
        else:
            input_image = inputs["images"]
            image.close()
            input_image = self.normalizer(input_image.npu().float()).to(dtype=inputs_embeds.dtype)
        images_features = self.vision(input_image) 

        image_size: int = self.config.vision_config['image_size']
        patch_size: int = self.config.vision_config['patch_size']
        num_patches = (image_size // patch_size // 2) ** 2

        new_input_embeds, new_position_ids = [], []
        position_ids = torch.arange(len(input_ids[0]), dtype=torch.long).unsqueeze(0).repeat(1, 1)
        for i, x in enumerate(input_ids):
            input_id = x.tolist()
            boi_token_pos = input_id.index(self.config.boi_token_id)
            eoi_token_pos = input_id.index(self.config.eoi_token_id)
            new_input_embeds.append(torch.cat(
                (inputs_embeds[i, :boi_token_pos], images_features[i].to(inputs_embeds.device),
                    inputs_embeds[i, eoi_token_pos + 1:])))
            new_position_ids.append(torch.cat(
                (position_ids[i, :boi_token_pos + 1], position_ids[i, boi_token_pos + 1].repeat(num_patches),
                    position_ids[i, eoi_token_pos:])
            ))
        
        inputs_embeds = torch.stack(new_input_embeds, dim=0)
        position_ids = torch.stack(new_position_ids, dim=0)
        return inputs_embeds.squeeze(0), position_ids.squeeze(0)
    
    def init_ascend_operation(self, config: PretrainedConfig):
        pass

    def init_ascend_weight(self):
        pass

    def forward(
            self,
            input_ids: torch.Tensor,
            position_ids: torch.Tensor,
            is_prefill: bool,
            kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
            block_tables: torch.Tensor,
            slots: torch.Tensor,
            input_lengths: torch.Tensor,
            max_seq_len: int,
            lm_head_indices: Optional[torch.Tensor] = None,
            **kwargs):
        if is_prefill and input_ids.dim() == 1:
            input_ids = self.prepare_prefill_token_service(input_ids)
        return self.language_model.forward(input_ids,
                                           position_ids,
                                           is_prefill,
                                           kv_cache,
                                           block_tables,
                                           slots,
                                           input_lengths,
                                           max_seq_len,
                                           lm_head_indices,
                                           **kwargs)