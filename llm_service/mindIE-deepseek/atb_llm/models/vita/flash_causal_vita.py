# Copyright (C) 2024 THL A29 Limited, a Tencent company.  All rights reserved. 
# The below software and/or models in this distribution may have been modified 
# by THL A29 Limited ("Tencent Modifications"). 
# All Tencent Modifications are Copyright (C) THL A29 Limited.

import os
from typing import Optional, List, Tuple
import importlib
import warnings

import math
import json
import yaml
import numpy as np 
from PIL import Image
import torch
from torch.functional import F
from torch import nn
from transformers import AutoTokenizer
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING

from atb_llm.utils.dist import initialize_torch_distributed
from atb_llm.utils.shm_utils import get_data_from_shm
from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.models.base.model_utils import safe_from_pretrained
from .modeling_vita_audio import GlobalCMVN, WhaleEncoder, AudioEncoder, AudioEncoderProcessor
from .tool import tokenizer_image_audio_token, tokenizer_image_token
from .tool import dynamic_preprocess_with_mean, dynamic_preprocess, get_rawvideo_dec
from .tool import VideoConfig, ImageConfig
from .vision_tower import InternViTVisionTower
from ..internvl.config_intern_vit import InternVisionConfig
from ..mixtral.config_mixtral import MixtralConfig
from ..qwen2.config_qwen2 import Qwen2Config
from ..base.flash_causal_lm import FlashForCausalLM
from ..base.config import QuantizationConfig
from .conversation import conv_templates


MODEL_TYPE = "model_type"
_PAD_TOKEN_ID = 32001
GLOBAL_WEIGHTS_PATH = "OpenGVLab"
IMAGE_TOKEN_INDEX = -200
AUDIO_TOKEN_INDEX = -500
IMAGE_TAG = -1
VIDEO_TAG = -2
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_AUDIO_TOKEN = "<audio>"
IGNORE_INDEX = -100
MAX_IMAGE_LENGTH = 16


def load_cmvn_json(json_cmvn_file):
    with open(json_cmvn_file) as f:
        cmvn_json = json.load(f)

    avg = cmvn_json["mean_stat"]
    var = cmvn_json["var_stat"]
    count = cmvn_json["frame_num"]
    for i, _ in enumerate(avg):
        avg[i] /= count
        var[i] = var[i] / count - avg[i] * avg[i]
        if var[i] < 1.0e-20:
            var[i] = 1.0e-20
        var[i] = 1.0 / math.sqrt(var[i])
    cmvn = [avg, var]
    return cmvn


def get_supported_models():
    current_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    supported_models = []
    for foldername in os.listdir(current_path):
        is_folder = os.path.isdir(os.path.join(current_path, foldername))
        skip_base_folder = foldername != "base"
        skip_invalid_folder = not foldername.startswith("_")
        if is_folder and skip_base_folder and skip_invalid_folder:
            supported_models.append(foldername)
    return supported_models


def get_llm_model(model_type):
    supported_models = get_supported_models()
    if model_type not in supported_models:
        raise NotImplementedError(
            f"unsupported model type: {model_type};"
            f"请确认atb_llm.models路径下是否存在名为{model_type}的文件夹。"
        )
    
    model_file_dir_name = f"atb_llm.models.{model_type}."
    model_file_name = 'flash_causal'
    module_path = f"{model_file_dir_name}{model_file_name}_{model_type}"
    module = importlib.import_module(module_path)
    model_cls_name = "Flash" + f"{model_type.capitalize()}ForCausalLM"
    model_cls = getattr(module, model_cls_name)
    return model_cls


class MultiModalConfig(PretrainedConfig):
    model_type = "vita-mixtral"
    is_composition = False

    def __init__(self, vision_config=None, text_config=None, audio_config=None, **kwargs):
        self._init_visionconfig(vision_config)
        self._init_textconfig(text_config)
        self._init_audioconfig(audio_config)
        super().__init__(**kwargs)
    
    
    def _init_visionconfig(self, vision_config):
        if isinstance(vision_config, dict):
            vision_config[MODEL_TYPE] = (
                vision_config[MODEL_TYPE] if MODEL_TYPE in vision_config else "clip_vision_model"
            )
            vision_config = CONFIG_MAPPING[vision_config[MODEL_TYPE]](**vision_config)
        self.vision_config = vision_config

    def _init_textconfig(self, text_config):
        self.text_config = text_config
        
    def _init_audioconfig(self, audio_config):
        if isinstance(audio_config, dict):
            audio_config[MODEL_TYPE] = (
                audio_config[MODEL_TYPE] if MODEL_TYPE in audio_config else "wav2vec2-conformer"
            )
            audio_config = CONFIG_MAPPING[audio_config[MODEL_TYPE]](**audio_config)
        self.audio_config = audio_config
    

class VitaConfig(MultiModalConfig):

    model_type = "vita-mixtral"
    is_composition = False

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        audio_config=None,
        ignore_index=-100,
        image_token_index=32000,
        projector_hidden_act="gelu",
        vision_feature_select_strategy="default",
        vision_feature_layer=-2,
        **kwargs,
    ):
        vision_config, audio_config = self._split_config(kwargs)
        super().__init__(vision_config=vision_config, text_config=kwargs, audio_config=audio_config, **kwargs)
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act
        if vision_feature_select_strategy not in ["default", "full"]:
            raise ValueError(
                "vision_feature_select_strategy should be one of 'default', 'full'."
                f"Got: {vision_feature_select_strategy}"
            )

        if "vocab_size" in kwargs:
            warnings.warn(
                "The `vocab_size` argument is deprecated and will be removed in v4.42, \
                since it can be inferred from the `text_config`. \
                Passing this argument has no effect",
                FutureWarning,
            )

        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer
        self._vocab_size = self.text_config.vocab_size if self.text_config is not None else 0 

    @property
    def vocab_size(self):
        warnings.warn(
            "The `vocab_size` attribute is deprecated and will be removed in v4.42, \
            Please use `text_config.vocab_size` instead.",
            FutureWarning,
        )
        return self._vocab_size

    @vocab_size.setter
    def vocab_size(self, value):
        self._vocab_size = value

    def to_dict(self):
        output = super().to_dict()
        output.pop("_vocab_size", None)
        return output
    
    def _init_visionconfig(self, vision_config):
        if isinstance(vision_config, dict):
            self.vision_config = InternVisionConfig(**vision_config)
            self.vision_config.use_flash_attn = False
            self.vision_config.norm_type = "layer_norm"
            self.vision_config.image_size = 448
            self.vision_config.model_type = "intern_vit_6b"
            self.vision_config.qk_normalization = False
            self.vision_config.num_hidden_layers = 24
        elif vision_config is None:
            self.vision_config = InternVisionConfig(

            )

    def _init_textconfig(self, text_config):
        if isinstance(text_config, dict):
            if self.model_type == 'vita-mixtral':
                self.text_config = MixtralConfig(**text_config)
                self.text_config.model_type = "mixtral"
            elif self.model_type == 'vita-Qwen2':
                self.text_config = Qwen2Config(**text_config)
                self.text_config.model_type = "qwen2"
        elif self.text_config is None:
            self.text_config = MixtralConfig(
            )

    def _init_audioconfig(self, audio_config):
        if isinstance(audio_config, dict):
            self.mm_audio_encoder = audio_config.get("mm_audio_encoder", None)

    def _split_config(self, config):
        vision_keys = [
                        "mm_hidden_size", 
                        "num_attention_heads", 
                        "mm_vision_tower", 
                        "intermediate_size", 
                        "torch_dtype", 
                        "hidden_act", 
                        "hidden_size", 
                        "max_position_embeddings", 
                        "vocab_size",
        ]
        audio_keys = ["mm_audio_encoder", "mm_hidden_size", "mm_projector_type", "torch_dtype"]
        vision_config = {k: v for k, v in config.items() if k in vision_keys}
        audio_config = {k: v for k, v in config.items() if k in audio_keys}
        return vision_config, audio_config



class VitaMultiModalProjector(nn.Module):
    def __init__(self, config: VitaConfig):
        super().__init__()
        self.mm_projector = nn.Sequential(
            nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size, bias=True),
            ACT2FN[config.projector_hidden_act],  
            nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True) 
        )

    @torch.no_grad()
    def forward(self, image_features):
        hidden_states = self.mm_projector(image_features)
        return hidden_states


class MultiModalLLm(FlashForCausalLM):
    def __init__(self, config, weights, **kwargs):
        if getattr(config, "text_config"):
            if not config.quantize:
                setattr(config.text_config, 'quantize', None)
            else:
                setattr(config.text_config, 'quantize', config.quantize)
            setattr(config.text_config, 'quantization_config', QuantizationConfig(**{}))
            super().__init__(config.text_config, weights, **kwargs)
        else:
            super().__init__(config, weights, **kwargs)
        self.config = config
        self.weights = weights
        self.vocab_size = config.text_config.vocab_size

        self.vision_tower = None
        self.language_model = None
        self.audio_tower = None
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.npu_id = weights.device.index
        self.process_group, self.device = initialize_torch_distributed(self.tp_rank, self.npu_id, self.tp_world_size)
        if self.config.vision_config:
            self.init_vit()
        self.init_llm()
        if self.config.mm_audio_encoder:
            self.init_audio()
        self.model_type = None
        
    @staticmethod
    def init_audiotower(module, weights):
        audio_weights = [audio_weight for audio_weight in module.state_dict().keys()]
        for audio_weight in audio_weights:
            saved_weight = torch.nn.Parameter(
                    weights.get_tensor(f"model.audio_encoder.{audio_weight}"),
                    requires_grad=False
                )
            audio_weight_list = audio_weight.split(".")
            target_module = module
            for nxt_module in audio_weight_list[:-1]:
                target_module = getattr(target_module, nxt_module)
            setattr(target_module, audio_weight_list[-1], saved_weight)

    def init_visiontowerweight(self, module, weights):
        vision_weights = [vision_weight for vision_weight in module.state_dict().keys()]
        for vision_weight in vision_weights:
            saved_weight = torch.nn.Parameter(
                    weights.get_tensor(f"model.vision_tower.vision_tower.{vision_weight}"),
                    requires_grad=False
            )
            vision_weight_list = vision_weight.split(".")
            target_module = module
            for nxt_module in vision_weight_list[:-1]:
                target_module = getattr(target_module, nxt_module)
            setattr(target_module, vision_weight_list[-1], saved_weight)

    def init_vit(self):
        self.vision_tower = InternViTVisionTower(self.config.model_name_or_path, self.config.vision_config, 
                                                 self.process_group).to(self.config.torch_dtype)
        self.init_visiontowerweight(self.vision_tower.vision_tower_model, self.weights)
        self.config.vision_config.mm_vision_tower = f"{GLOBAL_WEIGHTS_PATH}/{self.config.vision_config.mm_vision_tower}"

    def init_llm(self):
        self.model_type = self.config.text_config.model_type
        model_cls = get_llm_model(self.model_type)
        self.language_model = model_cls(self.config.text_config, 
                                  self.weights,
                                  lmhead_prefix="lm_head",
                                  model_prefix="model")
        if self.model_type == 'qwen2':
            self.language_model.skip_word_embedding = True
        else:
            self.language_model.prefill_skip_word_embedding = True
        
    def init_audio(self):
        audio_path = os.path.join(self.config.model_name_or_path, self.config.mm_audio_encoder) 
        with open(os.path.join(audio_path, "train.yaml"), "r") as fin:
            audio_configs = yaml.safe_load(fin)
        cmvn_file = "cmvn_file"
        audio_configs[cmvn_file] = os.path.join(audio_path, "global_cmvn")
        model_conf = "model_conf"
        audio_configs[model_conf]["freeze_encoder"] = getattr(
            self.config, "freeze_audio_encoder", True
        )
        audio_configs[model_conf]["freeze_adpter"] = getattr(
            self.config, "freeze_audio_encoder_adapter", True
        )
        audio_configs[model_conf]["audio_prompt_finetune"] = getattr(
            self.config, "audio_prompt_finetune", False
        )
        audio_configs[model_conf]["audio_prompt_num"] = getattr(
            self.config, "audio_prompt_num", 0
        )
        if audio_configs[cmvn_file] is not None:
            mean, istd = load_cmvn_json(audio_configs[cmvn_file])
            global_cmvn = GlobalCMVN(torch.tensor(mean).float(), torch.tensor(istd).float())
        else:
            global_cmvn = None

        input_dim = audio_configs["input_dim"]
        encoder = WhaleEncoder(input_dim, global_cmvn=global_cmvn, **audio_configs["encoder_conf"])
        self.audio_tower = AudioEncoder(encoder=encoder, **audio_configs[model_conf])
        audio_processor = AudioEncoderProcessor(dataset_conf=audio_configs["dataset_conf"])

        self.audio_tower.audio_processor = audio_processor
        self.init_audiotower(self.audio_tower, self.weights)
    
    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if isinstance(vision_tower, list):
            vision_tower = vision_tower[0]
        return vision_tower
    
    def get_audio_tower(self):
        audio_tower = getattr(self, "audio_tower", None)
        return audio_tower


class FlashVitaForCausalLM(MultiModalLLm):
    def __init__(self, config, weights, **kwargs):
        super().__init__(config, weights, **kwargs)
        self.config = config
        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else _PAD_TOKEN_ID
        self.multi_modal_projector = None
        self.image_token_id = self.config.image_token_index
        self.architectures = self.config.architectures
        self.processor = safe_from_pretrained(AutoTokenizer, config.model_name_or_path, use_fast=True)
        self.device = "npu"
        self.model_type = self.config.text_config.model_type
        if self.model_type == 'qwen2':
            self.conv_mode = "qwen2p5_instruct"
        else:
            self.conv_mode = "mixtral_two"
        self.init_multimodal()
        if self.vocab_size != len(self.processor):
            self.resize_token_embeddings(len(self.processor))
        
    @staticmethod
    def init_multi_modal_projectorweight(module, weights):
        multimodel_weights = [multimodel_weight for multimodel_weight in module.state_dict().keys()]
        for multimodel_weight in multimodel_weights:
            saved_weight = torch.nn.Parameter(
                    weights.get_tensor(f"model.{multimodel_weight}"),
                    requires_grad=False
                )
            multimodel_weight_list = multimodel_weight.split(".")
            target_module = module
            for nxt_module in multimodel_weight_list[:-1]:
                target_module = getattr(target_module, nxt_module)
            setattr(target_module, multimodel_weight_list[-1], saved_weight)

    def init_multimodal(self):
        self.multi_modal_projector = VitaMultiModalProjector(self.config)
        self.init_multi_modal_projectorweight(self.multi_modal_projector, self.weights)

    def get_input_embeddings(self):
        if self.model_type == 'qwen2':
            return self.language_model.transformer.wte
        return self.language_model.model.embed_tokens

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.lm_head.linear

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()
    
    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        embeddings = self.get_resized_embeddings(new_num_tokens)
        self.config.text_config.vocab_size = new_num_tokens
        self.vocab_size = new_num_tokens
        self.get_resized_lm_head(new_num_tokens)
        return embeddings
    
    def get_resized_embeddings(self, new_num_tokens):
        embeddings = self.get_input_embeddings()
        old_num_tokens, old_embedding_dim = embeddings.weight.size()
        n = min(old_num_tokens, new_num_tokens)
        weights = nn.Parameter(F.pad(torch.randn(new_num_tokens, old_embedding_dim), (0, 0, 0, 1)))\
            .to(embeddings.weight.dtype).to(embeddings.weight.device)
        weights.data[:n, :] = embeddings.weight.data[:n, :]
        embeddings.weight.data = weights.data
        block_size = new_num_tokens // embeddings.process_group.size()
        rank = embeddings.process_group.rank()
        embeddings.min_id = rank * block_size
        embeddings.max_id = min(new_num_tokens, (rank + 1) * block_size)
        embeddings.null_idx = block_size
        return embeddings
    
    def get_resized_lm_head(self, new_num_tokens):
        lm_head = self.get_output_embeddings()
        if (os.getenv("MATMUL_ND_NZ_ENABLE", "0") == "0") and (os.getenv("ENABLE_ACLNN_MATMUL_BACKEND", "0") == "0"):
            old_num_tokens, old_lm_head_dim = lm_head.weight.size()
            num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
            weights = nn.Parameter(torch.randn(new_num_tokens, old_lm_head_dim)).to(lm_head.weight.dtype)\
                .to(lm_head.weight.device)
            weights.data[:num_tokens_to_copy, :] = lm_head.weight.data[:num_tokens_to_copy, :]
        else:
            old_lm_head_dim, old_num_tokens = lm_head.weight.size()
            num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
            weights = nn.Parameter(torch.randn(old_lm_head_dim, new_num_tokens)).to(lm_head.weight.dtype)\
                .to(lm_head.weight.device)
            weights.data[:, :num_tokens_to_copy] = lm_head.weight.data[:, :num_tokens_to_copy]
        lm_head.weight.data = weights.data
        if lm_head.bias:
            bias = nn.Parameter(torch.randn(new_num_tokens)).to(lm_head.bias.dtype).to(lm_head.bias.device)
            bias.data[:num_tokens_to_copy] = lm_head.bias.data[:num_tokens_to_copy]
            lm_head.weight.data = weights.data
        return lm_head
    
    def process_images(self, images, model_cfg):
        vision_tower = self.vision_tower
        image_processor = vision_tower.image_processor
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
    
    def fusionfeature(self, image_list, audios_list, input_ids):
        input_embeds = "inputs_embeds"
        if len(image_list) > 1:
            concat_images = torch.cat([image for image in image_list], dim=0)
            image_features = self.vision_tower(concat_images)
            image_features = self.multi_modal_projector(image_features).to(self.device)
        elif len(image_list) == 1:
            image_features = self.vision_tower(image_list[0])
            image_features = self.multi_modal_projector(image_features).to(self.device)
        
        if audios_list:
            audio_features = []
            for audio in audios_list:
                audio_feature = self.audio_tower(audio["audios"], audio["lengths"])
                audio_feature[input_embeds] = audio_feature[input_embeds].squeeze()
                audio_features.append(audio_feature)  
                
        labels = None
        attention_mask = None
        position_ids = None
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            )
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        
        input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        labels = [
            cur_labels[cur_attention_mask]
            for cur_labels, cur_attention_mask in zip(labels, attention_mask)
        ]
        new_input_embeds = []
        cur_image_idx = 0
        cur_audio_idx = 0

        cur_input_ids = input_ids[0]
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        num_audio_frames = (cur_input_ids == AUDIO_TOKEN_INDEX).sum()
        if num_images == 0 and num_audio_frames == 0:
            new_input_embeds = self.get_input_embeddings()(cur_input_ids)
        else:
            image_audio_token_indices = (
                [-1]
                + torch.where(
                    (cur_input_ids == IMAGE_TOKEN_INDEX) | (cur_input_ids == AUDIO_TOKEN_INDEX)
                )[0].tolist()
                + [cur_input_ids.shape[0]]
            )
            cur_input_ids_noim_noau = []
            cur_labels = labels[0]
            cur_labels_noim_noau = []
            for i in range(len(image_audio_token_indices) - 1):
                cur_input_ids_noim_noau.append(
                    cur_input_ids[
                        image_audio_token_indices[i] + 1 : image_audio_token_indices[i + 1]
                    ]
                )
                cur_labels_noim_noau.append(
                    cur_labels[image_audio_token_indices[i] + 1 : image_audio_token_indices[i + 1]]
                )

            split_sizes = [x.shape[0] for x in cur_labels_noim_noau]
            cur_input_embeds = self.get_input_embeddings()(torch.cat(cur_input_ids_noim_noau))
            cur_input_embeds_no_im_no_au = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            for i in range(num_images + num_audio_frames + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im_no_au[i])
                if i < num_images + num_audio_frames:
                    if cur_input_ids[image_audio_token_indices[i + 1]] == IMAGE_TOKEN_INDEX:
                        cur_image_features = image_features[cur_image_idx]
                        cur_image_idx += 1
                        cur_new_input_embeds.append(cur_image_features)
                    elif cur_input_ids[image_audio_token_indices[i + 1]] == AUDIO_TOKEN_INDEX:
                        cur_audio_features = audio_features[cur_audio_idx][input_embeds]
                        cur_audio_idx += 1
                        cur_new_input_embeds.append(cur_audio_features)
                    else:
                        error_msg = "Error: token_indices is illegal"
                        logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
                        raise ValueError(error_msg)
            new_input_embeds = torch.cat(cur_new_input_embeds)

        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = new_input_embeds[:tokenizer_model_max_length] 
        position_ids = torch.arange(
                        0, new_input_embeds.shape[0], dtype=position_ids.dtype, device=position_ids.device
                    )

        if _position_ids is None:
            position_ids = None
        return new_input_embeds, position_ids

    def prepare_prefill_token(self, multimodalinputs, processor):
        text = multimodalinputs.text
        image = multimodalinputs.image
        video = multimodalinputs.video
        audio = multimodalinputs.audio
        tokenizer = processor
        qs = text if text else ''
        audios_list = []
        images_list = []
        modality = "lang"
        if audio is not None:
            audio_processor = self.audio_tower.audio_processor
            audio_input, audio_for_llm_lens = audio_processor.process(os.path.join(audio))
            audio_length = audio_input.shape[0]
            audio_input = torch.unsqueeze(audio_input, dim=0)
            audio_length = torch.unsqueeze(torch.tensor(audio_length), dim=0)
            audio_for_llm_lens = torch.unsqueeze(torch.tensor(audio_for_llm_lens), dim=0)
            audios = dict()
            audios["audios"] = audio_input.half().npu()
            audios["lengths"] = audio_length.half().npu()
            audios["lengths_for_llm"] = audio_for_llm_lens.npu()
            audios_list.append(audios)

        vision_tower = self.vision_tower
        image_processor = vision_tower.image_processor
        if image is not None:
            if "vita" in self.architectures[0].lower() :
                self.config.image_token_index = IMAGE_TOKEN_INDEX
            else:
                raise ValueError("Only support vita for now")
            image = Image.open(image).convert("RGB")
            if multimodalinputs.framecat:
                image, p_num = dynamic_preprocess_with_mean(
                    image, 
                    ImageConfig(min_num=2, 
                                max_num=12, 
                                image_size=448, 
                                use_thumbnail=True, 
                                img_mean=image_processor.image_mean),
                )
            else:
                image, p_num = dynamic_preprocess(
                    image, min_num=1, max_num=12, image_size=448, use_thumbnail=True
                )
            if len(p_num) != 1:
                raise ValueError("length of p_num should be 1.")
            image_tensor = self.process_images(image, self.config).to(dtype=self.dtype, device=self.device)
            qs = DEFAULT_IMAGE_TOKEN * p_num[0] + "\n" + qs
            images_list.append(image_tensor)
            modality = "image"

        elif video is not None:
            max_frames = MAX_IMAGE_LENGTH
            video_framerate = 1
            video_frames, slice_len = get_rawvideo_dec(
                VideoConfig(video_path=video,
                    image_processor=image_processor,
                    max_frames=max_frames,
                    video_framerate=video_framerate,
                    image_aspect_ratio=getattr(self.config, "image_aspect_ratio", None)),
            )
            image_tensor = video_frames.half().npu()
            qs = DEFAULT_IMAGE_TOKEN * slice_len + "\n" + qs
            images_list.append(image_tensor)
            modality = "video"

        if audio:
            qs = qs + DEFAULT_AUDIO_TOKEN
    
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt(modality)
        if audio is not None:
            input_ids = (
                tokenizer_image_audio_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .npu()
            )
        else:
            input_ids = (
                tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .npu()
            )
        new_input_embeds, position_ids = self.fusionfeature(images_list, audios_list, input_ids)
        return new_input_embeds, position_ids.squeeze(0)
    
    def prepare_prefill_token_service(self, input_ids):
        image_pos = torch.where(torch.eq(input_ids, IMAGE_TOKEN_INDEX))[0]
        audio_pos = torch.where(torch.eq(input_ids, AUDIO_TOKEN_INDEX))[0]
        pad_pos = torch.where(torch.eq(input_ids, IGNORE_INDEX))[0]
        image_final_pos = []
        audio_final_pos = []
        ignore_final_pos = []
        for idx in image_pos:
            if input_ids[idx + 1] != IMAGE_TOKEN_INDEX:
                image_final_pos.append(idx)
        for idx in audio_pos:
            if input_ids[idx + 1] != AUDIO_TOKEN_INDEX:
                audio_final_pos.append(idx)
        for idx in pad_pos:
            if input_ids[idx + 1] != IGNORE_INDEX:
                ignore_final_pos.append(idx)
        
        image_final_pos.extend(audio_final_pos)
        image_final_pos.sort()
        media_pos = image_final_pos 
        if len(media_pos) == 0:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            return inputs_embeds

        start = 0
        audios_list = []
        images_list = []
        for i, mideaidx in enumerate(media_pos):
            if i == 0:
                new_input_ids = input_ids[start: mideaidx + 1]
            else:
                new_input_ids = torch.cat([
                        new_input_ids,
                        input_ids[start: mideaidx + 1]
                    ])
            shm_value = input_ids[mideaidx + 1]
            shape_value = input_ids[mideaidx + 2]
            tensor = get_data_from_shm(shm_value, shape_value, dtype=np.float16, device="npu")
            if input_ids[mideaidx] == IMAGE_TOKEN_INDEX:
                images_list.append(tensor)
            else:
                audio = dict()
                audio["audios"] = tensor
                audio_length = audio["audios"].shape[1]
                audio_length = torch.unsqueeze(torch.tensor(audio_length), dim=0)
                audio["lengths"] = audio_length.half().npu()
                audios_list.append(audio)
            start = ignore_final_pos[i] + 1
        new_input_ids = torch.cat([
                        new_input_ids,
                        input_ids[start:]
                    ]).unsqueeze(0)

        new_input_embeds, _ = self.fusionfeature(images_list, audios_list, new_input_ids)
        return new_input_embeds

    def init_ascend_operations(self, config: PretrainedConfig):
        pass

    def init_ascend_weight(self):
        pass

    @torch.no_grad()
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
                                          lm_head_indices)
    
    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, bos_pos):
        inputs_embeds[bos_pos[0]: bos_pos[1] + 1] = image_features
        return inputs_embeds