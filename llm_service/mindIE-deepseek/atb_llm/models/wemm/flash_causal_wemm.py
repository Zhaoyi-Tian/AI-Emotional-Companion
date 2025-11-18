# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

"""WeMM model."""

import abc
import importlib
from typing import Optional, List, Tuple

import numpy as np
import torch
from PIL import Image

from atb_llm.models.base.flash_causal_lm import FlashForCausalLM
from atb_llm.utils.layers import TensorEmbedding
from atb_llm.utils.log.logging import logger
from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.utils.multimodal_utils import safe_open_image
from atb_llm.utils.env import ENV
from atb_llm.models.base.config import QuantizationConfig
from atb_llm.models.base.flash_causal_multimodal import get_supported_models
from atb_llm.utils.shm_utils import get_data_from_shm
from .vision_model import Idefics2VisionTransformer
from .image_processor_2k import Idefics2ImageProcessor
from .modeling_downsampler import DownsamplerModel
from .data_preprocess_wemm import merge_visual_embed


DEFAULT_IMAGE_TOKEN = "<image>"
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_BEGIN_TOKEN = "<img>"
IMAGE_BEGIN_TOKEN_INDEX = -300
DEFAULT_IMAGE_END_TOKEN = "</img>"
IMAGE_END_TOKEN_INDEX = -400

IGNORE_INDEX = -100
EOS_TOKEN_ID = 92542


def format_lora_a_key(base_weight_prefix):
    return f"{base_weight_prefix}.Plora_A.weight"


def format_lora_b_key(base_weight_prefix):
    return f"{base_weight_prefix}.Plora_B.weight"


class MultiModalLLm(FlashForCausalLM):
    def __init__(self, config, weights, **kwargs):
        if getattr(config, "text_config", None):
            if not config.quantize:
                setattr(config.text_config, "quantize", None)
            else:
                setattr(config.text_config, "quantize", config.quantize)
            setattr(config.text_config, "quantization_config", QuantizationConfig(**{}))
            super().__init__(config.text_config, weights, **kwargs)
        else:
            super().__init__(config, weights, **kwargs)
        self.config = config
        self.weights = weights
        self.vocab_size = config.vocab_size
        self.vision_tower = None
        self.language_model = None
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.init_vit()
        self.init_llm()

    def init_vit(self):
        self.vision_tower = Idefics2VisionTransformer(self.config.vision_config)
        self.init_module_weight(self.vision_tower, self.weights, "vision_tower")

    def init_llm(self):
        model_type = "internlm2"
        supported_models = get_supported_models()
        if model_type not in supported_models:
            msg = (
                f"unsupported model type: {model_type};" f"请确认atb_llm.models路径下是否存在名为{model_type}的文件夹。"
            )
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise NotImplementedError(msg)

        model_file_dir_name = f"atb_llm.models.{model_type}.v2."
        model_file_name = "flash_causal"
        module_path = f"{model_file_dir_name}{model_file_name}_{model_type}"
        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError as e:
            raise ImportError(f"无法导入模块{module_path}, 请确认路径是否正确") from e
        model_cls_name = "Flash" + f"{model_type.capitalize()}ForCausalLM"
        cls = getattr(module, model_cls_name)
        prefix = "language_model."
        self.language_model = cls(
            self.config, self.weights, lmhead_prefix=f"{prefix}output", model_prefix=f"{prefix}model"
        )
        self.language_model.skip_word_embedding = True

    @abc.abstractmethod
    def init_multimodal(self):
        pass

    @abc.abstractmethod
    def prepare_prefill_token(self, text, image, video, processor, batch_size):
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
        **kwargs,
    ):
        self.language_model.adapter_manager = self.adapter_manager
        return self.language_model.forward(
            input_ids,
            position_ids,
            is_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_seq_len,
            lm_head_indices,
            im_mask=self.im_mask,
            **kwargs,
        )


class FlashWemmForCausalLM(MultiModalLLm):
    def __init__(self, config, weights, **kwargs):
        super().__init__(config, weights, **kwargs)
        self.im_mask = None
        self.rank = ENV.rank
        self.config = config
        self.config.eos_token_id = EOS_TOKEN_ID
        self.vocab_size = self.config.vocab_size

        self.tok_embeddings = TensorEmbedding(prefix="language_model.model.tok_embeddings", weights=weights)
        for p in self.tok_embeddings.parameters():
            p.requires_grad = False
        self.init_multimodal()

    def update_adapter_manager(self):
        self.adapter_manager.base_model = self
        self.adapter_manager.format_lora_a_key = format_lora_a_key
        self.adapter_manager.format_lora_b_key = format_lora_b_key
        self.adapter_manager.enable_single_adapter_only = True

    def init_module_weight(self, module, weights, prefix="vision_model", prefixskip=None):
        model_weights = [model_weight for model_weight in module.state_dict().keys()]
        for model_weight in model_weights:
            if prefixskip and prefixskip in model_weight:
                continue
            saved_weight = torch.nn.Parameter(weights.get_tensor(f"{prefix}.{model_weight}"), requires_grad=False)
            model_weight_list = model_weight.split(".")
            target_module = module
            for nxt_module in model_weight_list[:-1]:
                target_module = getattr(target_module, nxt_module)
            setattr(target_module, model_weight_list[-1], saved_weight)

    def init_multimodal(self):
        self.image_processor = Idefics2ImageProcessor(self.config.image_processor)
        self.downsampler = DownsamplerModel(self.config.downsampler_config)
        self.init_module_weight(self.downsampler, self.weights, prefix="downsampler")

        self.visual_source_spliter_emb = torch.nn.Embedding(**self.config.spliter_emb_config)
        self.init_module_weight(self.visual_source_spliter_emb, self.weights, prefix="visual_source_spliter_emb")

        self.do_image_splitting = self.config.do_image_splitting

    def prepare_prefill_token(self, multimodalparams, processor):
        text = multimodalparams.text
        image = multimodalparams.image
        batch_size = multimodalparams.batch_size

        prompt = "<image>" + "\n" + text
        prompt = f"<|im_start|>user\n{prompt}<|im_end|><|im_start|>assistant\n"

        image = safe_open_image(Image, image)
        image_rgb = image.convert("RGB")
        image_size = self.config.image_processor["size"]
        navit980_images = self.image_processor(
            [[image_rgb]], size=image_size, return_tensors="pt", do_image_splitting=self.do_image_splitting
        )
        image.close()

        merged_visual_embeddings = merge_visual_embed(
            navit980_images, self.vision_tower, self.downsampler, self.visual_source_spliter_emb
        )
        chunk_encode = []
        for idx, chunk in enumerate(prompt.split(DEFAULT_IMAGE_TOKEN)):
            if idx == 0:
                cur_encode = processor.encode(chunk)
            else:
                cur_encode = processor.encode(chunk, add_special_tokens=False)
            chunk_encode.append(cur_encode)

        if len(chunk_encode) != 2:
            raise ValueError("Length of chunk_encode should be 2")
        
        ids = []
        for idx, cur_chunk_encode in enumerate(chunk_encode):
            ids.extend(cur_chunk_encode)
            if idx != len(chunk_encode) - 1:
                ids.append(IMAGE_TOKEN_INDEX)
        ids = torch.tensor(ids).npu().unsqueeze(0)
        pixel_values = None
        mm_inputs = self.prepare_inputs_labels_for_multimodal(
            llm=self.language_model, 
            input_ids=ids, 
            pixel_values=pixel_values, 
            clip_embeddings=merged_visual_embeddings
        )

        self.im_mask = mm_inputs.get("im_mask", None)
        if self.im_mask is not None:
            self.im_mask = self.im_mask.view(1, -1)
            self.im_mask = self.im_mask.squeeze(0).unsqueeze(-1).to(torch.float16)
            self.im_mask = torch.cat([self.im_mask for _ in range(batch_size)], dim=0)

        inputs_embeds = mm_inputs.get("inputs_embeds", None)
        inputs_embeds = inputs_embeds.view(inputs_embeds.shape[0] * inputs_embeds.shape[1], inputs_embeds.shape[2])
        return inputs_embeds

    def prepare_prefill_token_service(self, ids):
        if not torch.any(torch.eq(ids, IMAGE_BEGIN_TOKEN_INDEX)):
            inputs_embeds = self.language_model.model.tok_embeddings(ids)
            self.im_mask = torch.zeros(inputs_embeds.shape[0], 1, device=inputs_embeds.device).to(torch.float16)
            return inputs_embeds

        bos_pos = torch.where(torch.eq(ids, IMAGE_BEGIN_TOKEN_INDEX))[0]
        eos_pos = torch.where(torch.eq(ids, IMAGE_END_TOKEN_INDEX))[0]
        values_shm_name = ids[bos_pos + 1]
        values_shape_value = ids[bos_pos + 2]
        mask_shm_name = ids[bos_pos + 3]
        mask_shape_value = ids[bos_pos + 4]

        navit980_images = {}
        navit980_images["navit_pixel_values"] = get_data_from_shm(
            values_shm_name, values_shape_value, np.float32, self.device
        )
        navit980_images["pixel_attention_mask"] = get_data_from_shm(
            mask_shm_name, mask_shape_value, np.bool8, self.device
        )
        merged_visual_embeddings = merge_visual_embed(
            navit980_images, self.vision_tower, self.downsampler, self.visual_source_spliter_emb
        )

        pixel_values = None
        in_ids = []
        in_ids.extend(ids[:bos_pos].cpu().numpy().tolist())
        in_ids.append(IMAGE_TOKEN_INDEX)
        in_ids.extend(ids[eos_pos + 1 :].cpu().numpy().tolist())
        in_ids = torch.tensor(in_ids).npu().unsqueeze(0)
        mm_inputs = self.prepare_inputs_labels_for_multimodal(
            llm=self.language_model,
            input_ids=in_ids,
            pixel_values=pixel_values,
            clip_embeddings=merged_visual_embeddings,
        )

        self.im_mask = mm_inputs.get("im_mask", None)
        if self.im_mask is not None:
            self.im_mask = self.im_mask.view(1, -1)
            self.im_mask = self.im_mask.squeeze(0).unsqueeze(-1).to(torch.float16)

        inputs_embeds = mm_inputs.get("inputs_embeds", None)
        inputs_embeds = inputs_embeds.view(inputs_embeds.shape[0] * inputs_embeds.shape[1], inputs_embeds.shape[2])
        return inputs_embeds

    def chat(self, conversations, gen_config=None):
        prompt = ""
        image_path = conversations[0]["images"][0]
        for _, ann in enumerate(conversations):
            if ann["role"] == "user":
                prompt += f"<|im_start|>user\n{ann['content']}<|im_end|>\n"
            elif ann["role"] == "assistant":
                prompt += f"<|im_start|>assistant\n{ann['content']}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        with torch.no_grad():
            output = self.generate(image_path, prompt, gen_config=gen_config)
        return output

    def chat_v2(self, conversations, images, gen_config=None):
        image_path = images["images"][0]
        with torch.no_grad():
            output = self.generate(image_path, conversations, gen_config=gen_config)
        return output

    def get_valid_visual_embedding(self, embedding, valid_token_shape):
        if valid_token_shape is None:
            return embedding
        h, w = valid_token_shape
        return embedding[:h, :w, :].reshape(h * w, -1)

    def prepare_inputs_labels_for_multimodal(
        self,
        llm,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        clip_embeddings: Optional[torch.FloatTensor] = None
    ):
        if pixel_values is None and clip_embeddings is None:
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "inputs_embeds": None,
                "labels": labels,
            }

        _labels = labels
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        input_ids = [
            cur_input_ids[cur_attention_mask] 
            for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        labels = [
            cur_labels[cur_attention_mask] 
            for cur_labels, cur_attention_mask in zip(labels, attention_mask)
        ]

        new_inputs_embeds = []
        new_labels = []
        new_img_masks = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_pixel_values = pixel_values[cur_image_idx] if pixel_values is not None else None
                cur_clip_emb = clip_embeddings[cur_image_idx] if clip_embeddings is not None else None
                cur_inputs_embeds_1 = llm.model.tok_embeddings(cur_input_ids)
                if cur_clip_emb is not None and cur_pixel_values is not None:
                    cur_inputs_embeds = torch.cat(
                        [cur_inputs_embeds_1, cur_pixel_values[0:0], cur_clip_emb[0:0]], dim=0
                    )
                elif cur_pixel_values is not None:
                    cur_inputs_embeds = torch.cat([cur_inputs_embeds_1, cur_pixel_values[0:0]], dim=0)
                elif cur_clip_emb is not None:
                    cur_inputs_embeds = torch.cat([cur_inputs_embeds_1, cur_clip_emb[0:0]], dim=0)
                else:
                    raise ValueError
                new_inputs_embeds.append(cur_inputs_embeds)
                new_labels.append(labels[batch_idx])
                new_img_masks.append(torch.zeros(cur_inputs_embeds.shape[0], device=cur_inputs_embeds.device).bool())
                cur_image_idx += 1
                continue

            image_token_indices = (
                [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            )
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_inputs_embeds = llm.model.tok_embeddings(torch.cat(cur_input_ids_noim))
            cur_inputs_embeds_no_im = torch.split(cur_inputs_embeds, split_sizes, dim=0)
            cur_new_inputs_embeds = []
            cur_new_labels = []
            cur_img_masks = []

            for i in range(num_images + 1):
                cur_new_inputs_embeds.append(cur_inputs_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                cur_img_masks.append(
                    torch.zeros(cur_inputs_embeds_no_im[i].shape[0], device=cur_inputs_embeds_no_im[i].device).bool()
                )
                if i < num_images:
                    cur_pixel_values = pixel_values[cur_image_idx] if pixel_values is not None else None
                    cur_clip_emb = clip_embeddings[cur_image_idx] if clip_embeddings is not None else None

                    cur_image_idx += 1

                    # discrete token embeddings
                    if cur_pixel_values is not None:
                        cur_new_inputs_embeds.append(cur_pixel_values)
                        cur_img_masks.append(
                            torch.ones(cur_pixel_values.shape[0], device=cur_pixel_values.device).bool()
                        )
                        cur_new_labels.append(
                            torch.full(
                                (cur_pixel_values.shape[0],),
                                IGNORE_INDEX,
                                device=cur_labels.device,
                                dtype=cur_labels.dtype,
                            )
                        )

                    # clip embeddings
                    if cur_clip_emb is not None:
                        cur_new_inputs_embeds.append(cur_clip_emb)
                        cur_img_masks.append(torch.ones(cur_clip_emb.shape[0], device=cur_clip_emb.device).bool())
                        cur_new_labels.append(
                            torch.full(
                                (cur_clip_emb.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype
                            )
                        )

            cur_new_inputs_embeds = torch.cat(cur_new_inputs_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            cur_img_masks = torch.cat(cur_img_masks)

            new_inputs_embeds.append(cur_new_inputs_embeds)
            new_labels.append(cur_new_labels)
            new_img_masks.append(cur_img_masks)

        # Combine them
        max_len = max(x.shape[0] for x in new_inputs_embeds)
        batch_size = len(new_inputs_embeds)

        new_inputs_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device
        )
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        new_img_masks_padded = torch.zeros((batch_size, max_len), device=new_img_masks[0].device).bool()

        for i, (cur_new_embed, cur_new_labels, cur_new_img_masks) in enumerate(
            zip(new_inputs_embeds, new_labels, new_img_masks)
        ):
            cur_new_embed = cur_new_embed[:max_len]
            cur_new_labels = cur_new_labels[:max_len]
            cur_new_img_masks = cur_new_img_masks[:max_len]

            cur_len = cur_new_embed.shape[0]
            new_inputs_embeds_padded.append(
                torch.cat(
                    (
                        cur_new_embed,
                        torch.zeros(
                            (max_len - cur_len, cur_new_embed.shape[1]),
                            dtype=cur_new_embed.dtype,
                            device=cur_new_embed.device,
                        ),
                    ),
                    dim=0,
                )
            )
            if cur_len > 0:
                new_labels_padded[i, :cur_len] = cur_new_labels
                attention_mask[i, :cur_len] = True
                new_img_masks_padded[i, :cur_len] = cur_new_img_masks

        new_inputs_embeds = torch.stack(new_inputs_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        prepared_data = {
            "input_ids": None,
            "attention_mask": attention_mask,
            "inputs_embeds": new_inputs_embeds,
            "labels": new_labels,
        }
        # if pixel_values is not None:
        prepared_data.update({"im_mask": new_img_masks_padded})
        return prepared_data

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
        **kwargs,
    ):
        self.language_model.adapter_manager = self.adapter_manager
        kwargs.update({"adapter_ids": ["wemm"]})
        if is_prefill and input_ids.dim() == 1:
            inputs_embeds = self.prepare_prefill_token_service(input_ids)
        else:
            inputs_embeds = input_ids

        return self.language_model.forward(
            inputs_embeds,
            position_ids,
            is_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_seq_len,
            lm_head_indices,
            im_mask=self.im_mask,
            **kwargs,
        )
