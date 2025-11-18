# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import json
import re
from functools import partial, reduce
from typing import Optional, List, Tuple, Dict

import torch
from PIL import Image
from torch import nn
from transformers.image_processing_utils import BatchFeature, get_size_dict
from transformers.image_transforms import (convert_to_rgb, normalize, rescale, resize, to_channel_dimension_format, )
from transformers.image_utils import (ChannelDimension, PILImageResampling, to_numpy_array, )

from atb_llm.models.base.flash_causal_lm import FlashForCausalLM
from atb_llm.utils.data.weight_wrapper import WeightWrapper, AttnWrapper, MlpWrapper
from atb_llm.utils.layers import load_column_multi
from atb_llm.utils.log.logging import logger
from atb_llm.utils.env import ENV
from atb_llm.utils.multimodal_utils import safe_open_image
from .configuration_bunny import SigLipVisionConfig
from .modeling_bunny import FlashBunnyMinicpmModel, BunnyConfig
from .modeling_siglipvision import build_vision_projector, SigLipVisionModel


IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200


class SigLipImageProcessor:
    def __init__(self,
                 image_mean=(0.5, 0.5, 0.5),
                 image_std=(0.5, 0.5, 0.5),
                 size=(384, 384),
                 crop_size: Dict[str, int] = None,
                 resample=PILImageResampling.BICUBIC,
                 rescale_factor=1 / 255,
                 data_format=ChannelDimension.FIRST):
        crop_size = crop_size if crop_size is not None else {"height": 384, "width": 384}
        crop_size = get_size_dict(crop_size, default_to_square=True, param_name="crop_size")

        self.image_mean = image_mean
        self.image_std = image_std
        self.size = size
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.data_format = data_format
        self.crop_size = crop_size

    def preprocess(self, images, return_tensors):
        if isinstance(images, Image.Image):
            images = [images]
        else:
            pass

        transforms = [
            convert_to_rgb,
            to_numpy_array,
            partial(resize, size=self.size, resample=self.resample, data_format=self.data_format),
            partial(rescale, scale=self.rescale_factor, data_format=self.data_format),
            partial(normalize, mean=self.image_mean, std=self.image_std, data_format=self.data_format),
            partial(to_channel_dimension_format, channel_dim=self.data_format, input_channel_dim=self.data_format),
        ]

        images = reduce(lambda x, f: [*map(f, x)], transforms, images)
        data = {"pixel_values": images}

        return BatchFeature(data=data, tensor_type=return_tensors)


class FlashBunnyForCausalLM(FlashForCausalLM):
    def __init__(self, config, weights, **kwargs):
        super().__init__(config, weights, **kwargs)
        self.transformer = FlashBunnyMinicpmModel(config, weights)

        self.lm_head = load_column_multi(
            config,
            prefixes=["model.embed_tokens"],
            weights=weights,
            head_size=1,
            lm_head=True,
        )

        self.config = config
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.in_tensor_length = 13
        self.total_head_nums = config.hidden_size // self.head_dim
        self.acl_encoder_operation_inputs: list[None | torch.Tensor] = [None] * self.in_tensor_length
        self.acl_decoder_operation_inputs: list[None | torch.Tensor] = [None] * self.in_tensor_length

        if self.compress_head_enable:
            self.acl_encoder_operation_inputs.append(None)
            self.acl_encoder_operation_inputs.append(None)
            self.acl_decoder_operation_inputs.append(None)
            self.acl_decoder_operation_inputs.append(None)

        self.placeholder = torch.zeros(1, dtype=self.dtype, device="npu")
        self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int64, device="npu")

        self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        self.transdata_param = json.dumps({})
        self.transdata_operation.set_param(self.transdata_param)

        self.rope_given_inv_feq_str = config.rope_given_inv_feq_str
        self.atten_mask_cpu = None
        self.skip_word_embedding = False
        self.cos_embed = None
        self.sin_embed = None
        self.wins_batch_1 = None
        self.decoder_slots = None
        self.all_wins_batch = None
        self.block_tables_global = None
        self.wins_global = None
        self.scale_emb = config.scale_emb
        self.scale_depth = config.scale_depth
        self.dim_model_base = config.dim_model_base
        self.num_hidden_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        self.acl_param = None
        self.ascend_weight = None

        # visual
        self.vision_config = SigLipVisionConfig()
        self.visual = SigLipVisionModel(self.vision_config).to(
            device=weights.device, dtype=weights.dtype
        )
        self.visual.requires_grad_(False)

        # vision_projector
        self.mlp1 = build_vision_projector(self.config)
        self.mlp1 = self.mlp1.to(device=weights.device, dtype=weights.dtype)

        self.init_module_weight(self.visual, weights, prefix="model.vision_tower.vision_tower")
        self.init_module_weight(self.mlp1, weights, prefix="model.mm_projector")

    def build_vision_projector(self, config, delay_load=False, **kwargs):
        projector_type = getattr(config, 'mm_projector_type', 'mlp2x_gelu')

        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
        
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    def init_module_weight(self, module, weights, prefix="vision_model"):
        model_weights = [model_weight for model_weight in module.state_dict().keys()]
        for model_weight in model_weights:
            saved_weight = torch.nn.Parameter(
                weights.get_tensor(f"{prefix}.{model_weight}"), requires_grad=False
            )
            model_weight_list = model_weight.split(".")
            target_module = module
            for nxt_module in model_weight_list[:-1]:
                target_module = getattr(target_module, nxt_module)
            setattr(target_module, model_weight_list[-1], saved_weight)

    def init_ascend_operations(self, config: BunnyConfig):
        if self.num_key_value_heads != self.num_attention_heads:
            raise ValueError("minicpm does not support GQA")

        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch(
            "minicpm_DecoderModel"
        )
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch(
            "minicpm_DecoderModel"
        )
        logger.info(">>>> minicpm_DecoderModel is called.")

    def get_weights(self):
        attn_wrapper = AttnWrapper(
            norm_name="ln_1",
            wrapper_name="attn",
            pack_name="c_attn",
            sep_names=None,
            o_name="c_proj",
        )
        mlp_wrapper = MlpWrapper(
            norm_name="ln_2",
            wrapper_name="mlp",
            pack_name="w2_w1",
            sep_names=None,
            down_name="c_proj",
        )
        weight_wrapper = WeightWrapper(
            self.soc_info, self.tp_rank, attn_wrapper, mlp_wrapper
        )
        weight_wrapper.register_embedding(self.transformer.wte)
        for i in range(self.num_layers):
            layer = self.transformer.h[i]
            weight_wrapper.register_layer(layer, self.quantize)
            if self.soc_info.need_nz:
                del layer.attn
                del layer.ln_2
                del layer.mlp
        weight_wrapper.register_model_norm(self.transformer.ln_f)
        weight_wrapper.register_model_lmhead(self.lm_head)
        return weight_wrapper

    def init_ascend_weight(self):
        weight_wrapper = self.get_weights()
        self.ascend_weight = weight_wrapper.weights
        linear_types = weight_wrapper.linear_type
        pack_quant_configs = weight_wrapper.pack_quant_type
        linear_transpose_types = weight_wrapper.linear_transpose_types


        rank_table_file = ENV.rank_table_file

        acl_param_dict = {
            "rmsNormEps": self.config.rms_norm_eps,
            "numAttentionHeadsPerRank": self.num_attention_heads,
            "hiddenSizePerAttentionHead": self.head_size,
            "numKeyValueHeadsPerRank": self.num_key_value_heads,
            "skipWordEmbedding": False,
            "isFA": False,
            "isBF16": self.dtype == torch.bfloat16,
            "packQuantType": pack_quant_configs,
            "linearQuantType": linear_types,
            "linearTransposeType": linear_transpose_types,
            "isEmbeddingParallel": False,
            "isLmHeadParallel": False,
            "lmHeadTransposeType": self.lm_head.linear.trans_flag,
            "supportSwiGLU": False if self.soc_info.need_nz else True,
            "kvQuant": self.config.quantization_config.kv_quant_type is not None,
            "rank": self.tp_rank,
            "worldSize": self.tp_world_size,
            "backend": self.soc_info.communication_backend,
            "rankTableFile": rank_table_file,
            "positionEmbeddingType": "ROPE",
            "enableAddNorm": False,
            "gemma": False,
            "supportCompressHead": self.compress_head_enable,
            "hiddenSize": self.hidden_size,
            "numHiddenLayers": self.config.num_hidden_layers,
            "scale_emb": 12.0,  # self.scale_emb,
            "scale_depth": 1.4,  # self.scale_depth,
            "dim_model_base": 256,  # self.dim_model_base,
            "num_hidden_layers": self.num_hidden_layers,
        }

        acl_param_encoder_content = {
            **acl_param_dict,
            "isPrefill": True,
            "supportLcoc": self.lcoc_enable,
            "supportSpeculate": False,
            "skipWordEmbedding": True
        }

        acl_param_decoder_content = {  
            **acl_param_dict,
            "isPrefill": False,
            "supportLcoc": False,
            "supportSpeculate": self.speculate_enable           
        }

        acl_param_encoder = json.dumps(acl_param_encoder_content)
        acl_param_decoder = json.dumps(acl_param_decoder_content)

        self.acl_encoder_operation.set_param(acl_param_encoder)
        self.acl_decoder_operation.set_param(acl_param_decoder)

        self.acl_encoder_operation.set_weight(self.ascend_weight)
        self.acl_decoder_operation.set_weight(self.ascend_weight)

    def expand2square(self, pil_img, background_color):
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

    def process_images(self, images, model_cfg):
        image_processor = SigLipImageProcessor()
        image_processor.crop_size = image_processor.size

        image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
        new_images = []
        if image_aspect_ratio == 'pad':
            for image in images:
                image = self.expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))
                image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                new_images.append(image)
        else:
            return image_processor(images, return_tensors='pt')['pixel_values']
        if all(x.shape == new_images[0].shape for x in new_images):
            new_images = torch.stack(new_images, dim=0)
        return new_images

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[-2]

        return image_features

    def encode_images(self, images):
        # siglip
        image_forward_outs = self.visual(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)

        # mm_projector
        image_features = self.mlp1(image_features)
        return image_features

    def prepare_prefill_token(self, text, image, video, processor, batch_size):
        text_prompt = f"A chat between a curious user and an artificial intelligence assistant. \
        The assistant gives helpful, detailed, and polite answers to the user's questions. \
        USER: <image>\n{text} ASSISTANT:"

        text_chunks = [processor(chunk).input_ids for chunk in text_prompt.split('<image>')]
        input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1][1:],
                                 dtype=torch.long).unsqueeze(0).to(self.device)

        if image is None or input_ids.shape[1] == 1:
            return self.transformer.wte(input_ids)
        image = safe_open_image(Image, image)
        images = self.process_images([image], self.config).to(dtype=self.dtype, device=self.device)
        image.close()
        if isinstance(images, list) or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1).to(self.device) for x in image_features]
        else:
            image_features = self.encode_images(images).to(self.device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        input_attention = zip(input_ids, attention_mask)
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in input_attention]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.transformer.wte(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [
                cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1:image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1:image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.transformer.wte(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)

            new_input_embeds.append(cur_new_input_embeds)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)

        new_input_embeds_padded = []

        for cur_new_embed in new_input_embeds:
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device)
                ), dim=0))
        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        new_input_embeds[0] *= self.scale_emb  # BunnyMiniCPMModel scale_emb：12
        return new_input_embeds[0]

    def init_cos_sin_table(self, max_seq_len, dim, dtype, device):
        self._init_rope_cos_sin(max_seq_len, dtype, device)

    def prepare_inputs_for_ascend(
            self, input_ids: torch.Tensor,
            position_ids: torch.Tensor,
            is_prefill: bool,
            kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
            block_tables: torch.Tensor,
            slots: torch.Tensor,
            input_lengths: torch.Tensor,
            max_seq_len: int,
            lm_head_indices: Optional[torch.Tensor] = None,
            **kwargs
    ):
        if is_prefill:
            atten_mask = self.attn_mask.get_attn_mask(self.max_base_len, self.dtype,
                                                      self.device)
            self.init_cos_sin_table(self.max_position_embeddings, self.head_dim, self.dtype, self.device)

            if self.soc_info.need_nz:
                atten_mask = self.transdata_operation.execute([atten_mask])[0]
            if lm_head_indices is None:
                lm_head_indices = torch.tensor(range(input_ids.shape[0]),
                                               dtype=torch.int64, device=input_ids.device)

            self.acl_param = json.dumps({
                "seqLen": input_lengths.tolist()
            })
            self.acl_encoder_operation_inputs[0] = self.placeholder
            self.acl_encoder_operation_inputs[1] = input_ids
            self.acl_encoder_operation_inputs[2] = position_ids.to(torch.int64)
            self.acl_encoder_operation_inputs[3] = self.cos_embed
            self.acl_encoder_operation_inputs[4] = self.sin_embed
            self.acl_encoder_operation_inputs[5] = atten_mask
            self.acl_encoder_operation_inputs[6] = block_tables.to(torch.int32)
            self.acl_encoder_operation_inputs[7] = slots.to(torch.int32)

            # IN_TENSOR_KV_CACHE_IDX
            self.acl_encoder_operation_inputs[8] = self.placeholder

            # IN_TENSOR_TOKEN_OFFSET
            self.acl_encoder_operation_inputs[9] = self.placeholder

            # IN_TENSOR_PLACE_HOLDER
            self.acl_encoder_operation_inputs[10] = self.placeholder

            # IN_TENSOR_SEQ_LEN
            self.acl_encoder_operation_inputs[11] = input_lengths.to(torch.int32)

            # IN_TENSOR_LOGTIS_INDICES
            self.acl_encoder_operation_inputs[12] = lm_head_indices.to(torch.int64)

            return self.acl_encoder_operation_inputs, self.acl_param
        else:
            q_lens = kwargs.get('q_lens', [])
            spec_mask = kwargs.get('spec_mask', None)

            self.acl_param = json.dumps({
                "seqLen": input_lengths.tolist(),
                "qLen": q_lens
            })
            # if self.speculate_enable and self.soc_info.need_nz:
            atten_mask = spec_mask if self.speculate_enable else self.attn_mask_fake

            self.acl_decoder_operation_inputs[0] = input_ids
            self.acl_decoder_operation_inputs[1] = self.placeholder
            self.acl_decoder_operation_inputs[2] = position_ids.to(torch.int64)
            self.acl_decoder_operation_inputs[3] = self.cos_embed
            self.acl_decoder_operation_inputs[4] = self.sin_embed
            self.acl_decoder_operation_inputs[5] = atten_mask
            self.acl_decoder_operation_inputs[6] = block_tables.to(torch.int32)
            self.acl_decoder_operation_inputs[7] = slots.to(torch.int32)
            self.acl_decoder_operation_inputs[8] = self.placeholder
            self.acl_decoder_operation_inputs[9] = self.placeholder
            self.acl_decoder_operation_inputs[10] = self.placeholder
            self.acl_decoder_operation_inputs[11] = input_lengths.to(torch.int32)
            self.acl_decoder_operation_inputs[12] = self.lm_head_indices_fake
            return self.acl_decoder_operation_inputs, self.acl_param

    def _init_rope_cos_sin(self, max_seq_len, dtype, device):
        if self.config.rope_scaling is None:
            self.rotary_embedding.update_cos_sin_cache_total(dtype,
                                                             device,
                                                             max_seq_len)

        else:
            scaling_type = self.config.rope_scaling["type"]
            if scaling_type == "linear":
                self.rotary_embedding.update_cos_sin_cache_total(dtype,
                                                                 device,
                                                                 max_seq_len)
            elif scaling_type == "dynamic":
                raise ValueError(f"not support RoPE scaling type {scaling_type}")
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

        self.cos_embed = self.rotary_embedding.get_cos_cached_total()
        self.sin_embed = self.rotary_embedding.get_sin_cached_total()