# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.

from collections import OrderedDict
from typing import Optional, List, Tuple
from einops import rearrange
import numpy as np

import torch
import torch_npu

from atb_llm.common_op_builders.linear_parallel.base_linear_parallel_common_op_builder import (
    CommunicationBackend,
)
from atb_llm.models.base.flash_causal_lm_atb import (
    AtbGraph,
    DECODE,
    PREFILL,
)
from atb_llm.models.cogvlm2.flash_causal_cogvlm2_atb import FlashCogvlm2ForCausalLMATB
from atb_llm.models.cogvlm2.modeling_cogvlm2 import CogvlmModel
from atb_llm.utils.layers import TensorEmbedding
from atb_llm.utils.log import logger
from atb_llm.utils.shm_utils import get_data_from_shm
from .config_cogvlm2_llama3_video import Cogvlm2Llama3VideoConfig
from ..cogvlm2.visual import EVA2CLIPModel


_NPU_FORMAT_CAST_TYPE = 29 # storage format: NZ
_LANGUAGE_INDICES = "language_indices"
_VISION_INDICES = "vision_indices"
_IMG_TOKEN_LEN = Cogvlm2Llama3VideoConfig.img_token_len
_PAD_TOKEN_ID = Cogvlm2Llama3VideoConfig.pad_token_id


class FlashCogvlm2llama3videoForCausalLMATB(FlashCogvlm2ForCausalLMATB): 
    def __init__(
        self, config, weights, lm_head_prefix="lm_head", model_prefix="model", **kwargs
    ):  
        
        super().__init__(config, weights, **kwargs)

        self.deltas = None
        if self.config.pe_type != "ROPE":
            logger.error(
                f"position_embedding_type is illegal, current value is: {self.config.pe_type}"
            )
        self.bos_token_id = config.bos_token_id
        self.image_token_id = config.pad_token_id
        self.prefill_with_vison_expert_graph = None
        self.weight = weights
        self.model_prefix = model_prefix
        self.lm_head_prefix = f"{model_prefix}.{lm_head_prefix}"
        self.rope_keep_local_base_windows = config.rope_keep_local_base_windows
        self.rope_vanilla_theta = config.rope_vanilla_theta
        self.rope_mscale = config.rope_mscale
        self.rope_given_inv_feq_str = config.rope_given_inv_feq_str
        self.dtype = config.torch_dtype
        self.with_vision_expert = True
        if self.config.architectures[0] == "CogVLMVideoForCausalLM":
            self.with_vision_expert = False

        # for vision language feature mixing
        self.embed_tokens = TensorEmbedding(
            prefix="model.embed_tokens", weights=weights
        )
        for p in self.embed_tokens.parameters():
            p.requires_grad = False

        backend = (
            CommunicationBackend.HCCL
            if self.soc_info.need_nz
            else CommunicationBackend.LCCL
        )
        self.model = CogvlmModel(config, weights, model_prefix, lm_head_prefix, backend, self.with_vision_expert)
        self.init_vit()

    @property
    def name(self):
        return "cogvlm2"

    @staticmethod
    def init_visiontowerweight(module, weights):
        vision_weights = [vision_weight for vision_weight in module.state_dict().keys()]
        for vision_weight in vision_weights:
            saved_weight = torch.nn.Parameter(
                    weights.get_tensor(f"model.vision.{vision_weight}"),
                    requires_grad=False
                )
            vision_weight_list = vision_weight.split(".")
            target_module = module
            for nxt_module in vision_weight_list[:-1]:
                target_module = getattr(target_module, nxt_module)
            setattr(target_module, vision_weight_list[-1], saved_weight)

    
    def init_model(self, config, weights, model_prefix, lm_head_prefix, backend):
        self.model = CogvlmModel(config, weights, model_prefix, lm_head_prefix, backend, False)
    
    def init_vit(self):
        self.vision_tower = EVA2CLIPModel(self.config, self.weight)
        self.vision_tower = self.vision_tower.to(self.dtype).npu()
        self.vision_tower.transformer.init_graph()
        self.vision_tower.linear_proj.init_graph()

    def get_weights(self):
        weights_dict = OrderedDict()
        weights_dict.update(self.model.get_weights(self.model_prefix))
        return weights_dict

    def get_in_tensor_names(self, is_prefill, with_vision_expert):
        default_input = ["hidden_states", "position_ids", "slots_mapping", "seq_len"]
        default_input.extend(["cos_table", "sin_table"])
        if is_prefill:
            default_input.extend(["attention_mask", "lm_head_indices"])
            if with_vision_expert:
                default_input.extend([_LANGUAGE_INDICES, _VISION_INDICES])
        else:
            default_input.extend(["block_tables"])
        return default_input

    def get_out_tensor_names(self):
        return ["model_out"]

    def build_graph(self, graph, is_prefill, with_vision_expert):
        # 设置输入输出
        kv_cache_names = []
        for i in range(self.config.num_hidden_layers):
            kv_cache_names.extend([f"layer_{i}_k_cache", f"layer_{i}_v_cache"])
        graph.add_input_output(
            input=list(self.weight.keys())
            + kv_cache_names
            + self.get_in_tensor_names(is_prefill, with_vision_expert),
            output=self.get_out_tensor_names(),
        )

        # 增加图节点
        self.model.build_graph(graph, is_prefill, with_vision_expert)
        graph.execute_as_single = False
        graph.build()

    def init_graph(self):
        # 获取权重键值对
        self.weight = self.get_weights()
        # 创建atb graph
        self.prefill_graph = AtbGraph(f"{self.name}_prefill_graph")
        self.build_graph(self.prefill_graph, is_prefill=True, with_vision_expert=self.with_vision_expert)
        if self.with_vision_expert:
            self.prefill_with_vison_expert_graph = AtbGraph(
                f"{self.name}_prefill_with_vision_expert_graph"
            )
            self.build_graph(
                self.prefill_with_vison_expert_graph,
                is_prefill=True,
                with_vision_expert=True,
            )
        else:
            self.decode_graph = AtbGraph(f"{self.name}_decode_graph")
            self.build_graph(self.decode_graph, is_prefill=False, with_vision_expert=False)
        
    def init_kvcache(self, kv_cache):
        kcache_id = not self.kcache_id or self.kcache_id != id(kv_cache[0][0])
        vcache_id = not self.vcache_id or self.vcache_id != id(kv_cache[0][1])
        if kcache_id or vcache_id:
            k_caches, v_caches = map(list, zip(*kv_cache))
            if self.soc_info.need_nz:
                k_caches = [
                    torch_npu.npu_format_cast_(k_cache, _NPU_FORMAT_CAST_TYPE)
                    for k_cache in k_caches
                ]
                v_caches = [
                    torch_npu.npu_format_cast_(v_cache, _NPU_FORMAT_CAST_TYPE)
                    for v_cache in v_caches
                ]
                logger.info(
                    f"<<<<<<<after transdata k_caches[0].shape={k_caches[0].shape}"
                )
            for i, (k_cache, v_cache) in enumerate(zip(k_caches, v_caches)):
                k_cache_name = f"layer_{i}_k_cache"
                v_cache_name = f"layer_{i}_v_cache"
                self.weight.update({k_cache_name: k_cache, v_cache_name: v_cache})
            self.prefill_graph.set_weights(self.weight)
            if self.with_vision_expert:
                self.prefill_with_vison_expert_graph.set_weights(self.weight)
            self.decode_graph.set_weights(self.weight)
            self.kcache_id = id(kv_cache[0][0])
            self.vcache_id = id(kv_cache[0][1])

    def init_cos_sin_table(self, max_seq_len, dim, dtype, device):
        if self.rope_given_inv_feq_str is None and self.rope_vanilla_theta is None:
            self._init_rope_cos_sin(max_seq_len, dtype, device)
        else:
            self.cos_embed, self.sin_embed = self._get_cos_sin_table(
                max_seq_len,
                dim,
                dtype,
                device,
                0,
                self.rope_mscale,
                self.rope_keep_local_base_windows,
                self.rope_theta,
                self.rope_vanilla_theta,
                self.rope_given_inv_feq_str,
            )


    def get_language_vision_indices(self, input_ids):
        language_idx = []
        vision_idx = []
        bos_idxs = torch.where(torch.eq(input_ids, self.bos_token_id))[0]
        for bos in bos_idxs:
            next_bos = (bos_idxs[bos_idxs > bos]).min() if (bos_idxs > bos).any() else input_ids.size(0)
            language_idx.append(bos)
            if torch.any(torch.eq(input_ids[bos: next_bos], self.image_token_id)):
                image_range = list(range(bos + 1, bos + _IMG_TOKEN_LEN))
                vision_idx.extend(image_range)
                text_range = list(range(bos + _IMG_TOKEN_LEN, next_bos))
            else:
                text_range = list(range(bos + 1, next_bos))
            language_idx.extend(text_range)

        language_indices = torch.LongTensor(language_idx).to(self.device)
        vision_indices = torch.LongTensor(vision_idx).to(self.device)
        return language_indices, vision_indices

    def prepare_prefill_token_service(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)
        bos_idxs = torch.where(torch.eq(input_ids, self.bos_token_id))[0]
        for _, bos in enumerate(bos_idxs):
            next_bos = (bos_idxs[bos_idxs > bos]).min() if (bos_idxs > bos).any() else input_ids.size(0)
            if torch.any(torch.eq(input_ids[bos: next_bos], self.image_token_id)):

                img_pos = torch.where(torch.eq(input_ids[bos: next_bos], self.image_token_id))[0]
                pixel_array = []
                shm_value = input_ids[img_pos[0] + bos + 1]
                shape_value = input_ids[img_pos[0] + bos + 2]
                shared_tensor = get_data_from_shm(shm_value, shape_value, np.float32, self.device)
                pixel_array.append(shared_tensor)
                pixel_array = torch.cat(pixel_array, dim=0)
                image_outputs = self.vision_tower(pixel_array.to(self.dtype))
                frame_numbers = image_outputs.size(0)

                images_features = rearrange(image_outputs, 'b n d -> (b n) d')
                images_features = images_features.to(dtype=hidden_states.dtype, device=hidden_states.device)

                hidden_states = self._merge_input_ids_with_video_features_service(images_features,
                                                                        hidden_states,
                                                                        bos,
                                                                        frame_numbers)
        return hidden_states

    def prepare_inputs(
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
        # 准备输入
        attention_mask = kwargs.get("attention_mask", None)
        with_vision_expert = kwargs.get("with_vision_expert", False)

        if is_prefill:
            attention_mask = self.attn_mask.get_attn_mask(
                self.max_base_len, self.dtype, self.device
            )

        if self.soc_info.need_nz and attention_mask is not None:
            attention_mask = self.transdata_operation.execute([attention_mask])[0]
        if is_prefill:
            self.init_cos_sin_table(
                self.max_position_embeddings, self.head_size, self.dtype, self.device
            )
        # 更新输入
        target_key = PREFILL if is_prefill else DECODE
        # 拼接图片
        hidden_states = self.prepare_prefill_token_service(input_ids)
        
        self.graph_inputs[target_key].update(
            {
                "hidden_states": hidden_states,
                "position_ids": position_ids.to(torch.int64),
                "slots_mapping": slots.to(torch.int32),
                "seq_len": input_lengths.to(torch.int32),
                "cos_table": self.cos_embed,
                "sin_table": self.sin_embed,
            }
        )
        if attention_mask is not None:  # attention mask
            self.graph_inputs[target_key]["attention_mask"] = attention_mask
        if is_prefill and lm_head_indices is None:  # lm head indices
            lm_head_indices = torch.tensor(
                range(input_ids.shape[0]), dtype=torch.int64, device=input_ids.device
            )
        if is_prefill:
            self.graph_inputs[target_key]["lm_head_indices"] = lm_head_indices.to(torch.int32)
        else:  # decode
            self.graph_inputs[target_key]["block_tables"] = block_tables.to(torch.int32)

        if with_vision_expert:
            (
                self.graph_inputs[target_key][_LANGUAGE_INDICES],
                self.graph_inputs[target_key][_VISION_INDICES],
            ) = self.get_language_vision_indices(input_ids)
        else:
            if _LANGUAGE_INDICES in self.graph_inputs[target_key].keys():
                del self.graph_inputs[target_key][_LANGUAGE_INDICES]
            if _VISION_INDICES in self.graph_inputs[target_key].keys():
                del self.graph_inputs[target_key][_VISION_INDICES]

        # 准备输出
        real_vocab_size = (
            self.weight.get(f"{self.lm_head_prefix}.weight").shape[0]
            * self.tp_world_size
        )
        batch_size = lm_head_indices.shape[0] if is_prefill else input_ids.shape[0]

        self.graph_outputs[target_key][self.get_out_tensor_names()[0]] = torch.empty(
            batch_size, real_vocab_size, dtype=self.dtype, device=self.device
        )

        # 准备bind tensor
        self.graph_param[target_key]["seq_len"] = input_lengths.cpu().to(torch.int32)

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
    ) -> torch.Tensor:
        self.init_kvcache(kv_cache)
        with_vision_expert = False

        self.prepare_inputs(
            input_ids,
            position_ids,
            is_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_seq_len,
            lm_head_indices,
            with_vision_expert=with_vision_expert,
        )
        if is_prefill:
            if with_vision_expert:
                atb_model_out = self.prefill_with_vison_expert_graph.forward(
                    self.graph_inputs[PREFILL],
                    self.graph_outputs[PREFILL],
                    self.graph_param[PREFILL],
                )
            else:
                atb_model_out = self.prefill_graph.forward(
                    self.graph_inputs[PREFILL],
                    self.graph_outputs[PREFILL],
                    self.graph_param[PREFILL],
                )
        else:
            atb_model_out = self.decode_graph.forward(
                self.graph_inputs[DECODE],
                self.graph_outputs[DECODE],
                self.graph_param[DECODE],
            )

        logits = atb_model_out[self.get_out_tensor_names()[0]]
        return logits

    def _merge_input_ids_with_video_features_service(self, images_features, hidden_states, bos, frames):
        for i in range(frames):
            hidden_states[bos + 1 + i + _IMG_TOKEN_LEN * i : bos + 1 + i + _IMG_TOKEN_LEN * (i + 1)] = \
             images_features[_IMG_TOKEN_LEN * i : _IMG_TOKEN_LEN * (i + 1)]
        return hidden_states

    def _init_rope_cos_sin(self, max_seq_len, dtype, device):
        if self.config.rope_scaling is None:
            self.rotary_embedding.update_cos_sin_cache_total(dtype, device, max_seq_len)

        else:
            scaling_type = self.config.rope_scaling.type
            if scaling_type == "linear":
                self.rotary_embedding.update_cos_sin_cache_total(
                    dtype, device, max_seq_len
                )
            elif scaling_type == "dynamic":
                raise ValueError(f"not support RoPE scaling type {scaling_type}")
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

        self.cos_embed = self.rotary_embedding.get_cos_cached_total()
        self.sin_embed = self.rotary_embedding.get_sin_cached_total()
