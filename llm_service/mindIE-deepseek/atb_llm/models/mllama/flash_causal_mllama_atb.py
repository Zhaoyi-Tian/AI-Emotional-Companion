# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from typing import Optional, List, Tuple

import torch
import torch_npu
from .modeling_mllama_vit_atb import MllamaVisionModelATB
from .modeling_mllama_atb import MllamaModelATB
from ..base.flash_causal_lm_atb import FlashForCausalLMATB, AtbGraph, PREFILL, DECODE
from ..llama.flash_causal_llama_atb import FlashLlamaForCausalLMATB
from ...utils.layers import load_column_multi, TensorHead
from ...utils.log import logger
from ...common_op_builders.linear_parallel.base_linear_parallel_common_op_builder import CommunicationBackend


_CROSS_CONTEXT_LENS = "cross_context_lens"


class FlashMllamaForCausalLMATB(FlashLlamaForCausalLMATB):
    def __init__(self, config, weights, lm_head_prefix="language_model.lm_head",
                 model_prefix="language_model.model", **kwargs):
        if getattr(config, "text_config"):
            FlashForCausalLMATB.__init__(self, config.text_config, weights, **kwargs)
        else:
            raise KeyError("Mllama Config does not have attribute 'text_config'")
        
        self.backend = CommunicationBackend.HCCL if self.soc_info.need_nz else CommunicationBackend.LCCL
        # vision_model 初始化
        self.vision_model = MllamaVisionModelATB(
            config.vision_config,
            weights,
            'vision_model',
            device=self.device,
            dtype=self.dtype,
            backend=self.backend,
            soc_info=self.soc_info
        )
        self.init_weights(
            self.vision_model,
            weights,
            'vision_model',
            exclude_weight_list=['transformer', 'global_transformer']
        )

        # multi_modal_projector 初始化
        self.multi_modal_projector = torch.nn.Linear(
            config.vision_config.vision_output_dim,
            config.text_config.hidden_size,
            bias=True,
        )
        self.init_weights(self.multi_modal_projector, weights, 'multi_modal_projector')

        self.vision_model.eval()
        self.vision_model.requires_grad_(False)
        self.multi_modal_projector.eval()
        self.multi_modal_projector.requires_grad_(False)

        # config处理
        config.text_config.image_token_index = config.image_token_index        
        self.config = config.text_config
        self.vision_config = config.vision_config
        config = self.config

        # 模型结构相关
        self.model_prefix = model_prefix
        self.model = MllamaModelATB(config, weights, model_prefix, lm_head_prefix, 
                                is_fa=False, speculate_enable=self.speculate_enable, backend=self.backend)
        self.final_norm_prefix = f"{model_prefix}.norm"
        self.lm_head_prefix = lm_head_prefix
        if self.quantize == "w8a8sc":
            self.lm_head = TensorHead.load_weight(
                config,
                prefix=lm_head_prefix,
                weights=weights,
                is_norm=False,
            )
        else:
            self.lm_head = load_column_multi(
                config,
                prefixes=[lm_head_prefix],
                weights=weights,
                head_size=1,
                lm_head=True,
            )

        self.position_embedding_type = config.pe_type
        self.alibi_bias_max = config.alibi_bias_max
        self.rope_keep_local_base_windows = config.rope_keep_local_base_windows
        self.rope_vanilla_theta = config.rope_vanilla_theta
        self.rope_mscale = config.rope_mscale
        self.rope_given_inv_feq_str = config.rope_given_inv_feq_str
        self.atten_mask_cpu = None
        self.alibi_mask_compress = True
        self.skip_word_embedding = False
        if self.position_embedding_type != "ROPE":
            logger.error("error: only support petype: ROPE, check your config.json: pe_type")
            raise AssertionError(f'petype: {self.position_embedding_type} not supported')
        self.wins_batch_1 = None
        self.decoder_slots = None
        self.all_wins_batch = None
        self.block_tables_global = None
        self.wins_global = None

    @property
    def name(self):
        return "mllama"

    def init_weights(self, module, weights, weight_prefix, exclude_weight_list=None):
        exclude_weight_list = exclude_weight_list or []
        module_weights = []
        for module_weight in module.state_dict().keys():
            is_excluded = False
            for exclude in exclude_weight_list:
                if module_weight.startswith(exclude):
                    is_excluded = True
                    break
            if not is_excluded:
                module_weights.append(module_weight)

        for module_weight in module_weights:
            saved_weight = torch.nn.Parameter(
                    weights.get_tensor(f"{weight_prefix}.{module_weight}"),
                    requires_grad=False
                )
            module_weight_names = module_weight.split(".")
            target_module = module
            for nxt_module_name in module_weight_names[:-1]:
                target_module = getattr(target_module, nxt_module_name)
            setattr(target_module, module_weight_names[-1], saved_weight)

    def init_graph(self):
        # 获取权重键值对
        self.weight = self.get_weights()
        # 创建atb graph
        self.vision_model.build_graph()

        self.prefill_graph = AtbGraph(f"{self.name}_prefill_graph")
        self.build_graph(self.prefill_graph, is_prefill=True, is_multimodal=True)
        self.decode_graph = AtbGraph(f"{self.name}_decode_graph")
        self.build_graph(self.decode_graph, is_prefill=False, is_multimodal=True)

        self.prefill_text_graph = AtbGraph(f"{self.name}_prefill_text_graph")
        self.build_graph(self.prefill_text_graph, is_prefill=True, is_multimodal=False)
        self.decode_text_graph = AtbGraph(f"{self.name}_decode_text_graph")
        self.build_graph(self.decode_text_graph, is_prefill=False, is_multimodal=False)


    def get_in_tensor_names(self, is_prefill, is_multimodal):
        default_input = ['input_ids', 'position_ids', 'slots_mapping', 'seq_len', 'block_tables']
        if is_multimodal:
            default_input.extend(['cross_attention_mask', 'full_text_row_masked_out_mask', 'cross_context_lens'])
        if is_prefill and is_multimodal:
            default_input.extend(['cross_attention_states', 'cross_slots_mapping'])
        if self.config.pe_type == "ROPE":
            default_input.extend(['cos_table', 'sin_table'])
        if is_prefill:
            default_input.extend(['attention_mask', 'lm_head_indices'])
        if not is_prefill and self.speculate_enable:
            default_input.extend(['attention_mask', 'q_len'])
        return default_input

    def get_out_tensor_names(self):
        return ['model_out']

    def build_graph(self, graph, is_prefill, is_multimodal):
        # 设置输入输出
        kv_cache_names = []
        for i in range(self.config.num_hidden_layers):
            kv_cache_names.extend([f"layer_{i}_k_cache", f"layer_{i}_v_cache"])
        graph.add_input_output(
            input=list(self.weight.keys()) + kv_cache_names + self.get_in_tensor_names(is_prefill, is_multimodal),
            output=self.get_out_tensor_names())

        # 增加图节点
        self.model.build_graph(graph, is_prefill, is_multimodal)
        self.build_lm_head(graph, is_prefill)

        # 构图
        graph.execute_as_single = False
        graph.build()

    def init_kvcache(self, kv_cache):
        kcache_id_for_text_graph = not self.kcache_id or self.kcache_id != id(kv_cache[0][0])
        vcache_id_for_text_graph = not self.vcache_id or self.vcache_id != id(kv_cache[0][1])

        super().init_kvcache(kv_cache)

        if kcache_id_for_text_graph or vcache_id_for_text_graph:
            self.prefill_text_graph.set_weights(self.weight)
            self.decode_text_graph.set_weights(self.weight)

    def prepare_inputs(
            self, input_ids: torch.Tensor,
            position_ids: torch.Tensor,
            is_prefill: bool,
            kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
            block_tables: torch.Tensor,
            slots: torch.Tensor,
            input_lengths: torch.Tensor,
            max_seq_len: int,
            lm_head_indices: Optional[torch.Tensor],
            is_multimodal: bool,
            **kwargs
    ):
        # 准备输入
        multi_modal_inputs = kwargs.get('multi_modal_inputs', None)
        cross_slots_mapping = kwargs.get('cross_slots_mapping', None)
        cross_attention_mask = kwargs.get('cross_attention_mask', None)
        cross_context_lens = kwargs.get('cross_context_lens', None)
        full_text_row_masked_out_mask = kwargs.get('full_text_row_masked_out_mask', None)
        cross_attention_states = self.forward_vision_model(multi_modal_inputs) if multi_modal_inputs else None

        if is_multimodal and is_prefill:
            block_tables = torch.cat([block_tables[i].unsqueeze(0).repeat(len, 1) 
                                      for i, len in enumerate(input_lengths.tolist())], dim=0)
            
        # q lens
        q_lens = kwargs.get('q_lens', [])
        # attention mask
        attention_mask = kwargs.get('attention_mask', None)
        spec_mask = kwargs.get('spec_mask', None)

        if self.position_embedding_type == "ROPE":
            if is_prefill:
                attention_mask = self.attn_mask.get_attn_mask(self.max_base_len, self.dtype, self.device)
            elif self.speculate_enable:
                attention_mask = spec_mask
        else:
            logger.error(f"position_embedding_type is inllegal {self.position_embedding_type}")
        if self.soc_info.need_nz:
            if attention_mask is not None:
                attention_mask = self.transdata_operation.execute([attention_mask])[0]
            if cross_attention_mask is not None:
                torch.npu.synchronize()
                cross_attention_mask = self.transdata_operation.execute([cross_attention_mask])[0]
        # cosine & sine embedding
        if is_prefill:
            self.init_cos_sin_table(self.max_position_embeddings, self.head_size, self.dtype, self.device)

        # 更新输入
        target_key = PREFILL if is_prefill else DECODE
        self.graph_inputs[target_key].update({
            "input_ids": input_ids,
            "position_ids": position_ids.to(torch.int64),
            "slots_mapping": slots.to(torch.int32),
            "seq_len": input_lengths.to(torch.int32),
            "block_tables":block_tables.to(torch.int32),
        })

        if is_multimodal:
            self.graph_inputs[target_key].update({
                "cross_attention_mask": cross_attention_mask,
                "full_text_row_masked_out_mask":full_text_row_masked_out_mask,
                _CROSS_CONTEXT_LENS: cross_context_lens.to(torch.int32)
            })

        if attention_mask is not None:  # attention mask
            self.graph_inputs[target_key]["attention_mask"] = attention_mask
        if self.position_embedding_type == "ROPE":  # cosine & sine embedding
            self.graph_inputs[target_key]["cos_table"] = self.cos_embed
            self.graph_inputs[target_key]["sin_table"] = self.sin_embed
        if is_prefill and lm_head_indices is None:  # lm head indices
            lm_head_indices = torch.tensor(range(input_ids.shape[0]),
                                           dtype=torch.int64, device=input_ids.device)
        if is_prefill:
            self.graph_inputs[target_key]["lm_head_indices"] = lm_head_indices
            if is_multimodal:
                self.graph_inputs[target_key]["cross_attention_states"] = cross_attention_states
                self.graph_inputs[target_key]["cross_slots_mapping"] = cross_slots_mapping.to(torch.int32)
        else:  # decode
            if self.speculate_enable:
                q_len = torch.tensor(q_lens, dtype=torch.int32, device=self.device)
                self.graph_inputs[target_key]["q_len"] = q_len

        # 准备输出
        real_vocab_size = self.weight.get(f"{self.lm_head_prefix}.weight").shape[0] * self.tp_world_size
        batch_size = lm_head_indices.shape[0] if is_prefill else input_ids.shape[0]

        self.graph_outputs[target_key][self.get_out_tensor_names()[0]] = \
            torch.empty(batch_size, real_vocab_size, dtype=self.dtype, device=self.device)

        # 准备bind tensor
        self.graph_param[target_key]['seq_len'] = input_lengths.cpu().to(torch.int32)
        if is_multimodal:
            self.graph_param[target_key][_CROSS_CONTEXT_LENS] = cross_context_lens.cpu().to(torch.int32)

        if self.speculate_enable and not is_prefill:
            self.graph_param[target_key]['q_len'] = q_len.cpu()

    def prepare_prefill_token(self, text, image=None, video=None, processor=None):
        if video:
            message = "mllama3.2 does not support video input"
            logger.error(message)
            raise RuntimeError(message)

        if not image:
            inputs = processor(text=text, return_tensors="pt")
            inputs.to(self.device)
            return inputs['input_ids'].view(-1), None
        
        inputs = processor(image, text, add_special_tokens=False, return_tensors="pt")

        cross_attention_mask = inputs.pop('cross_attention_mask')
        batch_size, text_total_length, *_ = cross_attention_mask.shape
        cross_attention_mask = cross_attention_mask.view(batch_size * text_total_length, -1)
        full_text_row_masked_out_mask = (
            (cross_attention_mask != 0).any(dim=-1).type(self.dtype)[..., None]
        )

        input_ids = inputs.pop('input_ids').view(-1)

        multi_modal_inputs = dict(
            pixel_values=inputs['pixel_values'],
            aspect_ratio_mask=inputs['aspect_ratio_mask'],
            aspect_ratio_ids=inputs['aspect_ratio_ids'],
            cross_attention_mask=cross_attention_mask,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            num_vision_tokens=self.vision_model.num_patches,
        )
        return input_ids, multi_modal_inputs

    def prepare_cross_attention_status(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        aspect_ratio_mask: Optional[torch.Tensor] = None,
        aspect_ratio_ids: Optional[torch.Tensor] = None,
    ):
        if pixel_values is not None:
            if aspect_ratio_ids is None:
                message = "`aspect_ratio_ids` must be provided if `pixel_values` is provided"
                logger.error(message)
                raise ValueError(message)
            # get vision tokens from vision model
            vision_outputs = self.vision_model(
                pixel_values=pixel_values,
                aspect_ratio_ids=aspect_ratio_ids,
                aspect_ratio_mask=aspect_ratio_mask,
            )
            cross_attention_states = vision_outputs[0]
            cross_attention_states = self.multi_modal_projector(cross_attention_states)

        return cross_attention_states

    def forward_vision_model(self, multi_modal_inputs):
        pixel_values = []
        aspect_ratio_mask = []
        aspect_ratio_ids = []
        
        for data in multi_modal_inputs:
            pixel_values.append(data['pixel_values'])
            aspect_ratio_mask.append(data['aspect_ratio_mask'])
            aspect_ratio_ids.append(data['aspect_ratio_ids'])
            
        pixel_values = torch.concat(pixel_values, dim=0).npu()
        aspect_ratio_mask = torch.concat(aspect_ratio_mask, dim=0).npu()
        aspect_ratio_ids = torch.concat(aspect_ratio_ids, dim=0).npu()

        with torch.no_grad():
            cross_attention_states = self.prepare_cross_attention_status(
                pixel_values=pixel_values,
                aspect_ratio_mask=aspect_ratio_mask,
                aspect_ratio_ids=aspect_ratio_ids,
            )
            cross_attention_states = cross_attention_states.reshape(-1, self.hidden_size)

        return cross_attention_states

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
        is_multimodal = True if kwargs.get('cross_attention_mask', None) is not None else False
        self.init_kvcache(kv_cache)
        self.prepare_inputs(input_ids, position_ids, is_prefill, kv_cache, \
                            block_tables, slots, input_lengths, max_seq_len, lm_head_indices, \
                            is_multimodal, **kwargs)
        
        prefill_graph = self.prefill_graph if is_multimodal else self.prefill_text_graph
        decode_graph = self.decode_graph if is_multimodal else self.decode_text_graph

        if is_prefill:
            atb_model_out = prefill_graph.forward(self.graph_inputs[PREFILL], self.graph_outputs[PREFILL],
                                                       self.graph_param[PREFILL])
        else:
            atb_model_out = decode_graph.forward(self.graph_inputs[DECODE], self.graph_outputs[DECODE],
                                                      self.graph_param[DECODE])
        try:
            logits = atb_model_out[self.get_out_tensor_names()[0]]
        except IndexError as e:
            raise RuntimeError("An error occurs. Enable logs to further locate the problem") from e
        return logits
