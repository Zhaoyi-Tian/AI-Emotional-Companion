# Copyright Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
import torch

from atb_llm.models.base.base_lm import BaseLM
from atb_llm.models.base.engine.engine_wrapper import EngineWrapper


class PagedAttentionDecodeEngineWrapper(EngineWrapper):
    def create_engine(self, context: BaseLM, **kwargs):
        if context.infer_param.enable_python_engine:
            raise NotImplementedError
        else:
            engine = super().create_engine_cpp(
                context.model_torch_class_name,
                {**context.engine_static_param, "isPrefill": False, "enableSpeculate": True,
                "skipWordEmbedding": context.infer_param.skip_word_embedding}
            )
        return engine

    def activate(self, context: BaseLM, **kwargs) -> bool:
        pa_enable = context.infer_param.inference_mode.enable_decode_pa
        activate_flag = False
        q_lens = kwargs.get("q_lens", None)
        is_prefill = kwargs.get("is_prefill", False)
        if q_lens is not None and not is_prefill and pa_enable:
            activate_flag = True
        return activate_flag
