# Copyright Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
import torch

from atb_llm.models.base.base_lm import BaseLM
from atb_llm.models.base.engine.engine_wrapper import EngineWrapper


class SingleLoraPrefillEngineWrapper(EngineWrapper):
    def create_engine(self, context: BaseLM, **kwargs):
        """Create an engine for single lora adapters, utilized in the prefill phase.
        For C++ engine, apart from default parameters passed through `context.engine_static_param`,
            the following key value pair will be updated: {"isPrefill": False,
            "skipWordEmbedding": context.infer_param.skip_word_embedding, "enableLora": True}.

        Args:
            context (BaseLM): All the necessary parameters needed to intialize the engine.

        Raises:
            NotImplementedError: if `context.infer_param.enable_python_engine` is true,
            an `NotImplementedError` will be raised. Python engine is not supported.

        Returns:
            An inference engine instance.
        """
        if context.infer_param.enable_python_engine:
            raise NotImplementedError
        else:
            engine = super().create_engine_cpp(
                context.model_torch_class_name,
                {**context.engine_static_param, "isPrefill": True,
                 "skipWordEmbedding": context.infer_param.skip_word_embedding, "enableLora": True}
            )
        return engine

    def activate(self, context: BaseLM, **kwargs) -> bool:
        """A function that determines whether to use the current `engine` for inference.
        This object is activated when `is_prefill` from `kwargs` is true and single adapter is enabled.

        Args:
            context (BaseLM): All the necessary parameters needed to intialize the engine.
            kwargs: All the inference inputs needed to choose the engine.

        Returns:
            bool: A flag indicating whether the current `engine` is activated.
        """
        is_prefill = kwargs.get("is_prefill", False)
        return is_prefill and context.lora_decorator.use_single_adapter()
