# Copyright Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
from atb_llm.models.base.engine.engine_wrapper import EngineWrapper
from atb_llm.models.base.engine import BasePrefillEngineWrapper, BaseDecodeEngineWrapper, \
    SingleLoraPrefillEngineWrapper, SingleLoraDecodeEngineWrapper, MultiLoraPrefillEngineWrapper, \
    MultiLoraDecodeEngineWrapper, PagedAttentionPrefillEngineWrapper, PagedAttentionDecodeEngineWrapper
from atb_llm.models.base.base_lm import BaseLM
from atb_llm.utils.log import logger, print_log


class EngineManager:
    """A class manages all the `Engine` objects."""
    def __init__(self):
        self._engine_wrapper_list = []

    def register_engine(self, engine: EngineWrapper) -> None:
        """Register an engine by adding it into the object's member list.

        Args:
            engine (EngineWrapper): The engine to be registered.
        """
        self._engine_wrapper_list.append(engine)

    def choose_engine(self, context: BaseLM, **kwargs):
        """Choose an engine by looping through all the engines registed in the object's member list.
        Return the engine object of the first match.

        Args:
            context (BaseLM): All the necessary parameters needed to intialize the engine.
            kwargs: All the inference inputs needed to choose the engine.

        Returns:
            An inference engine instance.
        """
        for engine_wrapper in self._engine_wrapper_list:
            if engine_wrapper.activate(context, **kwargs):
                print_log(
                    context.mapping.rank,
                    logger.debug,
                    f"{type(engine_wrapper).__name__} is activated"
                )
                return engine_wrapper.engine
        return None

    def get_engines(self):
        """Returns all the registered engine objects.

        Yields:
            the engine member of the `EngineWrapper` object that is registered in this object's member list.
        """
        for engine_wrapper in self._engine_wrapper_list:
            yield engine_wrapper.engine

    def create_engines(self, context: BaseLM) -> None:
        """Create `EngineWrapper` objects by `context`.
        A multiple of `EngineWrapper` objects will be created based on parameters from `context`.

        Args:
            context (BaseLM): All the necessary parameters needed to intialize the engine.
        """
        # Pay attention to the order of engines
        # single feature
        if hasattr(context, "lora_decorator") and context.lora_decorator.active:
            self.register_engine(MultiLoraPrefillEngineWrapper(context))
            self.register_engine(MultiLoraDecodeEngineWrapper(context))
            self.register_engine(SingleLoraPrefillEngineWrapper(context))
            self.register_engine(SingleLoraDecodeEngineWrapper(context))
        
        if context.infer_param.inference_mode is not None and \
            context.infer_param.inference_mode.enable_prefill_pa:
            self.register_engine(PagedAttentionPrefillEngineWrapper(context))
        
        if context.infer_param.inference_mode is not None and \
            context.infer_param.inference_mode.enable_decode_pa:
            self.register_engine(PagedAttentionDecodeEngineWrapper(context))

        self.register_engine(BasePrefillEngineWrapper(context))
        self.register_engine(BaseDecodeEngineWrapper(context))

        # feature stacking


engine_manager = EngineManager()
