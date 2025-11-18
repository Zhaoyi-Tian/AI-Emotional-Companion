# Copyright Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
import json
from abc import abstractmethod

import torch

from atb_llm.models.base.base_lm import BaseLM


class EngineWrapper:
    """A wrapper for the inference engine.

    Attributes:
        engine: the inference engine.
    """
    def __init__(self, context: BaseLM, **kwargs):
        """Constructor

        Args:
            context (BaseLM): All the necessary parameters needed to intialize the engine.
            This argument will be passed to `create_engine`.
        """
        self.engine = self.create_engine(context, **kwargs)

    @classmethod
    def create_engine_cpp(
        cls,
        model_torch_class_name: str,
        engine_static_param: dict
    ):
        """A function that creates an engine built using the C++ programming language.

        Args:
            model_torch_class_name (str): The name of the C++ class responsible for creating the engine.
            engine_static_param (dict): The parameters used to intialize the C++ model class.

        Returns:
            An inference engine instance.
        """
        engine = torch.classes.ModelTorch.ModelTorch(model_torch_class_name)
        engine.set_param(json.dumps({**engine_static_param}))
        return engine

    @classmethod
    def create_engine_python(cls):
        """A function that creates an engine built using the Python programming language.

        Raises:
            NotImplementedError: This method is not implemented.
        """
        raise NotImplementedError

    @abstractmethod
    def create_engine(self, context: BaseLM, **kwargs):
        """Create an inference engine.

        Args:
            context (BaseLM): All the necessary parameters needed to intialize the engine.

        Returns:
            An inference engine instance.
        """
        pass

    @abstractmethod
    def activate(self, context: BaseLM, **kwargs) -> bool:
        """A function that determines whether to use the current `engine` for inference.

        Args:
            context (BaseLM): All the necessary parameters needed to choose the engine.
            kwargs: All the inference inputs needed to choose the engine.

        Returns:
            bool: A flag indicating whether the current `engine` is activated.
        """
        pass
