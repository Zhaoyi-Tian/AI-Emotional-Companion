# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from typing import Dict, List, Any
import importlib
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoConfig, AutoProcessor
from ..base.model_utils import safe_from_pretrained
from ..base.router import BaseRouter


@dataclass
class ClipRouter(BaseRouter):

    @property
    def image_processor(self):
        processor = self.processor
        if hasattr(processor, "image_processor"):
            return self.processor.image_processor
        elif hasattr(processor, "feature_extractor"):
            return self.processor.feature_extractor
        else:
            raise AttributeError("No image_processor attribute in processor")

    @property
    def processor(self):
        return self.get_processor()

    @staticmethod
    def mm_tokenization_func(processor: Any, tokenizer: Any, mm_inputs: List[Dict]):
        pass

    def get_model_cls(self):
        """
        Non-CausalLM model
        """
        if "clip" in self.model_type:
            tmp_model_type = "clip"
        model_file_dir_name = f"atb_llm.models.{tmp_model_type}."
        module_path = f"{model_file_dir_name}base_{tmp_model_type}"
        module = importlib.import_module(module_path)
        tmp_model_type_cap = tmp_model_type.capitalize()
        model_cls_name = f"{tmp_model_type_cap}Model"
        return getattr(module, model_cls_name)

    def get_config(self):
        config = safe_from_pretrained(AutoConfig, self.model_name_or_path)
        return config

    def get_processor(self):
        processor = safe_from_pretrained(AutoProcessor, self.model_name_or_path)
        return processor

    def get_tokenizer(self):
        tokenizer = safe_from_pretrained(AutoTokenizer, self.model_name_or_path)
        return tokenizer
