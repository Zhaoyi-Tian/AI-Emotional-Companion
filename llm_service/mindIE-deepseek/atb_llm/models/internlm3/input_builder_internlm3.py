# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from atb_llm.models.internlm2.input_builder_internlm2 import Internlm2InputBuilder


class Internlm3InputBuilder(Internlm2InputBuilder):
    def __init__(self, tokenizer, model_version, generation_config, **kwargs):
        super().__init__(tokenizer, model_version, generation_config, **kwargs)
