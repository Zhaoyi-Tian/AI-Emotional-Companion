# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from dataclasses import dataclass
from ..base.router import BaseRouter
from ..base.model_utils import safe_get_tokenizer_from_pretrained
from .v3.config_internlm3 import Internlm3Config
from .input_builder_internlm3 import Internlm3InputBuilder


@dataclass
class Internlm3Router(BaseRouter):

    @property
    def model_version(self):
        """
        次级模型名称
        :return:
        """
        return "v3"

    def get_config(self):
        config = Internlm3Config.from_dict(self.config_dict)
        super().check_config(config)
        return config

    def get_tokenizer(self):
        tokenizer = safe_get_tokenizer_from_pretrained(
            self.tokenizer_path,
            revision=self.revision,
            padding_side="left",
            trust_remote_code=self.trust_remote_code,
            use_fast=False,
        )
        return tokenizer

    def get_input_builder(self):
        return Internlm3InputBuilder(self.tokenizer, self.model_version, self.generation_config)