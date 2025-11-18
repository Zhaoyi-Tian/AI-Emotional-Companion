# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.
from dataclasses import dataclass
from ..base.router import BaseRouter
from .config_cohere import CohereConfig


@dataclass
class CohereRouter(BaseRouter):
    def get_config(self):
        config = CohereConfig.from_pretrained(self.model_name_or_path)
        if self.max_position_embeddings:
            config.max_position_embeddings = self.max_position_embeddings
        super().check_config(config)
        return config

    def get_tokenizer(self):
        tokenizer = super().get_tokenizer()
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        return tokenizer

