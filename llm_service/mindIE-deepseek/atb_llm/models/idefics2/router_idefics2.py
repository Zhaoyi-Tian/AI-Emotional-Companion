# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from dataclasses import dataclass
from ..base.model_utils import safe_from_pretrained
from ..base.router import BaseRouter
from ..idefics2.flash_causal_idefics2 import Idefics2Config
from ..base.config import QuantizationConfig
from ..base.model_utils import safe_get_tokenizer_from_pretrained


@dataclass
class Idefics2Router(BaseRouter):
    def check_config_idefics2(self, config):
        super().check_config(config)
        attribute_ranges = {
            'mm_hidden_size': (1, 2147483647),
            'num_key_value_heads': (1, 2147483647),
        }
        for attr, (min_val, max_val) in attribute_ranges.items():
            if not hasattr(config, attr) or getattr(config, attr) is None:
                continue
            value = getattr(config, attr)
            if value < min_val or value > max_val:
                raise ValueError(f"self._config.{attr} must be between {min_val} and {max_val}")

    def get_config(self):
        config = safe_from_pretrained(Idefics2Config, self.model_name_or_path)
        setattr(config, 'quantization_config', QuantizationConfig(**{}))
        if self.max_position_embeddings:
            config.max_position_embeddings = self.max_position_embeddings
        self.check_config_idefics2(config)
        return config

    def get_tokenizer(self):
        use_fast = True
        return safe_get_tokenizer_from_pretrained(
            self.model_name_or_path,
            revision=self.revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=self.trust_remote_code,
            use_fast=use_fast
        )
