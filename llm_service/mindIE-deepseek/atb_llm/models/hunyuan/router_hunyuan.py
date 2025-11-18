# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from dataclasses import dataclass
from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode
from ..base.router import BaseRouter
from .config_hunyuan import HunyuanConfig
from ..base.model_utils import safe_get_tokenizer_from_pretrained


@dataclass
class HunyuanRouter(BaseRouter):
    def get_config(self):
        config = HunyuanConfig.from_pretrained(self.model_name_or_path)
        self.check_config_hunyuan(config)
        return config

    def get_tokenizer(self):
        tokenizer = safe_get_tokenizer_from_pretrained(
            self.tokenizer_path,
            use_fast=False,
            trust_remote_code=self.trust_remote_code,
        )
        return tokenizer

    def check_config_hunyuan(self, config) -> None:
        super().check_config(config)

        attribute_ranges = {
            "num_key_value_heads" : (1, 2147483647),
            "n_routed_experts" : (1, 2147483647),
            "num_experts_per_tok" : (1, config.n_routed_experts),
            "n_shared_experts": (0, 2147483647),
            "cla_share_factor" : (1, config.num_hidden_layers),
            "vocab_size" : (0, 2147483647)
        }
        for attr, (min_val, max_val) in attribute_ranges.items():
            if not hasattr(config, attr) or getattr(config, attr) is None:
                msg = f"{attr} is not found in config"
                logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
                raise AssertionError
            value = getattr(config, attr)
            if value < min_val or value > max_val:
                msg = f"self._config.{attr} must be between {min_val} and {max_val}"
                logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
                raise AssertionError
                   
        # 校验rope_scaling参数数目，需要等于3
        if not isinstance(config.rope_scaling_dict, dict) or len(config.rope_scaling_dict) != 3:
            logger.error("`rope_scaling` must be a dictionary with with 3 fields, "
                f"got {len(config.rope_scaling_dict)}", ErrorCode.ATB_MODELS_PARAM_INVALID)
            raise AssertionError

        rope_scaling_type = config.rope_scaling_dict.get("type", None)
        if rope_scaling_type is None or rope_scaling_type not in ["dynamic"]:
            msg = f"`rope_scaling`'s type field must be one of ['dynamic'], got {rope_scaling_type}"
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise AssertionError