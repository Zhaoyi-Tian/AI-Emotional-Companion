# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

from dataclasses import dataclass

from ..base.router import BaseRouter


@dataclass
class MllamaRouter(BaseRouter):

    @staticmethod
    def check_config(config):
        BaseRouter.check_config(config.text_config)