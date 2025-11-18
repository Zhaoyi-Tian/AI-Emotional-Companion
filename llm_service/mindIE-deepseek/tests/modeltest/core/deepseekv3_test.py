# Copyright Huawei Technologies Co., Ltd. 2023-2025. All rights reserved.
from core.deepseekv2_test import DeepseekV2ModelTest


class DeepseekV3ModelTest(DeepseekV2ModelTest):
    def __init__(self, *args) -> None:
        model_name = "deepseek_v3"
        updated_args = args[:3] + (model_name,) + args[4:]
        super().__init__(*updated_args)

    def get_supported_model_type(self):
        return ["deepseek_v3"]


def main():
    DeepseekV3ModelTest.create_instance()

if __name__ == "__main__":
    main()