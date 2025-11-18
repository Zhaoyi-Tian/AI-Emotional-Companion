# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from base import model_test


class CohereModelTest(model_test.ModelTest):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.tokenizer_params = None

    @staticmethod  
    def get_chip_num():
        return 8
    
    @staticmethod
    def get_dataset_list():
        return ["CEval", "BoolQ"]

    def get_model(self, hardware_type, model_type, data_type):
        pass

    def set_fa_tokenizer_params(self):
        self.tokenizer_params = {
            'revision': None,
            'trust_remote_code': True
        }

    def get_supported_model_type(self):
        return ["cohere"]


def main():
    CohereModelTest.create_instance()


if __name__ == "__main__":
    main()
