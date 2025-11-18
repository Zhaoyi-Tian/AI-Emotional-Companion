# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import argparse
import torch
from transformers import AutoModelForVision2Seq
from atb_llm.models.base.model_utils import safe_from_pretrained


def convert_to_float16(config_path, target_config_path):
    model = safe_from_pretrained(AutoModelForVision2Seq, config_path)
    model = model.to(torch.float16)
    model.save_pretrained(target_config_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default='model_dir/siglip-so400m-patch14-384')
    parser.add_argument("--target_config_path", type=str, default='model_dir/siglip-so400m-patch14-384-float16')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    convert_to_float16(args.config_path, args.target_config_path)


if __name__ == "__main__":
    main()
