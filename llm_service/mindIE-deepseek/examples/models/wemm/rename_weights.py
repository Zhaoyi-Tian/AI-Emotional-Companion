# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
import shutil

from typing import List
import torch
from safetensors.torch import save_file, load_file

from atb_llm.utils.hub import weight_files
from atb_llm.utils.log import logger
from atb_llm.utils import file_utils
from atb_llm.utils.convert import _remove_duplicate_names

MAX_TOKENIZER_FILE_SIZE = 1024 * 1024 * 1024
INFIX_WEIGHT_NAME = "original_linear."


def copy_remaining_files(model_dir, dest_dir):
    model_dir = file_utils.standardize_path(model_dir, check_link=False)
    file_utils.check_path_permission(model_dir)
    if os.path.exists(dest_dir):
        dest_dir = file_utils.standardize_path(dest_dir, check_link=False)
        file_utils.check_path_permission(dest_dir)
    else:
        os.makedirs(dest_dir, exist_ok=True)
        dest_dir = file_utils.standardize_path(dest_dir, check_link=False)
    
    suffix = '.safetensors'
    for filename in file_utils.safe_listdir(model_dir):
        if not filename.endswith(suffix):
            src_filepath = os.path.join(model_dir, filename)
            src_filepath = file_utils.standardize_path(src_filepath, check_link=False)
            file_utils.check_file_safety(src_filepath, 'r', max_file_size=MAX_TOKENIZER_FILE_SIZE)
            dest_filepath = os.path.join(dest_dir, filename)
            dest_filepath = file_utils.standardize_path(dest_filepath, check_link=False)
            file_utils.check_file_safety(dest_filepath, 'w', max_file_size=MAX_TOKENIZER_FILE_SIZE)
            shutil.copyfile(src_filepath, dest_filepath)


def rename_safetensor_file(src_file: Path, dst_file: Path, discard_names: List[str]):
    src_file = file_utils.standardize_path(str(src_file), check_link=False)
    file_utils.check_file_safety(src_file, 'r', is_check_file_size=False)

    loaded_state_dict = load_file(src_file)
    if "state_dict" in loaded_state_dict:
        loaded_state_dict = loaded_state_dict["state_dict"]
    to_remove_dict = _remove_duplicate_names(loaded_state_dict, discard_names=discard_names)

    metadata = {"format": "pt"}
    for kept_name, to_remove_list in to_remove_dict.items():
        for to_remove in to_remove_list:
            if to_remove not in metadata:
                metadata[to_remove] = kept_name
            del loaded_state_dict[to_remove]
    
    renamed_loaded_state_dict = {}
    for k, v in loaded_state_dict.items():
        if INFIX_WEIGHT_NAME in k:
            k = k.replace(INFIX_WEIGHT_NAME, "")
        renamed_loaded_state_dict[k] = v.contiguous()

    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
    dst_file = file_utils.standardize_path(str(dst_file), check_link=False)
    file_utils.check_file_safety(dst_file, 'w', is_check_file_size=False)
    save_file(renamed_loaded_state_dict, dst_file, metadata=metadata)

    reloaded_state_dict = load_file(dst_file)
    for k, pt_tensor in loaded_state_dict.items():
        k = k.replace(INFIX_WEIGHT_NAME, "")
        sf_tensor = reloaded_state_dict[k]
        if not torch.equal(pt_tensor, sf_tensor):
            raise RuntimeError(f"The output tensors do not match for key {k}")


def rename_safetensor_files(src_files: List[Path], dst_files: List[Path], discard_names: List[str]):
    num_src_files = len(src_files)

    for i, (src_file, dst_file) in enumerate(zip(src_files, dst_files)):
        blacklisted_keywords = ["arguments", "args", "training"]
        if any(substring in src_file.name for substring in blacklisted_keywords):
            continue

        start_time = datetime.now(tz=timezone.utc)
        rename_safetensor_file(src_file, dst_file, discard_names)
        elapsed_time = datetime.now(tz=timezone.utc) - start_time
        try:
            logger.info(f"Rename: [{i + 1}/{num_src_files}] -- Took: {elapsed_time}")
        except ZeroDivisionError as e:
            raise ZeroDivisionError from e


def rename_weights(model_path, save_directory):
    local_src_files = weight_files(model_path, extension=".safetensors")
    local_dst_files = [
        Path(save_directory) / f"{s.stem}.safetensors"
        for s in local_src_files
    ]
    rename_safetensor_files(local_src_files, local_dst_files, discard_names=[])
    _ = weight_files(model_path)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_model_path', type=str, help="model and tokenizer path")
    parser.add_argument('--save_directory', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    model_path = args.src_model_path
    save_directory = args.save_directory

    input_model_path = file_utils.standardize_path(model_path, check_link=False)
    file_utils.check_path_permission(input_model_path)

    if not os.path.exists(save_directory):
        os.makedirs(save_directory, exist_ok=True)
    input_save_directory = file_utils.standardize_path(save_directory, check_link=False)
    file_utils.check_path_permission(input_save_directory)

    rename_weights(input_model_path, input_save_directory)
    copy_remaining_files(input_model_path, input_save_directory)
