# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import argparse
import math
import os
import time
import json
from dataclasses import dataclass
from typing import List
from PIL import Image

import torch
import torch_npu
from transformers import AutoProcessor

from atb_llm.utils.file_utils import standardize_path, check_path_permission, check_file_safety
from atb_llm.utils.cpu_binding import NpuHbmInfo
from atb_llm.utils.env import ENV
from atb_llm.utils import file_utils
from atb_llm.utils.log import logger, print_log
from examples.server.cache import CacheConfig, CacheManager
from examples.server.generate import decode_token, generate_req
from examples.server.request import MultiModalRequestParams
from examples.run_pa import PARunner, parse_ids


STORE_TRUE = "store_true"
PERF_FILE = os.path.join(os.path.dirname(__file__), "mllama.csv")
PERF_COLUMNS = "batch, input_len, output_len, embedding_len, fisrt_token_time(ms), \
                non_first_token_time(ms), ResponseTime(ms),E2E Throughout Average(Tokens/s)\n"
PRED_FILE = os.path.join(os.path.dirname(__file__), "predict_result.json")


@dataclass
class MultiModalInput:
    input_texts: List | None
    image_path: str | None


def is_image(file_name):
    ext = os.path.splitext(file_name)[1]
    ext = ext.lower()
    if ext in [".jpg", ".png", ".jpeg", ".bmp"]:
        return True
    return False


def parse_images_from_string(image):
    if not image:
        return None
    image_file_list = [path.strip() for path in image.split(',')]
    image_list = []

    for image_file in image_file_list:
        # 安全校验
        image_file = standardize_path(image_file)
        check_path_permission(image_file)
        check_file_safety(image_file, is_check_file_size=False)

        if is_image(image_file):
            image_list.append(Image.open(image_file))

    return image_list


def request_from_text_and_image_mllama(processor, model, multimodalparams):
    text = multimodalparams.text
    image = multimodalparams.image
    video = multimodalparams.video
    max_out_length = multimodalparams.max_out_length
    block_size = multimodalparams.block_size
    req_idx = multimodalparams.req_idx
    adapter_id = multimodalparams.adapter_id
    position_ids = None

    images = parse_images_from_string(image)
    input_ids, multi_modal_inputs = model.model.prepare_prefill_token(
        text, images, video, processor)
    request = MllamaRequest(max_out_length, block_size, req_idx,
                            input_ids, multi_modal_inputs, adapter_id, position_ids)
    return request


class MllamaRequest():
    def __init__(self, max_out_length: int, block_size: int, req_id: int,
                 input_ids: torch.Tensor, multi_modal_inputs: dict | None, adapter_id: None | str, position_ids=None):
        self.req_id = req_id
        self.input_ids = input_ids
        self.adapter_id = adapter_id
        self.input_length = self.input_ids.shape[0]
        self.adapter_id = adapter_id
        self.position_ids = position_ids
        self.context_length = None if position_ids is None else position_ids[-1] + 1
        self.image_context_length = 0
        self.has_image = False
        if multi_modal_inputs:
            self.cross_attention_mask = multi_modal_inputs.pop(
                'cross_attention_mask')
            self.full_text_row_masked_out_mask = multi_modal_inputs.pop(
                'full_text_row_masked_out_mask')
            self.num_vision_tokens = multi_modal_inputs.pop(
                'num_vision_tokens')
            self.multi_modal_inputs = multi_modal_inputs
            self.image_context_length = self.cross_attention_mask.shape[-1] * \
                self.num_vision_tokens
            self.has_image = True
        try:
            self.need_blocks = max(math.ceil((self.input_length + max_out_length) /
                                   block_size), math.ceil(self.image_context_length / block_size))
        except ZeroDivisionError as e:
            raise ZeroDivisionError from e
        self.need_slots = self.need_blocks * block_size
        self.block_tables = None
        self.slot_tables = None
        self.out_token_list = []


class MultiModalPARunner(PARunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init_processor()
        self.predict_result = kwargs.get('prediction_result', False)
        self.performance = kwargs.get('performance', False)
        self.max_prefill_tokens = kwargs.get('max_prefill_tokens', None)
        self.warm_up_num_blocks = 0
        self.warm_up_memory = 0
        self.cache_manager = None
        self.input_attrs = MultiModalInput(kwargs.get('input_texts', None),
                                           kwargs.get('image_path', None))
        self.max_image_num = kwargs.get('max_image_num', 1)
        self.max_num_tiles = self.model.model.vision_model.max_num_tiles
        self.num_patches = self.model.model.vision_model.num_patches

    def init_processor(self):
        self.processor = AutoProcessor.from_pretrained(self.model_path)

    def init_cache_manager(self, max_num_blocks: int = None):
        # 1 << 30正好是1G
        max_memory = ENV.memory_fraction * self.max_memory
        free_memory = max_memory - ENV.reserved_memory_gb * (1 << 30) - (
            self.warm_up_memory if self.warm_up_memory != 0 else self.init_memory)
        print_log(self.rank, logger.info,
                  f"infer max_memory(GB): {max_memory / (1024 ** 3): .2f}, "
                  f"warm_up_memory(GB): {self.warm_up_memory / (1024 ** 3): .2f}, "
                  f"free_memory(GB): {free_memory / (1024 ** 3): .2f}")
        try:
            num_blocks = (math.ceil(max(self.max_input_length + self.max_output_length,
                                        self.max_image_num * self.max_num_tiles * self.num_patches) /
                                    self.block_size)) * self.max_batch_size
        except ZeroDivisionError as e:
            raise ZeroDivisionError from e
        if max_num_blocks:
            num_blocks = min(num_blocks, max_num_blocks)
        print_log(self.rank, logger.info,
                  f"num_blocks: {num_blocks}, free_memory: {free_memory}")
        cache_config = CacheConfig(num_blocks, self.block_size)
        self.cache_manager = CacheManager(cache_config, self.model_config)
    
    def apply_chat_template(self, text, image_path):
        content = []
        if image_path:
            content.append({"type": "image"})
        content.append({"type": "text", "text": text})
        messages = [
            {"role": "user", "content": content}
        ]
        return self.processor.apply_chat_template(messages, add_generation_prompt=True)

    def warm_up(self):
        input_text = "<|begin_of_text|>If I had to write a haiku for this one"
        image_path = self.input_attrs.image_path
        if image_path:
            input_text = "<|image|>" + input_text
        if self.max_prefill_tokens == -1:
            self.max_prefill_tokens = self.max_batch_size * \
                (self.max_input_length + self.max_output_length)
        print_log(self.rank, logger.info,
                  "---------------begin warm_up---------------")
        try:
            self.init_cache_manager()
        except ZeroDivisionError as e:
            raise ZeroDivisionError from e

        req_list = [request_from_text_and_image_mllama(
            self.processor,
            self.model,
            MultiModalRequestParams(input_text,
                                    image_path,
                                    None,
                                    self.max_output_length,
                                    self.block_size,
                                    req_idx=0))]
        self.model.postprocessor.max_new_tokens = 2
        generate_req(req_list, self.model, self.max_batch_size,
                     self.max_prefill_tokens, self.cache_manager)
        self.warm_up_memory = int(
            self.max_memory * NpuHbmInfo.get_hbm_usage(self.local_rank, self.world_size, self.model.soc_info.need_nz))
        print_log(self.rank, logger.info,
                  f'warmup_memory(GB): {self.warm_up_memory / (1024 ** 3): .2f}')
        print_log(self.rank, logger.info,
                  "---------------end warm_up---------------")

    def infer(self, inputs, batch_size, max_output_length, ignore_eos, is_chat_model=False, **kwargs):
        print_log(self.rank, logger.info,
                  "---------------begin inference---------------")
        input_text = inputs.input_texts        
        image_path = inputs.image_path

        if not self.cache_manager:
            if self.max_prefill_tokens == -1:
                self.max_prefill_tokens = self.max_batch_size * \
                    (self.max_input_length + self.max_output_length)
            self.init_cache_manager()

        self.model.postprocessor.max_new_tokens = max_output_length
        all_input_texts = []
        all_generate_text_list = []
        all_token_num_list = []
        e2e_time_all = 0
        file_list = None
        req_list = None
        req_count = len(input_text)
        if (image_path) and req_count != len(image_path):
            message = "input_text length must equal input_images length"
            print_log(self.rank, logger.error, message)
            raise RuntimeError(message)
        
        if is_chat_model:
            for i in range(req_count):
                input_text[i] = self.apply_chat_template(input_text[i], image_path[i])

        if not ENV.profiling_enable:
            print_log(self.rank, logger.debug, "no profiling")
            torch.npu.synchronize()
            e2e_start = time.time()
            if ignore_eos:
                self.model.postprocessor.eos_token_id = []
            max_iters = math.ceil(req_count / self.max_batch_size)
            for current_iter in range(max_iters):
                if req_count % self.max_batch_size != 0 and current_iter == max_iters - 1:
                    current_batch_size = min(
                        batch_size, req_count % batch_size)
                else:
                    current_batch_size = batch_size
                req_list = [request_from_text_and_image_mllama(
                            self.processor,
                            self.model,
                            MultiModalRequestParams(input_text[current_iter * batch_size + single_batch],
                                                    image_path[current_iter *
                                                               batch_size + single_batch]
                                                    if image_path else None,
                                                    None,
                                                    max_output_length,
                                                    self.block_size,
                                                    req_idx=single_batch))
                            for single_batch in range(current_batch_size)]
                print_log(self.rank, logger.debug,
                          f'req_list[0].input_ids: {req_list[0].input_ids}')
                print_log(self.rank, logger.info,
                          f'current iter: {current_iter}')
                generate_req(req_list, self.model, self.max_batch_size,
                             self.max_prefill_tokens, self.cache_manager)
                generate_text_list, token_num_list = decode_token(
                    req_list, self.tokenizer)
                torch.npu.synchronize()
                e2e_end = time.time()
                e2e_time = e2e_end - e2e_start
                e2e_time_all += e2e_time
                all_generate_text_list.extend(generate_text_list)
                all_token_num_list.extend(token_num_list)
                all_input_texts.extend(
                    input_text[current_iter * batch_size: current_iter * batch_size + current_batch_size])
        else:
            print_log(self.rank, logger.debug, "no profiling")
            profiling_path = ENV.profiling_filepath
            profiler_level = torch_npu.profiler.ProfilerLevel
            actual_profiler_level = getattr(
                profiler_level, ENV.profiling_level)
            if not os.path.exists(profiling_path):
                os.makedirs(profiling_path, exist_ok=True)
            torch.npu.synchronize()
            e2e_start = time.time()
            experimental_config = torch_npu.profiler._ExperimentalConfig(
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                profiler_level=actual_profiler_level,
                l2_cache=False,
                data_simplification=False
            )
            current_batch_size = min(batch_size, req_count)
            with torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU
                ],
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    profiling_path),
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
                with_flops=False,
                with_modules=False,
                    experimental_config=experimental_config):
                req_list = [request_from_text_and_image_mllama(
                            self.processor,
                            self.model,
                            MultiModalRequestParams(input_text[single_batch],
                                                    image_path[single_batch]
                                                    if image_path else None,
                                                    None,
                                                    max_output_length,
                                                    self.block_size,
                                                    req_idx=single_batch))
                            for single_batch in range(current_batch_size)]
                generate_req(req_list, self.model, self.max_batch_size,
                             self.max_prefill_tokens, self.cache_manager)
            torch.npu.synchronize()
            e2e_end = time.time()
            e2e_time = e2e_end - e2e_start
            e2e_time_all += e2e_time
            all_input_texts.append(input_text[:current_batch_size])
        if self.predict_result:
            if self.local_rank == 0:
                image_answer_pairs = {}
                for text_index in range(len(all_input_texts)):
                    image_answer_pairs[file_list[text_index]
                                       ] = all_generate_text_list[text_index]
                    image_answer_pairs = dict(
                        sorted(image_answer_pairs.items()))
                if not os.path.exists(PRED_FILE):
                    with file_utils.safe_open(PRED_FILE, "w") as f:
                        json.dump(image_answer_pairs, f)
                else:
                    with file_utils.safe_open(PRED_FILE, "r") as f:
                        old_data = json.load(f)
                    old_data.update(image_answer_pairs)
                    old_data = dict(sorted(old_data.items()))
                    with file_utils.safe_open(PRED_FILE, "w") as f:
                        json.dump(old_data, f)

        if ENV.token_ids_save_enable:
            if self.local_rank == 0:
                for idx, req in enumerate(req_list):
                    input_ids_save_filename = f"input_ids_{idx}.pth"
                    input_ids_save_path = os.path.join(
                        ENV.token_ids_save_folder, input_ids_save_filename)
                    input_ids_save_path = file_utils.standardize_path(
                        input_ids_save_path)
                    file_utils.check_file_safety(input_ids_save_path, 'w')
                    torch.save(req.input_ids.cpu(), input_ids_save_path)

                    output_ids_save_filename = f"output_ids_{idx}.txt"
                    output_ids_save_path = os.path.join(
                        ENV.token_ids_save_folder, output_ids_save_filename)
                    with file_utils.safe_open(output_ids_save_path, 'w', encoding='utf-8') as f:
                        f.write(' '.join(map(str, req.out_token_list)))
        print_log(self.rank, logger.info, "---------------end inference---------------")
        return all_generate_text_list, all_token_num_list, e2e_time_all


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        help="model and tokenizer path",
                        required=True)
    parser.add_argument('--image_path',
                        help="image path",
                        default=None)
    parser.add_argument(
        '--input_texts',
        type=str,
        nargs='+',
        default="If I had to write a haiku for this one, it would be: ")
    parser.add_argument(
        '--input_ids',
        type=parse_ids,
        nargs='+',
        default=None)
    parser.add_argument(
        '--prediction_result',
        action=STORE_TRUE)
    parser.add_argument('--max_position_embeddings', type=int, default=None)
    parser.add_argument('--max_input_length', type=int, default=4096)
    parser.add_argument('--max_output_length', type=int, default=256)
    parser.add_argument('--max_prefill_tokens', type=int, default=-1)
    parser.add_argument("--max_batch_size", type=int, default=1)
    parser.add_argument("--max_image_num", type=int, default=1)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument('--ignore_eos', action=STORE_TRUE)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    input_texts = [args.input_texts] * args.max_batch_size
    image_path = [args.image_path] * args.max_batch_size

    rank = ENV.rank
    local_rank = ENV.local_rank
    world_size = ENV.world_size
    input_dict = {
        'rank': rank,
        'world_size': world_size,
        'local_rank': local_rank,
        'enable_atb_torch': True,
        **vars(args)
    }

    pa_runner = MultiModalPARunner(**input_dict)

    infer_params = {
        "inputs": MultiModalInput(input_texts,
                                  image_path),
        "batch_size": args.max_batch_size,
        "max_output_length": args.max_output_length,
        "ignore_eos": args.ignore_eos,
        "is_chat_model": True,
    }
    pa_runner.warm_up()
    generate_texts, token_nums, _ = pa_runner.infer(**infer_params)

    length = len(input_texts)
    for i, generate_text in enumerate(generate_texts):
        if i < length:
            print_log(rank, logger.info, f'Question[{i}]: {input_texts[i]}', need_filter=True)
            print_log(rank, logger.info, f'Image[{i}]: {image_path[i]}', need_filter=True)
            print_log(rank, logger.info, f'Answer[{i}]: {generate_text}')
            print_log(rank, logger.info, f'Generate[{i}] token num: {token_nums[i]}')
