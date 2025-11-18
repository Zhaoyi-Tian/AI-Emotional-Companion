# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json
import math
import os
import time

import torch
import torch_npu
from PIL import Image
from transformers import AutoImageProcessor
from atb_llm.models.base.model_utils import safe_from_pretrained
from atb_llm.models.qwen2_vl.router_qwen2_vl import process_shared_memory
from atb_llm.runner.tokenizer_wrapper import TokenizerWrapper
from atb_llm.utils import argument_utils
from atb_llm.utils.cpu_binding import NpuHbmInfo
from atb_llm.utils.env import ENV
from atb_llm.utils.file_utils import safe_open, is_path_exists, safe_listdir, standardize_path, check_file_safety
from atb_llm.utils.log import logger, print_log
from atb_llm.utils.shm_utils import decode_shape_from_int64, release_shared_memory
from examples.multimodal_runner import MultimodalPARunner, parser
from examples.multimodal_runner import path_validator
from examples.server.cache import CacheManager, CacheConfig
from examples.server.generate import decode_token, generate_req
from examples.server.request import MultiModalRequest


VISION_START_TOKEN_ID = 151652
_IMAGE_TOKEN_ID = 151655
VISION_END_TOKEN_ID = 151653
_IMAGE_FEATURE_LENS = 64
IMAGE_THW_TOKEN_OFFSET = 3
SUPPORTED_IMAGE_MODE = "RGB"
PYTORCH_TENSOR = "pt"


def request_from_token_qwen2_vl(input_ids, max_out_length, block_size,
                                req_idx=0, adapter_id=None):
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids, dtype=torch.int64)
    position_ids = torch.arange(len(input_ids), dtype=torch.int64)
    if torch.any(torch.eq(input_ids, VISION_START_TOKEN_ID)):
        bos_pos = torch.where(torch.eq(input_ids, VISION_START_TOKEN_ID))[0]
        eos_pos = torch.where(torch.eq(input_ids, VISION_END_TOKEN_ID))[0]
        vision_num = bos_pos.shape[0]
        deltas = 0
        for i in range(vision_num):
            thw_shape_value = input_ids[bos_pos[i] + IMAGE_THW_TOKEN_OFFSET]
            thw_shape = decode_shape_from_int64(thw_shape_value)

            vision_feature_len = eos_pos[i] - bos_pos[i] - 1
            max_hw = max(thw_shape[1:])
            if thw_shape[0] > (max_hw // 2):
                deltas += vision_feature_len - thw_shape[0]
            else:
                deltas += vision_feature_len - max_hw // 2
        position_ids[-1] = position_ids[-1] - deltas

    request = MultiModalRequest(max_out_length, block_size, req_idx, input_ids, adapter_id, position_ids)
    return request


class PARunner(MultimodalPARunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.shm_name_save_path = kwargs.get("shm_name_save_path", None)
        self.tokenizer_wrapper = TokenizerWrapper(self.model_path)
        self.tokenizer = self.tokenizer_wrapper.tokenizer

    def init_processor(self):
        try:
            self.processor = safe_from_pretrained(AutoImageProcessor, self.model_path)
        except AssertionError:
            self.processor = self.model.tokenizer

    def warm_up(self):
        all_input_length = self.max_batch_size * self.max_input_length
        input_ids_list = (
                [VISION_START_TOKEN_ID]
                + [_IMAGE_TOKEN_ID] * _IMAGE_FEATURE_LENS
                + [VISION_END_TOKEN_ID]
                + [1] * (all_input_length - _IMAGE_FEATURE_LENS - 2)
        )
        image = Image.new(SUPPORTED_IMAGE_MODE, (224, 224), (255, 255, 255))
        warmup_image_processor = safe_from_pretrained(AutoImageProcessor, self.model_path)
        images_inputs = warmup_image_processor(images=image,
                                               videos=None,
                                               return_tensors=PYTORCH_TENSOR)
        image.close()
        input_ids_list[1:4] = process_shared_memory(
            images_inputs.pixel_values, self.shm_name_save_path, images_inputs.image_grid_thw
        )

        input_ids = torch.tensor(input_ids_list, dtype=torch.int64).to(self.device)
        print_log(self.rank, logger.info, "---------------begin warm_up---------------")
        try:
            self.warm_up_num_blocks = math.ceil((self.max_input_length + self.max_output_length) /
                                                self.block_size) * self.max_batch_size
        except ZeroDivisionError as e:
            raise ZeroDivisionError from e
        cache_config = CacheConfig(self.warm_up_num_blocks, self.block_size)
        self.cache_manager = CacheManager(cache_config, self.model_config)
        max_output_length = 2
        self.model.postprocessor.max_new_tokens = max_output_length
        if self.max_prefill_tokens == -1:
            self.max_prefill_tokens = self.max_batch_size * (self.max_input_length + self.max_output_length)
        single_req = request_from_token_qwen2_vl(
            input_ids,
            max_output_length,
            self.block_size,
            req_idx=0,
        )
        generate_req([single_req], self.model, self.max_batch_size, self.max_prefill_tokens, self.cache_manager)
        if is_path_exists(self.shm_name_save_path):
            try:
                release_shared_memory(self.shm_name_save_path)
            except Exception as e:
                print_log(self.rank, logger.error, f"Release shared memory failed: {e}")
            try:
                os.remove(self.shm_name_save_path)
            except Exception as e:
                print_log(self.rank, logger.error, f"Remove shared memory file failed: {e}")
        self.warm_up_memory = int(
            self.max_memory
            * NpuHbmInfo.get_hbm_usage(
                self.local_rank, self.world_size, self.model.soc_info.need_nz
            )
        )
        print_log(
            self.rank,
            logger.info,
            f"warmup_memory(GB): {self.warm_up_memory / (1024 ** 3): .2f}",
        )
        print_log(self.rank, logger.info, "---------------end warm_up---------------")

    def infer(self, mm_inputs, max_output_length, shm_name_save_path, **kwargs):
        self.make_cache_manager()

        self.model.postprocessor.max_new_tokens = max_output_length

        req_list = []
        if not ENV.profiling_enable:
            torch.npu.synchronize()
            e2e_start = time.time()
            for i, mm_input in enumerate(mm_inputs):
                input_ids = self.tokenizer_wrapper.tokenize(mm_input, shm_name_save_path=shm_name_save_path)
                single_req = request_from_token_qwen2_vl(
                    input_ids,
                    max_output_length,
                    self.block_size,
                    req_idx=i,
                )
                req_list.append(single_req)
            generate_req(
                req_list,
                self.model,
                self.max_batch_size,
                self.max_prefill_tokens,
                self.cache_manager,
            )
            generate_text_list, token_num_list = decode_token(req_list, self.tokenizer, skip_special_tokens=True)
            torch.npu.synchronize()
            e2e_end = time.time()
            e2e_time = e2e_end - e2e_start
        else:
            for i, mm_input in enumerate(mm_inputs):
                input_ids = self.tokenizer_wrapper.tokenize(mm_input, shm_name_save_path=shm_name_save_path)
                single_req = request_from_token_qwen2_vl(
                    input_ids,
                    max_output_length,
                    self.block_size,
                    req_idx=i,
                )
                req_list.append(single_req)
            profiling_path = ENV.profiling_filepath
            if not os.path.exists(profiling_path):
                os.makedirs(profiling_path, exist_ok=True)
            torch.npu.synchronize()
            e2e_start = time.time()
            experimental_config = torch_npu.profiler._ExperimentalConfig(
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                l2_cache=False,
                data_simplification=False,
            )
            with torch_npu.profiler.profile(
                    activities=[
                        torch_npu.profiler.ProfilerActivity.CPU,
                        torch_npu.profiler.ProfilerActivity.NPU,
                    ],
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                        profiling_path
                    ),
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=False,
                    with_flops=False,
                    with_modules=False,
                    experimental_config=experimental_config,
            ) as _:
                generate_req(
                    req_list,
                    self.model,
                    self.max_batch_size,
                    self.max_prefill_tokens,
                    self.cache_manager,
                )
            torch.npu.synchronize()
            e2e_end = time.time()
            e2e_time = e2e_end - e2e_start
            generate_text_list, token_num_list = decode_token(req_list, self.tokenizer, skip_special_tokens=True)
        if is_path_exists(shm_name_save_path):
            try:
                release_shared_memory(shm_name_save_path)
            except Exception as e:
                print_log(self.rank, logger.error, f"Release shared memory failed: {e}")
            try:
                os.remove(shm_name_save_path)
            except Exception as e:
                print_log(self.rank, logger.error, f"Remove shared memory file failed: {e}")
        return generate_text_list, token_num_list, e2e_time


def parse_arguments():
    string_validator = argument_utils.StringArgumentValidator(min_length=0, max_length=1000)
    parser_qwen2_vl = parser

    parser_qwen2_vl.add_argument(
        "--input_text",
        default="Describe the image.",
        validator=string_validator
    )
    parser_qwen2_vl.add_argument(
        "--input_image",
        default="",
        validator=path_validator
    )
    parser_qwen2_vl.add_argument(
        "--dataset_path",
        help="precision test dataset path",
        default="",
        validator=path_validator
    )
    parser_qwen2_vl.add_argument(
        "--shm_name_save_path",
        type=str,
        help='This path is used to temporarily store the shared '
             'memory addresses that occur during the inference process.',
        default='./shm_name.txt',
        validator=path_validator
    )
    parser_qwen2_vl.add_argument(
        "--results_save_path",
        help="precision test result path",
        default="./results.json",
        validator=path_validator,
    )

    return parser_qwen2_vl.parse_args()


def is_image(file_image_name):
    ext = os.path.splitext(file_image_name)[1]
    ext = ext.lower()
    if ext in [".jpg", ".png", ".jpeg", ".bmp"]:
        return True
    return False


def is_video(file_video_name):
    video_ext = os.path.splitext(file_video_name)[1]
    video_ext = video_ext.lower()
    if video_ext in [".mp4", ".wmv", ".avi"]:
        return True
    return False


def deal_dataset(dataset_path, text):
    input_images = []
    dataset_path = standardize_path(dataset_path)
    check_file_safety(dataset_path)
    images_list = safe_listdir(dataset_path)
    for img_name in images_list:
        image_path = os.path.join(dataset_path, img_name)
        input_images.append(image_path)
    input_texts = [text] * len(
        input_images
    )
    return input_images, input_texts


def replace_crlf(mm_input):
    result = []
    for single_input in mm_input:
        res = {}
        for k, v in single_input.items():
            input_text_filter = v
            input_text_filter = input_text_filter.replace('\n', ' ').replace('\r', ' ').replace('\f', ' ')
            input_text_filter = input_text_filter.replace('\t', ' ').replace('\v', ' ').replace('\b', ' ')
            input_text_filter = input_text_filter.replace('\u000A', ' ').replace('\u000D', ' ').replace('\u000C', ' ')
            input_text_filter = input_text_filter.replace('\u000B', ' ').replace('\u0008', ' ').replace('\u007F', ' ')
            input_text_filter = input_text_filter.replace('\u0009', ' ').replace('    ', ' ')
            res[k] = input_text_filter.replace("\n", "_").replace("\r", "_")
        result.append(res)
    return result


if __name__ == '__main__':
    args = parse_arguments()
    rank = ENV.rank
    local_rank = ENV.local_rank
    world_size = ENV.world_size
    input_dict = {
        'rank': rank,
        'world_size': world_size,
        'local_rank': local_rank,
        **vars(args)
    }

    pa_runner = PARunner(**input_dict)
    print_log(rank, logger.info, f"pa_runner: {pa_runner}")
    pa_runner.warm_up()
    npu_results_dict = {}
    if args.dataset_path:
        dataset_images, dataset_texts = deal_dataset(args.dataset_path, args.input_text)
        mm_inputs = []
        for dataset_image, dataset_text in zip(dataset_images, dataset_texts):
            if is_video(dataset_image):
                key = "video"
            elif is_image(dataset_image):
                key = "image"
            else:
                raise TypeError("The input field currently only needs to support 'image', 'video'.")
            single_inputs = [{key: dataset_image}, {"text": dataset_text}]
            mm_inputs.append(single_inputs)
    else:
        if is_video(args.input_image):
            key = "video"
        elif is_image(args.input_image):
            key = "image"
        else:
            raise TypeError("The input field currently only needs to support 'image', 'video'.")
        mm_inputs = [
                        [
                            {key: args.input_image},
                            {"text": args.input_text},
                        ]
                    ] * args.max_batch_size

    generate_texts, token_nums, latency = pa_runner.infer(
        mm_inputs,
        args.max_output_length,
        args.shm_name_save_path,
    )
    token_nums_prev = 0
    for i, generate_text in enumerate(generate_texts):
        inputs = mm_inputs
        if args.dataset_path:
            rst_key = dataset_images[i].split("/")[-1]
            npu_results_dict[rst_key] = generate_text
        question = replace_crlf(inputs[i])
        print_log(rank, logger.info, f"Question[{i}]: {question}")
        print_log(rank, logger.info, f"Answer[{i}]: {generate_text}")
        print_log(rank, logger.info, f"Generate[{i}] token num: {token_nums[i][1] - token_nums_prev}")
        token_nums_prev = token_nums[i][1]
    print_log(rank, logger.info, f"Latency(s): {latency}")
    print_log(rank, logger.info, f"Throughput(tokens/s): {token_nums[-1][1] / latency}")

    if args.dataset_path:
        sorted_dict = dict(sorted(npu_results_dict.items()))
        with safe_open(
                args.results_save_path,
                "w",
                override_flags=os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        ) as f:
            json.dump(sorted_dict, f)
