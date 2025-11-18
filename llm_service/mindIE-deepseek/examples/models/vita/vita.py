# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import argparse
from dataclasses import dataclass
import math
import os

from transformers import AutoTokenizer

from atb_llm.utils.cpu_binding import NpuHbmInfo
from atb_llm.utils.env import ENV
from atb_llm.utils.file_utils import safe_listdir, standardize_path, check_file_safety
from atb_llm.models.base.model_utils import safe_from_pretrained
from atb_llm.utils.log import logger, print_log
from atb_llm.utils.multimodal_utils import MultimodalInput, MultimodalRequestOut
from examples.multimodal_runner import MultimodalPARunner
from examples.server.cache import CacheManager, CacheConfig
from examples.server.request import request_from_multimodalinputs, MultiModalReqParams
from examples.server.generate import generate_req

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_AUDIO_TOKEN = "<audio>"
IMAGE_TOKEN_INDEX = -200
AUDIO_TOKEN_INDEX = -500
IGNORE_INDEX = -100
MAX_IMAGE_LENGTH = 12

#will be removed
STORE_TRUE = "store_true"
PERF_FILE = "./examples/models/vita_mixtral/vita_mixtral_performance.csv"
PERF_COLUMNS = "batch, input_len, output_len, embedding_len, fisrt_token_time(ms), \
                non_first_token_time(ms), ResponseTime(ms),E2E Throughout Average(Tokens/s)\n"
PRED_FILE = "./examples/models/vita_mixtral/predict_result.json"


@dataclass
class VitaReqParams(MultiModalReqParams):
    framecat: bool = False


class VitaMixtralRunner(MultimodalPARunner):
    def __init__(self, **kwargs):
        self.processor = None
        super().__init__(**kwargs)
        self.framecat = kwargs.get('framecat', False)

    def init_processor(self):
        self.processor = safe_from_pretrained(AutoTokenizer, self.model_path, use_fast=True)

    def warm_up(self):
        image_path = self.warm_up_inputs.image_path
        video_path = self.warm_up_inputs.video_path
        audio_path = self.warm_up_inputs.audio_path
        input_texts = self.warm_up_inputs.input_texts
        if self.max_prefill_tokens == -1:
            self.max_prefill_tokens = self.max_batch_size * (self.max_input_length + self.max_output_length)
        print_log(self.rank, logger.info, "---------------begin warm_up---------------")
        try:
            self.warm_up_num_blocks = math.ceil((self.max_input_length + self.max_output_length) /
                                                self.block_size) * self.max_batch_size
        except ZeroDivisionError as e:
            raise ZeroDivisionError from e
        cache_config = CacheConfig(self.warm_up_num_blocks, self.block_size)
        self.cache_manager = CacheManager(cache_config, self.model_config)
        req_list = [request_from_multimodalinputs(
                        self.processor,
                        self.model,
                        VitaReqParams(input_texts[0] if input_texts else None,
                                                image_path[0] if image_path else None,
                                                video_path[0] if video_path else None,
                                                audio_path[0] if audio_path else None,
                                                self.max_output_length,
                                                self.block_size,
                                                req_idx=single_batch,
                                                adapter_id=self.adapter_id,
                                                batch_size=self.max_batch_size,
                                                framecat=self.framecat))
                        for single_batch in range(self.max_batch_size)
                    ]
        self.model.postprocessor.max_new_tokens = 2
        
        generate_req(req_list, self.model, self.max_batch_size, self.max_prefill_tokens, self.cache_manager)
        self.warm_up_memory = int(
            self.max_memory * NpuHbmInfo.get_hbm_usage(self.local_rank, self.world_size, self.model.soc_info.need_nz))
        print_log(self.rank, logger.info, f'warmup_memory(GB): {self.warm_up_memory / (1024 ** 3): .2f}')
        print_log(self.rank, logger.info, "---------------end warm_up---------------")
        
    def prepare_request(self, multimodalinputs, batch_size, max_output_length, current_iter):
        input_texts = multimodalinputs.input_texts
        img_path = multimodalinputs.image_path
        vid_path = multimodalinputs.video_path
        audio_path = multimodalinputs.audio_path
        req_list = [request_from_multimodalinputs(
                    self.processor,
                    self.model,
                    VitaReqParams(input_texts[current_iter * batch_size
                                                        + single_batch] if input_texts else None,
                                        img_path[current_iter * batch_size
                                                        + single_batch] if img_path else None,
                                        vid_path[current_iter * batch_size
                                                        + single_batch] if vid_path else None,
                                        audio_path[current_iter * batch_size
                                                        + single_batch] if audio_path else None,
                                        max_output_length,
                                        self.block_size,
                                        req_idx=single_batch,
                                        adapter_id=self.adapter_id,
                                        batch_size=batch_size,
                                        framecat=self.framecat))  
                    for single_batch in range(batch_size)]
        print_log(self.rank, logger.debug, f'req_list[0].input_ids: {req_list[0].input_ids}')
        return MultimodalRequestOut(req_list, batch_size, img_path, vid_path, None, input_texts)

    def precision_save(self, precision_inputs, **kwargs):
        all_input_texts = precision_inputs.all_input_texts
        all_generate_text_list = precision_inputs.all_generate_text_list
        image_file_list = precision_inputs.image_file_list
        video_file_list = precision_inputs.video_file_list
        audio_file_list = precision_inputs.audio_file_list
        answer_length = len(audio_file_list) if audio_file_list else len(all_input_texts)
        image_answer_pairs = {}
        for text_index in range(answer_length):
            if image_file_list:
                image_answer_pairs[image_file_list[text_index]] = all_generate_text_list[text_index]
            elif video_file_list:
                image_answer_pairs[video_file_list[text_index]] = all_generate_text_list[text_index]
            else:
                raise ValueError("No image or video provided.")
            image_answer_pairs = dict(sorted(image_answer_pairs.items()))
        super().precision_save(precision_inputs, answer_pairs=image_answer_pairs)

    def performance_test(self, performance_inputs):
        all_input_texts = performance_inputs.all_input_texts
        all_token_num_list = performance_inputs.all_token_num_list
        e2e_time_all = performance_inputs.e2e_time_all
        batch = performance_inputs.batch
        if all_input_texts:
            input_len = self.tokenizer([str(all_input_texts[0])], return_tensors="pt")["input_ids"].flatten().shape[0]
        else:
            input_len = 1
        self.performance_save(input_len, all_token_num_list, e2e_time_all, batch)

    def infer(self, mm_inputs, batch_size, max_output_length, ignore_eos, max_iters=None, **kwargs):
        input_texts = mm_inputs.input_texts
        image_path_list = mm_inputs.image_path
        video_path_list = mm_inputs.video_path
        audio_path_list = mm_inputs.audio_path
        if image_path_list:
            file_length = len(image_path_list)
        elif video_path_list:
            file_length = len(video_path_list)
        elif audio_path_list:
            file_length = len(audio_path_list)
        else:
            file_length = len(input_texts)
        if not ENV.profiling_enable:
            if self.max_batch_size > 0:
                max_iters = math.ceil(file_length / self.max_batch_size)
            else:
                raise ValueError(f"{self.max_batch_size} max_batch_size should > 0, please check")
        return super().infer(mm_inputs, batch_size, max_output_length, ignore_eos, max_iters=max_iters) 


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None, help="Path to the model directory")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--audio_path", type=str, default=None)
    parser.add_argument("--model_type", type=str, default="mixtral-8x7b")
    parser.add_argument("--conv_mode", type=str, default="mixtral_two")
    parser.add_argument("--question", type=str, default="")
    parser.add_argument(
        '--prediction_result',
         action=STORE_TRUE)
    parser.add_argument(
        '--performance',
         action=STORE_TRUE)
    parser.add_argument('--max_input_length', type=int, default=2048)
    parser.add_argument('--max_output_length', type=int, default=256)
    parser.add_argument('--max_prefill_tokens', type=int, default=-1)
    parser.add_argument("--max_batch_size", type=int, default=1)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument('--ignore_eos', action=STORE_TRUE)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    rank = ENV.rank
    local_rank = ENV.local_rank
    world_size = ENV.world_size
    input_dict = {
        'rank': rank,
        'world_size': world_size,
        'local_rank': local_rank,
        'perf_file': PERF_FILE,
        'pred_file': PRED_FILE,
        **vars(args)
    }

    if (args.audio_path is None) == (args.question == ""):
        raise ValueError("Exactly one of audio_path or qs must be non-None")

    file_length = 0
    image_path = None
    video_path = None
    audio_path = None
    new_input_texts = None
    if args.image_path:
        image_path = standardize_path(args.image_path)
        check_file_safety(image_path, 'r')
        file_name = safe_listdir(image_path)
        image_length = len(file_name)
        image_path = [os.path.join(image_path, f) for f in file_name]
        remainder = image_length % args.max_batch_size
        if remainder != 0:
            num_to_add = args.max_batch_size - remainder
            image_path.extend([image_path[-1]] * num_to_add)
        file_length = len(image_path)
    elif args.video_path:
        video_path = standardize_path(args.video_path)
        check_file_safety(video_path, 'r')
        file_name = safe_listdir(video_path)
        video_length = len(file_name)
        video_path = [os.path.join(video_path, f) for f in file_name]
        remainder = video_length % args.max_batch_size
        if remainder != 0:
            num_to_add = args.max_batch_size - remainder
            video_path.extend([video_path[-1]] * num_to_add)
        file_length = len(video_path)

    if args.audio_path:
        audio_path = standardize_path(args.audio_path)
        check_file_safety(audio_path, 'r')
        file_name = safe_listdir(audio_path)
        file_length_audio = len(file_name)
        audio_path = [os.path.join(audio_path, f) for f in file_name]
        if file_length != 0:
            if file_length_audio > file_length:
                raise ValueError(f"audios has {file_length_audio} files," + \
                    f"which is not less than image/video's {file_length} files.")
            audio_path.extend([audio_path[-1]] * (file_length - file_length_audio))
        remainder = len(audio_path) % args.max_batch_size
        if remainder != 0:
            num_to_add = args.max_batch_size - remainder
            audio_path.extend([audio_path[-1]] * num_to_add)
    else:
        new_input_texts = [args.question]
        length_question = len(new_input_texts)
        if file_length != 0:
            if length_question > file_length:
                raise ValueError(f"audios has {length_question} files," + \
                    f"which is not less than image/video's {file_length} files.")
            new_input_texts.extend([new_input_texts[-1]] * (file_length - length_question))
        remainder = len(new_input_texts) % args.max_batch_size
        if remainder != 0:
            num_to_add = args.max_batch_size - remainder
            new_input_texts.extend([new_input_texts[-1]] * num_to_add)

    input_dict['input_texts'] = new_input_texts
    input_dict['image_path'] = image_path
    input_dict['video_path'] = video_path
    input_dict['audio_path'] = audio_path
    pa_runner = VitaMixtralRunner(**input_dict)

    infer_params = {
        "mm_inputs": MultimodalInput(
                             new_input_texts,
                             image_path,
                             video_path,
                             audio_path),
        "batch_size": args.max_batch_size,
        "max_output_length": args.max_output_length,
        "ignore_eos": args.ignore_eos,
    }

    pa_runner.warm_up()
    all_generate_text_list, all_token_num_list, e2e_time_all = pa_runner.infer(**infer_params)
    for i, generate_text in enumerate(all_generate_text_list):
        print_log(rank, logger.info, f'Answer[{i}]: {generate_text}')
        print_log(rank, logger.info, f'Generate[{i}] token num: {all_token_num_list[i]}')