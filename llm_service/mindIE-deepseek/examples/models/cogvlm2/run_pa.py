# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import argparse
import copy
import json
import math
import os
import time
from dataclasses import dataclass
from typing import List

import torch
import torch_npu
from atb_llm.runner.tokenizer_wrapper import TokenizerWrapper
from atb_llm.runner.model_runner import ModelRunner
from atb_llm.utils.cpu_binding import NpuHbmInfo
from atb_llm.utils.env import ENV
from atb_llm.utils.log import logger, print_log
from atb_llm.utils import file_utils
from atb_llm.utils.initial import check_profiling_level
from examples.multimodal_runner import MultimodalPARunner
from examples.server.cache import CacheConfig, ModelConfig, CacheManager
from examples.server.generate import decode_token, generate_req
from examples.server.request import MultiModalRequest


OPERATOR_BOUND = 576
IMAGE_TOKEN_ID = 128002
_IMG_TOKEN_LEN = 2306


@dataclass
class InputAttrs:
    input_texts: List | None
    image_path: str | None


def request_from_token_cogvlm2(input_ids, max_out_length, block_size,
                               req_idx=0, adapter_id=None):
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids, dtype=torch.int64)
    position_ids = torch.arange(len(input_ids), dtype=torch.int64)
    if torch.any(torch.eq(input_ids, IMAGE_TOKEN_ID)):
        pos_ids_list = [0, 1]
        pos_ids_list += [2] * (_IMG_TOKEN_LEN - 2)
        pos_ids_list += [3]
        input_text_len = input_ids.shape[0] - _IMG_TOKEN_LEN - 1
        pos_ids_list += range(4, input_text_len + 4)
        position_ids = torch.tensor(pos_ids_list, dtype=torch.int64)

    request = MultiModalRequest(max_out_length, block_size, req_idx, input_ids, adapter_id, position_ids)
    return request


class PARunner(MultimodalPARunner):
    def __init__(self, **kwargs):
        self.rank = kwargs.get('rank', '0')
        self.local_rank = kwargs.get('local_rank', self.rank)
        self.world_size = kwargs.get('world_size', '1')

        self.model_path = kwargs.get('model_path', None)
        self.lora_adapter = kwargs.get('lora_adapter', None)
        self.input_text = kwargs.get('input_text', None)

        self.max_prefill_batch_size = kwargs.get('max_prefill_batch_size', None)
        self.max_batch_size = kwargs.get('max_batch_size', None)
        self.max_input_length = kwargs.get('max_input_length', None)
        self.max_output_length = kwargs.get('max_output_length', None)
        self.max_position_embeddings = kwargs.get('max_position_embeddings', None)
        self.max_prefill_tokens = kwargs.get('max_prefill_tokens', None)
        self.enable_atb_torch = kwargs.get('enable_atb_torch', False)

        self.block_size = kwargs.get('block_size', None)
        self.chat_template = kwargs.get('chat_template', None)
        self.load_tokenizer = kwargs.get('load_tokenizer', True)

        self.check_limits()

        kw_args = kwargs.get('kw_args', '')
        if kw_args:
            try:
                kw_args = json.loads(kw_args)
            except json.decoder.JSONDecodeError:
                print_log(self.local_rank, logger.warning, 'Please checkout input of kw_args!')
                kw_args = {}
        else:
            kw_args = {}

        self.tokenizer_wrapper = TokenizerWrapper(self.model_path)
        self.model = ModelRunner(
            self.model_path, rank=self.rank, world_size=self.world_size,
            local_rank=self.local_rank,
            max_position_embeddings=self.max_position_embeddings,
            enable_atb_torch=self.enable_atb_torch,
            load_tokenizer=self.load_tokenizer,
            lora_adapter=self.lora_adapter,
            **kw_args
        )
        self.tokenizer = self.model.tokenizer
        if self.chat_template:
            self.tokenizer.chat_template = self._load_chat_template(self.chat_template)
        self.dtype = self.model.dtype
        self.quantize = self.model.quantize
        self.kv_quant_type = self.model.kv_quant_type
        self.model.load_weights(**kw_args)

        self.device = self.model.device
        self.model_config = ModelConfig(self.model.num_heads,
                                        self.model.num_kv_heads,
                                        self.model.config.num_key_value_heads \
                                            if hasattr(self.model.config, 'num_key_value_heads') \
                                            else self.model.num_kv_heads,
                                        self.model.k_head_size,
                                        self.model.v_head_size,
                                        self.model.num_layers,
                                        self.model.device,
                                        self.model.dtype,
                                        self.model.soc_info,
                                        self.kv_quant_type)

        self.max_memory = NpuHbmInfo.get_hbm_capacity(self.local_rank, self.world_size, self.model.soc_info.need_nz)
        self.init_memory = int(
            self.max_memory * NpuHbmInfo.get_hbm_usage(self.local_rank, self.world_size, self.model.soc_info.need_nz))
        print_log(self.rank, logger.info, f'hbm_capacity(GB): {self.max_memory / (1024 ** 3)}, '
                                          f'init_memory(GB): {self.init_memory / (1024 ** 3)}')

        self.warm_up_memory = 0
        self.warm_up_num_blocks = 0
        self.cache_manager = None
        self.compress_head_enable = ENV.compress_head_enable

    def __repr__(self):
        return (
                "PARunner("
                + f"model_path={self.model_path}, "
                + f"input_text={self.input_text}, "
                + f"max_position_embeddings={self.max_position_embeddings}, "
                + f"max_input_length={self.max_input_length}, "
                + f"max_output_length={self.max_output_length}, "
                + f"max_prefill_tokens={self.max_prefill_tokens}, "
                + f"load_tokenizer={self.load_tokenizer}, "
                + f"enable_atb_torch={self.enable_atb_torch}, "
                + f"max_prefill_batch_size={self.max_prefill_batch_size}, "
                + f"max_batch_size={self.max_batch_size}, "
                + f"dtype={self.dtype}, "
                + f"block_size={self.block_size}, "
                + f"model_config={self.model_config}, "
                + f"max_memory={self.max_memory}, "
        )

    @staticmethod
    def _load_chat_template(chat_template: str):
        if os.path.exists(chat_template):
            with file_utils.safe_open(chat_template, 'r', encoding='utf-8') as f:
                chat_template_content = f.read()
        else:
            chat_template_content = chat_template
        return chat_template_content

    def check_limits(self):
        if self.max_input_length <= 0:
            print_log(self.rank, logger.warning,
                      f"'max_input_length' must be greater than 0, but is {self.max_input_length}. "
                      "Please increase the value of 'max_input_length'. ")
        if self.max_output_length <= 0:
            print_log(self.rank, logger.warning,
                      f"'max_output_length' must be greater than 0, but is {self.max_output_length}. "
                      "Please increase the value of 'max_output_length'. ")

        if self.max_position_embeddings and \
                self.max_position_embeddings < self.max_input_length + self.max_output_length:
            print_log(self.rank, logger.warning,
                      f"'max_position_embeddings' must be greater than "
                      f"{self.max_input_length + self.max_output_length}, but is {self.max_position_embeddings}. "
                      "Please increase the value of 'max_position_embeddings'. ")

        if self.max_prefill_batch_size and self.max_prefill_batch_size <= 0:
            print_log(self.rank, logger.warning,
                      f"'max_prefill_batch_size' must be greater than 0, but is {self.max_prefill_batch_size}. "
                      "Please increase the value of 'max_prefill_batch_size'. ")

        if self.max_batch_size <= 0:
            print_log(self.rank, logger.warning,
                      f"'max_batch_size' must be greater than 0, but is {self.max_batch_size}. "
                      "Please increase the value of 'max_batch_size'. ")

        if self.block_size > OPERATOR_BOUND:
            print_log(self.rank, logger.warning,
                      f"'block_size' must be lower than or equal to 576, but is {self.block_size}. "
                      "Please decrese the value of 'block_size'. ")

        if self.block_size <= 0:
            print_log(self.rank, logger.warning,
                      f"'block_size' is too small: {self.block_size}. "
                      "Please increse the value of 'block_size'. ")

        if self.max_prefill_tokens != -1 and \
                self.max_prefill_tokens <= self.max_input_length:
            print_log(self.rank, logger.warning,
                      f"'max_prefill_tokens' must be greater than or equal to {self.max_input_length}, "
                      "but is {self.block_size}. Please increse the value of 'max_prefill_tokens'. ")

    def warm_up(self):
        if self.max_prefill_batch_size is None:
            self.max_prefill_batch_size = self.max_batch_size
        if self.max_prefill_tokens == -1:
            self.max_prefill_tokens = self.max_prefill_batch_size * (self.max_input_length + self.max_output_length)
        all_input_length = self.max_prefill_batch_size * self.max_input_length
        input_ids = torch.ones(all_input_length, dtype=torch.int64).to(self.device)
        position_ids = torch.arange(self.max_input_length, dtype=torch.int32).repeat(self.max_prefill_batch_size).to(
            self.device)
        cu_seqlen_prefill = torch.tensor([1])
        try:
            block_num = math.ceil(all_input_length / self.block_size)
        except ZeroDivisionError as e:
            raise ZeroDivisionError from e
        block_tables_tensor = torch.arange(block_num, dtype=torch.int32).view(1, -1).to(self.device)
        slots = torch.arange(all_input_length, dtype=torch.int32).to(self.device)
        input_lengths_tensor = torch.tensor(
            [self.max_input_length] * self.max_prefill_batch_size, dtype=torch.int64
        ).to(self.device)
        prefill_head_indices = torch.tensor([all_input_length - 1], dtype=torch.int64).to(self.device)
        print_log(self.rank, logger.info, "---------------begin warm_up---------------")
        try:
            self.warm_up_num_blocks = math.ceil((self.max_input_length + self.max_output_length) /
                                                self.block_size) * self.max_batch_size
        except ZeroDivisionError as e:
            raise ZeroDivisionError from e
        cache_config = CacheConfig(self.warm_up_num_blocks, self.block_size)
        if self.compress_head_enable:
            cache_config = CacheConfig(self.warm_up_num_blocks, self.block_size, \
                                       self.max_input_length, self.max_output_length, self.max_batch_size, self.rank,
                                       self.world_size)
        self.cache_manager = CacheManager(cache_config, self.model_config)
        self.model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            is_prefill=cu_seqlen_prefill is not None,
            block_tables=block_tables_tensor,
            kv_cache=self.cache_manager.kv_cache,
            slots=slots,
            input_lengths=input_lengths_tensor,
            max_seq_len=self.max_input_length,
            lm_head_indices=prefill_head_indices
        )
        self.warm_up_memory = int(
            self.max_memory * NpuHbmInfo.get_hbm_usage(self.local_rank, self.world_size, self.model.soc_info.need_nz))
        print_log(self.rank, logger.info, f'warmup_memory(GB): {self.warm_up_memory / (1024 ** 3): .2f}')
        print_log(self.rank, logger.info, "---------------end warm_up---------------")

    def infer(self, mm_inputs, batch_size, max_output_length, ignore_eos, is_chat_model=False, **kwargs):
        print_log(self.rank, logger.info, "---------------begin inference---------------")
        if ignore_eos:
            self.model.postprocessor.eos_token_id = []
        is_truncation = kwargs.get("truncation", False)

        req_list = []
        for idx, mm_input in enumerate(mm_inputs):
            input_ids, adapter_ids = self._build_model_inputs(mm_input.input_texts, mm_input.image_path, is_chat_model,
                                                              is_truncation)
            single_req = request_from_token_cogvlm2(
                input_ids[0],
                max_output_length,
                self.block_size,
                req_idx=idx,
                adapter_id=adapter_ids[0]
            )
            req_list.append(single_req)

        if not self.cache_manager:
            if self.max_prefill_tokens == -1:
                self.max_prefill_tokens = self.max_batch_size * (self.max_input_length + self.max_output_length)
            cache_block_size = self.block_size * self.model.num_kv_heads * self.model.head_size
            dtype_size = CacheManager.get_dtype_size(self.dtype)
            total_cache_size = self.model.num_layers * cache_block_size * 2 * dtype_size

            max_memory = ENV.memory_fraction * self.max_memory
            free_memory = max_memory - ENV.reserved_memory_gb * (1 << 30) - (
                self.warm_up_memory if self.warm_up_memory != 0 else self.init_memory)
            print_log(self.rank, logger.info,
                      f"infer max_memory(GB): {max_memory / (1024 ** 3): .2f}, "
                      f"warm_up_memory(GB): {self.warm_up_memory / (1024 ** 3): .2f}, "
                      f"free_memory(GB): {free_memory / (1024 ** 3): .2f}")

            num_blocks = int(free_memory // total_cache_size)
            print_log(self.rank, logger.info, f"num_blocks: {num_blocks}, free_memory: {free_memory}")
            cache_config = CacheConfig(num_blocks, self.block_size)
            if self.compress_head_enable:
                cache_config = CacheConfig(self.warm_up_num_blocks, self.block_size, \
                                           self.max_input_length, self.max_output_length, self.max_batch_size,
                                           self.rank, self.world_size)
            self.cache_manager = CacheManager(cache_config, self.model_config)

        if ENV.benchmark_enable:
            req_list_dummy = copy.deepcopy(req_list)
            self.model.postprocessor.max_new_tokens = 2
            generate_req(req_list_dummy, self.model, self.max_batch_size, self.max_prefill_tokens, self.cache_manager)

        self.model.postprocessor.max_new_tokens = max_output_length
        skip_special_tokens = kwargs.get("skip_special_tokens", False)
        if not ENV.profiling_enable:
            print_log(self.rank, logger.debug, "no profiling")
            torch.npu.synchronize()
            e2e_start = time.time()
            generate_req(req_list, self.model, self.max_batch_size, self.max_prefill_tokens, self.cache_manager)
            _, _ = decode_token(req_list, self.tokenizer, skip_special_tokens)
            torch.npu.synchronize()
            e2e_end = time.time()
            e2e_time = e2e_end - e2e_start
        else:
            print_log(self.rank, logger.debug, "enter profiling")
            profiling_path = ENV.profiling_filepath
            check_profiling_level()
            profiler_level = torch_npu.profiler.ProfilerLevel
            actual_profiler_level = getattr(profiler_level, ENV.profiling_level)
            torch.npu.synchronize()
            e2e_start = time.time()
            experimental_config = torch_npu.profiler._ExperimentalConfig(
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                profiler_level=actual_profiler_level,
                l2_cache=False,
                data_simplification=False
            )
            with torch_npu.profiler.profile(
                    activities=[
                        torch_npu.profiler.ProfilerActivity.CPU,
                        torch_npu.profiler.ProfilerActivity.NPU
                    ],
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profiling_path),
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=False,
                    with_flops=False,
                    with_modules=False,
                    experimental_config=experimental_config):
                generate_req(req_list, self.model, self.max_batch_size, self.max_prefill_tokens, self.cache_manager)
            torch.npu.synchronize()
            e2e_end = time.time()
            e2e_time = e2e_end - e2e_start

        generate_text_list, token_num_list = decode_token(req_list, self.tokenizer, skip_special_tokens)
        if ENV.modeltest_dataset_specified:
            ENV.update()
            if ENV.token_ids_save_enable:
                if self.local_rank == 0:
                    for idx, req in enumerate(req_list):
                        input_ids_save_filename = f"input_ids_{idx}.pth"
                        input_ids_save_path = os.path.join(ENV.token_ids_save_folder, input_ids_save_filename)
                        input_ids_save_path = file_utils.standardize_path(input_ids_save_path)
                        file_utils.check_file_safety(input_ids_save_path, 'w', is_check_file_size=False)
                        torch.save(req.input_ids.cpu(), input_ids_save_path)

                        output_ids_save_filename = f"output_ids_{idx}.txt"
                        output_ids_save_path = os.path.join(ENV.token_ids_save_folder, output_ids_save_filename)
                        with file_utils.safe_open(output_ids_save_path, 'w', encoding='utf-8') as f:
                            f.write(' '.join(map(str, req.out_token_list)))
        print_log(self.rank, logger.info, "---------------end inference---------------")
        return generate_text_list, token_num_list, e2e_time

    def _build_model_inputs(self, inputs, image_path, is_chat_model, is_truncation=False):

        def _history_to_prompt(signal_type, history, query):
            if signal_type == 'base':
                return query
            elif signal_type == 'vqa':
                answer_format = 'Short answer:'
            elif signal_type == 'chat':
                answer_format = 'Answer:'
            else:
                print_log(self.rank, logger.error, "error catched: "
                          + f"Unknown signal type {signal_type}")
            prompt = ""
            for _, (old_query, response) in enumerate(history):
                prompt += 'Question: ' + old_query + " {} ".format(answer_format) + response + "\n"
            prompt += 'Question: {} {}'.format(query, answer_format)
            return prompt

        input_texts, input_ids, input_conversations, input_adapters = [], [], [], []
        if isinstance(inputs, list) and inputs:
            if isinstance(inputs[0], str):
                input_texts = inputs
            elif isinstance(inputs[0], torch.Tensor):
                input_ids = inputs
            elif isinstance(inputs[0], list) and inputs[0]:
                if isinstance(inputs[0][0], int):
                    input_ids = inputs
                elif isinstance(inputs[0][0], dict):
                    input_conversations = inputs
            elif isinstance(inputs[0], dict):
                prompt_key = "prompt"
                if isinstance(inputs[0].get(prompt_key), torch.Tensor):
                    input_ids = [input.get(prompt_key) for input in inputs]
                else:
                    input_texts = [input.get(prompt_key) for input in inputs]
                input_adapters = [input.get("adapter") for input in inputs]
        if not (input_texts or input_ids or input_conversations):
            raise ValueError(f"The inputs of `PARunner.infer` must be as List[str], List[torch.Tensor], List[List[int]]"
                             f" or List[List[Dict]]. Now the inputs ({inputs}) is not acceptable or is empty.")
        if not input_ids and not self.tokenizer:
            raise ValueError("Can‘t encode input_texts，because load_tokenizer=False, "
                             "please enter input_ids or enable load_tokenizer")
        if is_chat_model:
            if input_conversations:
                input_ids = self.model.build_inputs(input_conversations)
            elif input_texts:
                input_conversations = [[{"role": "user", "content": t}] for t in input_texts]
                input_ids = self.model.build_inputs(input_conversations)
            else:
                print_log(self.rank, logger.warning, "Neither conversations nor input_texts exist, "
                                                     "'chat' parameter is not effective.")
        elif input_texts:
            input_texts = [_history_to_prompt("chat", [], text) for text in input_texts]
            mm_inputs = [
                [
                    {"image": image_path},
                    {"text": input_texts},
                ]
            ]
            input_ids = [self.tokenizer_wrapper.tokenize(single_input)
                         for single_input in mm_inputs]

            # token长度校验
            for item in input_ids:
                if len(item) > self.max_input_length:
                    print_log(self.rank, logger.warning,
                              "Num of tokens in input_text is larger than max_input_length. "
                              "Please shorten input_text to avoid out of memory.")
        if not input_adapters:
            input_adapters = [None] * len(input_ids)
        return input_ids, input_adapters


def cmd_bool(cmd_arg):
    if cmd_arg == "True":
        return True
    elif cmd_arg == "False":
        return False
    raise ValueError(f"{cmd_arg} should be a boolean")


def parse_bool(bool_str):
    bool_str = bool_str.lower()
    return bool_str == 'true'


def parse_list_of_json(list_json):
    return json.loads(list_json)


def parse_ids(list_str):
    return [int(item) for item in list_str.split(',')]


def input_texts_parser(value):
    if os.path.isfile(value):
        with file_utils.safe_open(value, 'r') as opened_file:
            return opened_file.read()
    else:
        return value


# define Argument Parser
def parse_arguments():
    store_true = 'store_true'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default="/data/datasets/cogvlm2-llama3-chinese-chat-19B/",
                        help="model and tokenizer path")
    parser.add_argument(
        '--input_texts',
        type=input_texts_parser,
        nargs='+',
        default=["<|reserved_special_token_0|> " * 0 + "Describe this image."])
    parser.add_argument(
        '--image_path',
        type=str,
        help="input image path",
        default="/path/of/image/demo.jpg")
    parser.add_argument(
        '--input_ids',
        type=parse_ids,
        nargs='+',
        default=None)
    parser.add_argument(
        '--input_file',
        type=str,
        help='This parameter is used to input multi-turn dialogue information in the form '
             'of a jsonl file, with each line in the format of a List[Dict]. Each dictionary '
             '(Dict) must contain at least two fields: "role" and "content".',
        default=None)
    parser.add_argument(
        '--input_dict',
        help="Lora input, accepted format: "
             "'[{\"prompt\": \"prompt in text\", \"adapter\": \"adapter id defined in lora_adapater param\"}]'",
        type=parse_list_of_json,
        default=None)
    parser.add_argument("--max_batch_size", type=int, default=1)
    parser.add_argument('--max_input_length', type=int, default=8192)
    parser.add_argument('--max_output_length', type=int, default=256)
    parser.add_argument('--max_position_embeddings', type=int, default=None)
    parser.add_argument('--max_prefill_tokens', type=int, default=-1)

    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument('--chat_template', type=str, default=None)
    parser.add_argument('--ignore_eos', action=store_true)
    parser.add_argument('--is_chat_model', action=store_true)
    parser.add_argument('--is_embedding_model', action=store_true)
    parser.add_argument('--load_tokenizer', type=parse_bool, default=True)
    parser.add_argument('--enable_atb_torch', default=True, action=store_true)
    parser.add_argument('--kw_args', type=str, default='', help='json input')

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
        **vars(args)
    }

    # 输入优先级input_dict > input_ids > input_texts
    if args.input_dict:
        infer_inputs = args.input_dict
    elif args.input_ids:
        infer_inputs = args.input_ids
    else:
        infer_inputs = args.input_texts

    if args.is_chat_model and args.input_file:
        conversations = []
        with file_utils.safe_open(args.input_file, 'r', encoding='utf-8') as file:
            for line in file_utils.safe_readlines(file):
                data_line = json.loads(line)
                conversations.append(data_line)
        infer_inputs = conversations

    pa_runner = PARunner(**input_dict)
    print_log(rank, logger.info, f'pa_runner: {pa_runner}')
    pa_runner.warm_up()

    infer_params = {
        # 手动配置多种输入
        "mm_inputs": [
            InputAttrs(args.input_texts, args.image_path),
            InputAttrs(["<|reserved_special_token_0|> " * 0 + "描述这张图片。"], args.image_path),
            InputAttrs(args.input_texts, "/path/of/image/demo.jpg"),
            InputAttrs(args.input_texts, "")
        ],
        "batch_size": args.max_batch_size,
        "max_output_length": args.max_output_length,
        "ignore_eos": args.ignore_eos,
        "is_chat_model": args.is_chat_model
    }

    generate_texts, token_nums, _ = pa_runner.infer(**infer_params)
    length = len(infer_inputs)
    for i, generate_text in enumerate(generate_texts):
        if i < length:
            print_log(rank, logger.info, f'Question[{i}]: {infer_inputs[i][-32:]}')
        if input_dict['is_embedding_model']:
            embedding_tensor_path = f"{os.getcwd()}/examples/embedding_tensor"
            print_log(rank, logger.info, f"Context[{i}]: \nembedding tensor path is: {embedding_tensor_path}")
        else:
            print_log(rank, logger.info, f'Answer[{i}]: {generate_text}')
            print_log(rank, logger.info, f'Generate[{i}] token num: {token_nums[i]}')