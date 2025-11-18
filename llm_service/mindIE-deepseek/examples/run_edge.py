# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import argparse
import json

import os
import time
import torch

from atb_llm.runner.model_runner import ModelRunner

from atb_llm.utils.env import ENV
from atb_llm.utils.log import logger, print_log
from atb_llm.utils import file_utils

from examples.server.cache import ModelConfig

OPERATOR_BOUND = 576
MAX_KW_ARGS_LENGTH = 4096
PERF_FILE = "./examples/models/performance.csv"
PERF_COLUMNS = "batch, input_len, output_len, embedding_len, fisrt_token_time(ms), \
                non_first_token_time(ms), ResponseTime(ms),E2E Throughput Average(Tokens/s)\n"
PRED_FILE = "./examples/models/result.json"


class OtherRunner:
    def __init__(self, **kwargs):
        self.rank = kwargs.get('rank', '0')
        self.local_rank = kwargs.get('local_rank', self.rank)
        self.world_size = kwargs.get('world_size', '1')

        self.model_path = kwargs.get('model_path', None)
        self.lora_adapter = kwargs.get('lora_adapter', None)
        self.input_text = kwargs.get('input_text', None)

        self.max_prefill_batch_size = kwargs.get('max_prefill_batch_size', None)
        self.max_batch_size = kwargs.get('max_batch_size', 1)
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
        if len(kw_args) > MAX_KW_ARGS_LENGTH:
            raise ValueError(f'The length of kw_args is too long and should be within (0, {MAX_KW_ARGS_LENGTH}]')
        if kw_args:
            try:
                kw_args = json.loads(kw_args)
            except json.decoder.JSONDecodeError:
                print_log(self.local_rank, logger.warning, 'Please checkout input of kw_args!')
                kw_args = {}
        else:
            kw_args = {}

        self.model = ModelRunner(
            self.model_path, rank=self.rank, world_size=self.world_size,
            local_rank=self.local_rank,
            is_flash_causal_lm=False,
            max_position_embeddings=self.max_position_embeddings,
            enable_atb_torch=self.enable_atb_torch,
            enable_edge=True
        )

        self.tokenizer = self.model.tokenizer
        if self.chat_template:
            self.tokenizer.chat_template = self._load_chat_template(self.chat_template)
        self.dtype = self.model.dtype
        self.quantize = self.model.quantize
        self.kv_quant_type = self.model.kv_quant_type
        self.fa_quant_type = self.model.fa_quant_type
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
                                        self.kv_quant_type,
                                        self.fa_quant_type,
                                        self.model.mapping)

        self.warm_up_memory = 0
        self.warm_up_num_blocks = 0
        self.cache_manager = None
        self.compress_head_enable = ENV.compress_head_enable

    def __repr__(self):
        return (
                "OtherRunner("
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

    def warm_up(self):
        print_log(self.rank, logger.info, "---------------begin warm_up---------------")

        input_ids = torch.ones(self.max_input_length, dtype=torch.int64)
        inputs = self.tokenizer(str(input_ids), return_tensors="pt")

        position_ids_test = torch.arange(start=0, end=len(inputs.input_ids[0]), step=1,
                                         dtype=torch.int32, device="npu")
        position_ids_test = position_ids_test.reshape((1, -1), torch.int32)

        # prefill 直接调用prefill，返回logits，kvcache
        with torch.no_grad():
            self.model.forward(
                input_ids=inputs.input_ids.npu(),
                attention_mask=inputs.attention_mask.npu(),
                position_ids=position_ids_test,
                past_key_values=None
            )
        print_log(self.rank, logger.info, "---------------end warm_up---------------")

    def infer(self, inputs, max_output_length):
        print_log(self.rank, logger.info, "---------------begin inference---------------")
        inputs = self.tokenizer(inputs, return_tensors="pt")

        position_ids_prefill = torch.arange(start=0, end=len(inputs.input_ids[0]), step=1,
                                            dtype=torch.int32, device="npu")
        position_ids_prefill = position_ids_prefill.reshape((1, -1), torch.int32)

        with torch.no_grad():
            logits, past_key_values = self.model.forward(
                input_ids=inputs.input_ids.npu(),
                attention_mask=inputs.attention_mask.npu(),
                position_ids=position_ids_prefill,
                past_key_values=None
            )
        logits = logits[0][-1].reshape(1, -1)

        decode_position_ids = torch.Tensor([len(inputs.input_ids[0])])
        decode_position_ids = decode_position_ids.reshape(1, -1)
        attention_mask = torch.ones(1, 3000).npu()

        logits = logits.argmax(dim=-1)
        output = []
        output.append(logits[0])
        logits = logits.reshape(1, -1)
        attention_mask = attention_mask[:, :len(inputs.input_ids[0]) + 1]

        torch.npu.synchronize()
        decode_start_time = time.time()
        decode_times = 0
        for index in range(max_output_length):
            if index == 1:
                decode_start_time = time.time()
            with torch.no_grad():
                logits, past_key_values = self.model.forward(
                    input_ids=logits,
                    attention_mask=attention_mask,
                    position_ids=decode_position_ids,
                    past_key_values=past_key_values
                )
                logits = logits[0][-1].reshape(1, -1)
                logits = logits.argmax(dim=-1)
                decode_times = decode_times + 1
                output.append(logits[0])
                logits = logits.reshape(1, -1)
                decode_position_ids = torch.Tensor([len(inputs.input_ids[0]) + index + 1])
                decode_position_ids = decode_position_ids.reshape(1, -1)

        decode_end_time = time.time()
        averaged_decode_time = (decode_end_time - decode_start_time) * 1000 / (decode_times - 1)
        print_log(rank, logger.info, f'Average decode time: {averaged_decode_time}')

        torch.npu.synchronize()
        out_list = []
        for out in output:
            out_list.append(out.item())
        response = self.tokenizer.decode(out_list)

        print_log(self.rank, logger.info, "---------------end inference---------------")
        return response, decode_times, averaged_decode_time


def cmd_bool(cmd_arg):
    if cmd_arg == "True":
        return True
    elif cmd_arg == "False":
        return False
    raise ValueError(f"{cmd_arg} should be a boolean")


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
    parser.add_argument('--model_path', type=str, help="model and tokenizer path")
    parser.add_argument(
        '--input_texts',
        type=input_texts_parser,
        nargs='+',
        default=["who are you?"])
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

    parser.add_argument("--max_prefill_batch_size", type=int, default=None)
    parser.add_argument("--max_batch_size", type=int, default=1)
    parser.add_argument('--max_input_length', type=int, default=1024)
    parser.add_argument('--max_output_length', type=int, default=3)
    parser.add_argument('--max_position_embeddings', type=int, default=None)
    parser.add_argument('--max_prefill_tokens', type=int, default=-1)

    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument('--chat_template', type=str, default=None)
    parser.add_argument('--ignore_eos', action=store_true)
    parser.add_argument('--is_chat_model', action=store_true)
    parser.add_argument('--is_embedding_model', action=store_true)
    parser.add_argument('--load_tokenizer', type=cmd_bool, default=True)
    parser.add_argument('--enable_atb_torch', action=store_true)
    parser.add_argument('--dp', type=int, default=-1)
    parser.add_argument('--tp', type=int, default=-1)
    parser.add_argument('--moe_tp', type=int, default=-1)
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

    other_runner = OtherRunner(**input_dict)
    print_log(rank, logger.info, f'other_runner: {other_runner}')

    infer_params = {
        "inputs": infer_inputs,
        "max_output_length": args.max_output_length
    }
    generate_texts, token_nums, average_decode_time = other_runner.infer(**infer_params)

    print_log(rank, logger.info, f'Question: {infer_inputs[0]}')
    print_log(rank, logger.info, f'Answer: {generate_texts}')
    print_log(rank, logger.info, f'Generate token num: {token_nums}')
    print_log(rank, logger.info, f'average_decode_time: {average_decode_time} ms')
