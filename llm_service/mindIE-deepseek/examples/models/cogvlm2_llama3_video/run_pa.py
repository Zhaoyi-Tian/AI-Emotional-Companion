# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import argparse
import json
import os
from dataclasses import dataclass
from typing import List

from atb_llm.utils.env import ENV
from atb_llm.utils.log import logger, print_log
from atb_llm.utils import file_utils
from examples.models.cogvlm2.run_pa import PARunner


OPERATOR_BOUND = 576


@dataclass
class InputAttrs:
    input_texts:List | None
    image_path:str | None


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


#define Argument Parser
def parse_arguments():
    store_true = 'store_true'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default="/data/datasets/cogvlm2-video-llama3-chat/",
                        help="model and tokenizer path")
    parser.add_argument(
        '--input_texts',
        type=input_texts_parser,
        nargs='+',
        default=["What's the general content of this video?"]) 
    parser.add_argument(
        '--image_path',
        type=str,
        help="input image path",
        default="/data/basketball.mp4")
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
    parser.add_argument('--max_input_length', type=int, default=4096)
    parser.add_argument('--max_output_length', type=int, default=128)
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
        "mm_inputs": [InputAttrs(args.input_texts, args.image_path)],
        "batch_size": args.max_batch_size,
        "max_output_length": args.max_output_length,
        "ignore_eos": args.ignore_eos,
        "is_chat_model": args.is_chat_model
    }
    generate_texts, token_nums, _ = pa_runner.infer(**infer_params)

    length = len(infer_inputs)
    for i, generate_text in enumerate(generate_texts):
        if i < length:
            print_log(rank, logger.info, f'Question[{i}]: {infer_inputs[i][-64:]}')
        if input_dict['is_embedding_model']:
            embedding_tensor_path = f"{os.getcwd()}/examples/embedding_tensor"
            print_log(rank, logger.info, f"Context[{i}]: \nembedding tensor path is: {embedding_tensor_path}")
        else:
            print_log(rank, logger.info, f'Answer[{i}]: {generate_text}')
            print_log(rank, logger.info, f'Generate[{i}] token num: {token_nums[i]}')
