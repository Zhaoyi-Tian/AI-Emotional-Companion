# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import argparse
import json
import os
import time
import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from atb_llm.runner.model_runner import ModelRunner
from atb_llm.utils.log import logger, print_log
from atb_llm.utils.file_utils import safe_open

MAX_KW_ARGS_LENGTH = 4096


def input_texts_parser(value):
    if os.path.isfile(value):
        with open(value, 'r') as opened_file:
            return opened_file.read()
    else:
        return value


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help="model and tokenizer path")
    parser.add_argument(
        '--input_text',
        type=input_texts_parser,
        nargs='+',
        default=["What's deep learning?"])
    parser.add_argument(
        '--input_file',
        type=str,
        help='CSV or Numpy file containing tokenized input. Alternative to text input.',
        default=None)
    parser.add_argument('--max_input_length', type=int, default=10)
    parser.add_argument('--max_output_length', type=int, default=20)
    parser.add_argument('--max_position_embeddings', type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument('--is_flash_causal_lm', action='store_true')

    parser.add_argument('--num_beams',
                        type=int,
                        help="Use beam search if num_beams >1",
                        default=1)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=0.0)
    parser.add_argument('--length_penalty', type=float, default=1.0)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)

    parser.add_argument('--inputs_embeds_dir', type=str, default=None,
                        help='Directory of .pt files containing inputs_embeds.')
    parser.add_argument('--min_length', type=int, default=10)
    parser.add_argument('--stop_words_ids', type=json.loads, default=None)
    parser.add_argument('--do_sample', type=bool, default=False)
    parser.add_argument('--results_save_path', type=str, default=None,
                        help='File path to save inference results.')
    parser.add_argument('--is_chat_model', action='store_true')

    return parser.parse_args()


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=None):
        super().__init__()
        if stops is None:
            stops = []
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        for stop in self.stops:
            if torch.all(torch.eq(input_ids[:, -len(stop):], stop)).item():
                return True
        return False


class InputData:
    def __init__(self):
        self.input_ids = None
        self.attention_mask = None


class FARunner:
    def __init__(self, **kwargs):
        self.rank = kwargs.get('rank', '0')
        self.local_rank = kwargs.get('local_rank', self.rank)
        self.world_size = kwargs.get('world_size', '1')

        self.model_path = kwargs.get('model_path', None)
        self.max_input_length = kwargs.get('max_input_length', None)    
        self.max_output_length = kwargs.get('max_output_length', None)
        self.max_position_embeddings = kwargs.get('max_position_embeddings', None)
        self.is_flash_causal_lm = kwargs.get('is_flash_causal_lm', False)
        self.batch_size = kwargs.get('batch_size', None)
        self.is_chat_model = kwargs.get('is_chat_model', False)
        
        
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
            self.model_path, 
            rank=self.rank, 
            world_size=self.world_size,
            local_rank=self.local_rank,
            is_flash_causal_lm=self.is_flash_causal_lm,
            max_position_embeddings=self.max_position_embeddings,
            enable_edge=True
        )
        self.tokenizer = self.model.tokenizer
        self.device = self.model.device
        self.dtype = self.model.dtype
        self.quantize = self.model.quantize

        self.model.load_weights(**kwargs)

        self.skip_word_embedding = False
        if hasattr(self.model.model, 'skip_word_embedding'):
            self.skip_word_embedding = self.model.model.skip_word_embedding

    def warm_up(self):
        print_log(self.rank, logger.info, "---------------begin warm_up---------------")
        dummy_input_ids_full = torch.randint(
            0, 32000, [self.batch_size, self.max_input_length], dtype=torch.long).npu()
        try:
            self.model.generate(inputs=dummy_input_ids_full, do_sample=False, max_new_tokens=10)
        except Exception as e:
            raise FileNotFoundError("warm up fail.") from e
        print_log(self.rank, logger.info, "---------------end warm_up---------------")

    def infer(self, input_text):
        print_log(self.rank, logger.info, "---------------begin inference---------------")
        if isinstance(input_text, str):
            input_text = [input_text] * self.batch_size
        is_chat_model = self.is_chat_model
        inputs = InputData()
        if is_chat_model:
            input_conversations = [[{"role": "user", "content": t}] for t in input_text]
            chat_input_ids = self.model.build_inputs(input_conversations)
            if not isinstance(chat_input_ids, torch.Tensor):
                chat_input_ids = torch.tensor(chat_input_ids)
            chat_attention_mask = torch.ones_like(chat_input_ids)
            inputs.input_ids = chat_input_ids
            inputs.attention_mask = chat_attention_mask
        else:
            inputs = self.tokenizer(input_text, return_tensors="pt", max_length=self.max_input_length)
        
        prefill_start_time = time.time()
        with torch.no_grad():
            self.model.generate(
                inputs=inputs.input_ids.npu(),
                attention_mask=inputs.attention_mask.npu(),
                max_new_tokens=1
            )
        prefill_end_time = time.time()

        decode_start_time = time.time()      
        with torch.no_grad():
            generate_ids = self.model.generate(
                inputs=inputs.input_ids.npu(),
                attention_mask=inputs.attention_mask.npu(),
                do_sample=False,
                max_new_tokens=self.max_output_length
            )
        decode_end_time = time.time()

        generate_text = self.tokenizer.batch_decode(
            generate_ids[:, len(inputs.input_ids[0]):], 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        if self.rank == 0:

            logger.info(f'{inputs.input_ids.shape=}')

            input_tokens_num = len(inputs.input_ids[0])
            generate_tokens_num = len(generate_ids[0]) - len(inputs.input_ids[0])
            logger.info(f'Question: {input_text[0]}')
            logger.info(f'Answer: {generate_text[0]}')
            logger.info(f'Input token num: {input_tokens_num}')
            logger.info(f'Generate token num: {generate_tokens_num}')

            logger.info("---------------end inference---------------")

            prefill_time = (prefill_end_time - prefill_start_time)
            decode_time = (decode_end_time - decode_start_time)
            total_time = prefill_time + decode_time

            try:
                decode_average_time = (decode_time) / (self.max_output_length - 1)
                logger.info(
                f"Decode average time(ms): {decode_average_time * 1000}")
            except ZeroDivisionError as e:
                raise ZeroDivisionError from e
            logger.info(
                f"Total time(ms): {total_time * 1000}, "
                f"First token time(ms): {prefill_time * 1000}, "
                f"None-first token time(ms): {decode_time * 1000}, "
                f"Non-first token Throughout(Token/s): {(self.max_output_length - 1) / decode_time}, "
                f"Throughout(Token/s): {self.max_output_length / total_time}")

    def infer_from_embeds(self, args):
        if rank == 0:
            logger.info("---------------begin inference---------------")

        stop_words_ids = [torch.tensor(ids).npu() for ids in args.stop_words_ids]
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

        generation_args = {
            "inputs_embeds": None,
            'min_length': args.min_length,
            'max_new_tokens': args.max_output_length,
            "stopping_criteria": stopping_criteria,
            'do_sample': args.do_sample,
            'num_beams': args.num_beams,
            'top_p': args.top_p,
            'temperature': args.temperature,
            'repetition_penalty': args.repetition_penalty,
            'length_penalty': args.length_penalty,
        }

        image_answer_pairs = {}
        for inputs_embeds_file_path in sorted([os.path.join(args.inputs_embeds_dir, _)
                                               for _ in os.listdir(args.inputs_embeds_dir)]):

            if not inputs_embeds_file_path.endswith(".pt"):
                continue

            if rank == 0:
                logger.info(f'NO.{len(image_answer_pairs) + 1}')
                logger.info(f'inputs_embeds_file_path: {inputs_embeds_file_path}')

            inputs_embeds = torch.load(inputs_embeds_file_path).npu()
            inputs_embeds = inputs_embeds.to(self.dtype)
            generation_args["inputs_embeds"] = inputs_embeds

            with torch.no_grad():
                generate_ids = self.model.generate(**generation_args)

            output_text = self.tokenizer.decode(generate_ids[0], skip_special_tokens=True)
            output_text = output_text.split('###')[0]  # remove the stop sign '###'
            output_text = output_text.split('Assistant:')[-1].strip()
            image_answer_pairs[inputs_embeds_file_path] = output_text

            if rank == 0:
                logger.info(f'Answer: {output_text}')
                with safe_open(args.results_save_path, "w", encoding='utf-8') as f:
                    json.dump(image_answer_pairs, f)
                logger.info('json dump finished')

        if rank == 0:
            logger.info("---------------end inference---------------")


if __name__ == '__main__':
    arguments = parse_arguments()

    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    input_dict = {
        'rank': rank,
        'world_size': world_size,
        'local_rank': local_rank,
        **vars(arguments)
    }

    fa_runner = FARunner(**input_dict)

    if fa_runner.skip_word_embedding:
        fa_runner.infer_from_embeds(arguments)
    else:
        fa_runner.warm_up()
        fa_runner.infer(arguments.input_text)
