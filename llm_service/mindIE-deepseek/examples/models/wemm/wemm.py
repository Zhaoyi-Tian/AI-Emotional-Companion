# Copyright Huawei Technologies Co., Ltd. 2024-2028. All rights reserved.
import math
import os

from atb_llm.utils.env import ENV
from atb_llm.utils.file_utils import safe_listdir, standardize_path, check_file_safety
from atb_llm.utils.log import logger, print_log
from atb_llm.utils import argument_utils
from atb_llm.utils.argument_utils import BooleanArgumentValidator, ArgumentAction
from atb_llm.utils.multimodal_utils import MultimodalInput
from examples.multimodal_runner import MultimodalPARunner, parser
from examples.multimodal_runner import path_validator


PERF_FILE = "./examples/models/wemm/wemm_performance.csv"
PRED_FILE = "./examples/models/wemm/predict_result.json"


class WeMMRunner(MultimodalPARunner):
    def __init__(self, **kwargs):
        self.processor = None
        super().__init__(**kwargs)
        self.adapter_id = kwargs.get("lora_adapter_id", None)

    def init_processor(self):
        self.processor = self.model.tokenizer

    def precision_save(self, precision_inputs, **kwargs):
        all_generate_text_list = precision_inputs.all_generate_text_list
        image_file_list = precision_inputs.image_file_list
        image_answer_pairs = {}
        for image_file, generate_text in zip(image_file_list, all_generate_text_list):
            image_answer_pairs[image_file] = generate_text
        image_answer_pairs = dict(sorted(image_answer_pairs.items()))
        super().precision_save(precision_inputs, answer_pairs=image_answer_pairs)

    def infer(self, mm_inputs, batch_size, max_output_length, ignore_eos, **kwargs):
        input_texts = mm_inputs.input_texts
        image_path_list = mm_inputs.image_path
        if len(input_texts) != len(image_path_list):
            raise RuntimeError("input_text length must equal input_images length")
        if not ENV.profiling_enable:
            if self.max_batch_size > 0:
                max_iters = math.ceil(len(mm_inputs.image_path) / self.max_batch_size)
            else:
                raise RuntimeError("f{self.max_batch_size} max_batch_size should > 0, please check")
        return super().infer(mm_inputs, batch_size, max_output_length, ignore_eos, max_iters=max_iters)


def parse_arguments():
    parser_wemm = parser
    string_validator = argument_utils.StringArgumentValidator(min_length=0, max_length=1000)
    list_str_validator = argument_utils.ListArgumentValidator(string_validator, max_length=1000)
    bool_validator = BooleanArgumentValidator()

    parser_wemm.add_argument(
        '--image_or_video_path',
        help="image_or_video path",
        default="/image/path",
        validator=path_validator)
    parser_wemm.add_argument(
        '--input_texts_for_image',
        type=str,
        nargs='+',
        default=["Explain the details in the image."],
        validator=list_str_validator)
    parser_wemm.add_argument(
        '--lora_adapter_id',
        help="Lora input, accepted adapter id defined in lora_adapater param",
        type=str,
        default="wemm",
        validator=string_validator)
    parser_wemm.add_argument(
        '--skip_special_tokens', 
        action=ArgumentAction.STORE_TRUE.value, 
        validator=bool_validator)

    return parser_wemm.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    rank = ENV.rank
    local_rank = ENV.local_rank
    world_size = ENV.world_size

    image_or_video_path = standardize_path(args.image_or_video_path)
    check_file_safety(image_or_video_path, 'r')
    file_name = safe_listdir(image_or_video_path)
    image_path = [os.path.join(image_or_video_path, f) for f in file_name]
    texts = args.input_texts_for_image
    image_length = len(image_path)

    if len(texts) != image_length:
        texts.extend([texts[-1]] * (image_length - len(texts)))

    remainder = image_length % args.max_batch_size
    if remainder != 0:
        num_to_add = args.max_batch_size - remainder
        image_path.extend([image_path[-1]] * num_to_add)
        texts.extend([texts[-1]] * num_to_add)
    
    input_dict = {
        'rank': rank,
        'world_size': world_size,
        'local_rank': local_rank,
        'perf_file': PERF_FILE,
        'pred_file': PRED_FILE,
        'image_path': image_path,
        'input_texts': texts,
        'lora_adapter_id': args.lora_adapter_id,
        **vars(args)
    }

    pa_runner = WeMMRunner(**input_dict)
    print_log(rank, logger.info, f'pa_runner: {pa_runner}')
    
    infer_params = {
        "mm_inputs": MultimodalInput(texts,
                                image_path,
                                None,
                                None),
        "batch_size": args.max_batch_size,
        "max_output_length": args.max_output_length,
        "ignore_eos": args.ignore_eos,
        "skip_special_tokens": True
    }
    pa_runner.warm_up()
    generate_texts, token_nums, latency = pa_runner.infer(**infer_params)
    all_token_nums = 0
    for i, generate_text in enumerate(generate_texts):
        print_log(rank, logger.info, f'Answer[{i}]: {generate_text}')
        if i % args.max_batch_size == 0:
            token_num = token_nums[i][1]
        else:
            token_num = token_nums[i][1] - token_nums[i - 1][1]
        print_log(rank, logger.info, f'Generate[{i}] token num: {token_num}')
        all_token_nums += token_num
    print_log(rank, logger.info, f"All Token Nums: {all_token_nums}")
    print_log(rank, logger.info, f"Latency(s): {latency}")
    print_log(rank, logger.info, f"E2E_Throughput(tokens/s): {all_token_nums / latency}")
    