# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import argparse
import json
import math
import os
import time
import torch
import torch_npu
from PIL import Image
from atb_llm.models import get_model
from atb_llm.utils.cpu_binding import bind_cpus, NpuHbmInfo
from atb_llm.utils.dist import initialize_distributed
from atb_llm.utils.env import ENV
from atb_llm.utils.file_utils import safe_open
from atb_llm.utils.initial import NPUSocInfo
from atb_llm.utils.log import logger, print_log


STORE_TRUE = "store_true"


class ClipRequest():
    def __init__(self,
                 image_embeds: torch.Tensor,
                 text_features: torch.Tensor,
                 req_id: int,
                 ):
        self.req_id = req_id
        self.image_embeds = image_embeds
        self.text_features = text_features

        self.out_probs = None
        self.out_label_idx = None


class MultiModalRunner:
    def __init__(self, **kwargs):
        self.rank = kwargs.get("rank", "0")
        self.local_rank = kwargs.get("local_rank", self.rank)
        self.world_size = kwargs.get("world_size", "1")
        self.npu_id = kwargs.get("npu_id", self.local_rank)
        self.model_path = kwargs.get("model_path", None)
        self.input_text = kwargs.get("input_text", None)
        self.max_batch_size = kwargs.get("max_batch_size", None)
        self.is_flash_model = kwargs.get("is_flash_model", False)
        self.using_cpu = kwargs.get("using_cpu", False)
        self.using_fp16 = kwargs.get("using_fp16", False)
        self.warmup_image_path = kwargs.get("warmup_image_path", None)

        self.soc_info = NPUSocInfo()
        self.max_memory = NpuHbmInfo.get_hbm_capacity(
            self.local_rank, self.world_size, self.soc_info.need_nz
        )
        self.used_memory = int(
            self.max_memory
            * NpuHbmInfo.get_hbm_usage(
                self.local_rank, self.world_size, self.soc_info.need_nz
            )
        )
        if self.using_cpu:
            self.device = "cpu"
        else:
            if ENV.bind_cpu:
                try:
                    bind_cpus(self.world_size, self.npu_id, ratio=1.0)
                except RuntimeError as e:
                    print_log(self.rank, logger.info, e)
                except Exception as _:
                    print_log(self.rank, logger.info, 'Skip binding cpu.')

            self.process_group, device = initialize_distributed(self.rank, self.npu_id, self.world_size)
            torch.npu.set_compile_mode(jit_compile=False)
            self.device = device

        router_ins = get_model(self.model_path, is_flash_causal_lm=False)
        self.model_cls = router_ins.model_cls
        self.config = router_ins.config
        self.torch_dtype = self.config.torch_dtype if not self.using_fp16 else torch.float16
        self.tokenizer = router_ins.tokenizer
        self.processor = router_ins.processor
        self.image_processor = router_ins.image_processor
        self.model = self.model_cls(self.config, using_fp16=self.using_fp16)
        self.model.to(self.device)
        print_log(self.rank, logger.info, f'model:\n {self.model}')

        self.max_memory = NpuHbmInfo.get_hbm_capacity(
            self.local_rank, self.world_size, self.soc_info.need_nz
        )
        self.init_memory = int(
            self.max_memory
            * NpuHbmInfo.get_hbm_usage(
                self.local_rank, self.world_size, self.soc_info.need_nz
            )
        )
        self.init_memory = self.init_memory - self.used_memory
        print_log(
            self.rank,
            logger.info,
            f"hbm_capacity(GB): {self.max_memory / (1024 ** 3): .2f}, "
            f"init_memory(GB): {self.init_memory / (1024 ** 3): .2f}",
        )

        self.warm_up_memory = 0

    def __repr__(self):
        return (
            f"MultiModalRunner(model_path={self.model_path}, "
            f"input_text={self.input_text}, "
            f"is_flash_model={self.is_flash_model}, "
            f"max_batch_size={self.max_batch_size}, "
            f"dtype={self.torch_dtype}, "
            f"max_memory={self.max_memory}, "
        )

    def warm_up(self):
        print_log(self.rank, logger.info, "--------------begin warm_up--------------")
        warmup_image_path = self.warmup_image_path
        image = Image.open(warmup_image_path)
        texts = ["Bulbasaur", "Ivysaur", "Charmander", "Pikachu"]
        candidate_labels = " ".join(texts)
        print_log(self.rank, logger.info, f"The warm_up image address is {warmup_image_path}")
        print_log(self.rank, logger.info, f"The candidate labels are {candidate_labels}")

        # using huggingface encapsulated processor and forward
        return_tensors_type = "pt"
        union_inputs = self.processor(text=texts, 
                                      images=image, 
                                      return_tensors=return_tensors_type, 
                                      padding=True).to(self.device)
        outputs = self.model(**union_inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # NPU: [[1.2670e-03, 5.5052e-02, 6.8762e-04, 9.4299e-01]]
        predicted_label = texts[probs.argmax(-1).item()]

        # using local split flow
        image_inputs = self.image_processor(images=image, return_tensors=return_tensors_type).to(self.device)
        texts_inputs = self.tokenizer(text=texts, padding=True, return_tensors=return_tensors_type).to(self.device)
        image_features = self.model.extract_image_features(**image_inputs)
        text_features = self.model.extract_text_features(**texts_inputs)
        logits_per_image = self.model.get_logits(image_features, text_features)
        probs = logits_per_image.softmax(dim=1)  # NPU: [[1.2670e-03, 5.5052e-02, 6.8762e-04, 9.4299e-01]]
        predicted_label = texts[probs.argmax(-1).item()]
        print_log(self.rank, logger.info, f"The label of the picture is {predicted_label}")
        image.close()
        self.warm_up_memory = int(
            self.max_memory
            * NpuHbmInfo.get_hbm_usage(
                self.local_rank, self.world_size, self.soc_info.need_nz
            )
        )
        self.warm_up_memory = self.warm_up_memory - self.used_memory
        print_log(
            self.rank,
            logger.info,
            f"warmup_memory(GB): {self.warm_up_memory / (1024 ** 3): .2f}",
        )
        print_log(self.rank, logger.info, "---------------end warm_up---------------")

    def prepare_text_features(
            self,
            text_inputs,
            model,
    ):
        all_text_features = []
        for input_text in text_inputs:
            tokenized_id = self.tokenizer(text=input_text, padding=True, return_tensors="pt").to(self.device)
            single_text_features = model.extract_text_features(**tokenized_id)
            all_text_features.append(single_text_features)
        text_features = torch.squeeze(torch.stack(all_text_features, dim=0), dim=1)
        return text_features

    def generate_request_list(
            self,
            mm_inputs,
            batch_size,
            model,
    ):
        req_list = []
        image_key = "image"
        # By default, there is only one dataset.
        if batch_size == 0:
            raise RuntimeError("generate request list failed, the batch size is 0. ")
        max_iters = math.ceil(len(mm_inputs[0][0][image_key]) / batch_size)
        text_features = self.prepare_text_features(mm_inputs[0][1]["text"], model)

        for current_iter in range(max_iters):
            process_begin = current_iter * batch_size
            image_list = []
            for i in range(batch_size):
                if process_begin + i >= len(mm_inputs[0][0][image_key]):
                    break
                image_list.append(Image.open(mm_inputs[0][0][image_key][process_begin + i]))
            image_embeds = self.image_processor(images=image_list, return_tensors="pt")  # using cpu
            for image in image_list:
                image.close()
            request = ClipRequest(image_embeds.pixel_values, text_features, current_iter)
            req_list.append(request)

        return req_list

    def generate_logits(
            self,
            req_list,
            model,
    ):
        with torch.no_grad():
            for single_req in req_list:
                image_embeds = single_req.image_embeds.to(self.device)
                image_features = model.extract_image_features(image_embeds)
                logits = model.get_logits(image_features, single_req.text_features)
                probs = logits.softmax(dim=1)
                single_req.out_label_idx = probs.argmax(-1).cpu()
                single_req.out_probs = probs.cpu()

    def infer(
            self,
            mm_inputs,
            batch_size,
            **kwargs
    ):
        print_log(
            self.rank, logger.info, "-------------begin inference-------------"
        )

        req_list = self.generate_request_list(mm_inputs, batch_size, self.model)

        print_log(
            self.rank, logger.info, "-------generate req_list finished--------"
        )

        if not ENV.profiling_enable:
            print_log(self.rank, logger.debug, "no profiling")
            torch.npu.synchronize()
            e2e_start = time.time()
            self.generate_logits(
                req_list,
                self.model,
            )
            torch.npu.synchronize()
            e2e_end = time.time()
            e2e_time = e2e_end - e2e_start
        else:
            print_log(self.rank, logger.debug, "enter profiling")
            profiling_path = ENV.profiling_filepath
            if not os.path.exists(profiling_path):
                os.makedirs(profiling_path, exist_ok=True)
            torch.npu.synchronize()
            e2e_start = time.time()
            experimental_config = torch_npu.profiler._ExperimentalConfig(
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
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
                self.generate_logits(
                    req_list,
                    self.model,
                )
            torch.npu.synchronize()
            e2e_end = time.time()
            e2e_time = e2e_end - e2e_start

        # By default, there is only one dataset.
        labels = []
        label_idx = [req.out_label_idx for req in req_list]
        for tmp in label_idx:
            labels.extend([mm_inputs[0][1]["text"][item] for item in tmp.tolist()])

        if req_list[-1].out_probs.shape != req_list[0].out_probs.shape:
            add_batch_pad = req_list[0].out_probs.shape[0] - req_list[-1].out_probs.shape[0]
            req_list[-1].out_probs = torch.nn.functional.pad(req_list[-1].out_probs, (0, 0, 0, add_batch_pad))
        probs_list = [req.out_probs for req in req_list]
        probs = torch.squeeze(torch.stack(probs_list, dim=0))

        print_log(self.rank, logger.info, "--------------end inference--------------")
        return probs, labels, e2e_time


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default=None,
    )
    parser.add_argument(
        "--input_text",
        default="The label for this picture is _",
        help="This is a label template. Underscore '_' is used to indicate the label position. \
              Only one underscore '_' is allowed in the sentence.",
    )
    parser.add_argument(
        "--label_file",
        help="Used when the number of labels is large. Each line contains a label in label file.",
        default=None,
    )
    parser.add_argument(
        "--label_list",
        help="Used when the number of labels is small. Separate the labels with ',' .",
        default=None,
    )
    parser.add_argument(
        "--input_image",
        help="single image path",
        default=None,
    )
    parser.add_argument(
        "--dataset_path",
        help="precision test dataset path",
        default=None
    )
    parser.add_argument(
        "--warmup_image_path",
        help="single warm_up image path",
        default=None
    )
    parser.add_argument(
        "--results_save_path",
        help="precision test result path",
        default="./res.json",
    )

    parser.add_argument("--max_position_embeddings", type=int, default=None)
    parser.add_argument("--max_input_length", type=int, default=1024)
    parser.add_argument("--max_output_length", type=int, default=65)
    parser.add_argument("--max_prefill_tokens", type=int, default=-1)
    parser.add_argument("--max_batch_size", type=int, default=8)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--quantize", type=str, default=None)
    parser.add_argument("--is_flash_model", action=STORE_TRUE)
    parser.add_argument("--using_cpu", action=STORE_TRUE)
    parser.add_argument("--using_fp16", action=STORE_TRUE)

    return parser.parse_args()


def deal_dataset(dataset_path):
    input_images = []
    images_list = os.listdir(dataset_path)
    for img_name in images_list:
        image_path = os.path.join(dataset_path, img_name)
        input_images.append(image_path)

    return input_images


def main():
    args = parse_arguments()
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    input_dict = {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        **vars(args),
    }

    runner = MultiModalRunner(**input_dict)
    print_log(rank, logger.info, f"runner: {runner}")
    runner.warm_up()

    try:
        if args.label_list:
            label_texts = args.label_list.split(",")
        else:
            f = open(args.label_file, "r", encoding="utf8")
            if args.input_text:
                label_texts = [
                    args.input_text.replace("_", line.strip())
                    for line in f.readlines()
                ]
            else:
                label_texts = [
                    line.strip()
                    for line in f.readlines()
                ]
            f.close()
    except FileNotFoundError as e:
        raise FileNotFoundError("label_file does not exist and the label_list is not provided.") from e

    if args.dataset_path:
        dataset_images = deal_dataset(args.dataset_path)
    else:
        dataset_images = [args.input_image]

    mm_inputs = [
        [
            {"image": [i for i in dataset_images]},
            {"text": label_texts}
        ]
    ]

    probs, labels, latency = runner.infer(
        mm_inputs,
        args.max_batch_size,
    )

    res = {}
    for i, label in enumerate(labels):
        rst_key = dataset_images[i].split("/")[-1]
        res[rst_key] = label

    if args.dataset_path:
        sorted_dict = dict(sorted(res.items()))
        with safe_open(
                args.results_save_path,
                "w",
                override_flags=os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        ) as f:
            json.dump(sorted_dict, f)

    elif args.input_image:
        print_log(
            rank, logger.info, f"label of {args.input_image}: {labels[0]}"
        )

    print_log(
        rank, logger.info, f"generate_time: {latency}"
    )

    print_log(
        rank, logger.info, "--------npu precision test finish--------"
    )


if __name__ == "__main__":
    main()
