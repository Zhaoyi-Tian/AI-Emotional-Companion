# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import torch
import torch.nn.functional as F
from atb_llm.utils.log import logger
from examples.models.clip.run import MultiModalRunner


def compare_output(args):
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    input_dict = {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "using_fp16": True,
        **vars(args),
    }

    mm_inputs = [
        [
            {"image": [args.input_image, args.input_image]},
            {"text": args.label_list.split(",")}
        ]
    ]

    npu_runner = MultiModalRunner(**input_dict)
    npu_prob, _, _ = npu_runner.infer(mm_inputs, batch_size=2)

    input_dict["using_cpu"] = True
    input_dict["using_fp16"] = False  # slow_con2d_cpu used in patch_embedding not implemented for fp16
    cpu_runner = MultiModalRunner(**input_dict)

    cpu_prob, _, _ = cpu_runner.infer(mm_inputs, batch_size=2)

    sim_res = F.cosine_similarity(npu_prob.to(torch.float32).cpu().flatten(), cpu_prob.flatten(), dim=0, eps=1e-6)
    output_res = sim_res.cpu().detach().item()
    return output_res


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/data/chinese-clip-vit-base-patch16/")
    parser.add_argument('--sim_threshold', type=float, default=0.99)
    parser.add_argument(
        "--label_list",
        help="Used when the number of labels is small. Separate the labels with ',' .",
        default="Bulbasaur,Ivysaur,Charmander,Pikachu",
    )
    parser.add_argument(
        "--input_image",
        help="single image path",
        default="./examples/models/clip/pokemon.jpeg",
    )
    return parser.parse_args()


if __name__ == "__main__":
    input_args = parse_arguments()
    res = compare_output(input_args)
    logger.info(f"cosine similarity: {res}")
