# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
# Copyright (c) 2023-2024 DeepSeek.

import argparse
import json
import os
import sys

import PIL
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, StoppingCriteria, AutoConfig

from atb_llm.models.base.model_utils import safe_from_pretrained
from atb_llm.utils import argument_utils
from atb_llm.utils.argument_utils import BooleanArgumentValidator, ArgumentAction
from atb_llm.utils.log import logger
from atb_llm.models.deepseekvl.conversation import Conversation
from atb_llm.models.deepseekvl.processing_vlm import VLChatProcessor
from examples.models.deepseekvl.precision.modeling_vlm import VisionConfig, AlignerConfig, DeepseekvlConfig, \
    DeepseekvlCausalLM

torch.manual_seed(1234)
OUTPUT_JSON_PATH = "./gpu_coco_predict.json"


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=None):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ):
        for stop in self.stops:
            if input_ids.shape[-1] < len(stop):
                continue
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False


def generate_prompt_with_conversation(conversation: Conversation):
    prompts = []
    messages = conversation.messages

    for i in range(0, len(messages), 2):
        prompt = {
            "role": messages[i][0],
            "content": (
                messages[i][1][0]
                if isinstance(messages[i][1], tuple)
                else messages[i][1]
            ),
            "images": [messages[i][1][1]] if isinstance(messages[i][1], tuple) else [],
        }
        response = {"role": messages[i + 1][0], "content": messages[i + 1][1]}
        prompts.extend([prompt, response])

    return prompts


def get_all_images(image_path):
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}  # Add or remove extensions as needed
    image_list = []

    for dirpath, _, filenames in os.walk(image_path):
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() in image_extensions:
                image_list.append(os.path.join(dirpath, filename))

    return image_list


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--model_path",
                        required=True,
                        help="Model and tokenizer path.")
    parser.add_argument("--image_path",
                        required=True,
                        help="Image path for inference.")
    bool_validator = BooleanArgumentValidator()
    parser = argument_utils.ArgumentParser(description="Demo")
    parser.add_argument('--trust_remote_code', action=ArgumentAction.STORE_TRUE.value, 
                        validator=bool_validator)
    return parser.parse_args()


def main():
    device = torch.device('cuda', 0)
    args = parse_args()
    model_path = args.model_path
    image_path = args.image_path
    if os.path.exists(model_path) and os.path.exists(image_path):
        images_list = get_all_images(image_path)
        processor = safe_from_pretrained(VLChatProcessor, model_path, trust_remote_code=args.trust_remote_code)
        tokenizer = processor.tokenizer
        model = safe_from_pretrained(AutoModelForCausalLM, model_path, torch_dtype=torch.float16).eval()
        model = model.to(device)
        image_answer = {}
        for _, img_path in enumerate(tqdm(images_list)):
            conversation = processor.new_chat_template()
            conversation.append_message(conversation.roles[0],
                                        ("<image_placeholder>\nDescribe this image in detail.", img_path))
            conversation.append_message(conversation.roles[1], "")
            prompts = generate_prompt_with_conversation(conversation=conversation)
            pil_images = []
            for message in prompts:
                if "images" in message:
                    for img in message["images"]:
                        pil_images.append(PIL.Image.open(img))
            inputs = processor(conversations=prompts, images=pil_images, force_batchify=True).to(device)
            inputs_embeds = model.prepare_inputs_embeds(**inputs)
            inputs_embeds = inputs_embeds.to(model.device)

            with torch.no_grad():
                outputs = model.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=inputs.attention_mask,
                    pad_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=30,
                    do_sample=False,
                    num_beams=1,
                    use_cache=True,
                )
            response = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
            img_name = os.path.basename(img_path)
            image_answer[img_name] = response.split("ASSISTANT:")[-1]
        torch.cuda.empty_cache()
        if not os.path.exists(OUTPUT_JSON_PATH):
            with open(OUTPUT_JSON_PATH, "w") as f:
                json.dump(image_answer, f)
        else:
            with open(OUTPUT_JSON_PATH, "r") as f:
                old_data = json.load(f)
            old_data.update(image_answer)
            with open(OUTPUT_JSON_PATH, "w") as f:
                json.dump(old_data, f)
        logger.info("run run_coco_gpu.py finish! output file: ./gpu_coco_predict.json")
    else:
        logger.info("model_path or image_path not exist")


if __name__ == "__main__":
    if sys.version_info >= (3, 10):
        logger.info("Python version is above 3.10, patching the collections module.")
        # Monkey patch collections
        import collections
        import collections.abc

        for type_name in collections.abc.__all__:
            setattr(collections, type_name, getattr(collections.abc, type_name))
    AutoConfig.register("vision", VisionConfig)
    AutoConfig.register("aligner", AlignerConfig)
    AutoConfig.register("deepseekvl", DeepseekvlConfig)
    AutoModelForCausalLM.register(DeepseekvlConfig, DeepseekvlCausalLM)
    main()
