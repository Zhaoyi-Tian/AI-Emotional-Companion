# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import argparse
import json
import os
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

from atb_llm.utils import file_utils
from atb_llm.utils.file_utils import safe_open
from atb_llm.utils.log import logger

torch.manual_seed(1234)
OUTPUT_JSON_PATH = "./gpu_coco_predict.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--model_path",
                        required=True,
                        help="Model and tokenizer path.")
    parser.add_argument("--image_path",
                        required=True,
                        help="Image path for inference.")
    return parser.parse_args()


def safe_get_model_from_pretrained(model_path, **kwargs):
    model_path = file_utils.standardize_path(model_path, check_link=False)
    file_utils.check_path_permission(model_path)
    try:
        model = AutoModelForVision2Seq.from_pretrained(model_path)
    except EnvironmentError:
        raise EnvironmentError("Get model from pretrained failed, "
                               "please check model weights files in the model path.") from None
    except Exception:
        raise ValueError("Get model from pretrained failed, "
                         "please check the input parameters model_path and kwargs.") from None
    return model


def safe_processor_from_pretrained(model_path, **kwargs):
    model_path = file_utils.standardize_path(model_path, check_link=False)
    file_utils.check_path_permission(model_path)
    try:
        processor = AutoProcessor.from_pretrained(model_path)
    except EnvironmentError:
        raise EnvironmentError("Get processor from pretrained failed, "
                               "please check processor files in the model path.") from None
    except Exception:
        raise ValueError("Get processor from pretrained failed, "
                         "please check the input parameters model_path and kwargs.") from None
    return processor


def main():
    device = torch.device('cuda', 0)
    args = parse_args()
    model_path = args.model_path
    image_path = args.image_path
    logger.info(f"===== model_path: {model_path}")
    logger.info(f"===== image_path: {image_path}")
    if os.path.exists(model_path) and os.path.exists(image_path):
        processor = None
        model = None
        images_list = os.listdir(image_path)

        processor = safe_processor_from_pretrained(model_path)
        model = safe_get_model_from_pretrained(model_path)

        model = model.to(device)
        image_answer = {}
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this image in detail. ASSISTANT:"},
                ]
            },
        ]

        for _, img_name in enumerate(tqdm(images_list)):
            img_path = os.path.join(image_path, img_name)
            image = Image.open(img_path)
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=prompt, images=[image], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            image.close()

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=500)
            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            image_answer[img_name] = generated_texts
        sorted_dict = dict(sorted(image_answer.items()))
        torch.cuda.empty_cache()
        if not os.path.exists(OUTPUT_JSON_PATH):
            with safe_open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as fw:
                json.dump(sorted_dict, fw)
        else:
            with safe_open(OUTPUT_JSON_PATH, "r") as f:
                old_data = json.load(f)
            old_data.update(sorted_dict)
            sorted_dict = dict(sorted(old_data.items()))
            with safe_open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as fw:
                json.dump(sorted_dict, fw)
        logger.info("run run_coco_gpu.py finish! output file: ./gpu_coco_predict.json")
    else:
        logger.info("model_path or image_path not exist")


if __name__ == "__main__":
    main()
