# Copyright (c) Huawei Technologies Co., Ltd. 2012-2020. All rights reserved.
from safetensors.torch import save_file
from safetensors import safe_open

ORIGIN_MODEL_PATH = ''
TARGET_MODEL_PATH = ''

tensors = {}
with safe_open(ORIGIN_MODEL_PATH, framework="pt", device='cpu') as f:
    for k in f.keys():
        tensors[k] = f.get_tensor(k)
    tensors['lm_head'] = f.get_tensor('model.embed_tokens.weight').clone()
    save_file(tensors, TARGET_MODEL_PATH, metadata={'format': 'pt'})