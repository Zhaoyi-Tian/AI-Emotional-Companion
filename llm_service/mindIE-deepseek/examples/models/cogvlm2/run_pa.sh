#!/bin/bash

# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
# 参数配置以及启动指令的说明见同级目录下的README.md文件
export ASCEND_RT_VISIBLE_DEVICES=0
export MASTER_PORT=20035

rm -rf /root/atb/log/atb_*

model_path="/data/datasets/cogvlm2-llama3-chinese-chat-19B/"
extra_param="--enable_atb_torch"
world_size=$(($(echo "${ASCEND_RT_VISIBLE_DEVICES}" | grep -o , | wc -l) +1))

torchrun --nproc_per_node $world_size --master_port $MASTER_PORT -m examples.models.cogvlm2.run_pa --model_path ${model_path} ${extra_param}