#!/bin/bash

# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

# 参数配置以及启动指令的说明见同级目录下的README.md文件
export ASCEND_RT_VISIBLE_DEVICES=0，1
export MASTER_PORT=20037

# 开启确定性计算
export LCCL_DETERMINISTIC=1
export HCCL_DETERMINISTIC=1
export ATB_MATMUL_SHUFFLE_K_ENABLE=0
export ATB_LLM_LCOC_ENABLE=0

# 以下环境变量与性能和内存优化相关，通常情况下无需修改
export INF_NAN_MODE_ENABLE=0
export INT8_FORMAT_NZ_ENABLE=1

if [ -n "$1" ]; then
    model_path="$1"
else
    model_path="/data/datasets/cogvlm2-video-llama3-chat/"
fi
extra_param="--enable_atb_torch"
world_size=$(($(echo "${ASCEND_RT_VISIBLE_DEVICES}" | grep -o , | wc -l) +1))

torchrun --nproc_per_node $world_size --master_port $MASTER_PORT -m examples.models.cogvlm2_llama3_video.run_pa --model_path ${model_path} ${extra_param}
