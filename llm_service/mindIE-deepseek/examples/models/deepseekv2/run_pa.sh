#!/bin/bash

# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
# 参数配置以及启动指令的说明见同级目录下的README.md文件
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=20030

# 以下环境变量与性能和内存优化相关，通常情况下无需修改
export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
export INF_NAN_MODE_ENABLE=0
export ATB_LLM_ENABLE_AUTO_TRANSPOSE=0
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

dp=-1
tp=-1
moe_tp=-1
moe_ep=-1

if [[ "$2" =~ ^[1-9]+$ ]]; then
    dp=$2
fi

if [[ "$3" =~ ^[1-9]+$ ]]; then
    tp=$3
fi

if [[ "$4" =~ ^[1-9]+$ ]]; then
    moe_tp=$4
fi
if [[ "$5" =~ ^[1-9]+$ ]]; then
    moe_ep=$5
fi

# 解析命令行参数
if [[ "$1" == "--trust_remote_code" ]]; then
    TRUST_REMOTE_CODE="--trust_remote_code"
    echo "[TRUST_REMOTE_CODE]: true"
    shift
else
    TRUST_REMOTE_CODE=""
fi

if [[ ! -z "$1" ]];then
    MODEL_PATH=$1
    echo "[MODEL_PATH]: $MODEL_PATH"
    shift
fi
extra_param="--model_path $MODEL_PATH $TRUST_REMOTE_CODE --is_chat_model"

world_size=$(($(echo "${ASCEND_RT_VISIBLE_DEVICES}" | grep -o , | wc -l) +1))

if [ "$TP_WORLD_SIZE" == "1" ]; then
    python -m examples.run_pa --block_size 128 --$extra_param
else
    torchrun --nproc_per_node $world_size --master_port $MASTER_PORT -m examples.run_pa --block_size 128 --dp $dp --tp $tp --moe_tp $moe_tp --moe_ep $moe_ep $extra_param
fi
