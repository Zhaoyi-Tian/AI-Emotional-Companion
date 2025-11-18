#!/bin/bash

# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
# 参数配置以及启动指令的说明见同级目录下的README.md文件
export ASCEND_RT_VISIBLE_DEVICES=6
export MASTER_PORT=20037
# export ATB_PROFILING_ENABLE=1

# 以下环境变量与性能和内存优化相关，通常情况下无需修改
export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
export INF_NAN_MODE_ENABLE=1
export ATB_OPERATION_EXECUTE_ASYNC=1
export TASK_QUEUE_ENABLE=1
export ATB_CONVERT_NCHW_TO_ND=1
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
export ATB_CONTEXT_WORKSPACE_SIZE=0
export INT8_FORMAT_NZ_ENABLE=1

IMAGE_OR_VIDEO_PATH=/data/acltransformer_testdata/images/llava
MODEL_TYPE=qwen2p5_instruct
CONV_MODE=qwen2p5_instruct
QUESTION=""
AUDIO_PATH=""
IMAGE_PATH=""
VIDEO_PATH=""
MAX_BATCH_SIZE=1
MAX_INPUT_LENGTH=8192
MAX_OUTPUT_LENGTH=80
world_size=$(($(echo "${ASCEND_RT_VISIBLE_DEVICES}" | grep -o , | wc -l) +1))

if [ $# -eq 0 ]; then
        echo "Error: require parameter. Please refer to README."
        exit 1
fi

if [[ ! -z "$1" ]];then
    RUN_OPTION=$1
    echo "[RUN_OPTION]: $RUN_OPTION"
    shift
fi

if [[ ! -z "$1" ]];then
    MODEL_PATH=$1
    echo "[MODEL_PATH]: $MODEL_PATH"
    shift
fi

while [[ "$#" -gt 0 ]]; do
    case $1 in 
        --question) QUESTION="$2"; shift ;;
        --audio_path) AUDIO_PATH="$2"; shift ;;
        --video_path) VIDEO_PATH="$2"; shift ;;
        --image_path) IMAGE_PATH="$2"; shift ;;
        --max_batch_size) MAX_BATCH_SIZE="$2"; shift ;;
        --max_input_length) MAX_INPUT_LENGTH="$2"; shift ;;
        --max_output_length) MAX_OUTPUT_LENGTH="$2"; shift ;;
        -h|--help) echo "Usage: $0 [--param1 <value>] [--param2 <value>] [--param3 <value>] [--param4 <value>]"; exit 0 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "QUESTION: $QUESTION"
echo "AUDIO_PATH: $AUDIO_PATH"
echo "VIDEO_PATH: $VIDEO_PATH"
echo "IMAGE_PATH: $IMAGE_PATH"
echo "MAX_BATCH_SIZE: $MAX_BATCH_SIZE"
echo "MAX_INPUT_LENGTH: $MAX_INPUT_LENGTH"
echo "MAX_OUTPUT_LENGTH: $MAX_OUTPUT_LENGTH"

extra_param=""
extra_param="${extra_param} --model_path $MODEL_PATH
                            --model_type $MODEL_TYPE
                            --conv_mode $CONV_MODE
                            --max_input_length $MAX_INPUT_LENGTH
                            --max_output_length $MAX_OUTPUT_LENGTH"

if [ -n "$IMAGE_PATH" ]; then 
    extra_param="$extra_param
                            --image_path $IMAGE_PATH" 
fi 
if [ -n "$AUDIO_PATH" ]; then 
    extra_param="$extra_param
                            --audio_path $AUDIO_PATH" 
fi 
if [ -n "$VIDEO_PATH" ]; then 
    extra_param="$extra_param
                            --video_path $VIDEO_PATH" 
fi 

case "${RUN_OPTION}" in
    "--run")
    extra_param="$extra_param
                                --max_batch_size $MAX_BATCH_SIZE"
    echo "extra_param: $extra_param"                    
    if [ "$TP_WORLD_SIZE" == "1" ]; then
        python -m examples.models.vita.vita $extra_param --question "what you see"
    else
        torchrun --nproc_per_node $world_size --master_port $MASTER_PORT -m examples.models.vita.vita $extra_param --question "$QUESTION"
    fi
    ;;
    "--performance")
        for bsz in {1..10}; do
            extra_param="$extra_param
                                --max_batch_size $bsz
                                --ignore_eos"  
            if [ "$TP_WORLD_SIZE" == "1" ]; then
                python -m examples.models.vita.vita $extra_param --performance --question "$QUESTION" 
            else
                torchrun --nproc_per_node $world_size --master_port $MASTER_PORT -m examples.models.vita.vita $extra_param --performance --question "$QUESTION" 
            fi
                done
            ;;
    "--precision")
    extra_param="$extra_param
                                --max_batch_size $MAX_BATCH_SIZE"
    if [ "$TP_WORLD_SIZE" == "1" ]; then
        python -m examples.models.vita.vita $extra_param --prediction_result --question "$QUESTION" 
    else
        torchrun --nproc_per_node $world_size --master_port $MASTER_PORT -m examples.models.vita.vita $extra_param --prediction_result --question "$QUESTION" 
    fi
    ;;
esac
