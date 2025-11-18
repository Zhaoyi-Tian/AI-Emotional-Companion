#!/bin/bash
# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

# 可修改配置
LORA_ADAPTER_ID="wemm"
TP_WORLD_SIZE=1
PERFORMANCE_MAX_BATCH_SIZE=1
PERFORMANCE_MAX_OUTPUT_LENGTH=256
PRECISION_MAX_BATCH_SIZE=1

IMAGE_PATH=""
MODEL_PATH=""
RUN_OPTION=""
TRUST_REMOTE_CODE=""
MULTI_MODAL_PA_PATH="examples.models.wemm.wemm"

TEXTS=("请详细描述这张图片。")

function performance()
{
    world_size=$(($(echo "${ASCEND_RT_VISIBLE_DEVICES}" | grep -o , | wc -l) +1))
    for ((bsz=1;bsz<=$PERFORMANCE_MAX_BATCH_SIZE;bsz++)); do
        extra_param=""
        extra_param="${extra_param} --model_path $MODEL_PATH
                                    --image_or_video_path $IMAGE_PATH
                                    --max_batch_size $bsz
                                    --max_output_length $PERFORMANCE_MAX_OUTPUT_LENGTH
                                    --ignore_eos
                                    --skip_special_tokens
                                    $TRUST_REMOTE_CODE"
                                    
        if [ "$TP_WORLD_SIZE" == "1" ]; then
            python -m $MULTI_MODAL_PA_PATH $extra_param --performance --lora_adapter_id $LORA_ADAPTER_ID --input_texts_for_image "${TEXTS[@]}"
        else
            torchrun --nproc_per_node $world_size --master_port $MASTER_PORT\
                -m $MULTI_MODAL_PA_PATH $extra_param \
                --performance \
                --lora_adapter_id $LORA_ADAPTER_ID \
                --input_texts_for_image "${TEXTS[@]}"
        fi
            done
}

function precision()
{
    world_size=$(($(echo "${ASCEND_RT_VISIBLE_DEVICES}" | grep -o , | wc -l) +1))
    extra_param=""
    extra_param="${extra_param} --model_path $MODEL_PATH
                                --image_or_video_path $IMAGE_PATH
                                --max_batch_size $PRECISION_MAX_BATCH_SIZE
                                --skip_special_tokens
                                $TRUST_REMOTE_CODE"
                                
    if [ "$TP_WORLD_SIZE" == "1" ]; then
        python -m $MULTI_MODAL_PA_PATH $extra_param --prediction_result --lora_adapter_id $LORA_ADAPTER_ID --input_texts_for_image "${TEXTS[@]}"
    else
        torchrun --nproc_per_node $world_size --master_port $MASTER_PORT\
                 -m $MULTI_MODAL_PA_PATH $extra_param \
                 --prediction_result \
                 --lora_adapter_id $LORA_ADAPTER_ID \
                 --input_texts_for_image "${TEXTS[@]}"
    fi
}

function run()
{
    case "${RUN_OPTION}" in
        "--performance")
            performance_env
            performance
            ;;
        "--precision")
            precision_env
            precision
            ;;
        *)
            echo "ERROR: invalid RUN_OPTION, only support --performance and --precision"
            ;;
    esac
}

function precision_env()
{
    # 精度测试开启确定性计算
    export LCCL_DETERMINISTIC=1
    export HCCL_DETERMINISTIC=true
    export ATB_MATMUL_SHUFFLE_K_ENABLE=0
    export ATB_LLM_LCOC_ENABLE=0
}

function performance_env()
{
    # 性能测试需关闭确定性计算，否则影响性能
    export LCCL_DETERMINISTIC=0
    export HCCL_DETERMINISTIC=false
    export ATB_MATMUL_SHUFFLE_K_ENABLE=1
}

function set_env()
{
    # 参数配置以及启动指令的说明见同级目录下的README.md文件
    export ASCEND_RT_VISIBLE_DEVICES=0
    export MASTER_PORT=20036

    # 以下环境变量与性能和内存优化相关，通常情况下无需修改
    export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
    export INF_NAN_MODE_ENABLE=1
    export ATB_OPERATION_EXECUTE_ASYNC=1
    export TASK_QUEUE_ENABLE=1
    export ATB_CONVERT_NCHW_TO_ND=1
    export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
    export ATB_CONTEXT_WORKSPACE_SIZE=0
    export INT8_FORMAT_NZ_ENABLE=1
}

function fn_main()
{
    if [ $# -eq 0 ]; then
        echo "Error: require parameter. Please refer to README."
        exit 1
    fi

    while [[ $# -gt 0 ]]; do
        case $1 in
            --precision)
                RUN_OPTION=$1
                echo "RUN_OPTION: $RUN_OPTION"
                shift 1
                ;;
            --trust_remote_code)
                TRUST_REMOTE_CODE="--trust_remote_code"
                echo "TRUST_REMOTE_CODE: $TRUST_REMOTE_CODE"
                shift 1
                ;;
            --model_path)
                MODEL_PATH=$2
                echo "[MODEL_PATH]: $MODEL_PATH"
                shift 2
                ;;
            --image_or_video_path)
                IMAGE_PATH=$2
                echo "[IMAGE_PATH]: $IMAGE_PATH"
                shift 2
                ;;
            *)
                echo "Unknown parameter: $1"
                exit 1
                ;;
        esac
    done

    set_env
    run
}

fn_main "$@"
