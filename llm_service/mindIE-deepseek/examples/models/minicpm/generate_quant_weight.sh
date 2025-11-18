#!/bin/bash

# 参数提示
show_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "-h -help  Show help message and exit."
    echo "-src FLOAT_WEIGHT_PATH  Specify source float weight path."
    echo "-dst QUANT_WEIGHT_PATH  Specify target quant weight path."
    echo "-type TYPE  Specify model tpye and quant type. Support minicpm_2b_w8a8."
    echo "-use_kvcache_quant  Whether to use kvcache int8 quant. Default value is false."
}

use_kvcache_quant=False

# 解析命令行参数
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            exit 0
            ;;
        -src)
            if [[ -n "$2" ]]; then
                src="$2"
                shift
            else
                echo "Error: -src requires a non-empty argument."
                exit 1
            fi
            ;;
        -dst)
            if [[ -n "$2" ]]; then
                dst="$2"
                shift
            else
                echo "Error: -dst requires a non-empty argument."
                exit 1
            fi
            ;;
        -type)
            if [[ -n "$2" ]]; then
                type="$2"
                shift
            else
                echo "Error: -type requires a non-empty argument."
                exit 1
            fi
            ;;
        -use_kvcache_quant)
            if [[ -n "$2" ]]; then
                use_kvcache_quant="$2"
                shift
            else
                echo "Error: -type requires a non-empty argument."
                exit 1
            fi
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
    shift
done

# 参数校验
if [[ -z "$src" ]]; then
    echo "Error: Missing required option: -src"
    show_help
    exit 1
fi

if [[ -z "$dst" ]]; then
    echo "Error: Missing required option: -dst"
    show_help
    exit 1
fi

if [[ -z "$type" ]]; then
    echo "Error: Missing required option: -type"
    show_help
    exit 1
fi

# 设置环境变量
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
export ASCEND_RT_VISIBLE_DEVICES=0

# 进入运行路径
cd ${ATB_SPEED_HOME_PATH}

param=""

get_down_proj_disable_name() {
    local num_layer=$1
    local disable_names=""
    for ((i=0; i<$num_layer; i++)); do
        disable_names="$disable_names model.layers.$i.mlp.down_proj"
    done
    echo "$disable_names"
}

case "$type" in
    minicpm_2b_w8a8)
        disable_names=$(get_down_proj_disable_name 40)
        param="--calib_file $ATB_SPEED_HOME_PATH/examples/convert/model_slim/boolq.jsonl --disable_names $disable_names
        --device_type cpu --disable_level L0 --act_method 1 --disable_last_linear False
        --tokenizer_args {\"padding_side\":\"left\",\"pad_token\":\"<unk>\"} --use_kvcache_quant $use_kvcache_quant"
        ;;
    minicpm_1b_w8a8)
            disable_names=$(get_down_proj_disable_name 52)
            param="--calib_file $ATB_SPEED_HOME_PATH/examples/convert/model_slim/boolq.jsonl --disable_names $disable_names
            --device_type cpu --disable_level L0 --act_method 1 --disable_last_linear False
            --tokenizer_args {\"padding_side\":\"left\",\"pad_token\":\"<unk>\"} --use_kvcache_quant $use_kvcache_quant"
            ;;
    *)
        echo "Unknown type: $type"
        show_help
        exit 1
        ;;
esac

python -m examples.convert.model_slim.quantifier --model_path $src --save_directory $dst $param
