#!/bin/bash

# 参数提示
show_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "-h -help  Show help message and exit."
    echo "-src FLOAT_WEIGHT_PATH  Specify source float weight path."
    echo "-dst QUANT_WEIGHT_PATH  Specify target quant weight path."
    echo "-type TYPE  Specify model type and quant type. Support qwen_w4a16 and qwen_w8a8 and qwencode_w8a8s."
    echo "-act_method LEVEL  act_method set level"
}

#默认使用npu 如果需要使用cpu 传入-device_type cpu
device_type="npu"
w_bit="8"
a_bit="8"
disable_level="L0"
data_list_index="1"

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
        -device_type)
            if [[ -n "$2" ]]; then
                device_type="$2"
                shift
            else
                echo "Error: -device_type requires a non-empty argument."
                exit 1
            fi
            ;;
        -act_method)
            if [[ -n "$2" ]]; then
                act_method="$2"
                shift
            else
                echo "Error: -act_method requires a non-empty argument."
                exit 1
            fi
            ;;
        -w_bit)
            if [[ -n "$2" ]]; then
                w_bit="$2"
                shift
            else
                echo "Error: -w_bit requires a non-empty argument."
                exit 1
            fi
            ;;
        -a_bit)
            if [[ -n "$2" ]]; then
                a_bit="$2"
                shift
            else
                echo "Error: -a_bit requires a non-empty argument."
                exit 1
            fi
            ;;
        -disable_level)
            if [[ -n "$2" ]]; then
                disable_level="$2"
                shift
            else
                echo "Error: -disable_level requires a non-empty argument."
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
export ASCEND_RT_VISIBLE_DEVICES=2,3
# 进入运行路径
# cd ${ATB_SPEED_HOME_PATH}

param=""

case "$type" in
    w8a8)
        param="--w_bit 8 --a_bit 8 --disable_level L0 --device_type ${device_type} --act_method 1 --trust_remote_code"
        ;;
    w8a16)
        param="--w_bit 8 --a_bit 16 --disable_level L0 --device_type ${device_type} --anti_method m1 --act_method 3 --trust_remote_code"
        ;;
    w8a8c8)
        param="--w_bit 8 --a_bit 8 --disable_level L0 --device_type ${device_type} --act_method 3 --use_kvcache_quant True --trust_remote_code"
        ;;
    *)
        echo "Unknown type: $type"
        show_help
        exit 1
        ;;
esac

# 根据模型类型生成 disable_names 列表
if [ "$type" == "w8a8" ] || [ "$type" == "w8a8c8" ]; then
    config_file="$src/config.json"
    num_layers=48
    for ((layer=0; layer<num_layers; layer++)); do
        disable_names+=("model.layers.$layer.feed_forward.w2")
    done
fi

# 运行 Python 脚本
if [[ "$type" == "w8a8" ]] || [[ "$type" == "w8a8c8" ]]; then
    python -m examples.convert.model_slim.quantifier --model_path $src --save_directory $dst $param --disable_names "${disable_names[@]}"
elif [[ "$type" == "w8a16" ]]; then
    python -m examples.convert.model_slim.quantifier --model_path $src --save_directory $dst $param
fi