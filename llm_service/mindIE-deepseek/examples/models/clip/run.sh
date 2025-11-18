#!/bin/bash
export BIND_CPU=1
export RESERVED_MEMORY_GB=0
export ASCEND_RT_VISIBLE_DEVICES=0

model_path="/data/chinese-clip-vit-base-patch16/"
dataset_path=""
input_image="/data/chinese-clip-vit-base-patch16/festival.jpg"
warmup_image_path="./examples/models/clip/pokemon.jpeg"
# If both input_image and dataset_path exist, dataset_path is preferred.
label_file="./examples/models/clip/label.txt"
label_list="Dragon-Boat-Festival,Mid-Autumn-Festival,Spring-Festival"
# If both label_list and label_file exist, label_list is preferred.
# each label in label_list should not have space.
atb_options="ATB_LAUNCH_KERNEL_WITH_TILING=1 ATB_LAYER_INTERNAL_TENSOR_REUSE=1 PYTORCH_NPU_ALLOC_CONF='max_split_size_mb:2048' HCCL_BUFFSIZE=120 ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1"
atb_async_options="ATB_OPERATION_EXECUTE_ASYNC=1 TASK_QUEUE_ENABLE=1"

base_cmd="python \
    -m examples.models.clip.run \
    --model_path ${model_path} \
    --warmup_image_path ${warmup_image_path} \
    --input_image ${input_image} \
    --label_list ${label_list} \
    --input_text '' \
    --max_batch_size "8"
    "
run_cmd="${atb_options} ${atb_async_options} ${base_cmd}"

if [[ -n ${model_path} ]];then
    eval "${run_cmd}"
fi