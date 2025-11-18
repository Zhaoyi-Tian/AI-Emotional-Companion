# README

- CLIP (Contrastive Language-Image Pre-Training，以下简称 CLIP) 模型是 OpenAI 在 2021
  年初发布的用于匹配图像和文本的预训练神经网络模型，是近年来在多模态研究领域的经典之作，可用于自然语言图像检索和零样本图像分类。
- Chinese-CLIP为CLIP模型的中文版本，使用大规模中文数据进行训练（~2亿图文对），旨在帮助用户快速实现中文领域的图文特征相似度计算、跨模态检索、零样本图片分类等任务。

## 特性矩阵

- 此矩阵罗列了各CLIP模型支持的特性

| 模型及参数量  | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | 800I A2 BF16 | MindIE Service | 纯模型支持模态 | 服务化支持模态 |
| ------------- | -------------------------- | --------------------------- | ---- | ---- | --------------- | --------------- | -------- |
| CLIP  | 支持world size 1     | 当前模型不支持            | √    | ×    | ×               | 文本、图片               | 当前模型不支持服务化        |
| Chinese-CLIP | 支持world size 1       | 当前模型不支持          | √    | ×    | ×               | 文本、图片               | 当前模型不支持服务化        | 


## 路径变量解释

| 参数名         | 含义                                                                                                                    |
|-------------|-----------------------------------------------------------------------------------------------------------------------|
| working_dir | 加速库及模型库下载后放置的目录                                                                                                       |
| llm_path    | 模型仓所在路径。若使用编译好的包，则路径为 `${working_dir}/MindIE-LLM/`；若使用gitee下载的代码，则路径为 `${working_dir}/MindIE-LLM/examples/atb_models` |
| runner_path | 运行脚本所在路径。`${working_dir}/MindIE-LLM/examples/atb_models/examples/models/clip/`                                        |
| model_path  | 模型所在路径。`${working_dir}/MindIE-LLM/examples/atb_models/atb_llm/models/clip/`                                           |

## 权重

**权重下载**

参考配置

- [CLIP](https://huggingface.co/openai/clip-vit-large-patch14/tree/main)
- [Chinese-CLIP](https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16/tree/main)

**基础环境变量**

- Toolkit，MindIE/ATB，ATB-SPEED等，参考[此README文件](../../../README.md)
- Python其他第三方库依赖，参考[requirements_clip.txt](../../../requirements/models/requirements_clip.txt)

## 推理

**运行CLIP FP16**

- 脚本参数说明

  | 变量名      | 含义 |
    | ---------- | ------- |
  | model_path | 模型权重路径 |
  | input_text | 标签模板，下划线 `_` 用来表示标签的位置，一个句子中只能有一个下划线 `_` 。 |
  | label_file | 标签数量较多时使用file来保留标签，标签文件中每行只放一个标签。 |
  | label_list | 标签较少时直接用参数传入，多个标签之间以英文逗号 `,` 隔开。 |
  | input_image | 单张图片路径  |
  | warmup_image_path | warmup图片路径，用于模型初始化  |
  | dataset_path | 数据集路径，路径下放置大量图片  |
  | results_save_path | 结果保存路径  |
  | ... | 其他参数请参考脚本parse_args部分  |

  注意：CLIP不支持中文label_file以及label_list


- 执行启动脚本

  下载warm_up以及精度测试案例图片（可根据场景切换）
  ```shell
  cd ${runner_path}
  wget https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/pokemon.jpeg
  ```

  修改`${runner_path}/run.sh`中变量`dataset_path`为本地数据集路径或者修改为`input_image`实际图片路径，在`${llm_path}`目录下, 执行以下指令
  ```shell
  bash ${runner_path}/run.sh
  ```

  输入数据集时，运行推理脚本成功后在执行目录下生成 res.json 文件，保存数据集图片名称和对应的分类标签

- 环境变量说明
    - `export ASCEND_RT_VISIBLE_DEVICES=0`
        - 指定当前机器上可用的逻辑NPU核心，当前实现的CLIP在线模型仅仅支持单卡推理
        - 核心ID查阅方式见[此README文件](../../README.md)的【启动脚本相关环境变量】章节
        - 各模型支持的核心数参考“特性矩阵”
    - 以下环境变量与性能和内存优化相关，通常情况下无需修改，详细信息可参考ATB官方文档
      ```shell
      export ATB_LAUNCH_KERNEL_WITH_TILING=1
      export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
      export PYTORCH_NPU_ALLOC_CONF='max_split_size_mb:2048'
      export HCCL_BUFFSIZE=120
      export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
      export ATB_OPERATION_EXECUTE_ASYNC=1
      export TASK_QUEUE_ENABLE=1
      ```

## 精度测试

### 测试方法

目前CLIP模型支持NPU和CPU在线推理，模型输出当前测试图片的标签分布，计算NPU和CPU输出分布的余弦相似度

### 测试步骤

- 运行脚本
    ```shell
    export PYTHONPATH=${llm_path}:$PYTHONPATH

    python ${runner_path}/precision/precision_test.py
      --model_path {CLIP权重路径} \
      --input_image {测试图片路径} \
      --label_list {分类标签}
    ```
  CPU运行时候batch_size需要设置为2，否则torch.matmul会报错，具体原因为当前torch版本使用2.1.0，在ARM架构的CPU上运行此算子会有错误，详细错误信息见[此链接](https://gitee.com/ascend/pytorch/issues/I8IG3N?from=project-issue)

- 案例结果
  执行结束后，期望输出 cosine_similarity > 0.99
