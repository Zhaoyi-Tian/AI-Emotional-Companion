# README

- [DeepSeek-VL](https://github.com/deepseek-ai/DeepSeek-VL)，是一种专为真实世界视觉和语言理解应用而设计的开源视觉语言 （VL） 模型。DeepSeek-VL具备通用的多模态理解能力，能够在复杂场景下处理逻辑图、网页、公式识别、科学文献、自然图像和具身智能。
- 此代码仓中实现了一套基于NPU硬件的DeepSeek-VL推理模型。配合加速库使用，旨在NPU上获得极致的推理性能。
- 支持DeepSeek-VL 7b 基于llama文本模型的多模态推理

# 使用说明

## 特性矩阵

- 此矩阵罗列了各deepseekvl模型支持的特性

| 模型及参数量  | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | 800I A2 BF16 | MindIE Service | 纯模型支持模态 | 服务化支持模态 |
| ------------- | -------------------------- | --------------------------- | ---- | ---- | --------------- | --------------- | -------- |
| deepseek-vl-7b-base  | 支持world size 1,2,4,8     | 支持world size 1,2,4,8            | √    | ×    | √               | 文本、图片               | 文本、图片        |
| deepseek-vl-7b-chat | 支持world size 1,2,4,8       | 支持world size 1,2,4,8          | √    | ×    | √               | 文本、图片               | 文本、图片        |

须知：
1. 当前版本服务化仅支持单个请求单张图片输入
2. 当前多模态场景, MindIE Service仅支持MindIE Service、TGI、Triton、vLLM Generate 4种服务化请求格式

## 路径变量解释

| 变量名      | 含义                                                         |
| ----------- | ------------------------------------------------------------ |
| working_dir | 加速库及模型库下载后放置的目录                               |
| llm_path    | 模型仓所在路径。若使用编译好的包，则路径为 `${working_dir}/MindIE-LLM/`；若使用gitee下载的代码，则路径为 `${working_dir}/MindIE-LLM/examples/atb_models` |
| script_path | 脚本所在路径；deepseekvl的工作脚本所在路径为 `${llm_path}/examples/models/deepseekvl |
| weight_path | 模型权重路径                                                 |
| image_path  | 图片所在路径                                                 |
## 权重

**权重下载**

- [deepseek-vl-7b-base](https://huggingface.co/deepseek-ai/deepseek-vl-7b-base)
- [deepseek-vl-7b-chat](https://huggingface.co/deepseek-ai/deepseek-vl-7b-chat)

下载依赖后，请将`config.json`中的`model_type`从`multi_modality`改为`deepseekvl`。

修改前：

```json
  "model_type": "multi_modality",
  "torch_dtype": "float16",
  "transformers_version": "4.38.2",
```

修改后：

```json
  "model_type": "deepseekvl",
  "torch_dtype": "float16",
  "transformers_version": "4.38.2",
```

此举仅为了适应代码规范而将`multi_modality`中的`_`去掉。

**基础环境变量**

- 参考[此README文件](../../../README.md)

## 依赖

python依赖如下：

```txt
torch==2.1.0
transformers==4.38.2
timm==0.9.16
accelerate==0.31.0
sentencepiece==0.2.0
attrdict==2.0.1
einops==0.8.0
open-clip-torch
```

## 推理

### 对话测试

**运行Paged Attention FP16**

- 运行启动脚本
  - 在`${llm_path}`目录下执行以下指令
    ```shell
    bash ${script_path}/run_pa.sh --run ${weight_path} ${image_path}
    ```
- 环境变量说明
  - `export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
    
    - 指定当前机器上可用的逻辑NPU核心，多个核心间使用逗号相连
    - 核心ID查阅方式见[此README文件](../../README.md)的【启动脚本相关环境变量】章节
    - 对于300I DUO卡而言，若要使用单卡双芯，请指定至少两个可见核心；若要使用双卡四芯，请指定至少四个可见核心
    - 各模型支持的核心数参考“特性矩阵”
  - `export MASTER_PORT=20030`
    - 设置卡间通信端口
    - 默认使用20030端口
    - 目的是为了避免同一台机器同时运行多个多卡模型时出现通信冲突
    - 设置时端口建议范围为：20000-20050
    - `export USE_REFACTOR=true`
    - 是否使用新版模型组图
    - 默认使用
  - 以下环境变量与性能和内存优化相关，通常情况下无需修改
    ```shell
    export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
    export INF_NAN_MODE_ENABLE=0
    export ATB_OPERATION_EXECUTE_ASYNC=1
    export TASK_QUEUE_ENABLE=1
    export ATB_CONVERT_NCHW_TO_ND=1
    export LCCL_ENABLE_FALLBACK=1
    export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
    export ATB_CONTEXT_WORKSPACE_SIZE=0
    ```

## 精度测试

### open_clip方案

我们采用的精度测试方案是这样的：使用同样的一组图片，分别在 GPU 和 NPU 上执行推理，得到两组图片描述。 再使用 open_clip 模型作为裁判，对两组结果分别进行评分，以判断优劣。

#### 实施

1. 下载[open_clip 的权重 open_clip_pytorch_model.bin](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/tree/main)，
   下载[测试图片（CoCotest 数据集）](https://cocodataset.org/#download)并随机抽取其中100张图片放入一个文件夹，建议都放置到 `./precision`下。
2. 收集推理结果
3. GPU 上：收集脚本参考 `./precision/run_coco_gpu.py`，
   将其放到 `${work_space}`目录下执行，注意脚本传参（主要是 `--model_path`和 `--image_path`以及`--trust_remote_code`）。
4. NPU 上：类似基本推理，只需增加一个参数（图片文件夹的路径）即可
   ```bash
   bash ${script_path}/run_pa.sh --precision ${weight_path} ${iamge_path}
   ```

   收集的结果应是类似 `./precision/GPU_NPU_result_example.json` 的形式。
 3. 对结果进行评分：执行脚本 `./precision/clip_score_deepseekvl.py`，参考命令：

```bash
   python clip_score_deepseekvl.py --model_weights_path open_clip_pytorch_model.bin（这个替换成实际的open_clip的bin的位置） --image_info GPU_NPU_result_example.json（这个替换成你的实际路径） --dataset_path 图片文件夹的路径
```

   得分高者精度更优。

### TextVQA方案
使用modeltest进行纯模型在TextVQA数据集上的精度测试
- 数据准备
    - 数据集下载 [textvqa](https://huggingface.co/datasets/maoxx241/textvqa_subset)
    - 保证textvqa_val.jsonl和textvqa_val_annotations.json在同一目录下
    - 将textvqa_val.jsonl文件中所有"image"属性的值改为相应图片的绝对路径
  ```json
  ...
  {
    "image": "/data/textvqa/train_images/003a8ae2ef43b901.jpg",
    "question": "what is the brand of this camera?",
    "question_id": 34602, 
    "answer": "dakota"
  }
  ...
  ```
- 设置环境变量
  ```shell
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  source /usr/local/Ascend/nnal/atb/set_env.sh 
  source ${llm_path}/set_env.sh 
  ```
- 进入以下目录 MindIE-LLM/examples/atb_models/tests/modeltest
  ```shell
  cd MindIE-LLM/examples/atb_models/tests/modeltest
  ```
- 安装modeltest及其三方依赖
 
  ```shell
  pip install --upgrade pip
  pip install -e .
  pip install tabulate termcolor 
  ```
   - 若下载有SSL相关报错，可在命令后加上'-i https://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com'参数使用阿里源进行下载
- 将modeltest/config/model/deepseekvl.yaml中的model_path的值修改为模型权重的绝对路径，mm_model.warm_up_image_path改为textvqa数据集中任一图片的绝对路径
  ```yaml
  model_path: /data_mm/weights/deepseek-vl-7b-chat
  mm_model:
    warm_up_image_path: ['/data_mm/datasets/textvqa_val/train_images/003a8ae2ef43b901.jpg']
  ```
- 将modeltest/config/task/textvqa.yaml中的model_path修改为textvqa_val.jsonl文件的绝对路径，以及将requested_max_input_length和requested_max_output_length的值分别改为20000和256
  ```yaml
  local_dataset_path: /data_mm/datasets/textvqa_val/textvqa_val.jsonl
  requested_max_input_length: 20000
  requested_max_output_length: 256
  ```
- 将textvqa_val.jsonl文件中所有"image"属性的值改为相应图片的绝对路径
  ```json
  ...
  {
    "image": "/data/textvqa/train_images/003a8ae2ef43b901.jpg",
    "question": "what is the brand of this camera?",
    "question_id": 34602, 
    "answer": "dakota"
  }
  ...
  ```
- 设置可见卡数，修改mm_run.sh文件中的ASCEND_RT_VISIBLE_DEVICES。依需求设置单卡或多卡可见。
  ```shell
  export ASCEND_RT_VISIBLE_DEVICES=0,1
  ```
- 运行测试命令
  ```shell
  bash scripts/mm_run.sh textvqa deepseekvl
  ```
- 测试结果保存于以下路径。其下的results/..(一系列文件夹嵌套)/\*\_result.csv中存放着modeltest的测试结果。debug/..(一系列文件夹嵌套)/output\_\*.txt中存储着每一条数据的运行结果，第一项为output文本，第二项为输入infer函数的第一个参数的值，即模型输入。第三项为e2e_time。
  ```shell
  output/$DATE/modeltest/$MODEL_NAME/precision_result/
  ```

## 性能测试

_性能测试时需要在 `${image_path}` 下仅存放一张图片_

测试模型侧性能数据，需要开启环境变量
  ```shell
  export ATB_LLM_BENCHMARK_ENABLE=1
  export ATB_LLM_BENCHMARK_FILEPATH=${script_path}/benchmark.csv
  ```
**在${llm_path}目录使用以下命令运行 `run_pa.sh`**，会自动输出batchsize为1-10时，输出token长度为 256时的吞吐。

```shell
bash examples/models/deepseekvl/run_pa.sh --performance ${weight_path} ${image_path}
```

可以在 `${script_path}` 路径下找到测试结果。

## FAQ

- 更多环境变量见[此README文件](../../README.md)
- 运行时，需要通过指令`pip list｜grep protobuf`确认protobuf版本，如果版本高于3.20.x，请运行指令`pip install protobuf==3.20.0`进行更新
