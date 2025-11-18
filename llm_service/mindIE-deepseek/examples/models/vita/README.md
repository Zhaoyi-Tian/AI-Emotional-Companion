# README

- [VITA](https://github.com/VITA-MLLM/VITA)是第一个开源的多模态大模型（MLLM），擅长同时处理和分析视频、图像、文本和音频模态，同时具有先进的多模态交互体验。
- 此代码仓中实现了一套基于NPU硬件的LLaVa推理模型。配合加速库使用，旨在NPU上获得极致的推理性能。

## 特性矩阵
| 模型及参数量      | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | 800I A2 BF16 | MindIE Service |纯模型支持模态  | 服务化支持模态 |
|-------------|----------------------------|-----------------------------|------|------------------|-----------------|-----|-----|
|  vita-1.5(Qwen2)    | 支持world size 1     | 支持world size 1        | x   |  √                   | √              | 文本、图片、音频           | 文本、图片、音频|

须知：
1. 当前版本服务化仅支持单个请求单张图片输入
2. 当前多模态场景，MindIE Service仅支持MindIE Service、TGI、Triton、vLLM Generate 4种服务化请求格式

## 路径变量解释

| 变量名      | 含义                                                                                                                                                         |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| working_dir | 加速库及模型库下载后放置的目录                                                                                                                               |
| llm_path    | 模型仓所在路径。若使用编译好的包，则路径为 `${working_dir}/MindIE-LLM/`；若使用gitee下载的代码，则路径为 `${working_dir}/MindIE-LLM/examples/atb_models` |
| script_path | 脚本所在路径；vita的工作脚本所在路径为 `${llm_path}/examples/models/vita`                                                                                           |
| weight_path | 模型权重路径
| audio_path  | 音频所在路径                                                                      |
| image_path  | 图片所在路径                                                                      |
| video_path  | 视频所在路径                                                                      |
| max_batch_size  | 最大bacth数                                                                  |
| video_frames | 输入为video时，抽取的帧数                                                        |
| max_input_length  | 多模态模型的最大embedding长度，                                             |
| max_output_length | 生成的最大token数                                                          |

-注意：
max_input_length长度设置可参考模型权重路径下config.json里的max_position_embeddings参数值
## 权重

**权重下载**

- [VITA](https://huggingface.co/VITA-MLLM/VITA-1.5/tree/main)
- [InternViT-300M-448px](https://huggingface.co/OpenGVLab/InternViT-300M-448px/tree/main)
-权重准备：
  在weight_path执行：
  ```shell
  cd VITA
  mv ../InternViT-300M-448px ./VITA_ckpt
  ```

**基础环境变量**

1、安装 CANN 8.0 的环境，并 `source /path/to/cann/set_env.sh`；

2、使用 Python 3.9 或更高；

3、使用 torch 2.0 或更高版本，并安装对应的 torch_npu；

4、安装依赖：

- Python其他第三方库依赖，参考[requirements_vita.txt](../../../requirements/models/requirements_vita.txt)
- 参考[此README文件](../../../README.md)

## 推理

### 对话测试

**运行Paged Attention FP16**

- 运行启动脚本
  - 在\${llm_path}目录下执行以下指令
    ```shell
    bash ${script_path}/run_pa.sh --run ${weight_path} --image_path ${image_path} --audio_path ${audio_path}
    bash ${script_path}/run_pa.sh --run ${weight_path} --video_path ${video_path} --question "你是谁？"
    ```
  - 注意：
    音频和文本不能同时设置
- 环境变量说明
  - `export ASCEND_RT_VISIBLE_DEVICES=0`
    - 指定当前机器上可用的逻辑NPU核心，多个核心间使用逗号相连
    - 核心ID查阅方式见[此README文件](../../README.md)的【启动脚本相关环境变量】章节
    - 对于300I DUO卡而言，若要使用单卡双芯，请指定至少两个可见核心；若要使用双卡四芯，请指定至少四个可见核心
    - 各模型支持的核心数参考“特性矩阵”
  - `export MASTER_PORT=20030`
    - 设置卡间通信端口
    - 默认使用20030端口
    - 目的是为了避免同一台机器同时运行多个多卡模型时出现通信冲突
    - 设置时端口建议范围为：20000-20050
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


