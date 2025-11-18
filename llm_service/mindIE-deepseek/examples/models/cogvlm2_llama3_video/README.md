# README

- [cogvlm2-video-llama3-chat](https://huggingface.co/THUDM/cogvlm2-video-llama3-chat)
是由智谱AI（Zhipu.AI）推出的新一代多模态大型语言模型。它是基于 Meta-Llama-3-8B-Instruct 构建的，当前只支持英文。CogVLM2-Video 模型通过抽取关键帧的方式，实现对连续画面的解读，该模型可以支持最高1分钟的视频，支持文本长度为2K和224 * 224 的视频(取前24帧)。
- 此代码仓中实现了一套基于NPU硬件的cogvlm2-video-llama3-chatB推理模型。配合加速库使用，旨在NPU上获得极致的推理性能。

# 使用说明

## 路径变量解释

| 变量名               | 含义                                                                                                                                             |
|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| working_dir       | 加速库及模型库下载后放置的目录                                                                                                                                |
| llm_path          | 模型仓所在路径。若使用编译好的包，则路径为 `${working_dir}/MindIE-LLM/`；若使用gitee下载的代码，则路径为 `${working_dir}/MindIE-LLM/examples/atb_models`                          |
| script_path       | 脚本所在路径；cogvlm2_llama3_chinese_chat_19B的工作脚本所在路径为 `${llm_path}/examples/models/cogvlm2_llama3_video`                                                         |
| weight_path       | 模型权重路径                                                                                                                                         |
| image_path        | 图片所在路径                                                                                                                                         |
| max_input_length  | 多模态模型的最大embedding长度。 |
| max_output_length | 生成的最大token数                                                                                                                                    |

## 权重

**权重下载**

- [cogvlm2-video-llama3-chat](https://huggingface.co/THUDM/cogvlm2-video-llama3-chat)

**基础环境变量**

-1.Python其他第三方库依赖，参考[requirements_cogvlm2_video_llama3_chat.txt](../../../requirements/models/requirements_cogvlm2_video_llama3_chat.txt)
-2.参考[此README文件](../../../README.md)
-注意：保证先后顺序，否则cogvlm2-video-llama3-chat的其余三方依赖会重新安装torch，导致出现别的错误

**原始文件拷贝与修改**

- 修改权重文件中的config.json，增加字段：

  ```shell
  "model_type": "cogvlm2_llama3_video",
  "torch_dtype": "bfloat16",
  ```

## 推理

### 模型侧对话测试

- 运行启动脚本
  - 在\${llm_path}目录下执行以下指令
    ```shell
    bash ${script_path}/run_pa.sh
    ```
    可以配置多个环境变量，说明如下
    ```shell
      # 开启确定性计算
      export LCCL_DETERMINISTIC=1
      export HCCL_DETERMINISTIC=true
      export ATB_MATMUL_SHUFFLE_K_ENABLE=0
      export ATB_LLM_LCOC_ENABLE=0
      
      # 日志相关
      export ATB_LOG_TO_STDOUT=1 ATB_LOG_LEVEL=DEBUG
      export ASDOPS_LOG_LEVEL=DEBUG ASDOPS_LOG_TO_STDOUT=1
      export ASCEND_SLOG_PRINT_TO_STDOUT=1 ASCEND_GLOBAL_LOG_LEVEL=0
      
      # 以下环境变量与性能和内存优化相关，通常情况下无需修改
      export INF_NAN_MODE_ENABLE=0
      export INT8_FORMAT_NZ_ENABLE=1
    
      # 配置使用的核心
      export ASCEND_RT_VISIBLE_DEVICES=0
    ```
    - 核心ID查阅方式见[此README文件](../../README.md)的【启动脚本相关环境变量】章节
    - 更多环境变量见[此README文件](../../README.md)

### 服务化对话测试

- 运行启动命令样例
  - vLLM 
    ```shell
    curl 127.0.0.1:1025/generate -d '{
      "prompt": [
        {
          "type": "text",
          "text": "Question: "what is the general content of this video? Answer:"
        },
        {
          "type": "video_url",
          "video_url": "/path/of/image/video.mp4"
        }
      ],
      "max_tokens": 256,
      "repetition_penalty": 1,
      "presence_penalty": 1,
      "frequency_penalty": 1,
      "temperature": 0.6,
      "top_k": 1,
      "top_p": 0.9,
      "stream": false,
      "model": "cogvlm2"
    }'
    ```
  - TGI
    ```shell
    curl 127.0.0.1:1025/generate -d '{
      "inputs": [
        {
          "type": "text",
          "text": "what is the general content of this video?"
        },
        {
          "type": "video_url",
          "video_url": "/path/of/image/video.mp4"
        }
      ],
      "parameters": {
        "decoder_input_details": false,
        "details": false,
        "do_sample": true,
        "max_new_tokens": 256,
        "repetition_penalty": 1.0,
        "return_full_text": false,
        "seed": null,
        "temperature": 0.6,
        "top_k": 10,
        "top_p": 0.9,
        "truncate": null,
        "typical_p": 0.5,
        "watermark": false,
        "adapter_id": "456"
      }
    }'
    ```
  - Trition Stream
    ```shell
    curl 127.0.0.1:1025/v2/models/cogvlm2/generate_stream -d '{
      "id": "a123",
      "text_input": [
        {
          "type": "text",
          "text": "what is the general content of this video?"
        },
        {
          "type": "video_url",
          "video_url": "/path/of/image/video.mp4"
        }
      ],
      "parameters": {
        "details": true,
        "do_sample": true,
        "max_new_tokens": 256,
        "repetition_penalty": 1.0,
        "temperature": 0.6,
        "top_k": 10,
        "top_p": 0.9,
        "batch_size": 100,
        "typical_p": 0.5,
        "watermark": false,
        "perf_stat": false
      }
    }'
    ```
  - Trition Generate
    ```shell
    curl 127.0.0.1:1025/v2/models/cogvlm2/generate -d '{
      "id": "a123",
      "text_input": [
        {
          "type": "text",
          "text": "what is the general content of this video?"
        },
        {
          "type": "video_url",
          "video_url": "/path/of/image/video.mp4"
        }
      ],
      "parameters": {
        "details": true,
        "do_sample": true,
        "max_new_tokens": 256,
        "repetition_penalty": 1.1,
        "temperature": 1,
        "top_k": 10,
        "top_p": 0.99,
        "batch_size": 100,
        "typical_p": 0.5,
        "watermark": false,
        "perf_stat": false
      }
    }'
    ```
  - OpenAI
    ```shell
    curl 127.0.0.1:1025/v1/chat/completions -d '{
      "model": "cogvlm2_llama3_video",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "video_url",
              "video_url": "/path/of/image/video.mp4"
            },
            {
              "type": "text",
              "text": "what is the general content of this video?"
            }
          ]
        }
      ],
      "max_tokens": 8192,
      "presence_penalty": 1.0,
      "frequency_penalty": 1.0,
      "temperature": 0.6,
      "top_p": 0.9,
      "stream": false,
      "repetition_penalty": 1.0,
      "top_k": 1,
      "do_sample": false,
      "stop": [
        "If"
      ],
      "adapter_id": "aaaaaaaaaa"
    }'
    ```
  - OpenAI多轮对话样例
    ```shell
    curl 127.0.0.1:1025/v1/chat/completions -d '{
      "model": "cogvlm2_llama3_video",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "what is the general content of this video?"
            }
          ]
        },
        {
          "role": "system",
          "content": [
            {
              "type": "text",
              "text": " The video showcases a variety of scenes featuring birds in flight and marine life. It begins with a single bird soaring against the sky, followed by two seagulls gliding over the ocean. A school of fish is then seen swimming in formation, creating ripples on the water's surface. Next, an aerial view captures dolphins leaping energetically from the sea into sunlight-dappled waters. Finally, gannets are shown diving towards baitfishes above them, highlighting their hunting behavior amidst a serene underwater environment devoid of human presence."
            }
          ]
        },
        {
          "role": "user",
          "content": [
            {
              "type": "video_url",
              "video_url": "/path/of/image/video.mp4"
            },
            {
              "type": "text",
              "text": "Please give me a more detailed description."
            }
          ]
        }
      ],
      "max_tokens": 8192,
      "presence_penalty": 1.0,
      "frequency_penalty": 1.0,
      "temperature": 0.6,
      "top_p": 0.9,
      "stream": false,
      "repetition_penalty": 1.0,
      "top_k": 1,
      "do_sample": false,
      "stop": [
        "If"
      ],
      "adapter_id": "aaaaaaaaaa"
    }'
    ```

## FAQ

- 更多环境变量见[此 README 文件](../../README.md)
- 对话测试实际执行的 Python 文件为`${llm_path}/examples/run_pa.py`；