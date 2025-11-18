# README

- [cogvlm2_llama3_chinese_chat_19B](https://huggingface.co/THUDM/cogvlm2-llama3-chinese-chat-19B)cogvlm2_llama3_chinese_chat_19B 是由智谱AI（Zhipu.AI）推出的新一代多模态大型语言模型。它是基于 Meta-Llama-3-8B-Instruct 构建的，拥有 19 亿参数，支持中文和英文两种语言。CogVLM2-Llama3-Chinese-Chat-19B 模型具备图像理解与对话模型的功能，能够处理高达 8K 的文本长度和 1344x1344 分辨率的图片。
- 此代码仓中实现了一套基于NPU硬件的cogvlm2_llama3_chinese_chat_19B推理模型。配合加速库使用，旨在NPU上获得极致的推理性能。

# 使用说明


# 特性矩阵
| 模型及参数量    | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | 800I A2 BF16 | MindIE Service | 纯模型支持模态  | 服务化支持模态 |
| --------------- |-------------------------|-----------------------------| ---- |--------------|----------------| -------------- | ------------ |
| cogvlm2-llama3-chinese-chat-19B | 支持world size 1,2,4,8    | 不支持  | √    | √            | √              | 文本、图片      | 文本、图片 |


## 路径变量解释

| 变量名               | 含义                                                                                                                                             |
|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| working_dir       | 加速库及模型库下载后放置的目录                                                                                                                                |
| llm_path          | 模型仓所在路径。若使用编译好的包，则路径为 `${working_dir}/MindIE-LLM/`；若使用gitee下载的代码，则路径为 `${working_dir}/MindIE-LLM/examples/atb_models`                          |
| script_path       | 脚本所在路径；cogvlm2_llama3_chinese_chat_19B的工作脚本所在路径为 `${llm_path}/examples/models/cogvlm2`                                                         |
| weight_path       | 模型权重路径                                                                                                                                         |
| image_path        | 图片所在路径                                                                                                                                         |
| max_input_length  | 多模态模型的最大embedding长度。 |
| max_output_length | 生成的最大token数                                                                                                                                    |

## 权重

**权重下载**

- [cogvlm2_llama3_chinese_chat_19B](https://huggingface.co/THUDM/cogvlm2-llama3-chinese-chat-19B)

**基础环境变量**

-1.Python其他第三方库依赖，参考[requirements_cogvlm2_llama3_chinese_chat_19B.txt](../../../requirements/models/requirements_cogvlm2_llama3_chinese_chat_19B.txt)
-2.参考[此README文件](../../../README.md)
-注意：保证先后顺序，否则cogvlm2_llama3_chinese_chat_19B的其余三方依赖会重新安装torch，导致出现别的错误

**原始文件拷贝与修改**

- 修改权重文件中的config.json，增加字段：

  ```shell
  "model_type": "cogvlm2",
  "torch_dtype": "bfloat16"或"float16",
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
    
      # 配置使用的核心
      export ASCEND_RT_VISIBLE_DEVICES=0
    ```
    - 核心ID查阅方式见[此README文件](../../README.md)的【启动脚本相关环境变量】章节
    - 更多环境变量见[此README文件](../../README.md)

    
- 在run_pa.py中，可以修改main函数的infer_params["mm_inputs"]，一次进行多种输入的测试，例如
    ```shell
    "mm_inputs": [
            InputAttrs(args.input_texts, args.image_path), 
            InputAttrs(["<|reserved_special_token_0|> " * 0 + "描述这张图片。"], args.image_path), 
            InputAttrs(args.input_texts, "/image/xxxxx.jpeg"), 
            InputAttrs(args.input_texts, "")
            ],
    ```

### 服务化对话测试

- 运行启动命令样例
  - VLLM 
    ```shell
    curl 127.0.0.1:1025/generate -d '{
      "prompt": [
        {
          "type": "text",
          "text": "Question: Describe this image. Answer:"
        },
        {
          "type": "image_url",
          "image_url": "/path/of/image/img.jpg"
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
          "text": "Describe this image."
        },
        {
          "type": "image_url",
          "image_url": "/path/of/image/img.jpg"
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
          "text": "Describe this image."
        },
        {
          "type": "image_url",
          "image_url": "/path/of/image/img.jpg"
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
          "text": "Describe this image."
        },
        {
          "type": "image_url",
          "image_url": "/path/of/image/img.jpg"
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
      "model": "cogvlm2",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "image_url",
              "image_url": "/path/of/image/img.jpg"
            },
            {
              "type": "text",
              "text": "Describe this image."
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
      "model": "cogvlm2",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "Describe this image."
            }
          ]
        },
        {
          "role": "system",
          "content": [
            {
              "type": "text",
              "text": "The image features a cartoon character that resembles a cat with its black color and ears, sitting on what appears to be the ground. The character has large, round eyes that are wide open, suggesting surprise or curiosity. The background is simplistic, implying an outdoor setting with shades of green that could represent grass or foliage. The art style is reminiscent of animated cartoons from the early 2000s, characterized by its clean lines and vibrant colors."
            }
          ]
        },
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "Can you translate it in Chinese?"
            }
          ]
        },
        {
          "role": "system",
          "content": [
            {
              "type": "text",
              "text": "图片中的卡通人物看起来像一只猫，它是黑色的，坐在绿色的地面上。它有着大大的、圆溜溜的眼睛，眼睛瞪得大大的，看起来很惊讶或好奇。背景非常简单，只有不同深浅的绿色，这可能代表草地或树叶。 这种艺术风格让人想起2000年代初的动画卡通，其线条简洁、色彩鲜艳。"
            }
          ]
        },
        {
          "role": "user",
          "content": [
            {
              "type": "image_url",
              "image_url": "/path/of/image/img.jpg"
            },
            {
              "type": "text",
              "text": "No you are wrong, this is a black dog."
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


## 精度测试

- 分别在GPU与NPU机器上测试textVqa数据集精度，其中config.json中torch_dtype为"bfloat16"。

GPU:
- L20得分：81.16

NPU:
- 800I_A2_64G得分：82.16

## FAQ

- 更多环境变量见[此 README 文件](../../README.md)
- 对话测试实际执行的 Python 文件为`${llm_path}/examples/run_pa.py`；                                                                                                                                                                                                                                                                                   