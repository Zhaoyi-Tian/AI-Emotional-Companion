# README

- [WeMM](https://github.com/scenarios/WeMM)，是 WeChatCV 推出的最新一代多模态大语言模型。WeMM 具备动态高分辨率图片下的中英双语对话能力，在多模态大语言模型的榜单中是百亿参数级别最强模型，整体测评结果（Avg Rank）位居第一梯队 (vlm_leaderboard)。
- 此代码仓中实现了一套基于NPU硬件的WeMM推理模型。配合加速库使用，旨在NPU上获得极致的推理性能。
- 参考并使用了Idefics2中融入了navit980结构的base vision backbone代码，以及Internlm2的LLM框架，实现了图片文本模型的多模态中英文双语推理。

# 特性矩阵
- 此矩阵罗列了 WeMM 系列模型支持的特性

|        模型及参数量        | 800I A2 Tensor Parallelism | FP16 | 800I A2 BF16 | MindIE Service | 纯模型支持模态  | 服务化支持模态 |
| ------------------------- | ---------------------------| ---- | ------------ | -------------- | -------------- | ------------ |
|      WeMM-Chat-2k-CN      |      支持 worldsize 1,2,4,8      | √    | ×            |        √       | 文本、图片      | 文本、图片 |

# 使用说明

## 路径变量解释

| 变量名              | 含义                                                                                                                                                         |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| working_dir         | 加速库及模型库下载后放置的目录                                                                                                                               |
| llm_path            | 模型仓所在路径。若使用编译好的包，则路径为 `${working_dir}`；若使用gitee下载的代码，则路径为 `${working_dir}/MindIE-LLM/examples/atb_models` |
| script_path         | 脚本所在路径；工作脚本所在路径为 `${llm_path}/examples/models/wemm`                                                                        |
| src_weight_path     | huggingface上下载的模型原始权重路径                                                   |                                                  |
| weight_path         | 重命名后的模型权重路径                                                   |                                                  |
| image_path          | 图片所在路径                                                    |                                         |
## 权重

### 权重下载

- [WeMM-Chat-2k-CN](https://huggingface.co/feipengma/WeMM-Chat-2k-CN/tree/main)

### 权重重命名
> **说明：**
> hugging face上下载下来的原始权重命名中带有`*.original_linear.*`，不符合加速库导入权重时遵循的命名规则。
> - 请按照如下方式进行转换
- 脚本：`${llm_path}/examples/models/wemm/rename_weights.py`
- 功能：将 hugging face 上下载下来的 WeMM-Chat-2k-CN 原始权重重命名为符合框架规范的权重，并保存到`${weight_path}`目录下
- 参数说明
  | 参数名称        | 是否为必选 | 类型    | 默认值 | 描述         |
  |----------------|-----------|---------|-------|--------------|
  | src_model_path | 是        | string  |        | 原始模型权重路径 |
  | save_directory | 是        | string  |        | 重命名后的模型权重路径 |  
- 示例
    ```shell
    cd ${llm_path}
    python examples/models/wemm/rename_weights.py --src_model_path ${src_weight_path} --save_directory ${weight_path}
    ```
  - 注意：必须先进入`${llm_path}`路径下执行以上命令，否则由于脚本中存在相对路径，会导致module not found的问题

## 推理前准备

**环境配置**

- Toolkit, MindIE/ATB，ATB-SPEED等，参考[此README文件](../../../README.md)
- 安装Python其他第三方库依赖，参考[requirements_wemm.txt](../../../requirements/models/requirements_wemm.txt)
  ```shell
  pip install -r ${llm_path}/requirements/models/requirements_wemm.txt
  ```

**运行 Mindie Paged Attention FP16前修改配置**

- 修改模型权重路径下 `config.json` 文件：
1. `model_type` 字段为 wemm
2. `tokenizer_path` 字段为 ${weight_path}
3. `torch_dtype` 字段为 ${float16}
  ```shell
  {
    ...
    "model_type": "wemm",
    ...
    "tokenizer_path": "$weight_path",
    "torch_dtype": "float16",
  }
  ```

- 增加lora适配文件：adapter_config.json 和 lora_adapter.json，放在`${weight_path}`路径下：

```shell
# adapter_config.json 文件内容为：
{
  "lora_alpha": 512,
  "r": 512
}

# 配置lora权重路径，WeMM-Chat-2k-CN的lora权重都融合在模型权重中，因此，这里配置模型权重实际路径 $weight_path 即可。
# lora_adapter.json文件内容为：
{"wemm":"$weight_path"}
```

## 推理

**运行**
- 修改启动脚本 `${llm_path}/examples/models/wemm/run_pa.sh`
    - 修改启动脚本中 `input_texts_for_image` 为输入prompt。

- 执行启动脚本 `${llm_path}/examples/models/wemm/run_pa.sh`
    ```shell
    bash ${llm_path}/examples/models/wemm/run_pa.sh --precision (--trust_remote_code) \
    --model_path ${weight_path} \
    --image_or_video_path ${image_path}
    ```
- 其他支持的推理参数请参考 `${llm_path}/examples/models/wemm/wemm.py` 文件。

## 精度测试
### TextVQA
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
- 进入以下目录 `${llm_path}/tests/modeltest`
  ```shell
  cd ${llm_path}/tests/modeltest
  ```
- 安装modeltest及其三方依赖
  ```shell
  pip install --upgrade pip
  pip install -e .
  pip install tabulate termcolor 
  ```
- 将 `modeltest/config/model/wemm.yaml` 中的model_path的值修改为模型权重的绝对路径
  ```yaml
  model_path: /data_mm/weights/WeMM-Chat-2k-CN
  ```
- 将 `modeltest/config/task/textvqa.yaml` 中的model_path修改为textvqa_val.jsonl文件的绝对路径
  ```yaml
  local_dataset_path: /data_mm/datasets/textvqa_val/textvqa_val.jsonl
  ```
- 设置可见卡数，修改 `mm_run.sh` 文件中的ASCEND_RT_VISIBLE_DEVICES。依需求设置单卡或多卡可见。
  ```shell
  export ASCEND_RT_VISIBLE_DEVICES=0
  ```
- 运行测试命令
  ```shell
  bash scripts/mm_run.sh textvqa wemm
  ```
- 测试结果保存于以下路径。其下的results/..(一系列文件夹嵌套)/\*\_result.csv中存放着modeltest的测试结果。debug/..(一系列文件夹嵌套)/output\_\*.txt中存储着每一条数据的运行结果，第一项为output文本，第二项为输入infer函数的第一个参数的值，即模型输入。第三项为e2e_time。
  ```shell
  output/$DATE/modeltest/$MODEL_NAME/precision_result/
  ```

## 性能测试

性能测试时需要在 `${image_path}` 下仅存放一张图片，使用以下命令运行 `run_pa.sh`，会自动输出batch_size为1，输出token长度为 256 时的吞吐。

```shell
bash ${script_path}/run_pa.sh --performance ${weight_path} ${image_path}
```

例如在 MindIE-ATB-Models 根目录，可以运行：

```shell
bash examples/models/wemm/run_pa.sh --performance ${weight_path} ${image_path}
```

可以在 `examples/models/wemm` 路径下找到测试结果。

## FAQ
- 在精度测试和性能测试时，用户如果需要修改输入prompt，max_batch_size，max_output_length时，可以修改{script_path}/run_pa.sh里的可修改配置
- 更多环境变量见[此README文件](../../README.md)