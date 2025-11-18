# README

- [Tencent-Hunyuan-Large](https://github.com/Tencent/Tencent-Hunyuan-Large)是腾讯混元LLM团队发布的专家混合（MoE）语言模型，其特点是高质量合成数据，KV缓存压缩，专家特定学习率缩放，长上下文处理能力和广泛的基准测试。

- 此代码仓中实现了一套基于NPU硬件的Tencent-Hunyuan-Large推理模型。配合加速库使用，旨在NPU上获得极致的推理性能。

# 特性矩阵
- 此矩阵罗列了Tencent-Hunyuan-Large模型支持的特性

| 模型及参数量 | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16（仅800I A2支持） | Flash Attention | Paged Attention | W8A8量化 | W8A16量化  |KV cache量化 | 稀疏量化（仅300I DUO支持） | MindIE Service | TGI | 长序列  |
|-------------|----------------------------|-----------------------------|------|----------------------|-----------------|-----------------|---------|-----------|-----------|--------------|--------------------------|--------|-----|
| Hunyuan-A52B-Instruct    | 支持world size 16(64G)     | ×                | √   | √                   | ×              | √              | ×       | ×              | ×           | ×                       | ×     | ×  | ×  |


## 路径变量解释

| 变量名      | 含义                                                                                                                                                     |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| working_dir     | 加速库及模型库下载后放置的目录                                                                                                                           |
| llm_path        | 模型仓所在路径。若使用编译好的包，则路径为`${working_dir}/MindIE-LLM/`；若使用 gitee 下载的代码，则路径为`${working_dir}/MindIE-LLM/examples/atb_models` |
| script_path     | 脚本所在路径；Deepseek-MoE 的工作脚本所在路径为`${llm_path}/examples/models/hunyuan`                                                                    |
| weight_path     | 模型权重路径                                                                                                                                             |
| rank_table_path | Rank table文件路径                                                                                                                                              |

## 权重

**权重下载**

- [Hunyuan-A52B-Instruct](https://huggingface.co/tencent/Tencent-Hunyuan-Large/tree/main/Hunyuan-A52B-Instruct)

## 生成量化权重

- 生成量化权重依赖msModelSlim工具，安装方式见[此README](https://gitee.com/ascend/msit/tree/dev/msmodelslim)。
- 量化权重统一使用`${llm_path}/examples/convert/model_slim/quantifier.py`脚本生成，以下提供Hunyuan-Large模型量化权重生成快速启动命令，各模型量化方式的具体参数配置见`${llm_path}/examples/models/hunyuan/generate_quant_weight.sh`
- 当前Hunyuan-Large支持W8A8 dynamic量化，通过以下命令生成量化权重：
```shell
# 设置CANN包的环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 生成w8a8 dynamic量化权重
bash examples/models/hunyuan/generate_quant_weight.sh -src {浮点权重路径} -dst {量化权重路径} -type w8a8_dynamic -trust_remote_code

**基础环境变量**

- 参考[此 README 文件](../../../README.md)

## 使用说明
- trust_remote_code为可选参数代表是否信任本地的可执行文件：默认不执行。传入此参数，则信任本地可执行文件。

## 对话测试
**运行Paged Attention FP16**
- 运行启动脚本
  - 在\${llm_path}目录下执行以下指令
    ```shell
    bash ${script_path}/run_pa.sh (--trust_remote_code) ${weight_path}
    ```

## 精度测试

- 参考[此 README 文件](../../../tests/modeltest/README.md)
  - 单机示例
    ```shell
    cd ${llm_path}/tests/modeltest
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    export ATB_LLM_ENABLE_AUTO_TRANSPOSE=0
    bash run.sh pa_bf16 full_BoolQ 1 hunyuan ${weight_path} (trust_remote_code) 16
    bash run.sh pa_bf16 full_CEval 5 1 hunyuan ${weight_path} (trust_remote_code) 16
    ```
  - 双机示例
    ```shell
    cd ${llm_path}/tests/modeltest
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export ATB_LLM_BENCHMARK_ENABLE=1
    export ATB_LLM_ENABLE_AUTO_TRANSPOSE=0

    # 以下两条命令需要在两个节点同步执行
    # 节点1
    bash run.sh pa_bf16 full_BoolQ 1 hunyuan ${weight_path} (trust_remote_code) 
    ${rank_table_path} 16 2 0 [master_address]
    # 节点2
    bash run.sh pa_bf16 full_BoolQ 1 hunyuan ${weight_path} (trust_remote_code) 
    ${rank_table_path} 16 2 8 [master_address]
    ```

## 性能测试

- 参考[此 README 文件](../../../tests/modeltest/README.md)
  - 单机示例
    ```shell
    cd ${llm_path}/tests/modeltest
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    export ATB_LLM_BENCHMARK_ENABLE=1
    bash run.sh pa_bf16 performance [[2048,2048],[1024,1024],[512,512],[256,256]] 1 hunyuan 
    ${weight_path} (trust_remote_code) 16
    ```
  - 双机示例
    ```shell
    cd ${llm_path}/tests/modeltest
    export HCCL_OP_EXPANSION_MODE="AIV"
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export ATB_LLM_BENCHMARK_ENABLE=1

    # 以下两条命令需要在两个节点同步执行
    # 节点1
    bash run.sh pa_bf16 performance [[2048,2048],[1024,1024],[512,512],[256,256]] 1 hunyuan 
    ${weight_path} (trust_remote_code) ${rank_table_path} 16 2 0 [master_address]
    # 节点2
    bash run.sh pa_bf16 performance [[2048,2048],[1024,1024],[512,512],[256,256]] 1 hunyuan 
    ${weight_path} (trust_remote_code) ${rank_table_path} 16 2 8 [master_address]
    ```

## FAQ

- 更多环境变量见[此 README 文件](../../README.md)
- 对话测试实际执行的 Python 文件为`${llm_path}/examples/run_pa.py`；这个文件的参数说明见[此 README 文件](../../README.md)