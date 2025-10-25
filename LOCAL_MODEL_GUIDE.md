# LLM本地模型使用指南

## 📋 支持的模型

1. **Qwen1.5-0.5B** - 轻量级中文对话模型
   - 模型大小: ~1GB
   - 推理速度: 较快
   - 适用场景: 中文对话、快速响应

2. **TinyLlama-1.1B** - 轻量级英文对话模型
   - 模型大小: ~2GB
   - 推理速度: 中等
   - 适用场景: 英文对话、通用任务

## 🚀 使用方法

### 1. 配置模型

编辑 `config.yaml`:

```yaml
llm:
  mode: "local"  # 切换到本地模式

  local:
    model_name: "qwen"  # 或 "tinyllama"
    qwen_model_path: "/home/HwHiAiUser/.mindnlp/model/Qwen/Qwen1.5-0.5B-Chat"
    tinyllama_model_path: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    max_tokens: 128
    temperature: 1.0
```

### 2. 启动服务

```bash
# 确保使用llm环境
conda activate llm

# 启动LLM服务
python llm_service/app_fastapi.py
```

或使用统一启动脚本:

```bash
python start_all.py
```

## ⚙️ 环境变量说明

为防止模型加载崩溃,服务会自动设置以下环境变量:

```bash
export TE_PARALLEL_COMPILER=1
export MAX_COMPILE_CORE_NUMBER=1
export MS_BUILD_PROCESS_NUM=1
export MAX_RUNTIME_CORE_NUMBER=1
export MS_ENABLE_IO_REUSE=1
```

这些变量会在模型加载前自动设置,无需手动配置。

## 📝 模型切换

### 切换到Qwen模型

```yaml
llm:
  mode: "local"
  local:
    model_name: "qwen"
```

### 切换到TinyLlama模型

```yaml
llm:
  mode: "local"
  local:
    model_name: "tinyllama"
```

### 切换回API模式

```yaml
llm:
  mode: "api"
```

## ⚠️ 注意事项

### 1. 模型加载时间

- **Qwen1.5-0.5B**: 首次加载约30-60秒
- **TinyLlama**: 首次加载约60-120秒

请耐心等待服务启动。

### 2. 内存要求

- **Qwen1.5-0.5B**: 至少2GB可用内存
- **TinyLlama**: 至少4GB可用内存

### 3. 崩溃预防

模型加载和推理过程中可能会崩溃,已采取以下措施:

1. ✅ 设置MindSpore环境变量限制编译和运行时核心数
2. ✅ 禁用多线程 (`disable_multi_thread()`)
3. ✅ 使用float16精度减少内存占用

如果仍然崩溃,请尝试:
- 重启服务
- 检查系统内存是否充足
- 切换到另一个模型

## 🧪 测试本地模型

### 方法1: Web界面测试

1. 启动服务
2. 访问 http://localhost:8080
3. 在"LLM配置"页面:
   - 设置模式为 `local`
   - 选择模型名称 (`qwen` 或 `tinyllama`)
   - 保存配置
4. 在测试区域输入文本测试

### 方法2: API测试

```bash
curl -X POST "http://localhost:5002/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "你好,请介绍一下自己",
    "history": []
  }'
```

### 方法3: 流式测试

```bash
curl -X POST "http://localhost:5002/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "讲个笑话",
    "history": []
  }'
```

## 📊 性能对比

| 模型 | 加载时间 | 首Token延迟 | Token/秒 | 内存占用 |
|------|---------|------------|----------|---------|
| Qwen1.5-0.5B | ~30s | ~2s | 5-10 | ~1.5GB |
| TinyLlama | ~60s | ~3s | 3-8 | ~3GB |
| DeepSeek API | 0s | ~0.5s | 20-50 | 0 |

## 💡 推荐使用场景

### 使用本地模型:
- ✅ 网络不稳定或断网环境
- ✅ 对隐私要求高
- ✅ 需要离线运行
- ✅ API费用较高

### 使用API模式:
- ✅ 需要高质量回复
- ✅ 需要快速响应
- ✅ 硬件资源有限
- ✅ 网络稳定

## 🔧 故障排查

### 问题1: 模型加载失败

**错误**: `ModuleNotFoundError: No module named 'mindspore'`

**解决**:
```bash
conda activate llm
pip install mindspore mindnlp
```

### 问题2: 模型路径错误

**错误**: `FileNotFoundError: model not found`

**解决**:
检查配置文件中的模型路径是否正确:
```yaml
llm:
  local:
    qwen_model_path: "/home/HwHiAiUser/.mindnlp/model/Qwen/Qwen1.5-0.5B-Chat"
```

### 问题3: 服务崩溃

**现象**: 服务启动后立即退出

**解决**:
1. 查看日志: `cat logs/LLM.log`
2. 检查内存是否充足: `free -h`
3. 尝试切换模型
4. 降低max_tokens值

### 问题4: 推理速度慢

**解决**:
```yaml
llm:
  local:
    max_tokens: 64  # 减少生成长度
    temperature: 0.7  # 降低随机性可能稍快
```

## 📚 更多信息

- Qwen文档: https://github.com/QwenLM/Qwen
- TinyLlama文档: https://github.com/jzhang38/TinyLlama
- MindSpore文档: https://www.mindspore.cn/

## 🆘 需要帮助?

遇到问题请:
1. 查看服务日志: `./view_logs.sh LLM`
2. 运行诊断: `python diagnose.py`
3. 查看本文档的故障排查部分
