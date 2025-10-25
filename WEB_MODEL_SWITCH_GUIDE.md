# Web界面模型切换使用指南

## 🎯 功能说明

Web配置界面现已支持完整的本地模型配置和切换功能,包括:

1. ✅ **可视化配置** - 通过Web界面配置本地模型
2. ✅ **模型切换** - 在Qwen和TinyLlama之间切换
3. ✅ **一键重载** - 切换后一键重新加载LLM服务
4. ✅ **实时测试** - 配置后立即测试模型效果

## 🚀 使用步骤

### 1. 访问Web界面

启动服务后访问:
- 本地: http://localhost:8080
- 内网: http://你的IP:8080

### 2. 进入LLM配置页面

点击 **"🧠 LLM配置"** 标签页

### 3. 配置本地模型

#### 切换到本地模式

1. 在 **"运行模式"** 中选择 `local`

#### 选择模型

2. 在 **"本地模型配置"** 区域:
   - **本地模型选择**: 选择 `qwen` 或 `tinyllama`
   - **模型路径**:
     - Qwen: `/home/HwHiAiUser/.mindnlp/model/Qwen/Qwen1.5-0.5B-Chat`
     - TinyLlama: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

#### 调整参数

3. 根据需要调整:
   - **最大Token数**: 32-512 (建议128)
   - **Temperature**: 0.0-2.0 (建议1.0)
   - **系统提示词**: 自定义助手角色

#### 保存配置

4. 点击 **"💾 保存LLM配置"** 按钮

   系统会提示:
   ```
   ✅ LLM配置已保存

   ⚠️ 如果切换了模式或本地模型,请点击下方'重新加载LLM服务'按钮使配置生效
   ```

### 4. 重新加载服务

#### 重要!必须执行此步骤

在 **"重新加载LLM服务"** 区域:

1. 点击 **"🔄 重新加载LLM服务"** 按钮

2. 等待加载完成(30-60秒)

   显示信息:
   ```
   ✅ LLM服务已重新加载

   模式: 本地模型
   模型: qwen

   ⚠️ 本地模型加载需要30-60秒,请耐心等待...
   ```

3. 如果超时,稍后在"📊 服务状态"页面检查LLM服务状态

### 5. 测试模型

在 **"测试LLM服务"** 区域:

1. 输入测试问题,例如: `你好,请介绍一下自己`

2. 点击 **"🧪 测试对话"** 按钮

3. 查看回复结果

## 📊 配置界面说明

### API配置 (在线模式)

```
运行模式: api

API配置:
├── API提供商: deepseek
├── API Key: sk-xxx
├── API URL: https://api.deepseek.com/v1/chat/completions
├── 模型名称: deepseek-chat
├── 最大Token数: 512
├── Temperature: 1.0
└── 系统提示词: You are a helpful assistant
```

### 本地模型配置 (离线模式)

```
运行模式: local

本地模型配置:
├── 本地模型选择: qwen / tinyllama
├── Qwen模型路径: /path/to/qwen
├── TinyLlama模型路径: /path/to/tinyllama
├── 最大Token数: 128
├── Temperature: 1.0
└── 系统提示词: You are a helpful chatbot
```

## 🔄 模型切换流程

### 从API切换到本地模型

1. **运行模式** → 选择 `local`
2. **本地模型选择** → 选择 `qwen` 或 `tinyllama`
3. **保存配置** → 点击"💾 保存LLM配置"
4. **重新加载** → 点击"🔄 重新加载LLM服务"
5. **等待** → 30-60秒模型加载
6. **测试** → 输入问题测试

### 在本地模型间切换

例如从Qwen切换到TinyLlama:

1. **本地模型选择** → 从 `qwen` 改为 `tinyllama`
2. **保存配置** → 点击"💾 保存LLM配置"
3. **重新加载** → 点击"🔄 重新加载LLM服务"
4. **等待** → 60-120秒模型加载
5. **测试** → 验证新模型

### 从本地模型切换回API

1. **运行模式** → 选择 `api`
2. **保存配置** → 点击"💾 保存LLM配置"
3. **重新加载** → 点击"🔄 重新加载LLM服务"
4. **测试** → 立即可用

## ⚠️ 注意事项

### 1. 必须重新加载

⚠️ **保存配置后必须点击"重新加载LLM服务"才能生效!**

配置文件更新后,服务不会自动重启,需要手动重新加载。

### 2. 加载时间

不同模型的加载时间:
- **Qwen1.5-0.5B**: 30-60秒
- **TinyLlama**: 60-120秒
- **API模式**: 立即生效

### 3. 超时提示

如果看到"⚠️ 请求超时",不要担心:
- 模型可能还在加载中
- 稍后在"📊 服务状态"页面检查
- 等待2-3分钟后再测试

### 4. 内存需求

确保有足够内存:
- **Qwen**: 至少2GB可用
- **TinyLlama**: 至少4GB可用

### 5. 模型路径

确保路径正确:
```bash
# 检查Qwen模型是否存在
ls /home/HwHiAiUser/.mindnlp/model/Qwen/Qwen1.5-0.5B-Chat

# 检查TinyLlama模型(可能需要首次下载)
```

## 🧪 测试建议

### 1. 测试问题示例

**中文测试** (Qwen推荐):
```
你好,请用一句话介绍你自己
今天天气怎么样?
给我讲个笑话
```

**英文测试** (TinyLlama推荐):
```
Hello, please introduce yourself
What's the weather like today?
Tell me a joke
```

### 2. 对比测试

1. 先测试API模式
2. 切换到Qwen本地模型测试
3. 切换到TinyLlama本地模型测试
4. 对比回复质量和速度

## 📋 常见问题

### Q1: 点击重新加载后没反应?

**A**: 本地模型加载需要时间,请:
1. 等待30-60秒
2. 到"📊 服务状态"页面查看LLM服务状态
3. 如果显示healthy,说明加载成功

### Q2: 重新加载失败?

**A**: 检查:
1. LLM服务是否正在运行
2. 模型路径是否正确
3. 是否有足够内存
4. 查看日志: `./view_logs.sh LLM`

### Q3: 切换模型后回复质量差?

**A**: 调整参数:
1. 降低Temperature (0.7-0.8)
2. 修改系统提示词
3. 尝试不同的模型

### Q4: 本地模型太慢?

**A**:
1. 降低max_tokens (64-128)
2. 考虑使用API模式
3. 检查系统负载

## 💡 推荐配置

### 快速响应 (推荐API)
```yaml
模式: api
模型: deepseek-chat
Max Tokens: 256
Temperature: 1.0
```

### 离线使用 (推荐Qwen)
```yaml
模式: local
模型: qwen
Max Tokens: 128
Temperature: 1.0
```

### 英文对话 (推荐TinyLlama)
```yaml
模式: local
模型: tinyllama
Max Tokens: 128
Temperature: 0.7
```

## 📞 需要帮助?

1. 查看服务日志: 在Web界面 → 📊 服务状态 → 检查服务状态
2. 查看详细日志: `./view_logs.sh LLM`
3. 参考文档: `LOCAL_MODEL_GUIDE.md`
4. 运行诊断: `python diagnose.py`
