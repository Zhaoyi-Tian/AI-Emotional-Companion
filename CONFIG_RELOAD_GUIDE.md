# 配置热重载功能说明

## 功能概述

配置热重载功能允许你在不重启服务的情况下更新大部分配置，让配置立即生效。这极大提升了开发和调试的效率。

## 支持的服务

所有主要服务现在都支持配置热重载：

- ✅ **ASR 服务** - 自动重新加载模型
- ✅ **LLM 服务** - 支持API配置和模型切换
- ✅ **TTS 服务** - 即时更新语音配置
- ✅ **Orchestrator 服务** - 更新流式处理配置
- ✅ **Voice Chat 服务** - 热更新VAD、唤醒词、打断词等配置

## 使用方法

### 方法 1: 通过 Web UI（推荐）

1. 访问 Web UI: http://localhost:8080
2. 在对应的配置页面修改参数
3. 点击"保存配置"按钮
4. **配置会自动热重载**，无需手动操作！

Web UI 会自动：
- 保存配置到 config.yaml
- 调用对应服务的 `/reload_config` API
- 显示热重载结果（成功/失败/需要重启）

### 方法 2: 手动调用 API

如果你直接编辑了 `config.yaml` 文件，可以手动调用热重载API：

```bash
# 重新加载 ASR 服务配置
curl -X POST http://localhost:5001/reload_config

# 重新加载 LLM 服务配置
curl -X POST http://localhost:5002/reload_config

# 重新加载 TTS 服务配置
curl -X POST http://localhost:5003/reload_config

# 重新加载 Orchestrator 配置
curl -X POST http://localhost:5000/reload_config

# 重新加载 Voice Chat 配置
curl -X POST http://localhost:5004/reload_config

# 一键重新加载所有服务（在 Web UI 中点击按钮，或调用 reload_all_services）
```

### 方法 3: 使用测试脚本

运行测试脚本来验证热重载功能：

```bash
python test_config_reload.py
```

该脚本会：
1. 检查所有服务的健康状态
2. 测试每个服务的配置热重载功能
3. 显示详细的测试结果

## 配置热重载详情

### ASR 服务

**支持热重载的配置:**
- ✅ 模型类型切换 (CN/EN)
- ✅ 自动重新加载对应的模型

**注意事项:**
- 模型切换需要重新加载模型，可能需要几秒钟
- 超时时间设置为 10 秒

### LLM 服务

**支持热重载的配置:**
- ✅ 模式切换 (api/local)
- ✅ API 配置（API Key、URL、模型名称等）
- ✅ 温度、最大tokens等参数
- ✅ 系统提示词
- ✅ 本地模型切换（会自动清理旧模型并加载新模型）

**注意事项:**
- 本地模型加载可能需要较长时间，超时设置为 30 秒
- 模型切换会自动进行垃圾回收释放内存

### TTS 服务

**支持热重载的配置:**
- ✅ 模式切换 (api/local)
- ✅ API 配置（API Key、模型、音色等）
- ✅ 即时更新语音合成参数

### Orchestrator 服务

**支持热重载的配置:**
- ✅ 服务 URL 配置
- ✅ 流式处理配置（句子分隔符、最小块长度等）

**工作原理:**
- Orchestrator 大部分配置是动态读取的，热重载主要用于通知配置已更新

### Voice Chat 服务

**支持热重载的配置:**
- ✅ VAD 参数（静音阈值、静音时长、最小音频长度）
- ✅ 音量参数（输出音量）
- ✅ 唤醒词配置（启用/禁用、唤醒词列表、确认回复）
- ✅ 打断词配置（启用/禁用、打断词列表）
- ✅ 音频设备配置（输入/输出设备）*

***需要重启的配置:**
- ⚠️ 音频设备切换需要停止并重新启动语音对话才能生效

**热重载返回信息:**
热重载成功后会返回具体的配置变更：
```json
{
  "success": true,
  "message": "配置已重新加载",
  "changes": {
    "silence_threshold": 300,
    "output_volume": 80,
    "wake_mode": true,
    "interrupt_mode": true
  }
}
```

## 常见问题

### Q1: 我修改了配置但没有生效？

**A:** 确保：
1. 配置文件 `config.yaml` 已正确保存
2. 通过 Web UI 保存配置（会自动调用热重载），或手动调用 `/reload_config` API
3. 查看服务日志确认热重载是否成功

### Q2: 哪些配置不能热重载？

**A:** 以下配置需要重启服务：
- 服务端口更改（需要重启对应服务）
- Voice Chat 的音频设备切换（需要停止并重新启动语音对话）
- 某些底层硬件相关配置

### Q3: 热重载失败怎么办？

**A:**
1. 查看服务日志，了解失败原因
2. 确认配置文件格式正确（YAML 格式）
3. 如果是模型加载失败，检查模型路径是否正确
4. 最后的办法：重启对应的服务

### Q4: 如何知道热重载是否成功？

**A:** 有三种方式：
1. **Web UI**: 保存配置后会显示热重载结果
2. **API 响应**: 调用 `/reload_config` API 会返回成功/失败信息
3. **服务日志**: 查看对应服务的日志文件（`logs/` 目录）

## 实现原理

### 配置加载器（ConfigLoader）

配置加载器是一个单例类，维护全局配置状态：
```python
# 重新加载配置
reload_config()  # 更新 ConfigLoader 内存中的 config 字典
```

### 服务热重载

每个服务实现了 `reload_config()` 方法或 API 端点：
```python
# 服务实例方法
assistant.reload_config()  # 更新实例变量

# API 端点
POST /reload_config  # HTTP API 接口
```

### Web UI 自动调用

配置保存函数会自动调用热重载：
```python
def save_xxx_config(...):
    set_config(...)  # 保存到文件
    requests.post(f"http://localhost:{port}/reload_config")  # 自动热重载
```

## 开发者注意事项

如果你要添加新的配置项：

1. **静态配置（在初始化时读取）**: 需要在 `reload_config()` 方法中添加重新读取逻辑
2. **动态配置（每次使用时读取）**: 直接调用 `get_config()`，无需特殊处理
3. **Web UI 保存函数**: 确保调用对应服务的 `/reload_config` API

## 测试建议

建议在以下场景测试配置热重载：

1. **开发调试**: 频繁调整参数时使用热重载
2. **性能优化**: 调整 VAD 阈值、音量等参数
3. **模型切换**: ASR/LLM 模型切换
4. **功能开关**: 启用/禁用唤醒词、打断模式等

## 日志监控

热重载操作会在日志中留下记录，方便调试：

```bash
# 查看服务日志
tail -f logs/语音对话.log
tail -f logs/ASR.log
tail -f logs/LLM.log
tail -f logs/TTS.log
tail -f logs/主控制.log
```

日志示例：
```
2025-01-15 10:30:45 - VoiceChat - INFO - 🔄 静音阈值已更新: 200 → 300
2025-01-15 10:30:45 - VoiceChat - INFO - 🔄 输出音量已更新: 50% → 80%
2025-01-15 10:30:45 - VoiceChat - INFO - ✅ VoiceAssistant 配置已重新加载
```

## 性能影响

配置热重载的性能影响：

- **ASR/LLM 模型切换**: 需要重新加载模型，可能需要几秒到几十秒
- **参数调整**: 几乎即时生效，毫秒级
- **音频设备切换**: 需要重启语音对话，几秒钟

总体来说，热重载比完全重启服务快得多！
