# 语音对话自动启动功能说明

## 功能概述

现在，当您使用 `start_all.py` 启动所有服务时，如果在 [config.yaml](config.yaml) 中启用了语音对话功能（`voice_chat.enable: true`），语音对话将会**自动启动**，无需手动在Web界面点击"启动"按钮。

## 工作原理

### 启动流程

```
start_all.py 启动
    ↓
启动语音对话服务 (voice_chat.py)
    ↓
API服务器开始运行 (端口 5004)
    ↓
后台线程自动执行
    ↓
检查 config.yaml 中 voice_chat.enable
    ↓
┌──────────────┬──────────────┐
│  enable: true │ enable: false│
├──────────────┼──────────────┤
│  自动启动     │  仅运行API   │
│  语音对话     │  服务器      │
│  ✅ 开始监听  │  ⏸️ 待命中    │
│  唤醒词      │              │
└──────────────┴──────────────┘
```

### 技术实现

修改位置：[voice_chat.py:1054-1089](voice_chat.py#L1054-L1089)

```python
def auto_start_voice_chat():
    """自动启动语音对话（在后台线程中调用API）"""
    import time
    # 等待API服务器完全启动
    time.sleep(2)

    try:
        voice_config = get_config('voice_chat')
        if voice_config.get('enable', False):
            logger.info("🤖 配置已启用，自动启动语音对话...")
            # 调用内部启动函数
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(start_voice_chat())
            if result.get('success'):
                logger.info("✅ 语音对话已自动启动")
            else:
                logger.warning(f"⚠️ 自动启动失败: {result.get('message')}")
        else:
            logger.info("ℹ️ 语音对话未启用，仅运行API服务器")
    except Exception as e:
        logger.error(f"自动启动语音对话失败: {e}")

def main():
    """主函数 - 启动API服务器"""
    port = get_config('services').get('voice_chat', 5004)
    logger.info(f"🚀 启动语音对话API服务器，端口: {port}")

    # 在后台线程启动自动启动逻辑
    auto_start_thread = threading.Thread(target=auto_start_voice_chat, daemon=True)
    auto_start_thread.start()

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
```

## 使用方法

### 1. 启用自动启动（默认）

确保配置文件中启用了语音对话：

**[config.yaml](config.yaml)**:
```yaml
voice_chat:
  enable: true  # ✅ 启用自动启动
  input_device: 1
  output_device: null
  output_volume: 50
  silence_threshold: 3000
  silence_duration: 1.0
  min_audio_length: 0.5
  wake_mode: true
  wake_words:
    - 小助手
    - 你好助手
    - 嘿助手
    - 小爱
  wake_reply: 你好，我在
```

启动系统：
```bash
python start_all.py
```

启动日志示例：
```
2025-10-25 16:00:00 - Launcher - INFO - 🚀 正在启动 语音对话服务...
2025-10-25 16:00:02 - VoiceChat - INFO - 🚀 启动语音对话API服务器，端口: 5004
2025-10-25 16:00:02 - VoiceChat - INFO - INFO: Uvicorn running on http://0.0.0.0:5004
2025-10-25 16:00:04 - VoiceChat - INFO - 🤖 配置已启用，自动启动语音对话...
2025-10-25 16:00:04 - VoiceChat - INFO - ✅ 语音对话已自动启动
2025-10-25 16:00:04 - VoiceChat - INFO - 🔍 监听唤醒词...
```

### 2. 禁用自动启动

如果您不想自动启动语音对话，只需在配置中禁用：

**[config.yaml](config.yaml)**:
```yaml
voice_chat:
  enable: false  # ❌ 禁用自动启动
```

重新启动系统后，语音对话服务将只运行API服务器，不会自动开始监听。您仍然可以通过Web界面手动启动。

启动日志示例：
```
2025-10-25 16:00:00 - Launcher - INFO - 🚀 正在启动 语音对话服务...
2025-10-25 16:00:02 - VoiceChat - INFO - 🚀 启动语音对话API服务器，端口: 5004
2025-10-25 16:00:02 - VoiceChat - INFO - INFO: Uvicorn running on http://0.0.0.0:5004
2025-10-25 16:00:04 - VoiceChat - INFO - ℹ️ 语音对话未启用，仅运行API服务器
```

### 3. 通过Web界面控制

即使启用了自动启动，您仍然可以在Web界面 (http://localhost:8080) 中：

- **查看状态**：在"🎙️ 语音对话"标签页查看运行状态
- **手动停止**：点击"🛑 停止"按钮
- **手动启动**：点击"▶️ 启动"按钮
- **修改配置**：调整设备、阈值等参数后保存

## 启动时序

```
时间线（秒）   事件
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
0.0         start_all.py 启动
            ↓
3.0         ASR服务启动完成
            ↓
6.0         LLM服务启动完成
            ↓
9.0         TTS服务启动完成
            ↓
12.0        主控制服务启动完成
            ↓
15.0        Web配置界面启动完成
            ↓
18.0        语音对话服务启动
            ├─ API服务器开始运行
            └─ 启动后台自动启动线程
            ↓
20.0        后台线程检查配置
            ↓
            ┌─────────────────┐
            │ enable: true?   │
            └────┬───────┬────┘
                 │       │
            YES  │       │  NO
                 ↓       ↓
21.0        自动启动   仅运行API
            语音对话   (待命)
            ↓
22.0        初始化VoiceAssistant
            - 加载音频设备
            - 设置音量 50%
            - 配置VAD参数
            ↓
23.0        🔍 开始监听唤醒词
            ✅ 系统就绪！
```

## 日志监控

查看语音对话服务的实时日志：

```bash
tail -f logs/语音对话.log
```

关键日志标记：

| 日志消息 | 含义 |
|---------|------|
| `🚀 启动语音对话API服务器` | API服务器开始启动 |
| `🤖 配置已启用，自动启动语音对话...` | 检测到enable=true，准备自动启动 |
| `✅ 语音对话已自动启动` | 自动启动成功 |
| `⚠️ 自动启动失败` | 自动启动失败（查看错误信息） |
| `ℹ️ 语音对话未启用，仅运行API服务器` | enable=false，不自动启动 |
| `🔍 监听唤醒词...` | 语音对话正在运行，等待唤醒 |

## 故障排查

### 问题1：自动启动失败

**症状**：日志显示 `⚠️ 自动启动失败`

**可能原因**：
1. 音频设备未正确配置
2. 配置文件中设备索引无效
3. 麦克风或音箱被占用

**解决方法**：
```bash
# 1. 检查音频设备
python -c "
import pyaudio
p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    print(f'{i}: {info[\"name\"]} (输入:{info[\"maxInputChannels\"]}, 输出:{info[\"maxOutputChannels\"]})')
"

# 2. 更新config.yaml中的设备索引
# voice_chat:
#   input_device: <正确的麦克风索引>
#   output_device: <正确的音箱索引或null>

# 3. 重启服务
pkill -f start_all.py
python start_all.py
```

### 问题2：自动启动但没有声音

**症状**：日志显示启动成功，但没有听到声音

**可能原因**：
1. 蓝牙音箱未连接
2. PulseAudio未设置正确的默认输出
3. 音量设置为0

**解决方法**：
```bash
# 1. 检查蓝牙连接
bluetoothctl devices
bluetoothctl info <MAC地址>

# 2. 检查PulseAudio设备
pactl list sinks short

# 3. 使用Web界面
# - 访问 http://localhost:8080
# - 点击"🔵 检查蓝牙连接"
# - 点击"🔊 设为默认输出"
# - 调整音量滑块到 50-100%
```

### 问题3：服务启动顺序问题

**症状**：语音对话启动失败，提示ASR/LLM/TTS服务不可用

**可能原因**：其他微服务尚未完全启动

**解决方法**：

修改 [start_all.py:242](start_all.py#L242)，增加语音对话服务的等待时间：

```python
{
    'name': '语音对话服务',
    'script': base_dir / 'voice_chat.py',
    'conda_env': None,
    'wait': 5,  # 从3秒增加到5秒
    'optional': True
}
```

或者在 [voice_chat.py:1058](voice_chat.py#L1058) 增加等待时间：

```python
def auto_start_voice_chat():
    import time
    # 从2秒增加到5秒
    time.sleep(5)  # 等待API服务器和其他服务完全启动
```

## 配置参考

完整的语音对话配置项：

```yaml
voice_chat:
  # 是否启用自动启动（true/false）
  enable: true

  # 输入设备索引（麦克风）
  # 使用Web界面的"刷新设备列表"查看可用设备
  # 或设为null使用系统默认设备
  input_device: 1

  # 输出设备索引（音箱/耳机）
  # null = 使用系统默认设备（推荐用于蓝牙）
  output_device: null

  # 输出音量（0-100）
  output_volume: 50

  # 静音阈值（建议范围：1500-3000）
  # 值越高 = 对噪音容忍度越高 = 需要更大声说话
  # 值越低 = 对噪音容忍度越低 = 容易误触发
  silence_threshold: 3000

  # 静音持续时间（秒）
  # 检测到静音后需持续多久才结束录音
  silence_duration: 1.0

  # 最短录音时长（秒）
  # 低于此时长的录音会被忽略
  min_audio_length: 0.5

  # 是否启用唤醒词模式（true/false）
  # true = 需要先说唤醒词才能对话
  # false = 持续监听，无需唤醒词
  wake_mode: true

  # 唤醒词列表
  wake_words:
    - 小助手
    - 你好助手
    - 嘿助手
    - 小爱

  # 唤醒后的快速回复
  wake_reply: 你好，我在
```

## 与之前版本的区别

### 之前（手动启动）

```
1. 运行 start_all.py
2. 等待所有服务启动
3. 打开浏览器访问 http://localhost:8080
4. 切换到"🎙️ 语音对话"标签页
5. 点击"▶️ 启动"按钮
6. 开始使用
```

### 现在（自动启动）

```
1. 运行 start_all.py
2. 等待2-3秒
3. 直接开始使用（说唤醒词即可）
4.（可选）打开Web界面查看状态或调整参数
```

## 优势

1. **开箱即用**：系统启动后立即可用，无需额外操作
2. **无人值守**：适合作为后台服务长期运行
3. **灵活控制**：仍可通过Web界面或配置文件控制
4. **快速响应**：减少从启动到可用的时间
5. **易于调试**：详细日志帮助定位问题

## API端点

即使启用了自动启动，所有API端点仍然可用：

| 端点 | 方法 | 功能 |
|-----|------|------|
| `/devices` | GET | 获取音频设备列表 |
| `/start` | POST | 手动启动语音对话 |
| `/stop` | POST | 停止语音对话 |
| `/status` | GET | 获取运行状态 |

示例：
```bash
# 获取状态
curl http://localhost:5004/status

# 手动停止
curl -X POST http://localhost:5004/stop

# 手动启动
curl -X POST http://localhost:5004/start
```

## 相关文档

- [BLUETOOTH_SETUP_GUIDE.md](BLUETOOTH_SETUP_GUIDE.md) - 蓝牙音箱配置指南
- [VAD_TROUBLESHOOTING.md](VAD_TROUBLESHOOTING.md) - VAD故障排查指南
- [TTS_ASYNC_OPTIMIZATION.md](TTS_ASYNC_OPTIMIZATION.md) - TTS异步优化说明
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - 完整实现总结

---

**最后更新**: 2025-10-25
**适用版本**: AI语音助手 v1.1+
**功能状态**: ✅ 已实现并测试
