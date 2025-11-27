# 🤖 AI语音助手 (AI Voice Assistant)

一个基于微服务架构的智能语音助手系统，支持完整的ASR → LLM → TTS流水线，具备流式处理和实时语音交互能力。

## 📋 目录 (Table of Contents)

- [🌟 项目概述](#-项目概述)
- [🏗️ 系统架构](#️-系统架构)
- [📁 目录结构](#-目录结构)
- [🚀 快速开始](#-快速开始)
- [⚙️ 配置指南](#️-配置指南)
- [🔧 核心组件](#-核心组件)
- [📡 API文档](#-api文档)
- [🧠 开发指南](#-开发指南)
- [🚀 部署指南](#-部署指南)
- [🔍 故障排除](#-故障排除)

## 🌟 项目概述

AI语音助手是一个功能完整的语音交互系统，提供以下核心功能：

- ✨ **完整的语音流水线**：ASR → LLM → TTS全链路支持
- 🎯 **实时语音交互**：低延迟的句子级流式响应
- 🧠 **长期记忆**：基于向量的语义记忆存储
- 🎭 **声音克隆**：通过CosyVoice实现个性化语音合成
- 🌐 **Web管理界面**：基于Gradio的配置和监控系统
- 🔄 **多语言支持**：中文/英文语音识别和合成
- 🎤 **智能VAD**：语音活动检测，支持唤醒词和打断
- 📹 **视觉识别**：YOLO物体检测集成

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    AI Voice Assistant                   │
├─────────────────────────────────────────────────────────┤
│  用户输入 (音频/文本)                                    │
│           ↓                                             │
│  ┌─────────────┐  ┌─────────────┐                      │
│  │   Web UI    │  │ Voice Chat  │ ← 唤醒词检测          │
│  │   (8080)    │  │   (5004)    │                      │
│  └─────────────┘  └─────────────┘                      │
│           ↓                                             │
│  ┌─────────────────────────────────────────────────┐   │
│  │           Orchestrator (5000)                   │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐         │   │
│  │  │   ASR   │  │   LLM   │  │   TTS   │         │   │
│  │  │ (5001)  │  │ (5002)  │  │ (5003)  │         │   │
│  │  └─────────┘  └─────────┘  └─────────┘         │   │
│  └─────────────────────────────────────────────────┘   │
│           ↓                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   Memory    │  │    YOLO     │  │  Audio Out   │    │
│  │  (5006)     │  │  (5005)     │  │    Playback  │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────┘
```

### 数据流程

1. **语音输入流程**：
   ```
   麦克风 → Voice Chat VAD → ASR Service → 文本
   ```

2. **LLM处理流程**：
   ```
   文本 → SentenceSplitter → LLM Service → 流式句子
   ```

3. **音频输出流程**：
   ```
   句子 → TTS Service → AudioPlayer → 扬声器
   ```

## 📁 目录结构

```
ai_助手/
├── 🎯 核心服务
│   ├── orchestrator.py (518行)        # 主流水线协调器
│   ├── voice_chat.py (2,292行)        # 语音交互系统
│   ├── web_ui.py (2,693行)            # Gradio管理界面
│   ├── start_all.py (343行)           # 统一服务启动器
│   └── config_loader.py (141行)       # 配置管理中心
│
├── 🎤 ASR服务 (asr_service/)
│   └── app_fastapi.py (201行)         # WeNet语音识别
│
├── 🧠 LLM服务 (llm_service/)
│   ├── app_fastapi.py (512行)         # 双模式LLM服务
│   ├── deepseek_api.py (87行)         # DeepSeek API集成
│   ├── Qwen1.5-0.5b.py (104行)        # 本地Qwen模型
│   └── tinyllama.py (104行)           # 本地TinyLlama模型
│
├── 🔊 TTS服务 (tts_service/)
│   ├── app_fastapi.py (610行)         # CosyVoice语音合成
│   └── cosyvoice_api.py               # 声音克隆API
│
├── 💾 记忆服务 (memory_service/)
│   ├── app_fastapi.py                 # 向量存储服务
│   └── memory_client.py (198行)       # 记忆操作客户端
│
├── 🎯 YOLO服务 (yolo_service/)
│   └── YOLOV5USBCamera/web_cpp/cpp_bridge_app.py  # 物体检测
│
├── ⚙️ 配置与工具
│   ├── main_controller.py (336行)     # 服务生命周期管理
│   ├── diagnose_vad.py (168行)       # VAD校准工具
│   └── check_yolo_status.py (117行)   # 系统状态监控
│
├── 📄 配置文件
│   └── config.yaml                   # 系统配置文件
│
├── 📋 日志目录
│   └── logs/                         # 各服务日志文件
│
└── 📚 文档
    ├── CLAUDE.md                     # 项目指导文档
    ├── DESIGN.md                     # 设计文档
    └── [其他技术文档...]
```

### 端口分配表

| 服务 | 端口 | 功能 | 状态 |
|------|------|------|------|
| Web UI | 8080 | Gradio管理界面 | ✅ 活跃 |
| Orchestrator | 5000 | 主控制器 | ✅ 活跃 |
| ASR | 5001 | 语音识别 | ✅ 活跃 |
| LLM | 5002 | 语言模型 | ✅ 活跃 |
| TTS | 5003 | 语音合成 | ✅ 活跃 |
| Voice Chat | 5004 | 语音交互 | ✅ 活跃 |
| YOLO | 5005 | 物体检测 | ✅ 活跃 |
| Memory | 5006 | 长期记忆 | ✅ 活跃 |

## 🚀 快速开始

### 前置要求

- Python 3.8+
- Conda环境管理器
- Linux系统（推荐Ubuntu 20.04+）
- 音频设备（麦克风和扬声器）

### 安装步骤

1. **克隆项目**
   ```bash
   git clone <repository_url>
   cd ai_助手
   ```

2. **启动所有服务**
   ```bash
   python start_all.py
   ```
   > 该脚本会自动检测并启动所有必要的conda环境

3. **访问Web界面**
   ```
   http://localhost:8080
   ```

4. **启动语音对话**
   ```bash
   python voice_chat.py
   ```

### 测试功能

- 通过Web UI进行文本对话测试
- 使用"助手"、"你好"等唤醒词激活语音交互
- 检查各服务状态页面确保正常运行

## ⚙️ 配置指南

### config.yaml 结构

配置文件采用YAML格式，主要包含以下部分：

#### ASR配置
```yaml
asr:
  model_type: CN                    # CN/EN模型选择
  sample_rate: 16000               # 采样率
  model_path_cn: asr_service/CN_model/offline_encoder.om
  model_path_en: asr_service/EN_model/offline_encoder.om
```

#### LLM配置
```yaml
llm:
  mode: api                         # api/local模式
  api:
    provider: deepseek              # API提供商
    api_key: your_api_key          # API密钥
    model: deepseek-chat           # 模型名称
    max_tokens: 512                # 最大token数
    temperature: 1.0               # 温度参数
```

#### TTS配置
```yaml
tts:
  mode: api                         # api/local模式
  api:
    provider: cosyvoice             # CosyVoice API
    model: cosyvoice-v3            # 模型版本
    voice: voice_id                 # 声音ID
    sample_rate: 22050             # 采样率
```

#### 语音对话配置
```yaml
voice_chat:
  enable: true                      # 启用语音对话
  wake_mode: true                   # 启用唤醒词检测
  wake_words: ["助手", "你好"]       # 唤醒词列表
  interrupt_mode: true              # 启用打断功能
  interrupt_words: ["停止", "暂停"]   # 打断词列表
  silence_threshold: 200           # VAD阈值
  silence_duration: 0.5            # 静音时长
  output_volume: 50                 # 输出音量(0-100)
```

### 环境要求

| 服务 | Conda环境 | 主要依赖 |
|------|------------|----------|
| ASR | asr | wenet, torch |
| TTS | tts/cosyvoice | dashscope, cosyvoice |
| LLM(本地) | mindspore | mindspore, mindnlp |
| 其他 | base | fastapi, gradio, pyaudio |

## 🔧 核心组件

### 🎯 Orchestrator (orchestrator.py)

**功能**：系统的核心协调器，负责整个AI流水线

**关键特性**：
- **SentenceSplitter**：智能句子边界检测，实现流式响应
- **AudioPlayer**：实时PCM音频播放管理
- **WebSocket支持**：双向实时通信
- **健康监控**：服务依赖检查

**核心流程**：
1. 接收语音/文本输入
2. 调用ASR服务转录音频
3. 使用SentenceSplitter流式处理LLM响应
4. 实时发送完整句子到TTS
5. 播放合成的音频

### 🎤 Voice Chat (voice_chat.py)

**功能**：离线语音交互系统，提供完整的对话体验

**关键特性**：
- **VAD实现**：基于RMS的语音活动检测
- **唤醒词模式**：连续实时监控
- **打断处理**：AI播放时的实时打断检测
- **音频缓存**：MD5哈希的音频缓存系统

**主要类**：
- `VoiceAssistant`：核心语音交互管理
- `AudioPlaybackQueue`：异步TTS播放队列
- `VAD`：语音活动检测器

**API端点**：
- `/start` - 启动语音对话
- `/stop` - 停止服务
- `/status` - 状态查询
- `/devices` - 设备列表
- `/volume/*` - 音量监控和校准

### 🌐 Web UI (web_ui.py)

**功能**：基于Gradio的综合管理界面

**主要模块**：
- 💬 AI聊天界面（文本/语音输入，流式响应）
- 📊 服务状态监控
- 🎤 ASR配置（模型选择、测试）
- 🧠 LLM配置（API/本地模式切换）
- 🔊 TTS配置（声音克隆、测试）
- 🎙️ 语音对话配置（VAD、唤醒词、音频设备）
- 🧠 记忆管理（增删查改）
- 🎨 声音克隆（CosyVoice API集成）

### 🎤 ASR服务

**功能**：基于WeNet的语音识别服务

**特性**：
- 支持中英文模型切换
- 热重载模型能力
- 流式音频处理
- 健康监控

**API端点**：
- `/transcribe` - 音频转录
- `/health` - 健康检查
- `/reload` - 重载模型

### 🧠 LLM服务

**功能**：双模式语言模型服务

**API模式**：
- DeepSeek集成
- SSE流式响应
- 可配置参数

**本地模式**：
- Qwen1.5-0.5B模型
- TinyLlama-1.1B模型
- MindSpore加速

**API端点**：
- `/chat/stream` - 流式对话
- `/chat` - 非流式对话
- `/model/switch` - 模型切换

### 🔊 TTS服务

**功能**：基于CosyVoice的文本转语音服务

**特性**：
- **语音克隆**：完整的CosyVoice API集成
- **流式合成**：实时音频流
- **连接池**：性能优化
- **多格式输出**：PCM 22050Hz单声道16位

**语音克隆API**：
- `/voice/create` - 创建定制声音
- `/voice/query` - 查询声音状态
- `/voice/list` - 列出所有声音
- `/voice/update` - 更新声音
- `/voice/delete` - 删除声音

### 💾 记忆服务

**功能**：基于向量的语义记忆存储

**特性**：
- 固定输入长度（NPU优化）
- 多种记忆类型（通用、偏好、个人、事件）
- 重要性评分
- 自动提取
- 相似度检索

## 📡 API文档

### 主要服务端点

#### Orchestrator API
```http
POST /conversation/voice     # 完整语音流水线
POST /conversation/text      # 跳过ASR的文本流水线
GET  /health                 # 服务健康检查
WS   /ws/conversation        # WebSocket实时通信
```

#### ASR API
```http
POST /transcribe
Content-Type: multipart/form-data
{
  "audio": <audio_file>
}
```

#### LLM API
```http
POST /chat/stream
Content-Type: application/json
{
  "message": "你好",
  "history": [],
  "stream": true
}
```

#### TTS API
```http
POST /synthesize/stream
Content-Type: application/json
{
  "text": "你好世界",
  "voice": "voice_id"
}
```

### WebSocket API

实时双向通信协议：
```javascript
// 连接
ws = new WebSocket('ws://localhost:5000/ws/conversation');

// 发送消息
ws.send(JSON.stringify({
  type: 'audio',
  data: audio_blob
}));

// 接收流式响应
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'audio') {
    playAudio(data.data);
  }
};
```

## 🧠 开发指南

### 代码组织原则

1. **微服务架构**：每个服务独立部署和扩展
2. **配置驱动**：所有配置通过config.yaml管理
3. **流式优先**：低延迟的句子级流式处理
4. **错误处理**：完善的错误处理和恢复机制
5. **日志记录**：结构化日志便于调试

### 添加新服务

1. 在服务目录创建`app_fastapi.py`
2. 实现标准的健康检查端点
3. 添加配置到`config.yaml`
4. 更新`start_all.py`启动逻辑
5. 在`orchestrator.py`中集成

### 测试流程

```bash
# 测试ASR
curl -X POST http://localhost:5001/transcribe \
  -F "audio=@test.wav"

# 测试LLM
curl -X POST http://localhost:5002/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "history": []}'

# 测试TTS
curl -X POST http://localhost:5003/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}' \
  --output test.pcm
```

### 调试技巧

- 使用`/logs/`目录查看服务日志
- 通过Web UI的监控页面检查服务状态
- 使用`diagnose_vad.py`校准VAD参数
- 检查`config.yaml`配置是否正确

## 🚀 部署指南

### 环境设置

1. **安装Conda**
   ```bash
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh
   ```

2. **创建环境**
   ```bash
   conda create -n asr python=3.8
   conda create -n tts python=3.8
   conda create -n mindspore python=3.9
   ```

3. **激活环境并安装依赖**
   ```bash
   conda activate asr
   pip install -r requirements/asr.txt
   ```

### 服务依赖

| 服务 | 依赖服务 | 说明 |
|------|----------|------|
| Orchestrator | ASR, LLM, TTS | 协调所有AI服务 |
| Voice Chat | Orchestrator | 语音交互核心 |
| Web UI | 所有服务 | 管理界面 |
| Memory | 无 | 独立记忆存储 |

### 生产配置

- 使用`systemd`管理服务进程
- 配置Nginx反向代理
- 设置SSL证书
- 配置日志轮转
- 监控服务健康状态

## 🔍 故障排除

### 常见问题

#### 1. 服务启动失败
**症状**：服务无法启动或立即退出
**解决方案**：
- 检查conda环境是否正确激活
- 确认端口未被占用
- 查看日志文件`logs/SERVICE.log`

#### 2. 语音识别不准确
**症状**：ASR转录结果错误率高
**解决方案**：
- 检查音频格式（必须16kHz WAV）
- 调整`silence_threshold`参数
- 使用`diagnose_vad.py`校准VAD

#### 3. TTS合成延迟高
**症状**：语音合成响应慢
**解决方案**：
- 检查网络连接（API模式）
- 使用音频缓存功能
- 调整`max_wait_time`参数

#### 4. 唤醒词不响应
**症状**：唤醒词检测失败
**解决方案**：
- 调低`silence_threshold`
- 增加唤醒词列表
- 检查麦克风设备

### 日志位置

- 主日志：`logs/ai_assistant.log`
- ASR日志：`logs/ASR.log`
- LLM日志：`logs/LLM.log`
- TTS日志：`logs/TTS.log`
- Web日志：`logs/Web配置界面.log`
- 语音对话：`logs/语音对话.log`

### 性能优化

1. **VAD优化**：
   - 使用`/volume/start`监控环境噪音
   - 设置合适的`silence_threshold`
   - 调整`silence_duration`

2. **流式优化**：
   - 调整`sentence_delimiters`
   - 设置`min_chunk_length`
   - 优化`max_wait_time`

3. **缓存策略**：
   - 启用音频缓存
   - 使用MD5哈希避免重复合成
   - 定期清理过期缓存

### FAQ

**Q: 如何更换TTS声音？**
A: 在Web UI的TTS配置页面，可以：
- 选择预置声音
- 上传声音样本进行克隆
- 查看声音状态和进度

**Q: 如何添加新的唤醒词？**
A: 修改`config.yaml`中的`voice_chat.wake_words`列表

**Q: 支持哪些音频格式？**
A: 系统自动处理音频格式转换：
- 输入：任何格式（自动转换为16kHz WAV）
- 输出：PCM 22050Hz单声道16位

## 📄 许可证

本项目采用MIT许可证 - 详见[LICENSE](LICENSE)文件

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至：[your-email@example.com]