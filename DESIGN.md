# AI语音助手系统设计文档

## 1. 项目概述

### 1.1 项目简介

AI语音助手是一个基于微服务架构的智能语音交互系统，支持完整的语音识别（ASR）→ 大语言模型（LLM）→ 语音合成（TTS）处理链路，并提供流式处理能力以实现低延迟的实时交互体验。

### 1.2 核心特性

- **微服务架构**：各功能模块独立部署，松耦合设计
- **流式处理**：基于句子级别的智能流式传输，显著降低首字延迟
- **唤醒词检测**：支持自定义唤醒词，实现免按键交互
- **打断机制**：用户可在AI回复过程中打断对话
- **语音克隆**：支持通过CosyVoice API进行声音克隆
- **双模式LLM**：支持API模式(DeepSeek)和本地模型(Qwen/TinyLlama)
- **可视化配置**：基于Gradio的Web配置界面
- **音频缓存**：智能缓存常用语音，提升响应速度

### 1.3 技术栈

| 类别 | 技术选型 |
|------|---------|
| Web框架 | FastAPI + Uvicorn |
| Web界面 | Gradio 4.8.0 |
| ASR引擎 | WeNet (CN/EN模型) |
| LLM | DeepSeek API / Qwen-1.5 / TinyLlama |
| TTS引擎 | CosyVoice API |
| 音频处理 | PyAudio + SoundFile + NumPy |
| 配置管理 | PyYAML |
| 进程管理 | Subprocess + psutil |

---

## 2. 系统架构

### 2.1 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                         用户交互层                                │
├──────────────────────┬──────────────────────────────────────────┤
│  Web UI (Gradio)     │  Voice Chat (Offline)                    │
│  Port: 8080          │  Port: 5004                              │
│  - 配置管理           │  - VAD语音活动检测                        │
│  - 服务监控           │  - 唤醒词检测                            │
│  - 在线测试           │  - 打断机制                              │
└──────────────────────┴──────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│               Orchestrator (主控制服务)                          │
│               Port: 5000                                         │
│  - 服务编排与流程控制                                             │
│  - SentenceSplitter (句子分割器)                                 │
│  - AudioPlayer (音频播放器)                                       │
│  - WebSocket实时通信                                             │
└─────────────────────────────────────────────────────────────────┘
                ↓              ↓              ↓
    ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
    │  ASR Service  │  │  LLM Service  │  │  TTS Service  │
    │  Port: 5001   │  │  Port: 5002   │  │  Port: 5003   │
    ├───────────────┤  ├───────────────┤  ├───────────────┤
    │ • WeNet CN/EN │  │ • DeepSeek API│  │ • CosyVoice   │
    │ • 16kHz WAV   │  │ • Qwen-1.5    │  │ • 声音克隆     │
    │ • 实时识别     │  │ • TinyLlama   │  │ • PCM输出     │
    └───────────────┘  │ • SSE流式输出  │  │ • 流式合成     │
                       └───────────────┘  └───────────────┘
```

### 2.2 服务端口分配

| 服务 | 端口 | 描述 |
|------|------|------|
| Orchestrator | 5000 | 主控制服务，编排ASR→LLM→TTS流程 |
| ASR Service | 5001 | 语音识别服务 |
| LLM Service | 5002 | 大语言模型服务 |
| TTS Service | 5003 | 语音合成服务 |
| Voice Chat | 5004 | 离线语音对话服务 |
| Web UI | 8080 | Web配置界面 |

### 2.3 数据流向

#### 2.3.1 语音对话完整流程

```
用户说话 → 麦克风录音
           ↓
    [VAD检测语音活动]
           ↓
    保存为16kHz WAV文件
           ↓
    POST /transcribe (ASR Service)
           ↓
    识别文本 → Orchestrator
           ↓
    POST /chat/stream (LLM Service)
           ↓
    SSE流式输出 → SentenceSplitter
           ↓
    完整句子 → POST /synthesize/stream (TTS Service)
           ↓
    PCM音频流 → AudioPlaybackQueue
           ↓
    PyAudio播放 → 扬声器输出
```

#### 2.3.2 文本对话流程

```
用户文本输入
    ↓
POST /conversation/text (Orchestrator)
    ↓
POST /chat/stream (LLM Service)
    ↓
SSE流式输出 → SentenceSplitter
    ↓
完整句子 → POST /synthesize/stream (TTS Service)
    ↓
PCM音频流 → 返回/播放
```

---

## 3. 核心模块设计

### 3.1 Orchestrator (主控制服务)

**文件**: [orchestrator.py](orchestrator.py)

**职责**:
- 串联ASR、LLM、TTS服务的完整对话流程
- 实现句子级别的流式处理
- 管理对话历史
- 提供WebSocket实时通信接口

**关键类**:

#### 3.1.1 SentenceSplitter (句子分割器)

```python
class SentenceSplitter:
    """
    功能: 将LLM流式输出按句子边界智能切分
    配置参数:
      - sentence_delimiters: 句子分隔符列表 ['。', '!', '?', '\n', '.', ';']
      - min_chunk_length: 最小句子长度 (默认10字符)
      - max_wait_time: 最大等待时间 (默认3.0秒)

    工作流程:
      1. 接收LLM输出的文本块 (add_chunk)
      2. 在缓冲区中查找句子分隔符
      3. 提取完整句子发送给TTS
      4. 保留不完整部分继续缓冲
    """
```

**算法优势**:
- 避免在词语中间切分，保证语义完整
- 支持中英文标点符号
- 可配置最小句子长度，避免过短的片段

#### 3.1.2 AudioPlayer (音频播放器)

```python
class AudioPlayer:
    """
    功能: 管理PyAudio音频播放
    配置参数:
      - sample_rate: 采样率 (默认22050Hz)
      - chunk_size: 缓冲区大小
      - output_device: 输出设备索引

    方法:
      - play_audio(audio_data): 播放PCM音频数据
      - set_volume(volume): 设置音量 (0-100)
    """
```

**API端点**:

| 端点 | 方法 | 描述 |
|------|------|------|
| `/conversation/voice` | POST | 完整语音对话 (含ASR) |
| `/conversation/text` | POST | 文本对话 (跳过ASR) |
| `/ws/conversation` | WebSocket | 实时双向通信 |
| `/health` | GET | 服务健康检查 |

### 3.2 ASR Service (语音识别服务)

**文件**: [asr_service/app_fastapi.py](asr_service/app_fastapi.py)

**职责**:
- 将音频文件转换为文本
- 支持中文(CN)和英文(EN)模型
- 自动音频预处理（重采样到16kHz）

**技术实现**:
- **引擎**: WeNet (开源语音识别框架)
- **模型**:
  - CN: `asr_service/CN_model/offline_encoder.om`
  - EN: `asr_service/EN_model/offline_encoder.om`
- **输入格式**: 16kHz, 单声道WAV文件
- **输出格式**: JSON `{"text": "识别结果"}`

**API端点**:

| 端点 | 方法 | 描述 |
|------|------|------|
| `/transcribe` | POST | 上传音频文件进行识别 |
| `/health` | GET | 健康检查 |

**配置项** (config.yaml):
```yaml
asr:
  model_type: CN              # 模型类型: CN/EN
  sample_rate: 44100          # 原始采样率
  channels: 1                 # 声道数
  model_path_cn: ...          # 中文模型路径
  model_path_en: ...          # 英文模型路径
  vocab_path_cn: ...          # 中文词表路径
  vocab_path_en: ...          # 英文词表路径
```

### 3.3 LLM Service (大语言模型服务)

**文件**: [llm_service/app_fastapi.py](llm_service/app_fastapi.py)

**职责**:
- 生成智能对话回复
- 支持流式和非流式输出
- 管理对话上下文

**运行模式**:

#### 3.3.1 API模式 (推荐)

- **提供商**: DeepSeek
- **模型**: deepseek-chat
- **优势**: 响应快速、质量高、免部署
- **流式输出**: Server-Sent Events (SSE)

#### 3.3.2 本地模式

- **支持模型**:
  - Qwen-1.5-0.5B-Chat
  - TinyLlama-1.1B-Chat-v1.0
- **框架**: MindNLP + MindSpore
- **优势**: 完全离线、数据私密

**对话历史格式**:
```python
history = [
    ["用户问题1", "AI回答1"],
    ["用户问题2", "AI回答2"]
]
```

**API端点**:

| 端点 | 方法 | 描述 |
|------|------|------|
| `/chat` | POST | 非流式对话 |
| `/chat/stream` | POST | 流式对话 (SSE) |
| `/health` | GET | 健康检查 |

**配置项** (config.yaml):
```yaml
llm:
  mode: api                   # 模式: api/local
  api:
    provider: deepseek
    api_url: https://api.deepseek.com/v1/chat/completions
    api_key: sk-xxx
    model: deepseek-chat
    temperature: 1.0
    top_p: 0.9
    max_tokens: 512
    system_prompt: "你是一个智能语音助手..."
  local:
    model_name: qwen          # qwen/tinyllama
    temperature: 1.0
    max_tokens: 128
```

### 3.4 TTS Service (语音合成服务)

**文件**: [tts_service/app_fastapi.py](tts_service/app_fastapi.py)

**职责**:
- 将文本转换为语音
- 支持声音克隆
- 流式音频生成

**技术实现**:

- **提供商**: 阿里云DashScope
- **模型**: cosyvoice-v3
- **输出格式**: PCM_22050HZ_MONO_16BIT
- **特色功能**:
  - 零样本声音克隆
  - 多种预训练音色
  - 流式合成

**声音克隆流程**:
```
上传音频样本 → POST /enroll
    ↓
返回task_id
    ↓
轮询状态 → GET /enroll/status/{task_id}
    ↓
获得voice_id → 用于合成
```

**API端点**:

| 端点 | 方法 | 描述 |
|------|------|------|
| `/synthesize` | POST | 非流式合成 |
| `/synthesize/stream` | POST | 流式合成 |
| `/enroll` | POST | 声音注册/克隆 |
| `/enroll/status/{task_id}` | GET | 查询注册状态 |
| `/voices` | GET | 获取可用音色列表 |
| `/health` | GET | 健康检查 |

**配置项** (config.yaml):
```yaml
tts:
  mode: api                   # 固定为API模式
  api:
    provider: cosyvoice
    api_key: sk-xxx
    model: cosyvoice-v3
    voice: cosyvoice-v3-elysia-xxx  # 音色ID
    format: PCM_22050HZ_MONO_16BIT
    sample_rate: 22050
  voice_enrollment:
    default_model: cosyvoice-v2
    default_prefix: myvoice
    poll_interval: 10         # 轮询间隔(秒)
    max_poll_attempts: 30     # 最大轮询次数
```

### 3.5 Voice Chat (离线语音对话)

**文件**: [voice_chat.py](voice_chat.py)

**职责**:
- 实现完全离线的语音交互循环
- VAD语音活动检测
- 唤醒词检测
- 打断机制
- 音频播放队列管理

**核心类**:

#### 3.5.1 VoiceAssistant

```python
class VoiceAssistant:
    """
    主控制类，管理完整的语音交互循环

    核心方法:
      - record_audio(): 录制音频
      - monitor_wake_word(): 监听唤醒词
      - monitor_interrupt(): 监听打断词
      - conversation_loop(): 对话循环
    """
```

**工作模式**:

1. **唤醒词模式** (wake_mode=true)
   ```
   持续监听 → VAD检测到语音 → ASR识别 → 检查唤醒词
       ↓ (检测到唤醒词)
   播放唤醒回复 → 进入对话模式
   ```

2. **普通对话模式**
   ```
   录音 → ASR识别 → LLM生成回复 → TTS合成 → 播放
      ↑_____________________________________________↓
                    (循环直到超时)
   ```

3. **打断模式** (interrupt_mode=true)
   ```
   AI播放回复时 → 实时监听 → 检测到打断词 → 停止播放
   ```

#### 3.5.2 AudioPlaybackQueue

```python
class AudioPlaybackQueue:
    """
    音频播放队列管理器
    实现生产者-消费者模式

    工作流程:
      1. TTS生成音频片段 → 入队
      2. 后台播放线程 → 出队播放
      3. 支持中途打断 → 清空队列
    """
```

**优势**:
- **异步处理**: TTS生成和音频播放并行
- **低延迟**: 首个音频片段立即播放
- **可打断**: 清空队列即可停止播放

#### 3.5.3 VAD (Voice Activity Detection)

```python
def is_speech(audio_chunk):
    """
    基于RMS能量的语音活动检测

    算法:
      1. 计算音频块的RMS值
      2. 与阈值(silence_threshold)比较
      3. 返回是否为语音

    参数调优:
      - silence_threshold: 200 (可通过/volume API校准)
      - silence_duration: 0.5秒 (静音多久判定结束)
    """
```

**音频缓存机制**:

为了提升响应速度，常用语音(唤醒回复、打断回复)会被缓存:

```python
audio_cache/
  ├── 45a3c8f1234567890abcdef.pcm  # 唤醒回复缓存
  └── 89b7d2e9876543210fedcba.pcm  # 打断回复缓存
```

- **缓存键**: MD5(文本内容)
- **失效策略**: 配置文件中文本变化时自动重新生成

**API端点**:

| 端点 | 方法 | 描述 |
|------|------|------|
| `/start` | POST | 启动语音对话 |
| `/stop` | POST | 停止语音对话 |
| `/status` | GET | 获取运行状态 |
| `/devices` | GET | 列出音频设备 |
| `/volume/start` | POST | 开始音量监测 |
| `/volume/data` | GET | 获取音量数据(SSE) |

**配置项** (config.yaml):
```yaml
voice_chat:
  enable: true                # 启动时自动开始对话
  wake_mode: true             # 唤醒词模式
  wake_words:                 # 唤醒词列表
    - "助手"
    - "你好"
  wake_reply: "哎呦，谁在叫我呀？"
  interrupt_mode: true        # 打断模式
  interrupt_words:            # 打断词列表
    - "停止"
    - "闭嘴"
  interrupt_reply: "好吧好吧，我不说了还不行吗~"
  thinking_reply: "好，我知道了，等我想一下"
  silence_threshold: 200      # VAD阈值
  silence_duration: 0.5       # 静音时长(秒)
  min_audio_length: 0.7       # 最小有效音频长度(秒)
  continue_dialogue_timeout: 10.0  # 对话超时(秒)
  input_device: 1             # 输入设备索引
  output_device: null         # 输出设备索引(null=默认)
  output_volume: 50           # 输出音量(0-100)
```

### 3.6 Web UI (Web配置界面)

**文件**: [web_ui.py](web_ui.py)

**职责**:
- 可视化配置管理
- 服务状态监控
- 在线功能测试

**功能模块**:

1. **服务状态监控**
   - 实时显示各服务运行状态
   - 健康检查
   - 启动/停止服务

2. **配置管理**
   - LLM配置 (API Key, 模型参数, System Prompt)
   - TTS配置 (音色选择, 声音克隆)
   - 语音对话配置 (唤醒词, 打断词, VAD参数)
   - 流式处理配置 (句子分隔符, 缓冲策略)

3. **功能测试**
   - ASR测试: 上传音频测试识别
   - LLM测试: 文本对话测试
   - TTS测试: 文本转语音测试
   - 完整流程测试: 端到端语音对话

4. **音频设备管理**
   - 列出可用输入/输出设备
   - 音量监测与校准

**技术实现**:
- **框架**: Gradio 4.8.0
- **特性**:
  - 自动刷新
  - 实时日志显示
  - 音频播放
  - Markdown渲染

**配置项** (config.yaml):
```yaml
web:
  enable: true                # 启动时自动打开Web UI
  share: true                 # 是否生成公网链接
  title: "AI语音助手配置中心"
```

### 3.7 Main Controller (服务管理器)

**文件**: [main_controller.py](main_controller.py)

**职责**:
- 统一管理所有微服务的生命周期
- 服务启动/停止/重启
- 进程监控
- PID管理

**核心类**:

```python
class ServiceManager:
    """
    服务管理器

    功能:
      - start_service(service_key): 启动指定服务
      - stop_service(service_key): 停止指定服务
      - restart_service(service_key): 重启服务
      - check_health(service_key): 健康检查
      - start_all(): 启动所有服务
      - stop_all(): 停止所有服务
    """
```

**服务定义**:
```python
services = {
    'asr': {
        'name': '语音识别服务',
        'path': 'asr_service/app_fastapi.py',
        'env': 'asr',           # conda环境名
        'port': 5001,
        'health_endpoint': '/health'
    },
    # ... 其他服务
}
```

**PID管理**:
- PID文件存储在 `pids/` 目录
- 格式: `{service_key}.pid`
- 用于服务重启和状态检查

### 3.8 Config Loader (配置管理)

**文件**: [config_loader.py](config_loader.py)

**职责**:
- 统一配置文件管理
- 支持嵌套键访问
- 配置热更新

**核心API**:

```python
# 获取配置
api_key = get_config('llm.api.api_key')
port = get_config('services.asr', default=5001)

# 设置配置
set_config('llm.mode', 'api')
set_config('voice_chat.wake_mode', True)

# 重新加载配置
reload_config()
```

**配置文件结构** (config.yaml):
```yaml
# 服务端口配置
services:
  asr: 5001
  llm: 5002
  tts: 5003
  orchestrator: 5000
  voice_chat: 5004
  web_ui: 8080

# ASR配置
asr:
  model_type: CN
  sample_rate: 44100
  # ...

# LLM配置
llm:
  mode: api
  api: { ... }
  local: { ... }

# TTS配置
tts:
  mode: api  # 固定为API模式
  api: { ... }

# 流式处理配置
streaming:
  sentence_delimiters: ["。", "!", "?", "\n", ".", ";"]
  min_chunk_length: 10
  max_wait_time: 3.0

# 语音对话配置
voice_chat:
  enable: true
  wake_mode: true
  # ...

# Web界面配置
web:
  enable: true
  share: true
  title: "AI语音助手配置中心"

# 日志配置
logging:
  level: INFO
  file: ai_assistant.log
  max_bytes: 10485760
  backup_count: 5
```

---

## 4. 流式处理设计

### 4.1 为什么需要流式处理？

传统的语音对话系统:
```
用户说话 → ASR → 等待LLM生成完整回复 → 等待TTS合成完整音频 → 播放
总延迟 = ASR时间 + LLM全部生成时间 + TTS全部合成时间
```

流式处理系统:
```
用户说话 → ASR → LLM流式生成 → 句子1 → TTS合成 → 立即播放
                           ↓
                      句子2 → TTS合成 → 继续播放
                           ↓
                      句子3 → TTS合成 → 继续播放
首字延迟 = ASR时间 + LLM第一句时间 + TTS第一句时间
```

**延迟对比** (示例):
- 传统模式: 3s (ASR) + 8s (LLM完整) + 5s (TTS完整) = **16秒**
- 流式模式: 3s (ASR) + 2s (LLM首句) + 1s (TTS首句) = **6秒**

### 4.2 句子分割算法

**核心挑战**: 如何确定"完整句子"？

**解决方案**: SentenceSplitter智能缓冲算法

```python
def process_llm_stream():
    splitter = SentenceSplitter()

    # 接收LLM流式输出
    for chunk in llm_stream:
        splitter.add_chunk(chunk)

        # 检查是否有完整句子
        sentences = splitter.get_complete_sentences()

        for sentence in sentences:
            # 立即发送给TTS合成
            tts_audio = synthesize(sentence)
            play_audio(tts_audio)

    # 处理最后的不完整部分
    final_sentence = splitter.flush()
    if final_sentence:
        tts_audio = synthesize(final_sentence)
        play_audio(tts_audio)
```

**分隔符配置**:
```yaml
streaming:
  sentence_delimiters:
    - "。"    # 中文句号
    - "!"     # 中文感叹号
    - "?"     # 中文问号
    - "!"     # 英文感叹号
    - "?"     # 英文问号
    - "\n"    # 换行符
    - "."     # 英文句号
    - ";"     # 分号
```

**优化策略**:
1. **最小句子长度**: 避免过短的片段（如"好。"）
2. **最大等待时间**: 避免无限等待完整句子
3. **智能合并**: 短句可以合并后再发送

### 4.3 Server-Sent Events (SSE)

LLM流式输出采用SSE协议:

**格式**:
```
data: {"delta": "今"}

data: {"delta": "天"}

data: {"delta": "天气"}

data: {"delta": "很好"}

data: {"delta": "。"}

data: {"done": true}
```

**Python实现** (LLM Service):
```python
async def stream_response():
    for chunk in llm_generate_stream():
        yield f"data: {json.dumps({'delta': chunk})}\n\n"
    yield f"data: {json.dumps({'done': true})}\n\n"

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream"
    )
```

**Python消费** (Orchestrator):
```python
response = requests.post(
    f"{llm_url}/chat/stream",
    json={"message": text, "history": history},
    stream=True
)

for line in response.iter_lines():
    if line.startswith(b'data: '):
        data = json.loads(line[6:])
        if 'delta' in data:
            # 处理增量文本
            process_delta(data['delta'])
        elif data.get('done'):
            # 流式结束
            break
```

---

## 5. 部署架构

### 5.1 Conda环境隔离

系统采用多conda环境隔离不同服务的依赖:

```
base (默认环境)
  ├── orchestrator.py
  ├── voice_chat.py
  ├── web_ui.py
  ├── main_controller.py
  └── start_all.py

asr (ASR专用环境)
  └── asr_service/
      ├── WeNet依赖
      └── sounddevice

tts (TTS专用环境)
  └── tts_service/
      └── FastAPI + Requests (API调用)

llm (LLM专用环境，可选)
  └── llm_service/
      ├── MindSpore
      ├── MindNLP
      └── Transformers
```

### 5.2 启动流程

#### 5.2.1 推荐方式: start_all.py

```bash
python start_all.py
```

**自动化流程**:
1. 检测conda环境路径
2. 启动ASR服务 (asr环境)
3. 启动LLM服务 (base/llm环境)
4. 启动TTS服务 (tts环境)
5. 启动Orchestrator (base环境)
6. 启动Voice Chat (base环境，如果enabled)
7. 启动Web UI (base环境，如果enabled)

**日志输出**:
```
logs/
  ├── ASR.log           # ASR服务日志
  ├── LLM.log           # LLM服务日志
  ├── TTS.log           # TTS服务日志
  ├── 主控制.log        # Orchestrator日志
  ├── 语音对话.log      # Voice Chat日志
  └── ai_assistant.log  # 主控制器日志
```

#### 5.2.2 手动启动

```bash
# 终端1: ASR服务
conda activate asr
cd asr_service
python app_fastapi.py

# 终端2: LLM服务
cd llm_service
python app_fastapi.py

# 终端3: TTS服务
conda activate tts
cd tts_service
python app_fastapi.py

# 终端4: Orchestrator
python orchestrator.py

# 终端5: Voice Chat
python voice_chat.py

# 终端6: Web UI
python web_ui.py
```

### 5.3 进程管理

**PID文件**:
```
kernel_meta/buildPidInfo.json
```

**格式**:
```json
{
  "ASR": 12345,
  "LLM": 12346,
  "TTS": 12347,
  "主控制": 12348,
  "语音对话": 12349,
  "WebUI": 12350
}
```

**健康检查**:
```python
# 单个服务
GET http://localhost:5001/health

# 聚合检查
GET http://localhost:5000/health
```

**响应示例**:
```json
{
  "status": "healthy",
  "services": {
    "asr": {"status": "up", "latency_ms": 15},
    "llm": {"status": "up", "latency_ms": 23},
    "tts": {"status": "up", "latency_ms": 18}
  }
}
```

### 5.4 配置热更新

系统支持运行时配置更新:

```python
from config_loader import set_config, reload_config

# 方式1: 程序更新
set_config('llm.mode', 'local')

# 方式2: 手动编辑config.yaml后重新加载
reload_config()
```

**自动重载**:
- Web UI修改配置后自动调用`set_config()`
- 各服务在需要时调用`reload_config()`重新读取


## 7. 音频处理

### 7.1 音频格式规范

| 服务 | 输入格式 | 输出格式 |
|------|---------|---------|
| ASR | 16kHz, 单声道, WAV | N/A |
| TTS | N/A | 22050Hz, 单声道, PCM 16-bit |
| Voice Chat | 16kHz, 单声道 (录音) | 22050Hz (播放) |

### 7.2 音频转换

**重采样** (voice_chat.py):
```python
from scipy import signal

def resample_audio(audio_data, orig_sr, target_sr):
    """
    重采样音频到目标采样率

    参数:
      audio_data: 原始音频数据 (numpy array)
      orig_sr: 原始采样率
      target_sr: 目标采样率

    返回:
      重采样后的音频数据
    """
    num_samples = int(len(audio_data) * target_sr / orig_sr)
    resampled = signal.resample(audio_data, num_samples)
    return resampled.astype(np.int16)
```

**WAV文件保存**:
```python
import wave

def save_wav(filename, audio_data, sample_rate=16000, channels=1):
    """保存音频为WAV文件"""
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
```

### 7.3 VAD算法

**RMS能量计算**:
```python
def calculate_rms(audio_chunk):
    """
    计算音频块的RMS (Root Mean Square) 值

    公式: RMS = sqrt(mean(x^2))
    """
    audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
    rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
    return rms

def is_speech(audio_chunk, threshold=200):
    """
    判断音频块是否为语音

    参数:
      audio_chunk: 音频数据
      threshold: RMS阈值

    返回:
      True: 检测到语音
      False: 静音
    """
    rms = calculate_rms(audio_chunk)
    return rms > threshold
```

**静音检测**:
```python
def detect_silence_end(audio_stream, silence_threshold, silence_duration):
    """
    检测静音结束 (用于判断用户说话结束)

    参数:
      audio_stream: 音频流
      silence_threshold: 静音阈值
      silence_duration: 静音持续时间(秒)

    返回:
      检测到静音结束时返回True
    """
    silence_frames = 0
    frames_per_second = 16000 / 1024  # 假设chunk_size=1024
    required_silence_frames = int(silence_duration * frames_per_second)

    for chunk in audio_stream:
        if is_speech(chunk, silence_threshold):
            silence_frames = 0
        else:
            silence_frames += 1

        if silence_frames >= required_silence_frames:
            return True

    return False
```

### 7.4 音量校准

Voice Chat提供音量监测API用于VAD参数校准:

```python
# 启动音量监测
POST /volume/start

# 获取实时音量数据 (SSE)
GET /volume/data

# SSE响应:
data: {"rms": 156, "timestamp": 1234567890.123}

data: {"rms": 289, "timestamp": 1234567890.246}

data: {"rms": 421, "timestamp": 1234567890.369}
```

**校准步骤**:
1. 启动音量监测
2. 在安静环境下测量背景噪音RMS (如: 50-100)
3. 正常说话测量语音RMS (如: 300-800)
4. 设置`silence_threshold`为两者中间值 (如: 200)

---

## 8. 错误处理与容错

### 8.1 服务健康检查

每个服务实现`/health`端点:

```python
@app.get("/health")
async def health_check():
    """健康检查端点"""
    try:
        # 执行基本功能测试
        # 例如: 加载模型、测试API连接等

        return {
            "status": "healthy",
            "timestamp": time.time(),
            "service": "ASR Service",
            "version": "1.0.0"
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )
```

### 8.2 超时控制

**HTTP请求超时**:
```python
response = requests.post(
    url,
    json=data,
    timeout=5.0  # 5秒超时
)
```

**健康检查超时** (orchestrator.py):
```python
def check_service_health(service_url, timeout=5):
    try:
        response = requests.get(
            f"{service_url}/health",
            timeout=timeout
        )
        return response.status_code == 200
    except requests.Timeout:
        logger.error(f"Health check timeout: {service_url}")
        return False
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False
```

### 8.3 重试机制

```python
import time

def retry_request(func, max_attempts=3, delay=1.0):
    """
    重试装饰器

    参数:
      func: 要重试的函数
      max_attempts: 最大尝试次数
      delay: 重试间隔(秒)
    """
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            if attempt == max_attempts - 1:
                raise
            logger.warning(f"Attempt {attempt+1} failed: {e}, retrying...")
            time.sleep(delay)
```

### 8.4 临时文件清理

```python
import tempfile
import os

def safe_temp_file_operation():
    """安全的临时文件操作"""
    temp_file = None
    try:
        # 创建临时文件
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_path = temp_file.name
        temp_file.close()

        # 使用临时文件
        process_audio(temp_path)

    finally:
        # 确保清理
        if temp_file and os.path.exists(temp_path):
            os.unlink(temp_path)
```

### 8.5 流式处理异常

```python
async def stream_with_error_handling():
    """流式响应的错误处理"""
    try:
        for chunk in generate_stream():
            yield chunk
    except Exception as e:
        logger.error(f"Stream error: {e}")
        # 发送错误信息给客户端
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    finally:
        # 发送结束标记
        yield f"data: {json.dumps({'done': true})}\n\n"
```

---

## 9. 性能优化

### 9.1 音频缓存

**缓存策略**:
- 缓存固定回复（唤醒回复、打断回复、思考回复）
- 使用MD5作为缓存键
- 配置变更时自动失效

**实现** (voice_chat.py):
```python
import hashlib

def get_audio_cache_path(text):
    """根据文本生成缓存路径"""
    text_hash = hashlib.md5(text.encode()).hexdigest()
    return f"audio_cache/{text_hash}.pcm"

def get_cached_audio(text):
    """获取缓存的音频"""
    cache_path = get_audio_cache_path(text)
    if os.path.exists(cache_path):
        logger.info(f"✅ 使用缓存音频: {text[:20]}...")
        with open(cache_path, 'rb') as f:
            return f.read()
    return None

def cache_audio(text, audio_data):
    """缓存音频"""
    os.makedirs('audio_cache', exist_ok=True)
    cache_path = get_audio_cache_path(text)
    with open(cache_path, 'wb') as f:
        f.write(audio_data)
    logger.info(f"💾 音频已缓存: {text[:20]}...")
```

**缓存效果**:
- 唤醒响应延迟: 从 ~500ms 降至 <50ms
- 打断响应延迟: 从 ~500ms 降至 <50ms

### 9.2 并发处理

**异步TTS生成与播放**:

使用`AudioPlaybackQueue`实现生产者-消费者模式:

```python
# 生产者: TTS生成线程
def tts_producer(sentence_queue, audio_queue):
    while True:
        sentence = sentence_queue.get()
        if sentence is None:
            break

        # 调用TTS API
        audio_data = synthesize(sentence)

        # 放入播放队列
        audio_queue.put(audio_data)

# 消费者: 音频播放线程
def audio_consumer(audio_queue):
    while True:
        audio_data = audio_queue.get()
        if audio_data is None:
            break

        # 播放音频
        play_audio(audio_data)
```

**优势**:
- TTS合成和音频播放并行
- 减少播放间隙
- 提升用户体验

### 9.3 模型预加载

```python
# 服务启动时预加载模型
@app.on_event("startup")
async def startup_event():
    """启动时预加载资源"""
    logger.info("预加载ASR模型...")
    global asr_model
    asr_model = load_wenet_model()
    logger.info("✅ ASR模型加载完成")
```

### 9.4 连接池复用

```python
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# 创建Session对象复用连接
session = requests.Session()

# 配置重试策略
retry_strategy = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504]
)

adapter = HTTPAdapter(
    max_retries=retry_strategy,
    pool_connections=10,
    pool_maxsize=20
)

session.mount("http://", adapter)
session.mount("https://", adapter)

# 使用session发送请求
response = session.post(url, json=data)
```

---

## 10. 安全性考虑

### 10.1 API密钥管理

**存储方式**:
1. 配置文件 (config.yaml) - 适用于个人部署
2. 环境变量 - 推荐生产环境

```python
import os
from config_loader import get_config

def get_api_key(service):
    """优先从环境变量获取API密钥"""
    env_var = f"{service.upper()}_API_KEY"
    api_key = os.getenv(env_var)

    if not api_key:
        # 回退到配置文件
        api_key = get_config(f'{service}.api.api_key')

    if not api_key:
        raise ValueError(f"API key not found for {service}")

    return api_key
```

**最佳实践**:
- ❌ 不要将API密钥提交到版本控制
- ✅ 使用`.gitignore`排除`config.yaml`
- ✅ 提供`config.yaml.example`作为模板

### 10.2 输入验证

**文本长度限制**:
```python
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    history: List[List[str]] = Field(default=[])
```

**音频文件验证**:
```python
MAX_AUDIO_SIZE = 10 * 1024 * 1024  # 10MB

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    # 检查文件大小
    content = await audio.read()
    if len(content) > MAX_AUDIO_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Audio file too large"
        )

    # 检查文件格式
    if not audio.filename.endswith('.wav'):
        raise HTTPException(
            status_code=400,
            detail="Only WAV format supported"
        )

    # 处理音频...
```

### 10.3 日志脱敏

```python
import re

def sanitize_log_message(message):
    """移除日志中的敏感信息"""
    # 脱敏API密钥
    message = re.sub(
        r'(api_key["\s:=]+)(sk-[a-zA-Z0-9]+)',
        r'\1sk-****',
        message
    )

    # 脱敏其他敏感字段...

    return message

# 自定义日志Handler
class SanitizingHandler(logging.Handler):
    def emit(self, record):
        record.msg = sanitize_log_message(str(record.msg))
        # 调用原始handler...
```

---

## 11. 测试

### 11.1 单元测试

**ASR Service测试**:
```python
import pytest
import requests

def test_asr_health():
    """测试ASR健康检查"""
    response = requests.get("http://localhost:5001/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_asr_transcribe():
    """测试ASR识别"""
    with open("test_audio.wav", "rb") as f:
        files = {"audio": f}
        response = requests.post(
            "http://localhost:5001/transcribe",
            files=files
        )

    assert response.status_code == 200
    assert "text" in response.json()
    assert len(response.json()["text"]) > 0
```

**LLM Service测试**:
```python
def test_llm_chat():
    """测试LLM对话"""
    response = requests.post(
        "http://localhost:5002/chat",
        json={
            "message": "你好",
            "history": []
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert len(data["history"]) == 1
```

**TTS Service测试**:
```python
def test_tts_synthesize():
    """测试TTS合成"""
    response = requests.post(
        "http://localhost:5003/synthesize",
        json={"text": "你好世界"}
    )

    assert response.status_code == 200
    assert response.headers["Content-Type"] == "application/octet-stream"
    assert len(response.content) > 0
```

### 11.2 集成测试

**端到端测试**:
```python
def test_full_conversation_flow():
    """测试完整对话流程"""
    # 1. 准备音频文件
    audio_path = "test_audio.wav"

    # 2. 调用完整流程
    with open(audio_path, "rb") as f:
        files = {"audio": f}
        response = requests.post(
            "http://localhost:5000/conversation/voice",
            files=files,
            data={"history": "[]"}
        )

    # 3. 验证响应
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "history" in data
    assert len(data["history"]) == 1
```

### 11.3 配置重载测试

**文件**: [test_config_reload.py](test_config_reload.py)

```python
from config_loader import get_config, set_config, reload_config

def test_config_reload():
    """测试配置热更新"""
    # 1. 获取初始值
    original_mode = get_config('llm.mode')

    # 2. 修改配置
    set_config('llm.mode', 'local')
    assert get_config('llm.mode') == 'local'

    # 3. 重新加载
    reload_config()

    # 4. 验证服务响应配置变化
    # (需要服务实现reload端点)

    # 5. 恢复原始配置
    set_config('llm.mode', original_mode)
```

---

## 12. 监控与日志

### 12.1 日志配置

**全局日志** (config.yaml):
```yaml
logging:
  level: INFO              # DEBUG/INFO/WARNING/ERROR
  file: ai_assistant.log   # 主日志文件
  max_bytes: 10485760      # 10MB
  backup_count: 5          # 保留5个备份
```

**服务独立日志**:
```
logs/
  ├── ASR.log           # ASR服务
  ├── LLM.log           # LLM服务
  ├── TTS.log           # TTS服务
  ├── 主控制.log        # Orchestrator
  ├── 语音对话.log      # Voice Chat
  └── ai_assistant.log  # Main Controller
```

**日志格式**:
```
2025-01-15 10:23:45,123 - VoiceChat - INFO - 🎤 开始录音...
2025-01-15 10:23:48,456 - VoiceChat - INFO - ✅ 录音结束，时长: 3.2秒
2025-01-15 10:23:48,789 - Orchestrator - INFO - 📝 ASR识别结果: 今天天气怎么样
2025-01-15 10:23:49,012 - Orchestrator - INFO - 🤖 LLM开始生成回复...
2025-01-15 10:23:50,234 - Orchestrator - INFO - ✅ 句子完整: 今天天气晴朗。
2025-01-15 10:23:50,567 - Orchestrator - INFO - 🔊 TTS开始合成...
2025-01-15 10:23:51,890 - VoiceChat - INFO - 🎵 开始播放音频
```

### 12.2 性能指标

**关键指标**:
- **ASR延迟**: 音频时长 → 文本结果的时间
- **LLM首字延迟**: 发送请求 → 第一个token的时间
- **TTS延迟**: 文本 → 音频的时间
- **端到端延迟**: 用户说话 → 听到回复的时间

**指标收集** (示例):
```python
import time

class PerformanceMonitor:
    def __init__(self):
        self.metrics = []

    def record_asr(self, audio_duration, processing_time):
        self.metrics.append({
            'type': 'asr',
            'audio_duration': audio_duration,
            'processing_time': processing_time,
            'timestamp': time.time()
        })

    def record_llm(self, first_token_time, total_time):
        self.metrics.append({
            'type': 'llm',
            'first_token_time': first_token_time,
            'total_time': total_time,
            'timestamp': time.time()
        })

    def get_stats(self):
        """计算统计信息"""
        asr_times = [m['processing_time'] for m in self.metrics if m['type'] == 'asr']
        llm_times = [m['first_token_time'] for m in self.metrics if m['type'] == 'llm']

        return {
            'asr_avg': sum(asr_times) / len(asr_times) if asr_times else 0,
            'llm_avg': sum(llm_times) / len(llm_times) if llm_times else 0
        }
```

### 12.3 服务监控

**Orchestrator聚合健康检查**:

```python
@app.get("/health")
async def aggregate_health_check():
    """聚合所有服务的健康状态"""
    services = get_service_urls()
    health_status = {"status": "healthy", "services": {}}

    for service_name, service_url in services.items():
        try:
            start_time = time.time()
            response = requests.get(
                f"{service_url}/health",
                timeout=5
            )
            latency = (time.time() - start_time) * 1000

            if response.status_code == 200:
                health_status["services"][service_name] = {
                    "status": "up",
                    "latency_ms": round(latency, 2)
                }
            else:
                health_status["status"] = "degraded"
                health_status["services"][service_name] = {
                    "status": "down",
                    "error": f"HTTP {response.status_code}"
                }
        except Exception as e:
            health_status["status"] = "degraded"
            health_status["services"][service_name] = {
                "status": "down",
                "error": str(e)
            }

    return health_status
```

---

## 13. 未来优化方向

### 13.1 功能增强

1. **多语言支持**
   - 自动语言检测
   - 多语言模型切换
   - 实时翻译

2. **情感识别**
   - 语音情感分析
   - 情感化TTS输出
   - 基于情感的回复策略

3. **多轮对话管理**
   - 意图识别
   - 槽位填充
   - 对话状态跟踪

4. **个性化**
   - 用户偏好学习
   - 自定义唤醒词训练
   - 声音克隆优化

### 13.2 性能优化

1. **模型优化**
   - 模型量化 (INT8)
   - 模型蒸馏
   - 边缘设备部署

2. **缓存策略**
   - LRU缓存常见问答
   - 预生成常用回复
   - CDN加速音频分发

3. **并发优化**
   - 异步I/O全面应用
   - 请求批处理
   - GPU推理加速

### 13.3 架构演进

1. **容器化部署**
   ```yaml
   # docker-compose.yml
   version: '3.8'
   services:
     asr:
       build: ./asr_service
       ports:
         - "5001:5001"
     llm:
       build: ./llm_service
       ports:
         - "5002:5002"
     # ...
   ```

2. **服务发现**
   - Consul/Etcd集成
   - 动态服务注册
   - 负载均衡

3. **消息队列**
   - RabbitMQ/Kafka解耦
   - 异步任务处理
   - 事件驱动架构

---

## 14. 故障排查指南

### 14.1 常见问题

#### 问题1: ASR服务无法启动

**现象**:
```
ModuleNotFoundError: No module named 'wenet'
```

**解决方案**:
```bash
conda activate asr
cd asr_service
pip install -r requirements.txt
```

#### 问题2: TTS音频无声音

**现象**: TTS返回音频但播放无声

**排查步骤**:
1. 检查音频设备配置
   ```bash
   curl http://localhost:5004/devices
   ```

2. 检查音量设置
   ```yaml
   voice_chat:
     output_volume: 50  # 调整此值
   ```

3. 测试音频输出
   ```bash
   # 使用aplay测试
   aplay -D plughw:0,0 test.pcm -f S16_LE -r 22050 -c 1
   ```

#### 问题3: VAD检测不准确

**现象**:
- 误将噪音识别为语音
- 说话被判定为静音

**解决方案**:
1. 校准静音阈值
   ```bash
   # 启动音量监测
   curl -X POST http://localhost:5004/volume/start

   # 观察RMS值
   curl http://localhost:5004/volume/data
   ```

2. 调整配置
   ```yaml
   voice_chat:
     silence_threshold: 200  # 根据监测结果调整
     silence_duration: 0.5   # 调整静音判定时长
   ```

#### 问题4: LLM响应慢

**排查步骤**:
1. 检查API模式配置
   ```yaml
   llm:
     mode: api  # API模式通常更快
   ```

2. 调整max_tokens
   ```yaml
   llm:
     api:
       max_tokens: 512  # 减少可加快响应
   ```

3. 检查网络连接
   ```bash
   curl -I https://api.deepseek.com
   ```

### 14.2 日志分析

**查看实时日志**:
```bash
# 主控制日志
tail -f logs/主控制.log

# ASR日志
tail -f logs/ASR.log

# 所有服务日志
tail -f logs/*.log
```

**搜索错误**:
```bash
# 查找ERROR级别日志
grep -r "ERROR" logs/

# 查找特定服务错误
grep "ERROR" logs/LLM.log
```

### 14.3 性能分析

**检查服务响应时间**:
```bash
# ASR
time curl -X POST http://localhost:5001/transcribe -F "audio=@test.wav"

# LLM
time curl -X POST http://localhost:5002/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "你好", "history": []}'

# TTS
time curl -X POST http://localhost:5003/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "测试"}' \
  --output test.pcm
```

---

## 15. 附录

### 15.1 术语表

| 术语 | 全称 | 说明 |
|------|------|------|
| ASR | Automatic Speech Recognition | 自动语音识别 |
| TTS | Text-To-Speech | 文本转语音 |
| LLM | Large Language Model | 大语言模型 |
| VAD | Voice Activity Detection | 语音活动检测 |
| SSE | Server-Sent Events | 服务器推送事件 |
| PCM | Pulse Code Modulation | 脉冲编码调制 |
| RMS | Root Mean Square | 均方根 |
| WAV | Waveform Audio File Format | 波形音频文件格式 |

### 15.2 参考资料

**技术文档**:
- [FastAPI官方文档](https://fastapi.tiangolo.com/)
- [WeNet项目](https://github.com/wenet-e2e/wenet)
- [CosyVoice文档](https://help.aliyun.com/zh/dashscope/cosyvoice)
- [DeepSeek API文档](https://api-docs.deepseek.com/)
- [Gradio文档](https://www.gradio.app/docs)

**相关论文**:
- WeNet: "WeNet: Production Oriented Streaming and Non-streaming End-to-End Speech Recognition Toolkit"
- Streaming ASR: "Streaming Automatic Speech Recognition with the Transformer Model"

### 15.3 项目文件结构

```
ai_助手/
├── asr_service/              # ASR服务
│   ├── app_fastapi.py        # FastAPI应用
│   ├── CN_model/             # 中文模型
│   └── EN_model/             # 英文模型
├── llm_service/              # LLM服务
│   ├── app_fastapi.py        # FastAPI应用
│   ├── deepseek_api.py       # DeepSeek API客户端
│   ├── Qwen1.5-0.5b.py       # Qwen本地模型
│   └── tinyllama.py          # TinyLlama本地模型
├── tts_service/              # TTS服务
│   ├── app_fastapi.py        # FastAPI应用
│   └── cosyvoice_api.py      # CosyVoice API客户端
├── logs/                     # 日志目录
│   ├── ASR.log
│   ├── LLM.log
│   ├── TTS.log
│   ├── 主控制.log
│   └── 语音对话.log
├── audio_cache/              # 音频缓存目录
├── kernel_meta/              # 进程元数据
│   └── buildPidInfo.json     # PID信息
├── config.yaml               # 主配置文件
├── config_loader.py          # 配置加载器
├── orchestrator.py           # 主控制服务
├── voice_chat.py             # 语音对话服务
├── web_ui.py                 # Web配置界面
├── main_controller.py        # 服务管理器
├── start_all.py              # 统一启动脚本
├── requirements.txt          # Python依赖
├── CLAUDE.md                 # Claude Code指南
├── DESIGN.md                 # 本设计文档
└── README.md                 # 项目说明
```

### 15.4 贡献指南

**代码风格**:
- 遵循PEP 8规范
- 使用类型注解
- 添加docstring文档
- 中英文混合注释

**提交规范**:
```
feat: 添加新功能
fix: 修复bug
docs: 更新文档
refactor: 重构代码
perf: 性能优化
test: 添加测试
```

**示例**:
```
feat: 添加多语言ASR支持

- 支持自动语言检测
- 增加日语模型
- 更新配置项文档
```

---

## 变更记录

| 版本 | 日期 | 作者 | 变更说明 |
|------|------|------|---------|
| 1.0.0 | 2025-01-15 | Claude | 初始版本 |

---

**文档状态**: 当前文档与代码库保持同步

**最后更新**: 2025-11-15
