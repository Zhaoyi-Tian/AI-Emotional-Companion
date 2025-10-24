# 🤖 AI语音助手

基于FastAPI的微服务架构智能语音助手,支持语音识别(ASR)、大模型对话(LLM)、语音合成(TTS)的完整流程。

## 📋 架构设计

```
用户
 ↓
Web界面 (端口8080)
 ↓
主控制服务 (端口5000)
 ↙  ↓  ↘
ASR  LLM  TTS
5001 5002 5003
```

## 🔄 数据流

```
音频输入 → ASR服务(非流式识别) → 文本
                                    ↓
                              LLM服务(流式生成) → 句子1, 句子2, ...
                                    ↓
                              TTS服务(流式合成) → 播放音频
```

## 📁 项目结构

```
ai_助手/
├── config.yaml              # 统一配置文件
├── config_loader.py         # 配置加载器
├── orchestrator.py          # 主控制服务
├── web_ui.py               # Web配置界面
├── start_all.py            # 统一启动脚本
├── requirements.txt        # Python依赖
│
├── asr_service/            # ASR语音识别服务
│   ├── app_fastapi.py     # FastAPI应用
│   └── wenet/             # WeNet模型
│
├── llm_service/            # LLM大模型服务
│   ├── app_fastapi.py     # FastAPI应用
│   ├── deepseek_api.py    # DeepSeek API调用
│   └── ...                # 本地模型
│
└── tts_service/            # TTS语音合成服务
    ├── app_fastapi.py     # FastAPI应用
    └── cosyvoice_api.py   # CosyVoice API调用
```

## 🚀 快速开始

### 1. 安装依赖

```bash
# 安装基础依赖
pip install -r requirements.txt

# 注意: ASR、LLM、TTS服务可能需要特定的conda环境
# 根据你的需求配置相应的环境
```

### 2. 配置服务

编辑 `config.yaml` 文件,配置API密钥和模型:

```yaml
llm:
  api:
    api_key: "你的DeepSeek API Key"

tts:
  api:
    api_key: "你的CosyVoice API Key"
```

### 3. 启动服务

#### 方式1: 使用统一启动脚本 (推荐)

```bash
python start_all.py
```

这会自动启动所有服务:
- ASR服务 (端口5001)
- LLM服务 (端口5002)
- TTS服务 (端口5003)
- 主控制服务 (端口5000)
- Web配置界面 (端口8080)

#### 方式2: 手动启动各服务

```bash
# 终端1: ASR服务
cd asr_service
conda activate asr
python app_fastapi.py

# 终端2: LLM服务
cd llm_service
python app_fastapi.py

# 终端3: TTS服务
cd tts_service
conda activate tts
python app_fastapi.py

# 终端4: 主控制服务
python orchestrator.py

# 终端5: Web界面
python web_ui.py
```

### 4. 访问服务

**本地访问**:
- **Web配置界面**: http://localhost:8080
- **API文档**:
  - 主控制服务: http://localhost:5000/docs
  - ASR服务: http://localhost:5001/docs
  - LLM服务: http://localhost:5002/docs
  - TTS服务: http://localhost:5003/docs

**内网访问** (局域网其他设备):
- 启动服务后,终端会显示内网IP地址
- 例如: `http://192.168.1.100:8080`
- 同一局域网下的手机、平板、其他电脑都可以通过此地址访问

> 💡 提示: 确保防火墙允许相应端口的访问

## 🎯 使用方式

### 方式1: Web界面 (推荐)

1. 访问 http://localhost:8080
2. 在"服务状态"页面检查所有服务是否正常
3. 在各配置页面修改参数(API Key、模型选择等)
4. 使用测试功能验证各服务

### 方式2: API调用

#### 文本对话

```bash
curl -X POST "http://localhost:5000/conversation/text" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好,今天天气怎么样?",
    "history": [],
    "play_audio": true
  }'
```

#### 语音对话

```bash
curl -X POST "http://localhost:5000/conversation/voice" \
  -F "audio=@test.wav"
```

### 方式3: Python调用

```python
import requests

# 文本对话
response = requests.post(
    "http://localhost:5000/conversation/text",
    json={
        "text": "讲个笑话",
        "history": [],
        "play_audio": True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))
```

## ⚙️ 配置说明

### ASR配置

```yaml
asr:
  model_type: "EN"  # CN(中文) 或 EN(英文)
```

### LLM配置

```yaml
llm:
  mode: "api"  # api(云端) 或 local(本地)
  api:
    provider: "deepseek"
    api_key: "sk-xxx"
    model: "deepseek-chat"
    max_tokens: 512
    temperature: 1.0
    system_prompt: "You are a helpful assistant"
```

### TTS配置

```yaml
tts:
  mode: "api"  # api(云端) 或 local(本地)
  api:
    provider: "cosyvoice"
    api_key: "sk-xxx"
    model: "cosyvoice-v2"
    voice: "longxiaochun_v2"  # 发音人
```

### 流式处理配置

```yaml
streaming:
  sentence_delimiters: ["。", "!", "?", "\n"]  # 句子分隔符
  min_chunk_length: 10  # 最小块长度
```

## 🔧 核心功能

### 1. 按句子切分流式处理

LLM的流式输出会被智能切分成完整句子,然后逐句发送给TTS合成,实现更流畅的对话体验:

```
LLM: "今天天气" → 累积中...
LLM: "很好。" → 检测到句号,发送"今天天气很好。"给TTS
TTS: 合成并播放 "今天天气很好。"
LLM: "适合出" → 累积中...
LLM: "去玩!" → 检测到感叹号,发送"适合出去玩!"给TTS
TTS: 合成并播放 "适合出去玩!"
```

### 2. 动态配置管理

- 支持在Web界面实时修改配置
- 修改后可重新加载服务配置,无需重启
- 配置持久化到 `config.yaml`

### 3. 服务健康监控

- 统一的健康检查接口
- Web界面实时显示各服务状态
- 支持单独重启某个服务

## 📝 API接口说明

### 主控制服务 (端口5000)

- `POST /conversation/voice` - 完整语音对话(音频→文本→LLM→语音)
- `POST /conversation/text` - 文本对话(文本→LLM→语音)
- `WS /ws/conversation` - WebSocket实时对话
- `GET /health` - 健康检查

### ASR服务 (端口5001)

- `POST /transcribe` - 语音识别
- `GET /health` - 健康检查

### LLM服务 (端口5002)

- `POST /chat/stream` - 流式对话
- `POST /chat` - 非流式对话
- `GET /health` - 健康检查

### TTS服务 (端口5003)

- `POST /synthesize/stream` - 流式语音合成
- `POST /synthesize` - 非流式语音合成
- `GET /health` - 健康检查

## 🐛 故障排查

### 服务无法启动

1. 检查端口是否被占用: `lsof -i :5000`
2. 检查conda环境是否正确激活
3. 查看服务日志输出

### API调用失败

1. 检查API Key是否正确配置
2. 检查网络连接
3. 查看服务健康状态: 访问 http://localhost:8080

### 音频无法播放

1. 检查系统音频设备
2. 确认pyaudio正确安装: `python -c "import pyaudio"`
3. 检查TTS服务是否正常

## 📦 依赖说明

- **FastAPI**: Web框架
- **Uvicorn**: ASGI服务器
- **Gradio**: Web界面
- **PyAudio**: 音频播放
- **Requests**: HTTP客户端
- **DashScope**: 阿里云TTS API

## 🔐 安全建议

1. 不要将API Key提交到版本控制
2. 使用环境变量或密钥管理服务
3. 在生产环境中配置防火墙规则
4. 定期更新依赖包

## 📄 License

MIT License

## 🤝 贡献

欢迎提交Issue和Pull Request!

## 📧 联系方式

如有问题,请提交Issue或联系维护者。
