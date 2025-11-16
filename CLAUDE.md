# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI语音助手 (AI Voice Assistant) - A microservices-based intelligent voice assistant system supporting complete ASR → LLM → TTS pipeline with streaming capabilities.

## Core Architecture

The system follows a microservices architecture with service orchestration:

```
User Input (Audio/Text)
    ↓
Web UI (port 8080) / Voice Chat (port 5004)
    ↓
Orchestrator (port 5000) - Main Controller
    ↓ ↓ ↓
ASR (5001)  LLM (5002)  TTS (5003)
```

### Service Communication Flow

1. **Voice Conversation Flow**: Audio → ASR (non-streaming) → Text → LLM (streaming) → Sentences → TTS (streaming) → Audio playback
2. **Text Conversation Flow**: Text → LLM (streaming) → Sentences → TTS (streaming) → Audio playback
3. **Streaming Architecture**: The orchestrator implements sentence-based streaming where LLM output is intelligently split into complete sentences and sent to TTS for synthesis while the LLM continues generating, enabling low-latency responses

### Key Architectural Components

- **orchestrator.py**: Main control service that chains ASR → LLM → TTS. Implements `SentenceSplitter` class for intelligent sentence boundary detection during streaming
- **voice_chat.py**: Offline voice conversation system with VAD (Voice Activity Detection), wake word detection, interrupt handling, and audio playback queue management using producer-consumer pattern
- **main_controller.py**: Service lifecycle manager that starts/stops/monitors all microservices
- **start_all.py**: Unified launcher script that bootstraps all services with proper conda environment detection
- **web_ui.py**: Gradio-based web configuration interface providing service status monitoring, configuration management, and interactive testing

## Common Development Commands

### Starting the System

```bash
# Start all services (recommended)
python start_all.py

# Individual service startup (for debugging)
cd asr_service && conda activate asr && python app_fastapi.py
cd llm_service && python app_fastapi.py
cd tts_service && conda activate tts && python app_fastapi.py
python orchestrator.py
python web_ui.py
python voice_chat.py
```

### Configuration Management

All configuration is centralized in `config.yaml`. Use `config_loader.py` to access config:

```python
from config_loader import get_config, set_config, reload_config

# Get nested config value
api_key = get_config('llm.api.api_key')
port = get_config('services.asr', default=5001)

# Update config
set_config('llm.mode', 'api')

# Reload after manual edits
reload_config()
```

### Service Health Checks

Each service exposes `/health` endpoint. Orchestrator provides aggregate health check at `http://localhost:5000/health`.

### Logging

Service logs are written to `logs/` directory. Main log file is `ai_assistant.log`. Each service has its own log file when started via `start_all.py`.

## Service-Specific Details

### ASR Service (asr_service/)

- Uses WeNet models for speech recognition
- Supports CN (Chinese) and EN (English) models configured via `config.yaml`
- Model files located in `CN_model/` and `EN_model/` directories
- Audio must be 16kHz WAV format (automatic resampling handled by voice_chat.py)

### LLM Service (llm_service/)

- Dual mode: API (DeepSeek) or local (Qwen/TinyLlama via MindNLP)
- Supports streaming via Server-Sent Events (SSE) with `data: {...}` format
- History format: `[[user_msg_1, ai_response_1], [user_msg_2, ai_response_2]]`
- Streaming endpoint: `/chat/stream`, non-streaming: `/chat`

### TTS Service (tts_service/)

- Dual mode: API (CosyVoice) or local
- Supports voice cloning/enrollment via CosyVoice API
- Returns PCM audio (22050Hz, mono, 16-bit)
- Streaming endpoint: `/synthesize/stream`, non-streaming: `/synthesize`

### Orchestrator Service (orchestrator.py)

- **SentenceSplitter**: Buffers LLM output and yields complete sentences based on delimiters configured in `streaming.sentence_delimiters`
- **AudioPlayer**: Manages PyAudio playback with configurable sample rates
- Provides `/conversation/voice` (full pipeline) and `/conversation/text` (skip ASR) endpoints
- WebSocket support at `/ws/conversation` for real-time bidirectional communication

### Voice Chat Service (voice_chat.py)

- **VoiceAssistant**: Core class managing offline voice interaction loop
- **AudioPlaybackQueue**: Producer-consumer queue for asynchronous TTS generation and playback
- **VAD Implementation**: RMS-based voice activity detection with configurable `silence_threshold` and `silence_duration`
- **Wake Word Mode**: Real-time continuous monitoring for wake words using `monitor_wake_word()` method
  - Continuously listens for audio input (not periodic 2-second chunks)
  - Detects speech using VAD (RMS threshold)
  - Recognizes text after 1 second of silence
  - Checks for wake words in recognized text
  - Returns immediately when wake word detected with any remaining text
- **Interrupt Mode**: Real-time monitoring for interrupt words during AI speech playback using `monitor_interrupt()` method
- **Audio Caching**: Caches wake reply and interrupt reply audio in `audio_cache/` directory to avoid regenerating same audio
  - Uses MD5 hash of text for cache filenames
  - Automatically loads cached audio on startup
  - Invalidates cache when reply text changes in config
- Audio device configuration via `input_device` and `output_device` in `voice_chat` config section
- API endpoints: `/start`, `/stop`, `/status`, `/devices`, `/volume/start`, `/volume/data`

## Important Implementation Notes

### Streaming Architecture

The sentence-based streaming is critical for low latency. When modifying streaming behavior:

1. LLM output is buffered in `SentenceSplitter`
2. Complete sentences are detected using `sentence_delimiters`
3. Each complete sentence is immediately sent to TTS
4. TTS synthesizes while LLM continues generating next sentence
5. Audio playback begins as soon as first sentence is synthesized

### Conversation History Format

Always use the two-dimensional list format:
```python
# Correct
history = [["user question 1", "ai answer 1"], ["user question 2", "ai answer 2"]]

# Incorrect
history = [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
```

### Conda Environment Requirements

- ASR service requires `asr` conda environment
- TTS service requires `tts` (or `cosyvoice`) conda environment
- LLM local mode may require specific environment depending on model
- Base environment used for orchestrator, web_ui, voice_chat

The `start_all.py` script automatically detects conda environments in standard locations:
- `~/.conda/envs/`
- `~/miniconda3/envs/`
- `~/anaconda3/envs/`
- `/opt/miniconda3/envs/`
- `/opt/anaconda3/envs/`

### Audio Device Handling

Voice chat uses PyAudio with careful error suppression for ALSA warnings. Device indices can be discovered via `/devices` endpoint or `list_audio_devices()` method. System default devices are used if not specified in config.

### Error Handling Best Practices

- All HTTP endpoints should return proper error messages with appropriate status codes
- Service health checks must timeout within 5 seconds
- Audio processing operations should cleanup temporary files in finally blocks
- Streaming generators should handle client disconnection gracefully

## Testing Services

Use the Web UI (port 8080) for interactive testing, or:

```bash
# Test ASR
curl -X POST http://localhost:5001/transcribe -F "audio=@test.wav"

# Test LLM (non-streaming)
curl -X POST http://localhost:5002/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "history": []}'

# Test TTS
curl -X POST http://localhost:5003/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}' \
  --output test.pcm

# Test full pipeline
curl -X POST http://localhost:5000/conversation/text \
  -H "Content-Type: application/json" \
  -d '{"text": "讲个笑话", "history": [], "play_audio": false}'
```

## File Organization Principles

- Service implementations: `*_service/app_fastapi.py`
- Service-specific utilities: Within service directories
- Shared utilities: Root level (config_loader.py)
- Configuration: `config.yaml` (single source of truth)
- Logs: `logs/` directory
- Documentation: Root level `*.md` files for specific features

## Key Configuration Parameters

### Voice Chat Configuration

```yaml
voice_chat:
  enable: true/false           # Auto-start voice chat when service starts
  wake_mode: true/false        # Enable wake word detection
  wake_words: [...]            # List of wake words
  wake_reply: "..."            # Confirmation speech after wake word
  interrupt_mode: true/false   # Enable interrupt detection during playback
  interrupt_words: [...]       # Words that trigger interruption
  silence_threshold: 200       # RMS threshold for VAD (use /volume APIs to calibrate)
  silence_duration: 2.0        # Seconds of silence to end recording
  min_audio_length: 1.5        # Minimum valid audio length
  input_device: null/int       # PyAudio device index for microphone
  output_device: null/int      # PyAudio device index for speaker
  output_volume: 50            # Output volume 0-100%
```

### Streaming Configuration

```yaml
streaming:
  sentence_delimiters: ["。", "!", "?", "\n", ".", ";"]
  min_chunk_length: 10         # Minimum characters before sending to TTS
  max_wait_time: 3.0          # Maximum seconds to wait for complete sentence
```

## Port Allocation

- 5000: Orchestrator (main control service)
- 5001: ASR service
- 5002: LLM service
- 5003: TTS service
- 5004: Voice chat service
- 8080: Web UI (Gradio)

All ports configurable in `config.yaml` under `services` section.
