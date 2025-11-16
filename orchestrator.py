"""
主控制服务 (Orchestrator)
串联 ASR -> LLM -> TTS 的完整语音对话流程
支持按句子切分流式处理
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import requests
import json
import logging
import tempfile
import os
import re
from pathlib import Path

from config_loader import get_config
import pyaudio


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Orchestrator")

# 创建FastAPI应用
app = FastAPI(
    title="AI语音助手主控制服务",
    description="串联ASR、LLM、TTS服务的完整对话流程",
    version="1.0.0"
)

# 请求体模型
class VoiceConversationRequest(BaseModel):
    history: Optional[List[List[str]]] = []


class TextConversationRequest(BaseModel):
    text: str
    history: Optional[List[List[str]]] = []
    play_audio: Optional[bool] = True


# ==================== 服务URL配置 ====================
def get_service_urls():
    """获取各服务的URL"""
    ports = get_config('services')
    return {
        'asr': f"http://localhost:{ports.get('asr', 5001)}",
        'llm': f"http://localhost:{ports.get('llm', 5002)}",
        'tts': f"http://localhost:{ports.get('tts', 5003)}"
    }


# ==================== 句子分割工具 ====================
class SentenceSplitter:
    """句子分割器,用于将LLM流式输出按句子切分"""

    def __init__(self):
        streaming_config = get_config('streaming', {})
        self.delimiters = streaming_config.get('sentence_delimiters', ['。', '!', '?', '\n'])
        self.min_chunk_length = streaming_config.get('min_chunk_length', 10)
        self.buffer = ""

    def reload_config(self):
        """重新加载配置"""
        streaming_config = get_config('streaming', {})
        self.delimiters = streaming_config.get('sentence_delimiters', ['。', '!', '?', '\n'])
        self.min_chunk_length = streaming_config.get('min_chunk_length', 10)
        logger.info(f"✅ SentenceSplitter 配置已重新加载: delimiters={self.delimiters}, min_chunk_length={self.min_chunk_length}")

    def add_chunk(self, chunk: str):
        """添加新的文本块"""
        self.buffer += chunk

    def get_complete_sentences(self):
        """获取完整的句子"""
        sentences = []

        # 查找所有分隔符的位置
        for delimiter in self.delimiters:
            if delimiter in self.buffer:
                parts = self.buffer.split(delimiter)

                # 最后一部分可能不完整,保留在buffer中
                for i, part in enumerate(parts[:-1]):
                    sentence = part + delimiter
                    if len(sentence.strip()) >= self.min_chunk_length:
                        sentences.append(sentence)

                # 更新buffer为最后一个不完整的部分
                self.buffer = parts[-1]
                break

        return sentences

    def get_remaining(self):
        """获取剩余的未完成句子"""
        if self.buffer.strip():
            remaining = self.buffer
            self.buffer = ""
            return remaining
        return None


# ==================== 音频播放工具 ====================
class AudioPlayer:
    """PCM音频播放器"""

    def __init__(self, sample_rate=22050, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.player = None
        self.stream = None
        self.available = PYAUDIO_AVAILABLE

    def start(self):
        """启动音频播放"""
        if not self.available:
            logger.warning("pyaudio不可用,无法播放音频")
            return

        if self.player is None:
            self.player = pyaudio.PyAudio()
            self.stream = self.player.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                output=True
            )
            logger.debug("音频播放器已启动")

    def play(self, audio_data: bytes):
        """播放音频数据"""
        if self.stream:
            self.stream.write(audio_data)

    def stop(self):
        """停止音频播放"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.player:
            self.player.terminate()
        self.player = None
        self.stream = None
        logger.debug("音频播放器已停止")


# ==================== 核心流程函数 ====================
async def process_voice_to_text(audio_file_path: str) -> str:
    """
    步骤1: 使用ASR服务将音频转换为文本
    """
    service_urls = get_service_urls()
    asr_url = f"{service_urls['asr']}/transcribe"

    try:
        with open(audio_file_path, 'rb') as f:
            files = {'audio': (os.path.basename(audio_file_path), f, 'audio/wav')}
            response = requests.post(asr_url, files=files, timeout=30)
            response.raise_for_status()

        result = response.json()
        if result.get('success'):
            text = result.get('text', '')
            logger.info(f"✅ ASR识别结果: {text}")
            return text
        else:
            raise Exception("ASR识别失败")

    except Exception as e:
        logger.error(f"❌ ASR服务调用失败: {e}")
        raise HTTPException(status_code=500, detail=f"语音识别失败: {str(e)}")


async def stream_llm_response(text: str, history: List[List[str]]):
    """
    步骤2: 使用LLM服务进行流式对话
    生成器,逐句yield
    """
    service_urls = get_service_urls()
    llm_url = f"{service_urls['llm']}/chat/stream"

    try:
        payload = {
            "message": text,
            "history": history,
            "stream": True
        }

        # 流式请求LLM
        with requests.post(llm_url, json=payload, stream=True, timeout=60) as response:
            response.raise_for_status()

            splitter = SentenceSplitter()

            for line in response.iter_lines():
                if line:
                    try:
                        # 解析SSE格式: "data: {...}"
                        line_str = line.decode('utf-8')
                        if line_str.startswith('data: '):
                            data_str = line_str[6:]
                            data = json.loads(data_str)

                            # 检查是否完成
                            if data.get('done'):
                                # 返回剩余的文本
                                remaining = splitter.get_remaining()
                                if remaining:
                                    yield remaining
                                break

                            # 检查是否有错误
                            if 'error' in data:
                                logger.error(f"LLM错误: {data['error']}")
                                break

                            # 获取增量文本
                            delta = data.get('delta', '')
                            if delta:
                                splitter.add_chunk(delta)

                                # 获取完整的句子
                                sentences = splitter.get_complete_sentences()
                                for sentence in sentences:
                                    logger.info(f"📝 LLM输出句子: {sentence}")
                                    yield sentence

                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        logger.error(f"处理LLM响应出错: {e}")
                        continue

    except Exception as e:
        logger.error(f"❌ LLM服务调用失败: {e}")
        raise HTTPException(status_code=500, detail=f"对话生成失败: {str(e)}")


async def synthesize_and_play(text: str, play: bool = True):
    """
    步骤3: 使用TTS服务合成语音并播放
    """
    service_urls = get_service_urls()
    tts_url = f"{service_urls['tts']}/synthesize/stream"

    try:
        payload = {"text": text, "stream": True}

        # 流式请求TTS
        with requests.post(tts_url, json=payload, stream=True, timeout=30) as response:
            response.raise_for_status()

            if play:
                # 初始化音频播放器
                player = AudioPlayer(sample_rate=22050, channels=1)
                player.start()

                # 流式播放音频
                for chunk in response.iter_content(chunk_size=4096):
                    if chunk:
                        player.play(chunk)
                        yield chunk

                player.stop()
            else:
                # 仅返回音频数据
                for chunk in response.iter_content(chunk_size=4096):
                    if chunk:
                        yield chunk

        logger.info(f"🔊 TTS合成并播放完成: {text[:30]}...")

    except Exception as e:
        logger.error(f"❌ TTS服务调用失败: {e}")
        raise HTTPException(status_code=500, detail=f"语音合成失败: {str(e)}")


# ==================== FastAPI 路由 ====================
@app.on_event("startup")
async def startup_event():
    """服务启动事件"""
    logger.info("🚀 主控制服务正在启动...")
    logger.info("✅ 主控制服务启动完成!")


@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "AI语音助手主控制服务",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """健康检查,检查所有依赖服务"""
    service_urls = get_service_urls()
    health_status = {
        "orchestrator": "healthy",
        "services": {}
    }

    # 检查各服务健康状态
    for service_name, base_url in service_urls.items():
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                health_status["services"][service_name] = "healthy"
            else:
                health_status["services"][service_name] = "unhealthy"
        except Exception as e:
            health_status["services"][service_name] = f"error: {str(e)}"

    # 判断整体状态
    all_healthy = all(status == "healthy" for status in health_status["services"].values())

    return {
        "status": "healthy" if all_healthy else "degraded",
        **health_status
    }


@app.post("/reload_config")
async def reload_config():
    """
    重新加载配置
    更新服务URL配置和流式处理配置
    """
    try:
        from config_loader import reload_config as reload_config_file
        reload_config_file()

        logger.info("✅ Orchestrator 配置已重新加载")

        # 获取更新后的配置
        service_urls = get_service_urls()
        streaming_config = get_config('streaming', {})

        return {
            "success": True,
            "message": "配置已重新加载",
            "service_urls": service_urls,
            "streaming_config": {
                "sentence_delimiters": streaming_config.get('sentence_delimiters'),
                "min_chunk_length": streaming_config.get('min_chunk_length')
            }
        }
    except Exception as e:
        logger.error(f"❌ 配置重新加载失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"配置重新加载失败: {str(e)}")


@app.post("/conversation/voice")
async def voice_conversation(audio: UploadFile = File(...)):
    """
    完整的语音对话流程
    1. ASR: 音频 -> 文本
    2. LLM: 文本 -> 流式响应
    3. TTS: 流式响应 -> 音频播放

    返回: 对话文本和完整音频
    """
    logger.info("🎤 开始语音对话流程...")

    # 保存上传的音频文件
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
        content = await audio.read()
        temp_file.write(content)
        temp_audio_path = temp_file.name

    try:
        # 步骤1: 语音识别
        user_text = await process_voice_to_text(temp_audio_path)

        # 步骤2 & 3: LLM生成 + TTS合成播放
        full_response = ""
        audio_player = AudioPlayer(sample_rate=22050, channels=1)
        audio_player.start()

        async for sentence in stream_llm_response(user_text, []):
            full_response += sentence

            # 为每个句子合成语音并播放
            async for audio_chunk in synthesize_and_play(sentence, play=False):
                audio_player.play(audio_chunk)

        audio_player.stop()

        # 清理临时文件
        os.unlink(temp_audio_path)

        logger.info("✅ 语音对话流程完成!")

        return JSONResponse(content={
            "success": True,
            "user_text": user_text,
            "assistant_text": full_response
        })

    except Exception as e:
        logger.error(f"❌ 语音对话流程失败: {e}")
        if os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/conversation/text")
async def text_conversation(request: TextConversationRequest):
    """
    文本对话流程(跳过ASR)
    1. LLM: 文本 -> 流式响应
    2. TTS: 流式响应 -> 音频播放

    返回: 流式文本响应
    """
    logger.info(f"💬 开始文本对话: {request.text[:50]}...")

    async def generate_response():
        """生成流式响应"""
        full_response = ""

        # 初始化音频播放器(如果需要)
        audio_player = None
        if request.play_audio:
            audio_player = AudioPlayer(sample_rate=22050, channels=1)
            audio_player.start()

        try:
            async for sentence in stream_llm_response(request.text, request.history):
                full_response += sentence

                # 发送文本响应
                yield f"data: {json.dumps({'type': 'text', 'content': sentence})}\n\n"

                # 合成并播放语音
                if request.play_audio and audio_player:
                    async for audio_chunk in synthesize_and_play(sentence, play=False):
                        audio_player.play(audio_chunk)

            # 发送完成标记
            yield f"data: {json.dumps({'type': 'done', 'full_response': full_response})}\n\n"

        finally:
            if audio_player:
                audio_player.stop()

    return StreamingResponse(
        generate_response(),
        media_type="text/event-stream"
    )


@app.websocket("/ws/conversation")
async def websocket_conversation(websocket: WebSocket):
    """
    WebSocket实时对话接口
    支持双向语音对话
    """
    await websocket.accept()
    logger.info("WebSocket连接已建立")

    try:
        while True:
            # 接收客户端消息
            data = await websocket.receive_json()
            message_type = data.get('type')

            if message_type == 'text':
                # 文本消息
                text = data.get('text')
                history = data.get('history', [])

                full_response = ""
                async for sentence in stream_llm_response(text, history):
                    full_response += sentence
                    await websocket.send_json({
                        'type': 'text',
                        'content': sentence
                    })

                await websocket.send_json({
                    'type': 'done',
                    'full_response': full_response
                })

    except WebSocketDisconnect:
        logger.info("WebSocket连接已断开")
    except Exception as e:
        logger.error(f"WebSocket错误: {e}")
        await websocket.close()


if __name__ == "__main__":
    # 从配置文件读取端口
    port = get_config('services.orchestrator', 5000)

    logger.info(f"主控制服务启动在端口: {port}")

    # 启动服务
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
