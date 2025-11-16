"""
TTS语音合成服务 - FastAPI版本
支持流式和非流式语音合成
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
from typing import Optional
import uvicorn
import sys
import logging
import io
import threading
from pathlib import Path

# 添加父目录到路径以导入配置
sys.path.insert(0, str(Path(__file__).parent.parent))
from config_loader import get_config

try:
    import dashscope
    from dashscope.audio.tts_v2 import SpeechSynthesizer, ResultCallback, AudioFormat, VoiceEnrollmentService
    DASHSCOPE_AVAILABLE = True
except ImportError:
    logger.warning("dashscope未安装,TTS服务将以受限模式运行")
    DASHSCOPE_AVAILABLE = False
    # 定义占位类
    class ResultCallback:
        pass
    class AudioFormat:
        PCM_22050HZ_MONO_16BIT = "pcm_22050hz_mono_16bit"
    class VoiceEnrollmentService:
        pass

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TTS_Service")

# 创建FastAPI应用
app = FastAPI(
    title="TTS语音合成服务",
    description="基于CosyVoice的语音合成服务",
    version="1.0.0"
)

# 请求体模型
class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    stream: Optional[bool] = True


class TTSResponse(BaseModel):
    success: bool
    message: str
    audio_format: str


# 音色克隆相关请求体模型
class VoiceEnrollmentRequest(BaseModel):
    """创建音色请求"""
    target_model: str  # 例如: cosyvoice-v2, cosyvoice-v3
    prefix: str  # 音色前缀,仅允许数字和小写字母,小于10个字符
    url: str  # 音频文件URL


class VoiceUpdateRequest(BaseModel):
    """更新音色请求"""
    voice_id: str
    url: str  # 新的音频文件URL


class VoiceListRequest(BaseModel):
    """查询音色列表请求"""
    prefix: Optional[str] = None
    page_index: int = 0
    page_size: int = 10


class VoiceQueryRequest(BaseModel):
    """查询指定音色请求"""
    voice_id: str


class VoiceDeleteRequest(BaseModel):
    """删除音色请求"""
    voice_id: str


# ==================== CosyVoice API 模式 ====================
class AudioBufferCallback(ResultCallback):
    """音频缓冲回调类,用于收集音频数据"""

    def __init__(self):
        self.audio_buffer = io.BytesIO()
        self.error_message = None
        self.is_complete = False

    def on_open(self):
        logger.debug("TTS WebSocket已打开")

    def on_complete(self):
        logger.debug("TTS合成完成")
        self.is_complete = True

    def on_error(self, message: str):
        logger.error(f"TTS合成失败: {message}")
        self.error_message = message

    def on_close(self):
        logger.debug("TTS WebSocket已关闭")

    def on_event(self, message):
        pass

    def on_data(self, data: bytes) -> None:
        """接收音频数据"""
        self.audio_buffer.write(data)

    def get_audio_data(self) -> bytes:
        """获取完整的音频数据"""
        return self.audio_buffer.getvalue()


class StreamingAudioCallback(ResultCallback):
    """流式音频回调类,用于生成器"""

    def __init__(self):
        self.error_message = None
        self.audio_chunks = []
        self.is_complete = False

    def on_open(self):
        logger.debug("TTS WebSocket已打开(流式)")

    def on_complete(self):
        logger.debug("TTS合成完成(流式)")
        self.is_complete = True

    def on_error(self, message: str):
        logger.error(f"TTS合成失败(流式): {message}")
        self.error_message = message

    def on_close(self):
        logger.debug("TTS WebSocket已关闭(流式)")

    def on_event(self, message):
        pass

    def on_data(self, data: bytes) -> None:
        """接收音频数据并添加到队列"""
        self.audio_chunks.append(data)


def init_cosyvoice_config():
    """初始化CosyVoice配置"""
    if not DASHSCOPE_AVAILABLE:
        logger.warning("⚠️ dashscope未安装,请运行: pip install dashscope")
        return

    tts_config = get_config('tts')
    api_config = tts_config.get('api', {})

    api_key = api_config.get('api_key')
    if not api_key:
        logger.warning("⚠️ CosyVoice API Key未配置")
        return

    dashscope.api_key = api_key
    logger.info("✅ CosyVoice API配置完成")


# ==================== SpeechSynthesizer 连接池 ====================
class SynthesizerPool:
    """
    SpeechSynthesizer 连接池
    复用 synthesizer 对象，避免重复创建，减少连接开销
    """
    def __init__(self, max_size=5):
        self.max_size = max_size
        self.pool = []
        self.lock = threading.Lock()
        logger.info(f"🔧 初始化 Synthesizer 连接池，最大容量: {max_size}")

    def _create_synthesizer(self, model: str, voice: str, audio_format, callback):
        """创建新的 synthesizer"""
        return SpeechSynthesizer(
            model=model,
            voice=voice,
            format=audio_format,
            callback=callback
        )

    def get(self, model: str, voice: str, audio_format, callback):
        """
        从池中获取或创建 synthesizer
        注意：由于每次请求的 callback 不同，目前策略是每次创建新对象
        但保留池的架构，便于未来优化
        """
        # 目前每次都创建新对象（因为callback不同）
        # 未来可以优化为复用对象，只更换callback
        synthesizer = self._create_synthesizer(model, voice, audio_format, callback)
        logger.debug(f"📦 创建新的 Synthesizer: model={model}, voice={voice}")
        return synthesizer

    def release(self, synthesizer):
        """
        释放 synthesizer 回池中
        目前不做实际回收，未来可以优化
        """
        # 当前实现不回收，因为 CosyVoice API 的 synthesizer 是一次性的
        # 每次调用都需要新的 callback
        pass

    def clear(self):
        """清空连接池"""
        with self.lock:
            self.pool.clear()
            logger.info("🧹 连接池已清空")


# 全局连接池实例
synthesizer_pool = SynthesizerPool(max_size=5)


async def synthesize_speech_stream(text: str, voice: Optional[str] = None):
    """流式语音合成"""
    if not DASHSCOPE_AVAILABLE:
        raise HTTPException(status_code=503, detail="dashscope未安装,请运行: pip install dashscope")

    tts_config = get_config('tts')
    api_config = tts_config.get('api', {})

    model = api_config.get('model', 'cosyvoice-v2')
    voice = voice or api_config.get('voice', 'longxiaochun_v2')
    audio_format = AudioFormat.PCM_22050HZ_MONO_16BIT

    try:
        callback = StreamingAudioCallback()
        # 从连接池获取 synthesizer
        synthesizer = synthesizer_pool.get(model, voice, audio_format, callback)

        # 启动流式合成
        synthesizer.streaming_call(text)

        # 生成音频流
        import time
        max_wait = 10  # 最大等待时间(秒)
        wait_time = 0
        chunk_index = 0

        while wait_time < max_wait:
            # 检查是否有新的音频块
            if chunk_index < len(callback.audio_chunks):
                chunk = callback.audio_chunks[chunk_index]
                chunk_index += 1
                yield chunk
                wait_time = 0  # 重置等待时间
            elif callback.is_complete:
                # 合成完成,退出循环
                break
            elif callback.error_message:
                # 发生错误
                logger.error(f"流式合成错误: {callback.error_message}")
                break
            else:
                # 等待新数据
                time.sleep(0.05)
                wait_time += 0.05

        # 完成合成
        synthesizer.streaming_complete()
        logger.info(f"流式合成完成,请求ID: {synthesizer.get_last_request_id()}")

    except Exception as e:
        logger.error(f"流式语音合成失败: {e}")
        raise HTTPException(status_code=500, detail=f"合成失败: {str(e)}")


async def synthesize_speech(text: str, voice: Optional[str] = None) -> bytes:
    """非流式语音合成"""
    if not DASHSCOPE_AVAILABLE:
        raise HTTPException(status_code=503, detail="dashscope未安装,请运行: pip install dashscope")

    tts_config = get_config('tts')
    api_config = tts_config.get('api', {})

    model = api_config.get('model', 'cosyvoice-v2')
    voice = voice or api_config.get('voice', 'longxiaochun_v2')
    audio_format = AudioFormat.PCM_22050HZ_MONO_16BIT

    try:
        callback = AudioBufferCallback()
        # 从连接池获取 synthesizer
        synthesizer = synthesizer_pool.get(model, voice, audio_format, callback)

        # 执行合成
        synthesizer.streaming_call(text)
        synthesizer.streaming_complete()

        # 检查错误
        if callback.error_message:
            raise Exception(callback.error_message)

        logger.info(f"非流式合成完成,请求ID: {synthesizer.get_last_request_id()}")
        return callback.get_audio_data()

    except Exception as e:
        logger.error(f"语音合成失败: {e}")
        raise HTTPException(status_code=500, detail=f"合成失败: {str(e)}")


# ==================== FastAPI 路由 ====================
@app.on_event("startup")
async def startup_event():
    """服务启动事件"""
    logger.info("🚀 TTS服务正在启动...")
    init_cosyvoice_config()
    logger.info("✅ TTS服务启动完成!")


@app.get("/")
async def root():
    """根路径"""
    tts_config = get_config('tts')
    return {
        "service": "TTS语音合成服务",
        "status": "running",
        "mode": tts_config.get('mode'),
        "model": tts_config.get('api', {}).get('model'),
        "voice": tts_config.get('api', {}).get('voice')
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "service": "tts",
        "mode": get_config('tts.mode')
    }


@app.post("/synthesize/stream")
async def synthesize_stream(request: TTSRequest):
    """
    流式语音合成接口
    返回音频流(PCM格式)
    """
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="文本不能为空")

    logger.info(f"收到流式合成请求: {request.text[:50]}...")

    return StreamingResponse(
        synthesize_speech_stream(request.text, request.voice),
        media_type="audio/pcm",
        headers={
            "Content-Disposition": f"attachment; filename=speech.pcm",
            "X-Sample-Rate": "22050",
            "X-Channels": "1",
            "X-Bit-Depth": "16"
        }
    )


@app.post("/synthesize")
async def synthesize(request: TTSRequest):
    """
    非流式语音合成接口
    返回完整音频文件
    """
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="文本不能为空")

    logger.info(f"收到合成请求: {request.text[:50]}...")

    try:
        audio_data = await synthesize_speech(request.text, request.voice)

        return Response(
            content=audio_data,
            media_type="audio/pcm",
            headers={
                "Content-Disposition": "attachment; filename=speech.pcm",
                "X-Sample-Rate": "22050",
                "X-Channels": "1",
                "X-Bit-Depth": "16"
            }
        )

    except Exception as e:
        logger.error(f"合成失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload_config")
async def reload_config():
    """重新加载配置"""
    try:
        from config_loader import reload_config
        reload_config()
        init_cosyvoice_config()
        return {"success": True, "message": "配置重新加载成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"配置重新加载失败: {str(e)}")


# ==================== 音色克隆 API ====================
def get_voice_service():
    """获取音色克隆服务实例"""
    if not DASHSCOPE_AVAILABLE:
        raise HTTPException(status_code=503, detail="dashscope未安装,请运行: pip install dashscope")

    return VoiceEnrollmentService()


@app.post("/voice/create")
async def create_voice(request: VoiceEnrollmentRequest):
    """
    创建音色克隆

    参数:
    - target_model: 声音复刻所使用的模型 (例如: cosyvoice-v2, cosyvoice-v3, cosyvoice-v3-plus)
    - prefix: 音色自定义前缀,仅允许数字和小写字母,长度小于10个字符
    - url: 用于复刻音色的音频文件URL,要求公网可访问

    返回:
    - voice_id: 新创建的音色ID
    """
    try:
        service = get_voice_service()
        voice_id = service.create_voice(
            target_model=request.target_model,
            prefix=request.prefix,
            url=request.url
        )

        request_id = service.get_last_request_id()
        logger.info(f"音色创建成功 - Voice ID: {voice_id}, Request ID: {request_id}")

        return {
            "success": True,
            "voice_id": voice_id,
            "request_id": request_id,
            "message": "音色创建成功,请使用 /voice/query 查询状态"
        }
    except Exception as e:
        logger.error(f"创建音色失败: {e}")
        raise HTTPException(status_code=500, detail=f"创建音色失败: {str(e)}")


@app.post("/voice/list")
async def list_voices(request: VoiceListRequest):
    """
    查询所有音色

    参数:
    - prefix: 可选,按前缀筛选
    - page_index: 页索引,默认0
    - page_size: 页大小,默认10

    返回:
    - voices: 音色列表
    """
    try:
        service = get_voice_service()
        voices = service.list_voices(
            prefix=request.prefix,
            page_index=request.page_index,
            page_size=request.page_size
        )

        request_id = service.get_last_request_id()
        logger.info(f"查询音色列表成功 - Request ID: {request_id}, 找到 {len(voices)} 个音色")

        return {
            "success": True,
            "voices": voices,
            "count": len(voices),
            "request_id": request_id
        }
    except Exception as e:
        logger.error(f"查询音色列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询音色列表失败: {str(e)}")


@app.post("/voice/query")
async def query_voice(request: VoiceQueryRequest):
    """
    查询指定音色的详细信息

    参数:
    - voice_id: 音色ID

    返回:
    - voice_info: 音色详细信息 (包含状态、创建时间、音频链接等)
      - status: DEPLOYING(审核中), OK(可用), UNDEPLOYED(不可用)
    """
    try:
        service = get_voice_service()
        voice_info = service.query_voice(voice_id=request.voice_id)

        request_id = service.get_last_request_id()
        logger.info(f"查询音色详情成功 - Voice ID: {request.voice_id}, Status: {voice_info.get('status')}")

        return {
            "success": True,
            "voice_info": voice_info,
            "request_id": request_id
        }
    except Exception as e:
        logger.error(f"查询音色详情失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询音色详情失败: {str(e)}")


@app.post("/voice/update")
async def update_voice(request: VoiceUpdateRequest):
    """
    更新音色

    参数:
    - voice_id: 音色ID
    - url: 新的音频文件URL

    返回:
    - success: 更新是否成功
    """
    try:
        service = get_voice_service()
        service.update_voice(
            voice_id=request.voice_id,
            url=request.url
        )

        request_id = service.get_last_request_id()
        logger.info(f"音色更新成功 - Voice ID: {request.voice_id}, Request ID: {request_id}")

        return {
            "success": True,
            "request_id": request_id,
            "message": "音色更新成功,请等待审核完成"
        }
    except Exception as e:
        logger.error(f"更新音色失败: {e}")
        raise HTTPException(status_code=500, detail=f"更新音色失败: {str(e)}")


@app.post("/voice/delete")
async def delete_voice(request: VoiceDeleteRequest):
    """
    删除音色

    参数:
    - voice_id: 音色ID

    返回:
    - success: 删除是否成功
    """
    try:
        service = get_voice_service()
        service.delete_voice(voice_id=request.voice_id)

        request_id = service.get_last_request_id()
        logger.info(f"音色删除成功 - Voice ID: {request.voice_id}, Request ID: {request_id}")

        return {
            "success": True,
            "request_id": request_id,
            "message": "音色删除成功"
        }
    except Exception as e:
        logger.error(f"删除音色失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除音色失败: {str(e)}")


@app.on_event("startup")
async def warmup():
    """服务启动时预热TTS"""
    logger.info("🔥 开始TTS预热...")
    try:
        # 使用短文本预热，激活模型和连接
        warmup_text = "你好"
        chunk_count = 0
        async for chunk in synthesize_speech_stream(warmup_text):
            chunk_count += 1
        logger.info(f"✅ TTS预热完成，接收到 {chunk_count} 个音频块")
    except Exception as e:
        logger.warning(f"⚠️ TTS预热失败: {e}，服务仍可正常使用")


if __name__ == "__main__":
    # 从配置文件读取端口
    port = get_config('services.tts', 5003)

    logger.info(f"TTS服务启动在端口: {port}")

    # 启动服务
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
