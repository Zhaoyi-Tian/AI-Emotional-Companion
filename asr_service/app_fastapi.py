"""
ASR语音识别服务 - FastAPI版本
提供RESTful API接口进行语音识别
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import os
import sys
import tempfile
import logging
from pathlib import Path

# 添加父目录到路径以导入配置
sys.path.insert(0, str(Path(__file__).parent.parent))
from config_loader import get_config

# 导入WeNet模型
from wenet.model_CN import WeNetASRCN
from wenet.model_EN import WeNetASREN

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ASR_Service")

# 创建FastAPI应用
app = FastAPI(
    title="ASR语音识别服务",
    description="基于WeNet的语音识别服务",
    version="1.0.0"
)

# 全局ASR模型实例
asr_model = None


def init_asr_model():
    """初始化ASR模型"""
    global asr_model

    try:
        # 从配置文件读取配置
        asr_config = get_config('asr')
        model_type = asr_config.get('model_type', 'EN')

        base_dir = Path(__file__).parent

        if model_type == "EN":
            model_path = base_dir / "EN_model" / "offline_encoder.om"
            vocab_path = base_dir / "EN_model" / "vocab.txt"
            asr_model = WeNetASREN(str(model_path), str(vocab_path))
            logger.info(f"✅ ASR英文模型加载成功: {model_path}")
        elif model_type == "CN":
            model_path = base_dir / "CN_model" / "offline_encoder.om"
            vocab_path = base_dir / "CN_model" / "vocab.txt"
            asr_model = WeNetASRCN(str(model_path), str(vocab_path))
            logger.info(f"✅ ASR中文模型加载成功: {model_path}")
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

    except Exception as e:
        logger.error(f"❌ ASR模型加载失败: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """服务启动时初始化模型"""
    logger.info("🚀 ASR服务正在启动...")
    init_asr_model()
    logger.info("✅ ASR服务启动完成!")


@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "ASR语音识别服务",
        "status": "running",
        "model_type": get_config('asr.model_type')
    }


@app.get("/health")
async def health_check():
    """健康检查接口"""
    if asr_model is None:
        raise HTTPException(status_code=503, detail="ASR模型未初始化")

    return {
        "status": "healthy",
        "service": "asr",
        "model_loaded": asr_model is not None,
        "model_type": get_config('asr.model_type')
    }


@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """
    语音识别接口
    接收音频文件,返回识别文本

    Args:
        audio: 音频文件 (WAV格式, 16kHz采样率)

    Returns:
        JSON: {"text": "识别结果", "success": true}
    """
    if asr_model is None:
        raise HTTPException(status_code=503, detail="ASR模型未初始化")

    # 检查文件格式
    if not audio.filename.endswith(('.wav', '.WAV')):
        raise HTTPException(
            status_code=400,
            detail="仅支持WAV格式音频文件"
        )

    try:
        # 保存上传的音频文件到临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            content = await audio.read()
            temp_file.write(content)
            temp_path = temp_file.name

        # 调用ASR模型进行识别
        logger.info(f"正在识别音频: {audio.filename}")
        text = asr_model.transcribe(temp_path)
        logger.info(f"识别结果: {text}")

        # 删除临时文件
        os.unlink(temp_path)

        return JSONResponse(content={
            "success": True,
            "text": text,
            "filename": audio.filename
        })

    except Exception as e:
        logger.error(f"语音识别失败: {e}")
        # 清理临时文件
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        raise HTTPException(status_code=500, detail=f"识别失败: {str(e)}")


@app.post("/reload_model")
async def reload_model():
    """重新加载模型(用于配置更新后)"""
    try:
        init_asr_model()
        return {
            "success": True,
            "message": "模型重新加载成功",
            "model_type": get_config('asr.model_type')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型重新加载失败: {str(e)}")


if __name__ == "__main__":
    # 从配置文件读取端口
    port = get_config('services.asr', 5001)

    logger.info(f"ASR服务启动在端口: {port}")

    # 启动服务
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
