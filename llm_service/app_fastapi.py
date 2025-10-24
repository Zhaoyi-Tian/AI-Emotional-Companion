"""
LLM大模型服务 - FastAPI版本
支持API和本地模型两种模式,提供流式和非流式对话
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import sys
import logging
import json
from pathlib import Path

# 添加父目录到路径以导入配置
sys.path.insert(0, str(Path(__file__).parent.parent))
from config_loader import get_config

import requests

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LLM_Service")

# 创建FastAPI应用
app = FastAPI(
    title="LLM大模型服务",
    description="支持DeepSeek API和本地模型的对话服务",
    version="1.0.0"
)

# 请求体模型
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[List[str]]] = []
    stream: Optional[bool] = True
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


class ChatResponse(BaseModel):
    success: bool
    message: str
    model: str


# ==================== DeepSeek API 模式 ====================
def build_messages_from_history(history: List[List[str]], user_msg: str) -> List[Dict[str, str]]:
    """从历史记录构建消息列表"""
    llm_config = get_config('llm')
    system_prompt = llm_config.get('api', {}).get('system_prompt', 'You are a helpful assistant.')

    messages = [{"role": "system", "content": system_prompt}]

    for user, ai in history:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": ai})

    messages.append({"role": "user", "content": user_msg})
    return messages


async def chat_with_deepseek_api_stream(message: str, history: List[List[str]],
                                       max_tokens: Optional[int] = None,
                                       temperature: Optional[float] = None):
    """使用DeepSeek API进行流式对话"""
    llm_config = get_config('llm')
    api_config = llm_config.get('api', {})

    api_key = api_config.get('api_key')
    api_url = api_config.get('api_url')
    model = api_config.get('model', 'deepseek-chat')

    if not api_key or not api_url:
        raise HTTPException(status_code=500, detail="DeepSeek API配置不完整")

    # 构建请求
    payload = {
        "model": model,
        "messages": build_messages_from_history(history, message),
        "stream": True,
        "max_tokens": max_tokens or api_config.get('max_tokens', 512),
        "temperature": temperature or api_config.get('temperature', 1.0),
        "top_p": api_config.get('top_p', 0.9)
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        with requests.post(api_url, headers=headers, json=payload, stream=True, timeout=60) as resp:
            resp.raise_for_status()

            for line in resp.iter_lines():
                if line:
                    try:
                        if line.startswith(b'data: '):
                            line = line[6:]
                        chunk = line.decode("utf-8").strip()

                        if chunk == "[DONE]":
                            break

                        data = json.loads(chunk)
                        delta = data.get("choices", [{}])[0].get("delta", {}).get("content", "")

                        if delta:
                            # 返回SSE格式
                            yield f"data: {json.dumps({'delta': delta})}\n\n"

                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        logger.error(f"处理流式响应出错: {e}")
                        continue

            # 发送结束标记
            yield f"data: {json.dumps({'done': True})}\n\n"

    except requests.exceptions.RequestException as e:
        logger.error(f"DeepSeek API请求失败: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


async def chat_with_deepseek_api(message: str, history: List[List[str]],
                                 max_tokens: Optional[int] = None,
                                 temperature: Optional[float] = None) -> str:
    """使用DeepSeek API进行非流式对话"""
    llm_config = get_config('llm')
    api_config = llm_config.get('api', {})

    api_key = api_config.get('api_key')
    api_url = api_config.get('api_url')
    model = api_config.get('model', 'deepseek-chat')

    payload = {
        "model": model,
        "messages": build_messages_from_history(history, message),
        "stream": False,
        "max_tokens": max_tokens or api_config.get('max_tokens', 512),
        "temperature": temperature or api_config.get('temperature', 1.0),
        "top_p": api_config.get('top_p', 0.9)
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        resp = requests.post(api_url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        logger.error(f"DeepSeek API请求失败: {e}")
        raise HTTPException(status_code=500, detail=f"API调用失败: {str(e)}")


# ==================== 本地模型模式 (预留) ====================
# 本地模型的实现可以根据需要添加
local_model = None
local_tokenizer = None


def init_local_model():
    """初始化本地模型"""
    # 这里可以添加本地模型加载逻辑
    # 例如加载 Qwen1.5-0.5b 或 TinyLlama
    pass


# ==================== FastAPI 路由 ====================
@app.on_event("startup")
async def startup_event():
    """服务启动事件"""
    logger.info("🚀 LLM服务正在启动...")

    llm_config = get_config('llm')
    mode = llm_config.get('mode', 'api')

    if mode == 'local':
        logger.info("使用本地模型模式")
        init_local_model()
    else:
        logger.info("使用API模式")

    logger.info("✅ LLM服务启动完成!")


@app.get("/")
async def root():
    """根路径"""
    llm_config = get_config('llm')
    return {
        "service": "LLM大模型服务",
        "status": "running",
        "mode": llm_config.get('mode'),
        "model": llm_config.get('api', {}).get('model') if llm_config.get('mode') == 'api' else llm_config.get('local', {}).get('model_name')
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "service": "llm",
        "mode": get_config('llm.mode')
    }


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    流式对话接口
    返回Server-Sent Events (SSE)流
    """
    llm_config = get_config('llm')
    mode = llm_config.get('mode', 'api')

    logger.info(f"收到流式对话请求: {request.message[:50]}...")

    if mode == 'api':
        return StreamingResponse(
            chat_with_deepseek_api_stream(
                request.message,
                request.history,
                request.max_tokens,
                request.temperature
            ),
            media_type="text/event-stream"
        )
    else:
        # 本地模型流式输出(预留)
        raise HTTPException(status_code=501, detail="本地模型流式输出暂未实现")


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    非流式对话接口
    返回完整响应
    """
    llm_config = get_config('llm')
    mode = llm_config.get('mode', 'api')

    logger.info(f"收到对话请求: {request.message[:50]}...")

    try:
        if mode == 'api':
            response_text = await chat_with_deepseek_api(
                request.message,
                request.history,
                request.max_tokens,
                request.temperature
            )
            return ChatResponse(
                success=True,
                message=response_text,
                model=llm_config.get('api', {}).get('model', 'deepseek-chat')
            )
        else:
            # 本地模型推理(预留)
            raise HTTPException(status_code=501, detail="本地模型推理暂未实现")

    except Exception as e:
        logger.error(f"对话失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload_config")
async def reload_config():
    """重新加载配置"""
    try:
        from config_loader import reload_config
        reload_config()
        return {"success": True, "message": "配置重新加载成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"配置重新加载失败: {str(e)}")


if __name__ == "__main__":
    # 从配置文件读取端口
    port = get_config('services.llm', 5002)

    logger.info(f"LLM服务启动在端口: {port}")

    # 启动服务
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
