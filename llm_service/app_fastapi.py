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


# ==================== 本地模型模式 ====================
local_model = None
local_tokenizer = None
local_model_type = None


def set_mindspore_env():
    """设置MindSpore环境变量,防止模型加载崩溃"""
    import os
    os.environ['TE_PARALLEL_COMPILER'] = '1'
    os.environ['MAX_COMPILE_CORE_NUMBER'] = '1'
    os.environ['MS_BUILD_PROCESS_NUM'] = '1'
    os.environ['MAX_RUNTIME_CORE_NUMBER'] = '1'
    os.environ['MS_ENABLE_IO_REUSE'] = '1'
    logger.info("✅ MindSpore环境变量已设置")


def init_local_model():
    """初始化本地模型"""
    global local_model, local_tokenizer, local_model_type

    llm_config = get_config('llm')
    local_config = llm_config.get('local', {})
    model_name = local_config.get('model_name', 'qwen')

    logger.info(f"正在加载本地模型: {model_name}")

    # 设置环境变量
    set_mindspore_env()

    try:
        import mindspore
        from mindnlp.transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
        from mindspore._c_expression import disable_multi_thread
        disable_multi_thread()

        if model_name.lower() in ['qwen', 'qwen1.5-0.5b']:
            # 加载Qwen模型
            model_path = local_config.get('qwen_model_path', '/home/HwHiAiUser/.mindnlp/model/Qwen/Qwen1.5-0.5B-Chat')
            logger.info(f"加载Qwen模型: {model_path}")

            local_tokenizer = AutoTokenizer.from_pretrained(model_path, ms_dtype=mindspore.float16)
            local_model = AutoModelForCausalLM.from_pretrained(model_path, ms_dtype=mindspore.float16)
            local_model_type = 'qwen'
            logger.info("✅ Qwen1.5-0.5B模型加载成功")

        elif model_name.lower() in ['tinyllama', 'tiny']:
            # 加载TinyLlama模型
            model_path = local_config.get('tinyllama_model_path', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')
            logger.info(f"加载TinyLlama模型: {model_path}")

            local_tokenizer = AutoTokenizer.from_pretrained(model_path)
            local_model = AutoModelForCausalLM.from_pretrained(model_path, ms_dtype=mindspore.float16)
            local_model_type = 'tinyllama'
            logger.info("✅ TinyLlama模型加载成功")

        else:
            raise ValueError(f"不支持的模型: {model_name}")

    except ImportError as e:
        logger.error(f"❌ 导入失败: {e}")
        logger.error("请确保在llm环境中安装了mindspore和mindnlp")
        raise
    except Exception as e:
        logger.error(f"❌ 模型加载失败: {e}")
        raise


async def chat_with_local_model_stream(message: str, history: List[List[str]],
                                       max_tokens: Optional[int] = None,
                                       temperature: Optional[float] = None):
    """使用本地模型进行流式对话"""
    global local_model, local_tokenizer, local_model_type

    if local_model is None or local_tokenizer is None:
        raise HTTPException(status_code=503, detail="本地模型未加载")

    from mindnlp.transformers import TextIteratorStreamer
    from threading import Thread
    import mindspore

    llm_config = get_config('llm')
    local_config = llm_config.get('local', {})
    system_prompt = local_config.get('system_prompt', 'You are a helpful and friendly chatbot')

    max_new_tokens = max_tokens or local_config.get('max_tokens', 128)
    temp = temperature or local_config.get('temperature', 1.0)

    try:
        if local_model_type == 'qwen':
            # Qwen模型的输入格式
            messages = [{'role': 'system', 'content': system_prompt}]
            for user_msg, ai_msg in history:
                messages.append({'role': 'user', 'content': user_msg})
                messages.append({'role': 'assistant', 'content': ai_msg})
            messages.append({'role': 'user', 'content': message})

            input_ids = local_tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="ms",
                tokenize=True
            )

        else:  # tinyllama
            # TinyLlama的输入格式
            history_format = history + [[message, ""]]
            messages = "</s>".join(["</s>".join(["\n<|user|>:" + item[0], "\n<|assistant|>:" + item[1]])
                                   for item in history_format])
            model_inputs = local_tokenizer([messages], return_tensors="ms")
            input_ids = model_inputs['input_ids']

        # 创建流式输出
        streamer = TextIteratorStreamer(local_tokenizer, timeout=300, skip_prompt=True, skip_special_tokens=True)

        generate_kwargs = dict(
            input_ids=input_ids,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=temp,
            num_beams=1,
            use_cache=True
        )

        # 在单独线程中生成
        thread = Thread(target=local_model.generate, kwargs=generate_kwargs)
        thread.start()

        # 流式输出
        partial_message = ""
        for new_token in streamer:
            if '</s>' in new_token:
                break
            partial_message += new_token
            yield f"data: {json.dumps({'delta': new_token})}\n\n"

        yield f"data: {json.dumps({'done': True})}\n\n"

    except Exception as e:
        logger.error(f"本地模型推理失败: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


async def chat_with_local_model(message: str, history: List[List[str]],
                                max_tokens: Optional[int] = None,
                                temperature: Optional[float] = None) -> str:
    """使用本地模型进行非流式对话"""
    full_response = ""

    async for chunk in chat_with_local_model_stream(message, history, max_tokens, temperature):
        if chunk.startswith('data: '):
            data = json.loads(chunk[6:])
            if data.get('delta'):
                full_response += data['delta']
            elif data.get('error'):
                raise HTTPException(status_code=500, detail=data['error'])

    return full_response


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
    else:  # local
        return StreamingResponse(
            chat_with_local_model_stream(
                request.message,
                request.history,
                request.max_tokens,
                request.temperature
            ),
            media_type="text/event-stream"
        )


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
            model_name = llm_config.get('api', {}).get('model', 'deepseek-chat')
        else:  # local
            response_text = await chat_with_local_model(
                request.message,
                request.history,
                request.max_tokens,
                request.temperature
            )
            model_name = llm_config.get('local', {}).get('model_name', 'local-model')

        return ChatResponse(
            success=True,
            message=response_text,
            model=model_name
        )

    except Exception as e:
        logger.error(f"对话失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload_config")
async def reload_config_endpoint():
    """重新加载配置并重新初始化模型"""
    global local_model, local_tokenizer, local_model_type

    try:
        from config_loader import reload_config as reload_config_file
        reload_config_file()

        llm_config = get_config('llm')
        mode = llm_config.get('mode', 'api')

        logger.info(f"配置已重新加载,当前模式: {mode}")

        # 如果是本地模式,重新加载模型
        if mode == 'local':
            logger.info("检测到本地模式,正在重新加载模型...")

            # 清理旧模型(释放内存)
            if local_model is not None:
                logger.info("清理旧模型...")
                local_model = None
                local_tokenizer = None
                local_model_type = None

                # 强制垃圾回收
                import gc
                gc.collect()

            # 重新加载模型
            init_local_model()

            model_name = llm_config.get('local', {}).get('model_name')
            return {
                "success": True,
                "message": f"配置重新加载成功,本地模型 {model_name} 已加载",
                "mode": "local",
                "model": model_name
            }
        else:
            # API模式不需要加载模型
            return {
                "success": True,
                "message": "配置重新加载成功,使用API模式",
                "mode": "api",
                "model": llm_config.get('api', {}).get('model')
            }

    except Exception as e:
        logger.error(f"配置重新加载失败: {e}")
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
