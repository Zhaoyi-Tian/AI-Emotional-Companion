"""
Web配置管理界面
使用Gradio提供友好的配置管理、测试界面和AI对话功能
"""

import gradio as gr
import requests
import soundfile as sf
import numpy as np
from pathlib import Path
import sys
import logging
import tempfile
import subprocess
import os
import time
import base64
import cv2
import json
from datetime import datetime

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent))
from config_loader import config, get_config, set_config, reload_config

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WebUI")


# ==================== AI对话助手类 ====================
class AIAssistant:
    """AI助手核心类，整合ASR、LLM、TTS服务"""

    def __init__(self):
        self.conversation_history = []

    def speech_to_text(self, audio_file):
        """语音转文字（ASR服务）"""
        try:
            port = get_config('services.asr', 5001)
            url = f"http://localhost:{port}/transcribe"

            with open(audio_file, 'rb') as f:
                files = {'audio': f}
                response = requests.post(url, files=files, timeout=30)

            if response.status_code == 200:
                result = response.json()
                return result.get('text', '')
            else:
                logger.error(f"ASR识别失败: {response.text}")
                return None

        except Exception as e:
            logger.error(f"ASR服务调用失败: {e}")
            return None

    def text_to_speech(self, text):
        """文字转语音（TTS服务）"""
        try:
            port = get_config('services.tts', 5003)
            url = f"http://localhost:{port}/synthesize"

            payload = {"text": text}
            # 根据文本长度动态设置超时时间
            # API模式：每10字符约需1秒，最少60秒
            timeout = max(60, len(text) // 10 + 30)
            logger.info(f"TTS请求超时设置: {timeout}秒 (文本长度: {len(text)} 字符)")

            response = requests.post(url, json=payload, timeout=timeout)

            if response.status_code == 200:
                # 保存PCM音频
                with tempfile.NamedTemporaryFile(suffix='.pcm', delete=False) as f:
                    f.write(response.content)
                    pcm_path = f.name

                # 转换PCM为WAV
                wav_path = pcm_path.replace('.pcm', '.wav')
                subprocess.run([
                    'ffmpeg', '-y', '-f', 's16le', '-ar', '22050', '-ac', '1',
                    '-i', pcm_path, wav_path
                ], check=True, capture_output=True)

                os.unlink(pcm_path)
                return wav_path
            else:
                logger.error(f"TTS合成失败: {response.text}")
                return None

        except Exception as e:
            logger.error(f"TTS服务调用失败: {e}")
            return None

    def chat_stream(self, message):
        """与LLM流式对话"""
        try:
            port = get_config('services.llm', 5002)
            url = f"http://localhost:{port}/chat/stream"

            payload = {
                "message": message,
                "history": self.conversation_history
            }

            response = requests.post(url, json=payload, stream=True, timeout=60)

            if response.status_code == 200:
                full_reply = ""
                import json
                # 流式读取SSE响应
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')

                        # SSE格式: data: {"delta": "文字"}
                        if line_str.startswith('data: '):
                            json_str = line_str[6:]  # 移除 "data: " 前缀

                            try:
                                data = json.loads(json_str)

                                # 检查是否完成
                                if data.get('done'):
                                    logger.info("流式对话完成")
                                    break

                                # 提取delta内容
                                chunk = data.get('delta', '')
                                if chunk:
                                    full_reply += chunk
                                    yield chunk

                            except json.JSONDecodeError as e:
                                logger.debug(f"JSON解析失败: {json_str[:50]}")
                                continue

                # 更新对话历史 - 使用LLM服务期望的二维列表格式
                self.conversation_history.append([message, full_reply])

                logger.info(f"流式对话完成，总长度: {len(full_reply)} 字符")
                return full_reply
            else:
                error_msg = "抱歉，我遇到了一些问题，请稍后再试。"
                logger.error(f"LLM流式对话失败: {response.text}")
                yield error_msg
                return error_msg

        except Exception as e:
            error_msg = "抱歉，我遇到了一些问题，请稍后再试。"
            logger.error(f"LLM服务调用失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            yield error_msg
            return error_msg

    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []
        logger.info("对话历史已清空")

    def process_text_input(self, user_text, history):
        """处理文字输入（流式输出）"""
        if not user_text or not user_text.strip():
            return history, "", None

        # 添加用户消息到显示历史
        history = history or []
        history.append([user_text, ""])

        # 流式调用LLM获取回复
        full_reply = ""
        for chunk in self.chat_stream(user_text):
            full_reply += chunk
            # 更新显示历史（流式显示）
            history[-1][1] = full_reply
            yield history, "", None

        # 生成语音（等待完整文本）
        text_length = len(full_reply)
        # 预估TTS时间：每10字符约1秒
        estimated_time = max(5, text_length // 10)
        logger.info(f"开始生成语音，文本长度: {text_length} 字符，预计需要 {estimated_time} 秒")

        # 显示生成提示
        yield history, "", None  # 清空输入框，但暂不返回音频

        audio_path = self.text_to_speech(full_reply)

        # 返回最终结果（包含音频）
        yield history, "", audio_path

    def process_voice_input(self, audio, history):
        """处理语音输入（流式输出）"""
        if audio is None:
            yield history, "⚠️ 请先录制语音", None
            return

        try:
            # 保存音频为临时文件
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                sample_rate, audio_data = audio
                sf.write(f.name, audio_data, sample_rate)
                temp_path = f.name

            # 语音识别
            user_text = self.speech_to_text(temp_path)
            os.unlink(temp_path)

            if not user_text:
                yield history, "❌ 语音识别失败，请重试", None
                return

            logger.info(f"语音识别结果: {user_text}")

            # 添加用户消息到显示历史（只显示音频符号）
            history = history or []
            history.append(["🎤 语音消息", ""])

            # 流式调用LLM获取回复（使用识别的文字）
            full_reply = ""
            for chunk in self.chat_stream(user_text):
                full_reply += chunk
                # 更新显示历史（流式显示）
                history[-1][1] = full_reply
                yield history, "✅ 识别成功，AI回复中...", None

            # 生成语音（等待完整文本）
            text_length = len(full_reply)
            # 预估TTS时间：每10字符约1秒
            estimated_time = max(5, text_length // 10)
            logger.info(f"开始生成语音，文本长度: {text_length} 字符，预计需要 {estimated_time} 秒")

            # 显示生成提示
            yield history, f"🎵 正在生成音频中，预计需要 {estimated_time} 秒...", None

            audio_path = self.text_to_speech(full_reply)

            # 返回最终结果（包含音频）
            yield history, "✅ 识别并回复成功", audio_path

        except Exception as e:
            logger.error(f"处理语音输入失败: {e}")
            yield history, f"❌ 处理失败: {str(e)}", None


# 创建全局助手实例
assistant = AIAssistant()


# ==================== 配置管理功能 ====================
def get_current_config():
    """获取当前配置"""
    return {
        # ASR配置
        "asr_model_type": get_config('asr.model_type', 'EN'),

        # LLM配置
        "llm_mode": get_config('llm.mode', 'api'),
        "llm_api_provider": get_config('llm.api.provider', 'deepseek'),
        "llm_api_key": get_config('llm.api.api_key', ''),
        "llm_api_url": get_config('llm.api.api_url', ''),
        "llm_model": get_config('llm.api.model', 'deepseek-chat'),
        "llm_max_tokens": get_config('llm.api.max_tokens', 512),
        "llm_temperature": get_config('llm.api.temperature', 1.0),
        "llm_system_prompt": get_config('llm.api.system_prompt', ''),

        # LLM本地模型配置
        "llm_local_model_name": get_config('llm.local.model_name', 'qwen'),
        "llm_local_qwen_path": get_config('llm.local.qwen_model_path', '/home/HwHiAiUser/.mindnlp/model/Qwen/Qwen1.5-0.5B-Chat'),
        "llm_local_tinyllama_path": get_config('llm.local.tinyllama_model_path', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'),
        "llm_local_max_tokens": get_config('llm.local.max_tokens', 128),
        "llm_local_temperature": get_config('llm.local.temperature', 1.0),
        "llm_local_system_prompt": get_config('llm.local.system_prompt', 'You are a helpful and friendly chatbot'),

        # TTS配置
        "tts_mode": get_config('tts.mode', 'api'),
        "tts_api_provider": get_config('tts.api.provider', 'cosyvoice'),
        "tts_api_key": get_config('tts.api.api_key', ''),
        "tts_model": get_config('tts.api.model', 'cosyvoice-v2'),
        "tts_voice": get_config('tts.api.voice', 'longxiaochun_v2'),

        # 服务端口
        "port_orchestrator": get_config('services.orchestrator', 5000),
        "port_asr": get_config('services.asr', 5001),
        "port_llm": get_config('services.llm', 5002),
        "port_tts": get_config('services.tts', 5003),
    }


def save_asr_config(model_type):
    """保存ASR配置"""
    try:
        set_config('asr.model_type', model_type, save=True)

        # 尝试热重载ASR服务的配置
        try:
            port = get_config('services.asr', 5001)
            url = f"http://localhost:{port}/reload_config"
            response = requests.post(url, timeout=10)  # ASR重新加载模型可能需要更长时间

            if response.status_code == 200:
                return "✅ ASR配置已保存并立即生效！模型已重新加载"
            else:
                return "✅ ASR配置已保存\n⚠️ 需要重启ASR服务才能生效"
        except Exception:
            return "✅ ASR配置已保存\n⚠️ ASR服务未运行，配置将在下次启动时生效"
    except Exception as e:
        return f"❌ 保存失败: {str(e)}"


def save_llm_config(mode, provider, api_key, api_url, model, max_tokens, temperature, system_prompt,
                   local_model_name, local_qwen_path, local_tinyllama_path,
                   local_max_tokens, local_temperature, local_system_prompt):
    """保存LLM配置"""
    try:
        set_config('llm.mode', mode, save=False)

        # API配置
        set_config('llm.api.provider', provider, save=False)
        set_config('llm.api.api_key', api_key, save=False)
        set_config('llm.api.api_url', api_url, save=False)
        set_config('llm.api.model', model, save=False)
        set_config('llm.api.max_tokens', int(max_tokens), save=False)
        set_config('llm.api.temperature', float(temperature), save=False)
        set_config('llm.api.system_prompt', system_prompt, save=False)

        # 本地模型配置
        set_config('llm.local.model_name', local_model_name, save=False)
        set_config('llm.local.qwen_model_path', local_qwen_path, save=False)
        set_config('llm.local.tinyllama_model_path', local_tinyllama_path, save=False)
        set_config('llm.local.max_tokens', int(local_max_tokens), save=False)
        set_config('llm.local.temperature', float(local_temperature), save=False)
        set_config('llm.local.system_prompt', local_system_prompt, save=True)

        # 尝试热重载LLM服务的配置
        try:
            port = get_config('services.llm', 5002)
            url = f"http://localhost:{port}/reload_config"
            response = requests.post(url, timeout=30)  # LLM重新加载模型可能需要更长时间

            if response.status_code == 200:
                result_data = response.json()
                if result_data.get('success'):
                    msg = result_data.get('message', 'LLM配置已重新加载')
                    return f"✅ LLM配置已保存并立即生效！\n{msg}"
                else:
                    return "✅ LLM配置已保存\n⚠️ 配置热重载失败，可能需要重启LLM服务"
            else:
                return "✅ LLM配置已保存\n⚠️ 如果切换了模式或本地模型，请重启LLM服务使配置生效"
        except Exception:
            return "✅ LLM配置已保存\n⚠️ LLM服务未运行，配置将在下次启动时生效"
    except Exception as e:
        return f"❌ 保存失败: {str(e)}"


def save_tts_config(provider, api_key, model, voice):
    """保存TTS配置（仅支持API模式）"""
    try:
        set_config('tts.mode', 'api', save=False)  # 固定为API模式
        set_config('tts.api.provider', provider, save=False)
        set_config('tts.api.api_key', api_key, save=False)
        set_config('tts.api.model', model, save=False)
        set_config('tts.api.voice', voice, save=True)

        # 尝试热重载TTS服务的配置
        try:
            port = get_config('services.tts', 5003)
            url = f"http://localhost:{port}/reload_config"
            response = requests.post(url, timeout=10)

            if response.status_code == 200:
                return "✅ TTS配置已保存并立即生效！"
            else:
                return "✅ TTS配置已保存\n⚠️ 需要重启TTS服务才能生效"
        except Exception:
            return "✅ TTS配置已保存\n⚠️ TTS服务未运行，配置将在下次启动时生效"
    except Exception as e:
        return f"❌ 保存失败: {str(e)}"


def reload_all_services():
    """重新加载所有服务配置"""
    try:
        reload_config()
        ports = get_config('services')

        # 尝试重新加载各服务配置
        results = []
        services = {
            'ASR': f"http://localhost:{ports['asr']}/reload_config",
            'LLM': f"http://localhost:{ports['llm']}/reload_config",
            'TTS': f"http://localhost:{ports['tts']}/reload_config",
            'Orchestrator': f"http://localhost:{ports['orchestrator']}/reload_config",
            'VoiceChat': f"http://localhost:{ports['voice_chat']}/reload_config",
            'YOLO': f"http://localhost:{ports['yolo']}/reload_config"
        }

        for name, url in services.items():
            try:
                response = requests.post(url, timeout=5)
                if response.status_code == 200:
                    result_data = response.json()
                    if result_data.get('success', True):
                        results.append(f"✅ {name}服务配置已重新加载")
                    else:
                        results.append(f"⚠️ {name}服务重新加载失败: {result_data.get('message', result_data.get('error', '未知错误'))}")
                else:
                    results.append(f"⚠️ {name}服务重新加载失败 (HTTP {response.status_code})")
            except Exception as e:
                results.append(f"❌ {name}服务不可达: {str(e)}")

        return "\n".join(results)
    except Exception as e:
        return f"❌ 重新加载失败: {str(e)}"


def reload_llm_service():
    """单独重新加载LLM服务(用于模型切换)"""
    try:
        reload_config()
        port = get_config('services.llm', 5002)
        url = f"http://localhost:{port}/reload_config"

        response = requests.post(url, timeout=30)  # 本地模型加载需要更长时间

        if response.status_code == 200:
            result = response.json()
            mode = get_config('llm.mode')
            if mode == 'local':
                model_name = get_config('llm.local.model_name')
                return f"✅ LLM服务已重新加载\n\n模式: 本地模型\n模型: {model_name}\n\n⚠️ 本地模型加载需要30-60秒,请耐心等待..."
            else:
                return f"✅ LLM服务已重新加载\n\n模式: API\n模型: {get_config('llm.api.model')}"
        else:
            return f"❌ LLM服务重新加载失败: HTTP {response.status_code}"

    except requests.exceptions.Timeout:
        return "⚠️ 请求超时\n\n本地模型加载时间较长,请稍后在'服务状态'页面检查LLM服务状态"
    except Exception as e:
        return f"❌ 重新加载失败: {str(e)}"


# ==================== 服务测试功能 ====================
def test_asr_service(audio):
    """测试ASR服务"""
    if audio is None:
        return "⚠️ 请先录制或上传音频"

    try:
        port = get_config('services.asr', 5001)
        url = f"http://localhost:{port}/transcribe"

        # 保存音频为临时文件
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            # audio是(sample_rate, audio_data)元组
            sample_rate, audio_data = audio
            sf.write(f.name, audio_data, sample_rate)
            temp_path = f.name

        # 发送请求
        with open(temp_path, 'rb') as f:
            files = {'audio': f}
            response = requests.post(url, files=files, timeout=30)

        import os
        os.unlink(temp_path)

        if response.status_code == 200:
            result = response.json()
            return f"✅ 识别成功!\n\n识别结果: {result.get('text', '')}"
        else:
            return f"❌ 识别失败: {response.text}"

    except Exception as e:
        return f"❌ 测试失败: {str(e)}"


def test_llm_service(text):
    """测试LLM服务"""
    if not text:
        return "⚠️ 请输入测试文本"

    try:
        port = get_config('services.llm', 5002)
        url = f"http://localhost:{port}/chat"

        payload = {
            "message": text,
            "history": []
        }

        response = requests.post(url, json=payload, timeout=60)

        if response.status_code == 200:
            result = response.json()
            return f"✅ 对话成功!\n\n回复: {result.get('message', '')}"
        else:
            return f"❌ 对话失败: {response.text}"

    except Exception as e:
        return f"❌ 测试失败: {str(e)}"


def test_tts_service(text):
    """测试TTS服务"""
    if not text:
        return "⚠️ 请输入测试文本", None

    try:
        port = get_config('services.tts', 5003)
        url = f"http://localhost:{port}/synthesize"

        payload = {"text": text}
        response = requests.post(url, json=payload, timeout=30)

        if response.status_code == 200:
            # 保存音频
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.pcm', delete=False) as f:
                f.write(response.content)
                audio_path = f.name

            # 转换PCM为WAV
            import subprocess
            wav_path = audio_path.replace('.pcm', '.wav')
            subprocess.run([
                'ffmpeg', '-y', '-f', 's16le', '-ar', '22050', '-ac', '1',
                '-i', audio_path, wav_path
            ], check=True, capture_output=True)

            import os
            os.unlink(audio_path)

            return f"✅ 合成成功!", wav_path
        else:
            return f"❌ 合成失败: {response.text}", None

    except Exception as e:
        return f"❌ 测试失败: {str(e)}", None


# ==================== 音色克隆功能 ====================
def create_voice_enrollment(target_model, prefix, audio_url):
    """创建音色克隆"""
    if not target_model or not prefix or not audio_url:
        return "⚠️ 请填写所有必填项", ""

    try:
        port = get_config('services.tts', 5003)
        url = f"http://localhost:{port}/voice/create"

        payload = {
            "target_model": target_model,
            "prefix": prefix,
            "url": audio_url
        }

        response = requests.post(url, json=payload, timeout=180)  # 音色创建需要较长时间,设置3分钟超时

        if response.status_code == 200:
            result = response.json()
            voice_id = result.get('voice_id', '')
            return f"✅ 音色创建成功!\n\nVoice ID: {voice_id}\n\n请使用下方的'查询音色状态'功能查看审核进度", voice_id
        else:
            return f"❌ 创建失败: {response.text}", ""

    except Exception as e:
        return f"❌ 创建失败: {str(e)}", ""


def query_voice_status(voice_id):
    """查询音色状态"""
    if not voice_id:
        return "⚠️ 请输入Voice ID"

    try:
        port = get_config('services.tts', 5003)
        url = f"http://localhost:{port}/voice/query"

        payload = {"voice_id": voice_id}
        response = requests.post(url, json=payload, timeout=10)

        if response.status_code == 200:
            result = response.json()
            voice_info = result.get('voice_info', {})

            status = voice_info.get('status', 'UNKNOWN')
            status_emoji = {
                'OK': '✅',
                'DEPLOYING': '⏳',
                'UNDEPLOYED': '❌'
            }.get(status, '❓')

            status_text = {
                'OK': '审核通过,可以使用',
                'DEPLOYING': '审核中,请稍候',
                'UNDEPLOYED': '审核未通过,无法使用'
            }.get(status, '未知状态')

            info = f"{status_emoji} 音色状态: {status_text}\n\n"
            info += f"Voice ID: {voice_id}\n"
            info += f"创建时间: {voice_info.get('gmt_create', 'N/A')}\n"
            info += f"修改时间: {voice_info.get('gmt_modified', 'N/A')}\n"
            info += f"目标模型: {voice_info.get('target_model', 'N/A')}\n"
            info += f"音频链接: {voice_info.get('resource_link', 'N/A')}\n"

            if status == 'OK':
                info += "\n✅ 该音色已可用,可以在TTS配置中使用该Voice ID作为发音人"

            return info
        else:
            return f"❌ 查询失败: {response.text}"

    except Exception as e:
        return f"❌ 查询失败: {str(e)}"


def list_all_voices(prefix, page_index, page_size):
    """列出所有音色"""
    try:
        port = get_config('services.tts', 5003)
        url = f"http://localhost:{port}/voice/list"

        payload = {
            "prefix": prefix if prefix else None,
            "page_index": int(page_index),
            "page_size": int(page_size)
        }

        response = requests.post(url, json=payload, timeout=10)

        if response.status_code == 200:
            result = response.json()
            voices = result.get('voices', [])
            count = result.get('count', 0)

            if count == 0:
                return "📋 未找到任何音色"

            info = f"📋 找到 {count} 个音色:\n\n"
            for i, voice in enumerate(voices, 1):
                status = voice.get('status', 'UNKNOWN')
                status_emoji = {
                    'OK': '✅',
                    'DEPLOYING': '⏳',
                    'UNDEPLOYED': '❌'
                }.get(status, '❓')

                info += f"{i}. {status_emoji} {voice.get('voice_id', 'N/A')}\n"
                info += f"   状态: {status}\n"
                info += f"   创建时间: {voice.get('gmt_create', 'N/A')}\n\n"

            return info
        else:
            return f"❌ 查询失败: {response.text}"

    except Exception as e:
        return f"❌ 查询失败: {str(e)}"


def update_voice_enrollment(voice_id, new_audio_url):
    """更新音色"""
    if not voice_id or not new_audio_url:
        return "⚠️ 请填写Voice ID和新音频URL"

    try:
        port = get_config('services.tts', 5003)
        url = f"http://localhost:{port}/voice/update"

        payload = {
            "voice_id": voice_id,
            "url": new_audio_url
        }

        response = requests.post(url, json=payload, timeout=30)

        if response.status_code == 200:
            return "✅ 音色更新成功!\n\n请等待审核完成,使用'查询音色状态'查看进度"
        else:
            return f"❌ 更新失败: {response.text}"

    except Exception as e:
        return f"❌ 更新失败: {str(e)}"


def delete_voice_enrollment(voice_id):
    """删除音色"""
    if not voice_id:
        return "⚠️ 请输入Voice ID"

    try:
        port = get_config('services.tts', 5003)
        url = f"http://localhost:{port}/voice/delete"

        payload = {"voice_id": voice_id}
        response = requests.post(url, json=payload, timeout=10)

        if response.status_code == 200:
            return "✅ 音色删除成功!"
        else:
            return f"❌ 删除失败: {response.text}"

    except Exception as e:
        return f"❌ 删除失败: {str(e)}"


def check_services_health():
    """检查所有服务健康状态"""
    try:
        ports = get_config('services')

        # 1. 检查 Orchestrator 及其管理的服务 (ASR, LLM, TTS)
        orchestrator_port = ports.get('orchestrator', 5000)
        orchestrator_url = f"http://localhost:{orchestrator_port}/health"

        status_text = "🔍 服务健康状态:\n\n"
        status_text += "=" * 40 + "\n"

        try:
            response = requests.get(orchestrator_url, timeout=5)
            if response.status_code == 200:
                result = response.json()
                services = result.get('services', {})

                status_text += "📡 核心服务:\n"
                for name, status in services.items():
                    emoji = "✅" if status == "healthy" else "❌"
                    status_text += f"  {emoji} {name.upper()}: {status}\n"
            else:
                status_text += "❌ Orchestrator 服务异常\n"
        except Exception as e:
            status_text += f"❌ Orchestrator 服务不可达: {str(e)[:50]}\n"

        status_text += "\n" + "=" * 40 + "\n"

        # 2. 检查 Voice Chat 服务
        voice_chat_port = ports.get('voice_chat', 5004)
        voice_chat_url = f"http://localhost:{voice_chat_port}/health"

        status_text += "🎤 语音对话服务:\n"
        try:
            response = requests.get(voice_chat_url, timeout=5)
            if response.status_code == 200:
                result = response.json()
                service_status = result.get('status', 'unknown')
                running = result.get('running', False)
                enabled = result.get('enabled', False)

                if service_status == "healthy":
                    status_text += "  ✅ 服务状态: 正常运行\n"

                    # 显示详细状态
                    if running:
                        status_text += "  🟢 对话状态: 正在运行\n"
                    else:
                        status_text += "  ⚪ 对话状态: 已停止\n"

                    if enabled:
                        status_text += "  🔛 自动启动: 已启用\n"
                    else:
                        status_text += "  🔘 自动启动: 已禁用\n"
                else:
                    status_text += f"  ⚠️ 服务状态: {service_status}\n"
            else:
                status_text += "  ❌ 服务异常 (无法连接)\n"
        except requests.exceptions.ConnectionError:
            status_text += "  ❌ 服务未启动\n"
        except Exception as e:
            status_text += f"  ❌ 服务不可达: {str(e)[:50]}\n"

        status_text += "\n" + "=" * 40 + "\n"

        # 3. 检查 YOLO 检测服务
        yolo_port = ports.get('yolo', 5005)
        yolo_url = f"http://localhost:{yolo_port}/health"

        status_text += "📹 YOLO检测服务:\n"
        try:
            response = requests.get(yolo_url, timeout=5)
            if response.status_code == 200:
                result = response.json()
                service_status = result.get('status', 'unknown')
                model_loaded = result.get('model_loaded', False)

                if service_status == "healthy":
                    status_text += "  ✅ 服务状态: 正常运行\n"
                    status_text += f"  {'✅' if model_loaded else '❌'} 模型加载: {'已加载' if model_loaded else '未加载'}\n"
                else:
                    status_text += f"  ❌ 服务状态: {service_status}\n"
            else:
                status_text += "  ❌ 服务异常 (无法连接)\n"
        except requests.exceptions.ConnectionError:
            status_text += "  ❌ 服务未启动\n"
        except Exception as e:
            status_text += f"  ❌ 服务不可达: {str(e)[:50]}\n"

        status_text += "\n" + "=" * 40 + "\n"

        # 4. 检查 Web UI (自身)
        status_text += "🌐 Web 配置界面:\n"
        status_text += "  ✅ 服务状态: 正常运行 (当前)\n"

        status_text += "\n💡 提示:\n"
        status_text += "  • 如果服务显示异常，请运行 python start_all.py 启动服务\n"
        status_text += "  • 语音对话服务可在 '🎤 语音对话' 标签页控制启动/停止\n"
        status_text += "  • YOLO检测服务可在 '📹 YOLO检测' 标签页控制启动/停止\n"

        return status_text

    except Exception as e:
        return f"❌ 检查失败: {str(e)}\n\n请确保所有服务已启动"






# ==================== 语音对话功能 ====================
def get_voice_devices():
    """获取音频设备列表"""
    try:
        port = get_config('services.voice_chat', 5004)
        url = f"http://localhost:{port}/devices"
        response = requests.get(url, timeout=5)

        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                devices = result.get('devices', [])
                # 返回输入设备和输出设备
                input_devices = [(f"{d['index']}: {d['name']}", d['index']) for d in devices if d['max_input_channels'] > 0]
                output_devices = [(f"{d['index']}: {d['name']}", d['index']) for d in devices if d['max_output_channels'] > 0]
                return input_devices, output_devices
        return [], []
    except Exception as e:
        logger.error(f"获取音频设备失败: {e}")
        return [], []


def save_voice_chat_config(enable, wake_mode, wake_words, wake_reply, interrupt_mode, interrupt_words, interrupt_reply, thinking_reply, input_device, output_device, volume, silence_threshold, silence_duration, min_audio_length, continue_timeout):
    """保存语音对话配置"""
    try:
        # 解析唤醒词（按逗号分隔）
        wake_words_list = [w.strip() for w in wake_words.split(',') if w.strip()]

        # 解析打断词（按逗号分隔）
        interrupt_words_list = [w.strip() for w in interrupt_words.split(',') if w.strip()]

        # 保存配置（除最后一个外都不立即保存到文件）
        set_config('voice_chat.enable', enable, save=False)
        set_config('voice_chat.wake_mode', wake_mode, save=False)
        set_config('voice_chat.wake_words', wake_words_list, save=False)
        set_config('voice_chat.wake_reply', wake_reply, save=False)
        set_config('voice_chat.interrupt_mode', interrupt_mode, save=False)
        set_config('voice_chat.interrupt_words', interrupt_words_list, save=False)
        set_config('voice_chat.interrupt_reply', interrupt_reply, save=False)
        set_config('voice_chat.thinking_reply', thinking_reply, save=False)
        set_config('voice_chat.input_device', input_device if input_device != -1 else None, save=False)
        set_config('voice_chat.output_device', output_device if output_device != -1 else None, save=False)
        set_config('voice_chat.output_volume', int(volume), save=False)
        set_config('voice_chat.silence_threshold', int(silence_threshold), save=False)
        set_config('voice_chat.silence_duration', float(silence_duration), save=False)
        set_config('voice_chat.min_audio_length', float(min_audio_length), save=False)
        set_config('voice_chat.continue_dialogue_timeout', float(continue_timeout), save=True)  # 最后一个才保存到文件

        # 尝试热重载语音对话服务的配置
        reload_result = ""
        try:
            port = get_config('services.voice_chat', 5004)
            url = f"http://localhost:{port}/reload_config"
            response = requests.post(url, timeout=5)

            if response.status_code == 200:
                result_data = response.json()
                if result_data.get('success'):
                    reload_result = "\n\n✅ 语音对话服务配置已热重载！配置立即生效"
                    if 'changes' in result_data:
                        changes = result_data['changes']
                        reload_result += "\n\n📊 当前配置:"
                        reload_result += f"\n   🔊 静音阈值: {changes.get('silence_threshold')}"
                        reload_result += f"\n   🔉 输出音量: {changes.get('output_volume')}%"
                        reload_result += f"\n   🎙️ 唤醒模式: {'启用' if changes.get('wake_mode') else '禁用'}"
                        if changes.get('wake_words'):
                            reload_result += f"\n   📢 唤醒词: {', '.join(changes.get('wake_words', []))}"
                        if changes.get('wake_reply'):
                            reload_result += f"\n   💬 唤醒回复: {changes.get('wake_reply')}"
                        reload_result += f"\n   🛑 打断模式: {'启用' if changes.get('interrupt_mode') else '禁用'}"
                        if changes.get('interrupt_words'):
                            reload_result += f"\n   ⏸️ 打断词: {', '.join(changes.get('interrupt_words', []))}"
                        if changes.get('interrupt_reply'):
                            reload_result += f"\n   💬 打断回复: {changes.get('interrupt_reply')}"
                    if '音频设备' in result_data.get('message', ''):
                        reload_result += "\n\n⚠️ 音频设备配置需要重启语音对话才能生效"
                else:
                    reload_result = f"\n\n⚠️ 配置热重载失败，需要重启语音对话服务: {result_data.get('error', '未知错误')}"
            else:
                reload_result = "\n\n⚠️ 无法热重载配置，请重启语音对话服务"
        except Exception as e:
            reload_result = f"\n\n⚠️ 语音对话服务未运行，配置将在下次启动时生效"

        return f"✅ 配置已保存!{reload_result}"
    except Exception as e:
        return f"❌ 保存失败: {str(e)}"


def start_voice_chat():
    """启动语音对话服务"""
    try:
        port = get_config('services.voice_chat', 5004)
        url = f"http://localhost:{port}/start"
        response = requests.post(url, timeout=5)

        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                return "✅ 语音对话服务已启动"
            else:
                return f"⚠️ {result.get('message', '未知错误')}"
        return f"❌ 启动失败: HTTP {response.status_code}"
    except Exception as e:
        return f"❌ 启动失败: {str(e)}"


def stop_voice_chat():
    """停止语音对话服务"""
    try:
        port = get_config('services.voice_chat', 5004)
        url = f"http://localhost:{port}/stop"
        response = requests.post(url, timeout=5)

        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                return "✅ 语音对话服务已停止"
            else:
                return f"⚠️ {result.get('message', '未知错误')}"
        return f"❌ 停止失败: HTTP {response.status_code}"
    except Exception as e:
        return f"❌ 停止失败: {str(e)}"


def get_voice_chat_status():
    """获取语音对话状态（详细版）"""
    try:
        port = get_config('services.voice_chat', 5004)

        # 检查服务健康状态
        health_url = f"http://localhost:{port}/health"
        try:
            health_response = requests.get(health_url, timeout=3)
            if health_response.status_code == 200:
                health_data = health_response.json()
                service_status = health_data.get('status', 'unknown')
                running = health_data.get('running', False)
                enabled = health_data.get('enabled', False)

                # 构建详细状态信息
                status_text = "📊 语音对话服务详细状态\n\n"
                status_text += "=" * 35 + "\n"

                # 服务状态
                if service_status == "healthy":
                    status_text += "✅ 服务状态: 正常运行\n"
                else:
                    status_text += f"⚠️ 服务状态: {service_status}\n"

                # 对话运行状态
                if running:
                    status_text += "🟢 对话状态: 正在运行\n"
                else:
                    status_text += "⚪ 对话状态: 已停止\n"

                # 自动启动配置
                if enabled:
                    status_text += "🔛 自动启动: 已启用\n"
                else:
                    status_text += "🔘 自动启动: 已禁用\n"

                status_text += "=" * 35 + "\n\n"

                # 获取当前配置
                voice_config = get_config('voice_chat')
                status_text += "⚙️ 当前配置:\n"
                status_text += f"  静音阈值: {voice_config.get('silence_threshold', 'N/A')}\n"
                status_text += f"  输出音量: {voice_config.get('output_volume', 'N/A')}%\n"
                status_text += f"  唤醒模式: {'启用' if voice_config.get('wake_mode') else '禁用'}\n"
                status_text += f"  打断模式: {'启用' if voice_config.get('interrupt_mode') else '禁用'}\n"

                status_text += "\n💡 提示:\n"
                if not running:
                    status_text += "  • 点击'启动语音对话'按钮开始使用\n"
                else:
                    status_text += "  • 语音对话正在运行中\n"
                    status_text += "  • 可以点击'停止语音对话'按钮暂停\n"

                return status_text
            else:
                return "❌ 服务异常: 无法获取健康状态"
        except requests.exceptions.ConnectionError:
            return "❌ 语音对话服务未启动\n\n💡 请在终端运行:\n  python start_all.py\n或单独启动:\n  python voice_chat.py"
        except Exception as e:
            return f"❌ 连接服务失败: {str(e)[:50]}"

    except Exception as e:
        return f"❌ 获取状态失败: {str(e)}"


def restart_voice_chat():
    """重启语音对话服务"""
    try:
        # 先停止
        stop_result = stop_voice_chat()
        if "失败" in stop_result and "未在运行" not in stop_result:
            return stop_result

        # 等待一秒
        import time
        time.sleep(1)

        # 再启动
        start_result = start_voice_chat()
        return start_result
    except Exception as e:
        return f"❌ 重启失败: {str(e)}"


def start_volume_monitor():
    """启动音量监测"""
    try:
        port = get_config('services.voice_chat', 5004)
        url = f"http://localhost:{port}/volume/start"

        # 发送POST请求，持续10秒
        response = requests.post(url, json={"duration": 10}, timeout=5)

        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                return (
                    "🎤 音量监测已启动，持续10秒...\n请保持安静，不要说话！",
                    0, 0, 0, 0, 0  # 重置所有数值显示
                )
            else:
                return (
                    f"⚠️ {result.get('message', '未知错误')}",
                    0, 0, 0, 0, 0
                )
        return (
            f"❌ 启动失败: HTTP {response.status_code}",
            0, 0, 0, 0, 0
        )
    except Exception as e:
        return (
            f"❌ 启动失败: {str(e)}",
            0, 0, 0, 0, 0
        )


def stop_volume_monitor():
    """停止音量监测"""
    try:
        port = get_config('services.voice_chat', 5004)
        url = f"http://localhost:{port}/volume/stop"
        response = requests.post(url, timeout=5)

        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                return "✅ 音量监测已停止"
            else:
                return f"⚠️ {result.get('message', '未知错误')}"
        return f"❌ 停止失败: HTTP {response.status_code}"
    except Exception as e:
        return f"❌ 停止失败: {str(e)}"


def get_volume_data():
    """获取音量监测数据"""
    try:
        port = get_config('services.voice_chat', 5004)
        url = f"http://localhost:{port}/volume/data"
        response = requests.get(url, timeout=2)

        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                data = result.get('data', {})
                running = result.get('running', False)

                # 返回状态和所有数据
                if running:
                    status = f"⏳ 监测中... (已采集 {data.get('sample_count', 0)} 个样本)"
                elif data.get('sample_count', 0) > 0:
                    status = f"✅ 监测完成！共采集 {data.get('sample_count', 0)} 个样本\n\n💡 推荐将下方\"静音阈值\"设置为: {data.get('recommended_threshold', 0)}"
                else:
                    status = "未开始监测"

                return (
                    status,
                    data.get('current_rms', 0),
                    data.get('avg_rms', 0),
                    data.get('min_rms', 0),
                    data.get('max_rms', 0),
                    data.get('recommended_threshold', 0)
                )

        return ("❌ 无法连接到服务", 0, 0, 0, 0, 0)
    except Exception as e:
        return (f"❌ 获取数据失败: {str(e)}", 0, 0, 0, 0, 0)


def refresh_devices():
    """刷新设备列表"""
    input_devices, output_devices = get_voice_devices()
    # 添加"默认设备"选项
    input_choices = [("默认设备", -1)] + input_devices
    output_choices = [("默认设备", -1)] + output_devices

    # 获取当前配置的设备
    current_input = get_config('voice_chat.input_device', None)
    current_output = get_config('voice_chat.output_device', None)

    # 检测蓝牙设备
    bluetooth_devices = []
    for name, idx in output_devices:
        name_lower = name.lower()
        if 'bluez' in name_lower or 'bluetooth' in name_lower or 'bt' in name_lower:
            bluetooth_devices.append(name)

    status_msg = "✅ 设备列表已刷新\n"
    if bluetooth_devices:
        status_msg += f"\n🔵 检测到 {len(bluetooth_devices)} 个蓝牙设备：\n"
        for dev in bluetooth_devices:
            status_msg += f"  • {dev}\n"
    else:
        status_msg += "\n⚠️ 未检测到蓝牙设备，请确保蓝牙音箱已连接"

    return (
        gr.Dropdown(choices=input_choices, value=current_input if current_input is not None else -1),
        gr.Dropdown(choices=output_choices, value=current_output if current_output is not None else -1),
        status_msg
    )


def check_bluetooth_status():
    """检查系统蓝牙连接状态（包括PulseAudio）"""
    status_msg = ""

    # 1. 检查PulseAudio蓝牙音频设备
    try:
        result = subprocess.run(
            ['pactl', 'list', 'sinks', 'short'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            sinks = result.stdout.strip().split('\n')
            bluetooth_sinks = [s for s in sinks if 'bluez' in s.lower() or 'bluetooth' in s.lower()]

            if bluetooth_sinks:
                status_msg += f"🔵 PulseAudio检测到 {len(bluetooth_sinks)} 个蓝牙音频设备：\n\n"
                for sink in bluetooth_sinks:
                    parts = sink.split('\t')
                    if len(parts) >= 2:
                        sink_name = parts[1]
                        # 尝试获取设备描述
                        desc_result = subprocess.run(
                            ['pactl', 'list', 'sinks'],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        if desc_result.returncode == 0:
                            for line in desc_result.stdout.split('\n'):
                                if sink_name in line:
                                    # 找到对应设备，获取描述
                                    for desc_line in desc_result.stdout.split('\n'):
                                        if 'Description:' in desc_line:
                                            desc = desc_line.split('Description:')[1].strip()
                                            status_msg += f"  • {desc} ({sink_name})\n"
                                            break
                                    break
                status_msg += "\n"
            else:
                status_msg += "⚠️ PulseAudio未检测到蓝牙音频设备\n\n"

    except FileNotFoundError:
        status_msg += "⚠️ 未找到pactl命令，无法检查PulseAudio设备\n\n"
    except Exception as e:
        status_msg += f"⚠️ PulseAudio检查失败: {str(e)}\n\n"

    # 2. 检查bluetoothctl连接状态
    try:
        # 使用info命令检查所有设备
        result = subprocess.run(
            ['bluetoothctl', 'devices'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            devices = result.stdout.strip().split('\n')
            device_list = [d for d in devices if d.strip() and d.startswith('Device')]

            if device_list:
                status_msg += f"🔵 蓝牙配对设备 ({len(device_list)} 个)：\n\n"

                # 检查每个设备的连接状态
                connected_count = 0
                for device in device_list:
                    # 提取MAC地址
                    parts = device.split()
                    if len(parts) >= 3:
                        mac = parts[1]
                        name = ' '.join(parts[2:])

                        # 检查连接状态
                        info_result = subprocess.run(
                            ['bluetoothctl', 'info', mac],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )

                        is_connected = 'Connected: yes' in info_result.stdout
                        if is_connected:
                            status_msg += f"  ✅ {name} (已连接)\n"
                            connected_count += 1
                        else:
                            status_msg += f"  ⚪ {name} (未连接)\n"

                status_msg += f"\n已连接设备: {connected_count}/{len(device_list)}\n"
            else:
                status_msg += "⚠️ 未找到配对的蓝牙设备\n"

    except FileNotFoundError:
        status_msg += "⚠️ 未找到bluetoothctl命令\n请安装bluez工具包：sudo apt install bluez\n"
    except Exception as e:
        status_msg += f"⚠️ 蓝牙检查失败: {str(e)}\n"

    if not status_msg:
        status_msg = "❌ 无法获取蓝牙状态"

    return status_msg


def set_default_audio_sink():
    """将蓝牙音箱设为系统默认音频输出（使用PulseAudio）"""
    try:
        # 获取所有音频输出设备
        result = subprocess.run(
            ['pactl', 'list', 'sinks', 'short'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode != 0:
            return "❌ 无法获取音频设备列表"

        sinks = result.stdout.strip().split('\n')
        bluetooth_sinks = [s for s in sinks if 'bluez' in s.lower()]

        if not bluetooth_sinks:
            return "⚠️ 未检测到蓝牙音频设备\n\n请确保蓝牙音箱已连接并在PulseAudio中可见"

        # 获取第一个蓝牙设备的名称
        sink_name = bluetooth_sinks[0].split('\t')[1] if '\t' in bluetooth_sinks[0] else bluetooth_sinks[0].split()[1]

        # 设置为默认输出设备
        set_result = subprocess.run(
            ['pactl', 'set-default-sink', sink_name],
            capture_output=True,
            text=True,
            timeout=5
        )

        if set_result.returncode == 0:
            # 获取设备描述
            desc_result = subprocess.run(
                ['pactl', 'list', 'sinks'],
                capture_output=True,
                text=True,
                timeout=5
            )

            device_desc = "蓝牙音箱"
            if desc_result.returncode == 0:
                lines = desc_result.stdout.split('\n')
                for i, line in enumerate(lines):
                    if sink_name in line:
                        for j in range(i, min(i+20, len(lines))):
                            if 'Description:' in lines[j]:
                                device_desc = lines[j].split('Description:')[1].strip()
                                break
                        break

            return f"✅ 已将默认音频输出设置为：{device_desc}\n\nSink: {sink_name}\n\n现在所有音频（包括语音对话）都会通过蓝牙音箱播放"
        else:
            return f"❌ 设置默认输出失败: {set_result.stderr}"

    except FileNotFoundError:
        return "❌ 未找到pactl命令\n\n请安装PulseAudio工具：sudo apt install pulseaudio-utils"
    except Exception as e:
        return f"❌ 操作失败: {str(e)}"


def set_audio_volume(volume):
    """设置音频输出音量（使用PulseAudio）"""
    try:
        volume = int(volume)
        if volume < 0 or volume > 100:
            return "❌ 音量必须在 0-100 之间"

        # 获取默认输出设备
        result = subprocess.run(
            ['pactl', 'info'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode != 0:
            return "❌ 无法获取音频设备信息"

        # 提取默认sink
        default_sink = None
        for line in result.stdout.split('\n'):
            if 'Default Sink:' in line:
                default_sink = line.split('Default Sink:')[1].strip()
                break

        if not default_sink:
            return "❌ 未找到默认音频输出设备\n\n请先设置默认输出设备"

        # 设置音量（PulseAudio使用百分比）
        set_result = subprocess.run(
            ['pactl', 'set-sink-volume', default_sink, f'{volume}%'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if set_result.returncode == 0:
            # 保存到配置文件
            set_config('voice_chat.output_volume', volume, save=True)

            return f"✅ 音量已设置为 {volume}%\n\n设备: {default_sink}"
        else:
            return f"❌ 设置音量失败: {set_result.stderr}"

    except ValueError:
        return "❌ 无效的音量值"
    except FileNotFoundError:
        return "❌ 未找到pactl命令\n\n请安装PulseAudio工具：sudo apt install pulseaudio-utils"
    except Exception as e:
        return f"❌ 操作失败: {str(e)}"


def get_current_volume():
    """获取当前音频输出音量"""
    try:
        # 获取默认输出设备
        result = subprocess.run(
            ['pactl', 'info'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode != 0:
            return 100  # 默认返回100%

        # 提取默认sink
        default_sink = None
        for line in result.stdout.split('\n'):
            if 'Default Sink:' in line:
                default_sink = line.split('Default Sink:')[1].strip()
                break

        if not default_sink:
            return 100

        # 获取音量
        volume_result = subprocess.run(
            ['pactl', 'list', 'sinks'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if volume_result.returncode == 0:
            lines = volume_result.stdout.split('\n')
            in_target_sink = False
            for line in lines:
                if default_sink in line:
                    in_target_sink = True
                if in_target_sink and 'Volume:' in line:
                    # 提取百分比，例如：Volume: front-left: 65536 / 100% / 0.00 dB
                    parts = line.split('/')
                    if len(parts) >= 2:
                        volume_str = parts[1].strip().replace('%', '')
                        try:
                            return int(volume_str)
                        except:
                            return 100
                    break

        return 100

    except Exception as e:
        logger.error(f"获取音量失败: {e}")
        return 100


# ==================== Gradio 界面 ====================
def create_ui():
    """创建Gradio界面"""

    current_config = get_current_config()

    with gr.Blocks(title="AI语音助手中心", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 AI语音助手中心")
        gr.Markdown("AI对话 + 服务配置管理 一体化界面")

        with gr.Tabs():
            # ==================== AI对话标签页 ====================
            with gr.Tab("💬 AI对话"):
                gr.Markdown("### 与AI智能对话")
                gr.Markdown("支持文字和语音输入，AI回复会同时显示文字和语音")

                with gr.Row():
                    with gr.Column(scale=2):
                        # 对话历史显示
                        chatbot = gr.Chatbot(
                            label="对话记录",
                            height=450,
                            show_label=True,
                            bubble_full_width=False
                        )

                        # 文字输入区域
                        with gr.Row():
                            text_input = gr.Textbox(
                                label="",
                                placeholder="输入消息...",
                                lines=2,
                                scale=4
                            )
                            text_submit_btn = gr.Button("📤 发送", variant="primary", scale=1)

                        # 语音输入区域
                        gr.Markdown("#### 🎤 语音输入")
                        with gr.Row():
                            audio_input = gr.Audio(
                                label="录制或上传音频",
                                type="numpy",
                                scale=3
                            )
                            voice_submit_btn = gr.Button("🎙️ 语音发送", variant="secondary", scale=1)

                        voice_status = gr.Textbox(label="", lines=1, show_label=False)

                    with gr.Column(scale=1):
                        # AI回复语音播放区域
                        gr.Markdown("### 🔊 AI语音回复")
                        audio_output = gr.Audio(
                            label="点击播放",
                            type="filepath",
                            autoplay=True
                        )

                        gr.Markdown("---")

                        # 控制按钮
                        clear_chat_btn = gr.Button("🗑️ 清空对话", variant="stop")

                        gr.Markdown("### 💡 使用提示")
                        gr.Markdown("""
                        **文字输入：**
                        - 在输入框输入消息
                        - 点击"发送"或按Enter

                        **语音输入：**
                        - 点击麦克风图标录音
                        - 或上传音频文件
                        - 点击"语音发送"

                        **语音播放：**
                        - AI回复后自动播放
                        - 可重复点击播放
                        """)

                # 事件绑定
                # 文字输入
                text_submit_btn.click(
                    assistant.process_text_input,
                    inputs=[text_input, chatbot],
                    outputs=[chatbot, text_input, audio_output]
                )

                text_input.submit(
                    assistant.process_text_input,
                    inputs=[text_input, chatbot],
                    outputs=[chatbot, text_input, audio_output]
                )

                # 语音输入
                voice_submit_btn.click(
                    assistant.process_voice_input,
                    inputs=[audio_input, chatbot],
                    outputs=[chatbot, voice_status, audio_output]
                )

                # 清空对话
                def clear_conversation():
                    assistant.clear_history()
                    return [], "", None, ""

                clear_chat_btn.click(
                    clear_conversation,
                    outputs=[chatbot, text_input, audio_output, voice_status]
                )

            # ==================== 服务状态标签页 ====================
            with gr.Tab("📊 服务状态"):
                gr.Markdown("### 检查所有服务的运行状态")

                health_output = gr.Textbox(label="健康状态", lines=8)
                check_btn = gr.Button("🔄 检查服务状态", variant="primary")
                check_btn.click(check_services_health, outputs=health_output)

                gr.Markdown("### 重新加载配置")
                reload_output = gr.Textbox(label="重新加载结果", lines=5)
                reload_btn = gr.Button("🔄 重新加载所有服务配置")
                reload_btn.click(reload_all_services, outputs=reload_output)

            # ==================== ASR配置标签页 ====================
            with gr.Tab("🎤 ASR配置"):
                gr.Markdown("### 语音识别服务配置")

                asr_model_type = gr.Radio(
                    choices=["CN", "EN"],
                    value=current_config["asr_model_type"],
                    label="模型类型"
                )

                asr_save_btn = gr.Button("💾 保存ASR配置", variant="primary")
                asr_status = gr.Textbox(label="状态")
                asr_save_btn.click(save_asr_config, inputs=asr_model_type, outputs=asr_status)

                gr.Markdown("### 测试ASR服务")
                asr_audio_input = gr.Audio(label="录制或上传音频", type="numpy")
                asr_test_btn = gr.Button("🧪 测试识别")
                asr_test_output = gr.Textbox(label="测试结果", lines=5)
                asr_test_btn.click(test_asr_service, inputs=asr_audio_input, outputs=asr_test_output)

            # ==================== LLM配置标签页 ====================
            with gr.Tab("🧠 LLM配置"):
                gr.Markdown("### 大模型服务配置")

                llm_mode = gr.Radio(
                    choices=["api", "local"],
                    value=current_config["llm_mode"],
                    label="运行模式"
                )

                with gr.Group():
                    gr.Markdown("#### API配置 (在线模式)")
                    llm_provider = gr.Textbox(
                        value=current_config["llm_api_provider"],
                        label="API提供商"
                    )
                    llm_api_key = gr.Textbox(
                        value=current_config["llm_api_key"],
                        label="API Key",
                        type="password"
                    )
                    llm_api_url = gr.Textbox(
                        value=current_config["llm_api_url"],
                        label="API URL"
                    )
                    llm_model = gr.Textbox(
                        value=current_config["llm_model"],
                        label="模型名称"
                    )
                    llm_max_tokens = gr.Slider(
                        minimum=64,
                        maximum=2048,
                        value=current_config["llm_max_tokens"],
                        label="最大Token数"
                    )
                    llm_temperature = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=current_config["llm_temperature"],
                        label="Temperature"
                    )
                    llm_system_prompt = gr.Textbox(
                        value=current_config["llm_system_prompt"],
                        label="系统提示词",
                        lines=3
                    )

                with gr.Group():
                    gr.Markdown("#### 本地模型配置 (离线模式)")
                    gr.Markdown("⚠️ 本地模型需要较长加载时间(30-60秒)")

                    llm_local_model_name = gr.Radio(
                        choices=["qwen", "tinyllama"],
                        value=current_config["llm_local_model_name"],
                        label="本地模型选择"
                    )

                    llm_local_qwen_path = gr.Textbox(
                        value=current_config["llm_local_qwen_path"],
                        label="Qwen模型路径",
                        placeholder="/home/HwHiAiUser/.mindnlp/model/Qwen/Qwen1.5-0.5B-Chat"
                    )

                    llm_local_tinyllama_path = gr.Textbox(
                        value=current_config["llm_local_tinyllama_path"],
                        label="TinyLlama模型路径",
                        placeholder="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                    )

                    llm_local_max_tokens = gr.Slider(
                        minimum=32,
                        maximum=512,
                        value=current_config["llm_local_max_tokens"],
                        label="最大Token数"
                    )

                    llm_local_temperature = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=current_config["llm_local_temperature"],
                        label="Temperature"
                    )

                    llm_local_system_prompt = gr.Textbox(
                        value=current_config["llm_local_system_prompt"],
                        label="系统提示词",
                        lines=3
                    )

                llm_save_btn = gr.Button("💾 保存LLM配置", variant="primary")
                llm_status = gr.Textbox(label="状态", lines=3)
                llm_save_btn.click(
                    save_llm_config,
                    inputs=[llm_mode, llm_provider, llm_api_key, llm_api_url,
                           llm_model, llm_max_tokens, llm_temperature, llm_system_prompt,
                           llm_local_model_name, llm_local_qwen_path, llm_local_tinyllama_path,
                           llm_local_max_tokens, llm_local_temperature, llm_local_system_prompt],
                    outputs=llm_status
                )

                gr.Markdown("### 重新加载LLM服务")
                gr.Markdown("⚠️ 切换模式或本地模型后,必须重新加载服务才能生效")
                llm_reload_btn = gr.Button("🔄 重新加载LLM服务", variant="secondary")
                llm_reload_output = gr.Textbox(label="重新加载结果", lines=4)
                llm_reload_btn.click(reload_llm_service, outputs=llm_reload_output)

                gr.Markdown("### 测试LLM服务")
                llm_test_input = gr.Textbox(label="测试输入", placeholder="输入测试问题...")
                llm_test_btn = gr.Button("🧪 测试对话")
                llm_test_output = gr.Textbox(label="测试结果", lines=5)
                llm_test_btn.click(test_llm_service, inputs=llm_test_input, outputs=llm_test_output)

            # ==================== TTS配置标签页 ====================
            with gr.Tab("🔊 TTS配置"):
                gr.Markdown("### 语音合成服务配置（仅支持API模式）")

                with gr.Group():
                    gr.Markdown("#### API配置")
                    tts_provider = gr.Textbox(
                        value=current_config["tts_api_provider"],
                        label="API提供商"
                    )
                    tts_api_key = gr.Textbox(
                        value=current_config["tts_api_key"],
                        label="API Key",
                        type="password"
                    )
                    tts_model = gr.Textbox(
                        value=current_config["tts_model"],
                        label="模型名称"
                    )
                    tts_voice = gr.Textbox(
                        value=current_config["tts_voice"],
                        label="发音人"
                    )

                tts_save_btn = gr.Button("💾 保存TTS配置", variant="primary")
                tts_status = gr.Textbox(label="状态")
                tts_save_btn.click(
                    save_tts_config,
                    inputs=[tts_provider, tts_api_key, tts_model, tts_voice],
                    outputs=tts_status
                )

                gr.Markdown("### 测试TTS服务")
                tts_test_input = gr.Textbox(label="测试文本", placeholder="输入要合成的文本...")
                tts_test_btn = gr.Button("🧪 测试合成")
                tts_test_status = gr.Textbox(label="测试状态", lines=2)
                tts_test_audio = gr.Audio(label="合成音频")
                tts_test_btn.click(
                    test_tts_service,
                    inputs=tts_test_input,
                    outputs=[tts_test_status, tts_test_audio]
                )


            # ==================== 语音对话配置标签页 ====================
            with gr.Tab("🎙️ 语音对话"):
                gr.Markdown("### 线下语音对话系统配置")
                gr.Markdown("使用USB麦克风和蓝牙音箱进行语音对话")

                # 蓝牙配置提示
                with gr.Accordion("📶 蓝牙音箱配置指南", open=False):
                    gr.Markdown("""
                    #### 蓝牙音箱连接步骤：

                    1. **开启蓝牙音箱**
                       - 打开您的蓝牙音箱，确保其处于配对模式
                       - 通常会有指示灯闪烁或语音提示

                    2. **在系统中连接蓝牙设备**
                       ```bash
                       # 使用系统蓝牙工具连接蓝牙音箱
                       bluetoothctl
                       > scan on
                       > pair [设备MAC地址]
                       > connect [设备MAC地址]
                       > trust [设备MAC地址]
                       > exit
                       ```

                    3. **验证蓝牙音箱连接**
                       - 连接成功后，点击下方"刷新设备列表"按钮
                       - 在"输出设备"下拉菜单中找到您的蓝牙音箱
                       - 通常名称包含"bluez"、"bluetooth"或音箱品牌名

                    4. **测试音频输出**
                       - 选择蓝牙音箱作为输出设备后，保存配置
                       - 使用TTS配置页面的"测试合成"功能验证音频输出
                       - 确保声音从蓝牙音箱而非系统扬声器播放

                    #### 常见问题：

                    - **找不到蓝牙音箱**：确保蓝牙音箱已配对并连接到系统
                    - **音频不从蓝牙播放**：检查系统音频输出默认设备设置
                    - **音质不佳**：某些蓝牙音箱可能需要调整采样率设置
                    - **连接断开**：重新连接蓝牙音箱后，需重启语音对话服务
                    """)

                # 获取当前配置
                voice_config = get_config('voice_chat')

                with gr.Group():
                    gr.Markdown("#### 基本设置")
                    voice_enable = gr.Checkbox(
                        label="启用语音对话服务",
                        value=voice_config.get('enable', False)
                    )
                    voice_wake_mode = gr.Checkbox(
                        label="启用唤醒词模式",
                        value=voice_config.get('wake_mode', True),
                        info="需要说出唤醒词才能激活对话"
                    )
                    voice_wake_words = gr.Textbox(
                        label="唤醒词列表（用逗号分隔）",
                        value=', '.join(voice_config.get('wake_words', ["小助手", "你好助手", "嘿助手", "小爱"])),
                        placeholder="小助手, 你好助手, 嘿助手, 小爱"
                    )
                    voice_wake_reply = gr.Textbox(
                        label="唤醒确认回复",
                        value=voice_config.get('wake_reply', "你好，我在"),
                        placeholder="你好，我在",
                        info="听到唤醒词后播放的确认语音（支持自定义）"
                    )

                with gr.Group():
                    gr.Markdown("#### 🛑 打断词设置")
                    gr.Markdown("""
                    **功能说明**：当AI正在说话时，您可以通过说打断词来立即停止AI播放，并继续下一轮对话

                    **使用场景**：
                    - AI回答太长，想要打断
                    - AI理解错了，需要立即停止
                    - 听够了，想问下一个问题

                    **工作原理**：
                    1. AI开始播放回答时，系统会同时监听麦克风
                    2. 每隔2秒检测一次是否说了打断词
                    3. 一旦检测到打断词，立即停止播放并清空播放队列
                    4. 系统重新进入监听状态，等待新的唤醒词
                    """)

                    voice_interrupt_mode = gr.Checkbox(
                        label="启用打断词模式",
                        value=voice_config.get('interrupt_mode', True),
                        info="允许在AI播放时通过说打断词来停止播放"
                    )

                    voice_interrupt_words = gr.Textbox(
                        label="打断词列表（用逗号分隔）",
                        value=', '.join(voice_config.get('interrupt_words', ["停止", "暂停", "别说了", "闭嘴"])),
                        placeholder="停止, 暂停, 别说了, 闭嘴, 停下"
                    )

                    voice_interrupt_reply = gr.Textbox(
                        label="打断确认回复",
                        value=voice_config.get('interrupt_reply', "好的，已停止"),
                        placeholder="好的，已停止",
                        info="检测到打断词后播放的确认语音（支持自定义）"
                    )

                    gr.Markdown("""
                    **提示**：
                    - 打断词应该简短易说，例如"停止"、"暂停"
                    - 可以添加多个打断词，系统会检测任意一个
                    - 打断后会播放确认回复（可自定义）
                    - 打断后不会保存被打断的对话到历史记录
                    - 打断后系统会立即重新监听唤醒词
                    """)

                with gr.Group():
                    gr.Markdown("#### 思考确认回复")

                    voice_thinking_reply = gr.Textbox(
                        label="思考确认回复",
                        value=voice_config.get('thinking_reply', "好，我知道了，等我想一下"),
                        placeholder="好，我知道了，等我想一下",
                        info="识别到问题后、开始AI思考前播放的确认语音（支持自定义，支持缓存）"
                    )

                    gr.Markdown("""
                    **提示**：
                    - 在识别完用户问题后立即播放，让用户知道系统已经收到问题
                    - 提升用户体验，避免等待AI思考时的尴尬沉默
                    - 音频会自动缓存，重复使用不需要重新生成
                    """)

                with gr.Group():
                    gr.Markdown("#### 音频设备设置")
                    gr.Markdown("⚠️ **重要**：连接蓝牙音箱后，必须点击刷新按钮才能检测到设备")

                    with gr.Row():
                        device_refresh_btn = gr.Button("🔄 刷新设备列表", variant="primary", scale=1)
                        bluetooth_check_btn = gr.Button("🔵 检查蓝牙连接", variant="secondary", scale=1)
                        set_default_btn = gr.Button("🔊 设为默认输出", variant="secondary", scale=1)

                    device_refresh_status = gr.Textbox(label="设备状态", lines=5, show_label=True)

                    # 获取设备列表
                    input_devices, output_devices = get_voice_devices()
                    input_choices = [("默认设备", -1)] + input_devices
                    output_choices = [("默认设备", -1)] + output_devices

                    current_input = voice_config.get('input_device', None)
                    current_output = voice_config.get('output_device', None)

                    voice_input_device = gr.Dropdown(
                        choices=input_choices,
                        value=current_input if current_input is not None else -1,
                        label="🎤 输入设备（麦克风）",
                        info="选择USB麦克风或其他输入设备，推荐使用外接USB麦克风以获得更好的录音质量"
                    )
                    voice_output_device = gr.Dropdown(
                        choices=output_choices,
                        value=current_output if current_output is not None else -1,
                        label="🔊 输出设备（蓝牙音箱）",
                        info="⚠️ 选择蓝牙音箱作为输出设备，名称通常包含'bluez'、'bluetooth'或音箱品牌名"
                    )

                    # 音量控制
                    gr.Markdown("#### 🔊 音量控制")
                    with gr.Row():
                        voice_volume_slider = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=voice_config.get('output_volume', 100),
                            step=5,
                            label="输出音量 (%)",
                            info="调整蓝牙音箱的播放音量（0-100%）",
                            scale=3
                        )
                        volume_set_btn = gr.Button("🎚️ 应用音量", variant="primary", scale=1)

                    volume_status = gr.Textbox(label="音量状态", lines=2, show_label=True)

                    device_refresh_btn.click(
                        refresh_devices,
                        outputs=[voice_input_device, voice_output_device, device_refresh_status]
                    )

                    bluetooth_check_btn.click(
                        check_bluetooth_status,
                        outputs=device_refresh_status
                    )

                    set_default_btn.click(
                        set_default_audio_sink,
                        outputs=device_refresh_status
                    )

                    volume_set_btn.click(
                        set_audio_volume,
                        inputs=voice_volume_slider,
                        outputs=volume_status
                    )

                with gr.Group():
                    gr.Markdown("#### 🎤 麦克风音量监测工具")
                    gr.Markdown("""
                    **功能说明**：实时监测麦克风环境噪音，帮助您找到最佳的静音阈值设置

                    **使用方法**：
                    1. 确保周围环境保持**安静**（不说话）
                    2. 点击"开始监测"按钮
                    3. 等待10秒，期间保持安静
                    4. 查看监测结果和推荐阈值
                    5. 使用推荐的阈值更新下方"静音阈值"设置
                    """)

                    with gr.Row():
                        volume_monitor_btn = gr.Button("🎤 开始监测（10秒）", variant="primary")
                        volume_stop_btn = gr.Button("⏹️ 停止监测", variant="secondary")

                    volume_monitor_status = gr.Textbox(
                        label="监测状态",
                        lines=1,
                        value="未开始监测"
                    )

                    with gr.Row():
                        volume_current_rms = gr.Number(
                            label="当前RMS",
                            value=0,
                            interactive=False
                        )
                        volume_avg_rms = gr.Number(
                            label="平均RMS",
                            value=0,
                            interactive=False
                        )

                    with gr.Row():
                        volume_min_rms = gr.Number(
                            label="最小RMS",
                            value=0,
                            interactive=False
                        )
                        volume_max_rms = gr.Number(
                            label="最大RMS",
                            value=0,
                            interactive=False
                        )

                    volume_recommended_threshold = gr.Number(
                        label="🎯 推荐静音阈值",
                        value=0,
                        interactive=False
                    )

                    gr.Markdown("""
                    **监测结果解读**：
                    - **当前RMS**：实时麦克风音量
                    - **平均RMS**：10秒内环境噪音平均值
                    - **最小/最大RMS**：噪音波动范围
                    - **推荐阈值**：基于环境噪音自动计算（平均值的1.3倍），确保能可靠检测静音

                    💡 **提示**：推荐阈值应该**高于**环境噪音，但**低于**说话音量
                    """)

                    # 绑定按钮事件
                    volume_monitor_btn.click(
                        start_volume_monitor,
                        outputs=[
                            volume_monitor_status,
                            volume_current_rms,
                            volume_avg_rms,
                            volume_min_rms,
                            volume_max_rms,
                            volume_recommended_threshold
                        ]
                    ).then(
                        # 启动后每秒自动刷新数据，持续11秒（监测10秒 + 1秒缓冲）
                        lambda: None,  # 空操作，用于触发后续的刷新
                        None,
                        None
                    )

                    volume_stop_btn.click(
                        stop_volume_monitor,
                        outputs=volume_monitor_status
                    )

                    # 添加自动刷新定时器（每500ms刷新一次数据）
                    volume_timer = gr.Timer(value=0.5, active=False)
                    volume_timer.tick(
                        get_volume_data,
                        outputs=[
                            volume_monitor_status,
                            volume_current_rms,
                            volume_avg_rms,
                            volume_min_rms,
                            volume_max_rms,
                            volume_recommended_threshold
                        ]
                    )

                    # 点击开始监测后激活定时器
                    volume_monitor_btn.click(
                        lambda: gr.Timer(active=True),
                        None,
                        volume_timer
                    )

                    # 监测完成或停止后禁用定时器
                    volume_stop_btn.click(
                        lambda: gr.Timer(active=False),
                        None,
                        volume_timer
                    )

                with gr.Group():
                    gr.Markdown("#### VAD参数设置")
                    gr.Markdown("💡 **提示**：如果录音一直到超时才结束，说明无法检测到静音")

                    voice_silence_threshold = gr.Slider(
                        minimum=100,
                        maximum=5000,
                        value=voice_config.get('silence_threshold', 500),
                        step=50,
                        label="静音阈值",
                        info="⚠️ 音量低于此值才视为静音。当前值：" + str(voice_config.get('silence_threshold', 500))
                    )

                    # 添加阈值建议
                    gr.Markdown("""
                    **阈值原理**：
                    - RMS值（音量）**高于**阈值 → 有声音，继续录音
                    - RMS值（音量）**低于**阈值 → 静音，开始计数
                    - 静音持续足够久 → 停止录音

                    **阈值参考**：
                    - **安静环境（噪音RMS ~300）**：推荐阈值 500-800
                    - **普通环境（噪音RMS ~500）**：推荐阈值 800-1500
                    - **嘈杂环境（噪音RMS ~1000）**：推荐阈值 1500-3000
                    - **当前配置**：""" + str(voice_config.get('silence_threshold', 500)) + """

                    ⚠️ **如果遇到录音一直到30秒超时才结束**：

                    **原因分析**：
                    - 说明停止说话后，环境音量（RMS）仍然高于阈值
                    - 系统认为还有声音，无法检测到"静音"

                    **解决方案**：
                    1. **提高阈值**（让系统更容易识别为"静音"）
                       - 如果当前是500，尝试调到1000-1500
                       - 如果当前是2000，可能已经合适，检查是否有持续噪音
                    2. **查看日志**获取实际RMS值：
                       ```bash
                       tail -f logs/语音对话.log | grep "RMS"
                       ```
                    3. **降低环境噪音**或远离噪音源
                    4. **调整麦克风增益**（降低输入音量）

                    **调试方法**：
                    - 观察日志中的RMS值
                    - 说话时RMS应该明显高于静音时
                    - 阈值应该设在两者之间
                    """)
                    voice_silence_duration = gr.Slider(
                        minimum=0.1,
                        maximum=5.0,
                        value=voice_config.get('silence_duration', 1.5),
                        step=0.1,
                        label="静音持续时间（秒）",
                        info="静音持续多久后停止录音（最低0.1秒，用于唤醒词快速检测）"
                    )
                    voice_min_audio_length = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=voice_config.get('min_audio_length', 0.5),
                        step=0.1,
                        label="最短音频长度（秒）",
                        info="录音时长少于此值将被忽略"
                    )
                    voice_continue_timeout = gr.Slider(
                        minimum=1.0,
                        maximum=30.0,
                        value=voice_config.get('continue_dialogue_timeout', 5.0),
                        step=0.5,
                        label="连续对话超时（秒）",
                        info="AI回答后等待多久无语音将返回待机模式（建议3-10秒）"
                    )

                voice_save_btn = gr.Button("💾 保存配置", variant="primary")
                voice_save_status = gr.Textbox(label="保存状态", lines=2)

                voice_save_btn.click(
                    save_voice_chat_config,
                    inputs=[
                        voice_enable,
                        voice_wake_mode,
                        voice_wake_words,
                        voice_wake_reply,
                        voice_interrupt_mode,
                        voice_interrupt_words,
                        voice_interrupt_reply,
                        voice_thinking_reply,
                        voice_input_device,
                        voice_output_device,
                        voice_volume_slider,
                        voice_silence_threshold,
                        voice_silence_duration,
                        voice_min_audio_length,
                        voice_continue_timeout
                    ],
                    outputs=voice_save_status
                )

                gr.Markdown("### 服务控制")
                with gr.Row():
                    voice_start_btn = gr.Button("▶️ 启动", variant="primary")
                    voice_stop_btn = gr.Button("⏹️ 停止", variant="secondary")
                    voice_restart_btn = gr.Button("🔄 重启", variant="secondary")
                    voice_status_btn = gr.Button("📊 查看状态", variant="secondary")

                voice_control_status = gr.Textbox(label="服务状态", lines=3)

                voice_start_btn.click(start_voice_chat, outputs=voice_control_status)
                voice_stop_btn.click(stop_voice_chat, outputs=voice_control_status)
                voice_restart_btn.click(restart_voice_chat, outputs=voice_control_status)
                voice_status_btn.click(get_voice_chat_status, outputs=voice_control_status)

                gr.Markdown("""
                ### 使用说明

                #### 1. 蓝牙音箱配置（推荐方式）
                - **首次使用**：在系统中配对并连接蓝牙音箱（查看上方"蓝牙音箱配置指南"）
                - **检查连接**：点击"🔵 检查蓝牙连接"确认蓝牙音箱已连接
                - **设为默认**：点击"🔊 设为默认输出"将蓝牙音箱设为系统默认音频输出
                  - ⚠️ **这一步很重要**：通过PulseAudio设置默认输出可确保所有音频都从蓝牙音箱播放
                  - 系统会使用`paplay`命令播放音频，完美支持蓝牙设备
                - **验证配置**：前往"🔊 TTS配置"页面测试音频输出

                #### 2. 音量调整
                - **调整音量**：使用"输出音量"滑块调整播放音量（0-100%）
                - **应用音量**：点击"🎚️ 应用音量"按钮立即生效
                  - 方式1：通过PulseAudio设置系统音量（推荐）
                  - 方式2：在音频播放前对PCM数据进行软件音量调整
                - **保存配置**：点击"保存配置"将音量设置保存到配置文件
                - **注意**：
                  - 音量调整对所有音频输出生效（唤醒确认、AI回复等）
                  - 如果蓝牙音箱本身音量很低，建议先调高音箱硬件音量

                #### 3. USB麦克风配置
                - 插入USB麦克风后，点击"🔄 刷新设备列表"
                - 在"输入设备"下拉菜单中选择USB麦克风
                - 如不选择，将使用系统默认麦克风

                #### 4. 唤醒词设置
                - **启用唤醒词模式**：需要先说唤醒词（如"小助手"）才能进行对话
                - **关闭唤醒词模式**：系统持续监听，直接说话即可对话（不推荐）
                - **自定义唤醒词**：在"唤醒词列表"中添加，用逗号分隔多个唤醒词

                #### 5. VAD参数调整
                - **静音阈值**：音量低于此值视为静音，建议500-2000
                - **静音持续时间**：静音持续多久后停止录音，建议1.0-2.0秒
                - **最短音频长度**：录音时长少于此值将被忽略，建议0.5秒

                #### 6. 启动服务
                - 确保配置已保存并且"启用语音对话服务"已勾选
                - 点击"重启"按钮使配置生效
                - 查看"服务状态"确认服务正在运行

                #### 7. 音频播放技术说明
                - **PulseAudio优先**：系统会优先使用`paplay`命令播放音频
                - **蓝牙兼容性好**：PulseAudio对蓝牙设备支持最佳
                - **软件音量控制**：在音频播放前对PCM数据进行音量调整
                - **PyAudio备用**：如果paplay不可用，会自动降级到PyAudio

                #### 常见问题排查
                - **蓝牙音箱无声音**：
                  1. 检查音箱是否已连接：`bluetoothctl info [MAC地址]`
                  2. 检查PulseAudio是否识别：`pactl list sinks short | grep bluez`
                  3. 确保已设为默认输出：点击"🔊 设为默认输出"按钮
                  4. 测试系统音频：`paplay /usr/share/sounds/alsa/Front_Center.wav`

                - **音量太小或太大**：
                  1. 调整Web界面中的"输出音量"滑块
                  2. 点击"应用音量"按钮
                  3. 也可以调整蓝牙音箱本身的硬件音量
                  4. 使用`pactl set-sink-volume @DEFAULT_SINK@ 50%`命令行调整

                - **录音无响应**：检查USB麦克风是否正确选择，调整静音阈值

                - **识别率低**：使用质量较好的USB麦克风，避免环境噪音

                - **服务异常**：查看"📊 服务状态"页面，确保所有服务正常运行
                """)


            # ==================== 音色克隆标签页 ====================
            with gr.Tab("🎨 音色克隆"):
                gr.Markdown("### CosyVoice音色克隆服务")
                gr.Markdown("""
                使用10~20秒音频样本即可生成高度相似且自然的定制声音。

                **音频要求:**
                - 格式: WAV (16bit), MP3, M4A
                - 时长: 10~20秒
                - 大小: ≤ 10 MB
                - 采样率: ≥ 16 kHz
                - 内容: 至少包含一段5秒以上的连续、清晰、无背景音的朗读
                - 语言: 中文、英文
                """)

                with gr.Group():
                    gr.Markdown("#### 创建新音色")
                    voice_create_model = gr.Dropdown(
                        choices=["cosyvoice-v1", "cosyvoice-v2", "cosyvoice-v3", "cosyvoice-v3-plus"],
                        value="cosyvoice-v2",
                        label="目标模型",
                        info="推荐使用v3-plus获得最佳效果"
                    )
                    voice_create_prefix = gr.Textbox(
                        label="音色前缀",
                        placeholder="myvoice (仅允许小写字母和数字,少于10个字符)",
                        max_lines=1
                    )
                    voice_create_url = gr.Textbox(
                        label="音频URL",
                        placeholder="https://your-audio-file-url.wav",
                        info="音频文件必须是公网可访问的URL"
                    )
                    voice_create_btn = gr.Button("🎨 创建音色", variant="primary")
                    voice_create_output = gr.Textbox(label="创建结果", lines=5)
                    voice_created_id = gr.Textbox(label="Voice ID", interactive=False)

                    voice_create_btn.click(
                        create_voice_enrollment,
                        inputs=[voice_create_model, voice_create_prefix, voice_create_url],
                        outputs=[voice_create_output, voice_created_id]
                    )

                with gr.Group():
                    gr.Markdown("#### 查询音色状态")
                    voice_query_id = gr.Textbox(
                        label="Voice ID",
                        placeholder="cosyvoice-v2-myvoice-xxxxxxxx"
                    )
                    voice_query_btn = gr.Button("🔍 查询状态")
                    voice_query_output = gr.Textbox(label="音色信息", lines=10)

                    voice_query_btn.click(
                        query_voice_status,
                        inputs=voice_query_id,
                        outputs=voice_query_output
                    )

                with gr.Group():
                    gr.Markdown("#### 列出所有音色")
                    with gr.Row():
                        voice_list_prefix = gr.Textbox(
                            label="前缀筛选 (可选)",
                            placeholder="myvoice"
                        )
                        voice_list_page_index = gr.Number(
                            label="页码",
                            value=0,
                            precision=0
                        )
                        voice_list_page_size = gr.Number(
                            label="每页数量",
                            value=10,
                            precision=0
                        )
                    voice_list_btn = gr.Button("📋 列出音色")
                    voice_list_output = gr.Textbox(label="音色列表", lines=15)

                    voice_list_btn.click(
                        list_all_voices,
                        inputs=[voice_list_prefix, voice_list_page_index, voice_list_page_size],
                        outputs=voice_list_output
                    )

                with gr.Group():
                    gr.Markdown("#### 更新音色")
                    voice_update_id = gr.Textbox(
                        label="Voice ID",
                        placeholder="cosyvoice-v2-myvoice-xxxxxxxx"
                    )
                    voice_update_url = gr.Textbox(
                        label="新音频URL",
                        placeholder="https://your-new-audio-file-url.wav"
                    )
                    voice_update_btn = gr.Button("🔄 更新音色")
                    voice_update_output = gr.Textbox(label="更新结果", lines=3)

                    voice_update_btn.click(
                        update_voice_enrollment,
                        inputs=[voice_update_id, voice_update_url],
                        outputs=voice_update_output
                    )

                with gr.Group():
                    gr.Markdown("#### 删除音色")
                    gr.Markdown("⚠️ 删除操作不可逆,请谨慎操作")
                    voice_delete_id = gr.Textbox(
                        label="Voice ID",
                        placeholder="cosyvoice-v2-myvoice-xxxxxxxx"
                    )
                    voice_delete_btn = gr.Button("🗑️ 删除音色", variant="stop")
                    voice_delete_output = gr.Textbox(label="删除结果", lines=2)

                    voice_delete_btn.click(
                        delete_voice_enrollment,
                        inputs=voice_delete_id,
                        outputs=voice_delete_output
                    )

            # ==================== YOLO检测标签页 ====================
            with gr.Tab("📹 YOLO检测"):
                gr.Markdown("### 实时目标检测")
                gr.Markdown("使用YOLOv5进行实时摄像头目标检测，支持80种COCO数据集类别")

                # YOLO检测显示区域
                with gr.Row():
                    with gr.Column(scale=2):
                        # 视频流显示
                        yolo_video = gr.Image(
                            label="📹 实时检测画面",
                            sources="webcam",
                            streaming=True,
                            interactive=False
                        )

                        # 控制按钮
                        with gr.Row():
                            yolo_start_btn = gr.Button("🎥 开始检测", variant="primary", scale=1)
                            yolo_stop_btn = gr.Button("⏹️ 停止检测", variant="stop", scale=1)
                            yolo_refresh_btn = gr.Button("🔄 刷新状态", variant="secondary", scale=1)

                    with gr.Column(scale=1):
                        # 检测参数控制
                        gr.Markdown("#### 检测参数")
                        yolo_confidence = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=get_config('yolo.confidence_threshold', 0.5),
                            step=0.05,
                            label="置信度阈值",
                            info="过滤低置信度的检测结果"
                        )

                        yolo_nms = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=get_config('yolo.nms_threshold', 0.4),
                            step=0.05,
                            label="NMS阈值",
                            info="非极大值抑制阈值"
                        )

                        # FPS显示
                        yolo_fps_display = gr.Textbox(
                            label="实时FPS",
                            value="0.0",
                            interactive=False
                        )

                        # 检测统计
                        yolo_stats = gr.JSON(
                            label="检测统计",
                            value={}
                        )

                # 检测结果显示
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 检测结果列表")
                        yolo_detections_list = gr.DataFrame(
                            headers=["类别", "置信度", "位置"],
                            datatype=["str", "number", "str"],
                            interactive=False
                        )

                        # 历史记录
                        with gr.Row():
                            yolo_clear_history_btn = gr.Button("🗑️ 清空历史", size="sm")
                            yolo_export_btn = gr.Button("💾 导出截图", size="sm")

                    with gr.Column():
                        # YOLO控制（C++版本）
                        gr.Markdown("#### C++版本控制")
                        with gr.Row():
                            cpp_start_btn = gr.Button("🚀 启动C++检测", variant="secondary")
                            cpp_stop_btn = gr.Button("🛑 停止C++检测", variant="secondary")

                        cpp_status = gr.Textbox(
                            label="C++状态",
                            lines=5,
                            value="未启动",
                            interactive=False
                        )

                # 摄像头配置
                with gr.Accordion("📷 高级配置", open=False):
                    yolo_camera_index = gr.Number(
                        label="摄像头索引",
                        value=get_config('yolo.camera_index', 0),
                        precision=0,
                        info="指定摄像头设备索引，-1为自动检测"
                    )

                    yolo_max_fps = gr.Slider(
                        minimum=5,
                        maximum=30,
                        value=get_config('yolo.max_fps', 15),
                        step=1,
                        label="最大FPS",
                        info="限制检测帧率以降低CPU负载"
                    )

                    gr.Markdown("""
                    **使用说明**:
                    - 点击"开始检测"启动实时检测
                    - 调整置信度阈值过滤不重要的检测
                    - 检测结果会实时显示在画面和列表中
                    - 可以导出当前检测截图保存
                    """)

                # 绑定事件处理函数
                yolo_start_btn.click(
                    fn=start_yolo_detection,
                    inputs=[yolo_camera_index, yolo_confidence],
                    outputs=[yolo_video, yolo_fps_display, yolo_detections_list]
                )

                yolo_stop_btn.click(
                    fn=stop_yolo_detection,
                    outputs=[yolo_video, yolo_fps_display]
                )

                yolo_refresh_btn.click(
                    fn=get_yolo_status,
                    outputs=[yolo_fps_display, yolo_stats]
                )

                yolo_confidence.change(
                    fn=update_yolo_settings,
                    inputs=[yolo_confidence, yolo_nms],
                    outputs=[]
                )

                cpp_start_btn.click(
                    fn=run_yolo_cpp_detection,
                    outputs=[cpp_status]
                )

                cpp_stop_btn.click(
                    fn=stop_yolo_cpp_detection,
                    outputs=[cpp_status]
                )

                # 使用定时器更新检测状态
                yolo_timer = gr.Timer(value=0.2)  # 200ms刷新一次
                yolo_timer.tick(
                    fn=update_yolo_cpp_stream,
                    inputs=[yolo_confidence],
                    outputs=[yolo_video, yolo_fps_display, yolo_detections_list]
                )

    return demo


# ==================== YOLO Detection Functions ====================

def start_yolo_detection(camera_index, confidence_threshold):
    """启动YOLO检测"""
    try:
        import requests
        port = get_config('services.yolo', 5005)

        # 转换摄像头索引
        cam_idx = None if camera_index == -1 else int(camera_index)

        # 启动检测
        response = requests.post(
            f"http://localhost:{port}/detect/start",
            json={
                "camera_index": cam_idx,
                "confidence_threshold": confidence_threshold
            },
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                logger.info(f"YOLO检测已启动: {result.get('message')}")
                # 等待一下让摄像头开始捕获
                time.sleep(1)
                # 返回初始状态，定时器会更新实际的图像
                return "检测已启动，正在加载...", "0.0", []
            else:
                logger.error(f"YOLO启动失败: {result.get('message')}")
                return None, "错误", []
        else:
            logger.error(f"YOLO启动请求失败: {response.status_code}")
            return None, f"HTTP {response.status_code}", []

    except Exception as e:
        logger.error(f"启动YOLO检测出错: {e}")
        return None, f"错误: {str(e)}", []

def stop_yolo_detection():
    """停止YOLO检测"""
    try:
        import requests
        port = get_config('services.yolo', 5005)

        response = requests.post(
            f"http://localhost:{port}/detect/stop",
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            logger.info(f"YOLO检测已停止: {result.get('message')}")

        return None, "0.0"

    except Exception as e:
        logger.error(f"停止YOLO检测出错: {e}")
        return None, "错误"

def get_yolo_status():
    """获取YOLO状态"""
    try:
        import requests
        port = get_config('services.yolo', 5005)

        # 获取检测状态
        response = requests.get(
            f"http://localhost:{port}/detect/status",
            timeout=5
        )

        if response.status_code == 200:
            result = response.json()
            status = result.get('status', {})
            fps = status.get('fps', 0.0)
            stats = {
                "is_running": status.get('is_running', False),
                "camera_index": status.get('camera_index'),
                "fps": round(fps, 1),
                "detections": status.get('last_detection_count', 0)
            }
            return str(round(fps, 1)), stats
        else:
            return "0.0", {"error": "无法获取状态"}

    except Exception as e:
        logger.error(f"获取YOLO状态出错: {e}")
        return "0.0", {"error": str(e)}

def update_yolo_settings(confidence_threshold, nms_threshold):
    """更新YOLO设置"""
    try:
        import requests
        port = get_config('services.yolo', 5005)

        response = requests.post(
            f"http://localhost:{port}/detect/update_settings",
            json={
                "confidence_threshold": confidence_threshold,
                "nms_threshold": nms_threshold
            },
            timeout=5
        )

        if response.status_code == 200:
            result = response.json()
            logger.info(f"YOLO设置已更新: {result}")

    except Exception as e:
        logger.error(f"更新YOLO设置出错: {e}")

def update_yolo_stream(confidence_threshold):
    """更新YOLO视频流"""
    try:
        import requests
        import base64
        from PIL import Image
        import io

        port = get_config('services.yolo', 5005)

        # 获取最新检测结果
        response = requests.get(
            f"http://localhost:{port}/detect/latest",
            timeout=5
        )

        if response.status_code == 200:
            result = response.json()

            # 解码图像
            frame_base64 = result.get('frame_base64', '')
            if frame_base64:
                image_data = base64.b64decode(frame_base64)
                image = Image.open(io.BytesIO(image_data))

                # 处理检测结果
                detections_data = result.get('detections', {})
                detections = detections_data.get('detections', [])
                fps = detections_data.get('fps', 0.0)

                # 转换检测结果为DataFrame格式
                detection_list = []
                for det in detections:
                    bbox = det.get('bbox', [])
                    pos_str = f"[{bbox[0]}, {bbox[1]}]"
                    detection_list.append([
                        det.get('label', ''),
                        round(det.get('confidence', 0), 3),
                        pos_str
                    ])

                return image, str(round(fps, 1)), detection_list

        return None, "0.0", []

    except Exception as e:
        logger.error(f"更新YOLO流出错: {e}")
        return None, "0.0", []

def update_yolo_cpp_stream(confidence_threshold):
    """更新C++ YOLO视频流"""
    try:
        import requests
        import base64
        from PIL import Image
        import io

        # C++服务运行在5007端口
        port = 5007

        # 获取最新帧
        response = requests.get(
            f"http://localhost:{port}/frame",
            timeout=5
        )

        if response.status_code == 200:
            result = response.json()

            # 解码图像
            image_data = result.get('image', '')
            if image_data and image_data.startswith('data:image/jpeg;base64,'):
                # 移除data URL前缀
                base64_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(base64_data)
                image = Image.open(io.BytesIO(image_bytes))

                # 处理检测结果
                detections = result.get('detections', [])
                fps = result.get('fps', 0.0)

                # 转换检测结果为DataFrame格式
                detection_list = []
                for det in detections:
                    detection_list.append([
                        det.get('label', ''),
                        round(det.get('confidence', 0), 3),
                        f"[{det.get('x', 0):.0f}, {det.get('y', 0):.0f}]"
                    ])

                return image, str(round(fps, 1)), detection_list

        return None, "0.0", []

    except Exception as e:
        logger.error(f"更新C++ YOLO流出错: {e}")
        return None, "0.0", []

def update_yolo_detection_info(confidence_threshold):
    """更新YOLO检测信息（保留兼容性）"""
    return get_yolo_status()[0]

def run_yolo_cpp_detection():
    """运行C++ YOLO检测"""
    try:
        import requests
        port = get_config('services.yolo', 5005)
        response = requests.post(f"http://localhost:{port}/yolo/start", timeout=10)

        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                return "✅ C++ YOLO检测已启动！\n" + result.get('message', '')
            else:
                return "❌ 启动失败: " + result.get('message', '未知错误')
        else:
            return f"❌ 请求失败: HTTP {response.status_code}"
    except Exception as e:
        return f"❌ 错误: {str(e)}"

def stop_yolo_cpp_detection():
    """停止C++ YOLO检测"""
    try:
        import requests
        port = get_config('services.yolo', 5005)
        response = requests.post(f"http://localhost:{port}/yolo/stop", timeout=10)

        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                return "✅ C++ YOLO检测已停止\n" + result.get('message', '')
            else:
                return "❌ 停止失败: " + result.get('message', '未知错误')
        else:
            return f"❌ 请求失败: HTTP {response.status_code}"
    except Exception as e:
        return f"❌ 错误: {str(e)}"

def check_yolo_cpp_status():
    """检查C++ YOLO状态"""
    try:
        import requests
        port = get_config('services.yolo', 5005)
        response = requests.get(f"http://localhost:{port}/yolo/status", timeout=10)

        if response.status_code == 200:
            result = response.json()
            status_msg = f"状态: {result.get('status', 'unknown')}\n"
            if result.get('pid'):
                status_msg += f"进程ID: {result['pid']}\n"
            if result.get('executable'):
                status_msg += f"可执行文件: {result['executable']}"
            return status_msg
        else:
            return f"❌ 请求失败: HTTP {response.status_code}"
    except Exception as e:
        return f"❌ 错误: {str(e)}"



if __name__ == "__main__":
    # 创建并启动界面
    demo = create_ui()

    port = get_config('services.web_ui', 8080)
    share = get_config('web.share', False)

    logger.info(f"🌐 Web配置界面启动在端口: {port}")

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=share
    )
