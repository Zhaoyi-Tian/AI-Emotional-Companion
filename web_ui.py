"""
Web配置管理界面
使用Gradio提供友好的配置管理和测试界面
"""

import gradio as gr
import requests
import soundfile as sf
import numpy as np
from pathlib import Path
import sys
import logging

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent))
from config_loader import config, get_config, set_config, reload_config

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WebUI")


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
        return "✅ ASR配置已保存"
    except Exception as e:
        return f"❌ 保存失败: {str(e)}"


def save_llm_config(mode, provider, api_key, api_url, model, max_tokens, temperature, system_prompt):
    """保存LLM配置"""
    try:
        set_config('llm.mode', mode, save=False)
        set_config('llm.api.provider', provider, save=False)
        set_config('llm.api.api_key', api_key, save=False)
        set_config('llm.api.api_url', api_url, save=False)
        set_config('llm.api.model', model, save=False)
        set_config('llm.api.max_tokens', int(max_tokens), save=False)
        set_config('llm.api.temperature', float(temperature), save=False)
        set_config('llm.api.system_prompt', system_prompt, save=True)
        return "✅ LLM配置已保存"
    except Exception as e:
        return f"❌ 保存失败: {str(e)}"


def save_tts_config(mode, provider, api_key, model, voice):
    """保存TTS配置"""
    try:
        set_config('tts.mode', mode, save=False)
        set_config('tts.api.provider', provider, save=False)
        set_config('tts.api.api_key', api_key, save=False)
        set_config('tts.api.model', model, save=False)
        set_config('tts.api.voice', voice, save=True)
        return "✅ TTS配置已保存"
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
            'TTS': f"http://localhost:{ports['tts']}/reload_config"
        }

        for name, url in services.items():
            try:
                response = requests.post(url, timeout=5)
                if response.status_code == 200:
                    results.append(f"✅ {name}服务配置已重新加载")
                else:
                    results.append(f"⚠️ {name}服务重新加载失败")
            except Exception as e:
                results.append(f"❌ {name}服务不可达: {str(e)}")

        return "\n".join(results)
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


def check_services_health():
    """检查所有服务健康状态"""
    try:
        port = get_config('services.orchestrator', 5000)
        url = f"http://localhost:{port}/health"

        response = requests.get(url, timeout=5)

        if response.status_code == 200:
            result = response.json()
            services = result.get('services', {})

            status_text = "🔍 服务健康状态:\n\n"
            for name, status in services.items():
                emoji = "✅" if status == "healthy" else "❌"
                status_text += f"{emoji} {name.upper()}: {status}\n"

            return status_text
        else:
            return "❌ 无法获取健康状态"

    except Exception as e:
        return f"❌ 检查失败: {str(e)}\n\n请确保所有服务已启动"


# ==================== Gradio 界面 ====================
def create_ui():
    """创建Gradio界面"""

    current_config = get_current_config()

    with gr.Blocks(title="AI语音助手配置中心", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 AI语音助手配置中心")
        gr.Markdown("管理和测试ASR、LLM、TTS服务的配置")

        with gr.Tabs():
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
                    gr.Markdown("#### API配置")
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

                llm_save_btn = gr.Button("💾 保存LLM配置", variant="primary")
                llm_status = gr.Textbox(label="状态")
                llm_save_btn.click(
                    save_llm_config,
                    inputs=[llm_mode, llm_provider, llm_api_key, llm_api_url,
                           llm_model, llm_max_tokens, llm_temperature, llm_system_prompt],
                    outputs=llm_status
                )

                gr.Markdown("### 测试LLM服务")
                llm_test_input = gr.Textbox(label="测试输入", placeholder="输入测试问题...")
                llm_test_btn = gr.Button("🧪 测试对话")
                llm_test_output = gr.Textbox(label="测试结果", lines=5)
                llm_test_btn.click(test_llm_service, inputs=llm_test_input, outputs=llm_test_output)

            # ==================== TTS配置标签页 ====================
            with gr.Tab("🔊 TTS配置"):
                gr.Markdown("### 语音合成服务配置")

                tts_mode = gr.Radio(
                    choices=["api", "local"],
                    value=current_config["tts_mode"],
                    label="运行模式"
                )

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
                    inputs=[tts_mode, tts_provider, tts_api_key, tts_model, tts_voice],
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

    return demo


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
