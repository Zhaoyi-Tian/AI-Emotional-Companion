"""
线下语音对话系统
使用USB麦克风录音，通过蓝牙音箱播放回复
"""

import os
import sys
import logging
from pathlib import Path
import time
import subprocess
import tempfile
import threading
import warnings
from contextlib import contextmanager
from queue import Queue, Empty

# 抑制ALSA警告信息
os.environ['PYAUDIO_ALSA_ERRORS'] = '0'
warnings.filterwarnings('ignore', category=DeprecationWarning)

# 重定向stderr以抑制ALSA错误信息
@contextmanager
def suppress_stderr():
    """临时抑制stderr输出（用于抑制ALSA错误）"""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    sys.stderr.flush()
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)

# 在导入pyaudio时抑制ALSA错误
with suppress_stderr():
    import pyaudio

import wave
import numpy as np
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from scipy import signal

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent))
from config_loader import get_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("VoiceChat")

# 创建FastAPI应用用于API接口
app = FastAPI(title="Voice Chat API")

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AudioPlaybackQueue:
    """
    音频播放队列管理器
    实现生产者-消费者模式，支持TTS异步生成和播放
    """

    def __init__(self, voice_assistant):
        self.queue = Queue()
        self.voice_assistant = voice_assistant
        self.is_playing = False
        self.stop_flag = False
        self.playback_thread = None
        self.output_device = None

    def start(self, output_device=None):
        """启动播放线程"""
        self.output_device = output_device
        self.stop_flag = False
        self.playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self.playback_thread.start()
        logger.info("🎵 音频播放队列已启动")

    def stop(self):
        """停止播放线程"""
        self.stop_flag = True
        # 清空队列
        while not self.queue.empty():
            try:
                audio_file, _ = self.queue.get_nowait()
                # 删除未播放的音频文件
                if audio_file and os.path.exists(audio_file):
                    os.unlink(audio_file)
            except Empty:
                break

        if self.playback_thread:
            self.playback_thread.join(timeout=2)
        logger.info("🛑 音频播放队列已停止")

    def add(self, audio_file, text=""):
        """添加音频到播放队列"""
        if audio_file:
            self.queue.put((audio_file, text))
            logger.debug(f"📥 音频已加入队列，当前队列长度: {self.queue.qsize()}")

    def _playback_worker(self):
        """播放工作线程"""
        logger.info("🎧 播放工作线程已启动")

        while not self.stop_flag:
            try:
                # 等待队列中的音频，超时1秒
                audio_file, text = self.queue.get(timeout=1)

                if audio_file and os.path.exists(audio_file):
                    self.is_playing = True
                    if text:
                        logger.info(f"🔊 正在播放: {text[:30]}...")

                    # 调用语音助手的播放方法
                    # 播放时会检查interrupt_flag
                    self.voice_assistant.play_audio(audio_file, self.output_device)
                    self.is_playing = False

                    # 检查是否被打断
                    if self.voice_assistant.interrupt_flag:
                        logger.info("⏹️ 检测到打断标志，清空播放队列")
                        # 清空剩余队列
                        while not self.queue.empty():
                            try:
                                remaining_file, _ = self.queue.get_nowait()
                                if remaining_file and os.path.exists(remaining_file):
                                    os.unlink(remaining_file)
                                self.queue.task_done()
                            except Empty:
                                break
                        # 标记当前任务完成
                        self.queue.task_done()
                        break

                    # 标记任务完成
                    self.queue.task_done()
                else:
                    logger.warning(f"⚠️ 音频文件不存在或无效: {audio_file}")

            except Empty:
                # 队列为空，继续等待
                continue
            except Exception as e:
                logger.error(f"❌ 播放音频时出错: {e}")
                import traceback
                logger.error(traceback.format_exc())
                self.is_playing = False

        logger.info("🛑 播放工作线程已退出")

    def wait_until_done(self):
        """等待所有音频播放完成"""
        self.queue.join()
        # 等待当前正在播放的音频完成
        while self.is_playing:
            time.sleep(0.1)

    def get_queue_size(self):
        """获取队列长度"""
        return self.queue.qsize()


class VoiceAssistant:
    """语音助手核心类"""

    def __init__(self):
        # 对话历史格式: [[用户问题1, AI回答1], [用户问题2, AI回答2]]
        self.conversation_history = []
        self.ports = get_config('services')

        # 从配置加载参数
        voice_config = get_config('voice_chat')

        # 音频参数
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000

        # VAD参数
        self.SILENCE_THRESHOLD = voice_config.get('silence_threshold', 500)
        self.SILENCE_DURATION = voice_config.get('silence_duration', 1.5)
        self.MIN_AUDIO_LENGTH = voice_config.get('min_audio_length', 0.5)

        # 音量参数
        self.OUTPUT_VOLUME = voice_config.get('output_volume', 100)

        # 唤醒词参数
        self.WAKE_WORDS = voice_config.get('wake_words', ["小助手", "你好助手", "嘿助手", "小爱"])
        self.WAKE_MODE = voice_config.get('wake_mode', True)
        self.WAKE_REPLY = voice_config.get('wake_reply', "你好，我在")  # 唤醒确认语音

        # 打断词参数
        self.INTERRUPT_MODE = voice_config.get('interrupt_mode', True)
        self.INTERRUPT_WORDS = voice_config.get('interrupt_words', ["停止", "暂停", "别说了"])
        self.INTERRUPT_REPLY = voice_config.get('interrupt_reply', "好的，已停止")  # 打断确认语音
        self.interrupt_flag = False  # 打断标志
        self.interrupt_monitor_thread = None  # 打断监听线程

        # 思考回复参数
        self.THINKING_REPLY = voice_config.get('thinking_reply', "好，我知道了，等我想一下")  # 开始思考确认语音

        # 连续对话参数
        self.CONTINUE_DIALOGUE_TIMEOUT = voice_config.get('continue_dialogue_timeout', 5.0)  # 连续对话超时时间（秒）

        # 音频缓存目录
        self.cache_dir = Path(__file__).parent / "audio_cache"
        self.cache_dir.mkdir(exist_ok=True)

        # 音频缓存（避免重复生成相同的回复音频）
        self.wake_reply_audio_cache = None  # 唤醒回复音频缓存
        self.interrupt_reply_audio_cache = None  # 打断回复音频缓存
        self.thinking_reply_audio_cache = None  # 思考回复音频缓存

        # 情绪监控相关
        emotion_config = get_config('emotion_context', {})
        self.EMOTION_ENABLE = emotion_config.get('enable', False)
        self.emotion_service_url = emotion_config.get('service_url', 'http://localhost:5005')
        self.current_emotion = None
        self.emotion_context = ""

        # 长时记忆相关
        memory_config = get_config('memory_service', {})
        self.MEMORY_ENABLE = memory_config.get('enable', False)
        self.memory_service_url = memory_config.get('service_url', 'http://localhost:5006')
        self.auto_extract = memory_config.get('auto_extract', True)
        self.memory_client = None

        # 初始化记忆客户端（如果启用）
        if self.MEMORY_ENABLE:
            try:
                from memory_service.memory_client import MemoryClient
                self.memory_client = MemoryClient(self.memory_service_url)
                if self.memory_client.check_service():
                    logger.info("✅ 长时记忆服务连接成功")
                else:
                    logger.warning("⚠️ 长时记忆服务未运行")
                    self.MEMORY_ENABLE = False
            except Exception as e:
                logger.error(f"初始化记忆客户端失败: {e}")
                self.MEMORY_ENABLE = False

        # 加载缓存的音频
        self._load_cached_audio()

        # 初始化PyAudio（抑制ALSA错误）
        with suppress_stderr():
            self.audio = pyaudio.PyAudio()

        # 存储设备配置
        self.input_device = voice_config.get('input_device')
        self.output_device = voice_config.get('output_device')

        # 检测并调整输入设备的采样率
        if self.input_device is not None:
            try:
                device_info = self.audio.get_device_info_by_index(self.input_device)
                device_rate = int(device_info['defaultSampleRate'])

                # 尝试测试设备是否支持16000Hz
                try:
                    with suppress_stderr():
                        test_stream = self.audio.open(
                            format=self.FORMAT,
                            channels=self.CHANNELS,
                            rate=16000,
                            input=True,
                            input_device_index=self.input_device,
                            frames_per_buffer=self.CHUNK,
                            start=False
                        )
                        test_stream.close()
                    logger.info(f"设备支持16000Hz采样率")
                except:
                    # 设备不支持16000Hz,使用设备默认采样率
                    logger.warning(f"设备不支持16000Hz采样率,将使用设备默认采样率: {device_rate}Hz")
                    self.RATE = device_rate
            except Exception as e:
                logger.warning(f"无法获取设备信息: {e},使用默认采样率")

        logger.info(f"使用采样率: {self.RATE}Hz")
        logger.info(f"输出音量: {self.OUTPUT_VOLUME}%")
        logger.info("语音助手初始化完成")
        if self.WAKE_MODE:
            logger.info(f"唤醒词模式已启用，支持的唤醒词: {', '.join(self.WAKE_WORDS)}")

    def _get_cache_filename(self, text, cache_type):
        """
        生成缓存文件名（基于文本内容的哈希）

        Args:
            text: 文本内容
            cache_type: 缓存类型 ('wake' 或 'interrupt')

        Returns:
            str: 缓存文件路径
        """
        import hashlib
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:16]
        return self.cache_dir / f"{cache_type}_reply_{text_hash}.pcm"

    def _load_cached_audio(self):
        """加载缓存的音频文件"""
        # 加载唤醒回复音频（使用带"一"前缀的文本）
        wake_text_with_prefix = "一" + self.WAKE_REPLY
        wake_cache_file = self._get_cache_filename(wake_text_with_prefix, 'wake')
        if wake_cache_file.exists():
            self.wake_reply_audio_cache = str(wake_cache_file)
            logger.info(f"✅ 加载唤醒回复音频缓存: {self.WAKE_REPLY}")
        else:
            logger.info(f"💾 唤醒回复音频缓存不存在，将在首次使用时生成")

        # 加载打断回复音频（使用带"一"前缀的文本）
        interrupt_text_with_prefix = "一" + self.INTERRUPT_REPLY
        interrupt_cache_file = self._get_cache_filename(interrupt_text_with_prefix, 'interrupt')
        if interrupt_cache_file.exists():
            self.interrupt_reply_audio_cache = str(interrupt_cache_file)
            logger.info(f"✅ 加载打断回复音频缓存: {self.INTERRUPT_REPLY}")
        else:
            logger.info(f"💾 打断回复音频缓存不存在，将在首次使用时生成")

        # 加载思考回复音频（使用带"一"前缀的文本）
        thinking_text_with_prefix = "一" + self.THINKING_REPLY
        thinking_cache_file = self._get_cache_filename(thinking_text_with_prefix, 'thinking')
        if thinking_cache_file.exists():
            self.thinking_reply_audio_cache = str(thinking_cache_file)
            logger.info(f"✅ 加载思考回复音频缓存: {self.THINKING_REPLY}")
        else:
            logger.info(f"💾 思考回复音频缓存不存在，将在首次使用时生成")

    def get_emotion_context(self):
        """获取当前情绪上下文"""
        if not self.EMOTION_ENABLE:
            return ""

        try:
            # 请求情绪统计接口
            response = requests.get(
                f"{self.emotion_service_url}/emotion/stats",
                timeout=2
            )

            if response.status_code == 200:
                data = response.json()

                if data.get("status") == "success":
                    dominant_emotion = data.get("dominant_emotion", "neutral")
                    confidence = data.get("confidence", 0)

                    # 更新当前情绪
                    self.current_emotion = dominant_emotion

                    # 生成情绪上下文描述
                    emotion_descriptions = {
                        "happy": "用户当前看起来很开心",
                        "sad": "用户当前可能情绪低落",
                        "angry": "用户当前看起来有些生气",
                        "surprise": "用户当前看起来很惊讶",
                        "neutral": "用户当前情绪平静",
                        "fear": "用户当前看起来有些紧张",
                        "disgust": "用户当前看起来有些不悦"
                    }

                    base_desc = emotion_descriptions.get(dominant_emotion, "用户当前情绪平静")

                    # 根据置信度添加描述
                    if confidence > 0.8:
                        return f"{base_desc}（非常确定）"
                    elif confidence > 0.6:
                        return f"{base_desc}（比较确定）"
                    else:
                        return f"{base_desc}（不太确定）"
                else:
                    logger.debug(f"情绪服务返回错误: {data.get('message', 'Unknown error')}")
            else:
                logger.debug(f"情绪服务请求失败: {response.status_code}")

        except requests.exceptions.RequestException as e:
            logger.debug(f"无法连接到情绪服务: {e}")
        except Exception as e:
            logger.error(f"获取情绪上下文时出错: {e}")

        return ""

    def update_emotion_context(self):
        """更新情绪上下文"""
        self.emotion_context = self.get_emotion_context()
        if self.emotion_context:
            logger.info(f"💭 情绪上下文: {self.emotion_context}")

    def _save_audio_cache(self, text, cache_type, audio_file):
        """
        保存音频到缓存

        Args:
            text: 文本内容
            cache_type: 缓存类型 ('wake' 或 'interrupt')
            audio_file: 临时音频文件路径

        Returns:
            str: 缓存文件路径
        """
        try:
            cache_file = self._get_cache_filename(text, cache_type)

            # 复制音频文件到缓存目录
            import shutil
            shutil.copy2(audio_file, cache_file)

            logger.info(f"💾 已保存{cache_type}回复音频缓存: {text} -> {cache_file.name}")
            return str(cache_file)
        except Exception as e:
            logger.error(f"保存音频缓存失败: {e}")
            return None

    def _clear_audio_cache(self):
        """清除所有音频缓存"""
        try:
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(exist_ok=True)
                logger.info("🗑️ 已清除所有音频缓存")

            # 重置缓存变量
            self.wake_reply_audio_cache = None
            self.interrupt_reply_audio_cache = None
        except Exception as e:
            logger.error(f"清除音频缓存失败: {e}")

    def list_audio_devices(self):
        """列出所有音频设备"""
        logger.info("=" * 60)
        logger.info("可用音频设备列表:")
        logger.info("=" * 60)

        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            logger.info(f"设备 {i}: {info['name']}")
            logger.info(f"  输入通道: {info['maxInputChannels']}")
            logger.info(f"  输出通道: {info['maxOutputChannels']}")
            logger.info(f"  采样率: {info['defaultSampleRate']}")
            logger.info("-" * 60)

    def reload_config(self):
        """
        重新加载配置参数（热重载）
        注意：某些配置（如音频设备）的更改可能需要重启语音对话才能生效
        """
        try:
            from config_loader import reload_config as reload_config_file
            reload_config_file()  # 先更新 ConfigLoader 的 config

            voice_config = get_config('voice_chat')
            self.ports = get_config('services')

            # 更新 VAD 参数
            old_threshold = self.SILENCE_THRESHOLD
            self.SILENCE_THRESHOLD = voice_config.get('silence_threshold', 500)
            self.SILENCE_DURATION = voice_config.get('silence_duration', 1.5)
            self.MIN_AUDIO_LENGTH = voice_config.get('min_audio_length', 0.5)

            if old_threshold != self.SILENCE_THRESHOLD:
                logger.info(f"🔄 静音阈值已更新: {old_threshold} → {self.SILENCE_THRESHOLD}")

            # 更新音量参数
            old_volume = self.OUTPUT_VOLUME
            self.OUTPUT_VOLUME = voice_config.get('output_volume', 100)
            if old_volume != self.OUTPUT_VOLUME:
                logger.info(f"🔄 输出音量已更新: {old_volume}% → {self.OUTPUT_VOLUME}%")

            # 更新唤醒词参数
            old_wake_mode = self.WAKE_MODE
            old_wake_words = self.WAKE_WORDS
            old_wake_reply = self.WAKE_REPLY
            self.WAKE_WORDS = voice_config.get('wake_words', ["小助手", "你好助手", "嘿助手", "小爱"])
            self.WAKE_MODE = voice_config.get('wake_mode', True)
            self.WAKE_REPLY = voice_config.get('wake_reply', "你好，我在")

            if old_wake_mode != self.WAKE_MODE:
                logger.info(f"🔄 唤醒词模式已{'启用' if self.WAKE_MODE else '禁用'}")
            if old_wake_words != self.WAKE_WORDS:
                logger.info(f"🔄 唤醒词已更新: {old_wake_words} → {self.WAKE_WORDS}")
            if old_wake_reply != self.WAKE_REPLY:
                logger.info(f"🔄 唤醒回复已更新: {old_wake_reply} → {self.WAKE_REPLY}")
                # 清除旧的缓存
                self.wake_reply_audio_cache = None
                logger.info("🗑️ 已清除唤醒回复音频缓存，将在下次使用时重新生成")

            # 更新打断词参数
            old_interrupt_mode = self.INTERRUPT_MODE
            old_interrupt_words = self.INTERRUPT_WORDS
            old_interrupt_reply = self.INTERRUPT_REPLY
            self.INTERRUPT_MODE = voice_config.get('interrupt_mode', True)
            self.INTERRUPT_WORDS = voice_config.get('interrupt_words', ["停止", "暂停", "别说了"])
            self.INTERRUPT_REPLY = voice_config.get('interrupt_reply', "好的，已停止")

            if old_interrupt_mode != self.INTERRUPT_MODE:
                logger.info(f"🔄 打断模式已{'启用' if self.INTERRUPT_MODE else '禁用'}")
            if old_interrupt_words != self.INTERRUPT_WORDS:
                logger.info(f"🔄 打断词已更新: {old_interrupt_words} → {self.INTERRUPT_WORDS}")
            if old_interrupt_reply != self.INTERRUPT_REPLY:
                logger.info(f"🔄 打断回复已更新: {old_interrupt_reply} → {self.INTERRUPT_REPLY}")
                # 清除旧的缓存
                self.interrupt_reply_audio_cache = None
                logger.info("🗑️ 已清除打断回复音频缓存，将在下次使用时重新生成")

            # 更新思考回复参数
            old_thinking_reply = self.THINKING_REPLY
            self.THINKING_REPLY = voice_config.get('thinking_reply', "好，我知道了，等我想一下")

            if old_thinking_reply != self.THINKING_REPLY:
                logger.info(f"🔄 思考回复已更新: {old_thinking_reply} → {self.THINKING_REPLY}")
                # 清除旧的缓存
                self.thinking_reply_audio_cache = None
                logger.info("🗑️ 已清除思考回复音频缓存，将在下次使用时重新生成")

            # 更新设备配置（注意：设备切换需要重启语音对话才能生效）
            old_input = self.input_device
            old_output = self.output_device
            self.input_device = voice_config.get('input_device')
            self.output_device = voice_config.get('output_device')

            if old_input != self.input_device or old_output != self.output_device:
                logger.warning("⚠️ 音频设备配置已更新，但需要重启语音对话才能生效")

            logger.info("✅ VoiceAssistant 配置已重新加载")

            return {
                "success": True,
                "message": "配置已重新加载",
                "changes": {
                    "silence_threshold": self.SILENCE_THRESHOLD,
                    "output_volume": self.OUTPUT_VOLUME,
                    "wake_mode": self.WAKE_MODE,
                    "wake_words": self.WAKE_WORDS,
                    "wake_reply": self.WAKE_REPLY,
                    "interrupt_mode": self.INTERRUPT_MODE,
                    "interrupt_words": self.INTERRUPT_WORDS,
                    "interrupt_reply": self.INTERRUPT_REPLY,
                    "thinking_reply": self.THINKING_REPLY
                }
            }

        except Exception as e:
            logger.error(f"❌ 配置重新加载失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }

    def get_default_input_device(self):
        """获取默认输入设备"""
        try:
            default_input = self.audio.get_default_input_device_info()
            logger.info(f"默认输入设备: {default_input['name']}")
            return default_input['index']
        except Exception as e:
            logger.error(f"获取默认输入设备失败: {e}")
            return None

    def get_default_output_device(self):
        """获取默认输出设备"""
        try:
            default_output = self.audio.get_default_output_device_info()
            logger.info(f"默认输出设备: {default_output['name']}")
            return default_output['index']
        except Exception as e:
            logger.error(f"获取默认输出设备失败: {e}")
            return None

    def calculate_rms(self, audio_data):
        """计算音频数据的RMS（均方根）值"""
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            # 避免空数组或全零数组导致的问题
            if len(audio_array) == 0:
                return 0
            # 计算RMS，使用float64避免溢出
            rms = np.sqrt(np.mean(audio_array.astype(np.float64) ** 2))
            # 处理NaN情况
            if np.isnan(rms):
                return 0
            return rms
        except Exception as e:
            logger.error(f"计算RMS失败: {e}")
            return 0

    def resample_audio(self, input_file, target_rate=16000):
        """
        重采样音频文件到目标采样率

        Args:
            input_file: 输入音频文件路径
            target_rate: 目标采样率,默认16000Hz (ASR服务要求)

        Returns:
            重采样后的音频文件路径
        """
        # 如果当前采样率就是目标采样率,直接返回
        if self.RATE == target_rate:
            return input_file

        try:
            # 读取原始音频
            with wave.open(input_file, 'rb') as wf:
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                frames = wf.readframes(wf.getnframes())

            # 转换为numpy数组
            audio_data = np.frombuffer(frames, dtype=np.int16)

            # 计算重采样比例
            num_samples = int(len(audio_data) * target_rate / framerate)

            # 使用scipy进行重采样
            resampled_data = signal.resample(audio_data, num_samples)
            resampled_data = resampled_data.astype(np.int16)

            # 保存重采样后的音频
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                output_file = f.name

            with wave.open(output_file, 'wb') as wf:
                wf.setnchannels(n_channels)
                wf.setsampwidth(sampwidth)
                wf.setframerate(target_rate)
                wf.writeframes(resampled_data.tobytes())

            # 不删除原始文件，由调用者决定是否删除

            return output_file

        except Exception as e:
            logger.error(f"音频重采样失败: {e}")
            return input_file

    def check_wake_word(self, text):
        """
        检查文本是否包含唤醒词

        Args:
            text: 识别的文本

        Returns:
            tuple: (是否包含唤醒词, 去除唤醒词后的文本)
        """
        if not text:
            return False, text

        text_lower = text.lower()

        for wake_word in self.WAKE_WORDS:
            if wake_word.lower() in text_lower:
                # 找到唤醒词，去除它
                remaining_text = text.replace(wake_word, "").strip()
                logger.info(f"✅ 检测到唤醒词: {wake_word}")
                return True, remaining_text

        return False, text

    def record_audio_with_vad(self, input_device=None, for_wake_word=False, custom_timeout=None):
        """
        使用VAD录音
        自动检测说话开始和结束

        Args:
            input_device: 输入设备索引
            for_wake_word: 是否用于唤醒词检测（唤醒词录音时间更短）
            custom_timeout: 自定义超时时间（秒），如果不指定则使用默认值
        """
        global assistant_running

        if for_wake_word:
            logger.info("🔍 监听唤醒词...")
            max_duration = 3  # 唤醒词最长3秒
        elif custom_timeout is not None:
            logger.info(f"🎤 准备录音，请开始说话...（最长{custom_timeout}秒）")
            max_duration = custom_timeout
        else:
            logger.info("🎤 准备录音，请开始说话...")
            max_duration = 30  # 正常对话最长30秒

        stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            input_device_index=input_device,
            frames_per_buffer=self.CHUNK
        )

        frames = []
        silent_chunks = 0
        started = False
        max_silent_chunks = int(self.SILENCE_DURATION * self.RATE / self.CHUNK)

        # 记录一些关键信息用于调试
        if not for_wake_word:
            logger.info(f"📊 VAD参数: 静音阈值={self.SILENCE_THRESHOLD}, 静音时长={self.SILENCE_DURATION}秒, 需要静音帧数={max_silent_chunks}")

        # 用于调试的计数器
        debug_counter = 0
        rms_values = []  # 记录最近的RMS值用于调试

        try:
            while assistant_running:  # 检查运行标志
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                frames.append(data)

                # 计算音量
                rms = self.calculate_rms(data)

                # 每隔一定帧数输出调试信息（避免日志过多）
                debug_counter += 1
                rms_values.append(rms)
                if not for_wake_word and debug_counter % 20 == 0:  # 每20帧（约0.5秒）输出一次
                    avg_rms = np.mean(rms_values[-20:]) if rms_values else 0
                    logger.debug(f"音量监测 - 当前RMS: {int(rms)}, 平均RMS: {int(avg_rms)}, 阈值: {self.SILENCE_THRESHOLD}, 已录制: {len(frames)}帧")

                if rms > self.SILENCE_THRESHOLD:
                    if not started:
                        logger.info(f"🗣️ 检测到语音，开始录音... (RMS: {int(rms)} > 阈值: {self.SILENCE_THRESHOLD})")
                        started = True
                    silent_chunks = 0
                else:
                    if started:
                        silent_chunks += 1
                        # 每隔一定帧数输出静音计数（改为INFO级别，更频繁）
                        if not for_wake_word and silent_chunks % 5 == 0:
                            logger.info(f"🔇 静音计数: {silent_chunks}/{max_silent_chunks} 帧 (RMS: {int(rms)}, 已录制: {len(frames)}帧, 时长: {len(frames)*self.CHUNK/self.RATE:.1f}秒)")

                # 检测到足够长的静音，停止录音
                if started and silent_chunks > max_silent_chunks:
                    if not for_wake_word:
                        logger.info(f"✅ 检测到静音，录音结束 (静音持续: {silent_chunks}帧 = {silent_chunks*self.CHUNK/self.RATE:.2f}秒)")
                    break

                # 防止无限录音
                if len(frames) > self.RATE / self.CHUNK * max_duration:
                    if not for_wake_word:
                        logger.warning("⚠️ 录音超时，自动停止")
                    break

        finally:
            stream.stop_stream()
            stream.close()

        # 如果被中断停止，返回None
        if not assistant_running:
            return None

        # 检查录音长度
        audio_duration = len(frames) * self.CHUNK / self.RATE
        if audio_duration < self.MIN_AUDIO_LENGTH:
            if not for_wake_word:
                logger.warning("⚠️ 录音时间过短，忽略")
            return None

        if not for_wake_word:
            logger.info(f"📝 录音完成，时长: {audio_duration:.2f} 秒, 总帧数: {len(frames)}")

        # 保存为临时WAV文件
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name

        wf = wave.open(temp_path, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        return temp_path

    def speech_to_text(self, audio_file):
        """调用ASR服务进行语音识别"""
        resampled_file = None
        try:
            # 重采样到16000Hz (ASR服务要求)
            resampled_file = self.resample_audio(audio_file, target_rate=16000)

            url = f"http://localhost:{self.ports['asr']}/transcribe"

            with open(resampled_file, 'rb') as f:
                files = {'audio': f}
                response = requests.post(url, files=files, timeout=30)

            if response.status_code == 200:
                result = response.json()
                text = result.get('text', '')
                logger.info(f"🗣️ 识别结果: {text}")
                return text
            else:
                logger.error(f"ASR识别失败: {response.text}")
                return None

        except Exception as e:
            logger.error(f"ASR服务调用失败: {e}")
            return None
        finally:
            # 清理重采样后的临时文件
            if resampled_file and os.path.exists(resampled_file):
                try:
                    os.unlink(resampled_file)
                except:
                    pass

    def chat(self, message):
        """调用LLM服务进行对话"""
        try:
            url = f"http://localhost:{self.ports['llm']}/chat"

            payload = {
                "message": message,
                "history": self.conversation_history
            }

            response = requests.post(url, json=payload, timeout=60)

            if response.status_code == 200:
                result = response.json()
                reply = result.get('message', '')

                # 更新对话历史 - 使用二维列表格式
                self.conversation_history.append([message, reply])

                # 自动提取并存储记忆
                if self.MEMORY_ENABLE and self.auto_extract and self.memory_client:
                    self._extract_and_store_memory(message, reply)

                logger.info(f"🤖 AI回复: {reply}")
                return reply
            else:
                logger.error(f"LLM对话失败: {response.text}")
                return "抱歉，我遇到了一些问题。"

        except Exception as e:
            logger.error(f"LLM服务调用失败: {e}")
            return "抱歉，我遇到了一些问题。"

    def text_to_speech(self, text):
        """调用TTS服务进行语音合成"""
        try:
            url = f"http://localhost:{self.ports['tts']}/synthesize"

            payload = {"text": text}
            timeout = max(60, len(text) // 10 + 30)

            response = requests.post(url, json=payload, timeout=timeout)

            if response.status_code == 200:
                # 保存PCM音频
                with tempfile.NamedTemporaryFile(suffix='.pcm', delete=False) as f:
                    f.write(response.content)
                    return f.name
            else:
                logger.error(f"TTS合成失败: {response.text}")
                return None

        except Exception as e:
            logger.error(f"TTS服务调用失败: {e}")
            return None

    def warmup_tts(self):
        """
        预热TTS服务
        在ASR识别期间后台预热，减少首句TTS延迟
        """
        try:
            url = f"http://localhost:{self.ports['tts']}/synthesize/stream"
            payload = {"text": "嗯", "stream": True}

            # 使用短超时，快速预热
            response = requests.post(url, json=payload, stream=True, timeout=5)

            if response.status_code == 200:
                # 只读取少量数据就结束，不需要完整接收
                for _ in response.iter_content(chunk_size=4096):
                    break
                logger.debug("🔥 TTS预热完成")
            else:
                logger.debug(f"TTS预热失败: {response.status_code}")
        except Exception as e:
            logger.debug(f"TTS预热异常(忽略): {e}")


    def chat_stream(self, message, output_device=None):
        """
        流式对话：LLM流式输出 + TTS异步生成和播放
        使用队列实现：播放一句话的同时生成下一句话

        Args:
            message: 用户消息
            output_device: 输出设备索引
        """
        try:
            # 获取情绪上下文
            self.update_emotion_context()

            # 获取记忆上下文
            memory_context = ""
            if self.MEMORY_ENABLE and self.memory_client:
                memory_context = self.memory_client.get_context(message)
                if memory_context:
                    logger.info(f"💭 找到相关记忆: {len(memory_context)} 字符")

            # 组合所有上下文
            context_parts = []
            if self.emotion_context:
                context_parts.append(self.emotion_context)
            if memory_context:
                context_parts.append(memory_context)

            # 如果有上下文，将其添加到消息中
            if context_parts:
                all_context = "。".join(context_parts)
                enhanced_message = f"{all_context}。{message}"
                logger.info(f"🎭 消息已添加上下文")
            else:
                enhanced_message = message

            url = f"http://localhost:{self.ports['llm']}/chat/stream"

            payload = {
                "message": enhanced_message,
                "history": self.conversation_history
            }

            # 使用流式请求
            response = requests.post(url, json=payload, stream=True, timeout=120)

            if response.status_code != 200:
                logger.error(f"LLM流式对话失败: {response.text}")
                return

            # 创建播放队列
            playback_queue = AudioPlaybackQueue(self)
            playback_queue.start(output_device)

            # 重置打断标志
            self.interrupt_flag = False

            # 启动打断监听线程（如果启用）
            if self.INTERRUPT_MODE:
                self.interrupt_monitor_thread = threading.Thread(
                    target=self.monitor_interrupt,
                    args=(self.input_device,),
                    daemon=True
                )
                self.interrupt_monitor_thread.start()
                logger.info("👂 打断监听线程已启动")

            full_reply = ""
            text_buffer = ""
            sentence_delimiters = ["。", "！", "?", "!", "?", "\n", ".", ";"]

            logger.info("🤖 AI开始回复...")

            try:
                # 逐块接收LLM输出
                for line in response.iter_lines():
                    # 检查是否被打断
                    if self.interrupt_flag:
                        logger.info("⏹️ 检测到打断，停止生成内容")
                        break

                    if not line:
                        continue

                    try:
                        # 解析SSE格式
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            data = line[6:]  # 去掉 'data: ' 前缀

                            if data == '[DONE]':
                                break

                            import json
                            chunk_data = json.loads(data)
                            chunk = chunk_data.get('delta', '')

                            if chunk:
                                text_buffer += chunk
                                full_reply += chunk

                                # 检查是否有完整的句子
                                for delimiter in sentence_delimiters:
                                    if delimiter in text_buffer:
                                        # 找到句子结束符，分割句子
                                        sentences = text_buffer.split(delimiter)
                                        for i in range(len(sentences) - 1):
                                            sentence = sentences[i] + delimiter
                                            if sentence.strip():
                                                logger.info(f"📝 生成文本片段: {sentence[:50]}...")

                                                # 异步生成TTS音频
                                                pcm_file = self.text_to_speech(sentence)
                                                if pcm_file:
                                                    # 加入播放队列（不等待播放完成）
                                                    playback_queue.add(pcm_file, sentence)
                                                    logger.debug(f"✅ TTS已生成并加入队列，队列长度: {playback_queue.get_queue_size()}")

                                        # 保留最后一个未完成的部分
                                        text_buffer = sentences[-1]
                                        break

                    except Exception as e:
                        logger.error(f"处理流式数据出错: {e}")
                        continue

                # 处理剩余的文本（如果没有被打断）
                if text_buffer.strip() and not self.interrupt_flag:
                    logger.info(f"📝 生成最后片段: {text_buffer[:50]}...")
                    pcm_file = self.text_to_speech(text_buffer)
                    if pcm_file:
                        playback_queue.add(pcm_file, text_buffer)

                # 等待所有音频播放完成（或被打断）
                if not self.interrupt_flag:
                    logger.info(f"⏳ 等待所有音频播放完成... (队列剩余: {playback_queue.get_queue_size()})")
                    playback_queue.wait_until_done()
                    logger.info("✅ 所有音频播放完成")
                else:
                    logger.info("⏹️ 对话已被打断")
                    # 播放打断确认音频（使用缓存）
                    if self.INTERRUPT_MODE and self.INTERRUPT_REPLY:
                        try:
                            self.quick_reply(self.INTERRUPT_REPLY, output_device)
                        except Exception as e:
                            logger.error(f"播放打断确认音频失败: {e}")

            finally:
                # 停止打断监听
                self.interrupt_flag = True  # 确保监听线程停止

                # 停止播放队列
                playback_queue.stop()

            # 更新对话历史 - 使用二维列表格式
            if full_reply and not self.interrupt_flag:  # 只有在有回复且未被打断时才添加到历史
                self.conversation_history.append([message, full_reply])

            logger.info(f"✅ 完整回复: {full_reply[:100]}..." if len(full_reply) > 100 else f"✅ 完整回复: {full_reply}")

        except Exception as e:
            logger.error(f"流式对话失败: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def monitor_interrupt(self, input_device=None):
        """
        监听打断词的后台线程（简化版）
        在AI播放时持续监听用户是否说话（基于音量检测）
        当检测到说话时，进行ASR识别检查是否为打断词
        """
        if not self.INTERRUPT_MODE:
            return

        logger.info("👂 开始监听打断（音量检测模式）...")

        try:
            # 使用独立的PyAudio实例避免冲突
            with suppress_stderr():
                monitor_audio = pyaudio.PyAudio()

            # 打开音频流（持续监听）
            stream = monitor_audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                input_device_index=input_device,
                frames_per_buffer=self.CHUNK
            )

            logger.info("👂 打断监听音频流已打开")
            silent_threshold = self.SILENCE_THRESHOLD * 2  # 需要比静音阈值高，才认为是说话

            while not self.interrupt_flag:
                try:
                    # 读取音频数据
                    data = stream.read(self.CHUNK, exception_on_overflow=False)

                    # 计算RMS
                    audio_array = np.frombuffer(data, dtype=np.int16)
                    if len(audio_array) > 0:
                        rms = int(np.sqrt(np.mean(audio_array.astype(np.float64) ** 2)))

                        # 检测到说话（RMS超过阈值）
                        if rms > silent_threshold:
                            logger.debug(f"👂 检测到声音，RMS: {rms} > {silent_threshold}")

                            # 录制完整的话（用于ASR识别）
                            # 暂时关闭流
                            stream.stop_stream()
                            stream.close()

                            # 录制一段音频用于识别
                            logger.debug("🎤 录制音频进行打断词识别...")
                            frames = [data]  # 包含刚才检测到的数据

                            # 继续录制1秒
                            temp_stream = monitor_audio.open(
                                format=self.FORMAT,
                                channels=self.CHANNELS,
                                rate=self.RATE,
                                input=True,
                                input_device_index=input_device,
                                frames_per_buffer=self.CHUNK
                            )

                            for _ in range(int(self.RATE / self.CHUNK * 1.0)):  # 1秒
                                data = temp_stream.read(self.CHUNK, exception_on_overflow=False)
                                frames.append(data)

                            temp_stream.stop_stream()
                            temp_stream.close()

                            # 保存音频并识别
                            temp_file = f"/tmp/interrupt_detect_{int(time.time() * 1000)}.wav"
                            wf = wave.open(temp_file, 'wb')
                            wf.setnchannels(self.CHANNELS)
                            wf.setsampwidth(monitor_audio.get_sample_size(self.FORMAT))
                            wf.setframerate(self.RATE)
                            wf.writeframes(b''.join(frames))
                            wf.close()

                            # ASR识别
                            text = self.speech_to_text(temp_file)

                            # 清理临时文件
                            try:
                                os.unlink(temp_file)
                            except:
                                pass

                            if text:
                                logger.debug(f"👂 监听到: {text}")

                                # 检查是否包含打断词
                                for interrupt_word in self.INTERRUPT_WORDS:
                                    if interrupt_word in text:
                                        logger.info(f"🛑 检测到打断词: {interrupt_word}")
                                        self.interrupt_flag = True
                                        monitor_audio.terminate()
                                        return

                            # 重新打开流继续监听
                            if not self.interrupt_flag:
                                stream = monitor_audio.open(
                                    format=self.FORMAT,
                                    channels=self.CHANNELS,
                                    rate=self.RATE,
                                    input=True,
                                    input_device_index=input_device,
                                    frames_per_buffer=self.CHUNK
                                )

                except Exception as e:
                    logger.error(f"监听循环出错: {e}")
                    break

            # 清理资源
            try:
                stream.stop_stream()
                stream.close()
            except:
                pass

            monitor_audio.terminate()
            logger.info("👂 打断监听已停止")

        except Exception as e:
            logger.error(f"打断监听失败: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def monitor_wake_word(self, input_device=None):
        """
        实时连续监听唤醒词
        类似于打断词监听，持续录音并识别是否包含唤醒词

        Args:
            input_device: 输入设备索引

        Returns:
            tuple: (has_wake_word, remaining_text) 是否检测到唤醒词和剩余文本
        """
        try:
            logger.info("👂 开始连续监听唤醒词...")

            # 创建独立的PyAudio实例用于监听
            with suppress_stderr():
                monitor_audio = pyaudio.PyAudio()

            stream = monitor_audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                input_device_index=input_device,
                frames_per_buffer=self.CHUNK
            )

            logger.info("👂 唤醒词监听音频流已打开")

            frames = []
            is_speaking = False
            silent_chunks = 0
            # 使用配置的静音持续时间,用于唤醒词检测
            max_silent_chunks = int(self.RATE / self.CHUNK * self.SILENCE_DURATION)
            chunk_counter = 0  # 用于定期输出状态

            silent_threshold = self.SILENCE_THRESHOLD * 0.8  # 唤醒词检测使用较低阈值,更灵敏

            while assistant_running and not self.interrupt_flag:
                try:
                    # 读取音频数据
                    data = stream.read(self.CHUNK, exception_on_overflow=False)
                    frames.append(data)
                    chunk_counter += 1

                    # 计算音量(RMS) - 使用相同的方法
                    rms = self.calculate_rms(data)

                    # 每50帧(约1秒)输出一次状态
                    if chunk_counter % 50 == 0:
                        logger.info(f"🎤 监听中... RMS={int(rms)}, 阈值={int(silent_threshold)}, 说话={is_speaking}, 静音帧={silent_chunks}")

                    # 检测是否在说话
                    if rms > silent_threshold:
                        if not is_speaking:
                            logger.info(f"🗣️ 检测到说话开始 (RMS={int(rms)} > {int(silent_threshold)})")
                        is_speaking = True
                        silent_chunks = 0
                    elif is_speaking:
                        silent_chunks += 1
                        if silent_chunks % 10 == 0:  # 每10帧输出一次
                            logger.info(f"🔇 静音计数: {silent_chunks}/{max_silent_chunks}")

                    # 如果说话后静音超过阈值，进行识别
                    if is_speaking and silent_chunks >= max_silent_chunks:
                        logger.info(f"🎤 检测到语音结束，开始识别... (累计 {len(frames)} 帧, 约{len(frames)*self.CHUNK/self.RATE:.1f}秒)")

                        # 暂停流
                        stream.stop_stream()
                        stream.close()

                        # 保存音频并识别
                        if len(frames) > 0:
                            temp_file = f"/tmp/wake_monitor_{int(time.time() * 1000)}.wav"
                            wf = wave.open(temp_file, 'wb')
                            wf.setnchannels(self.CHANNELS)
                            wf.setsampwidth(monitor_audio.get_sample_size(self.FORMAT))
                            wf.setframerate(self.RATE)
                            wf.writeframes(b''.join(frames))
                            wf.close()

                            # ASR识别
                            text = self.speech_to_text(temp_file)

                            # 清理临时文件
                            try:
                                os.unlink(temp_file)
                            except:
                                pass

                            if text:
                                logger.info(f"👂 监听到: {text}")

                                # 检查是否包含唤醒词
                                has_wake_word, remaining_text = self.check_wake_word(text)

                                if has_wake_word:
                                    logger.info(f"🎯 检测到唤醒词! 剩余文本: {remaining_text}")
                                    # 清理资源
                                    monitor_audio.terminate()
                                    return True, remaining_text
                                else:
                                    logger.info(f"❌ 未检测到唤醒词，继续监听...")
                            else:
                                logger.info(f"⚠️ 识别结果为空，继续监听...")

                        # 重置状态，继续监听
                        frames = []
                        is_speaking = False
                        silent_chunks = 0

                        # 重新打开流继续监听
                        if assistant_running and not self.interrupt_flag:
                            stream = monitor_audio.open(
                                format=self.FORMAT,
                                channels=self.CHANNELS,
                                rate=self.RATE,
                                input=True,
                                input_device_index=input_device,
                                frames_per_buffer=self.CHUNK
                            )

                    # 限制缓冲区大小，避免无限累积（最多保留5秒）
                    max_frames = int(self.RATE / self.CHUNK * 5)
                    if len(frames) > max_frames:
                        frames = frames[-max_frames:]

                except Exception as e:
                    logger.error(f"监听循环出错: {e}")
                    break

            # 清理资源
            try:
                stream.stop_stream()
                stream.close()
            except:
                pass

            monitor_audio.terminate()
            logger.info("👂 唤醒词监听已停止")
            return False, ""

        except Exception as e:
            logger.error(f"唤醒词监听失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False, ""

    def record_audio_short(self, input_device=None, duration=2.0):
        """
        录制短音频（用于打断词检测）
        不使用VAD，直接录制指定时长

        Args:
            input_device: 输入设备索引
            duration: 录制时长（秒）

        Returns:
            str: 录制的音频文件路径
        """
        try:
            stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                input_device_index=input_device,
                frames_per_buffer=self.CHUNK
            )

            frames = []
            num_chunks = int(self.RATE / self.CHUNK * duration)

            for _ in range(num_chunks):
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                frames.append(data)

            stream.stop_stream()
            stream.close()

            # 保存为临时文件
            temp_file = f"/tmp/interrupt_detect_{int(time.time() * 1000)}.wav"
            wf = wave.open(temp_file, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            return temp_file

        except Exception as e:
            logger.error(f"短音频录制失败: {e}")
            return None

    def quick_reply(self, text, output_device=None, use_cache=True):
        """
        快速响应：直接合成并播放指定文本（支持缓存）

        Args:
            text: 要播放的文本
            output_device: 输出设备索引
            use_cache: 是否使用缓存（默认True）
        """
        try:
            logger.info(f"💬 快速回复: {text}")

            pcm_file = None
            cache_type = None
            should_cleanup = True  # 是否需要清理临时文件

            # 判断是唤醒回复、打断回复还是思考回复，以使用对应的缓存
            if use_cache:
                if text == self.WAKE_REPLY and self.wake_reply_audio_cache:
                    # 检查缓存文件是否真的存在
                    if os.path.exists(self.wake_reply_audio_cache):
                        pcm_file = self.wake_reply_audio_cache
                        should_cleanup = False
                        logger.info(f"🎵 使用唤醒回复音频缓存")
                    else:
                        logger.warning(f"⚠️ 唤醒回复缓存文件不存在，重新生成")
                        self.wake_reply_audio_cache = None
                elif text == self.INTERRUPT_REPLY and self.interrupt_reply_audio_cache:
                    # 检查缓存文件是否真的存在
                    if os.path.exists(self.interrupt_reply_audio_cache):
                        pcm_file = self.interrupt_reply_audio_cache
                        should_cleanup = False
                        logger.info(f"🎵 使用打断回复音频缓存")
                    else:
                        logger.warning(f"⚠️ 打断回复缓存文件不存在，重新生成")
                        self.interrupt_reply_audio_cache = None
                elif text == self.THINKING_REPLY and self.thinking_reply_audio_cache:
                    # 检查缓存文件是否真的存在
                    if os.path.exists(self.thinking_reply_audio_cache):
                        pcm_file = self.thinking_reply_audio_cache
                        should_cleanup = False
                        logger.info(f"🎵 使用思考回复音频缓存")
                    else:
                        logger.warning(f"⚠️ 思考回复缓存文件不存在，重新生成")
                        self.thinking_reply_audio_cache = None

            # 如果没有缓存，则生成新的音频
            if not pcm_file:
                # 对唤醒、思考、打断回复添加"一"前缀
                tts_text = text
                cache_key = text  # 用于缓存文件名的key
                if text in [self.WAKE_REPLY, self.THINKING_REPLY, self.INTERRUPT_REPLY]:
                    tts_text = "一" + text
                    cache_key = tts_text  # 使用带前缀的文本作为缓存key
                    logger.debug(f"🔤 添加前缀: {tts_text}")

                pcm_file = self.text_to_speech(tts_text)
                if not pcm_file:
                    logger.error("TTS生成失败")
                    return

                # 保存到缓存（使用带前缀的文本作为key）
                if use_cache:
                    if text == self.WAKE_REPLY:
                        cache_type = 'wake'
                        cached_file = self._save_audio_cache(cache_key, cache_type, pcm_file)
                        if cached_file:
                            self.wake_reply_audio_cache = cached_file
                    elif text == self.INTERRUPT_REPLY:
                        cache_type = 'interrupt'
                        cached_file = self._save_audio_cache(cache_key, cache_type, pcm_file)
                        if cached_file:
                            self.interrupt_reply_audio_cache = cached_file
                    elif text == self.THINKING_REPLY:
                        cache_type = 'thinking'
                        cached_file = self._save_audio_cache(cache_key, cache_type, pcm_file)
                        if cached_file:
                            self.thinking_reply_audio_cache = cached_file

            # 播放音频
            if pcm_file:
                self.play_audio(pcm_file, output_device)

                # 清理临时文件（缓存文件不清理）
                if should_cleanup:
                    try:
                        os.unlink(pcm_file)
                    except:
                        pass

        except Exception as e:
            logger.error(f"快速回复失败: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def play_audio(self, pcm_file, output_device=None):
        """
        播放PCM音频到指定输出设备（蓝牙音箱）
        优先使用paplay（PulseAudio）确保蓝牙兼容性
        """
        try:
            logger.info("🔊 开始播放音频...")

            # PCM参数（与TTS服务一致）
            RATE = 22050
            CHANNELS = 1
            FORMAT = pyaudio.paInt16

            # 方法1: 优先使用paplay (PulseAudio) - 对蓝牙支持最好
            try:
                # 将PCM转换为WAV格式（paplay需要WAV格式）
                wav_file = pcm_file.replace('.pcm', '.wav')

                # 读取PCM数据
                with open(pcm_file, 'rb') as f:
                    pcm_data = f.read()

                # 应用音量调整
                if self.OUTPUT_VOLUME < 100:
                    # 将PCM数据转换为numpy数组
                    audio_array = np.frombuffer(pcm_data, dtype=np.int16)
                    # 应用音量（0-100% 对应 0.0-1.0）
                    volume_factor = self.OUTPUT_VOLUME / 100.0
                    audio_array = (audio_array * volume_factor).astype(np.int16)
                    pcm_data = audio_array.tobytes()
                    logger.info(f"应用音量调整: {self.OUTPUT_VOLUME}%")

                # 写入WAV文件
                import wave
                with wave.open(wav_file, 'wb') as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(2)  # 16-bit = 2 bytes
                    wf.setframerate(RATE)
                    wf.writeframes(pcm_data)

                # 使用paplay播放（会自动使用系统默认输出设备，包括蓝牙音箱）
                paplay_cmd = ['paplay', wav_file]

                # 如果指定了输出设备，需要查找对应的PulseAudio sink名称
                if output_device is not None:
                    try:
                        # 获取PyAudio设备信息
                        device_info = self.audio.get_device_info_by_index(output_device)
                        device_name = device_info['name']

                        # 查找对应的PulseAudio sink
                        # 这里我们使用默认sink，因为已经通过Web界面设置了
                        logger.info(f"目标输出设备: {device_name}")

                    except Exception as e:
                        logger.warning(f"无法获取设备信息，使用默认输出: {e}")

                # 执行播放
                result = subprocess.run(
                    paplay_cmd,
                    capture_output=True,
                    timeout=30
                )

                if result.returncode == 0:
                    logger.info("✅ 音频播放完成 (使用paplay)")
                    # 清理临时文件
                    os.unlink(pcm_file)
                    os.unlink(wav_file)
                    return
                else:
                    logger.warning(f"paplay播放失败: {result.stderr.decode('utf-8', errors='ignore')}")
                    # 继续尝试PyAudio方式

            except FileNotFoundError:
                logger.warning("未找到paplay命令，尝试使用PyAudio播放")
            except Exception as e:
                logger.warning(f"paplay播放出错: {e}，尝试使用PyAudio播放")

            # 方法2: 使用PyAudio播放（备用方案）
            logger.info("使用PyAudio播放音频...")

            # 读取PCM数据
            with open(pcm_file, 'rb') as f:
                pcm_data = f.read()

            # 应用音量调整
            if self.OUTPUT_VOLUME < 100:
                audio_array = np.frombuffer(pcm_data, dtype=np.int16)
                volume_factor = self.OUTPUT_VOLUME / 100.0
                audio_array = (audio_array * volume_factor).astype(np.int16)
                pcm_data = audio_array.tobytes()
                logger.info(f"应用音量调整: {self.OUTPUT_VOLUME}%")

            # 打开音频流
            stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                output=True,
                output_device_index=output_device,
                frames_per_buffer=1024
            )

            # 播放音频
            stream.write(pcm_data)

            # 关闭流
            stream.stop_stream()
            stream.close()

            logger.info("✅ 音频播放完成 (使用PyAudio)")

            # 删除临时文件
            os.unlink(pcm_file)

        except Exception as e:
            logger.error(f"音频播放失败: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _extract_and_store_memory(self, user_message: str, ai_response: str):
        """
        提取并存储对话记忆

        Args:
            user_message: 用户消息
            ai_response: AI回复
        """
        try:
            # 调用记忆服务自动提取记忆
            extracted = self.memory_client.auto_extract(user_message, ai_response)

            if extracted:
                total_extracted = sum(extracted.values())
                logger.info(f"🧠 记忆提取: 偏好={extracted.get('preferences_found', 0)}, "
                          f"事实={extracted.get('facts_found', 0)}, "
                          f"事件={extracted.get('events_found', 0)}")

                # 如果提取到了重要信息，记录更详细的信息
                if total_extracted > 0:
                    logger.debug(f"📝 对话记忆已存储: 用户='{user_message[:50]}...' "
                               f"助手='{ai_response[:50]}...'")

        except Exception as e:
            logger.error(f"记忆提取失败: {e}")
            # 不影响对话流程，静默处理错误

    def run(self, input_device=None, output_device=None):
        """
        运行语音对话循环

        Args:
            input_device: 输入设备索引（USB麦克风），None使用默认
            output_device: 输出设备索引（蓝牙音箱），None使用默认
        """
        global assistant_running

        logger.info("=" * 60)
        logger.info("🤖 线下语音对话系统启动")
        logger.info("=" * 60)

        # 显示设备信息
        if input_device is None:
            input_device = self.get_default_input_device()
        if output_device is None:
            output_device = self.get_default_output_device()

        logger.info(f"使用输入设备: {input_device}")
        logger.info(f"使用输出设备: {output_device}")
        logger.info("按 Ctrl+C 退出程序")
        logger.info("=" * 60)

        try:
            while assistant_running:  # 改为检查assistant_running标志
                # 唤醒词模式
                if self.WAKE_MODE:
                    # 重置打断标志(唤醒词监听阶段不应该被打断)
                    self.interrupt_flag = False

                    # 1. 实时连续监听唤醒词（类似打断词的实现）
                    has_wake_word, remaining_text = self.monitor_wake_word(input_device)

                    if not has_wake_word:
                        # 没有检测到唤醒词或者被中断，继续下一轮
                        continue

                    logger.info("🎯 已唤醒！")

                    # 2. 立即播放确认语音
                    self.quick_reply(self.WAKE_REPLY, output_device)

                    # 3. 检查唤醒词后面是否有内容
                    prefix_text = ""
                    if remaining_text and remaining_text.strip():
                        # 唤醒词后面已经有部分内容，保存起来
                        prefix_text = remaining_text.strip()
                        logger.info(f"📌 检测到前缀内容: {prefix_text}")

                    # 4. 重新录音等待完整问题（使用完整的静音检测时长）
                    logger.info("💬 请说出您的问题...")
                    dialogue_audio = self.record_audio_with_vad(input_device, for_wake_word=False)

                    if dialogue_audio is None:
                        # 如果没有录到新内容，但有前缀内容，可以直接使用
                        if prefix_text:
                            user_text = prefix_text
                            logger.info(f"📝 使用前缀内容作为问题: {user_text}")
                        else:
                            logger.warning("⚠️ 未检测到语音，重新监听唤醒词")
                            continue
                    else:
                        # 5. 识别新录音的内容
                        new_text = self.speech_to_text(dialogue_audio)
                        os.unlink(dialogue_audio)

                        if not new_text or not new_text.strip():
                            # 新录音为空，使用前缀内容（如果有）
                            if prefix_text:
                                user_text = prefix_text
                                logger.info(f"📝 新录音为空，使用前缀内容: {user_text}")
                            else:
                                logger.warning("⚠️ 识别结果为空，重新监听唤醒词")
                                continue
                        else:
                            # 6. 合并前缀内容和新内容
                            if prefix_text:
                                # 有前缀内容，拼接起来
                                user_text = f"{prefix_text}，{new_text.strip()}"
                                logger.info(f"🔗 合并内容: 前缀'{prefix_text}' + 新内容'{new_text.strip()}' = '{user_text}'")
                            else:
                                # 没有前缀内容，直接使用新内容
                                user_text = new_text.strip()

                    logger.info(f"📝 完整问题: {user_text}")

                    # 7. 播放思考确认语音的同时，后台预热TTS
                    warmup_thread = threading.Thread(target=self.warmup_tts, daemon=True)
                    warmup_thread.start()

                    self.quick_reply(self.THINKING_REPLY, output_device)

                    # 8. 使用流式对话：LLM流式输出 + TTS流式播放
                    self.chat_stream(user_text, output_device)

                    # 9. 对话完成后，进入连续对话模式（等待用户继续提问，无需再次唤醒）
                    while True:
                        logger.info(f"💬 等待继续对话（{self.CONTINUE_DIALOGUE_TIMEOUT}秒内无语音将返回待机）...")

                        # 尝试录音，使用配置的超时时间
                        continue_audio = self.record_audio_with_vad(
                            input_device,
                            for_wake_word=False,
                            custom_timeout=self.CONTINUE_DIALOGUE_TIMEOUT
                        )

                        if continue_audio is None:
                            # 没有录到音频，返回待机模式
                            logger.info("⏸️ 未检测到继续对话，返回待机模式")
                            break

                        # 识别新的问题
                        continue_text = self.speech_to_text(continue_audio)
                        os.unlink(continue_audio)

                        if not continue_text or not continue_text.strip():
                            logger.info("⏸️ 识别结果为空，返回待机模式")
                            break

                        logger.info(f"📝 继续对话: {continue_text}")

                        # 播放思考确认语音的同时，后台预热TTS
                        warmup_thread = threading.Thread(target=self.warmup_tts, daemon=True)
                        warmup_thread.start()

                        self.quick_reply(self.THINKING_REPLY, output_device)

                        # 继续对话
                        self.chat_stream(continue_text, output_device)

                else:
                    # 非唤醒词模式，直接录音
                    audio_file = self.record_audio_with_vad(input_device)

                    if audio_file is None:
                        continue

                    # 语音识别
                    user_text = self.speech_to_text(audio_file)
                    os.unlink(audio_file)

                    if not user_text or not user_text.strip():
                        logger.warning("⚠️ 识别结果为空，请重试")
                        continue

                    logger.info(f"📝 用户问题: {user_text}")

                    # 播放思考确认语音的同时，后台预热TTS
                    warmup_thread = threading.Thread(target=self.warmup_tts, daemon=True)
                    warmup_thread.start()

                    self.quick_reply(self.THINKING_REPLY, output_device)

                    # 使用流式对话：LLM流式输出 + TTS流式播放
                    self.chat_stream(user_text, output_device)

                logger.info("-" * 60)
                if self.WAKE_MODE:
                    logger.info("💤 对话结束，等待下次唤醒...")
                    logger.info("-" * 60)

        except KeyboardInterrupt:
            logger.info("\n👋 用户退出，再见！")
        finally:
            # 不再调用terminate()，让PyAudio对象保持可用
            # 这样可以重复启动/停止语音对话
            logger.info("🛑 语音对话循环已退出")


# 全局语音助手实例
assistant = None
assistant_thread = None
assistant_running = False


@app.get("/devices")
async def get_audio_devices():
    """获取所有音频设备列表"""
    try:
        with suppress_stderr():
            audio = pyaudio.PyAudio()
            devices = []

            for i in range(audio.get_device_count()):
                info = audio.get_device_info_by_index(i)
                devices.append({
                    "index": i,
                    "name": info['name'],
                    "max_input_channels": info['maxInputChannels'],
                    "max_output_channels": info['maxOutputChannels'],
                    "default_sample_rate": info['defaultSampleRate']
                })

            audio.terminate()

        return {
            "success": True,
            "devices": devices
        }
    except Exception as e:
        logger.error(f"获取音频设备列表失败: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/start")
async def start_voice_chat():
    """启动语音对话"""
    global assistant, assistant_thread, assistant_running

    try:
        # 检查是否已启用
        voice_config = get_config('voice_chat')
        if not voice_config.get('enable', False):
            return {
                "success": False,
                "message": "语音对话未启用，请在配置中启用"
            }

        # 检查是否已经在运行
        if assistant_running:
            return {
                "success": False,
                "message": "语音对话已经在运行中"
            }

        # 创建助手实例
        assistant = VoiceAssistant()

        # 获取设备配置
        input_device = voice_config.get('input_device')
        output_device = voice_config.get('output_device')

        # 在后台线程运行
        def run_assistant():
            global assistant_running
            assistant_running = True
            try:
                assistant.run(input_device=input_device, output_device=output_device)
            except Exception as e:
                logger.error(f"语音对话运行出错: {e}")
            finally:
                assistant_running = False

        assistant_thread = threading.Thread(target=run_assistant, daemon=True)
        assistant_thread.start()

        return {
            "success": True,
            "message": "语音对话已启动"
        }
    except Exception as e:
        logger.error(f"启动语音对话失败: {e}")
        assistant_running = False
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/stop")
async def stop_voice_chat():
    """停止语音对话"""
    global assistant, assistant_running

    try:
        if not assistant_running:
            return {
                "success": False,
                "message": "语音对话未在运行"
            }

        # 只设置标志位为False，让run函数自然退出
        # 不要调用terminate()，因为这会使PyAudio对象失效
        assistant_running = False

        return {
            "success": True,
            "message": "语音对话已停止"
        }
    except Exception as e:
        logger.error(f"停止语音对话失败: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/status")
async def get_status():
    """获取语音对话状态"""
    return {
        "running": assistant_running,
        "enabled": get_config('voice_chat').get('enable', False)
    }


@app.post("/reload_config")
async def reload_config_endpoint():
    """
    重新加载配置
    如果语音对话正在运行，会热更新配置参数
    如果未运行，只重新加载配置文件
    """
    global assistant

    try:
        if assistant and assistant_running:
            # 语音对话正在运行，调用实例的 reload_config 方法
            result = assistant.reload_config()
            return result
        else:
            # 语音对话未运行，只重新加载配置文件
            from config_loader import reload_config
            reload_config()
            return {
                "success": True,
                "message": "配置文件已重新加载（语音对话未运行）"
            }
    except Exception as e:
        logger.error(f"配置重新加载失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "service": "voice_chat",
        "running": assistant_running,
        "enabled": get_config('voice_chat').get('enable', False)
    }


# 全局变量用于音量监测
volume_monitor_running = False
volume_monitor_thread = None
latest_volume_data = {
    "current_rms": 0,
    "min_rms": 99999,
    "max_rms": 0,
    "avg_rms": 0,
    "samples": [],
    "recommended_threshold": 0
}


def calculate_recommended_threshold(samples, percentile=80):
    """
    基于样本数据计算推荐的静音阈值

    Args:
        samples: RMS样本列表
        percentile: 百分位数（默认80%，即高于80%的环境噪音）

    Returns:
        推荐的阈值
    """
    if not samples or len(samples) < 5:
        return 0

    sorted_samples = sorted(samples)
    index = int(len(sorted_samples) * percentile / 100)
    base_threshold = sorted_samples[index]

    # 添加30%的安全边际，确保能够检测到说话
    recommended = int(base_threshold * 1.3)

    return recommended


def volume_monitor_worker(input_device=None, duration=10):
    """
    后台线程：监测麦克风音量

    Args:
        input_device: 输入设备索引
        duration: 监测持续时间（秒）
    """
    global volume_monitor_running, latest_volume_data

    try:
        logger.info(f"🎤 开始音量监测，持续 {duration} 秒...")

        # 重置数据
        latest_volume_data = {
            "current_rms": 0,
            "min_rms": 99999,
            "max_rms": 0,
            "avg_rms": 0,
            "samples": [],
            "recommended_threshold": 0
        }

        with suppress_stderr():
            audio = pyaudio.PyAudio()

        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1

        # 获取设备信息，检测支持的采样率
        if input_device is not None:
            device_info = audio.get_device_info_by_index(input_device)
        else:
            device_info = audio.get_default_input_device_info()

        # 尝试多个常用采样率，找到设备支持的
        supported_rates = [16000, 44100, 48000, 22050, 8000]
        RATE = None

        for rate in supported_rates:
            try:
                # 测试是否支持该采样率
                with suppress_stderr():
                    test_stream = audio.open(
                        format=FORMAT,
                        channels=CHANNELS,
                        rate=rate,
                        input=True,
                        input_device_index=input_device,
                        frames_per_buffer=CHUNK
                    )
                    test_stream.close()
                RATE = rate
                logger.info(f"✅ 使用采样率: {RATE} Hz")
                break
            except Exception:
                continue

        if RATE is None:
            # 如果都不支持，使用设备默认采样率
            RATE = int(device_info.get('defaultSampleRate', 44100))
            logger.warning(f"⚠️ 使用设备默认采样率: {RATE} Hz")

        # 打开音频流
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=input_device,
            frames_per_buffer=CHUNK
        )

        start_time = time.time()
        sample_count = 0

        while volume_monitor_running and (time.time() - start_time) < duration:
            # 读取音频数据
            audio_data = stream.read(CHUNK, exception_on_overflow=False)

            # 计算RMS
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            if len(audio_array) > 0:
                rms = np.sqrt(np.mean(audio_array.astype(np.float64) ** 2))
                if not np.isnan(rms):
                    rms = int(rms)

                    # 更新统计数据
                    latest_volume_data["current_rms"] = rms
                    latest_volume_data["samples"].append(rms)

                    if rms < latest_volume_data["min_rms"]:
                        latest_volume_data["min_rms"] = rms
                    if rms > latest_volume_data["max_rms"]:
                        latest_volume_data["max_rms"] = rms

                    # 每10个样本计算一次平均值和推荐阈值
                    sample_count += 1
                    if sample_count % 10 == 0:
                        latest_volume_data["avg_rms"] = int(np.mean(latest_volume_data["samples"]))
                        latest_volume_data["recommended_threshold"] = calculate_recommended_threshold(
                            latest_volume_data["samples"]
                        )

            time.sleep(0.05)  # 50ms间隔

        # 清理
        stream.stop_stream()
        stream.close()
        audio.terminate()

        # 最终计算
        if latest_volume_data["samples"]:
            latest_volume_data["avg_rms"] = int(np.mean(latest_volume_data["samples"]))
            latest_volume_data["recommended_threshold"] = calculate_recommended_threshold(
                latest_volume_data["samples"]
            )

        logger.info(f"✅ 音量监测完成")
        logger.info(f"  平均RMS: {latest_volume_data['avg_rms']}")
        logger.info(f"  范围: {latest_volume_data['min_rms']} - {latest_volume_data['max_rms']}")
        logger.info(f"  推荐阈值: {latest_volume_data['recommended_threshold']}")

    except Exception as e:
        logger.error(f"音量监测失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        volume_monitor_running = False


@app.post("/volume/start")
async def start_volume_monitor(input_device: int = None, duration: int = 10):
    """
    开始监测麦克风音量

    Args:
        input_device: 输入设备索引（null使用默认设备）
        duration: 监测持续时间（秒，默认10秒）
    """
    global volume_monitor_running, volume_monitor_thread

    try:
        if volume_monitor_running:
            return {
                "success": False,
                "message": "音量监测已在运行中"
            }

        # 获取配置的输入设备
        if input_device is None:
            voice_config = get_config('voice_chat')
            input_device = voice_config.get('input_device')

        # 启动监测线程
        volume_monitor_running = True
        volume_monitor_thread = threading.Thread(
            target=volume_monitor_worker,
            args=(input_device, duration),
            daemon=True
        )
        volume_monitor_thread.start()

        return {
            "success": True,
            "message": f"音量监测已启动，持续 {duration} 秒",
            "duration": duration
        }

    except Exception as e:
        logger.error(f"启动音量监测失败: {e}")
        volume_monitor_running = False
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/volume/stop")
async def stop_volume_monitor():
    """停止音量监测"""
    global volume_monitor_running

    try:
        if not volume_monitor_running:
            return {
                "success": False,
                "message": "音量监测未在运行"
            }

        volume_monitor_running = False

        return {
            "success": True,
            "message": "音量监测已停止"
        }

    except Exception as e:
        logger.error(f"停止音量监测失败: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/volume/data")
async def get_volume_data():
    """获取当前音量监测数据"""
    global latest_volume_data, volume_monitor_running

    return {
        "success": True,
        "running": volume_monitor_running,
        "data": {
            "current_rms": latest_volume_data["current_rms"],
            "min_rms": latest_volume_data["min_rms"] if latest_volume_data["min_rms"] != 99999 else 0,
            "max_rms": latest_volume_data["max_rms"],
            "avg_rms": latest_volume_data["avg_rms"],
            "sample_count": len(latest_volume_data["samples"]),
            "recommended_threshold": latest_volume_data["recommended_threshold"]
        }
    }


def auto_start_voice_chat():
    """自动启动语音对话（在后台线程中调用API）"""
    import time
    # 等待API服务器完全启动
    time.sleep(2)

    try:
        voice_config = get_config('voice_chat')
        if voice_config.get('enable', False):
            logger.info("🤖 配置已启用，自动启动语音对话...")
            # 调用内部启动函数
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(start_voice_chat())
            if result.get('success'):
                logger.info("✅ 语音对话已自动启动")
            else:
                logger.warning(f"⚠️ 自动启动失败: {result.get('message', result.get('error'))}")
        else:
            logger.info("ℹ️ 语音对话未启用，仅运行API服务器")
    except Exception as e:
        logger.error(f"自动启动语音对话失败: {e}")


def main():
    """主函数 - 启动API服务器"""
    # 默认运行API服务器模式
    port = get_config('services').get('voice_chat', 5004)
    logger.info(f"🚀 启动语音对话API服务器，端口: {port}")

    # 在后台线程启动自动启动逻辑
    auto_start_thread = threading.Thread(target=auto_start_voice_chat, daemon=True)
    auto_start_thread.start()

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


def standalone_mode():
    """独立运行模式（直接运行语音对话，不启动API服务器）"""
    assistant = VoiceAssistant()

    # 列出所有音频设备
    assistant.list_audio_devices()

    # 运行对话系统
    # 可以手动指定设备索引，例如：
    # assistant.run(input_device=1, output_device=2)
    assistant.run()


if __name__ == "__main__":
    main()
