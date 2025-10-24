"""
统一配置加载器
支持从YAML文件加载配置,并提供动态更新功能
"""

import yaml
import os
from pathlib import Path
from typing import Any, Dict
import threading


class ConfigLoader:
    """配置加载器单例类"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.config_path = Path(__file__).parent / "config.yaml"
            self.config: Dict[str, Any] = {}
            self.load_config()
            self.initialized = True

    def load_config(self) -> Dict[str, Any]:
        """从YAML文件加载配置"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            return self.config
        except FileNotFoundError:
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"配置文件格式错误: {e}")

    def save_config(self) -> None:
        """保存配置到YAML文件"""
        with self._lock:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, allow_unicode=True, default_flow_style=False)

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置值,支持点分隔的路径
        例如: get("llm.api.api_key")
        """
        keys = key_path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def set(self, key_path: str, value: Any, save: bool = True) -> None:
        """
        设置配置值,支持点分隔的路径
        例如: set("llm.api.api_key", "new_key")
        """
        keys = key_path.split('.')
        config = self.config

        # 导航到最后一个键的父级
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        # 设置值
        config[keys[-1]] = value

        # 可选保存到文件
        if save:
            self.save_config()

    def reload(self) -> Dict[str, Any]:
        """重新加载配置文件"""
        return self.load_config()

    # 便捷访问方法
    def get_asr_config(self) -> Dict[str, Any]:
        """获取ASR配置"""
        return self.get('asr', {})

    def get_llm_config(self) -> Dict[str, Any]:
        """获取LLM配置"""
        return self.get('llm', {})

    def get_tts_config(self) -> Dict[str, Any]:
        """获取TTS配置"""
        return self.get('tts', {})

    def get_service_ports(self) -> Dict[str, int]:
        """获取所有服务端口"""
        return self.get('services', {})

    def get_streaming_config(self) -> Dict[str, Any]:
        """获取流式处理配置"""
        return self.get('streaming', {})


# 全局配置实例
config = ConfigLoader()


# 便捷函数
def get_config(key_path: str, default: Any = None) -> Any:
    """获取配置值"""
    return config.get(key_path, default)


def set_config(key_path: str, value: Any, save: bool = True) -> None:
    """设置配置值"""
    config.set(key_path, value, save)


def reload_config() -> Dict[str, Any]:
    """重新加载配置"""
    return config.reload()


if __name__ == "__main__":
    # 测试配置加载
    print("=== 配置加载测试 ===")
    print(f"LLM API Key: {get_config('llm.api.api_key')}")
    print(f"TTS Voice: {get_config('tts.api.voice')}")
    print(f"ASR Model Type: {get_config('asr.model_type')}")
    print(f"Service Ports: {config.get_service_ports()}")
    print(f"Sentence Delimiters: {get_config('streaming.sentence_delimiters')}")
