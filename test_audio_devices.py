#!/usr/bin/env python3
"""
测试音频设备的脚本
帮助找到正确的音频设备配置
"""

import pyaudio
import numpy as np
import time

def test_audio_device(device_index=None):
    """测试音频设备"""
    pa = pyaudio.PyAudio()

    print(f"\n=== 测试设备 {device_index if device_index is not None else '默认'} ===")

    try:
        # 尝试打开音频流
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=1024
        )

        print("✅ 音频流打开成功！")

        # 获取设备信息
        if device_index is not None:
            device_info = pa.get_device_info_by_index(device_index)
            print(f"设备名称: {device_info['name']}")
            print(f"最大输入通道: {device_info['maxInputChannels']}")
            print(f"最大输出通道: {device_info['maxOutputChannels']}")

        # 测试录音5秒
        print("\n开始5秒录音测试...")
        frames = []
        for i in range(0, int(16000 / 1024 * 5)):  # 5秒
            data = stream.read(1024, exception_on_overflow=False)
            frames.append(data)

            # 计算音量
            rms = np.sqrt(np.mean(np.square(np.frombuffer(data, dtype=np.int16))))
            if i % 50 == 0:  # 每秒显示一次
                print(f"录音中... RMS: {int(rms)}")

        print("✅ 录音测试完成！")

        # 关闭流
        stream.stop_stream()
        stream.close()

    except Exception as e:
        print(f"❌ 错误: {e}")
        return False

    return True

def list_audio_devices():
    """列出所有音频设备"""
    pa = pyaudio.PyAudio()

    print("\n=== 可用音频设备列表 ===")
    print("索引 | 设备名称 | 输入通道 | 输出通道")
    print("-" * 50)

    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        print(f"{i:4d} | {info['name'][:30]:30s} | {info['maxInputChannels']:10d} | {info['maxOutputChannels']:10d}")

if __name__ == "__main__":
    print("🎤 音频设备测试工具")
    print("1. 列出所有音频设备")
    list_audio_devices()

    print("\n\n2. 测试每个可用的输入设备")
    pa = pyaudio.PyAudio()

    # 测试有输入通道的设备
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"\n测试设备 {i}: {info['name']}")
            test_audio_device(i)
            time.sleep(1)

    print("\n✅ 测试完成！")
    print("\n建议：")
    print("1. 选择能成功打开音频流的设备")
    print("2. 在config.yaml中设置 voice_chat.input_device 为对应的索引")
    print("3. 如果仍有问题，尝试设置 output_device 以解决音频输出问题")