#!/usr/bin/env python3
"""
VAD问题诊断工具
实时显示麦克风RMS值，帮助找到问题
"""

import sys
import os
from pathlib import Path

# 抑制ALSA警告
os.environ['PYAUDIO_ALSA_ERRORS'] = '0'

import pyaudio
import numpy as np
import time

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent))
from config_loader import get_config

def diagnose_vad():
    """诊断VAD问题"""

    print("=" * 60)
    print("VAD问题诊断工具")
    print("=" * 60)

    # 读取配置
    voice_config = get_config('voice_chat')
    input_device = voice_config.get('input_device', 1)
    threshold = voice_config.get('silence_threshold', 500)

    print(f"\n当前配置:")
    print(f"  输入设备: {input_device}")
    print(f"  静音阈值: {threshold}")
    print(f"  静音持续时间: {voice_config.get('silence_duration', 1.0)}秒")

    # 初始化PyAudio
    audio = pyaudio.PyAudio()

    # 获取设备信息
    if input_device is not None:
        device_info = audio.get_device_info_by_index(input_device)
        print(f"\n设备信息:")
        print(f"  名称: {device_info['name']}")
        print(f"  默认采样率: {device_info['defaultSampleRate']} Hz")
        print(f"  输入通道数: {device_info['maxInputChannels']}")

    # 尝试打开音频流
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1

    # 尝试不同的采样率
    supported_rates = [16000, 44100, 48000, 22050]
    RATE = None

    for rate in supported_rates:
        try:
            stream = audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=rate,
                input=True,
                input_device_index=input_device,
                frames_per_buffer=CHUNK
            )
            stream.close()
            RATE = rate
            break
        except:
            continue

    if RATE is None:
        print("\n❌ 无法打开音频设备！")
        audio.terminate()
        return

    print(f"\n✅ 使用采样率: {RATE} Hz")

    # 打开流
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=input_device,
        frames_per_buffer=CHUNK
    )

    print("\n" + "=" * 60)
    print("开始实时监测麦克风音量（Ctrl+C停止）")
    print("=" * 60)
    print(f"\n当前阈值: {threshold}")
    print("请保持安静几秒，然后说话，观察RMS变化...\n")

    rms_values = []
    try:
        while True:
            # 读取音频
            data = stream.read(CHUNK, exception_on_overflow=False)

            # 计算RMS
            audio_array = np.frombuffer(data, dtype=np.int16)
            rms = int(np.sqrt(np.mean(audio_array.astype(np.float64) ** 2)))

            rms_values.append(rms)
            if len(rms_values) > 100:
                rms_values.pop(0)

            avg_rms = int(np.mean(rms_values))
            min_rms = int(np.min(rms_values))
            max_rms = int(np.max(rms_values))

            # 判断状态
            if rms > threshold:
                status = "🔴 有声音 (会录音)"
            else:
                status = "🟢 静音 (会计数)"

            # 实时显示
            print(f"\r当前RMS: {rms:5d}  |  平均: {avg_rms:5d}  |  范围: {min_rms:5d}-{max_rms:5d}  |  阈值: {threshold}  |  {status}", end='', flush=True)

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("监测结束")
        print("=" * 60)

        if rms_values:
            print(f"\n统计数据:")
            print(f"  平均RMS: {int(np.mean(rms_values))}")
            print(f"  最小RMS: {int(np.min(rms_values))}")
            print(f"  最大RMS: {int(np.max(rms_values))}")

            # 推荐阈值
            sorted_rms = sorted(rms_values)
            percentile_80 = sorted_rms[int(len(sorted_rms) * 0.8)]
            recommended = int(percentile_80 * 1.3)

            print(f"\n推荐阈值: {recommended}")
            print(f"  (基于80百分位数 {percentile_80} × 1.3)")

            print(f"\n诊断结果:")

            avg = np.mean(rms_values)
            if threshold < avg:
                print(f"  ❌ 问题：阈值({threshold})低于平均RMS({int(avg)})")
                print(f"     → 系统会认为环境噪音是\"有声音\"，一直录音")
                print(f"     → 解决：提高阈值到{recommended}以上")
            elif threshold > max_rms:
                print(f"  ❌ 问题：阈值({threshold})高于最大RMS({max_rms})")
                print(f"     → 系统永远检测不到\"有声音\"，无法开始录音")
                print(f"     → 解决：降低阈值到{recommended}左右")
            else:
                print(f"  ✅ 阈值设置合理")
                print(f"     → 静音RMS < {threshold} < 说话RMS")

    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

if __name__ == "__main__":
    diagnose_vad()
