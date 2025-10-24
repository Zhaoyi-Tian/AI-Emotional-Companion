import sounddevice as sd
import numpy as np
import time
import os
from wenet.model_CN import WeNetASRCN
from wenet.model_EN import WeNetASREN
import soundfile as sf

base_dir = os.path.dirname(__file__)
model_type="EN"
if model_type=="EN":
    model_path = os.path.join(base_dir, "EN_model/offline_encoder.om")
    vocab_path = os.path.join(base_dir, "EN_model/vocab.txt")
elif model_type=="CN":
    model_path = os.path.join(base_dir, "CN_model/offline_encoder.om")
    vocab_path = os.path.join(base_dir, "CN_model/vocab.txt")


SAMPLE_RATE = 16000
CHANNELS = 1

asr = WeNetASREN(model_path, vocab_path)

print("开始实时语音识别，按 Ctrl+C 退出。")
try:
    while True:
        print("请说话（10秒后自动识别）：")
        audio_block = sd.rec(int(10 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
        sd.wait()
        temp_wav = "temp_realtime.wav"
        sf.write(temp_wav, audio_block, SAMPLE_RATE)
        text = asr.transcribe(temp_wav)
        print("识别结果：", text)
except KeyboardInterrupt:
    print("识别结束")