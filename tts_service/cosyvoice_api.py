import pyaudio
import dashscope
from dashscope.audio.tts_v2 import *
from http import HTTPStatus
from dashscope import Generation

import sys

# 若没有将API Key配置到环境变量中，需将下面这行代码注释放开，并将apiKey替换为自己的API Key
dashscope.api_key = "sk-3daa6c3cbffa4aeebce31a49f75a2c7b"
model = "cosyvoice-v2"
voice = "longxiaochun_v2"

class Callback(ResultCallback):
    _player = None
    _stream = None

    def on_open(self):
        print("WebSocket is open.")
        self._player = pyaudio.PyAudio()
        self._stream = self._player.open(
            format=pyaudio.paInt16, channels=1, rate=22050, output=True
        )

    def on_complete(self):
        print("Speech synthesis task complete successfully.")

    def on_error(self, message: str):
        print(f"Speech synthesis task failed: {message}")

    def on_close(self):
        print("WebSocket is closed.")
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._player:
            self._player.terminate()

    def on_event(self, message):
        pass  # Can be used for debug: print(f"Received synth event: {message}")

    def on_data(self, data: bytes) -> None:
        self._stream.write(data)

def synthesizer_with_cli():
    callback = Callback()
    synthesizer = SpeechSynthesizer(
        model=model,
        voice=voice,
        format=AudioFormat.PCM_22050HZ_MONO_16BIT,
        callback=callback,
    )

    print("请输入你想转换为语音的文本，按 Ctrl+D (或 Ctrl+Z 并回车, Windows) 结束输入：")
    try:
        # 读取标准输入直到 EOF
        input_text = sys.stdin.read().strip()
    except KeyboardInterrupt:
        print("\n用户终止输入。")
        return

    if not input_text:
        print("未检测到输入文本。")
        return

    synthesizer.streaming_call(input_text)
    synthesizer.streaming_complete()
    print('requestId: ', synthesizer.get_last_request_id())

if __name__ == "__main__":
    synthesizer_with_cli()