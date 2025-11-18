# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.ed.
import subprocess
import os
import json
import ffmpeg
import requests
import gradio as gr

from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.utils.log import logger

IMAGE_TOKEN_INDEX = -200
AUDIO_TOKEN_INDEX = -500
IMAGE_TAG = -1
VIDEO_TAG = -2
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_AUDIO_TOKEN = "<audio>"
MAX_IMAGE_LENGTH = 16
PUNCTUATION = "！？。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."


def is_video(file_path):
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    _, ext = os.path.splitext(file_path)
    return ext.lower() in video_extensions


def is_image(file_path):
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
    _, ext = os.path.splitext(file_path)
    return ext.lower() in image_extensions


def is_wav(file_path):
    wav_extensions = {'.wav'}
    _, ext = os.path.splitext(file_path)
    return ext.lower() in wav_extensions


def convert_webm_to_mp4(input_file, output_file):
    try:
        (
            ffmpeg
            .input(input_file)
            .output(output_file, vcodec='libx264', acodec='aac')
            .run()
        )
    except ffmpeg.Error as e:
        raise RuntimeError("运行时报错，请开启日志进一步定位问题") from e


def _parse_text(text):
    text = text.replace("\\n", "\n")
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0

    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split("`")
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = "<br></code></pre>"
        else:
            if i > 0 and count % 2 == 1:
                line = line.replace("`", r"\`")
                line = line.replace("<", "&lt;")
                line = line.replace(">", "&gt;")
                line = line.replace(" ", "&nbsp;")
                line = line.replace("*", "&ast;")
                line = line.replace("_", "&lowbar;")
                line = line.replace("-", "&#45;")
                line = line.replace(".", "&#46;")
                line = line.replace("!", "&#33;")
                line = line.replace("(", "&#40;")
                line = line.replace(")", "&#41;")
                line = line.replace("$", "&#36;")
            lines[i] = "<br>" + line

    return "".join(lines)


def _launch_demo():
    def predict(_chatbot, task_history):
        if task_history[-1]["type"] == "text":
            chat_query = task_history[-1]["text"]
            chat_query = _parse_text(chat_query)
        else:
            chat_query = task_history[-1]["audio_url"]
            chat_query = (chat_query,)

        query = {}
        query["prompt"] = task_history
        query["model"] = "vita_mixtral"
        json_data = json.dumps(query)
        headers = {'Content-Type': 'application/json'}
        url = 'http://127.0.0.1:1080/generate'
        response = requests.post(url, data=json_data, headers=headers)

        if response.status_code == 200:
            # 处理成功响应
            indices = [i for i, char in enumerate(response.text) if char == ']']
            if len(indices) >= 2: 
                text_start_index = indices[-2]
            text_end_index = response.text.rfind('"')
            text_str = response.text[text_start_index + 1: text_end_index] 
            _chatbot[-1] = (chat_query, _parse_text(text_str))
        else:
            # 处理错误响应
            error_message = response.status_code + " " + response.text
            logger.error(error_message, ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
        task_history.clear()

        yield _chatbot

    def add_text(history, task_history, text):
        task_text = text
        history = history + [(_parse_text(text), None)]
        task_history = task_history + [{"type":"text", "text": task_text}]
        return history, task_history

    def add_file(history, task_history, file):
        history = history + [((file.name,), None)]
        if is_image(file):
            task_history = task_history + [{"type":"image_url", "image_url": file}]
        elif is_video(file):
            task_history = task_history + [{"type":"video_url", "video_url": file}]

        return history, task_history

    def add_audio(history, task_history, file):
        if file is None:
            return history, task_history
        history = history + [((file,), None)]
        task_history = task_history + [{"type":"audio_url", "audio_url": file}]
        return history, task_history

    def add_video(history, task_history, file):
        if file is None:
            return history, task_history
        new_file_name = file.replace(".webm", ".mp4")
        if file.endswith(".webm"):
            convert_webm_to_mp4(file, new_file_name)
        task_history = task_history + [{"type":"video_url", "video_url": file}]
        return history, task_history

    def reset_user_input():
        return gr.update(value="")

    def reset_state(task_history):
        task_history.clear()
        return []

    with gr.Blocks(title="VideoMLLM") as demo:
        gr.Markdown("""<center><font size=8>VITA</center>""")
        chatbot = gr.Chatbot(label='VITA', elem_classes="control-height", height=500)
        query = gr.Textbox(lines=2, label='Text Input')
        task_history = gr.State([])
        with gr.Row():
            add_text_button = gr.Button("Submit Text (提交文本)")
            add_audio_button = gr.Button("Submit Audio (提交音频)")
        with gr.Row():
            with gr.Column(scale=2):
                addfile_btn = gr.UploadButton("📁 Upload (上传文件[视频,图片])", file_types=["video", "image"])
                video_input = gr.Video(sources=["webcam"], height=400, width=700, container=True, interactive=True,
                    show_download_button=True, label="📹 Video Recording (视频录制)")
   
            with gr.Column(scale=1):
                empty_bin = gr.Button("🧹 Clear History (清除历史)")
                record_btn = gr.Audio(sources=["microphone", "upload"], type="filepath",
                    label="🎤 Record or Upload Audio (录音或上传音频)", show_download_button=True,
                    waveform_options=gr.WaveformOptions(sample_rate=16000))

        add_text_button.click(add_text, [chatbot, task_history, query], [chatbot, task_history], show_progress=True
        ).then(reset_user_input, [], [query]
        ).then(predict, [chatbot, task_history], [chatbot], show_progress=True)

        video_input.stop_recording(add_video, [chatbot, task_history, video_input], [chatbot, task_history],
            show_progress=True)
        empty_bin.click(reset_state, [task_history], [chatbot], show_progress=True)
        addfile_btn.upload(add_file, [chatbot, task_history, addfile_btn], [chatbot, task_history], show_progress=True)

        add_audio_button.click(add_audio, [chatbot, task_history, record_btn], [chatbot, task_history],
            show_progress=True).then(predict, [chatbot, task_history], [chatbot], show_progress=True)

    server_port = 7876
    demo.launch(
        share=False,
        debug=True,
        server_name="127.0.0.1",
        server_port=server_port,
        show_api=False,
        show_error=False,
        auth=('123', '123'),
        )


if __name__ == '__main__':
    cwd = os.getcwd()
    config_path = '/usr/local/Ascend/mindie/latest/mindie-service/conf/config.json'
    with open(config_path, 'r') as file:
        data = json.load(file)
    data['ServerConfig']['port'] = 1080
    data['ServerConfig']['managementPort'] = 1081
    data['ServerConfig']['httpsEnabled'] = False
    data['BackendConfig']['npuDeviceIds'] = [[0, 1, 2, 3, 4, 5, 6, 7]]
    data['BackendConfig']['ModelDeployConfig']['maxSeqLen'] = 16384
    data['BackendConfig']['ModelDeployConfig']['maxInputTokenLen'] = 8192
    data['BackendConfig']['ModelDeployConfig']['ModelConfig'][0]['modelName'] = "vita_mixtral"
    data['BackendConfig']['ModelDeployConfig']['ModelConfig'][0]['modelWeightPath'] = \
        "/data2/tingfu/Downloads/VITA_Weights/vita/"
    data['BackendConfig']['ModelDeployConfig']['ModelConfig'][0]['worldSize'] = 8
    data['BackendConfig']['ModelDeployConfig']['ModelConfig'][0]['cpuMemSize'] = 8
    data['BackendConfig']['ModelDeployConfig']['ModelConfig'][0]['npuMemSize'] = 8
    with open(config_path, 'w') as file: 
        json.dump(data, file, indent=4)

    new_directory = '/usr/local/Ascend/mindie/latest/mindie-service/bin/' 
    os.chdir(new_directory)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '1234'
    binary_file_path = './mindieservice_daemon'
    subprocess.Popen([binary_file_path])
    os.chdir(cwd)
    _launch_demo()
