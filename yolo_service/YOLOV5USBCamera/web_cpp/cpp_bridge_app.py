#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI Bridge for C++ YOLO Detection
桥接 C++ YOLO 检测程序，通过 Web 界面展示结果
"""

import os
import sys
import json
import asyncio
import base64
import subprocess
import threading
import queue
import time
from typing import Optional, List, Dict
import signal

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# 创建 FastAPI 应用
app = FastAPI(title="C++ YOLO Bridge", version="1.0.0")

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
is_running = False
detection_queue = queue.Queue(maxsize=100)
latest_data = None
current_model = "detection"  # "detection" or "emotion"
stats = {
    "fps": 0,
    "total_detections": 0,
    "frame_count": 0,
    "start_time": None
}


class CPPBridge:
    """C++ 程序桥接器"""

    def __init__(self):
        self.process = None
        self.output_thread = None
        self.model_paths = {
            "detection": "../out/main",
            "emotion": "../out_emotion/main"
        }

    def start_cpp_program(self, model_type="detection"):
        """启动指定模型的 C++ 检测程序"""
        global current_model

        try:
            current_model = model_type
            cpp_dir = os.path.dirname(os.path.abspath(__file__))

            # 根据模型类型选择目录
            if model_type == "emotion":
                cpp_dir = os.path.join(cpp_dir, "../out_emotion")
            else:
                cpp_dir = os.path.join(cpp_dir, "../out")

            # 先停止可能已运行的程序
            os.system("pkill -f './main stdout' 2>/dev/null")
            time.sleep(1)

            # 启动 C++ 程序
            model_name = "人脸检测" if model_type == "detection" else "情感识别"
            print(f"启动 C++ {model_name} 程序...")

            self.process = subprocess.Popen(
                ["./main", "stdout"],
                cwd=cpp_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )

            # 检查是否成功启动
            time.sleep(2)
            if self.process.poll() is not None:
                # 程序已退出，读取错误信息
                stderr = self.process.stderr.read()
                print(f"C++ {model_name}程序启动失败: {stderr}")
                return False

            # 启动输出读取线程
            self.output_thread = threading.Thread(target=self._read_output)
            self.output_thread.daemon = True
            self.output_thread.start()

            print(f"C++ {model_name}程序启动成功")
            return True
        except Exception as e:
            print(f"启动 C++ 程序失败: {e}")
            return False

    def _read_output(self):
        """读取 C++ 程序输出"""
        global latest_data, stats
        fps_counter = 0
        fps_start_time = time.time()

        while self.process and self.process.poll() is None:
            try:
                line = self.process.stdout.readline()
                if line:
                    line = line.strip()

                    # 尝试解析 JSON
                    if line.startswith("{"):
                        try:
                            data = json.loads(line)

                            # 更新最新数据
                            latest_data = data

                            # 更新统计
                            stats["frame_count"] += 1
                            stats["total_detections"] += len(data.get("detections", []))

                            # 更新 FPS
                            fps_counter += 1
                            current_time = time.time()
                            if current_time - fps_start_time >= 1.0:
                                stats["fps"] = fps_counter / (current_time - fps_start_time)
                                fps_counter = 0
                                fps_start_time = current_time

                            # 放入队列
                            if not detection_queue.full():
                                detection_queue.put(data)

                        except json.JSONDecodeError:
                            print(f"JSON 解析失败: {line}")
                    elif line and "[INFO]" in line:
                        print(f"C++: {line}")

            except Exception as e:
                print(f"读取输出错误: {e}")
                break

    def stop(self):
        """停止所有进程"""
        if self.process:
            self.process.terminate()
            time.sleep(0.5)
            if self.process.poll() is None:
                self.process.kill()
            self.process = None


# 创建桥接器实例
bridge = CPPBridge()


@app.get("/")
async def get_index():
    """返回主页"""
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 端点"""
    await websocket.accept()

    try:
        while True:
            # 获取最新的数据
            data = None
            if not detection_queue.empty():
                data = detection_queue.get()
            elif latest_data:
                data = latest_data

            # 发送数据
            if data:
                # 添加 FPS 和模型信息
                data["fps"] = stats.get("fps", 0)
                data["model"] = current_model
                await websocket.send_json(data)

            await asyncio.sleep(0.05)  # 20 FPS 最大

    except WebSocketDisconnect:
        print("WebSocket 连接断开")
    except Exception as e:
        print(f"WebSocket 错误: {e}")


@app.post("/start")
async def start_detection():
    """启动检测"""
    global is_running, stats

    if is_running:
        return {"status": "already_running"}

    # 使用当前模型启动 C++ 程序
    if not bridge.start_cpp_program(current_model):
        return {"status": "error", "message": "无法启动 C++ 程序"}

    # 重置统计
    stats = {
        "fps": 0,
        "total_detections": 0,
        "frame_count": 0,
        "start_time": time.time()
    }

    is_running = True
    return {"status": "success", "model": current_model}


class ModelRequest(BaseModel):
    model_type: str

class ConfigRequest(BaseModel):
    display_confidence_threshold: float = 0.7
    nms_threshold: float = 0.45
    max_fps: int = 15

# 全局配置
global_config = ConfigRequest()

@app.post("/switch_model")
async def switch_model(request: ModelRequest):
    """切换模型"""
    global is_running, stats, current_model

    model_type = request.model_type
    if model_type not in ["detection", "emotion"]:
        return {"status": "error", "message": "Invalid model type"}

    # 如果模型相同，无需切换
    if model_type == current_model:
        return {"status": "success", "model": current_model, "message": "Model already active"}

    # 记录之前是否在运行
    was_running = is_running

    # 停止当前运行
    if is_running:
        is_running = False
        bridge.stop()

        # 清空队列
        while not detection_queue.empty():
            detection_queue.get()

    # 更新当前模型
    current_model = model_type

    # 重置统计
    stats = {
        "fps": 0,
        "total_detections": 0,
        "frame_count": 0,
        "start_time": None
    }

    # 如果之前在运行，自动启动新模型
    if was_running:
        if bridge.start_cpp_program(current_model):
            is_running = True
            model_name = "人脸检测" if current_model == "detection" else "情感识别"
            return {"status": "success", "model": current_model, "message": f"已切换到{model_name}模式", "auto_started": True}
        else:
            return {"status": "error", "message": "无法启动新的模型"}

    return {"status": "success", "model": current_model, "message": "模型已切换"}


@app.get("/config")
async def get_config():
    """获取配置"""
    return {
        "display_confidence_threshold": global_config.display_confidence_threshold,
        "nms_threshold": global_config.nms_threshold,
        "max_fps": global_config.max_fps
    }


@app.post("/config")
async def update_config(config: ConfigRequest):
    """更新配置"""
    global global_config
    global_config = config
    return {"status": "success", "message": "配置已更新"}


@app.post("/stop")
async def stop_detection():
    """停止检测"""
    global is_running

    if not is_running:
        return {"status": "not_running"}

    is_running = False
    bridge.stop()

    # 清空队列
    while not detection_queue.empty():
        detection_queue.get()

    return {"status": "success"}


@app.get("/stats")
async def get_stats():
    """获取统计信息"""
    runtime = 0
    if stats["start_time"]:
        runtime = time.time() - stats["start_time"]

    return {
        "fps": stats.get("fps", 0),
        "total_detections": stats.get("total_detections", 0),
        "frame_count": stats.get("frame_count", 0),
        "runtime_seconds": round(runtime, 1),
        "is_running": is_running
    }


# 处理生命周期事件
@app.on_event("startup")
async def startup_event():
    """应用启动时"""
    pass


@app.on_event("shutdown")
async def shutdown_event():
    """关闭时清理"""
    global is_running
    is_running = False
    bridge.stop()


# 创建静态文件目录
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")