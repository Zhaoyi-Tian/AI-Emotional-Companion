#!/bin/bash

# C++ YOLO + Python Web 启动脚本

echo "========================================="
echo "  C++ YOLO Detection & Emotion Web System"
echo "========================================="

# 设置工作目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 检查 C++ 程序是否存在
CPP_EXEC_DETECTION="../out/main"
CPP_EXEC_EMOTION="../out_emotion/main"
CPP_DETECTION_EXISTS=true
CPP_EMOTION_EXISTS=true

if [ ! -f "$CPP_EXEC_DETECTION" ]; then
    echo "警告: 人脸检测程序不存在: $CPP_EXEC_DETECTION"
    CPP_DETECTION_EXISTS=false
fi

if [ ! -f "$CPP_EXEC_EMOTION" ]; then
    echo "警告: 情感识别程序不存在: $CPP_EXEC_EMOTION"
    CPP_EMOTION_EXISTS=false
fi

if [ "$CPP_DETECTION_EXISTS" = false ] && [ "$CPP_EMOTION_EXISTS" = false ]; then
    echo "错误: 两个 C++ 程序都不存在"
    echo "请先运行: bash ../scripts/sample_build.sh"
    echo "和: bash ../scripts/sample_build_emotion.sh"
    exit 1
fi

# 检查模型文件
MODEL_DETECTION="../model/yolov5s.om"
MODEL_EMOTION="../model/yolov5s_emotion.om"

if [ ! -f "$MODEL_DETECTION" ]; then
    echo "警告: 人脸检测模型不存在: $MODEL_DETECTION"
fi

if [ ! -f "$MODEL_EMOTION" ]; then
    echo "警告: 情感识别模型不存在: $MODEL_EMOTION"
fi

# 检查 Python 环境
echo "检查 Python 环境..."
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 Python3"
    exit 1
fi

# 安装依赖（如果需要）
echo "检查依赖..."
python3 -c "import fastapi" 2>/dev/null || {
    echo "安装 FastAPI 依赖..."
    pip3 install fastapi uvicorn websockets opencv-python numpy
}

# 启动 Web 服务
echo ""
echo "启动 C++ YOLO Bridge Web 服务..."
echo "----------------------------------------"
echo "Web 界面地址: http://localhost:8001"
echo "API 文档地址: http://localhost:8001/docs"
echo "----------------------------------------"
echo ""
echo "使用说明："
echo "1. 打开浏览器访问 http://localhost:8001"
echo "2. 选择检测模式：物体识别 或 人脸情感识别"
echo "3. 点击 '开始检测' 按钮"
echo "4. 查看实时检测结果"
echo ""
echo "按 Ctrl+C 停止服务"
echo ""

# 进入 src 目录并启动服务
python3 cpp_bridge_app.py