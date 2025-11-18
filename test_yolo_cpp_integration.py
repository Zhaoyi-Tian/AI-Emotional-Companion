#!/usr/bin/env python3
"""
测试C++ YOLO集成系统
验证C++程序、共享内存和Web服务的完整工作流程
"""

import time
import requests
import json
import logging
import sys
import subprocess
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 服务配置
CPP_SERVICE_PORT = 5007
CPP_SERVICE_URL = f"http://localhost:{CPP_SERVICE_PORT}"

def test_cpp_service():
    """测试C++ YOLO服务"""
    logger.info("=== 测试C++ YOLO服务 ===\n")

    # 1. 健康检查
    logger.info("1. 健康检查...")
    try:
        response = requests.get(f"{CPP_SERVICE_URL}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            logger.info("✅ 服务健康")
            logger.info(f"   - 检测器状态: {health_data.get('detector', {})}")
            logger.info(f"   - C++程序运行: {health_data.get('cpp_running')}")
            logger.info(f"   - C++进程PID: {health_data.get('cpp_pid')}")
        else:
            logger.error(f"❌ 健康检查失败: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ 无法连接到服务: {e}")
        logger.info("请确保C++ YOLO服务已启动: python yolo_cpp_service.py")
        return False

    # 2. 测试获取检测结果
    logger.info("\n2. 获取检测结果...")
    try:
        response = requests.get(f"{CPP_SERVICE_URL}/detections", timeout=5)
        if response.status_code == 200:
            data = response.json()
            fps = data.get('fps', 0)
            detection_count = len(data.get('detections', []))
            has_frame = data.get('has_frame', False)

            logger.info(f"✅ 检测结果获取成功")
            logger.info(f"   - FPS: {fps:.1f}")
            logger.info(f"   - 检测数量: {detection_count}")
            logger.info(f"   - 有帧数据: {has_frame}")

            if detection_count > 0:
                logger.info("\n   检测对象:")
                for det in data.get('detections', [])[:5]:  # 只显示前5个
                    logger.info(f"   - {det['label']}: {det['confidence']:.2f} "
                              f"位置({det['x']:.0f}, {det['y']:.0f})")
        else:
            logger.error(f"❌ 获取检测结果失败: {response.status_code}")
    except Exception as e:
        logger.error(f"❌ 请求失败: {e}")

    # 3. 测试获取帧数据
    logger.info("\n3. 获取帧数据...")
    try:
        response = requests.get(f"{CPP_SERVICE_URL}/frame", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if 'image' in data:
                logger.info("✅ 帧数据获取成功")
                logger.info(f"   - 图像大小: {len(data['image'])} 字符")
                logger.info(f"   - FPS: {data.get('fps', 0):.1f}")
                logger.info(f"   - 检测数量: {len(data.get('detections', []))}")
            else:
                logger.warning("⚠️ 无图像数据（可能需要等待摄像头初始化）")
        else:
            logger.error(f"❌ 获取帧失败: {response.status_code}")
    except Exception as e:
        logger.error(f"❌ 请求失败: {e}")

    # 4. 测试视频流
    logger.info("\n4. 测试视频流...")
    try:
        response = requests.get(f"{CPP_SERVICE_URL}/video_feed", timeout=5, stream=True)
        if response.status_code == 200:
            logger.info("✅ 视频流响应正常")
            # 读取几帧数据
            count = 0
            for line in response.iter_lines():
                if line and b'Content-Type' in line:
                    count += 1
                    if count >= 3:  # 只测试3帧
                        break
            logger.info(f"   - 成功读取视频流帧")
        else:
            logger.error(f"❌ 视频流失败: {response.status_code}")
    except Exception as e:
        logger.error(f"❌ 视频流测试失败: {e}")

    return True

def test_web_ui():
    """测试Web UI集成"""
    logger.info("\n\n=== 测试Web UI集成 ===\n")

    # 检查Web UI是否运行
    try:
        response = requests.get("http://localhost:8080", timeout=5)
        if response.status_code == 200:
            logger.info("✅ Web UI运行正常")
        else:
            logger.error("❌ Web UI未运行，请执行: python web_ui.py")
            return False
    except:
        logger.error("❌ 无法连接Web UI，请执行: python web_ui.py")
        return False

    logger.info("\n请在浏览器中执行以下操作：")
    logger.info("1. 访问 http://localhost:8080")
    logger.info("2. 点击 '📹 YOLO检测' 标签页")
    logger.info("3. 点击 '🚀 启动C++检测' 按钮")
    logger.info("4. 等待2秒观察视频流")
    logger.info("5. 应能看到带检测框的实时视频")

    return True

def check_system_requirements():
    """检查系统要求"""
    logger.info("=== 检查系统要求 ===\n")

    # 检查摄像头
    camera_devices = list(Path("/dev").glob("video*"))
    if camera_devices:
        logger.info(f"✅ 找到摄像头设备: {[str(d) for d in camera_devices[:3]]}")
    else:
        logger.warning("⚠️ 未找到摄像头设备 (/dev/video*)")

    # 检查共享内存
    shm_file = Path("/dev/shm/_yolo_detection")
    if shm_file.exists():
        logger.info(f"✅ 共享内存文件存在: {shm_file}")
    else:
        logger.info("ℹ️ 共享内存文件不存在（C++程序未运行）")

    # 检查C++可执行文件
    exe_path = Path("yolo_service/YOLOV5USBCamera/out/main")
    if exe_path.exists():
        logger.info(f"✅ C++可执行文件存在: {exe_path}")
    else:
        logger.error(f"❌ C++可执行文件不存在: {exe_path}")
        logger.info("请先编译: cd yolo_service/YOLOV5USBCamera && ./build_with_shared_memory.sh")
        return False

    # 检查模型文件
    model_path = Path("yolo_service/models/yolov5s.om")
    if model_path.exists():
        logger.info(f"✅ YOLO模型文件存在: {model_path}")
    else:
        logger.error(f"❌ YOLO模型文件不存在: {model_path}")
        return False

    return True

def main():
    """主测试函数"""
    logger.info("C++ YOLO集成系统测试\n")
    logger.info("="*50)

    # 1. 检查系统要求
    if not check_system_requirements():
        logger.error("\n❌ 系统要求检查失败，请先解决问题")
        sys.exit(1)

    # 2. 测试C++服务
    logger.info("\n" + "="*50)
    if not test_cpp_service():
        logger.error("\n❌ C++服务测试失败")
        sys.exit(1)

    # 3. 测试Web UI
    logger.info("\n" + "="*50)
    test_web_ui()

    # 总结
    logger.info("\n" + "="*50)
    logger.info("\n✅ 测试完成！")
    logger.info("\n系统工作流程：")
    logger.info("1. C++程序执行YOLO检测")
    logger.info("2. 结果写入共享内存")
    logger.info("3. Python服务读取共享内存")
    logger.info("4. Web界面显示实时视频流")
    logger.info("\n如果一切正常，您可以在Web界面看到带检测框的实时视频！")

if __name__ == "__main__":
    main()