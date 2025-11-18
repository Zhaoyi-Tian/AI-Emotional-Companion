#!/usr/bin/env python3
"""
YOLO系统状态检查脚本
"""

import requests
import json
import sys
from datetime import datetime

def print_status(status, text):
    """打印带颜色的状态"""
    colors = {
        "✅": "\033[92m",  # 绿色
        "❌": "\033[91m",  # 红色
        "⚠️": "\033[93m",  # 黄色
        "ℹ️": "\033[94m",  # 蓝色
    }
    reset = "\033[0m"
    symbol = text[0] if text[0] in colors else "ℹ️"
    color = colors.get(symbol, "")
    print(f"{color}{text}{reset}")

def check_service(url, name):
    """检查服务状态"""
    try:
        response = requests.get(f"{url}/health", timeout=3)
        if response.status_code == 200:
            data = response.json()
            print_status("✅", f"{name} - 运行正常")
            return True
        else:
            print_status("❌", f"{name} - HTTP错误: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print_status("❌", f"{name} - 无法连接")
        return False
    except Exception as e:
        print_status("❌", f"{name} - 错误: {e}")
        return False

def check_yolo_detection():
    """检查YOLO检测状态"""
    try:
        response = requests.get("http://localhost:5005/detect/status", timeout=3)
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                status = data.get("status", {})
                is_running = status.get("is_running", False)
                fps = status.get("fps", 0)
                camera = status.get("camera_index")

                if is_running:
                    print_status("✅", f"YOLO检测 - 运行中 (摄像头: {camera}, FPS: {fps:.1f})")
                else:
                    print_status("⚠️", f"YOLO检测 - 已停止")
                return is_running
            else:
                print_status("❌", f"YOLO检测 - {data.get('message')}")
                return False
    except Exception as e:
        print_status("❌", f"YOLO检测 - 错误: {e}")
        return False

def main():
    """主检查函数"""
    print("=" * 60)
    print("YOLO实时检测系统状态检查")
    print("=" * 60)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)

    # 检查各个服务
    services = [
        ("http://localhost:5005", "YOLO检测服务 (Port 5005)"),
        ("http://localhost:8080", "Web UI界面 (Port 8080)"),
    ]

    all_ok = True
    for url, name in services:
        if not check_service(url, name):
            all_ok = False

    print("-" * 60)

    # 检查YOLO检测状态
    detection_running = check_yolo_detection()

    print("-" * 60)

    # 访问地址
    print("\n📍 访问地址:")
    print("   • Web UI: http://localhost:8080")
    print("   • YOLO API: http://localhost:5005")
    print("   • 视频流页面: http://localhost:5005/stream")
    print("   • MJPEG流: http://localhost:5005/camera/detect/stream")

    if not detection_running:
        print("\n💡 提示: YOLO检测未运行，可以在Web UI中点击'开始检测'")

    print("\n" + "=" * 60)

    # 总结
    if all_ok:
        print_status("✅", "系统状态正常！")
    else:
        print_status("❌", "部分服务异常，请检查日志")

    # 快速操作
    print("\n🚀 快速操作:")
    print("   • 启动检测: curl -X POST http://localhost:5005/detect/start")
    print("   • 停止检测: curl -X POST http://localhost:5005/detect/stop")
    print("   • 查看日志: tail -f logs/YOLO.log")

if __name__ == "__main__":
    main()