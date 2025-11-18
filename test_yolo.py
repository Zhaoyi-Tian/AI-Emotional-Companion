#!/usr/bin/env python3
"""
YOLO检测测试脚本
"""

import cv2
import requests
import time
import json
import base64
import numpy as np
from PIL import Image
import io

# YOLO服务配置
YOLO_PORT = 5005
YOLO_URL = f"http://localhost:{YOLO_PORT}"

def test_yolo_service():
    """测试YOLO服务的完整功能"""
    print("=" * 50)
    print("YOLO检测服务测试")
    print("=" * 50)

    # 1. 测试健康检查
    print("\n1. 测试健康检查...")
    try:
        response = requests.get(f"{YOLO_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 服务状态: {data['status']}")
            print(f"   - 版本: {data['version']}")
            print(f"   - 检测器初始化: {'是' if data.get('detector') else '否'}")
        else:
            print(f"❌ 健康检查失败: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 无法连接到YOLO服务: {e}")
        return False

    # 2. 测试启动检测
    print("\n2. 启动检测...")
    try:
        response = requests.post(
            f"{YOLO_URL}/detect/start",
            json={"confidence_threshold": 0.5},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"✅ 检测已启动: {data['message']}")
                print(f"   - 摄像头索引: {data.get('camera_index')}")
            else:
                print(f"❌ 启动失败: {data.get('message')}")
                return False
        else:
            print(f"❌ 请求失败: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 启动检测出错: {e}")
        return False

    # 3. 等待几秒获取检测结果
    print("\n3. 获取检测结果（等待5秒）...")
    time.sleep(5)

    try:
        response = requests.get(f"{YOLO_URL}/detect/latest", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                detections = data.get('detections', {})
                fps = detections.get('fps', 0)
                detection_count = len(detections.get('detections', []))

                print(f"✅ 检测状态:")
                print(f"   - FPS: {fps:.1f}")
                print(f"   - 检测到的对象数: {detection_count}")

                # 显示检测到的对象
                if detection_count > 0:
                    print("\n   检测到的对象:")
                    for i, det in enumerate(detections.get('detections', []), 1):
                        label = det.get('label', 'Unknown')
                        confidence = det.get('confidence', 0)
                        bbox = det.get('bbox', [])
                        print(f"   {i}. {label}: {confidence:.2%} 位置:{bbox}")
                else:
                    print("   - 未检测到对象（可能摄像头没有连接或环境中没有对象）")
            else:
                print(f"❌ 获取失败: {data.get('message')}")
        else:
            print(f"❌ 请求失败: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ 获取检测结果出错: {e}")

    # 4. 测试更新设置
    print("\n4. 测试更新设置...")
    try:
        response = requests.post(
            f"{YOLO_URL}/detect/update_settings",
            json={
                "confidence_threshold": 0.7,
                "nms_threshold": 0.5
            },
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"✅ 设置已更新")
                settings = data.get('settings', {})
                print(f"   - 置信度阈值: {settings.get('confidence_threshold', 0.5)}")
                print(f"   - NMS阈值: {settings.get('nms_threshold', 0.4)}")
            else:
                print(f"❌ 更新失败: {data.get('message')}")
        else:
            print(f"❌ 请求失败: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ 更新设置出错: {e}")

    # 5. 测试停止检测
    print("\n5. 停止检测...")
    try:
        response = requests.post(f"{YOLO_URL}/detect/stop", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"✅ 检测已停止: {data['message']}")
            else:
                print(f"❌ 停止失败: {data['message']}")
        else:
            print(f"❌ 请求失败: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ 停止检测出错: {e}")

    # 6. 测试视频流端点
    print("\n6. 测试视频流端点...")
    print("   - MJPEG流地址: http://localhost:5005/camera/detect/stream")
    print("   - WebSocket地址: ws://localhost:5005/ws/detect/stream")
    print("   - HTML页面地址: http://localhost:5005/stream")

    print("\n" + "=" * 50)
    print("测试完成！")
    print("=" * 50)

    # 提供使用说明
    print("\n使用说明:")
    print("1. 访问 Web UI: http://localhost:8080")
    print("   - 点击 '📹 YOLO检测' 标签页")
    print("   - 点击 '开始检测' 启动实时检测")
    print("\n2. 直接访问视频流页面: http://localhost:5005/stream")
    print("   - 使用WebSocket实时传输检测结果")
    print("   - 支持调整检测参数")
    print("\n3. 使用MJPEG流: http://localhost:5005/camera/detect/stream")
    print("   - 可以在VLC等播放器中打开")
    print("   - 或嵌入到其他Web应用中")

    return True

if __name__ == "__main__":
    test_yolo_service()