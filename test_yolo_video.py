#!/usr/bin/env python3
"""
测试YOLO视频流获取
"""

import requests
import base64
from PIL import Image
import io
import cv2
import numpy as np
import time

def test_yolo_video():
    print("测试YOLO视频流...")
    print("=" * 50)

    # API地址
    url = "http://localhost:5005/detect/latest"

    # 连续获取10帧
    for i in range(10):
        try:
            print(f"\n获取第 {i+1} 帧...")
            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                data = response.json()

                if data.get('success'):
                    # 获取图像数据
                    frame_base64 = data.get('frame_base64', '')
                    detections = data.get('detections', {})

                    if frame_base64:
                        # 解码并显示图像信息
                        image_data = base64.b64decode(frame_base64)
                        image = Image.open(io.BytesIO(image_data))

                        print(f"  ✓ 图像尺寸: {image.size}")
                        print(f"  ✓ 数据大小: {len(frame_base64)} 字节")

                        # 检测信息
                        fps = detections.get('fps', 0)
                        detection_list = detections.get('detections', [])
                        print(f"  ✓ FPS: {fps:.1f}")
                        print(f"  ✓ 检测到 {len(detection_list)} 个对象")

                        # 显示检测对象
                        if detection_list:
                            print("  检测到的对象:")
                            for j, det in enumerate(detection_list[:3]):  # 只显示前3个
                                label = det.get('label', 'Unknown')
                                conf = det.get('confidence', 0)
                                bbox = det.get('bbox', [])
                                print(f"    {j+1}. {label}: {conf:.2%} 位置:{bbox}")

                        # 保存一帧用于验证
                        if i == 0:
                            image.save("test_yolo_frame.jpg")
                            print("  ✓ 已保存测试帧到 test_yolo_frame.jpg")
                    else:
                        print("  ✗ 无图像数据")
                else:
                    print(f"  ✗ 请求失败: {data.get('message')}")
            else:
                print(f"  ✗ HTTP错误: {response.status_code}")

        except Exception as e:
            print(f"  ✗ 错误: {e}")

        time.sleep(0.5)  # 等待500ms再获取下一帧

    print("\n" + "=" * 50)
    print("测试完成！")

if __name__ == "__main__":
    test_yolo_video()