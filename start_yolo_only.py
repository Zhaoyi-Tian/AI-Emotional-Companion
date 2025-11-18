#!/usr/bin/env python3
"""
仅启动YOLO服务的脚本
"""

import subprocess
import sys
import time
import os

def check_port(port):
    """检查端口是否可用"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return True
        except:
            return False

def main():
    """启动YOLO服务"""
    print("=" * 50)
    print("YOLO检测服务启动器")
    print("=" * 50)

    # 检查端口
    port = 5005
    if not check_port(port):
        print(f"❌ 端口 {port} 已被占用")
        print("尝试清理...")
        os.system(f"lsof -ti:{port} | xargs -r kill -9")
        time.sleep(2)

    if not check_port(port):
        print(f"❌ 端口 {port} 仍然被占用，请手动清理")
        return

    print(f"✅ 端口 {port} 可用")

    # 启动服务
    print("\n启动YOLO检测服务...")
    cmd = [sys.executable, "yolo_service/app_fastapi.py", "--host", "0.0.0.0", "--port", str(port)]

    try:
        # 使用subprocess运行，保持实时输出
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

        print(f"服务已启动，PID: {process.pid}")
        print(f"访问地址: http://localhost:{port}")
        print(f"健康检查: http://localhost:{port}/health")
        print(f"视频流: http://localhost:{port}/stream")
        print("\n按 Ctrl+C 停止服务\n")

        # 实时输出日志
        for line in process.stdout:
            if "error" in line.lower() or "error" in line.lower():
                print(f"❌ {line.strip()}")
            elif "warning" in line.lower() or "warn" in line.lower():
                print(f"⚠️ {line.strip()}")
            else:
                print(f"   {line.strip()}")

    except KeyboardInterrupt:
        print("\n\n正在停止服务...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("强制终止...")
            process.kill()
        print("✅ 服务已停止")

    except Exception as e:
        print(f"\n❌ 启动失败: {e}")

if __name__ == "__main__":
    main()