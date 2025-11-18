#!/usr/bin/env python3
"""
清理端口占用脚本
清理可能被占用的端口
"""

import subprocess
import sys

def kill_processes_on_port(port):
    """杀死占用指定端口的进程"""
    try:
        # 查找占用端口的进程
        result = subprocess.run(
            ['lsof', '-ti', f':{port}'],
            capture_output=True,
            text=True
        )

        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                try:
                    subprocess.run(['kill', '-9', pid])
                    print(f"✅ 已杀死进程 {pid} (端口 {port})")
                except Exception as e:
                    print(f"❌ 无法杀死进程 {pid}: {e}")
        else:
            print(f"✅ 端口 {port} 未被占用")

    except Exception as e:
        print(f"❌ 清理端口 {port} 失败: {e}")

def main():
    """主函数"""
    print("清理端口占用...")
    print("-" * 50)

    # 要清理的端口列表
    ports = [5005, 8080]

    for port in ports:
        print(f"\n清理端口 {port}...")
        kill_processes_on_port(port)

    print("\n" + "-" * 50)
    print("清理完成！")

if __name__ == "__main__":
    main()