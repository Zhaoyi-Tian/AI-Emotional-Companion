"""
统一启动脚本
自动启动所有微服务(ASR, LLM, TTS, Orchestrator, Web UI)
"""

import subprocess
import time
import sys
import os
import signal
import logging
import socket
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Launcher")

# 服务进程列表
processes = []


def get_local_ip():
    """获取本机内网IP地址"""
    try:
        # 创建一个UDP socket连接到外网(不会真正发送数据)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"


def get_conda_python(env_name):
    """获取指定conda环境的Python路径"""
    home = os.path.expanduser("~")

    # 优先查找.conda目录(conda的新默认位置)
    possible_paths = [
        f"{home}/.conda/envs/{env_name}/bin/python",
        f"{home}/miniconda3/envs/{env_name}/bin/python",
        f"{home}/anaconda3/envs/{env_name}/bin/python",
        f"/opt/miniconda3/envs/{env_name}/bin/python",
        f"/opt/anaconda3/envs/{env_name}/bin/python",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            logger.info(f"  找到conda环境 {env_name}: {path}")
            return path

    # 如果找不到直接路径,尝试使用conda run
    logger.warning(f"  未找到conda环境 {env_name} 的直接路径,将使用 conda run")
    return None  # 返回None表示需要使用conda run


def start_service(name, script_path, conda_env=None, wait_time=3):
    """启动单个服务"""
    logger.info(f"🚀 正在启动 {name}...")

    # 创建日志文件
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)

    service_name_clean = name.replace('服务', '').replace(' ', '_')
    log_file = log_dir / f"{service_name_clean}.log"

    try:
        if conda_env:
            # 使用conda环境
            python_path = get_conda_python(conda_env)

            if python_path is None:
                # 使用conda run命令
                cmd = ["conda", "run", "-n", conda_env, "python", str(script_path)]
                logger.info(f"  使用命令: conda run -n {conda_env} python {script_path}")
                with open(log_file, 'w') as log_f:
                    process = subprocess.Popen(
                        cmd,
                        stdout=log_f,
                        stderr=subprocess.STDOUT
                    )
            else:
                # 直接使用Python路径
                with open(log_file, 'w') as log_f:
                    process = subprocess.Popen(
                        [python_path, str(script_path)],
                        stdout=log_f,
                        stderr=subprocess.STDOUT
                    )
        else:
            # 使用当前Python环境
            logger.info(f"  使用当前Python环境: {sys.executable}")
            with open(log_file, 'w') as log_f:
                process = subprocess.Popen(
                    [sys.executable, str(script_path)],
                    stdout=log_f,
                    stderr=subprocess.STDOUT
                )

        processes.append({
            'name': name,
            'process': process,
            'pid': process.pid,
            'log_file': str(log_file)
        })

        logger.info(f"✅ {name} 已启动 (PID: {process.pid})")
        logger.info(f"  日志文件: {log_file}")
        time.sleep(wait_time)
        return True

    except Exception as e:
        logger.error(f"❌ {name} 启动失败: {e}")
        return False


def stop_all_services():
    """停止所有服务"""
    logger.info("\n🛑 正在停止所有服务...")

    for service in reversed(processes):
        try:
            logger.info(f"停止 {service['name']} (PID: {service['pid']})...")
            service['process'].terminate()

            # 等待进程结束
            try:
                service['process'].wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning(f"强制终止 {service['name']}")
                service['process'].kill()

            logger.info(f"✅ {service['name']} 已停止")

        except Exception as e:
            logger.error(f"停止 {service['name']} 失败: {e}")

    logger.info("所有服务已停止")


def signal_handler(sig, frame):
    """处理Ctrl+C信号"""
    logger.info("\n收到停止信号...")
    stop_all_services()
    sys.exit(0)


def main():
    """主函数"""
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("="*60)
    logger.info("🤖 AI语音助手 - 统一启动器")
    logger.info("="*60)

    base_dir = Path(__file__).parent

    # 定义服务启动顺序和配置
    services = [
        {
            'name': 'ASR服务',
            'script': base_dir / 'asr_service' / 'app_fastapi.py',
            'conda_env': 'asr',  # ASR的conda环境
            'wait': 5
        },
        {
            'name': 'LLM服务',
            'script': base_dir / 'llm_service' / 'app_fastapi.py',
            'conda_env': 'llm',  # LLM的conda环境
            'wait': 5
        },
        {
            'name': 'TTS服务',
            'script': base_dir / 'tts_service' / 'app_fastapi.py',
            'conda_env': 'tts',  # TTS的conda环境
            'wait': 5
        },
        {
            'name': '主控制服务',
            'script': base_dir / 'orchestrator.py',
            'conda_env': None,  # 使用base环境
            'wait': 3
        },
        {
            'name': 'Web配置界面',
            'script': base_dir / 'web_ui.py',
            'conda_env': None,  # 使用base环境
            'wait': 3
        }
    ]

    # 启动所有服务
    success_count = 0
    for service in services:
        if start_service(
            service['name'],
            str(service['script']),
            service.get('conda_env'),
            service.get('wait', 3)
        ):
            success_count += 1

    logger.info("\n" + "="*60)
    logger.info(f"✅ 成功启动 {success_count}/{len(services)} 个服务")
    logger.info("="*60)

    if success_count > 0:
        # 获取内网IP
        local_ip = get_local_ip()

        logger.info("\n📋 服务访问地址:")
        logger.info(f"  本地访问:")
        logger.info("    • 主控制服务: http://localhost:5000")
        logger.info("    • ASR服务: http://localhost:5001")
        logger.info("    • LLM服务: http://localhost:5002")
        logger.info("    • TTS服务: http://localhost:5003")
        logger.info("    • Web配置界面: http://localhost:8080")

        logger.info(f"\n  内网访问 (局域网其他设备可访问):")
        logger.info(f"    • 主控制服务: http://{local_ip}:5000")
        logger.info(f"    • ASR服务: http://{local_ip}:5001")
        logger.info(f"    • LLM服务: http://{local_ip}:5002")
        logger.info(f"    • TTS服务: http://{local_ip}:5003")
        logger.info(f"    • Web配置界面: http://{local_ip}:8080  ⭐")

        logger.info("\n💡 提示:")
        logger.info(f"  - 在本机访问: http://localhost:8080")
        logger.info(f"  - 在局域网其他设备访问: http://{local_ip}:8080")
        logger.info("  - 按 Ctrl+C 停止所有服务")
        logger.info("\n服务运行中...")

        # 保持运行
        try:
            while True:
                time.sleep(1)
                # 检查是否有进程异常退出
                for service in processes:
                    if service['process'].poll() is not None:
                        logger.error(f"⚠️ {service['name']} 异常退出!")
                        logger.error(f"  查看日志: {service.get('log_file', 'N/A')}")
                        # 只显示一次
                        service['process'] = type('obj', (object,), {'poll': lambda: None})()

        except KeyboardInterrupt:
            pass
    else:
        logger.error("❌ 没有服务成功启动")
        stop_all_services()
        sys.exit(1)


if __name__ == "__main__":
    main()
