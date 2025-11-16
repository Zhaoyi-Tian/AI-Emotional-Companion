#!/usr/bin/env python3
"""
配置热重载功能测试脚本

测试各个服务的配置热重载功能是否正常工作
"""

import requests
import time
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent))
from config_loader import get_config

# 服务端口
SERVICES = {
    'ASR': 5001,
    'LLM': 5002,
    'TTS': 5003,
    'Orchestrator': 5000,
    'VoiceChat': 5004
}


def test_service_health(service_name, port):
    """测试服务健康状态"""
    try:
        url = f"http://localhost:{port}/health"
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            print(f"✅ {service_name} 服务运行正常 (端口 {port})")
            return True
        else:
            print(f"⚠️ {service_name} 服务响应异常 (HTTP {response.status_code})")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ {service_name} 服务未运行 (端口 {port})")
        return False
    except Exception as e:
        print(f"❌ {service_name} 服务检查失败: {e}")
        return False


def test_service_reload(service_name, port):
    """测试服务配置重新加载"""
    try:
        url = f"http://localhost:{port}/reload_config"
        print(f"\n📡 测试 {service_name} 配置热重载...")

        response = requests.post(url, timeout=30)

        if response.status_code == 200:
            result = response.json()
            if result.get('success', True):
                print(f"✅ {service_name} 配置热重载成功")
                if 'message' in result:
                    print(f"   消息: {result['message']}")
                if 'changes' in result:
                    print(f"   变更: {result['changes']}")
                if 'streaming_config' in result:
                    print(f"   流式配置: {result['streaming_config']}")
                return True
            else:
                error_msg = result.get('message', result.get('error', '未知错误'))
                print(f"⚠️ {service_name} 配置热重载失败: {error_msg}")
                return False
        else:
            print(f"❌ {service_name} 配置热重载失败 (HTTP {response.status_code})")
            return False

    except requests.exceptions.Timeout:
        print(f"⏱️ {service_name} 配置重新加载超时（可能模型较大）")
        return False
    except Exception as e:
        print(f"❌ {service_name} 配置重新加载出错: {e}")
        return False


def main():
    """主测试函数"""
    print("=" * 60)
    print("🔧 配置热重载功能测试")
    print("=" * 60)

    # 第一步：检查所有服务健康状态
    print("\n第一步：检查服务健康状态")
    print("-" * 60)

    healthy_services = []
    for service_name, port in SERVICES.items():
        if test_service_health(service_name, port):
            healthy_services.append(service_name)

    if not healthy_services:
        print("\n❌ 没有服务在运行，请先启动服务")
        print("   提示：运行 python start_all.py")
        return 1

    print(f"\n✅ 发现 {len(healthy_services)}/{len(SERVICES)} 个服务正在运行")

    # 第二步：测试配置热重载
    print("\n第二步：测试配置热重载功能")
    print("-" * 60)

    reload_success = []
    for service_name in healthy_services:
        port = SERVICES[service_name]
        if test_service_reload(service_name, port):
            reload_success.append(service_name)
        time.sleep(0.5)  # 避免请求过快

    # 总结
    print("\n" + "=" * 60)
    print("📊 测试总结")
    print("=" * 60)
    print(f"运行中的服务: {len(healthy_services)}/{len(SERVICES)}")
    print(f"配置热重载成功: {len(reload_success)}/{len(healthy_services)}")

    if reload_success:
        print(f"\n✅ 以下服务支持配置热重载:")
        for service in reload_success:
            print(f"   - {service}")

    failed_services = set(healthy_services) - set(reload_success)
    if failed_services:
        print(f"\n⚠️ 以下服务热重载失败:")
        for service in failed_services:
            print(f"   - {service}")

    # 使用说明
    print("\n" + "=" * 60)
    print("💡 使用说明")
    print("=" * 60)
    print("1. 在 Web UI (http://localhost:8080) 中修改配置并保存")
    print("   - 配置会自动调用对应服务的热重载功能")
    print("   - 大部分配置立即生效，无需重启服务")
    print("")
    print("2. 手动测试热重载:")
    print("   - 修改 config.yaml 文件")
    print("   - 运行此测试脚本验证热重载是否成功")
    print("")
    print("3. 注意事项:")
    print("   - ASR/LLM 模型切换需要更长的重载时间")
    print("   - Voice Chat 的音频设备配置需要重启才能生效")
    print("   - 其他配置大多支持热重载")

    return 0 if len(reload_success) == len(healthy_services) else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
