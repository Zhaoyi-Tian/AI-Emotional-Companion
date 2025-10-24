import os
import sys
import time
import logging
import subprocess
import threading
import signal
import psutil
import requests
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ai_assistant.log')
    ]
)
logger = logging.getLogger('AI_Assistant')

class ServiceManager:
    def __init__(self):
        self.services = {
            'asr': {
                'name': '语音识别服务',
                'path': 'asr_service/app.py',
                'env': 'asr_environment',
                'port': 5001,
                'health_endpoint': '/health',
                'process': None,
                'pid': None
            },
            'llm': {
                'name': '大模型服务',
                'path': 'llm_service/app.py',
                'env': 'llm_environment',
                'port': 5002,
                'health_endpoint': '/health',
                'process': None,
                'pid': None
            },
            'tts': {
                'name': '语音合成服务',
                'path': 'tts_service/app.py',
                'env': 'cosyvoice',  # 使用cosyvoice虚拟环境
                'port': 5003,
                'health_endpoint': '/health',
                'process': None,
                'pid': None
            }
        }
        self.pid_dir = 'pids'
        self.main_controller = 'main_controller.py'
        
        # 创建PID目录
        os.makedirs(self.pid_dir, exist_ok=True)
        
    def start_service(self, service_key):
        """启动单个服务"""
        service = self.services[service_key]
        logger.info(f"正在启动 {service['name']}...")
        
        try:
            # 获取虚拟环境的Python解释器路径
            if service['env'] == 'cosyvoice':
                # 特殊处理cosyvoice环境
                python_path = self._get_cosyvoice_python_path()
            else:
                python_path = os.path.join(service['env'], 'bin', 'python')
            
            # 构建启动命令
            cmd = [
                python_path,
                service['path']
            ]
            
            # 启动服务进程
            process = subprocess.Popen(
                cmd,
                stdout=open(f"{service_key}_service.log", 'a'),
                stderr=subprocess.STDOUT,
                start_new_session=True  # 创建新会话组
            )
            
            # 保存进程信息
            service['process'] = process
            service['pid'] = process.pid
            
            # 保存PID到文件
            with open(os.path.join(self.pid_dir, f"{service_key}.pid"), 'w') as f:
                f.write(str(process.pid))
                
            logger.info(f"{service['name']} 启动成功! PID: {process.pid}")
            return True
            
        except Exception as e:
            logger.error(f"启动 {service['name']} 失败: {str(e)}")
            return False
    
    def _get_cosyvoice_python_path(self):
        """获取cosyvoice环境的Python路径"""
        try:
            # 方法1: 尝试使用conda run
            conda_path = self._find_conda_path()
            if conda_path:
                return f"{conda_path} run -n cosyvoice python"
            
            # 方法2: 尝试查找cosyvoice环境的Python解释器
            possible_paths = [
                os.path.expanduser("~/miniconda3/envs/cosyvoice/bin/python"),
                os.path.expanduser("~/anaconda3/envs/cosyvoice/bin/python"),
                "/opt/miniconda3/envs/cosyvoice/bin/python",
                "/opt/anaconda3/envs/cosyvoice/bin/python"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    return path
            
            # 方法3: 使用which命令查找
            try:
                result = subprocess.run(
                    ['which', 'python'],
                    capture_output=True,
                    text=True,
                    check=True
                )
                return result.stdout.strip()
            except:
                pass
            
            # 所有方法都失败，返回默认Python
            return sys.executable
            
        except Exception as e:
            logger.warning(f"获取cosyvoice Python路径失败，使用系统Python: {str(e)}")
            return sys.executable
    
    def _find_conda_path(self):
        """查找conda可执行文件路径"""
        possible_paths = [
            os.path.expanduser("~/miniconda3/bin/conda"),
            os.path.expanduser("~/anaconda3/bin/conda"),
            "/opt/miniconda3/bin/conda",
            "/opt/anaconda3/bin/conda",
            "/usr/bin/conda",
            "/usr/local/bin/conda"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None
    
    def start_all_services(self):
        """启动所有服务"""
        logger.info("="*50)
        logger.info("启动所有AI服务...")
        logger.info("="*50)
        
        # 创建线程启动所有服务
        threads = []
        for service_key in self.services:
            thread = threading.Thread(
                target=self.start_service,
                args=(service_key,),
                daemon=True
            )
            thread.start()
            threads.append(thread)
            time.sleep(2)  # 服务间启动间隔
        
        # 等待所有服务启动线程完成
        for thread in threads:
            thread.join(timeout=30)
        
        # 检查服务健康状态
        time.sleep(5)  # 给服务启动时间
        self.check_services_health()
        
        logger.info("所有服务启动完成!")
    
    def check_service_health(self, service_key):
        """检查单个服务的健康状态"""
        service = self.services[service_key]
        url = f"http://localhost:{service['port']}{service['health_endpoint']}"
        
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                logger.info(f"✅ {service['name']} 健康状态: 正常")
                return True
            else:
                logger.warning(f"⚠️ {service['name']} 健康状态异常: HTTP {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ 无法连接到 {service['name']}: {str(e)}")
            return False
    
    def check_services_health(self):
        """检查所有服务的健康状态"""
        logger.info("检查服务健康状态...")
        
        all_healthy = True
        for service_key in self.services:
            if not self.check_service_health(service_key):
                all_healthy = False
        
        return all_healthy
    
    def stop_service(self, service_key):
        """停止单个服务"""
        service = self.services[service_key]
        if service['process'] and service['process'].poll() is None:
            logger.info(f"正在停止 {service['name']} (PID: {service['pid']})...")
            
            try:
                # 尝试优雅终止
                service['process'].terminate()
                
                # 等待最多10秒
                for _ in range(10):
                    if service['process'].poll() is not None:
                        break
                    time.sleep(1)
                
                # 如果仍然运行，强制终止
                if service['process'].poll() is None:
                    service['process'].kill()
                    logger.warning(f"强制终止 {service['name']}")
                
                logger.info(f"{service['name']} 已停止")
                return True
            except Exception as e:
                logger.error(f"停止 {service['name']} 失败: {str(e)}")
                return False
        else:
            logger.info(f"{service['name']} 未运行或已停止")
            return True
    
    def stop_all_services(self):
        """停止所有服务"""
        logger.info("="*50)
        logger.info("停止所有AI服务...")
        logger.info("="*50)
        
        # 按依赖关系逆序停止服务
        stop_order = ['tts', 'llm', 'asr']
        
        for service_key in stop_order:
            self.stop_service(service_key)
        
        logger.info("所有服务已停止")
    
    def restart_service(self, service_key):
        """重启单个服务"""
        logger.info(f"重启 {self.services[service_key]['name']}...")
        self.stop_service(service_key)
        time.sleep(1)
        return self.start_service(service_key)
    
    def restart_all_services(self):
        """重启所有服务"""
        self.stop_all_services()
        time.sleep(2)
        self.start_all_services()
    
    def run_main_controller(self):
        """运行主控制器"""
        logger.info("="*50)
        logger.info("启动AI助手主控制器...")
        logger.info("="*50)
        
        try:
            # 运行主控制器
            subprocess.run([sys.executable, self.main_controller], check=True)
        except KeyboardInterrupt:
            logger.info("主控制器已停止")
        except Exception as e:
            logger.error(f"主控制器运行失败: {str(e)}")
        finally:
            self.stop_all_services()
    
    def monitor_services(self):
        """监控服务状态，自动恢复"""
        logger.info("启动服务监控...")
        
        while True:
            try:
                for service_key in self.services:
                    if not self.check_service_health(service_key):
                        logger.warning(f"{self.services[service_key]['name']} 异常，尝试重启...")
                        self.restart_service(service_key)
                
                # 每30秒检查一次
                time.sleep(30)
            except KeyboardInterrupt:
                logger.info("服务监控已停止")
                break
            except Exception as e:
                logger.error(f"服务监控出错: {str(e)}")
                time.sleep(60)

def main():
    manager = ServiceManager()
    
    # 启动所有服务
    manager.start_all_services()
    
    # 启动服务监控线程
    monitor_thread = threading.Thread(target=manager.monitor_services, daemon=True)
    monitor_thread.start()
    
    # 运行主控制器
    manager.run_main_controller()
    
    # 主控制器退出后停止所有服务
    manager.stop_all_services()

if __name__ == "__main__":
    main()