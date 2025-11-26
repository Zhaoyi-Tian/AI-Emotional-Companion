"""
记忆服务客户端
为语音助手提供记忆功能
"""

import requests
import logging
from typing import List, Dict, Optional
from datetime import datetime
import time

logger = logging.getLogger("MemoryClient")


class MemoryClient:
    """记忆服务客户端"""

    def __init__(self, service_url: str = "http://localhost:5006"):
        self.service_url = service_url
        self.timeout = 5

    def add_memory(self, text: str, tags: List[str] = None,
                   importance: float = 0.5, memory_type: str = "general") -> Optional[Dict]:
        """添加记忆"""
        try:
            response = requests.post(
                f"{self.service_url}/memory/add",
                json={
                    "text": text,
                    "tags": tags or [],
                    "importance": importance,
                    "memory_type": memory_type
                },
                timeout=self.timeout
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "success":
                    logger.info(f"添加记忆成功: {text[:50]}...")
                    return result.get("memory")
                else:
                    logger.error(f"添加记忆失败: {result}")
            else:
                logger.error(f"添加记忆请求失败: {response.status_code}")

        except Exception as e:
            logger.error(f"添加记忆出错: {e}")

        return None

    def search_memories(self, query: str, limit: int = 5,
                         min_similarity: float = 0.3) -> List[Dict]:
        """搜索记忆"""
        try:
            response = requests.get(
                f"{self.service_url}/memory/search",
                params={
                    "query": query,
                    "limit": limit,
                    "min_similarity": min_similarity
                },
                timeout=self.timeout
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "success":
                    return result.get("results", [])

        except Exception as e:
            logger.error(f"搜索记忆出错: {e}")

        return []

    def get_context(self, query: str, max_items: int = 3) -> str:
        """获取上下文"""
        try:
            response = requests.get(
                f"{self.service_url}/memory/context",
                params={
                    "query": query,
                    "max_items": max_items
                },
                timeout=self.timeout
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "success":
                    return result.get("context", "")

        except Exception as e:
            logger.error(f"获取上下文出错: {e}")

        return ""

    def auto_extract(self, user_message: str, ai_response: str) -> Dict:
        """自动提取记忆"""
        try:
            response = requests.post(
                f"{self.service_url}/memory/extract",
                json={
                    "user_message": user_message,
                    "ai_response": ai_response,
                    "auto_save": True
                },
                timeout=10  # 需要更长时间，因为要处理文本
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "success":
                    extracted = result.get("extracted", {})
                    logger.info(f"自动提取记忆: {extracted}")
                    return extracted

        except Exception as e:
            logger.error(f"自动提取记忆出错: {e}")

        return {}

    def list_memories(self, limit: int = 50) -> List[Dict]:
        """列出记忆"""
        try:
            response = requests.get(
                f"{self.service_url}/memory/list",
                params={"limit": limit},
                timeout=self.timeout
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "success":
                    return result.get("memories", [])

        except Exception as e:
            logger.error(f"列出记忆出错: {e}")

        return []

    def delete_memory(self, index: int) -> bool:
        """删除记忆"""
        try:
            response = requests.delete(
                f"{self.service_url}/memory/{index}",
                timeout=self.timeout
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "success":
                    logger.info(f"删除记忆 {index} 成功")
                    return True

        except Exception as e:
            logger.error(f"删除记忆出错: {e}")

        return False

    def check_service(self) -> bool:
        """检查服务是否可用"""
        try:
            response = requests.get(
                f"{self.service_url}/health",
                timeout=2
            )
            return response.status_code == 200
        except:
            return False


# 测试代码
if __name__ == "__main__":
    client = MemoryClient()

    # 测试服务
    if client.check_service():
        print("记忆服务运行正常")

        # 添加测试记忆
        memory = client.add_memory(
            "用户喜欢听古典音乐，特别是巴赫的作品",
            tags=["音乐", "偏好"],
            importance=0.9,
            memory_type="preference"
        )
        print(f"添加的记忆: {memory}")

        # 搜索记忆
        results = client.search_memories("音乐")
        print(f"搜索结果: {results}")

        # 获取上下文
        context = client.get_context("音乐相关")
        print(f"上下文: {context}")
    else:
        print("记忆服务未运行")