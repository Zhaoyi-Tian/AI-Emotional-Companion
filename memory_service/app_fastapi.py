#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语义记忆服务 FastAPI 实现
支持向量化存储和检索记忆
"""

import os
import sys
import json
import time
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from config_loader import get_config

# 导入向量编码器
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MemoryService")

# 创建 FastAPI 应用
app = FastAPI(
    title="语义记忆服务",
    description="基于向量的语义记忆存储和检索",
    version="1.0.0"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求模型
class MemoryRequest(BaseModel):
    text: str
    tags: List[str] = []
    importance: float = 0.5  # 0-1之间，重要度
    memory_type: str = "general"  # personal, preference, event, general

class SearchRequest(BaseModel):
    query: str
    limit: int = 5
    min_similarity: float = 0.3

class ExtractRequest(BaseModel):
    user_message: str
    ai_response: str
    auto_save: bool = True


class SemanticMemoryManager:
    """语义记忆管理器"""

    def __init__(self):
        # 初始化目录
        self.memory_dir = Path("memory_data")
        self.memory_dir.mkdir(exist_ok=True)

        # 存储文件
        self.memory_file = self.memory_dir / "memories.json"
        self.vectors_file = self.memory_dir / "vectors.npy"

        try:
            import torch
            import torch_npu
            # 使用 torch_npu 库来判断 NPU 是否可用
            if hasattr(torch, 'npu') and torch.npu.is_available():
                self.device = torch.device("npu:0")
                logger.info("设备检测: 成功切换到昇腾 NPU")
            elif torch.cuda.is_available(): # 兼容性检查
                self.device = torch.device("cuda")
                logger.info("设备检测: 切换到 CUDA GPU")
            else:
                self.device = torch.device("cpu")
                logger.info("设备检测: 使用 CPU")
        except Exception:
            self.device = "cpu"
            logger.warning("未检测到 PyTorch 环境，使用 CPU。")
        
        # 加载编码器
        try:
            model_path = '/home/HwHiAiUser/ai_助手/model/models--moka-ai--m3e-small/snapshots/44c696631b2a8c200220aaaad5f987f096e986df'
            
            # SentenceTransformer 会自动处理 safetensors 文件，
            # 关键是传入 device 参数
            self.encoder = SentenceTransformer(model_path, device=self.device) 
            logger.info(f"成功加载 sentence-transformers 模型到设备: {self.device}")
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")

        # 加载已有记忆
        self.memories = self._load_memories()
        self.vectors = self._load_vectors()

    def _load_memories(self) -> List[Dict]:
        """加载记忆数据"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"加载记忆失败: {e}")
        return []

    def _load_vectors(self) -> np.ndarray:
        """加载向量数据"""
        if self.vectors_file.exists():
            try:
                return np.load(self.vectors_file)
            except Exception as e:
                logger.error(f"加载向量失败: {e}")
        return np.array([])

    def _save_memories(self):
        """保存记忆数据"""
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.memories, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存记忆失败: {e}")

    def _save_vectors(self):
        """保存向量数据"""
        try:
            np.save(self.vectors_file, self.vectors)
        except Exception as e:
            logger.error(f"保存向量失败: {e}")

    def encode_text(self, text: str) -> np.ndarray:
        """将文本编码为向量"""
        try:
            return self.encoder.encode(text)
        except Exception as e:
            logger.error(f"编码文本失败: {e}")
            return np.zeros(384)  # 模型维度是384

    def add_memory(self, text: str, tags: List[str] = None,
                   importance: float = 0.5, memory_type: str = "general"):
        """添加记忆"""
        # 编码文本
        vector = self.encode_text(text)

        # 创建记忆项
        memory = {
            "text": text,
            "tags": tags or [],
            "importance": importance,
            "type": memory_type,
            "timestamp": datetime.now().isoformat(),
            "access_count": 0
        }

        # 添加到列表
        self.memories.append(memory)

        # 添加向量
        if len(self.vectors) == 0:
            self.vectors = vector.reshape(1, -1)
        else:
            self.vectors = np.vstack([self.vectors, vector])

        # 限制记忆数量（保留最重要的1000条）
        if len(self.memories) > 1000:
            # 按重要性和访问次数排序
            self.memories.sort(
                key=lambda x: (x["importance"], x["access_count"]),
                reverse=True
            )
            self.memories = self.memories[:1000]

            # 重新编码向量
            self.vectors = np.array([
                self.encode_text(m["text"]) for m in self.memories
            ])

        # 保存到文件
        self._save_memories()
        self._save_vectors()

        logger.info(f"添加记忆: {text[:50]}...")
        return memory

    def search_memories(self, query: str, limit: int = 5,
                        min_similarity: float = 0.3) -> List[Dict]:
        """搜索相关记忆"""
        if len(self.vectors) == 0:
            return []

        # 编码查询
        query_vector = self.encode_text(query).reshape(1, -1)

        # 计算相似度
        similarities = cosine_similarity(query_vector, self.vectors)[0]

        # 获取最相关的记忆
        results = []
        indices = np.argsort(similarities)[::-1][:limit]

        for idx in indices:
            if similarities[idx] >= min_similarity:
                memory = self.memories[idx].copy()
                memory["similarity"] = float(similarities[idx])

                # 更新访问次数
                self.memories[idx]["access_count"] += 1

                results.append(memory)

        # 保存更新的访问次数
        self._save_memories()

        return results

    def get_context_for_query(self, query: str, max_items: int = 3) -> str:
        """获取查询相关的上下文"""
        memories = self.search_memories(query, limit=max_items)

        if not memories:
            return ""

        context_parts = ["【相关记忆】"]
        for mem in memories:
            context_parts.append(f"- {mem['text']}")
            if mem.get('type') == 'preference':
                context_parts.append("  (用户偏好)")
            elif mem.get('type') == 'personal':
                context_parts.append("  (个人信息)")
            elif mem.get('type') == 'event':
                context_parts.append("  (重要事件)")

        return "\n".join(context_parts)

    def auto_extract_memories(self, user_message: str, ai_response: str):
        """自动从对话中提取记忆"""
        # 1. 提取偏好
        preferences = self._extract_preferences(user_message)
        for pref in preferences:
            self.add_memory(pref, tags=["preference"], importance=0.8, memory_type="preference")

        # 2. 提取个人信息
        facts = self._extract_facts(user_message)
        for fact in facts:
            self.add_memory(fact, tags=["fact"], importance=0.9, memory_type="personal")

        # 3. 提取事件
        events = self._extract_events(user_message)
        for event in events:
            self.add_memory(event, tags=["event"], importance=0.7, memory_type="event")

        # 4. 存储完整对话
        self.add_memory(
            f"用户: {user_message}\n助手: {ai_response}",
            tags=["conversation"],
            importance=0.5,
            memory_type="general"
        )

        return {
            "preferences_found": len(preferences),
            "facts_found": len(facts),
            "events_found": len(events)
        }

    def _extract_preferences(self, text: str) -> List[str]:
        """提取偏好信息"""
        preferences = []

        # 偏好关键词
        patterns = [
            "我喜欢", "我不喜欢", "我爱", "我讨厌",
            "我偏爱", "我喜欢的是", "我不喜欢的是",
            "我喜欢听", "我喜欢吃", "我喜欢看"
        ]

        for pattern in patterns:
            if pattern in text:
                sentences = text.split("。")
                for sentence in sentences:
                    if pattern in sentence:
                        # 简单清理
                        pref = sentence.strip()
                        if len(pref) > 10:  # 过滤太短的
                            preferences.append(pref)

        return preferences

    def _extract_facts(self, text: str) -> List[str]:
        """提取事实信息"""
        facts = []

        # 事实关键词
        patterns = [
            "我叫", "我的名字是", "我是", "我住在",
            "我的工作", "我的专业", "我毕业于",
            "我家有", "我岁", "我今年"
        ]

        for pattern in patterns:
            if pattern in text:
                sentences = text.split("。")
                for sentence in sentences:
                    if pattern in sentence:
                        fact = sentence.strip()
                        if len(fact) > 10:
                            facts.append(fact)

        return facts

    def _extract_events(self, text: str) -> List[str]:
        """提取事件信息"""
        events = []

        # 时间和事件关键词
        patterns = [
            "明天", "下周", "下个月", "即将",
            "计划", "要去", "要参加", "会议",
            "约定", "安排", "预约"
        ]

        for pattern in patterns:
            if pattern in text:
                # 提取包含时间词的整句话
                events.append(text.strip())
                break  # 避免重复

        return events

    def get_all_memories(self) -> List[Dict]:
        """获取所有记忆"""
        return self.memories

    def delete_memory(self, index: int) -> bool:
        """删除指定索引的记忆"""
        if 0 <= index < len(self.memories):
            # 删除记忆
            del self.memories[index]

            # 删除向量
            if index < len(self.vectors):
                self.vectors = np.delete(self.vectors, index, axis=0)

            # 保存
            self._save_memories()
            self._save_vectors()

            return True
        return False


# 创建全局记忆管理器
memory_manager = SemanticMemoryManager()


@app.post("/memory/add")
async def add_memory(request: MemoryRequest):
    """添加记忆"""
    try:
        memory = memory_manager.add_memory(
            text=request.text,
            tags=request.tags,
            importance=request.importance,
            memory_type=request.memory_type
        )
        return {
            "status": "success",
            "memory": memory
        }
    except Exception as e:
        logger.error(f"添加记忆失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/search")
async def search_memories(query: str, limit: int = 5, min_similarity: float = 0.3):
    """搜索记忆"""
    try:
        memories = memory_manager.search_memories(
            query=query,
            limit=limit,
            min_similarity=min_similarity
        )
        return {
            "status": "success",
            "query": query,
            "results": memories,
            "count": len(memories)
        }
    except Exception as e:
        logger.error(f"搜索记忆失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/context")
async def get_context(query: str, max_items: int = 3):
    """获取上下文"""
    try:
        context = memory_manager.get_context_for_query(query, max_items)
        return {
            "status": "success",
            "query": query,
            "context": context,
            "has_context": bool(context)
        }
    except Exception as e:
        logger.error(f"获取上下文失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/extract")
async def auto_extract_memories(request: ExtractRequest):
    """自动提取记忆"""
    try:
        result = memory_manager.auto_extract_memories(
            user_message=request.user_message,
            ai_response=request.ai_response
        )
        return {
            "status": "success",
            "extracted": result
        }
    except Exception as e:
        logger.error(f"自动提取失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/list")
async def list_memories(limit: int = 50):
    """列出记忆"""
    try:
        memories = memory_manager.get_all_memories()
        # 按时间倒序
        memories = sorted(memories, key=lambda x: x["timestamp"], reverse=True)

        return {
            "status": "success",
            "memories": memories[:limit],
            "total": len(memories)
        }
    except Exception as e:
        logger.error(f"列出记忆失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memory/{index}")
async def delete_memory(index: int):
    """删除记忆"""
    try:
        success = memory_manager.delete_memory(index)
        if success:
            return {
                "status": "success",
                "message": f"记忆 {index} 已删除"
            }
        else:
            raise HTTPException(status_code=404, detail="记忆索引不存在")
    except Exception as e:
        logger.error(f"删除记忆失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "service": "Memory Service",
        "total_memories": len(memory_manager.get_all_memories()),
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    # 使用端口5006
    uvicorn.run(app, host="0.0.0.0", port=5006, log_level="info")