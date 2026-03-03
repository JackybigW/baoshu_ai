import asyncio
import json
from typing import Dict, Any, Optional
import redis.asyncio as redis
from utils.logger import logger

class MessageBuffer:
    def __init__(self, host="localhost", port=6379, db=0, wait_time=2.0):
        self.redis = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self.wait_time = wait_time 
        self.active_tasks: Dict[str, asyncio.Task] = {}

    async def add_message(self, session_id: str, message: str) -> bool:
        # 如果当前有处理锁，说明 AI 正在说话，直接拦截，不让插话
        if await self.redis.get(f"lock:{session_id}"):
            return False

        if message.strip():
            await self.redis.rpush(f"buffer:{session_id}", message.strip())
            await self.redis.expire(f"buffer:{session_id}", 600)

        # 刷新计时器
        if session_id in self.active_tasks:
            self.active_tasks[session_id].cancel()
        
        self.active_tasks[session_id] = asyncio.create_task(self._timer_callback(session_id))
        return True

    async def _timer_callback(self, session_id: str):
        try:
            await asyncio.sleep(self.wait_time)
            # 标记为就绪
            await self.redis.set(f"ready:{session_id}", "1", ex=60)
            logger.info(f"⏳ Buffer Ready ({session_id})")
        except asyncio.CancelledError:
            pass

    async def get_merged_message(self, session_id: str) -> Optional[str]:
        """
        原子化领取任务：
        使用 delete 操作。Redis 保证只有一个请求能得到返回值为 1。
        """
        # 尝试删除 ready 键
        if await self.redis.delete(f"ready:{session_id}") == 1:
            # 我是胜出者，负责取走所有消息
            messages = await self.redis.lrange(f"buffer:{session_id}", 0, -1)
            await self.redis.delete(f"buffer:{session_id}")
            # 设置处理锁，防止 AI 运行期间新的消息进来把逻辑搞乱
            await self.redis.set(f"lock:{session_id}", "1", ex=60)
            return "\n".join(messages)
        return None

    async def release_lock(self, session_id: str):
        await self.redis.delete(f"lock:{session_id}")
