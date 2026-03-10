import asyncio
from typing import Dict, Optional
import redis.asyncio as redis
from utils.logger import logger

class MessageBuffer:
    """
    高级真人行为缓冲区：支持消息随时入队，AI 串行处理。
    """
    def __init__(self, host="localhost", port=6379, db=0, wait_time=2.5):
        self.redis = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self.wait_time = wait_time 
        self.active_tasks: Dict[str, asyncio.Task] = {}

    async def add_message(self, session_id: str, message: str):
        """
        不论 AI 是否在忙，统统存入 Redis。
        """
        if message.strip():
            await self.redis.rpush(f"buffer:{session_id}", message.strip())
            await self.redis.expire(f"buffer:{session_id}", 600)

        # 只要有新消息进来，就刷新（或启动）2.5s 的倒计时
        if session_id in self.active_tasks:
            self.active_tasks[session_id].cancel()
        
        self.active_tasks[session_id] = asyncio.create_task(
            self._timer_callback(session_id)
        )

    async def _timer_callback(self, session_id: str):
        try:
            await asyncio.sleep(self.wait_time)
            # 静默期结束，标记为“可取走”
            await self.redis.set(f"ready:{session_id}", "1", ex=60)
            logger.info(f"⏳ Buffer Ready ({session_id}): 用户停顿，准备处理新消息")
        except asyncio.CancelledError:
            pass

    async def get_message_batch(self, session_id: str) -> Optional[dict[str, list[str] | str]]:
        """
        尝试领取任务。
        规则：必须满足 (1) 没人在跑 (lock不存在) 且 (2) 消息已就绪 (ready存在)
        """
        # 1. 检查是否有别人正在跑
        if await self.redis.get(f"lock:{session_id}"):
            return None

        # 2. 检查消息是否已经过了 2.5s 的静默期
        if await self.redis.delete(f"ready:{session_id}") == 1:
            # 只有删除成功的那个请求才是 Winner
            messages = await self.redis.lrange(f"buffer:{session_id}", 0, -1)
            if not messages:
                return None
            
            await self.redis.delete(f"buffer:{session_id}")
            # 设置处理锁，保护本次 AI 运行
            await self.redis.set(f"lock:{session_id}", "1", ex=60)
            return {
                "messages": messages,
                "combined_text": "\n".join(messages),
            }
        
        return None

    async def release_lock(self, session_id: str):
        await self.redis.delete(f"lock:{session_id}")
