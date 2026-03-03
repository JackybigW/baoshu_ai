import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv, find_dotenv
import uuid
import asyncio
from typing import Optional
from utils.logger import logger
from utils.buffer import MessageBuffer

_ = load_dotenv(find_dotenv(), override=True)
from agent_graph import app 

class ChatRequest(BaseModel):
    message: str = "" 
    session_id: Optional[str] = None

api = FastAPI()
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2.5s 仍然是黄金等待期
buffer = MessageBuffer(wait_time=2.5)

@api.post("/chat")
async def chat_endpoint(req: ChatRequest):
    if not req.session_id:
        req.session_id = str(uuid.uuid4())
    
    session_id = req.session_id
    message_text = req.message.strip()

    # 1. 开场白逻辑
    if not message_text:
        config = {"configurable": {"thread_id": session_id}}
        output = await asyncio.to_thread(app.invoke, {"messages": []}, config=config)
        return {"response": output["messages"][-1].content, "session_id": session_id}

    # 2. 直接入队，不再拦截插嘴
    await buffer.add_message(session_id, message_text)

    # 3. 轮询等待领取任务权 (或被合并)
    max_wait = 25 # 考虑到排队，等待时间稍微延长到 25 秒
    start_time = asyncio.get_event_loop().time()
    
    while (asyncio.get_event_loop().time() - start_time) < max_wait:
        combined_text = await buffer.get_merged_message(session_id)
        
        if combined_text:
            # 我是这一波消息的执行者（Winner）
            try:
                logger.info(f"🚀 AI Winner Processing ({session_id}): {combined_text}")
                config = {"configurable": {"thread_id": session_id}}
                inputs = {"messages": [HumanMessage(content=combined_text)]}
                
                output = await asyncio.to_thread(app.invoke, inputs, config=config)
                
                all_messages = output.get("messages", [])
                ai_contents = []
                last_human_index = -1
                for i in range(len(all_messages) - 1, -1, -1):
                    if isinstance(all_messages[i], HumanMessage):
                        last_human_index = i
                        break
                for i in range(last_human_index + 1, len(all_messages)):
                    msg = all_messages[i]
                    if isinstance(msg, AIMessage) and msg.content:
                        ai_contents.append(msg.content.strip())
                
                response_text = "|||".join(ai_contents) if ai_contents else "好的。"
                await buffer.release_lock(session_id) # AI 跑完，释放锁，让下一波排队的进来
                return {"response": response_text, "session_id": session_id}
            
            except Exception as e:
                await buffer.release_lock(session_id)
                logger.error(f"❌ AI Error: {str(e)}")
                raise HTTPException(status_code=500, detail="系统故障。")
        
        # 这里的检查依然重要：如果我发现缓冲区空了且锁被释放了，
        # 说明我的消息已经被同一波的某个 Winner 领走并处理完了。
        buf_len = await buffer.redis.llen(f"buffer:{session_id}")
        if buf_len == 0 and not await buffer.redis.get(f"lock:{session_id}") and not await buffer.redis.get(f"ready:{session_id}"):
             # 安全退出：这里可能有点复杂，为了 Web 端不出错，我们返回一个信号
             return {"response": "__MERGED__", "session_id": session_id}

        await asyncio.sleep(0.5)

    return {"response": "__MERGED__", "session_id": session_id}

api.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(api, host="0.0.0.0", port=8000)
