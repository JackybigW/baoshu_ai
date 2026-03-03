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

buffer = MessageBuffer(wait_time=2.5)

@api.post("/chat")
async def chat_endpoint(req: ChatRequest):
    if not req.session_id:
        req.session_id = str(uuid.uuid4())
    
    session_id = req.session_id
    message_text = req.message.strip()

    # 1. 纯开场白逻辑（不进入缓冲区）
    if not message_text:
        config = {"configurable": {"thread_id": session_id}}
        output = await asyncio.to_thread(app.invoke, {"messages": []}, config=config)
        return {"response": output["messages"][-1].content, "session_id": session_id}

    # 2. 存入缓冲区
    # 如果 AI 正在处理（锁在），提示用户稍等，防止插话导致 LangGraph 报错
    if await buffer.redis.get(f"lock:{session_id}"):
        return {"response": "暴叔正在思考中，请稍后再发...", "session_id": session_id}

    await buffer.add_message(session_id, message_text)

    # 3. 轮询等待领取任务
    max_wait = 12 
    start_time = asyncio.get_event_loop().time()
    
    while (asyncio.get_event_loop().time() - start_time) < max_wait:
        # 尝试领取任务
        combined_text = await buffer.get_merged_message(session_id)
        
        if combined_text:
            # 【我是唯一的执行者】
            try:
                logger.info(f"🚀 AI Processing ({session_id}): 合并消息为 -> {combined_text}")
                config = {"configurable": {"thread_id": session_id}}
                inputs = {"messages": [HumanMessage(content=combined_text)]}
                
                output = await asyncio.to_thread(app.invoke, inputs, config=config)
                
                # 提取回复
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
                
                response_text = "|||".join(ai_contents) if ai_contents else "好的，收到了。"
                await buffer.release_lock(session_id)
                return {"response": response_text, "session_id": session_id}
            
            except Exception as e:
                await buffer.release_lock(session_id)
                logger.error(f"❌ AI Error: {str(e)}")
                raise HTTPException(status_code=500, detail="系统故障。")
        
        # 【检查我是否是陪跑者】
        # 如果还没领到任务，但发现锁已经被别人（同一 session 的其他请求）拿走了
        # 说明我已经“交接”成功了
        if await buffer.redis.get(f"lock:{session_id}"):
            return {"response": "__MERGED__", "session_id": session_id}

        await asyncio.sleep(0.5)

    return {"response": "__MERGED__", "session_id": session_id}

api.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(api, host="0.0.0.0", port=8000)
