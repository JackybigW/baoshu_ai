import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv, find_dotenv
import uuid
from typing import Optional # 🔥 必须引入这个
from utils.logger import logger

# 1. 加载环境变量
_ = load_dotenv(find_dotenv(), override=True)

# 2. 导入你的图
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

@api.post("/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        # 如果没有 session_id，生成一个新的
        if not req.session_id:
            req.session_id = str(uuid.uuid4())
            logger.info(f"🆕 New Session Created: {req.session_id}")
        
        config = {"configurable": {"thread_id": req.session_id}}
        log_msg = req.message if req.message.strip() else "[触发开场白]"
        logger.info(f"👤 User ({req.session_id}): {log_msg}")
        if not req.message.strip():
            inputs = {"messages": []} # 空列表触发 route_entry -> first_greeting
        else:
            inputs = {"messages": [HumanMessage(content=req.message)]}
        
        # 调用 LangGraph
        output = app.invoke(inputs, config=config)
        
        content = ""
        if output["messages"] and len(output["messages"]) > 0:
            last_message = output["messages"][-1]
            content = last_message.content
        preview_content = content[:50] + "..." if len(content) > 50 else content
        logger.info(f"🤖 暴叔 ({req.session_id}): {preview_content}")
        
        
        return {
            "response": content,
            "session_id": req.session_id
        }

    except Exception as e:
        logger.error(f"❌ System Error ({req.session_id}): {str(e)}")
        # 返回 500 状态码，前端会捕获
        raise HTTPException(status_code=500, detail=str(e))

api.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    print("🚀 Server is running on http://0.0.0.:8000")
    uvicorn.run(api, host="0.0.0.0", port=8000)
