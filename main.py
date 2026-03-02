import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
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
        
        # 优化消息提取逻辑：
        # 1. 找到最后一个 HumanMessage 的位置
        # 2. 收集此后所有的 AIMessage，过滤掉 ToolMessage 和静默的 tool_calls
        all_messages = output.get("messages", [])
        last_human_index = -1
        for i in range(len(all_messages) - 1, -1, -1):
            if isinstance(all_messages[i], HumanMessage):
                last_human_index = i
                break
        
        ai_contents = []
        for i in range(last_human_index + 1, len(all_messages)):
            msg = all_messages[i]
            if isinstance(msg, AIMessage) and msg.content:
                # 过滤掉加粗（如果有遗留），并提取内容
                clean_content = msg.content.replace("**", "").strip()
                if clean_content:
                    ai_contents.append(clean_content)
            # 自动跳过 ToolMessage 和没有 content 的 AIMessage
        
        # 3. 组合内容，使用 ||| 方便前端分段处理
        if ai_contents:
            content = "|||".join(ai_contents)
        else:
            # 兜底内容
            content = "好的，我刚才没听清，您能再说一遍吗？"

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
