from langchain_core.messages import HumanMessage
from agent_graph import app
from dotenv import load_dotenv
import os

load_dotenv()

def test_debug():
    print("🚀 Starting Local Debug (Clean state)...")
    session_id = "debug_session_123"
    config = {"configurable": {"thread_id": session_id}}
    inputs = {"messages": [HumanMessage(content="预算5000w 去哪留学？")]}
    
    try:
        print("📡 Invoking app...")
        output = app.invoke(inputs, config=config)
        print("✅ Success!")
        
        # 打印最后几条 AI 消息
        all_messages = output.get("messages", [])
        for msg in all_messages[-3:]:
             if hasattr(msg, "content") and msg.content:
                 print(f"[{type(msg).__name__}]: {msg.content}")
                 
    except Exception as e:
        print(f"❌ Caught Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_debug()
