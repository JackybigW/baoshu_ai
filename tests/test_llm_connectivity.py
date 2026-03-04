import sys
import os
from langchain_core.messages import HumanMessage

# 设置项目根目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.llm_factory import get_backend_llm, get_frontend_llm

def test_connectivity():
    print("🔍 LLM 备份链路连通性测试 (Fallback System)...\n")

    test_chains = [
        {"name": "Backend Chain (Perception)", "factory": get_backend_llm},
        {"name": "Frontend Chain (Consultants)", "factory": get_frontend_llm},
    ]

    for chain in test_chains:
        print(f"--- 🛠️ 测试链路: {chain['name']} ---")
        try:
            # 获取链路（包含 Fallback 逻辑）
            llm = chain['factory']()
            if not llm:
                print("⚠️ 跳过: 链路未配置 API Key")
                continue
                
            print(f"📡 正在请求: {chain['name']}...")
            res = llm.invoke([HumanMessage(content="你好，请用三个字回复我。")])
            
            content = res.content
            # 处理可能的 list 类型回复
            if isinstance(content, list):
                content = " ".join([block.get("text", "") if isinstance(block, dict) else str(block) for block in content])
            
            print(f"✅ 成功! 响应: {content.strip()}")
            
        except Exception as e:
            print(f"❌ 链路崩溃! 错误详情: {str(e)}")
        print("\n")

if __name__ == "__main__":
    test_connectivity()
