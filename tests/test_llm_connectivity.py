import sys
import os
from langchain_core.messages import HumanMessage

# 设置项目根目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.llm_factory import get_backend_llm, get_frontend_llm, get_llm

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


def test_model_registry_connectivity():
    print("🔍 LLM 工厂模型连通性测试...\n")

    model_cases = [
        {"name": "DeepSeek", "model_id": "deepseek"},
        {"name": "Gemini Flash", "model_id": "gemini_flash"},
        {"name": "GLM", "model_id": "glm"},
        {"name": "Doubao Pro", "model_id": "doubao"},
        {"name": "Doubao Lite", "model_id": "doubao_lite"},
    ]

    for case in model_cases:
        print(f"--- 🛠️ 测试模型: {case['name']} ({case['model_id']}) ---")
        try:
            llm = get_llm(case["model_id"])
            if not llm:
                print("⚠️ 跳过: 模型缺少必要配置（通常是 API Key / Base URL）")
                continue

            print(f"📡 正在请求: {case['name']}...")
            res = llm.invoke([HumanMessage(content="你好，请只回复'收到'两个字。")])

            content = res.content
            if isinstance(content, list):
                content = " ".join(
                    [block.get("text", "") if isinstance(block, dict) else str(block) for block in content]
                )

            print(f"✅ 成功! 响应: {str(content).strip()}")
        except Exception as e:
            print(f"❌ 失败! 错误详情: {str(e)}")
        print("\n")

if __name__ == "__main__":
    test_connectivity()
    test_model_registry_connectivity()
