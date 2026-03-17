# -*- coding: utf-8 -*-
import os
import sys
from typing import List

# --- 1. 确保能导入项目中的模块 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# --- 2. 导入组件 ---
from langchain_core.messages import HumanMessage, AIMessage
from state import AgentState, CustomerProfile, IntentType
from nodes.consultants import high_value_node
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def print_header(title: str):
    print(f"\n{'='*25} {title} {'='*25}")

def test_high_value_progression():
    """
    自动化测试：验证 High Value Node 在不同消息长度下的阶段表现。
    """
    print_header("VIP 顾问节点：动态阶段演进测试")
    
    # 模拟一个高净值客户画像
    profile = CustomerProfile(
        user_role="学生",
        educationStage="本科",
        destination_preference=["美国"],
        academic_background="GPA 3.8, 本科 Top 50"
    )
    profile.budget.amount = 500  # 500w 预算
    
    state: AgentState = {
        "messages": [],
        "profile": profile,
        "has_proposed_solution": False,
        "dialog_status": "VIP_SERVICE",
        "last_intent": IntentType.HIGH_VALUE
    }

    # 测试场景 A：前期 (建立信任)
    print("\n[阶段 1: 建立信任 (Message Count < 7)]")
    state["messages"] = [HumanMessage(content="预算500w，想去美国读研，有什么建议？")]
    res = high_value_node(state)
    for msg in res["messages"]:
        if msg.content:
            print(f"🤖 暴叔: {msg.content}")

    # 测试场景 B：中期 (指出软肋)
    print("\n[阶段 2: 指出软肋 (Message Count 7-12)]")
    # 构造一些历史消息使 count 达到 8
    state["messages"] = [HumanMessage(content="Hi")] * 8 + [HumanMessage(content="我觉得美国挺好的，但我担心现在的背景够不够？")]
    res = high_value_node(state)
    for msg in res["messages"]:
        if msg.content:
            print(f"🤖 暴叔: {msg.content}")

    # 测试场景 C：后期 (强力收网)
    print("\n[阶段 3: 强力收网 (Message Count >= 13)]")
    # 构造消息使 count 达到 14
    state["messages"] = [HumanMessage(content="Hi")] * 14 + [HumanMessage(content="咱们具体怎么操作这个内部通道？")]
    res = high_value_node(state)
    for msg in res["messages"]:
        if msg.content:
            print(f"🤖 暴叔: {msg.content}")
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print(f"🔧 [工具触发] 调用拉群工具: {msg.tool_calls[0]['name']}")

    # 测试场景 D：Sales Ready (无论长度，立即收网)
    print("\n[特殊场景: SALES_READY (即时收网)]")
    state["messages"] = [HumanMessage(content="我觉得这个保录方案非常靠谱，怎么签约？")]
    state["last_intent"] = IntentType.SALES_READY
    res = high_value_node(state)
    for msg in res["messages"]:
        if msg.content:
            print(f"🤖 暴叔: {msg.content}")
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print(f"🔧 [工具触发] 立即调用拉群工具")

if __name__ == "__main__":
    test_high_value_progression()
