# 决策层: 纯逻辑路由，不调用 LLM
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from state import AgentState, IntentType


from utils.logger import logger

def core_router(state: AgentState):
    """
    【决策层 - 核心路由器】
    纯逻辑判断，根据 state 决定下一步走向。不调用 LLM。
    """
    logger.info("--- 🎯 Decision: 核心路由 ---")

    intent = state.get("last_intent")
    profile = state["profile"]
    status = state.get("dialog_status")

    # 1. 最高优先级
    if intent == IntentType.TRANSFER_TO_HUMAN:
        return "human_handoff"
    if intent in [IntentType.GREETING, IntentType.CHIT_CHAT]:
        return "chit_chat"

    # 2. 身份赛道分流
    if intent == IntentType.HIGH_VALUE or status == "VIP_SERVICE":
        return "high_value"
        
    if intent == IntentType.ART_CONSULTING:
        return "art_director"
        
    if intent == IntentType.LOW_BUDGET:  # 直接看意图，干净清爽
        return "low_budget"

    # 3. 业务补漏 (针对 intent 是 NEED_CONSULTING 的情况)
    # 如果意图没识别出穷鬼，但画像里确实没钱，补刀：
    if profile.budget.amount > 0 and profile.budget.amount < 10:
        return "low_budget"
    
    if intent == IntentType.DECISION_SUPPORT:
        return "consultant"
    if intent == IntentType.SALES_READY:
        return "consultant"

    # 4. 常规流程
    if not profile.is_complete:
        return "interviewer"

    return "consultant"


def route_high_value(state: AgentState):
    """
    【决策层 - VIP 路由】
    检查 High Value Node 是否触发了摇人工具。
    """
    from langgraph.graph import END
    messages = state["messages"]
    last_msg = messages[-1]

    # 检查是否有工具调用请求
    if hasattr(last_msg, "tool_calls") and len(last_msg.tool_calls) > 0:
        print(">>> 🔘 检测到 VIP 摇人信号，转入人工对接...")
        return "human_handoff"

    # 没有摇人 -> 结束当前轮次，等待用户回复
    return END


def route_low_budget(state: AgentState):
    """
    【决策层 - 低预算路由】
    检查 Low Budget Node 是否触发了摇人工具。
    """
    from langgraph.graph import END
    messages = state["messages"]
    last_msg = messages[-1]

    # 检查是否有工具调用请求
    if hasattr(last_msg, "tool_calls") and len(last_msg.tool_calls) > 0:
        print(">>> 🔘 检测到 Low Budget 摇人信号，转入人工对接...")
        return "human_handoff"

    # 没有摇人 -> 结束当前轮次，等待用户回复
    return END


def route_art_director(state: AgentState):
    """
    【决策层 - 艺术留学路由】
    检查 Art Node 是否触发了摇人工具。
    """
    from langgraph.graph import END
    messages = state["messages"]
    last_msg = messages[-1]

    # 检查是否有工具调用请求
    if hasattr(last_msg, "tool_calls") and len(last_msg.tool_calls) > 0:
        print(">>> 🔘 检测到 Art 摇人信号，转入人工对接...")
        return "human_handoff"

    # 没有摇人 -> 结束当前轮次，等待用户回复
    return END


def route_consultant(state: AgentState):
    """
    【决策层 - Consultant 路由】
    检查 Consultant Node 是否触发了摇人工具。
    """
    from langgraph.graph import END
    messages = state["messages"]
    last_msg = messages[-1]

    # 检查是否有工具调用请求
    if hasattr(last_msg, "tool_calls") and len(last_msg.tool_calls) > 0:
        print(">>> 🔘 检测到 Consultant 摇人信号，转入人工对接...")
        return "human_handoff"

    # 没有摇人 -> 结束当前轮次，等待用户回复
    return END