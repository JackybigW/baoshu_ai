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

    # 1. 绝对最高优先级：保命逻辑（人工、破冰、闲聊）
    if intent == IntentType.TRANSFER_TO_HUMAN:
        return "human_handoff"
    if intent in [IntentType.GREETING, IntentType.CHIT_CHAT]:
        return "chit_chat"

    # 2. 业务功能优先级：成交信号 > 专家建议 > 身份赛道
    # 只要触发了成交意图或决策辅助，不论身份（VIP/艺术/穷），统一进 consultant 收网版
    if intent == IntentType.SALES_READY:
        return "consultant"
    if intent == IntentType.DECISION_SUPPORT:
        return "consultant"

    # 3. 身份赛道锁定（此时意图是普通咨询或赛道意图）
    if intent == IntentType.HIGH_VALUE or status == "VIP_SERVICE":
        return "high_value"
        
    if intent == IntentType.ART_CONSULTING:
        return "art_director"
        
    if intent == IntentType.LOW_BUDGET: 
        return "low_budget"

    # 4. 业务补漏
    if profile.budget.amount > 0 and profile.budget.amount < 10:
        return "low_budget"
    
    # 5. 常规资料补全
    if not profile.is_complete:
        return "interviewer"

    return "consultant"


def common_tool_router(state: AgentState):
    """
    【决策层 - 通用执行节点路由】
    检查各 Consultant 节点是否触发了摇人工具。
    如果触发了，统一路由到 human_handoff。
    """
    from langgraph.graph import END
    messages = state["messages"]
    last_msg = messages[-1]

    # 检查是否有工具调用请求
    if hasattr(last_msg, "tool_calls") and len(last_msg.tool_calls) > 0:
        logger.info(">>> 🔘 检测到摇人信号，转入人工对接...")
        return "human_handoff"

    # 没有摇人 -> 结束当前轮次，等待用户回复
    return END