# ==========================================
# 暴叔AI - 留学顾问智能体
# 三层架构：感知层 -> 决策层 -> 执行层
# 🔥 核心特性：感知层 classifier 和 extractor 并行执行
# ==========================================

from langgraph.checkpoint.memory import MemorySaver
from state import AgentState
from langgraph.graph import StateGraph, END

# 感知层
from nodes.perception import classifier_node, extractor_node

# 决策层路由逻辑
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from router import core_router, route_high_value, route_low_budget, route_art_director, route_consultant

# 执行层节点
from nodes.consultants import (
    first_greeting_node,
    interviewer_node,
    consultant_node,
    high_value_node,
    low_budget_node,
    art_node,
    chit_chat_node,
    human_handoff_node
)


def route_entry(state: AgentState):
    """
    【并行入口】
    这里返回一个 list，LangGraph 会同时启动这些节点
    """
    messages = state.get("messages", [])
    if not messages:
        return "first_greeting"
    last_intent = state.get("last_intent")
    
    sticky_intents = ["ART_CONSULTING", "HIGH_VALUE", "LOW_BUDGET", "SALES_READY"]
    
    if last_intent in sticky_intents:
        print(f">>> 🔒 身份锁定 ({last_intent}) -> 跳过 Classifier")
        return ["extractor"]

    # 🔥 核心修正：返回列表，实现并行！
    return ["classifier", "extractor"]


def wait_node(state: AgentState):
    """
    【汇合节点】
    这个节点啥都不干，就是一个空的缓冲站。
    它的作用是等待 classifier 和 extractor 都跑完，
    然后在这里汇合，再统一触发 core_router。
    """
    return {}


# ==========================================
# 组装图
# ==========================================

workflow = StateGraph(AgentState)

# --- 注册节点 ---
workflow.add_node("classifier", classifier_node)
workflow.add_node("extractor", extractor_node)
workflow.add_node("wait_node", wait_node)  # 新增汇合点

workflow.add_node("first_greeting", first_greeting_node)
workflow.add_node("interviewer", interviewer_node)
workflow.add_node("consultant", consultant_node)
workflow.add_node("high_value", high_value_node)
workflow.add_node("low_budget", low_budget_node)
workflow.add_node("art_director", art_node)
workflow.add_node("chit_chat", chit_chat_node)
workflow.add_node("human_handoff", human_handoff_node)

# --- 设置入口 ---
workflow.set_conditional_entry_point(
    route_entry,
    {
        "first_greeting": "first_greeting",
        "classifier": "classifier",
        "extractor": "extractor"
    }
)

# --- 连接 First Greeting ---
workflow.add_edge("first_greeting", END)

# --- 🔥 核心：并行汇合 ---
# 无论谁先跑完，都去 wait_node 等着
workflow.add_edge("classifier", "wait_node")
workflow.add_edge("extractor", "wait_node")

# --- 决策分流 (从 wait_node 出发) ---
# 此时 state 里的 last_intent 和 profile 都已经更新完毕了
workflow.add_conditional_edges(
    "wait_node",
    core_router,
    {
        "human_handoff": "human_handoff",
        "chit_chat": "chit_chat",
        "high_value": "high_value",
        "art_director": "art_director",
        "interviewer": "interviewer",
        "low_budget": "low_budget",
        "consultant": "consultant"
    }
)

# --- 连接各业务节点的出口 ---

# Interviewer 问完问题 -> 结束 (等用户回话)
workflow.add_edge("interviewer", END)

# Consultant (自含 Sales 逻辑)
workflow.add_conditional_edges(
    "consultant",
    route_consultant,
    {
        "human_handoff": "human_handoff",
        END: END
    }
)

# High Value
workflow.add_conditional_edges(
    "high_value",
    route_high_value,
    {
        "human_handoff": "human_handoff",
        END: END
    }
)

# Low Budget
workflow.add_conditional_edges(
    "low_budget",
    route_low_budget,
    {
        "human_handoff": "human_handoff",
        END: END
    }
)

# Art Director
workflow.add_conditional_edges(
    "art_director",
    route_art_director,
    {
        "human_handoff": "human_handoff",
        END: END
    }
)

# Chit Chat & Handoff
workflow.add_edge("chit_chat", END)
workflow.add_edge("human_handoff", END)

# ==========================================
# 编译
# ==========================================
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
