from langgraph.checkpoint.memory import MemorySaver
from state import AgentState, IntentType
from langgraph.graph import StateGraph
from nodes.consultants import first_greeting_node, high_value_node, art_node
from nodes.consultants import sales_node,interviewer_node,human_handoff_node, consultant_node
from nodes.tools import  extractor_node
from nodes.router import classifier_node, sales_detector_node
from langgraph.graph import StateGraph, END


def route_entry(state: AgentState):
    """
    
    【智能入口】
    判断是“刚加好友(空记录)”还是“用户已回复”。
    """
    messages = state.get("messages",[])
    if not messages:
        return "first_greeting"
    return "classifier"

def route_classifier(state: AgentState):
    """
    【初筛分流】
    Classifier 跑完后，根据意图分发。
    """
    intent = state.get("last_intent")
    status = state.get("dialog_status")
    
    # 1. 极其罕见的直接成交 -> Sales
    if intent == IntentType.TRANSFER_TO_HUMAN:
        return "human_handoff"
    
    # 2. 纯闲聊/打招呼 -> Interviewer (省去Extractor的Token，让Interviewer陪聊)
    if intent == IntentType.GREETING and status == "START":
        return "interviewer"
    
    if intent == "ART_CONSULTING":
        return "art_director"
        
    # 3. 其他所有情况 (VIP / 普通咨询 / 咨询中途) -> 统统去 Extractor
    # 为什么 VIP 也要去 Extractor？
    # 因为 VIP 刚才说的 "我预算800w" 需要被 Extractor 存进 Profile 里！
    # 存完之后，route_extractor 会负责把它导向 high_value_node
    return "extractor"

def route_extractor(state: AgentState):
    """
    【中枢路由】
    Extractor 更新完画像后，决定下一步去哪。
    这里是 VIP 轨道和普通轨道的分岔口。
    """
    profile = state["profile"]
    status = state.get("dialog_status")
    has_proposed = state.get("has_proposed_solution", False)
    
    # ==========================
    # 🚀 VIP 快速通道
    # ==========================
    if status == "VIP_SERVICE":
        # 如果已经给过 VIP 方案 -> 去博弈检测
        return "high_value"
    # ==========================
    # 🐢 普通咨询通道
    # ==========================
    
    # 1. 画像还不全 -> Interviewer 追问
    # (注意：我们在 interviewer 内部加了更严格的门禁，这里主要靠 Pydantic 判断)
    if not profile.is_complete:
        return "interviewer"
    
    # 2. 画像齐了，还没给方案 -> Consultant 专家诊断
    if not has_proposed:
        return "consultant"
    
    # 3. 方案给过了 -> Sales Detector 博弈分析
    return "sales_detector"

def route_sales_detector(state: AgentState):
    """
    【博弈路由】
    Detector 分析完用户对方案的反应后...
    """
    intent = state.get("last_intent")
    status = state.get("dialog_status")
    
    if intent == IntentType.TRANSFER_TO_HUMAN:
        return "human_handoff"
    # 1. 上钩了 -> 收网
    if intent == IntentType.SALES_READY:
        return "sales"
        
    # 2. 还有疑虑 / 没反应过来 -> 回去继续聊
    else:
        # 如果是 VIP 客户 -> 回 VIP 接待室继续哄
        if status == "VIP_SERVICE":
            return "high_value"
        # 如果是普通客户 -> 回 Consultant 继续解释
        else:
            return "consultant"
        
def route_high_value(state: AgentState):
    """
    【VIP 专属路由】
    检查 High Value Node 是否触发了摇人工具。
    """
    messages = state["messages"]
    last_msg = messages[-1]
    
    # 检查是否有工具调用请求
    if hasattr(last_msg, "tool_calls") and len(last_msg.tool_calls) > 0:
        print(">>> 🔘 检测到 VIP 摇人信号，转入人工对接...")
        return "human_handoff"
        
    # 没有摇人 -> 结束当前轮次，等待用户回复 (继续在 High Value 节点循环)
    return END


# ==========================================
# 2. 组装图 (The Architecture)
# ==========================================

# ==========================================
# 2. 组装图 (The Architecture) - 最终修正版
# ==========================================

workflow = StateGraph(AgentState)

# --- 添加所有节点 ---
workflow.add_node("first_greeting", first_greeting_node)
workflow.add_node("classifier", classifier_node)
workflow.add_node("extractor", extractor_node)
workflow.add_node("interviewer", interviewer_node)
workflow.add_node("consultant", consultant_node)
workflow.add_node("high_value", high_value_node)
workflow.add_node("sales_detector", sales_detector_node)
workflow.add_node("sales", sales_node)
workflow.add_node("human_handoff", human_handoff_node)
workflow.add_node("art_director", art_node)
# --- 设置入口 (核心差异点) ---
workflow.set_conditional_entry_point(
    route_entry, # 使用新的智能入口
    {
        "first_greeting": "first_greeting", # 没说话 -> AI先说
        "classifier": "classifier"          # 说了话 -> 雷达检测
    }
)

# --- 连接 First Greeting ---
# AI 说完“欢迎”后，就停下来(END)，等待用户打字
workflow.add_edge("first_greeting", END)

# --- 连接 Classifier (分流) ---
workflow.add_conditional_edges(
    "classifier",
    route_classifier,
    {
        "human_handoff":"human_handoff",
        "sales": "sales",
        "interviewer": "interviewer", # 仅限打招呼
        "extractor": "extractor",
        "art_director": "art_director"# 其他所有情况先去提取数据
    }
)

# --- 连接 Extractor (中枢) ---
workflow.add_conditional_edges(
    "extractor",
    route_extractor,
    {
        "high_value": "high_value",        # VIP 专用通道
        "interviewer": "interviewer",      # 补全信息
        "consultant": "consultant",        # 给方案
        "sales_detector": "sales_detector" # 方案后的博弈
    }
)

# --- 连接 Sales Detector (博弈) ---
workflow.add_conditional_edges(
    "sales_detector",
    route_sales_detector,
    {
        "human_handoff": "human_handoff",
        "sales": "sales",             # 收网
        "high_value": "high_value",   # VIP 回炉继续聊
        "consultant": "consultant"    # 普通 回炉继续聊
    }
)
workflow.add_conditional_edges(
    "high_value",
    route_high_value, # 使用上面写的新路由
    {
        "human_handoff": "human_handoff", # 摇人了 -> 去 Handoff 节点收尾
        END: END                          # 没摇人 -> 等用户回话
    }
)
workflow.add_conditional_edges(
    "art_director",
    route_high_value, # 复用 High Value 的路由逻辑 (有Tool去Handoff，没Tool去End)
    {
        "human_handoff": "human_handoff",
        END: END
    }
)
# --- 设置终点 ---
# 所有负责"说话"的节点，说完后都挂起(END)，等待用户输入
workflow.add_edge("interviewer", END)
workflow.add_edge("consultant", END)
workflow.add_edge("high_value", END)
workflow.add_edge("art_director", END)# 🔥 确保 VIP 节点也有出口
workflow.add_edge("sales", END)
workflow.add_edge("human_handoff", END)

# ==========================================
# 3. 编译
# ==========================================
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)