# 感知层: classifier 和 extractor (并行运行)
import os
import sys
from typing import List

# 🛠️ 【防报错补丁】
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
# -------------------------------------------------

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
from state import AgentState, IntentResult, CustomerProfile, BudgetPeriod
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
deepseek_api_key = os.environ['DEEPSEEK_API_KEY']

llm = init_chat_model(
    "deepseek-chat",
    model_provider="deepseek",
    temperature=0,
    api_key=deepseek_api_key
)

llm_chat = init_chat_model(
    "deepseek-chat",
    model_provider="deepseek",
    temperature=0.7,
    api_key=deepseek_api_key
)


def classifier_node(state: AgentState):
    """
    【感知层 - 意图分类器】
    分析用户消息，判断意图类型。与 extractor 并行运行。
    """
    print("--- 🧠 Perception: 意图分类 ---")
    recent_msg = state["messages"][-12:]
    current_status = state.get("dialog_status")
    profile = state.get("profile")
    classifier = llm.with_structured_output(IntentResult)
    
    last_intent = state.get("last_intent")

    prompt = [
        SystemMessage(content=f"""
        你是留学顾问暴叔的"首席分诊台"。请根据用户的最新回复判断客户层级和真实意图
        当前客户画像摘要: {profile.model_dump_json(exclude_none=True)}

        【判定逻辑 - 优先级从高到低】

        1. 🆘 **TRANSFER_TO_HUMAN (转人工)**:
           - **逻辑**: 用户想要跳过ai咨询，直接进入真人对接环节或者想要咨询非常规业务
           - **用户提及留学灰产&保录**：
           - **负面情绪/障碍信号**: "我不懂", "太复杂了", "不知道", "直接找人跟我说", "别问了"，"我不想折腾".
           - **要求非文本沟通**: 电话，微信，语音沟通
           

        2. 🎨 **ART_CONSULTING (艺术生通道)**:
            - **关键词**: "作品集", "交互设计"，提及艺术类院校，"游戏设计", "服装设计", "纯艺", "插画", "动画", "电影","导演".
            - **逻辑**: 只要涉及需要**作品集(Portfolio)**的专业，无论预算多少，统统归为此类。因为艺术申请逻辑完全不同。

        3. 🔥 **HIGH_VALUE (高价值/VIP客户)**:
           - **关键词**:
             - 预算高: 预算充足，年预算50w+ 或总预算100w+
             - 背景强: "美本", "英本", "澳本", "加本", "海高", "美高", "A-level", "IB".
             - 目标高: "只想去滕校", "G5"
           - **逻辑**: 只要用户流露出"不缺钱"或者"出身海外院校"的信号，一律归为此类。
           

        4. 💰 **LOW_BUDGET (低预算客户)**:
           - **核心判定**：
             - 用户有任何形式的负债
             - 用户表示总预算或年预算 10w 以内
           - **逻辑**: 预算有限的客户，需要走专门的低预算通道，提供性价比方案
           
        5. ⚖️ **DECISION_SUPPORT (决策辅助/Offer对比/路径PK)**:
           - **场景**: 用户手里有offer需要选，或者在多个学校/路径/项目之间犹豫不决。
           - **关键词**: "offer", "ad", "vs", "对比", "哪个好", "怎么选", "纠结", "还是去".
           - **逻辑**: 用户不再需要常规规划，而是需要专家进行**利弊分析**和**拍板**。
           
        6. **SALES_READY
           - 用户表现出对方案/项目的质疑
           - 用户表现出对方案/项目的兴趣
           - 用户表现出对方案/项目的赞扬，肯定
           - **逻辑**： 此时是最好的销售切入点
           
        7. 📋 **NEED_CONSULTING (普通咨询)**:
           - 普通背景，预算正常或未提及，需要常规规划。

        8. 👋 **GREETING / CHIT_CHAT**:
           - 纯打招呼或闲或简单的语气词，且没有包含任何业务信息
        """)
    ] + recent_msg
    res = classifier.invoke(prompt)
    final_intent = res.intent
    
    # Python 守门员：负债关键词检测
    user_content = recent_msg[-1].content if recent_msg else ""
    debt_keywords = ["贷", "负债", "欠款", "欠债", "还债", "借钱", "还钱", "周转", "抵押", "上岸"]
    if any(k in user_content for k in debt_keywords):
        if final_intent != "TRANSFER_TO_HUMAN":  # 转人工优先级最高，不拦截
            print(f"--- ⚠️ (Python矫正) 检测到负债关键词，强制修正为 LOW_BUDGET ---")
            final_intent = "LOW_BUDGET"
    
    # 如果 classifier 没识别出低预算，但画像显示低预算，强制设为 LOW_BUDGET
    if profile and profile.budget.amount > 0 and profile.budget.amount < 10:
        if final_intent not in ["TRANSFER_TO_HUMAN", "ART_CONSULTING"]:  # 艺术生允许穷
            final_intent = "LOW_BUDGET"
    
    # 身份继承逻辑 (Sticky Intents)
    sticky_intents = ["ART_CONSULTING", "HIGH_VALUE", "LOW_BUDGET", "SALES_READY"]
    
    # 只有当 LLM 判定为"普通咨询"时，才去检查上一轮
    if final_intent == "NEED_CONSULTING":
        if last_intent in sticky_intents:
            print(f"--- 🔒 触发身份继承: {last_intent} (忽略本次普通判定) ---")
            final_intent = last_intent

    updates = {"last_intent": final_intent}

    if final_intent == "HIGH_VALUE":
        updates["dialog_status"] = "VIP_SERVICE"
    elif final_intent == "LOW_BUDGET":
        updates["dialog_status"] = "CONSULTING"
    elif final_intent == "NEED_CONSULTING":
        if current_status != "VIP_SERVICE":
            updates["dialog_status"] = "CONSULTING"

    return updates


def extractor_node(state: AgentState):
    """
    【感知层 - 信息提取器】
    从用户消息中提取结构化画像信息。
    与 classifier 并行运行。
    """
    print("--- 🕵️ Perception: 信息提取 ---")

    # 1. 读取旧档案
    current_profile = state.get("profile")
    if current_profile is None:
        from state import CustomerProfile
        current_profile = CustomerProfile()

    # 2. 获取上下文
    msgs = state["messages"]
    last_user_msg = msgs[-1].content
    last_ai_msg = msgs[-2].content if len(msgs) > 1 else ""

    # 3. 构造 Prompt
    system_prompt = f"""
    你是留学行业信息提取专家，你的任务是根据对话更新客户画像。

    【上下文信息】
    1. 上文 AI 问: "{last_ai_msg}"
    2. 当前 用户 答: "{last_user_msg}"

    【提取规则（优先级由高到低）】
    1. **显式提取 (Explicit)**:
       - 如果 User 直接陈述了数据（如"预算50万"、"我想去美国"），以用户原话为准，绝对优先。

    2. **隐式确认 (Implicit)**:
       - 如果 AI 上文给出了具体信息，User 表达了同意（"没问题"、"可以"），请直接提取 AI 提供的数值。

    3. **逻辑推理 (Inference)**:
       - **当且仅当**用户没有提供上述两条信息时，利用你的常识进行推理补全。
       - **推理场景**:
         - **预算**: 对于已经在海外读书的用户，根据国家地区合理推理预算（如英国高中，70w/年, 美国高中 90w/年，日本高中 20w/年 等等）
         - **目的地**: 根据用户当前所在地，就读院校所在地或意向国家进行合理推断。

    【特殊判断】
    - 区分现状与意向："想读本科"是意向(educationStage=None)，"我是大三"是现状(educationStage=本科)。
    - 判断身份：主语是"我"->学生；主语是"孩子/儿子"->家长；不包含主语-> None
    - **预算提取**：特别注意识别低预算表达，如"5万"、"8万"、"10万以内"、"预算不多"等，金额统一提取为数字（万）
    - **⚠️ 负债识别**：如果用户提到"负债"、"欠款"、"背着贷款"、"还债中"等，**不要提取为预算**！这是负面财务信号，不是留学预算。
    - **预算周期判断**：如果用户说"一年X万"->ANNUAL；"总共X万"->TOTAL；只说数字没说周期->UNKNOWN

    如果某字段既无显示信息，也无法逻辑推理，请返回None
    """
    # 4. 调用 LLM
    messages = [SystemMessage(content=system_prompt)]

    from state import CustomerProfile
    extractor = llm.with_structured_output(CustomerProfile)
    new_data = extractor.invoke(messages)

    # 5. Python 守门员逻辑 (The Gatekeeper)
    final_profile = current_profile.model_copy()

    # --- 逻辑 A：普通字段 (非空覆盖) ---
    if new_data.user_role is not None:
        final_profile.user_role = new_data.user_role
    if new_data.educationStage is not None:
        final_profile.educationStage = new_data.educationStage
    if new_data.destination_preference is not None:
        final_profile.destination_preference = new_data.destination_preference
    if new_data.budget.amount != -1:
        final_profile.budget.amount = new_data.budget.amount
    if new_data.budget.period != BudgetPeriod.UNKNOWN:
        final_profile.budget.period = new_data.budget.period

    # --- 逻辑 B：文本字段 (追加合并) ---
    if new_data.academic_background:
        if final_profile.academic_background:
            final_profile.academic_background += f"；{new_data.academic_background}"
        else:
            final_profile.academic_background = new_data.academic_background

    if new_data.language_level:
        if final_profile.language_level:
            final_profile.language_level += f"；{new_data.language_level}"
        else:
            final_profile.language_level = new_data.language_level

    print(f"最终画像: {final_profile.model_dump_json(exclude_none=True)}")

    return {"profile": final_profile}
