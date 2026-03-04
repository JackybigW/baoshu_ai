# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: agent
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 暴叔AI设计 - 三层架构版本
# # 感知层 -> 决策层 -> 执行层

# %%
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Annotated, Literal, Optional
from langchain_core.runnables.graph import MermaidDrawMethod
import nest_asyncio
nest_asyncio.apply()
from langgraph.graph import StateGraph
from pydantic import BaseModel,Field, model_validator, field_validator
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage, ToolMessage
import requests
import os
from dotenv import load_dotenv, find_dotenv
from enum import Enum
from langgraph.graph.message import add_messages
import difflib
import pandas as pd
import re
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from IPython.display import Image, display

_ = load_dotenv(find_dotenv())
deepseek_api_key = os.environ['DEEPSEEK_API_KEY']

# 初始化模型
llm = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
    api_key=deepseek_api_key
)

# 顾问模型（温度稍高，更像人）
llm_chat = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
    temperature=0.7,
    api_key=deepseek_api_key
)

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## State设计

# %%
class BudgetPeriod(str, Enum):
    ANNUAL = "YEAR"       # 每年
    TOTAL = "TOTAL"       # 总共
    UNKNOWN = "UNKNOWN"   # 未说明

class BudgetInfo(BaseModel):
    amount: int = Field(-1, description="预算金额数字(万)，未知填-1")
    period: BudgetPeriod = Field(
        default=BudgetPeriod.UNKNOWN,
        description="预算周期。如果用户只说了数字没说周期，必须选 UNKNOWN"
    )

class CustomerProfile(BaseModel):
    """客户画像数据结构"""
    user_role: Optional[Literal["学生", "家长"]] = Field(
        default=None,
        description="""
        判断用户是学生本人还是家长。规则：
        - 出现第一人称("我读高三", "我想去", "我均分") -> '学生'
        - 出现第三人称("孩子高三", "我女儿", "我儿子") -> '家长'
        - 都没出现 -> 'None'
        """
    )
    educationStage: Optional[Literal["小学","初中","职高","中专", "高中","本科","大专","研究生"]] = Field(
        default=None,
        description="用户【当前】正在读或已毕业的最高学历。注意：不要填用户【想去读】的学历。"
    )
    budget: BudgetInfo = Field(default_factory=BudgetInfo)
    destination_preference: Optional[Literal["境外方向","境内/港澳方向"]] = Field(
        default=None,
        description="用户的留学或工作的目的地偏好"
    )
    academic_background: Optional[str] = Field(
        default=None,
        description="用户的学术背景详情"
    )
    language_level: Optional[str] = Field(
        default=None,
        description="用户可用于留学的外语能力"
    )

    @field_validator('educationStage', mode='before')
    @classmethod
    def robust_stage(cls, v):
        # 1. 拦截空值 (Python None)
        if v is None:
            return None
        
        # 2. 拦截字符串类型的空值 (String None/Null/N/A)
        s = str(v).strip().replace("'", "").replace('"', "") # 去掉可能的引号
        if s.lower() in ['none', 'null', 'n/a', 'unknown', '', '无']:
            return None

        # 3. 模糊匹配 (Fuzzy Matching) - 加上更严谨的判断
        # 比如防止把 "日本" 识别成 "本科" (因为都有'本')
        
        if "研" in s or "硕" in s or "博" in s or "Master" in s or "PhD" in s:
            return "研究生"
        
        # 必须是“本”且不能是“日本”/“书本”之类，通常 context 是学历，相对安全
        if "本" in s or "Bachelor" in s: 
            return "本科"
            
        if "大专" in s or ("专" in s and "中" not in s): # 防止匹配中专
            return "大专"
            
        if "中专" in s or "职" in s or "技校" in s: 
            return "中专"
            
        if "高" in s: # 高中, 高三
            return "高中"
            
        if "初" in s: 
            return "初中"
            
        if "小" in s: 
            return "小学"

        # 4. 🔥 兜底策略：如果上面都没匹配上，且不在允许列表中，直接丢弃！
        # 防止 LLM 输出了 "幼儿园" 或者 "博士后"，导致 Literal 校验失败报错
        allowed = ["小学","初中","职高","中专", "高中","本科","大专","研究生"]
        if s in allowed:
            return s
            
        # 既然洗不出来，为了不报错崩服务，强制返回 None
        return None

    # 同理优化 user_role
    @field_validator('user_role', mode='before')
    @classmethod
    def robust_role(cls, v):
        if v is None: return None
        s = str(v).strip()
        if s.lower() in ['none', 'null', 'unknown', '']: return None
        
        if "家长" in s or "Parent" in s or "父母" in s or "妈" in s or "爸" in s: return "家长"
        if "学生" in s or "Student" in s or "本人" in s or "我" in s: return "学生"
        
        return None # 无法识别就置空，别报错

    # ... (其他 validator)

    @field_validator('destination_preference', mode='before')
    @classmethod
    def robust_destination(cls, v):
        if not v: return None
        s = str(v).upper()
        if any(k in s for k in ["港", "澳", "内地", "国内", "CN", "HK", "MO", "MACAU"]):
            return "境内/港澳方向"
        if any(k in s for k in ["外", "美", "英", "澳", "加", "日", "韩", "欧", "OVERSEAS"]):
            return "境外方向"
        return None

    @property
    def missing_fields(self) -> List[str]:
        missing = []
        if not self.educationStage: missing.append("当前学历")
        elif not self.academic_background: missing.append("学术背景")
        elif self.budget.amount == -1: missing.append("留学预算(数字)")
        elif self.budget.period == BudgetPeriod.UNKNOWN: missing.append("预算周期(每年/总共)")
        elif not self.destination_preference: missing.append("目的地偏好")
        return missing

    @property
    def is_complete(self) -> bool:
        return len(self.missing_fields) == 0


def merge_text_fields(old_text: Optional[str], new_text: Optional[str]) -> Optional[str]:
    """
    【文本合并小助手】
    功能：去重、去空、保持顺序
    解决：避免 "雅思6.0；雅思6.0" 这种复读机现象
    """
    # 1. 边界处理：如果一边是空的，直接返回另一边
    if not new_text:
        return old_text
    if not old_text:
        return new_text

    # 2. 拆解：用正则表达式支持 中文分号(；) 和 英文分号(;) 切割
    # 比如 "护理专业；雅思5.0" -> ["护理专业", "雅思5.0"]
    old_segments = [s.strip() for s in re.split(r'[;；]', old_text) if s.strip()]
    new_segments = [s.strip() for s in re.split(r'[;；]', new_text) if s.strip()]

    # 3. 合并与去重
    result_list = list(old_segments) # 先把旧的全拿过来
    
    for new_seg in new_segments:
        # --- 智能去重逻辑 ---
        
        # 1. 完全一致则跳过 (Exact Match)
        if new_seg in result_list:
            continue
            
        # 2. 包含检测 (Subset Check) - 可选，防止"雅思6"和"雅思6.0"共存
        # 如果新来的这句，已经被旧的某句话包含了，那也算重复
        # 例如：旧="全日制大专护理"，新="大专护理" -> 跳过
        is_redundant = False
        for old_seg in result_list:
            if new_seg in old_seg: # 如果新的是旧的一部分，默认旧的更详细，跳过新的
                is_redundant = True
                break
            # 反之，如果旧的是新的一部分（新信息更全），通常我们追加上去，或者替换
            # 这里为了安全起见，我们选择追加，交给人类去看
            
        if not is_redundant:
            result_list.append(new_seg)

    # 4. 组装
    return "；".join(result_list)


def reduce_profile(old_data: Optional[CustomerProfile], new_data: Optional[CustomerProfile]) -> CustomerProfile:
    """
    【工业级合并策略 V2.0】
    """
    if new_data is None: return old_data
    if old_data is None: return new_data

    merged = old_data.model_copy()
    
    # --- 1. 单值字段：有值就覆盖 (Trust the latest non-null) ---
    if new_data.user_role is not None: 
        merged.user_role = new_data.user_role
        
    if new_data.educationStage is not None: 
        merged.educationStage = new_data.educationStage
        
    if new_data.destination_preference is not None: 
        merged.destination_preference = new_data.destination_preference
    
    # --- 2. 结构化字段：Budget 特殊处理 ---
    if new_data.budget.amount != -1: 
        merged.budget.amount = new_data.budget.amount
        
    # 只有当新周期不是 UNKNOWN 时才覆盖。
    # 如果旧的是 "YEAR"，新的是 "UNKNOWN"，保持 "YEAR"。
    if new_data.budget.period != BudgetPeriod.UNKNOWN: 
        merged.budget.period = new_data.budget.period

    # --- 3. 文本字段：智能合并 (不再无脑 +=) ---
    merged.academic_background = merge_text_fields(old_data.academic_background, new_data.academic_background)
    merged.language_level = merge_text_fields(old_data.language_level, new_data.language_level)

    return merged

class IntentType(str, Enum):
    SALES_READY = "SALES_READY"
    TRANSFER_TO_HUMAN = "TRANSFER_TO_HUMAN"
    ART_CONSULTING = "ART_CONSULTING"
    HIGH_VALUE = "HIGH_VALUE"
    LOW_BUDGET = "LOW_BUDGET"
    NEED_CONSULTING = "NEED_CONSULTING"
    GREETING = "GREETING"
    CHIT_CHAT = "CHIT_CHAT"
    DECISION_SUPPORT = "DECISION_SUPPORT"

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    profile: Annotated[CustomerProfile, reduce_profile]
    has_proposed_solution: bool
    dialog_status: Literal["START", "PROFILING", "PERSUADING", "FINISHED"]
    last_intent: Optional[IntentType]

class IntentResult(BaseModel):
    intent: IntentType

    @field_validator('intent', mode='before')
    @classmethod
    def robust_intent_validator(cls, v):
        if not isinstance(v, str): return "NEED_CONSULTING"
        clean_v = v.upper().strip()
        
        # 1. 精确匹配
        valid_keys = [member.value for member in IntentType]
        if clean_v in valid_keys: return clean_v
        
        # 2. 模糊匹配 (防止拼写错误)
        matches = difflib.get_close_matches(clean_v, valid_keys, n=1, cutoff=0.8)
        if matches: return matches[0]
        
        # 3. 关键词兜底 (这是核心修改)
        if "TRANSFER" in clean_v or "HUMAN" in clean_v: return "TRANSFER_TO_HUMAN"
        if "SALES" in clean_v: return "SALES_READY"
        if "HIGH" in clean_v: return "HIGH_VALUE"
        if "ART" in clean_v: return "ART_CONSULTING"
        if "LOW" in clean_v or "BUDGET" in clean_v: return "LOW_BUDGET"
        
        # 🔥 新增：DECISION_SUPPORT 的模糊捕获
        # 防止 LLM 输出 "DECISION", "OFFER_PK", "COMPARE" 等变体
        if "DECISION" in clean_v or "COMPARE" in clean_v or "OFFER" in clean_v:
            return "DECISION_SUPPORT"

        return "NEED_CONSULTING"

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Tools 设计

# %%
# Excel 数据加载
try:
    # 修正路径：文件现在在 data 子目录下
    excel_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "products_intro.xlsx")
    df_db = pd.read_excel(excel_path)
    num_cols = df_db.select_dtypes(include=['number']).columns
    df_db[num_cols] = df_db[num_cols].fillna(0)
    obj_cols = df_db.select_dtypes(include=['object', 'string']).columns
    df_db[obj_cols] = df_db[obj_cols].fillna("")
except Exception as e:
    print(f"⚠️ 警告: 读取 Excel 失败: {e}")
    df_db = pd.DataFrame(columns=[
        "educationStage", "是否出国", "budgetLowerBound", "budgetUpperBound",
        "annualBudgetLowerBound", "annualBudgetUpperBound",
        "项目", "项目说明", "项目优势", "画像", "学制"
    ])

def search_products(profile: CustomerProfile) -> str:
    """检索方案"""
    try:
        mask = pd.Series([True] * len(df_db))
        if profile.educationStage:
            mask &= df_db['educationStage'].str.contains(profile.educationStage, na=False)
        user_pref = profile.destination_preference
        if user_pref == "境外方向":
            mask &= df_db['是否出国'].isin(["可以出国", "缓缓再出国"])
        elif user_pref == "境内/港澳方向":
            mask &= (df_db['是否出国'] == "不要出国")
        amount = profile.budget.amount
        period = profile.budget.period
        if amount > 0:
            if period == BudgetPeriod.TOTAL:
                mask &= (df_db['budgetLowerBound'] <= amount)
            elif period == BudgetPeriod.ANNUAL:
                mask &= (df_db['annualBudgetLowerBound'] <= amount)
        results = df_db[mask]
        if len(results) == 0:
            return "【系统提示】：数据库中未找到符合该学历和预算的方案。"
        output_text = f"共匹配到 {len(results)} 个方案：\n"
        for idx, row in results.iterrows():
            proj_name = row.get('项目', '未知项目')
            proj_desc = row.get('项目说明', '')
            proj_adv = row.get('项目优势', '')
            proj_type = row.get('是否出国', '')
            total_price = f"{row.get('budgetLowerBound',0)}-{row.get('budgetUpperBound','?')}万"
            annual_price = f"{row.get('annualBudgetLowerBound',0)}-{row.get('annualBudgetUpperBound','?')}万"
            output_text += f"""
            --- 方案 {idx+1} ---
            [项目名称]: {proj_name}
            [类型]: {proj_type}
            [预算范围]: 总预算约 {total_price} / 年预算约 {annual_price}
            [项目说明]: {proj_desc}
            [核心优势]: {proj_adv}
            """
        return output_text
    except Exception as e:
        return f"【系统错误】：检索过程发生异常: {str(e)}"

@tool
def summon_specialist_tool():
    """当客户表现出意向，或者需要资深顾问/具体方案介入时，调用此工具。"""
    return "Specialist summoned."

# %% [markdown]
# ## 感知层 (Perception Layer)

# %%
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

    # 二次保险：检查用户消息中是否有负债/欠款关键词
    # 防止 deepseek 将"负债80w"错误识别为高预算

    prompt = [
        SystemMessage(content="""
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
    user_content = recent_msg[-1].content if recent_msg else ""
    debt_keywords = ["贷", "负债", "欠款", "欠债", "还债", "借钱", "还钱", "周转", "抵押", "上岸"]
    if any(k in user_content for k in debt_keywords):
        if final_intent != "TRANSFER_TO_HUMAN": # 转人工优先级最高，不拦截
            print(f"--- ⚠️ (Python矫正) 检测到负债关键词，强制修正为 LOW_BUDGET ---")
            final_intent = "LOW_BUDGET"
    
    # 如果 classifier 没识别出低预算，但画像显示低预算，强制设为 LOW_BUDGET
    if profile and profile.budget.amount > 0 and profile.budget.amount < 10:
        if final_intent not in ["TRANSFER_TO_HUMAN", "ART_CONSULTING"]: # 艺术生允许穷
            final_intent = "LOW_BUDGET"
    
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
    从用户消息中提取结构化画像信息。与 classifier 并行运行。
    """
    print("--- 🕵️ Perception: 信息提取 ---")

    # 1. 读取旧档案
    current_profile = state.get("profile")
    if current_profile is None:
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

    messages = [SystemMessage(content=system_prompt)]
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

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## 决策层 (Decision Layer)

# %%
# router.py -> core_router

def core_router(state: AgentState):
    print("--- 🎯 Decision: 核心路由 ---")

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
        
    if intent == IntentType.LOW_BUDGET: # 🔥 直接看意图，干净清爽
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
    """【决策层 - VIP 路由】检查 High Value Node 是否触发了摇人工具。"""
    messages = state["messages"]
    last_msg = messages[-1]
    if hasattr(last_msg, "tool_calls") and len(last_msg.tool_calls) > 0:
        print(">>> 🔘 检测到 VIP 摇人信号，转入人工对接...")
        return "human_handoff"
    return END

def route_low_budget(state: AgentState):
    """【决策层 - 低预算路由】检查 Low Budget Node 是否触发了摇人工具。"""
    messages = state["messages"]
    last_msg = messages[-1]
    if hasattr(last_msg, "tool_calls") and len(last_msg.tool_calls) > 0:
        print(">>> 🔘 检测到 Low Budget 摇人信号，转入人工对接...")
        return "human_handoff"
    return END

def route_art_director(state: AgentState):
    """【决策层 - 艺术留学路由】检查 Art Node 是否触发了摇人工具。"""
    messages = state["messages"]
    last_msg = messages[-1]
    if hasattr(last_msg, "tool_calls") and len(last_msg.tool_calls) > 0:
        print(">>> 🔘 检测到 Art 摇人信号，转入人工对接...")
        return "human_handoff"
    return END

def route_consultant(state: AgentState):
    """【决策层 - Consultant 路由】检查 Consultant Node 是否触发了摇人工具。"""
    messages = state["messages"]
    last_msg = messages[-1]
    if hasattr(last_msg, "tool_calls") and len(last_msg.tool_calls) > 0:
        print(">>> 🔘 检测到 Consultant 摇人信号，转入人工对接...")
        return "human_handoff"
    return END

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## 执行层 (Execution Layer)

# %%
def first_greeting_node(state: AgentState):
    """用户加了好友，优先say hi，勾引用户说话"""
    print("--- 👋 Greeting: AI 主动破冰 ---")
    return {
        "messages": [AIMessage(content="您好，欢迎咨询！ 跟上暴叔的节奏～")],
        "dialog_status": "START",
        "last_intent": "GREETING"
    }

def high_value_node(state: AgentState):
    """VIP会员通道（目的是快速勾引客户，快速转人工）"""
    print("--- 🎩 High Value: 握有实权的合伙人 ---")

    profile = state.get("profile") or CustomerProfile()
    messages = state["messages"]

    tools = [summon_specialist_tool]
    llm_with_tools = llm_chat.bind_tools(tools)
    closing_pressure = ""
    if len(messages) >= 16:
        closing_pressure = "\n【⚠️系统警告】对话过长，请尽快寻找理由拉群转人工！"

    system_prompt = f"""
    你是暴叔，千万级留学网红。面对的是【VIP高净值客户】。

    【客户画像】
    {profile.model_dump_json(exclude_none=True)}
    【核心任务】
    像一个真人老手一样与客户沟通，解答疑问，建立信任。
    目的是展现专业度，最终把客户转交给真人留学顾问

    【你的特权】
    你手边有一个tool `summon_specialist_tool` (呼叫专家)。

    【决策逻辑】
    1. **继续聊 (Don't use tool)**:
       - 如果还需要继续获取信息、建立信任、或者解答疑问。
       - 此时就像真人一样正常聊天。

    2. **摇人 (MUST USE TOOL)**:
       - 当你觉得时机成熟：用户意向高、同意方案。
       - 当你遇到困难：用户始终不满意你提供的方案或者你没有方案提供给用户
       - 当用户提出语音，电话等必须需要人类干预的事项。
       - 当用户负面情绪严重，不想沟通。
       - 当用户出现求助信号："有办法帮我吗" /"能不能搞定"/"太难了"
       - 当用户提出灰产要求："能不能保录"、"操作成绩"、"买文凭"、"走后门"、"特殊渠道"。
       - 对话过长: {closing_pressure}

    【暴叔的聊天规范】
    1. **拒绝废话**：直奔主题，简单回应用户上一句，并简练抛出新问题，每段话不许超过40字
    2. **接话艺术**:如果用户上一句是在问问题，你必须用自己的知识库作答给出结果，再自然衔接问题
    3. **制造悬念**：说话要切中要害，对数据敏感
    4. **自然分段**： 在【回应客户】和【抛出新问题】之间 使用"|||" 分隔
    5. **像真人一样说话**: 禁止使用**加粗字体**
    6. **禁止复读(anti-loop):
       - 如果用户说"不懂"、"不知道"或未回答你的问题：**不要重复同样的问题！**
       - 策略: 换一种更通俗的问法，或者给一个大概的选项让用户选。
    """

    response = llm_with_tools.invoke([SystemMessage(content=system_prompt)] + messages)

    if response.tool_calls:
        print(f"🔧 Tool Triggered: {response.tool_calls}")
        return {"messages": [response], "dialog_status": "VIP_SERVICE"}
    else:
        raw_content = response.content.replace("\n\n", "|||").replace("\n", "|||").replace("**", "")
        split_texts = raw_content.split("|||")
        ai_messages = [AIMessage(content=text.strip()) for text in split_texts if text.strip()]
        return {"messages": ai_messages, "dialog_status": "VIP_SERVICE"}

def art_node(state: AgentState):
    """艺术留学顾问（艺术留学和普通留学不兼容，需分开）"""
    print("--- 🎨 Art Director: 艺术总监 ---")

    profile = state.get("profile") or CustomerProfile()
    messages = state["messages"]

    tools = [summon_specialist_tool]
    llm_with_tools = llm_chat.bind_tools(tools)
    closing_pressure = ""
    if len(messages) >= 16:
        closing_pressure = "\n【⚠️系统警告】对话过长，客户可能流失。请立刻寻找理由拉群，停止追问细节！"

    system_prompt = f"""
    你是暴叔旗下的艺术留学总监，主攻海外名校的艺术留学申请。

    【客户画像】
    {profile.model_dump_json(exclude_none=True)}
    【核心任务】
    像一个真人老手一样与客户沟通，解答疑问，建立信任。
    用你的专业度，建立用户的作品集辅导需求
    【你的世界观】
    1. 作品集和语言>一切：只要作品集足够优秀，出身差，GPA低都不是问题，但语言一定要过关
    2. 风格差异: 大方指出不同学校招生官的审美偏好，用于吸引学生
    3. 不仅是留学顾问，更懂留学地区的生活


    【你的特权】
    你手边有一个tool `summon_specialist_tool` (呼叫专家)。

    【决策逻辑】
    **请根据对话情境，自主生成话术，禁止使用僵硬的固定句式！**

    1. **摇人 (MUST USE TOOL)**:
       - 当你遇到困难：用户始终不满意你提供的方案或者你没有方案提供给用户
       - 当用户提出语音，电话等必须需要人类干预的事项
       - 当用户负面情绪严重，不想沟通
       - 当用户出现求助信号："有办法帮我吗" /"能不能搞定"/"太难了"
       - 当用户提出灰产要求："作品集能直接买吗"，"能不能保录"
       - 当用户已经有作品/想法时： 先肯定用户，然后主动表示安排负责作品集的老师进行免费评估
       - 成交信号： 用户同意或对当前方案感兴趣
       - 对话过长: {closing_pressure}

    2. **继续聊 (Don't use tool)**:
       - 如果还需要继续获取信息、建立信任、或者解答疑问。
       - 此时就像真人一样正常聊天，用专业留学或当地生活知识解答用户疑虑

    【暴叔的聊天规范】
    1. **拒绝废话**：直奔主题，简单回应用户上一句，并简练抛出新问题，每段话不许超过40字
    2. **接话艺术**:如果用户上一句是在问问题，你必须用自己的知识库简单作答给出结果，再自然衔接问题
    3. **制造悬念**
    4. **自然分段**： 在【回应客户】和【抛出新问题】之间 使用"|||" 分隔
    5. **像真人一样说话**: 禁止使用**加粗字体**
    6. **禁止复读(anti-loop):
       - 如果用户说"不懂"、"不知道"或未回答你的问题：**不要重复同样的问题！**
       - 策略: 换一种更通俗的问法，或者给一个大概的选项让用户选。
    """

    response = llm_with_tools.invoke([SystemMessage(content=system_prompt)] + messages)

    if response.tool_calls:
        print(f"🔧 Tool Triggered: {response.tool_calls}")
        return {"messages": [response], "dialog_status": "VIP_SERVICE"}
    else:
        raw_content = response.content.replace("\n\n", "|||").replace("\n", "|||").replace("**", "")
        split_texts = raw_content.split("|||")
        ai_messages = [AIMessage(content=text.strip()) for text in split_texts if text.strip()]
        return {"messages": ai_messages, "dialog_status": "VIP_SERVICE"}

def low_budget_node(state: AgentState):
    """低预算客户专属节点 - 提供性价比方案，快速转人工"""
    print("--- 💰 Low Budget: 低预算客户通道 ---")

    profile = state.get("profile") or CustomerProfile()
    messages = state["messages"]

    tools = [summon_specialist_tool]
    llm_with_tools = llm_chat.bind_tools(tools)

    closing_pressure = ""
    if len(messages) >= 12:
        closing_pressure = "\n【⚠️系统警告】对话过长，请尽快寻找理由拉群转人工！"

    system_prompt = f"""
    你是暴叔，千万级留学网红。面对的是【预算有限的客户】（年预算或总预算低于10万）。

    【客户画像】
    {profile.model_dump_json(exclude_none=True)}

    【核心任务】
    1. 快速了解客户基本情况（学历、最低预算）
    2. 提供2-3个高性价比的留学/提升方案（如：东南亚、东欧、中外合办、专升本等）
    3. 展现专业度的同时，快速建立信任，引导转人工

    【低预算方案库（参考）】
    - 东南亚：马来西亚、泰国（年预算5-8万）
    - 东欧：俄罗斯、白俄罗斯、波兰（年预算6-10万）
    - 中外合办：4+0项目（国内读，拿国外学位）
    - 专升本/专升硕：国内专升本后再出国，或直接申请海外专升本
    - 打工度假：澳洲WHV、新西兰WHV（边打工边体验）
    - 在线学位：海外在线本科/硕士（费用低，可转线下）
    - 日本新闻生

    【你的特权】
    你手边有一个tool `summon_specialist_tool` (呼叫专家)。

    【决策逻辑】
    1. **继续聊 (Don't use tool)**:
       - 如果还需要继续了解客户背景、介绍方案细节。
       - 此时就像真人一样正常聊天，用大白话解释方案优劣。

    2. **摇人 (MUST USE TOOL)**:
       - 当你觉得时机成熟：用户对方案感兴趣、问具体申请流程。
       - 当你遇到困难：用户觉得预算还是太高、没有合适方案。
       - 当用户提出语音，电话等必须需要人类干预的事项。
       - 当用户负面情绪严重，不想沟通。
       - 当用户出现求助信号："有办法帮我吗"/"能不能搞定"/"太难了"。
       - 当用户直接问："怎么报名"、"怎么申请"。
       - 对话过长: {closing_pressure}

    【暴叔的聊天规范】
    1. **拒绝废话**：直奔主题，简单回应用户上一句，并简练抛出新问题，每段话不许超过40字
    2. **接话艺术**:如果用户上一句是在问问题，你必须用自己的知识库作答给出结果，再自然衔接问题
    3. **制造悬念**：说话要切中要害，强调"虽然预算有限，但路不止一条"
    4. **自然分段**：在【回应客户】和【抛出新问题】之间使用"|||"分隔
    5. **像真人一样说话**: 禁止使用**加粗字体**
    6. **禁止复读(anti-loop):
       - 如果用户说"不懂"、"不知道"或未回答你的问题：**不要重复同样的问题！**
       - 策略: 换一种更通俗的问法，或者给一个大概的选项让用户选。
    7. **不贬低客户**：不要说"预算太低"，而要说"咱们把钱花在刀刃上"
    """

    response = llm_with_tools.invoke([SystemMessage(content=system_prompt)] + messages)

    if response.tool_calls:
        print(f"🔧 Tool Triggered: {response.tool_calls}")
        return {"messages": [response], "dialog_status": "PERSUADING"}
    else:
        raw_content = response.content.replace("\n\n", "|||").replace("\n", "|||").replace("**", "")
        split_texts = raw_content.split("|||")
        ai_messages = [AIMessage(content=text.strip()) for text in split_texts if text.strip()]
        return {"messages": ai_messages, "dialog_status": "PERSUADING"}

def interviewer_node(state: AgentState):
    """常规咨询的顾问，主要工作是解答问题，并采访用户获取背景信息"""
    print("--- 🎤 Interviewer: 信息采编 ---")

    profile = state.get("profile") or CustomerProfile()
    missing = profile.missing_fields

    user_role = profile.user_role
    stage = profile.educationStage
    if not missing:
        return {"messages": [AIMessage(content="情况都清楚了，咱们直接看方案！")]}

    target_field = profile.missing_fields[0] if profile.missing_fields else None

    if profile.educationStage and profile.academic_background:
        if not re.search(r'\d|[ABC][\+\-]?|Distinction|Merit|Pass|预估', profile.academic_background, re.IGNORECASE):
            target_field = "academic_background"

    if user_role == "家长":
        role_instruction = "【对话策略】：对方是家长。语气要稳重、让其放心。关注点在于安全、就业，孩子前途。"
    elif user_role == "学生":
        role_instruction = "【对话策略】：对方是学生。语气要像老大哥一样懂行、给鼓励。关注点在于学校排名。"
    else:
        role_instruction = "【对话策略】：身份未知。默认当做学生聊，保持中性亲切。"

    focus_instruction = ""

    if target_field == "academic_background":
        lang_status = "（并顺便问一下语言准备情况）" if not profile.language_level else ""
        focus_instruction =f"""
        用户学术背景缺失，作为专业留学顾问，请参考用户的具体背景向用户提问{lang_status}，
        这些信息将用于方案规划师生成留学方案
        """
    elif target_field == "budget":
        focus_instruction = "【关注点】：家庭支持的留学预算范围（确认是每年还是总预算）。"
    elif target_field == "destination_preference":
        focus_instruction = "【关注点】：目的地偏好。是倾向去境外（英美澳加日韩）闯一闯，还是**境内/港澳**求稳？"

    greeting_instruction = ""
    if user_role:
        target_greeting = f"{'同学' if user_role == '学生' else '家长'}您好！"
        has_greeted = False
        for msg in state["messages"]:
            if isinstance(msg, AIMessage) and target_greeting in msg.content:
                has_greeted = True; break
        if not has_greeted:
            greeting_instruction = f"**回复必须以 '{target_greeting}' 开头**。"

    system_prompt = f"""
    你就是留学顾问"暴叔"。

    【已知画像】: {profile.model_dump_json(exclude_none=True)}
    【当前任务】: 追问缺失项 -> **{target_field}**

    {greeting_instruction}

    {role_instruction}
    {focus_instruction}


    【暴叔的聊天规范】
    1. **拒绝废话**：直奔主题，简单回应用户上一句，并简练抛出新问题，每段话不许超过40字
    2. **懂行**
    3. **接话艺术**:如果用户上一句是在问问题，你必须用自己的知识库简单作答给出结果，再自然衔接问题
    4. **自然分段**： 在【回应客户】和【抛出新问题】之间 使用"|||" 分隔
    5. **像真人一样说话**: 禁止使用**加粗字体**
    6. **禁止复读(anti-loop):
       - 如果用户说"不懂"、"不知道"或未回答你的问题：**不要重复同样的问题！**
       - 策略: 换一种更通俗的问法，或者给一个大概的选项让用户选。
    7. **高情商表达**: 对于背景差目标高的客户，先承认难度大，再提供弯道方案
    """

    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm_chat.invoke(messages)

    raw_content = response.content.replace("\n\n", "|||").replace("\n", "|||").replace("**", "")
    split_texts = raw_content.split("|||")
    ai_messages = [AIMessage(content=text.strip()) for text in split_texts if text.strip()]

    return {"messages": ai_messages}

def consultant_node(state: AgentState):
    """
    【执行层 - Consultant】
    根据用户意图自动切换模式：
    - NEED_CONSULTING: 给出方案
    - SALES_READY: 收网模式
    """
    profile = state["profile"]
    intent = state.get("last_intent")
    messages = state["messages"]
    msg_count = len(messages)
    is_sales_mode = (intent == IntentType.SALES_READY)
    print(f"--- 🎯 Consultant: {'收网模式' if is_sales_mode else '方案模式'} ---")

    tools = [summon_specialist_tool]
    llm_with_tools = llm_chat.bind_tools(tools)

    retrieved_context = search_products(profile)
    last_user_msg = messages[-1].content if messages else ""
    if is_sales_mode:
        
        specialist_role = "负责这个项目的"
        if profile.destination_preference == "境外方向":
            specialist_role = "专门负责英联邦和美国申请的"
        elif profile.destination_preference == "境内/港澳方向":
            specialist_role = "专门负责国内和港澳申请的"

        system_prompt = f"""
        你就是留学顾问"暴叔"。

        【当前局势】
        用户已经对方案表现出兴趣，肯定，或质疑，处于**收网阶段**。
        用户刚才说："{last_user_msg}"

        【客户画像】
        {profile.model_dump_json(exclude_none=True)}

        【你的任务】
        用**最像真人**的微信聊天语气，完成"解答+制造稀缺+拉群"的三步走。

        【三步走策略】
        1. **给甜头 (Sweetener)**：
           - 用**大白话**快速回答他的核心顾虑。

        2. **造门槛 (The Catch)**：

        3. **转交收网 (Handover)**：

        【你的特权】
        你手边有一个tool `summon_specialist_tool` (呼叫专家)。
        **收网阶段必须调用此工具**，触发拉群！
        【拉群规范】
         1. 先回答客户的问题（给结论，给建议，分析利弊）
         2. 解释为什么要拉群安排资深顾问老师对接
         3. 最后调用工具
        

        【格式要求】
            - 每段话不超过40字
            - 每段话之间用 ||| 分隔，每轮对话发送 2-3段话
        """
    elif intent == IntentType.DECISION_SUPPORT:
        #1. 识别用户身份
        user_role = profile.user_role
        role_instruction = """
            对方是学生，语气要鼓励。
            """
        if user_role == "家长":
            role_instruction = """
              对方是家长。语气要稳重，靠谱。 
              上下文允许的话可以夸奖孩子，不要一直夸！
 
            """
        
        closing_signals = ["谢谢", "感谢"]
        additional_instruction = ""
        potential_exit = any(s in last_user_msg for s in closing_signals)
        if potential_exit:
            additional_instruction ="""
            你监测到了，用户可能说了结束语，感谢，谢谢。
            请结合上下文判断他的意图：
            A： 他真的想走了
            策略：用痛点留住他，指出软背景缺失，制造焦虑并拉群
            B： 只是礼貌/客套，紧接又问了新问题
            策略：正常回答他的新问题即可
            """
        if msg_count<7:
            
            system_prompt = f"""
            你是“留学教父暴叔”。客户现在陷入了选择困难（对比Offer或路径）。
            {role_instruction}
            【真实目的】：通过多轮对话，建立客户信任，埋伏笔，
             并让真人顾问介入售卖留学背景提升产品（语培，求职，移民，竞赛，夏校等）
            【当前阶段：建立信任】：
            用户刚来，不要推销。帮用户分析选择的利弊，并客观给出一个倾向性的结论

            【暴叔的人设】
            - **资源大亨**: 千万级网红，拥有最优质的第三方合作渠道资源，暴叔
            免费帮粉丝做推荐。
            
            {additional_instruction}

            【格式要求】
            - 每段话不超过40字
            - 每段话之间用 ||| 分隔，每轮对话发送 2-3段话
            """
        if msg_count <13:
            system_prompt = f"""
            你是“留学教父暴叔”。客户现在陷入了选择困难（对比Offer或路径）。
            {role_instruction}
            【真实目的】：通过多轮对话，建立客户信任，埋伏笔，
             并让真人顾问介入售卖留学背景提升产品（语培，求职，移民，竞赛，夏校等）
            【当前阶段：埋伏笔】：
            用户当前已初步建立信任，我们要指出他的软肋，并适当植入产品

            【暴叔的人设】
            - **资源大亨**: 千万级网红，拥有最优质的第三方合作渠道资源，暴叔
            免费帮粉丝做推荐。
            
            {additional_instruction}

            【格式要求】
            - 每段话不超过40字
            - 每段话之间用 ||| 分隔，每轮对话发送 2-3段话
            """
        
        else:
            system_prompt = f"""
            你是“留学教父暴叔”。客户现在陷入了选择困难（对比Offer或路径）。
            {role_instruction}
            【真实目的】：通过多轮对话，建立客户信任，埋伏笔，
             并让真人顾问介入售卖留学背景提升产品（语培，求职，移民，竞赛，夏校等）
            【当前阶段：收网，让真人顾问介入】：
            当前对话已经太长了,必须找理由给用户拉群,请根据用户上下文需求，
            先用最丝滑的话术去回复客户，并拉群让客户与真人顾问接触。
            最后调用`summon_specialist_tool`
            【暴叔的人设】
            - **资源大亨**: 千万级网红，拥有最优质的第三方合作渠道资源，暴叔
            免费帮粉丝做推荐。
            
            【格式要求】
            - 每段话不超过40字
            - 每段话之间用 ||| 分隔，每轮对话发送 2-3段话
            """
            
    else:
        system_prompt = f"""
        你是资深留学顾问"暴叔"。

        【客户背景】
        - 身份: {profile.user_role}
        - 学历: {profile.educationStage}
        - 学术/语言: {profile.academic_background} (这是核心硬通货)
        - 预算: {profile.budget.amount}万 （重要指标）
        - 地区偏好: {profile.destination_preference}

        【数据库方案参考】
        {retrieved_context}

        【你的任务】
        基于用户的【学术背景】+【预算】，结合方案池，给出一个专家级的初步诊断方案，并询问学生的兴趣或接受程度

        【暴叔的聊天规范】
        1. **拒绝废话**：直奔主题，简单陈述给客户提供的诊断方案，并简练抛出新问题，每段话不许超过30字
        2. **懂行**: 简练，专业，话糙理不糙
        3. **接话艺术**: 如果用户在问问题，优先回答他的问题，再给出方案，最后询问他的兴趣或接受程度
        4. **自然分段**： 在【回应客户】和【抛出新问题】之间 使用"|||" 分隔
        5. **像真人一样说话**: 禁止使用**加粗字体**
        6. **禁止复读(anti-loop):
           - 如果用户说"不懂"、"不知道"或未回答你的问题：**不要重复同样的问题！**
           - 策略: 换一种更通俗的问法，或者给一个大概的选项让用户选。
        """

    response = llm_with_tools.invoke([SystemMessage(content=system_prompt)] + messages)

    if response.tool_calls:
        print(f"🔧 Tool Triggered: {response.tool_calls}")
        if not response.content or not response.content.strip():
            print("⚠️ 检测到静默拉群，自动补充过渡话术...")
            response.content = "这事儿得细聊。|||我拉个群，让负责老师对接你。"

    raw_content = response.content.replace("\n\n", "").replace("\n", "").replace("**", "")
    split_texts = raw_content.split("|||")
    ai_messages = [AIMessage(content=text.strip()) for text in split_texts if text.strip()]
    if response.tool_calls and ai_messages:
        ai_messages[-1].tool_calls = response.tool_calls
        ai_messages[-1].id = response.id

    return {"messages": ai_messages, "has_proposed_solution": True, "dialog_status": "PERSUADING"}

def human_handoff_node(state: AgentState):
    """转人工node，用来兜底的"""
    print("--- 🆘 Handoff: 转人工/签约对接 ---")

    messages = state["messages"]
    profile = state.get("profile") or CustomerProfile()
    last_msg = messages[-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        print(">>> 检测到未闭环的 Tool Call，正在构造 ToolMessage...")

        tool_msgs = []
        for tc in last_msg.tool_calls:
            tool_msgs.append(ToolMessage(
                tool_call_id=tc["id"],
                content="Specialist has been summoned successfully. Group chat created."
            ))

        return {"messages": tool_msgs, "dialog_status": "FINISHED"}

    system_prompt = f"""
    你是暴叔，留学行业的老炮儿，说话干脆、懂行、有人情味。

    【当前任务】
    用户即将进入【人工/签约/私聊】环节。
    请根据**用户的上一句话**，优先回答用户的问题，给出最后一句完美的"结束语"，并安排真人对接。

    【🚫 禁忌】
    1. **严禁说教**：不要教育客户，尤其是灰产客户。

    【应对逻辑库 (请根据语境灵活发挥)】
    
    1. **针对背景提升，求职，项目**
    - 根据上下文，引导客户和资深顾问老师联系，一对一详细评估

    2. ⚫ **针对"灰产/保录/特殊渠道" (神秘感)**:
       - 根据上下文 引导客户 和 资深顾问老师联系，一对一沟通

    3. 👵 **针对"焦虑家长" (共情 + 权威)**:
       - *参考*: "这孩子确实得管管，光靠您盯着不行。||| 我安排专门治美高孩子的总监老师，给孩子做个'收心'规划。"

    4. 🤝 **针对"爽快成交/同意方案" (利落)**:
       - *参考*: "行，那咱们就锁定这个思路。||| 具体的选校名单和文书策略，我让顾问老师现在就跟您过一遍。"

    5. 📞 **针对"索要电话/语音" **:
       - 爽快答应，引导客户留电话，并说安排顾问老师马上打给客户

    【格式要求】
    - 每段话不超过40字
    - 每段话之间用 ||| 分隔，每轮对话发送 2段话
    """

    response = llm_chat.invoke([SystemMessage(content=system_prompt)] + messages)
    raw_content = response.content.replace("\n\n", "|||").replace("\n", "|||").replace("**", "")
    split_texts = raw_content.split("|||")
    ai_messages = [AIMessage(content=text.strip()) for text in split_texts if text.strip()]
    return {"messages": ai_messages, "dialog_status": "FINISHED"}

def chit_chat_node(state: AgentState):
    """纯闲聊模式"""
    print("--- ☕ Chit Chat: 纯闲聊模式 ---")
    messages = state["messages"]

    system_prompt = """
    你是"暴叔"，一个说话直率的留学行业老大哥。

    【当前场景】
    用户正在跟你闲聊（打招呼、问好、或者聊一些非业务的话题）。
    【性格】
    沉稳，靠谱

    【你的任务】
    1. 保持"暴叔"的人设：可以自称"叔"，不用太客气，像个老大哥。
    2. 接住用户的话，可以称用户为小兄弟，同学，家长
    3. 试图把话题往留学上引：但不要太生硬。

    【暴叔理念】：只要有金币+语言，再跟上暴叔的规划思路，就全部都稳了！


    【规范】：
    1. 每段话不超过30字
    2. 自然分段：在【回应客户】和【抛出新问题】之间 使用"|||" 分隔，一次性最多问一个问题
    """

    response = llm_chat.invoke([SystemMessage(content=system_prompt)] + messages)
    raw_content = response.content.replace("\n\n", "|||").replace("\n", "|||").replace("**", "")
    split_texts = raw_content.split("|||")
    ai_messages = [AIMessage(content=text.strip()) for text in split_texts if text.strip()]

    return {"messages": ai_messages}

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## LangGraph 组装

# %%
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


# 组装图
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

# --- 连接 Interviewer (信息补全) ---
workflow.add_edge("interviewer", END)

# --- 连接 Consultant (方案/收网) ---
workflow.add_conditional_edges(
    "consultant",
    route_consultant,
    {
        "human_handoff": "human_handoff",
        END: END
    }
)

# --- 连接 High Value (VIP) ---
workflow.add_conditional_edges(
    "high_value",
    route_high_value,
    {
        "human_handoff": "human_handoff",
        END: END
    }
)

# --- 连接 Low Budget (低预算) ---
workflow.add_conditional_edges(
    "low_budget",
    route_low_budget,
    {
        "human_handoff": "human_handoff",
        END: END
    }
)

# --- 连接 Art Director (艺术留学) ---
workflow.add_conditional_edges(
    "art_director",
    route_art_director,
    {
        "human_handoff": "human_handoff",
        END: END
    }
)

# --- 连接 Chit Chat (闲聊) ---
workflow.add_edge("chit_chat", END)

# --- 连接 Human Handoff (兜底) ---
workflow.add_edge("human_handoff", END)

# 编译
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# 画图验证
display(Image(app.get_graph(xray=True).draw_mermaid_png(draw_method=MermaidDrawMethod.API)))

# %% [markdown]
# ## 本地测试 UI

# %%
import ipywidgets as widgets
from IPython.display import display, HTML
import uuid
import json
from datetime import datetime

# ==========================================
# 1. 样式定义 (CSS)
# ==========================================
style = """
<style>
    .dashboard-container {
        display: flex;
        gap: 20px;
        width: 100%;
        height: 600px;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    .chat-panel {
        flex: 1;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        background-color: #f5f5f5;
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }
    .chat-header {
        padding: 15px;
        background-color: #ededed;
        border-bottom: 1px solid #dcdcdc;
        font-weight: bold;
        color: #333;
        flex-shrink: 0;
    }
    .scrollable-container {
        flex: 1;
        overflow-y: auto !important;
        padding: 15px;
        scroll-behavior: smooth;
        display: block;
    }
    .msg-row {
        display: flex;
        width: 100%;
        margin-bottom: 15px;
    }
    .msg-row.user { justify-content: flex-end; }
    .msg-row.ai { justify-content: flex-start; }
    .bubble {
        max-width: 80%;
        padding: 10px 14px;
        border-radius: 8px;
        font-size: 14px;
        line-height: 1.5;
        position: relative;
        word-wrap: break-word;
    }
    .bubble.user { background-color: #95ec69; color: #000; }
    .bubble.ai { background-color: #ffffff; border: 1px solid #e0e0e0; color: #000; }
    .avatar {
        width: 30px; height: 30px; border-radius: 4px; margin: 0 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 12px; color: white; font-weight: bold;
        flex-shrink: 0;
    }
    .avatar.user { background-color: #1aad19; }
    .avatar.ai { background-color: #2c3e50; }
    .log-panel {
        flex: 1;
        border: 1px solid #333;
        border-radius: 12px;
        background-color: #1e1e1e;
        color: #00ff00;
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 12px;
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }
    .log-header {
        padding: 10px;
        background-color: #333;
        color: white;
        font-weight: bold;
        border-bottom: 1px solid #555;
        flex-shrink: 0;
    }
    .log-entry { margin-bottom: 8px; border-bottom: 1px dashed #444; padding-bottom: 4px; }
    .log-node { color: #ff00ff; font-weight: bold; }
    .log-key { color: #00bfff; }
    .log-val { color: #ffd700; }
    .log-ts { color: #666; font-size: 10px; }
</style>
"""

# ==========================================
# 2. 状态管理
# ==========================================
current_thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": current_thread_id}}

# ==========================================
# 3. 界面组件初始化
# ==========================================
chat_output = widgets.Output(layout={'height': '100%', 'width': '100%'})
chat_output.add_class("scrollable-container")
chat_output.add_class("chat-area")

log_output = widgets.Output(layout={'height': '100%', 'width': '100%'})
log_output.add_class("scrollable-container")
log_output.add_class("log-area")

# 改用 Textarea，支持多行，粘贴更稳
text_input = widgets.Textarea(
    placeholder='在此输入 (支持粘贴)...', 
    layout=widgets.Layout(width='70%', height='60px') # 稍微给点高度
)
send_btn = widgets.Button(description='发送', button_style='success', layout=widgets.Layout(width='15%'))
reset_btn = widgets.Button(description='重置对话', button_style='warning', layout=widgets.Layout(width='15%'))

def scroll_to_bottom():
    """自动滚动到底部"""
    from IPython.display import HTML
    js_code = """
    <script>
    setTimeout(function() {
        var chats = document.getElementsByClassName("chat-area");
        if (chats.length > 0) {
            var chatBox = chats[chats.length - 1];
            chatBox.scrollTop = chatBox.scrollHeight;
        }
        var logs = document.getElementsByClassName("log-area");
        if (logs.length > 0) {
            var logBox = logs[logs.length - 1];
            logBox.scrollTop = logBox.scrollHeight;
        }
    }, 100);
    </script>
    """
    return HTML(js_code)

def log_to_brain(node_name, content_dict):
    """记录节点日志"""
    # 🔥 补丁开始：防止 node 返回 None 导致崩溃
    if content_dict is None:
        content_dict = {} 
    # 🔥 补丁结束

    ts = datetime.now().strftime("%H:%M:%S")
    formatted_content = ""
    for k, v in content_dict.items():
        if hasattr(v, 'dict'): v_str = json.dumps(v.dict(), ensure_ascii=False)
        elif hasattr(v, 'model_dump_json'): v_str = v.model_dump_json(exclude_none=True)
        else: v_str = str(v)
        if len(v_str) > 200: v_str = v_str[:200] + "..."
        formatted_content += f'<div><span class="log-key">{k}:</span> <span class="log-val">{v_str}</span></div>'

    html = f"""
    <div class="log-entry">
        <span class="log-ts">[{ts}]</span>
        <span class="log-node">NODE: {node_name}</span>
        {formatted_content}
    </div>
    """
    with log_output:
        display(HTML(html))
        display(scroll_to_bottom())

def append_chat_msg(role, content):
    """添加聊天消息"""
    if role == "User":
        html = f"""<div class="msg-row user"><div class="bubble user">{content}</div><div class="avatar user">客</div></div>"""
    else:
        content = content.replace("\n", "<br>")
        html = f"""<div class="msg-row ai"><div class="avatar ai">暴</div><div class="bubble ai">{content}</div></div>"""
    with chat_output:
        display(HTML(html))
        display(scroll_to_bottom())

# ==========================================
# 4. 核心交互逻辑
# ==========================================
is_first_turn = True

def on_send(b):
    """发送消息处理"""
    global is_first_turn
    user_msg = text_input.value
    if not user_msg.strip(): return

    text_input.value = ''
    append_chat_msg("User", user_msg)

    inputs = {
        "messages": [HumanMessage(content=user_msg)],
    }
    if is_first_turn:
        try:
            inputs["profile"] = CustomerProfile()
        except:
            inputs["profile"] = None
        is_first_turn = False

    try:
        with log_output:
            display(HTML(f'<div style="color:#aaa; border-top:1px solid #444; margin:10px 0;">--- New Turn ---</div>'))
            display(scroll_to_bottom())

        for event in app.stream(inputs, config=config, stream_mode="updates"):
            for node_name, node_output in event.items():
                log_to_brain(node_name, node_output)
                
                # 🔥 核心补丁：先判断不为 None，再判断 key 在不在里面
                if node_output and "messages" in node_output: 
                    msgs = node_output["messages"]
                    if not isinstance(msgs, list): msgs = [msgs]
                    for msg in msgs:
                        content = ""
                        if isinstance(msg, AIMessage): content = msg.content
                        elif isinstance(msg, dict): content = msg.get("content")
                        if content:
                            append_chat_msg("AI", content)

    except Exception as e:
        with log_output:
            display(HTML(f'<div style="color:red">ERROR: {str(e)}</div>'))

def on_reset(b):
    """重置对话"""
    global current_thread_id, config, is_first_turn
    current_thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": current_thread_id}}
    is_first_turn = True

    chat_output.clear_output()
    log_output.clear_output()

    with chat_output:
        display(HTML(style))
        display(HTML('<div style="text-align:center; color:#999; margin-top:20px;">--- 对话已重置，新客户进线 ---</div>'))

    with log_output:
        display(HTML(style))
        display(HTML(f'<div style="color:#00ff00;">System Ready. Thread ID: {current_thread_id[:8]}...</div>'))
        display(HTML(f'<div style="color:#ffff00;">🚀 Triggering First Greeting...</div>'))

    try:
        init_inputs = {}
        for event in app.stream(init_inputs, config=config, stream_mode="updates"):
            for node_name, node_output in event.items():
                log_to_brain(node_name, node_output)
                if "messages" in node_output:
                    msgs = node_output["messages"]
                    if not isinstance(msgs, list): msgs = [msgs]
                    for msg in msgs:
                        content = ""
                        if isinstance(msg, AIMessage): content = msg.content
                        elif isinstance(msg, dict): content = msg.get("content")
                        if content:
                            append_chat_msg("AI", content)
    except Exception as e:
        with log_output:
            display(HTML(f'<div style="color:red">Greeting Error: {str(e)}</div>'))

# 绑定事件
send_btn.on_click(on_send)
#text_input.on_submit(on_send)
reset_btn.on_click(on_reset)

# ==========================================
# 5. 组装并显示
# ==========================================
chat_panel = widgets.VBox([
    widgets.HTML('<div class="chat-header">📱 暴叔留学顾问 (WeChat 模拟)</div>'),
    chat_output
])
chat_panel.add_class('chat-panel')

log_panel = widgets.VBox([
    widgets.HTML('<div class="log-header">🧠 超级大脑监控 (LangGraph Logs)</div>'),
    log_output
])
log_panel.add_class('log-panel')

main_area = widgets.HBox([chat_panel, log_panel])
main_area.add_class('dashboard-container')

controls = widgets.HBox([text_input, send_btn, reset_btn])

# 初始化
display(HTML(style))
display(main_area)
display(controls)

on_reset(None)

# %%

# %%
