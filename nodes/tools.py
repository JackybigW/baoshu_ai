#这里全是干苦力的工具人
import os
import sys # 正则提取必用
import pandas as pd# 处理 Excel 必用

# 🛠️ 【防报错补丁】
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# -------------------------------------------------

# 工具定义相关
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, SystemMessage

# 状态与数据结构 (预算、枚举等通常在提取时用到)
from typing import List, Optional
from state import AgentState, BudgetInfo, BudgetPeriod, CustomerProfile
from enum import Enum
# LLM (如果你的提取器是用 LLM 做的，不需要的话可以删掉下面这两行)
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
deepseek_api_key = os.environ['DEEPSEEK_API_KEY']

llm = init_chat_model(
    "deepseek-chat", 
    model_provider="deepseek", 
    temperature=0, 
    api_key=deepseek_api_key
    ) # 提取信息必须严谨，温度设为0

#1 Excel Retriever Tool
#通过客户的预算和目的地偏好以及学历，给用户匹配合适的规划路径，给LLM参考。
#（技术上用的只有excel 和 pandas，可以理解为简单实用版RAG）
#（必须吐槽！！ 真正的rag 就是垃圾！！！也可能是我太菜 不会rag QAQ）

try:
    # 1. 读取 Excel
    df_db = pd.read_excel("products_intro.xlsx")

    # =========================================================
    # 🔥 核心修复：按列类型自动填充 (不再写死列名)
    # =========================================================
    
    # 2. 找出所有的【数字列】 (float, int)，把空值填为 0
    # include=['number'] 会自动抓取 float64, int64 等所有数字类型
    num_cols = df_db.select_dtypes(include=['number']).columns
    df_db[num_cols] = df_db[num_cols].fillna(0)
    
    # 3. 找出所有的【文本列】 (object, string)，把空值填为 ""
    obj_cols = df_db.select_dtypes(include=['object', 'string']).columns
    df_db[obj_cols] = df_db[obj_cols].fillna("")

    # =========================================================

except Exception as e:
    print(f"⚠️ 警告: 读取 Excel 失败: {e}")
    # 兜底：创建一个空的 DataFrame，防止后面代码报错
    # 这里的列名要和你 Excel 的表头一致
    df_db = pd.DataFrame(columns=[
        "educationStage", "是否出国", "budgetLowerBound", "budgetUpperBound", 
        "annualBudgetLowerBound", "annualBudgetUpperBound", 
        "项目", "项目说明", "项目优势", "画像", "学制"
    ])
#直接 define 一个 function 让 LLM强制用，让暴叔的方案大于LLM自己知识库的方案即可
def search_products(profile: CustomerProfile) -> str:
    """
    【翻译层 + 检索层】
    将 CustomerProfile 的 'destination_preference' 映射到 Excel 的 '是否出国' 字段
    """
    try:
        # 初始化 Mask (全选)
        mask = pd.Series([True] * len(df_db))
        
        # --- A. 学历筛选 ---
        if profile.educationStage:
            mask &= df_db['educationStage'].str.contains(profile.educationStage, na=False)
            
        # --- B. 意向筛选 (逻辑映射核心修改) ---
        user_pref = profile.destination_preference # "境外方向" / "境内/港澳方向"
        
        if user_pref == "境外方向":
            # 对应原逻辑：包括常规出国 + 缓缓再出国(预科/跳板)
            mask &= df_db['是否出国'].isin(["可以出国", "缓缓再出国"])
            
        elif user_pref == "境内/港澳方向":
            # 对应原逻辑：Excel里的"不要出国"分类 (即港澳/4+0)
            mask &= (df_db['是否出国'] == "不要出国")
            
        # --- C. 预算筛选 (保持不变) ---
        amount = profile.budget.amount
        period = profile.budget.period
        
        if amount > 0: 
            if period == BudgetPeriod.TOTAL:
                mask &= (df_db['budgetLowerBound'] <= amount)
            elif period == BudgetPeriod.ANNUAL:
                mask &= (df_db['annualBudgetLowerBound'] <= amount)
                
        # --- D. 执行筛选 ---
        results = df_db[mask]
        
        # --- E. 格式化输出 ---
        if len(results) == 0:
            return "【系统提示】：数据库中未找到符合该学历和预算的方案。请委婉告知用户目前没有完美匹配的项目，建议调整预算或考虑其他路径。"
            
        output_text = f"共匹配到 {len(results)} 个方案：\n"
        
        for idx, row in results.iterrows():
            proj_name = row.get('项目', '未知项目')
            proj_desc = row.get('项目说明', '')
            proj_adv = row.get('项目优势', '')
            # 这里虽然 Excel 还是叫"是否出国"，但展示给 LLM 时我们可以不强调这个字段，或者保留给它做参考
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
    
#2 企业微信拉群Tool，不在github展示了，意思一下。聊完之后 LLM用这个tool 直接拉客户和真人顾问进群。  
@tool
def summon_specialist_tool():
    """
    当客户表现出意向，或者需要资深顾问/具体方案介入时，调用此工具。
    这将触发后台系统，拉取真人顾问进入群聊。
    """
    return "Specialist summoned."


#3 用户信息提取node，用户给客户打标签
def extractor_node(state: AgentState):
    print("--- 🕵️ Extractor: 混合更新 (原子覆盖 + 文本合并) ---")
    
    # 1. 读取旧档案
    current_profile = state.get("profile")
    if current_profile is None:
        current_profile = CustomerProfile()
     
    # 2. 获取上下文
    msgs = state["messages"]
    last_user_msg = msgs[-1].content
    last_ai_msg = msgs[-2].content if len(msgs) > 1 else ""

    # 3. 构造 Prompt (核心修改)
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
    
    如果某字段既无显示信息，也无法逻辑推理，请返回None
    """
    # 4. 调用 LLM
    # 我们依然只传 prompt，不需要把整个 profile 对象传进去，避免干扰
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
        
    if new_data.academic_background:
        if final_profile.academic_background:
            # 如果原来有，就拼在后面
            final_profile.academic_background += f"；{new_data.academic_background}"
        else:
            # 如果原来没有，直接赋值
            final_profile.academic_background = new_data.academic_background

    print(f"最终画像: {final_profile.model_dump_json(exclude_none=True)}")

    return {"profile": final_profile}