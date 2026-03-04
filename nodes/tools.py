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
    # 1. 获取当前脚本所在目录 (.../nodes)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. 获取项目根目录 (.../baoshu_ai) - 🔥 关键修改：往上跳一级
    project_root = os.path.dirname(current_dir) 

    # 3. 拼凑出 Excel 的绝对路径
    excel_path = os.path.join(project_root, "data", "products_intro.xlsx")
    
    df_db = pd.read_excel(excel_path)

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


