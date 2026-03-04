#这里是负责前台和用户交互的顾问 agents 们
import os
import sys
import re

# 🛠️ 【防报错补丁】
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# -------------------------------------------------

from typing import List, Optional, Any, Union, Dict, Sequence, Literal
from pydantic import BaseModel
from langchain_core.messages import SystemMessage,  AIMessage, ToolMessage
from state import AgentState, CustomerProfile, IntentType
from nodes.tools import summon_specialist_tool, search_products
from utils.llm_factory import get_frontend_llm
from utils.logger import logger
from config.prompts import (
    HIGH_VALUE_PERSONA,
    ART_DIRECTOR_SYSTEM_PROMPT,
    LOW_BUDGET_SYSTEM_PROMPT,
    CHIT_CHAT_SYSTEM_PROMPT,
    HUMAN_HANDOFF_SYSTEM_PROMPT
)

# 结构化输出模型 (温度低，稳定)
llm = get_frontend_llm(temperature=0)

# 顾问聊天模型 (温度稍高，更像人)
llm_chat = get_frontend_llm(temperature=0.7)

#1 用户加了好友，优先say hi，勾引用户说话
def first_greeting_node(state: AgentState):
    """首句破冰"""
    logger.info("--- 👋 Greeting: AI 主动破冰 ---")
    return {
        "messages": [AIMessage(content="您好，欢迎咨询！ 跟上暴叔的节奏～")],
        "dialog_status": "START",
        "last_intent": "GREETING"
    }

#2 VIP会员通道（重构版：动态时期感知）
def high_value_node(state: AgentState):
    """
    【执行层 - High Value】
    针对高净值/VIP客户，采用与 Consultant 一致的动态三阶段逻辑：
    1. 信任建立 -> 2. 软肋植入 -> 3. 强力收网 (拉群)
    """
    from state import IntentType
    
    profile = state.get("profile") or CustomerProfile()
    intent = state.get("last_intent")
    messages = state["messages"]
    msg_count = len(messages)
    is_sales_mode = (intent == IntentType.SALES_READY)
    
    logger.info(f"--- 🎩 High Value: {'收网模式' if is_sales_mode else '动态阶段模式'} (Count: {msg_count}) ---")
    last_user_msg = state["messages"][-1]
    tools = [summon_specialist_tool]
    
    # 🔥 核心控制：阶段一 (msg_count < 7) 禁用工具，强制文字咨询
    # 阶段二、阶段三及 Sales Ready 模式开启工具，允许摇人
    if is_sales_mode or msg_count >= 7:
        active_llm = llm_chat.bind_tools(tools)
        can_use_tool = True
    else:
        active_llm = llm_chat
        can_use_tool = False
    
    # 核心人设：懂行的暴叔，资源大亨，对 VIP 客户极度负责
    persona = HIGH_VALUE_PERSONA

    if is_sales_mode:
        system_prompt = f"""
        你是留学顾问暴叔

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
    elif msg_count < 7:
        system_prompt = f"""
        {persona}
        【当前阶段：建立信任】
        用户刚开始咨询。
        【任务】: 
        1. 表现出对 VIP 画像的精准理解。
        2. 结合画像悉心解答用户的问题，并给客观建议
        3. 严禁此时推销！目的是建立“暴叔很懂行”的第一印象。
        【客户画像】: {profile.model_dump_json(exclude_none=True)}
        """
    elif msg_count < 13:
        system_prompt = f"""
        {persona}
        【当前阶段：埋伏笔】
        用户已初步建立信任。
        【任务】: 
        1. 结合画像适当暗示其申请中的致命隐患。
        2. 适当暗示暴叔手里资源优秀可弯道超车，以及团队的专业程度，会辅佐客户留学申请。
        3. 引起用户的焦虑感和好奇心。
        【客户画像】: {profile.model_dump_json(exclude_none=True)}
        【你的特权】
        你手边有一个tool `summon_specialist_tool` (呼叫专家)。
        当前是埋伏笔阶段，只有在用户肯定方案，或者要求语音通话时使用
        【拉群规范】
         1. 先回答客户的问题（给结论，给建议，分析利弊）
         2. 解释为什么要拉群安排资深顾问老师对接
         3. 最后调用工具
        """
    else:
        system_prompt = f"""
        {persona}
        【客户画像】: {profile.model_dump_json(exclude_none=True)}
        
        【当前阶段：收网，让真人顾问介入】：
            当前对话已经太长了,必须找理由给用户拉群,请根据用户上下文需求，
            先用最丝滑的话术去回复客户，并拉群让客户与真人顾问接触。
            最后调用`summon_specialist_tool`
        
        """

    response = active_llm.invoke([SystemMessage(content=system_prompt)] + messages)

    # 结构化处理回复 (防静默逻辑只在允许使用工具时生效)
    if can_use_tool and response.tool_calls:
        logger.info(f"🔧 High Value Tool Triggered: {response.tool_calls}")
        if not response.content or len(response.content.strip()) < 5:
            logger.warning("⚠️ 检测到回复过短，自动补偿 VIP 话术...")
            response.content = "这件事儿细节挺多，一句两句说不完。|||我直接拉个群，安排最资深的顾问老师来跟你一对一对接。"

    raw_content = response.content.replace("\n\n", "").replace("\n", "").replace("**", "")
    split_texts = raw_content.split("|||")
    ai_messages = [AIMessage(content=text.strip()) for text in split_texts if text.strip()]
    
    if can_use_tool and response.tool_calls and ai_messages:
        ai_messages[-1].tool_calls = response.tool_calls
        ai_messages[-1].id = response.id

    return {"messages": ai_messages, "dialog_status": "VIP_SERVICE"}

#3 艺术留学顾问（艺术留学和普通留学不兼容，需分开）
def art_node(state: AgentState):
    logger.info("--- 🎩 Art Director: 艺术总监 ---")
    
    profile = state.get("profile") or CustomerProfile()
    messages = state["messages"]
    
    # 1. 绑定工具
    tools = [summon_specialist_tool]
    # 强制绑定工具，但允许它不调用 (即继续聊天)
    llm_with_tools = llm_chat.bind_tools(tools)
    closing_pressure = ""
    if len(messages) >= 16: # 稍微给点空间，大概5-6轮对话
        closing_pressure = "\n【⚠️系统警告】对话过长，客户可能流失。请立刻寻找理由拉群，停止追问细节！"
    
    system_prompt = ART_DIRECTOR_SYSTEM_PROMPT.format(
        profile=profile.model_dump_json(exclude_none=True),
        closing_pressure=closing_pressure
    )
    
    # 调用带工具的 LLM
    response = llm_with_tools.invoke([SystemMessage(content=system_prompt)] + messages)
    
    if response.tool_calls:
        logger.info(f"🔧 Tool Triggered: {response.tool_calls}")
        return {"messages": [response], "dialog_status": "VIP_SERVICE"}
    
    # 情况 B: 纯聊天 (Chat)
    # 手动处理分段，为了 UI 好看
    else:
        raw_content = response.content.replace("\n\n", "|||").replace("\n", "|||").replace("**", "")
        split_texts = raw_content.split("|||")
        # 把切分后的文本重新封装成多个 AIMessage
        ai_messages = [AIMessage(content=text.strip()) for text in split_texts if text.strip()]
        
        return {"messages": ai_messages, "dialog_status": "VIP_SERVICE"}

#5 常规咨询的顾问，主要工作是解答问题，并采访用户获取背景信息
def interviewer_node(state: AgentState):
    logger.info("--- 🎤 Interviewer ---")
    profile = state.get("profile") or CustomerProfile()
    missing = profile.missing_fields
    # 获取上下文
    user_role = profile.user_role      # "学生" / "家长"
    stage = profile.educationStage     # "高中" / "本科" ...
    if not missing:
        return {"messages": [AIMessage(content="情况都清楚了，咱们直接看方案！")]}
    
    # 1. 先拿 Pydantic 默认缺项 (通常是 "当前学历")
    target_field = profile.missing_fields[0] if profile.missing_fields else None
    
    # 初始化拦截标记
    force_ask_score = False

    # 2. 只有当用户【已经填了背景】时，才去检查有没有分
    if profile.educationStage and profile.academic_background:
        if not re.search(r'\d|[ABC][\+\-]?|Distinction|Merit|Pass|预估', profile.academic_background, re.IGNORECASE):
             target_field = "academic_background"
    # ============================================================
    # 1. 动态人设 (Role Persona) - 决定语气
    # ============================================================
    # 这部分保持不变，因为它决定了“屁股坐在哪一边”
    if user_role == "家长":
        role_instruction = "【对话策略】：对方是家长。语气要稳重、让其放心。关注点在于安全、就业，孩子前途。"
    elif user_role == "学生":
        role_instruction = "【对话策略】：对方是学生。语气要像老大哥一样懂行、给鼓励。关注点在于学校排名。"
    else:
        role_instruction = "【对话策略】：身份未知。默认当做学生聊，保持中性亲切。"

    # ============================================================
    # 2. 动态关注点 (Focus Points) - 决定问什么方向
    # ============================================================
    # 这里的关键是：只给方向，不给具体句子！
    
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

    # ============================================================
    # 3. 强制打招呼 (Greeting) - 保持不变
    # ============================================================
    greeting_instruction = ""
    if user_role:
        target_greeting = f"{'同学' if user_role == '学生' else '家长'}您好！"
        has_greeted = False
        for msg in state["messages"]:
            if isinstance(msg, AIMessage) and target_greeting in msg.content:
                has_greeted = True; break
        if not has_greeted:
            greeting_instruction = f"**回复必须以 “{target_greeting}” 开头**。"

    # ============================================================
    # 4. 组装极简 Prompt
    # ============================================================
    
    system_prompt = f"""
    你就是留学顾问“暴叔”。
    
    【已知画像】: {profile.model_dump_json(exclude_none=True)}
    【当前任务】: 追问缺失项 -> **{target_field}**
    
    {greeting_instruction}
    
    {role_instruction}
    {focus_instruction}
    
    
    【暴叔的聊天规范】
    1. **拒绝废话**：直奔主题，简单回应用户上一句，并简练抛出新问题，每段话不许超过40字
    2. **懂行**：说话要切中要害，对数据敏感。如果问成绩，表现出对分数敏感；如果问预算，表现出对性价比关注
    3. **接话艺术**:如果用户上一句是在问问题，你必须用自己的知识库简单作答给出结果，再自然衔接问题
    4. **自然分段**： 在【回应客户】和【抛出新问题】之间 使用"|||" 分隔
    5. **像真人一样说话**: 禁止使用**加粗字体**
    6. **禁止复读(anti-loop):
       - 如果用户说“不懂”、“不知道”或未回答你的问题：**不要重复同样的问题！**
       - 策略: 换一种更通俗的问法，或者给一个大概的选项让用户选。
    7. **高情商表达**: 对于背景差目标高的客户，先承认难度大，再提供弯道方案
    """

    # 调用
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm_chat.invoke(messages) # 前台对话使用 llm_chat
    
    # Split处理
    raw_content = response.content.replace("\n\n", "|||").replace("\n", "|||").replace("**", "")
    split_texts = raw_content.split("|||")
    ai_messages = [AIMessage(content=text.strip()) for text in split_texts if text.strip()]
    
    return {"messages": ai_messages}

#6 提供方案的顾问（含 Sales 收网功能）
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
    
    logger.info(f"--- 🎯 Consultant: {'收网模式' if is_sales_mode else '方案模式'} ---")
    
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
            请结合上下文判断 his 意图：
            A： 他真的想走了
            策略：用痛点留住他，指出软背景缺失，制造焦虑并拉群
            B： 只是礼貌/客套，紧接又问了新问题
            策略：正常回答他的新问题即可
            """
        if msg_count<7:
            
            system_prompt = f"""
            你是"留学教父暴叔"。客户现在陷入了选择困难（对比Offer或路径）。
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
            你是"留学教父暴叔"。客户现在陷入了选择困难（对比Offer或路径）。
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
            你是"留学教父暴叔"。客户现在陷入了选择困难（对比Offer或路径）。
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
        logger.info(f"🔧 Tool Triggered: {response.tool_calls}")
        if not response.content or not response.content.strip():
            logger.warning("⚠️ 检测到静默拉群，自动补充过渡话术...")
            response.content = "这件事儿细节挺多，一句两句说不完。|||我直接拉个群，安排最资深的顾问老师来跟你一对一对接。"

    raw_content = response.content.replace("\n\n", "").replace("\n", "").replace("**", "")
    split_texts = raw_content.split("|||")
    ai_messages = [AIMessage(content=text.strip()) for text in split_texts if text.strip()]
    if response.tool_calls and ai_messages:
        ai_messages[-1].tool_calls = response.tool_calls
        ai_messages[-1].id = response.id

    return {"messages": ai_messages, "has_proposed_solution": True, "dialog_status": "PERSUADING"}

#7 低预算客户专属节点 - 提供性价比方案，快速转人工
def low_budget_node(state: AgentState):
    logger.info("--- 💰 Low Budget ---")
    
    profile = state.get("profile") or CustomerProfile()
    messages = state["messages"]
    
    # 绑定工具
    tools = [summon_specialist_tool]
    llm_with_tools = llm_chat.bind_tools(tools)
    
    closing_pressure = ""
    if len(messages) >= 12:
        closing_pressure = "\n【⚠️系统警告】对话过长，请尽快寻找理由拉群转人工！"
    
    system_prompt = LOW_BUDGET_SYSTEM_PROMPT.format(
        profile=profile.model_dump_json(exclude_none=True),
        closing_pressure=closing_pressure
    )
    
    # 调用带工具的 LLM
    response = llm_with_tools.invoke([SystemMessage(content=system_prompt)] + messages)
    
    if response.tool_calls:
        logger.info(f"🔧 Tool Triggered: {response.tool_calls}")
        return {"messages": [response], "dialog_status": "PERSUADING"}
    
    # 情况 B: 纯聊天 (Chat)
    else:
        raw_content = response.content.replace("\n\n", "|||").replace("\n", "|||").replace("**", "")
        split_texts = raw_content.split("|||")
        # 把切分后的文本重新封装成多个 AIMessage
        ai_messages = [AIMessage(content=text.strip()) for text in split_texts if text.strip()]
        
        return {"messages": ai_messages, "dialog_status": "PERSUADING"}

#8 转人工node，用来兜底的。。
def human_handoff_node(state: AgentState):

    logger.info("--- 🆘 Handoff ---")
    
    messages = state["messages"]
    profile = state.get("profile") or CustomerProfile()
    last_msg = messages[-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        logger.info(">>> 检测到未闭环的 Tool Call，正在构造 ToolMessage...")
        
        tool_msgs = []
        for tc in last_msg.tool_calls:
            # 构造一个假的工具执行结果，骗过 LLM 协议，防止 400 Error
            tool_msgs.append(ToolMessage(
                tool_call_id=tc["id"],
                content="Specialist has been summoned successfully. Group chat created."
            ))
            
        return {"messages": tool_msgs, "dialog_status": "FINISHED"}
    
    system_prompt = HUMAN_HANDOFF_SYSTEM_PROMPT

    response = llm_chat.invoke([SystemMessage(content=system_prompt)] + messages)
    raw_content = response.content.replace("\n\n", "|||").replace("\n", "|||").replace("**", "")
    split_texts = raw_content.split("|||")
    ai_messages = [AIMessage(content=text.strip()) for text in split_texts if text.strip()]
    return {"messages": ai_messages, "dialog_status": "FINISHED"}

#2.28增加
def chit_chat_node(state: AgentState):
    logger.info("--- ☕ Chit Chat: 纯闲聊模式 ---")
    messages = state["messages"]
    
    system_prompt = CHIT_CHAT_SYSTEM_PROMPT
    
    # 简单的直接调用
    response = llm_chat.invoke([SystemMessage(content=system_prompt)] + messages)
    raw_content = response.content.replace("\n\n", "|||").replace("\n", "|||").replace("**", "")
    split_texts = raw_content.split("|||")
    # 把切分后的文本重新封装成多个 AIMessage
    ai_messages = [AIMessage(content=text.strip()) for text in split_texts if text.strip()]
    
    return {"messages": ai_messages} # 闲聊不改变 dialog_status，保持原样即可
