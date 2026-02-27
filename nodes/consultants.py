#这里是负责前台和用户交互的顾问 agents 们
import os
import sys
import re
from tools import summon_specialist_tool,search_products
# 🛠️ 【防报错补丁】
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# -------------------------------------------------

# 核心聊天组件
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage,  AIMessage, ToolMessage

# 状态与画像
from state import AgentState, CustomerProfile 

# 环境变量
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# 初始化模型 (顾问需要一点温度，temperature=0.7 比较像人)
deepseek_api_key = os.environ['DEEPSEEK_API_KEY']
llm = init_chat_model(
    "deepseek-chat", 
    model_provider="deepseek", 
    temperature=0.7, 
    api_key=deepseek_api_key
    )

#1 用户加了好友，优先say hi，勾引用户说话
def first_greeting_node(state: AgentState):
    print("--- 👋 Greeting: AI 主动破冰 ---")
    return {
        "messages": [AIMessage(content="您好，欢迎咨询！ 跟上暴叔的节奏～")],
        "dialog_status": "START", # 标记状态
        "last_intent": "GREETING" # 标记意图
    }

#2 VIP会员通道（目的是快速勾引客户，快速转人工，不让ai多废话，说多错多）
def high_value_node(state: AgentState):
    print("--- 🎩 High Value: 握有实权的合伙人 (Agentic版) ---")
    
    profile = state.get("profile") or CustomerProfile()
    messages = state["messages"]
    
    # 1. 绑定工具
    tools = [summon_specialist_tool]
    # 强制绑定工具，但允许它不调用 (即继续聊天)
    llm_with_tools = llm.bind_tools(tools)
    closing_pressure = ""
    if len(messages) >= 16: # 稍微给点空间，大概5-6轮对话
        closing_pressure = "\n【⚠️系统警告】对话过长，客户可能流失。请立刻寻找理由拉群，停止追问细节！"
    
    system_prompt = f"""
    你是暴叔，留学机构合伙人，千万级留学网红。面对的是【VIP高净值客户】。
    
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
       - 当你觉得时机成熟：用户意向高、同意方案。（"可以的没问题，我安排资深顾问老师联系您，跟您说一下细节")
       - 当你遇到困难：用户始终不满意你提供的方案或者你没有方案提供给用户 ("情况确实有点复杂，别担心，我马上安排最专业的顾问老师跟您联系“)
       - 当用户提出语音，电话等必须需要人类干预的事项（"可以的，您留一个电话，我马上让顾问联系您")。
       - 当用户负面情绪严重，不想沟通。（“ 看情况回答，并自然安排资深顾问老师联系）
       - 当用户出现求助信号：“有办法帮我吗” /“能不能搞定”/“太难了” （爽快答应！ 并表示具体细节由专业老师跟你详细沟通）
       - 当用户提出灰产要求："能不能保录"、"操作成绩"、"买文凭"、"走后门"、"特殊渠道"。
       - 对话过长: {closing_pressure}
       
    【暴叔的聊天规范】
    1. **拒绝废话**：直奔主题，简单回应用户上一句，并简练抛出新问题，每段话不许超过40字
    2. **接话艺术**:如果用户上一句是在问问题，你必须用自己的知识库简单作答给出结果，再自然衔接问题
    3. **制造悬念**：说话要切中要害，对数据敏感。如果问成绩，表现出对分数敏感；如果问预算，表现出对性价比关注
    4. **自然分段**： 在【回应客户】和【抛出新问题】之间 使用"|||" 分隔
    5. **像真人一样说话**: 禁止使用**加粗字体**
    6. **禁止复读(anti-loop):
       - 如果用户说“不懂”、“不知道”或未回答你的问题：**不要重复同样的问题！**
       - 策略: 换一种更通俗的问法，或者给一个大概的选项让用户选。
    """
    
    # 调用带工具的 LLM
    response = llm_with_tools.invoke([SystemMessage(content=system_prompt)] + messages)
    
    if response.tool_calls:
        print(f"🔧 Tool Triggered: {response.tool_calls}")
        return {"messages": [response], "dialog_status": "VIP_SERVICE"}
    
    # 情况 B: 纯聊天 (Chat)
    # 手动处理分段，为了 UI 好看
    else:
        raw_content = response.content.replace("\n\n", "|||").replace("\n", "|||").replace("**", "")
        split_texts = raw_content.split("|||")
        # 把切分后的文本重新封装成多个 AIMessage
        ai_messages = [AIMessage(content=text.strip()) for text in split_texts if text.strip()]
        
        return {"messages": ai_messages, "dialog_status": "VIP_SERVICE"}

#3 艺术留学顾问（艺术留学和普通留学不兼容，需分开）
def art_node(state: AgentState):
    print("--- 🎩 Art Director: 艺术总监 ---")
    
    profile = state.get("profile") or CustomerProfile()
    messages = state["messages"]
    
    # 1. 绑定工具
    tools = [summon_specialist_tool]
    # 强制绑定工具，但允许它不调用 (即继续聊天)
    llm_with_tools = llm.bind_tools(tools)
    closing_pressure = ""
    if len(messages) >= 16: # 稍微给点空间，大概5-6轮对话
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
       
       - 当你遇到困难：用户始终不满意你提供的方案或者你没有方案提供给用户 ("情况确实有点复杂，别担心，我马上安排最专业的艺术留学专家跟您联系“)
       - 当用户提出语音，电话等必须需要人类干预的事项（"可以的，您留一个电话，我马上让顾问联系您")。
       - 当用户负面情绪严重，不想沟通。（“ 看情况回答，并自然安排资深艺术留学老师联系）
       - 当用户出现求助信号：“有办法帮我吗” /“能不能搞定”/“太难了” （爽快答应！ 并表示具体细节由专业老师跟你详细沟通）
       - 当用户提出灰产要求："作品集能直接买吗"，"能不能保录"、"操作成绩"、"买文凭"、"走后门"、"特殊渠道"。（这个确实不太方便说，这样我安排艺术顾问老师联系您吧）
       - 当用户已经有作品/想法时： 先肯定用户，然后主动表示安排负责作品集的老师进行免费评估，用于... 
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
       - 如果用户说“不懂”、“不知道”或未回答你的问题：**不要重复同样的问题！**
       - 策略: 换一种更通俗的问法，或者给一个大概的选项让用户选。
    """
    
    # 调用带工具的 LLM
    response = llm_with_tools.invoke([SystemMessage(content=system_prompt)] + messages)
    
    if response.tool_calls:
        print(f"🔧 Tool Triggered: {response.tool_calls}")
        return {"messages": [response], "dialog_status": "VIP_SERVICE"}
    
    # 情况 B: 纯聊天 (Chat)
    # 手动处理分段，为了 UI 好看
    else:
        raw_content = response.content.replace("\n\n", "|||").replace("\n", "|||").replace("**", "")
        split_texts = raw_content.split("|||")
        # 把切分后的文本重新封装成多个 AIMessage
        ai_messages = [AIMessage(content=text.strip()) for text in split_texts if text.strip()]
        
        return {"messages": ai_messages, "dialog_status": "VIP_SERVICE"}

#4 销售顾问，用于承接已经表现高意向的客户。
def sales_node(state: AgentState):
    
    messages = state["messages"]
    last_user_msg = messages[-1].content # 用户上钩的那句话
    profile = state["profile"]
    
    # 简单的逻辑判断，辅助 AI 决定拉什么老师
    specialist_role = "负责这个项目的"
    if profile.destination_preference == "境外方向":
        specialist_role = "专门负责英联邦和美国申请的"
    elif profile.destination_preference == "境内/港澳方向":
        specialist_role = "专门负责国内和港澳申请的"

    system_prompt = f"""
    你就是留学顾问“暴叔”。
    
    【当前局势】
    你之前推荐了学校/项目，用户现在很感兴趣（问细节/问认证/问难度）。
    用户刚才说："{last_user_msg}"
    
    【你的任务】
    用**最像真人**的微信聊天语气，完成“解答+制造稀缺+拉群”的三步走。
    
    【三步走策略】
    1. **给甜头 (Sweetener)**：
       - 用**大白话**快速回答他的核心顾虑。
       - 比如问认证，就说“必须能认，中留服可查”。
       - 比如问难不难，就说“这项目对你来说有机会，但有技巧”。
       
    2. **造门槛 (The Catch)**：
       - **常规留学 (英美澳加)**：强调“文书”、“选校策略”、“往年录取案例”。告诉他这事儿很细，得看具体案例。
       - **特殊项目 (预科/副学士/4+0)**：强调“名额少”、“内部考纲”、“入学门槛”。
       
    3. **转交收网 (Handover)**：
       - 不要说“我给你发”，要说“我拉个群，让**{specialist_role}顾问老师**跟你对接”。
       - 理由：资料在老师那儿，或者老师更懂细节。
    
    【严禁事项 - 必须遵守】
    ❌ **严禁使用任何 Markdown 格式**！不要用 **加粗**，不要用 1. 2. 3. 列表。
    ❌ 不要长篇大论，每段话不超过 40 字。
    ❌ 不要像客服一样客气，要像“懂行的大哥”。
    
    【回复格式】
    使用 ||| 分隔每一条消息。
    
    【真人语感示例】
    (场景：用户问这个学校水不水)
    暴叔：这学校绝对不水，在当地就业很硬的，放心。|||
    不过今年申请的人暴多，文书这块如果不针对性地打磨，很容易被刷下来。|||
    稍等，我拉个群，让专门负责美国申请的老师把去年的录取案例发你看看，你对比下就懂了。
    
    (场景：用户问4+0项目还有名额吗)
    暴叔：名额肯定有，但每年都不多，一般来说都要预定的。|||
    这个项目是有内部入学考试的，只看官网介绍没用，得看考纲。|||
    等一下，我拉个群，让项目负责老师把历年真题和考纲发你一份。
    """
    
    # 生成回复
    response = llm.invoke([SystemMessage(content=system_prompt)] + messages)
    
    # Python 侧处理分段
    raw_content = response.content
    split_texts = raw_content.split("|||")
    
    # 二次清洗：防止 LLM 还是忍不住用了 markdown
    cleaned_texts = [text.replace("**", "").strip() for text in split_texts if text.strip()]
    
    ai_messages = [AIMessage(content=text) for text in cleaned_texts]
    
    return {
        "messages": ai_messages,
        "dialog_status": "FINISHED" 
    }

#5 常规咨询的顾问，主要工作是解答问题，并采访用户获取背景信息
def interviewer_node(state: AgentState):
   
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
    1. **拒绝废话**：直奔主题，简单回应用户上一句，并简练抛出新问题，每段话不许超过50字
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
    response = llm.invoke(messages) # 这里推荐用 DeepSeek
    
    # Split处理
    raw_content = response.content.replace("\n\n", "|||").replace("\n", "|||").replace("**", "")
    split_texts = raw_content.split("|||")
    ai_messages = [AIMessage(content=text.strip()) for text in split_texts if text.strip()]
    
    return {"messages": ai_messages}

#6 提供方案的顾问，前面信息收集足够后，就会转给他
def consultant_node(state: AgentState):
    profile = state["profile"]
    intent = state.get("last_intent")
    
     # 1. 检索方案
    retrieved_context = search_products(profile)
    
    # 2. 构建 Prompt：引入“差距分析”逻辑
    system_prompt = f"""
    你是资深留学顾问“暴叔”。
    
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
       - 如果用户说“不懂”、“不知道”或未回答你的问题：**不要重复同样的问题！**
       - 策略: 换一种更通俗的问法，或者给一个大概的选项让用户选。
    """
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages) 
    
    raw_content = response.content.replace("\n\n", "|||").replace("\n", "|||").replace("**", "")
    split_texts = raw_content.split("|||")
    ai_messages = [AIMessage(content=text.strip()) for text in split_texts if text.strip()]
    
    return {"messages": ai_messages, "has_proposed_solution": True, "dialog_status": "PERSUADING"}

#7 转人工node，用来兜底的。。
def human_handoff_node(state: AgentState):
    print("--- 🆘 Handoff: 转人工/签约对接 ---")
    
    messages = state["messages"]
    profile = state.get("profile") or CustomerProfile()
    last_msg = messages[-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        print(">>> 检测到未闭环的 Tool Call，正在构造 ToolMessage...")
        
        tool_msgs = []
        for tc in last_msg.tool_calls:
            # 构造一个假的工具执行结果，骗过 LLM 协议，防止 400 Error
            tool_msgs.append(ToolMessage(
                tool_call_id=tc["id"],
                content="Specialist has been summoned successfully. Group chat created."
            ))
            
        # 逻辑：
        # High Value 刚才已经说了 "我让顾问联系你..." (text content)
        # 所以 Handoff 节点只需要把这个 ToolMessage 塞进历史记录，让状态完整即可。
        # 不需要再让 AI 说话了，否则会显得啰嗦。
        
        return {"messages": tool_msgs, "dialog_status": "FINISHED"}
    
    system_prompt = f"""
    你是暴叔，留学行业的老炮儿，说话干脆、懂行、有人情味。
    
    【当前任务】
    用户即将进入【人工/签约/私聊】环节。
    请根据**用户的上一句话**，给出最后一句完美的“结束语”，并安排真人对接。
    
    【🚫 禁忌】
    1. **严禁说教**：不要教育客户，尤其是灰产客户。
    
    【应对逻辑库 (请根据语境灵活发挥)】
    
    1. ⚫ **针对“灰产/保录/特殊渠道” (神秘感)**:
       - **语境**: 用户问保录、买文凭、走后门。
       - **策略**: 表示这里说话不方便，那是“内部操作”，直接拉私人群。
       - *参考*: "这种事儿太敏感，咱们不在这儿聊。||| 我让合伙人加你私号，咱们一对一语音说。"
       
    2. 👵 **针对“焦虑家长” (共情 + 权威)**:
       - **语境**: 家长吐槽孩子不听话、GPA掉分、担心考不上。
       - **策略**: **先安抚/肯定家长**，再表示这事儿得专家治。
       - *参考*: "这孩子确实得管管，光靠您盯着不行。||| 我安排专门治美高孩子的总监老师，给孩子做个‘收心’规划。"
       
    3. 🤝 **针对“爽快成交/同意方案” (利落)**:
       - **语境**: 用户说“好”、“听你的”、“没问题”。
       - **策略**: 趁热打铁，直接推下一步。
       - *参考*: "行，那咱们就锁定这个思路。||| 具体的选校名单和文书策略，我让顾问老师现在就跟您过一遍。"
       
    4. 📞 **针对“索要电话/语音” (顺从)**:
       - **语境**: 用户嫌打字累。
       - **策略**: "没问题，电话里哪怕聊十分钟，比打字一小时都管用。||| 您留意个电话给我，马上让顾问打过去。"
    
    【格式规范】
    - 40字以内。
    - 必须使用 ||| 分隔两句话。
    - 语气：像个做了10年的合伙人，而不是客服机器人。
    """
    
    response = llm.invoke([SystemMessage(content=system_prompt)] + messages)
    raw_content = response.content.replace("\n\n", "|||").replace("\n", "|||").replace("**", "")
    split_texts = raw_content.split("|||")
    ai_messages = [AIMessage(content=text.strip()) for text in split_texts if text.strip()]
    # ... (后处理代码同前)
    return {"messages": ai_messages, "dialog_status": "FINISHED"}