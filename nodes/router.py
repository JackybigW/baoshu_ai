#这里是专门用来 分流的 节点们（调度中心）
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage

# 状态与类型
from state import AgentState, IntentResult # 假设 IntentResult 在 state 里

# 环境变量 (加载 API Key)
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# 初始化模型
deepseek_api_key = os.environ['DEEPSEEK_API_KEY']
llm = init_chat_model(
    "deepseek-chat", 
    model_provider="deepseek", 
    temperature=0, 
    api_key=deepseek_api_key
    )

#1 分类器，是用户每轮进入对话后的第一个节点，用于分类用户类型。
def classifier_node(state: AgentState):
    print("--- 🧠 正在给客户分类 ---")
    recent_msg = state["messages"][-6:]
    current_status = state.get("dialog_status")
    classifier = llm.with_structured_output(IntentResult)
    
    prompt = [
            SystemMessage(content="""
            你是留学顾问暴叔。请根据用户的最新回复判断客户层级和意图
            
            【判定逻辑 - 优先级从高到低】
            
            1. 🆘 **TRANSFER_TO_HUMAN (转人工)**:
               - **逻辑**: 用户想要跳过ai咨询，直接进入真人对接环节或者想要咨询非常规业务
               - **用户提及留学灰产&保录**：
               - **负面情绪/障碍信号**: "我不懂", "太复杂了", "不知道", "直接找人跟我说", "别问了"，“我不想折腾”.
               - **要求非文本沟通**: 电话，微信，语音沟通
               - **成交/同意信号**： 用户表示同意或对方案的赞成，"可以","没问题"，"听你的"
               
            2. 🎨 **ART_CONSULTING (艺术生通道)**:
                - **关键词**: "作品集", "交互设计"，提及艺术类院校，"游戏设计", "服装设计", "纯艺", "插画", "动画", "电影","导演“.
                - **逻辑**: 只要涉及需要**作品集(Portfolio)**的专业，无论预算多少，统统归为此类。因为艺术申请逻辑完全不同。  
            
            3. 🔥 **HIGH_VALUE (高价值/VIP客户)**:
               - **关键词**: 
                 - 预算高: "不差钱", "预算没问题", "50w+", "80w", "100w".
                 - 背景强: "美本", "英本", "澳本", "加本", "海高", "美高", "A-level", "IB".
                 - 目标高: "只想去滕校", "G5", "港三".
               - **逻辑**: 只要用户流露出“不缺钱”或者“出身海外院校”的信号，一律归为此类。
            
            4. 📋 **NEED_CONSULTING (普通咨询)**:
               - 普通背景，预算正常或未提及，需要常规规划。
               
            5. 👋 **GREETING / CHIT_CHAT**:
               - 纯打招呼或闲或简单的语气词，且没有包含任何业务信息
            """)
    ] + recent_msg
    res = classifier.invoke(prompt)
    updates = {"last_intent": res.intent}
    
    if res.intent == "HIGH_VALUE":
        updates["dialog_status"] = "VIP_SERVICE"
    
    elif res.intent == "NEED_CONSULTING":
        if current_status != "VIP_SERVICE":
            updates["dialog_status"] = "CONSULTING"
    
    return updates

#2 用户意图识别器，如果有意向，直接转人工拉群！
def sales_detector_node(state: AgentState):
    print("--- 🧠 Sales Detector: 客户上钩了吗？ ---")
    
    # 只看最近几轮，节省 token
    recent_msgs = state["messages"]
    
    classifier = llm.with_structured_output(IntentResult)
    prompt = [
        SystemMessage(content="""
        你现在处于销售谈判的关键阶段。AI顾问已经给出了初步建议。
        请分析用户最新的回复，判断意图：
        
        1. SALES_READY (上钩了):
           - 用户问细节（"学校在哪？" "学费多少？"）。
           - 用户表示赞同（"听起来不错" "可以考虑"）。
           - 用户问流程（"怎么申请？"）。
        2. 🆘 **TRANSFER_TO_HUMAN (转人工/直接报名/不耐烦)**:
           - 用户明确要求："我要报名", "电话联系", "人工客服".
           - 用户表现出负面情绪："太麻烦了", "我不懂", "直接找人跟我说".
            
        3. NEED_CONSULTING (还有疑虑/没兴趣):
           - 用户反驳或拒绝（"我不去韩国" "太贵了"）。
           - 用户提出新问题（"那欧洲呢？"）。
           - 用户沉默或顾左右而言他。
        """),
    ] + recent_msgs
    
    res = classifier.invoke(prompt)
    return {"last_intent": res.intent}