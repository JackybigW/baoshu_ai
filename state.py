
from pydantic import BaseModel,Field, model_validator,field_validator
from enum import Enum
from typing import TypedDict, List, Annotated, Literal, Optional
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage, ToolMessage
from langgraph.graph.message import add_messages
import difflib
import re
#核心组件

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
    #--- 0. 用户角色 ----
    user_role: Optional[Literal["学生", "家长"]] = Field(
        default=None,
        description="""
        判断用户是学生本人还是家长。规则：
        - 出现第一人称("我读高三", "我想去", "我均分") -> '学生'
        - 出现第三人称("孩子高三", "我女儿", "我儿子") -> '家长'
        - 都没出现 -> 'None'
        """
    )
    #--- 1. 当前学历 ----
    educationStage: Optional[Literal["小学","初中","职高","中专", "高中","本科","大专","研究生"]]= Field(
        default=None,
        description="用户【当前】正在读或已毕业的最高学历。注意：不要填用户【想去读】的学历。"
    )
    #--- 2. 留学预算 ----
    budget: BudgetInfo = Field(default_factory=BudgetInfo)
    #--- 3. 出国意向 ----
    destination_preference: Optional[Literal["境外方向","境内/港澳方向"]] = Field(
        default=None,
        description="用户的留学或工作的目的地偏好，'境外方向'代表欧美澳加日韩以及东南亚等海外国家；'境内/港澳方向'代表只考虑香港、澳门、或者内地"
    )
    #--- 4. 学术背景&经历 ----
    academic_background: Optional[str] = Field(
        default = None,
        description="""
        用户的学术背景详情。包括但不限于:
        - 高中生：平时成绩, 排名, 科目偏好
        - 大学生: 专业(Major), 均分（GPA）， 院校层次(985/211/一本/二本/民办)，实习经历，科研经历
        - 专升本： 专科成绩，本科成绩，是否有学位证，实习经历
        - 职高，中专生: 在校成绩，专业，实习经历
        """
    )
    #--- 5. 语言水平 ---
    language_level: Optional[str] = Field(
        default=None,
        description="用户可用于留学的外语能力。包含但不限于：雅思，托福，PTE等英语成绩，日语成绩，韩语Topik成绩，欧洲小语种等"
    )
    
    @field_validator('user_role', mode='before')
    @classmethod
    def robust_role(cls, v):
        if not v or str(v).lower() in ['none', 'null', 'unknown']: return None
        s = str(v)
        if "家长" in s or "Parent" in s or "父母" in s: return "家长"
        if "学生" in s or "Student" in s or "本人" in s or "我" in s: return "学生"
        return None # 无法识别就置空，别报错

    @field_validator('educationStage', mode='before')
    @classmethod
    def robust_stage(cls, v):
        if not v: return None
        s = str(v)
        # 关键词映射
        if "研" in s or "Master" in s or "PhD" in s: return "研究生"
        if "本" in s or "Bachelor" in s or "大" in s: return "本科" # 大一/大二 -> 本科
        if "专" in s and "大" not in s: return "中专" # 防止把大专识别成中专，这里简单处理
        if "大专" in s: return "大专"
        if "高" in s: return "高中"
        if "初" in s: return "初中"
        if "小" in s: return "小学"
        return v # 让他尝试匹配 Literal，匹配不上再报错（或者你可以 return None 兜底）

    @field_validator('destination_preference', mode='before')
    @classmethod
    def robust_destination(cls, v):
        """
        这里是最容易出错的，必须暴力映射。
        LLM 可能会输出：'国外', '欧美', '香港', 'HK', '英美澳加'
        """
        if not v: return None
        s = str(v).upper()
        
        # 1. 优先判断境内/港澳 (关键词少)
        # 只要出现 港、澳、内地、国内、CN、HK、MO -> 归为境内
        if any(k in s for k in ["港", "澳", "内地", "国内", "CN", "HK", "MO", "MACAU"]):
            return "境内/港澳方向"
            
        # 2. 其他只要不是空的，基本就是境外 (欧美澳加日韩...)
        # 关键词：外、美、英、澳、加、日、韩、欧
        if any(k in s for k in ["外", "美", "英", "澳", "加", "日", "韩", "欧", "OVERSEAS"]):
            return "境外方向"
            
        return None # 无法判断
    
    @field_validator('user_role', 'educationStage', 'destination_preference', mode='before')
    @classmethod
    def parse_none_string(cls, v):
        # 如果 LLM 返回了字符串类型的 "None", "null" 或者空字符串，强制转为 Python None
        if isinstance(v, str) and v.lower() in ('none', 'null', '', 'n/a', 'unknown'):
            return None
        return v
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
    LOW_BUDGET = "LOW_BUDGET"         # 低预算客户 -> 单独处理
    NEED_CONSULTING = "NEED_CONSULTING" # 正常咨询 -> 提取+问询
    GREETING = "GREETING"             # 打招呼 -> 破冰
    CHIT_CHAT = "CHIT_CHAT"           # 闲聊 -> 敷衍
    DECISION_SUPPORT = "DECISION_SUPPORT"  # 决策辅助/Offer对比/路径PK

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    profile: Annotated[CustomerProfile,reduce_profile]
    # 增加这个，用于记录是否已经给过初步方案
    # 如果是 True，说明我们进入了"博弈阶段"
    has_proposed_solution: bool 
    dialog_status: Literal["START", "PROFILING", "PERSUADING", "FINISHED"]
    last_intent: Optional[IntentType]
    # 低预算标记，由 classifier 设置
    is_low_budget: bool
    
class IntentResult(BaseModel):
    intent: IntentType

    @field_validator('intent', mode='before')
    @classmethod
    def robust_intent_validator(cls, v):
        # 1. 防御性编程：如果不是字符串（比如None），直接抛错或返回默认
        if not isinstance(v, str):
            return "NEED_CONSULTING"
            
        # 2. 基础清洗：转大写，去前后空格
        clean_v = v.upper().strip()
        
        # 3. 尝试直接匹配 (最快)
        valid_keys = [member.value for member in IntentType]
        if clean_v in valid_keys:
            return clean_v
            
        # 4. 模糊匹配 (Fuzzy Match) - 解决 Typo 的神器
        # n=1 表示只找最像的那个，cutoff=0.6 表示相似度至少60%
        matches = difflib.get_close_matches(clean_v, valid_keys, n=1, cutoff=0.6)
        
        if matches:
            # 可以在日志里记一下，方便以后优化 Prompt
            # print(f"🔧 自动修复 Typo: {v} -> {matches[0]}")
            return matches[0]
            
        # 5. 特殊规则兜底 (处理 LLM 的顽固幻觉)
        # 比如它非要输出 "CONSULTING_ONLY" 或者 "TRANSFER"
        if "TRANSFER" in clean_v or "HUMAN" in clean_v:
            return "TRANSFER_TO_HUMAN"
        if "SALES" in clean_v:
            return "SALES_READY"
        if "HIGH" in clean_v:
            return "HIGH_VALUE"
        if "ART" in clean_v:
            return "ART_CONSULTING"
        if "LOW" in clean_v or "BUDGET" in clean_v:
            return "LOW_BUDGET"
        
        # 🔥 新增：DECISION_SUPPORT 的模糊捕获
        # 防止 LLM 输出 "DECISION", "OFFER_PK", "COMPARE" 等变体
        if "DECISION" in clean_v or "COMPARE" in clean_v or "OFFER" in clean_v:
            return "DECISION_SUPPORT"
            
        # 6. 实在没救了，为了不崩系统，返回一个默认的安全选项
        # (通常归类为普通咨询是最安全的，因为 Extractor 会接住它)
        return "NEED_CONSULTING"
    