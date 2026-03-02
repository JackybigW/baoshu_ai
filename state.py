from pydantic import BaseModel, Field, field_validator
from enum import Enum
from typing import TypedDict, List, Annotated, Literal, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
import difflib
import re

# 核心组件

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

    @field_validator('user_role', mode='before')
    @classmethod
    def robust_role(cls, v):
        if v is None: return None
        s = str(v).strip()
        if s.lower() in ['none', 'null', 'unknown', '']: return None
        
        if "家长" in s or "Parent" in s or "父母" in s or "妈" in s or "爸" in s: return "家长"
        if "学生" in s or "Student" in s or "本人" in s or "我" in s: return "学生"
        
        return None # 无法识别就置空，别报错

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
    if not new_text:
        return old_text
    if not old_text:
        return new_text

    old_segments = [s.strip() for s in re.split(r'[;；]', old_text) if s.strip()]
    new_segments = [s.strip() for s in re.split(r'[;；]', new_text) if s.strip()]

    result_list = list(old_segments)
    
    for new_seg in new_segments:
        if new_seg in result_list:
            continue
            
        is_redundant = False
        for old_seg in result_list:
            if new_seg in old_seg:
                is_redundant = True
                break
            
        if not is_redundant:
            result_list.append(new_seg)

    return "；".join(result_list)

def reduce_profile(old_data: Optional[CustomerProfile], new_data: Optional[CustomerProfile]) -> CustomerProfile:
    """
    【工业级合并策略 V2.0】
    """
    if new_data is None: return old_data
    if old_data is None: return new_data

    merged = old_data.model_copy()
    
    if new_data.user_role is not None: 
        merged.user_role = new_data.user_role
        
    if new_data.educationStage is not None: 
        merged.educationStage = new_data.educationStage
        
    if new_data.destination_preference is not None: 
        merged.destination_preference = new_data.destination_preference
    
    if new_data.budget.amount != -1: 
        merged.budget.amount = new_data.budget.amount
        
    if new_data.budget.period != BudgetPeriod.UNKNOWN: 
        merged.budget.period = new_data.budget.period

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
        
        valid_keys = [member.value for member in IntentType]
        if clean_v in valid_keys: return clean_v
        
        matches = difflib.get_close_matches(clean_v, valid_keys, n=1, cutoff=0.8)
        if matches: return matches[0]
        
        if "TRANSFER" in clean_v or "HUMAN" in clean_v: return "TRANSFER_TO_HUMAN"
        if "SALES" in clean_v: return "SALES_READY"
        if "HIGH" in clean_v: return "HIGH_VALUE"
        if "ART" in clean_v: return "ART_CONSULTING"
        if "LOW" in clean_v or "BUDGET" in clean_v: return "LOW_BUDGET"
        
        if "DECISION" in clean_v or "COMPARE" in clean_v or "OFFER" in clean_v:
            return "DECISION_SUPPORT"

        return "NEED_CONSULTING"
