import ast
import difflib
import re
from enum import Enum
from typing import Annotated, List, Literal, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, field_validator

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
    destination_preference: Optional[List[str]] = Field(
        default=None,
        description="用户明确提到的意向国家和地区，如没有明确提及返回None。"
    )
    
    abroad_readiness: Optional[Literal["直接出国", "需要过渡/暂缓", "坚决不出国"]] = Field(
    default=None,
    description="""
    判断用户出国的心理准备度和时间线：
    - '坚决不出国'：明确表示不想去国外，或者只接受国内/港澳。
    - '需要过渡/暂缓'：觉得孩子太小、成绩不够、想先在国内读预科/中外合作办学，后面等ready了再出国。
    - '直接出国'：明确要申请海外学校，不在国内过度，直接出国。
    """
    )
    target_school: Optional[str] = Field(
        default=None,
        description="用户的留学目标院校范围，此处可以填具体的院校名称，也可以填排名的要求，如没有明确提及返回None。"
    )
    
    target_major: Optional[str] = Field(
        default = None,
        description ="用户的留学目标专业"
    )
    
    
    academic_background: Optional[str] = Field(
        default=None,
        description="用户的学术背景详情"
    )
    language_level: Optional[str] = Field(
        default=None,
        description="""
        用户可用于留学的外语能力，雅思托福，小语种，或是别的语言考试成绩 level.
        如用户说自己xx不好，可以填入此栏，如果用户完全没提及语言能力，返回None。
        """
    )

    @field_validator('educationStage', mode='before')
    @classmethod
    def robust_stage(cls, v):
        if v is None:
            return None

        s = str(v).strip().replace("'", "").replace('"', "")
        if s.lower() in ['none', 'null', 'n/a', 'unknown', '', '无']:
            return None

        allowed = ["小学","初中","职高","中专", "高中","本科","大专","研究生"]
        if s in allowed:
            return s

        stage_patterns = [
            ("研究生", [r"研究生", r"硕士", r"博士", r"\bmaster\b", r"\bphd\b", r"研[一二三]", r"硕[一二三]", r"博[一二三]"]),
            ("本科", [r"本科", r"\bbachelor\b", r"大[一二三四五]"]),
            ("大专", [r"大专", r"专科"]),
            ("中专", [r"中专", r"技校"]),
            ("职高", [r"职高"]),
            ("高中", [r"高中", r"高[一二三]"]),
            ("初中", [r"初中", r"初[一二三]"]),
            ("小学", [r"小学", r"小[一二三四五六]"]),
        ]
        for normalized, patterns in stage_patterns:
            if any(re.search(pattern, s, re.IGNORECASE) for pattern in patterns):
                return normalized

        return None

    @field_validator('user_role', mode='before')
    @classmethod
    def robust_role(cls, v):
        if v is None:
            return None
        s = str(v).strip()
        if s.lower() in ['none', 'null', 'unknown', '']:
            return None

        if any(token in s for token in ["家长", "父母", "妈妈", "爸爸", "母亲", "父亲"]):
            return "家长"
        if any(token in s for token in ["学生", "本人", "我"]) or s.lower() in ["student", "self"]:
            return "学生"

        return None

    @field_validator('destination_preference', mode='before')
    @classmethod
    def robust_destination(cls, v):
        if not v:
            return None

        raw_items: List[str] = []
        if isinstance(v, list):
            raw_items = [str(item).strip() for item in v]
        else:
            text = str(v).strip()
            if text.lower() in ['none', 'null', 'unknown', '', '无']:
                return None

            if text.startswith("[") and text.endswith("]"):
                try:
                    parsed = ast.literal_eval(text)
                except (ValueError, SyntaxError):
                    parsed = None
                if isinstance(parsed, list):
                    raw_items = [str(item).strip() for item in parsed]

            if not raw_items:
                raw_items = [item.strip() for item in re.split(r'[，,、/|；;]+', text) if item.strip()]

        cleaned: List[str] = []
        for item in raw_items:
            token = item.strip().strip('"').strip("'")
            if not token or token.lower() in ['none', 'null', 'unknown', '无']:
                continue
            if token not in cleaned:
                cleaned.append(token)

        return cleaned or None

    @property
    def requires_abroad_readiness(self) -> bool:
        return self.educationStage in {"初中", "高中"}

    @property
    def missing_fields(self) -> List[str]:
        missing = []
        if not self.educationStage:
            missing.append("educationStage")
        elif not self.academic_background:
            missing.append("academic_background")
        elif self.budget.amount == -1 or self.budget.period == BudgetPeriod.UNKNOWN:
            missing.append("budget")
        elif not self.destination_preference:
            missing.append("destination_preference")
        elif self.requires_abroad_readiness and not self.abroad_readiness:
            missing.append("abroad_readiness")
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
            
        # 优化：避免长文本下 O(N^2) 的性能问题，只比较较短的片段或直接全文本查找
        if not any(new_seg in old_seg for old_seg in result_list):
            result_list.append(new_seg)

    return "；".join(result_list)


def merge_list_fields(old_items: Optional[List[str]], new_items: Optional[List[str]]) -> Optional[List[str]]:
    if not new_items:
        return old_items
    if not old_items:
        return list(new_items)

    merged = list(old_items)
    for item in new_items:
        if item and item not in merged:
            merged.append(item)
    return merged

def reduce_profile(old_data: Optional[CustomerProfile], new_data: Optional[CustomerProfile]) -> CustomerProfile:
    """
    【工业级合并策略 V2.0】
    """
    if new_data is None:
        return old_data or CustomerProfile()
    if old_data is None:
        return new_data

    merged = old_data.model_copy()
    
    if new_data.user_role is not None: 
        merged.user_role = new_data.user_role
        
    if new_data.educationStage is not None: 
        merged.educationStage = new_data.educationStage
        
    merged.destination_preference = merge_list_fields(old_data.destination_preference, new_data.destination_preference)

    if new_data.abroad_readiness is not None:
        merged.abroad_readiness = new_data.abroad_readiness
    
    if new_data.budget.amount != -1: 
        merged.budget.amount = new_data.budget.amount
        
    if new_data.budget.period != BudgetPeriod.UNKNOWN: 
        merged.budget.period = new_data.budget.period

    merged.academic_background = merge_text_fields(old_data.academic_background, new_data.academic_background)
    merged.language_level = merge_text_fields(old_data.language_level, new_data.language_level)
    merged.target_school = merge_text_fields(old_data.target_school, new_data.target_school)
    merged.target_major = merge_text_fields(old_data.target_major, new_data.target_major)

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

        low_budget_aliases = {
            "LOW_BUDGET",
            "LOW-BUDGET",
            "LOW BUDGET",
            "TIGHT_BUDGET",
            "TIGHT-BUDGET",
            "BUDGET_LIMITED",
            "BUDGET-LIMITED",
            "BUDGET LIMITED",
        }
        if clean_v in low_budget_aliases:
            return "LOW_BUDGET"
        
        if "DECISION" in clean_v or "COMPARE" in clean_v or "OFFER" in clean_v:
            return "DECISION_SUPPORT"

        return "NEED_CONSULTING"
