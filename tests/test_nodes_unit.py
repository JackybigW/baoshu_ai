from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from nodes import consultants, perception
from nodes.tools import search_products
from state import BudgetInfo, BudgetPeriod, CustomerProfile, IntentType


class RecordingStructuredLLM:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def with_structured_output(self, _schema):
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        return self.response


class RecordingChatLLM:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return self.response


class RecordingToolLLM:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def bind_tools(self, _tools):
        outer = self

        class BoundLLM:
            def invoke(self, messages):
                outer.calls.append(messages)
                return outer.response

        return BoundLLM()


def test_classifier_node_forces_low_budget_on_debt_keyword(monkeypatch):
    stub_llm = RecordingStructuredLLM(SimpleNamespace(intent="NEED_CONSULTING"))
    monkeypatch.setattr(perception, "llm", stub_llm)

    state = {
        "messages": [HumanMessage(content="我现在还在还债，预算也不多")],
        "profile": CustomerProfile(),
        "last_intent": None,
        "dialog_status": "START",
    }

    result = perception.classifier_node(state)

    assert result["last_intent"] == "LOW_BUDGET"
    assert result["dialog_status"] == "CONSULTING"


def test_classifier_node_keeps_sticky_intent(monkeypatch):
    stub_llm = RecordingStructuredLLM(SimpleNamespace(intent="NEED_CONSULTING"))
    monkeypatch.setattr(perception, "llm", stub_llm)

    profile = CustomerProfile()
    profile.budget.amount = 50
    profile.budget.period = BudgetPeriod.TOTAL
    state = {
        "messages": [HumanMessage(content="先聊聊方案")],
        "profile": profile,
        "last_intent": "HIGH_VALUE",
        "dialog_status": "CONSULTING",
    }

    result = perception.classifier_node(state)

    assert result["last_intent"] == "HIGH_VALUE"
    assert result["dialog_status"] == "VIP_SERVICE"


def test_extractor_node_merges_profile_via_reduce_profile(monkeypatch):
    new_data = CustomerProfile(
        destination_preference=["英国", "美国"],
        abroad_readiness="直接出国",
        target_school="曼彻斯特大学",
        target_major="金融",
        academic_background="GPA 3.7",
    )
    stub_llm = RecordingStructuredLLM(new_data)
    monkeypatch.setattr(perception, "llm", stub_llm)

    current_profile = CustomerProfile(
        user_role="学生",
        educationStage="本科",
        destination_preference=["美国"],
        target_school="QS前100",
        academic_background="211本科",
    )
    current_profile.budget.amount = 40
    current_profile.budget.period = BudgetPeriod.TOTAL

    state = {
        "messages": [
            AIMessage(content="你想去哪些国家，目标学校和专业是什么？"),
            HumanMessage(content="想去英国，也保留美国，目标曼大金融"),
        ],
        "profile": current_profile,
    }

    result = perception.extractor_node(state)
    profile = result["profile"]

    assert profile.destination_preference == ["美国", "英国"]
    assert profile.abroad_readiness == "直接出国"
    assert profile.target_school == "QS前100；曼彻斯特大学"
    assert profile.target_major == "金融"
    assert profile.academic_background == "211本科；GPA 3.7"


def test_interviewer_node_asks_readiness_for_high_school(monkeypatch):
    stub_chat = RecordingChatLLM(AIMessage(content="先确认路线|||再看细方案"))
    monkeypatch.setattr(consultants, "llm_chat", stub_chat)

    profile = CustomerProfile(
        user_role="家长",
        educationStage="高中",
        academic_background="高二，均分85",
        destination_preference=["英国"],
    )
    profile.budget.amount = 50
    profile.budget.period = BudgetPeriod.TOTAL

    state = {
        "messages": [HumanMessage(content="先看看有什么路可以走")],
        "profile": profile,
    }

    result = consultants.interviewer_node(state)
    prompt = stub_chat.calls[0][0].content

    assert "**abroad_readiness**" in prompt
    assert "先在国内/港澳过渡" in prompt
    assert "家长您好" in prompt
    assert [msg.content for msg in result["messages"]] == ["先确认路线", "再看细方案"]


def test_interviewer_node_reasks_background_when_score_missing(monkeypatch):
    stub_chat = RecordingChatLLM(AIMessage(content="把成绩和语言一起发我"))
    monkeypatch.setattr(consultants, "llm_chat", stub_chat)

    profile = CustomerProfile(
        user_role="学生",
        educationStage="本科",
        academic_background="本科在读，商科方向",
    )
    profile.budget.amount = 40
    profile.budget.period = BudgetPeriod.TOTAL

    state = {
        "messages": [HumanMessage(content="我先了解一下")],
        "profile": profile,
    }

    consultants.interviewer_node(state)
    prompt = stub_chat.calls[0][0].content

    assert "**academic_background**" in prompt
    assert "语言准备情况" in prompt


def test_consultant_node_sales_mode_keeps_tool_call(monkeypatch):
    response = SimpleNamespace(
        content="",
        tool_calls=[{"id": "tool_1", "name": "summon_specialist_tool", "args": {}}],
        id="response_1",
    )
    stub_tool_llm = RecordingToolLLM(response)
    monkeypatch.setattr(consultants, "llm_chat", stub_tool_llm)
    monkeypatch.setattr(consultants, "search_products", lambda _profile: "mocked context")

    profile = CustomerProfile(
        educationStage="高中",
        destination_preference=["香港"],
        abroad_readiness="坚决不出国",
        academic_background="高三，均分88",
    )
    profile.budget.amount = 40
    profile.budget.period = BudgetPeriod.TOTAL

    state = {
        "messages": [HumanMessage(content="这个项目靠谱吗")],
        "profile": profile,
        "last_intent": IntentType.SALES_READY,
    }

    result = consultants.consultant_node(state)
    prompt = stub_tool_llm.calls[0][0].content

    assert "专门负责国内和港澳申请的" in prompt
    assert result["messages"][-1].tool_calls[0]["name"] == "summon_specialist_tool"
    assert len(result["messages"]) == 2


def test_consultant_node_normal_mode_uses_search_results(monkeypatch):
    response = SimpleNamespace(content="方案A|||你能接受吗", tool_calls=[], id="response_2")
    stub_tool_llm = RecordingToolLLM(response)
    monkeypatch.setattr(consultants, "llm_chat", stub_tool_llm)
    monkeypatch.setattr(consultants, "search_products", lambda _profile: "共匹配到 1 个方案：英国方向")

    profile = CustomerProfile(
        user_role="学生",
        educationStage="本科",
        destination_preference=["英国", "香港"],
        academic_background="GPA 3.6",
        abroad_readiness="直接出国",
    )
    profile.budget.amount = 45
    profile.budget.period = BudgetPeriod.ANNUAL

    state = {
        "messages": [HumanMessage(content="给我一个初步方案")],
        "profile": profile,
        "last_intent": IntentType.NEED_CONSULTING,
    }

    result = consultants.consultant_node(state)
    prompt = stub_tool_llm.calls[0][0].content

    assert "共匹配到 1 个方案：英国方向" in prompt
    assert "英国、香港" in prompt
    assert [msg.content for msg in result["messages"]] == ["方案A", "你能接受吗"]


def test_search_products_supports_annual_budget_filter():
    profile = CustomerProfile(
        educationStage="本科",
        academic_background="GPA 3.6",
        destination_preference=["英国"],
        abroad_readiness="直接出国",
        budget=BudgetInfo(amount=45, period=BudgetPeriod.ANNUAL),
    )

    results = search_products(profile)

    assert "英国方向" in results
    assert "新加坡方向" not in results
