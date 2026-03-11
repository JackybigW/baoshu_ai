import os
import sys

from langchain_core.messages import HumanMessage

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from state import CustomerProfile, IntentResult, IntentType
from nodes import perception


def test_intent_validator_does_not_map_generic_budget_labels_to_low_budget():
    result = IntentResult(intent="BUDGET_SENSITIVE")
    assert result.intent == IntentType.NEED_CONSULTING


def test_classifier_corrects_high_budget_low_budget_misfire(monkeypatch):
    class FakeClassifier:
        def invoke(self, _prompt):
            return IntentResult(intent="LOW_BUDGET")

    class FakeLLM:
        def with_structured_output(self, _schema):
            return FakeClassifier()

    monkeypatch.setattr(perception, "llm", FakeLLM())

    profile = CustomerProfile()
    profile.budget.amount = 40

    state = {
        "messages": [HumanMessage(content="预算40个是总共")],
        "profile": profile,
        "last_intent": None,
        "dialog_status": "CONSULTING",
    }

    result = perception.classifier_node(state)
    assert result["last_intent"] == IntentType.NEED_CONSULTING
