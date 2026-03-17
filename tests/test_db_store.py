import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from db.store import _json_ready, _text_ready
from state import CustomerProfile, IntentType


def test_json_ready_handles_pydantic_model():
    profile = CustomerProfile(
        user_role="学生",
        educationStage="本科",
        destination_preference=["美国"],
    )
    profile.budget.amount = 40
    payload = _json_ready(profile)

    assert payload["user_role"] == "学生"
    assert payload["budget"]["amount"] == 40


def test_text_ready_handles_enum():
    assert _text_ready(IntentType.SALES_READY) == "SALES_READY"
