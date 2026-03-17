from state import BudgetInfo, BudgetPeriod, CustomerProfile, reduce_profile
from nodes.tools import search_products


def test_missing_fields_requires_readiness_only_for_middle_and_high_school():
    high_school_profile = CustomerProfile(
        educationStage="高中",
        academic_background="高二，均分85",
        destination_preference=["英国"],
        budget=BudgetInfo(amount=40, period=BudgetPeriod.TOTAL),
    )
    assert high_school_profile.missing_fields == ["abroad_readiness"]

    undergraduate_profile = CustomerProfile(
        educationStage="本科",
        academic_background="大三，均分82",
        destination_preference=["英国"],
        budget=BudgetInfo(amount=40, period=BudgetPeriod.TOTAL),
    )
    assert undergraduate_profile.missing_fields == []


def test_reduce_profile_merges_list_and_text_fields():
    old_profile = CustomerProfile(
        destination_preference=["美国"],
        abroad_readiness="需要过渡/暂缓",
        target_school="QS前100",
        target_major="商科",
    )
    new_profile = CustomerProfile(
        destination_preference=["英国", "美国"],
        abroad_readiness="直接出国",
        target_school="曼彻斯特大学",
        target_major="金融",
    )

    merged = reduce_profile(old_profile, new_profile)

    assert merged.destination_preference == ["美国", "英国"]
    assert merged.abroad_readiness == "直接出国"
    assert merged.target_school == "QS前100；曼彻斯特大学"
    assert merged.target_major == "商科；金融"


def test_reduce_profile_does_not_overwrite_budget_with_unknown_amount():
    old_profile = CustomerProfile(budget=BudgetInfo(amount=40, period=BudgetPeriod.TOTAL))
    new_profile = CustomerProfile(budget=BudgetInfo(amount=None, period=BudgetPeriod.UNKNOWN))

    merged = reduce_profile(old_profile, new_profile)

    assert merged.budget.amount == 40
    assert merged.budget.period == BudgetPeriod.TOTAL


def test_search_products_filters_by_abroad_readiness():
    direct_profile = CustomerProfile(
        educationStage="高中",
        academic_background="高二，均分85",
        destination_preference=["英国"],
        abroad_readiness="直接出国",
        budget=BudgetInfo(amount=40, period=BudgetPeriod.TOTAL),
    )
    direct_results = search_products(direct_profile)
    assert "马来西亚方向" in direct_results
    assert "国际本科2+2" not in direct_results

    transition_profile = CustomerProfile(
        educationStage="高中",
        academic_background="高二，均分85",
        destination_preference=["英国"],
        abroad_readiness="需要过渡/暂缓",
        budget=BudgetInfo(amount=40, period=BudgetPeriod.TOTAL),
    )
    transition_results = search_products(transition_profile)
    assert "国际本科2+2" in transition_results
    assert "马来西亚方向" not in transition_results

    domestic_profile = CustomerProfile(
        educationStage="高中",
        academic_background="高二，均分85",
        destination_preference=["香港"],
        abroad_readiness="坚决不出国",
        budget=BudgetInfo(amount=40, period=BudgetPeriod.TOTAL),
    )
    domestic_results = search_products(domestic_profile)
    assert "香港副学士" in domestic_results
    assert "国际本科2+2" not in domestic_results
