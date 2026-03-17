from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


OUTPUT_PATH = Path(__file__).resolve().parent / "golden_dataset.json"


def budget(amount: Optional[int] = None, period: str = "UNKNOWN") -> Dict[str, Any]:
    return {"amount": amount, "period": period}


def profile(**kwargs: Any) -> Dict[str, Any]:
    base = {
        "user_role": None,
        "educationStage": None,
        "budget": budget(),
        "destination_preference": None,
        "abroad_readiness": None,
        "target_school": None,
        "target_major": None,
        "academic_background": None,
        "language_level": None,
    }
    base.update(kwargs)
    return base


def unique_items(items: List[str]) -> List[str]:
    ordered: List[str] = []
    for item in items:
        if item not in ordered:
            ordered.append(item)
    return ordered


def build_special_cases() -> List[Dict[str, Any]]:
    return [
        {
            "tags": ["annual_budget", "anxious_parent"],
            "input": {
                "last_ai_msg": "把孩子年级、预算周期和想去的国家说清楚。",
                "last_user_msg": "孩子高一，校内均分86，一年预算25万人民币，考虑澳洲商科。",
            },
            "expected": profile(
                user_role="家长",
                educationStage="高中",
                budget=budget(25, "YEAR"),
                destination_preference=["澳洲"],
                target_major="商科",
                academic_background="高一，校内均分86",
            ),
        },
        {
            "tags": ["annual_budget", "top_student", "unit_confusion", "currency_conversion"],
            "input": {
                "last_ai_msg": "预算如果是按年说，直接给我年预算。",
                "last_user_msg": "我大二，GPA 3.6，一年预算3万美金，想去美国读传媒。",
            },
            "expected": profile(
                user_role="学生",
                educationStage="本科",
                budget=budget(22, "YEAR"),
                destination_preference=["美国"],
                target_major="传媒",
                academic_background="大二；GPA 3.6",
            ),
        },
        {
            "tags": ["hkd_budget", "mixed_scores", "currency_conversion"],
            "input": {
                "last_ai_msg": "把背景、预算、目标项目一次说完。",
                "last_user_msg": "我本科会计，均分83，总共40万港币，想去香港读一年传媒硕士。",
            },
            "expected": profile(
                user_role="学生",
                educationStage="本科",
                budget=budget(37, "TOTAL"),
                destination_preference=["香港"],
                target_major="传媒",
                academic_background="本科会计；均分83",
            ),
        },
        {
            "tags": ["overseas_inference", "anxious_parent"],
            "input": {
                "last_ai_msg": "如果已经在海外读书，也把目前学校体系告诉我。",
                "last_user_msg": "孩子现在在英国读A-level，想本科继续留英国冲G5，预算还没细算。",
            },
            "expected": profile(
                user_role="家长",
                educationStage="高中",
                budget=budget(70, "YEAR"),
                destination_preference=["英国"],
                target_school="G5",
                academic_background="英国A-level在读",
            ),
        },
        {
            "tags": ["overseas_inference", "top_student"],
            "input": {
                "last_ai_msg": "你如果已经在海外，就按现在的在读国家来描述。",
                "last_user_msg": "我现在在美国读11年级，想继续在美国本科CS，预算家里还没细说。",
            },
            "expected": profile(
                user_role="学生",
                educationStage="高中",
                budget=budget(90, "YEAR"),
                destination_preference=["美国"],
                target_major="CS",
                academic_background="美国11年级在读",
            ),
        },
        {
            "tags": ["hk_macao_only", "no_abroad"],
            "input": {
                "last_ai_msg": "先说能不能接受真正出国，还是只看港澳。",
                "last_user_msg": "孩子高三，只接受港澳，不出国，学校先看港大和澳门大学。",
            },
            "expected": profile(
                user_role="家长",
                educationStage="高中",
                destination_preference=["香港", "澳门"],
                abroad_readiness="坚决不出国",
                target_school="港大；澳门大学",
                academic_background="高三",
            ),
        },
        {
            "tags": ["transition_needed", "low_cognition"],
            "input": {
                "last_ai_msg": "如果不想马上出去，也直接说。",
                "last_user_msg": "孩子初三，不想太早出去，最好先在国内国际部或者港澳过渡两年。",
            },
            "expected": profile(
                user_role="家长",
                educationStage="初中",
                destination_preference=["港澳"],
                abroad_readiness="需要过渡/暂缓",
                academic_background="初三",
            ),
        },
        {
            "tags": ["desire_vs_current", "cross_major"],
            "input": {
                "last_ai_msg": "注意区分你现在在哪个阶段，和你未来想读什么。",
                "last_user_msg": "我现在大二，想出去读硕士，新加坡和英国都看，方向还是数据分析。",
            },
            "expected": profile(
                user_role="学生",
                educationStage="本科",
                destination_preference=["新加坡", "英国"],
                target_major="数据分析",
                academic_background="大二",
            ),
        },
        {
            "tags": ["school_only", "mixed_scores"],
            "input": {
                "last_ai_msg": "学校和专业都可以直接提。",
                "last_user_msg": "想冲UCL或者港大的传媒硕士，我本科新闻，均分84。",
            },
            "expected": profile(
                user_role="学生",
                educationStage="本科",
                target_school="UCL；港大",
                target_major="传媒",
                academic_background="本科新闻；均分84",
            ),
        },
        {
            "tags": ["destination_exclusion", "budget_sensitive"],
            "input": {
                "last_ai_msg": "哪些国家可以，哪些一定不要，也直接说。",
                "last_user_msg": "除了美国都行，英国香港新加坡都可以，预算40万总共。",
            },
            "expected": profile(
                user_role="学生",
                budget=budget(40, "TOTAL"),
                destination_preference=["英国", "香港", "新加坡"],
            ),
        },
        {
            "tags": ["jpy_budget", "currency_conversion"],
            "input": {
                "last_ai_msg": "把预算和目标国家一起说，我会自己换成人民币万。",
                "last_user_msg": "总预算大概700万日元，想去日本读经营学修士。",
            },
            "expected": profile(
                budget=budget(34, "TOTAL"),
                destination_preference=["日本"],
                target_major="经营学",
            ),
        },
        {
            "tags": ["debt_not_budget", "budget_sensitive"],
            "input": {
                "last_ai_msg": "债务和留学预算分开说，我只记留学那部分。",
                "last_user_msg": "家里欠款50万，但我真正留学预算总共只有15万，想去马来西亚或者香港。",
            },
            "expected": profile(
                user_role="学生",
                budget=budget(15, "TOTAL"),
                destination_preference=["马来西亚", "香港"],
            ),
        },
        {
            "tags": ["noisy_asr", "low_cognition"],
            "input": {
                "last_ai_msg": "哪怕说得乱一点也行，我会帮你拆。",
                "last_user_msg": "我二本马上毕业 想去英国 预算20w major media。",
            },
            "expected": profile(
                user_role="学生",
                educationStage="本科",
                budget=budget(20, "UNKNOWN"),
                destination_preference=["英国"],
                target_major="传媒",
                academic_background="二本马上毕业",
            ),
        },
        {
            "tags": ["art_portfolio", "target_conflict"],
            "input": {
                "last_ai_msg": "如果是作品集方向，要把具体方向说出来。",
                "last_user_msg": "我做交互和作品集方向，学校想申UAL或者RCA。",
            },
            "expected": profile(
                user_role="学生",
                target_school="UAL；RCA",
                target_major="交互设计",
                academic_background="作品集方向",
            ),
        },
        {
            "tags": ["overwrite_conflict", "profile_merge"],
            "input": {
                "last_ai_msg": "你之前说金融，现在确认一下，是不是改成传媒了？",
                "last_user_msg": "对，改传媒，不读金融了。",
                "current_profile": profile(
                    user_role="学生",
                    educationStage="本科",
                    target_major="金融",
                    academic_background="大三",
                ),
            },
            "expected": profile(
                user_role="学生",
                educationStage="本科",
                target_major="传媒",
                academic_background="大三",
            ),
        },
        {
            "tags": ["overwrite_conflict", "profile_merge", "destination_exclusion"],
            "input": {
                "last_ai_msg": "之前记成美国，现在你是只看香港和新加坡，对吗？",
                "last_user_msg": "对，美国不看了，就香港和新加坡。",
                "current_profile": profile(
                    user_role="学生",
                    destination_preference=["美国"],
                    target_major="商科",
                ),
            },
            "expected": profile(
                user_role="学生",
                destination_preference=["香港", "新加坡"],
                target_major="商科",
            ),
        },
        {
            "tags": ["krw_budget", "currency_conversion"],
            "input": {
                "last_ai_msg": "预算如果不是人民币，也直接说原币种。",
                "last_user_msg": "我本科传媒，家里能给我2800万韩币，想去韩国读一年制传媒硕士。",
            },
            "expected": profile(
                user_role="学生",
                educationStage="本科",
                budget=budget(15, "TOTAL"),
                destination_preference=["韩国"],
                target_major="传媒",
                academic_background="本科传媒",
            ),
        },
        {
            "tags": ["negative_language", "mixed_scores"],
            "input": {
                "last_ai_msg": "语言好坏都直接说，不要省略。",
                "last_user_msg": "英语巨烂，雅思还没考，四六级也就刚过，本科均分81。",
            },
            "expected": profile(
                user_role="学生",
                educationStage="本科",
                academic_background="本科；均分81",
                language_level="英语巨烂；雅思还没考；四六级刚过",
            ),
        },
        {
            "tags": ["country_abbrev", "top_student"],
            "input": {
                "last_ai_msg": "国家缩写我也能识别，你直接写。",
                "last_user_msg": "想去美国/英国/HK读BA，budget 50w total，本科经济学，GPA 3.7。",
            },
            "expected": profile(
                user_role="学生",
                educationStage="本科",
                budget=budget(50, "TOTAL"),
                destination_preference=["美国", "英国", "香港"],
                target_major="BA",
                academic_background="本科经济学；GPA 3.7",
            ),
        },
        {
            "tags": ["annual_budget", "unit_confusion", "school_only"],
            "input": {
                "last_ai_msg": "预算是按年还是总额，学校偏好也一起说。",
                "last_user_msg": "高三均分88，去新二，预算£15k/year能走吗？",
            },
            "expected": profile(
                user_role="学生",
                educationStage="高中",
                budget=budget(14, "YEAR"),
                target_school="新二",
                academic_background="高三；均分88",
            ),
        },
    ]


def build_cases() -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    variant_bank = [
        {
            "stage": "高中",
            "background": "高二，校内均分82",
            "language": "雅思还没考",
            "dest": ["英国"],
            "school": "QS前100",
            "major": "经济",
            "budget_foreign": "5万美金",
            "budget_amount": 36,
            "budget_period": "TOTAL",
            "readiness": "直接出国",
            "parent_word": "孩子",
        },
        {
            "stage": "本科",
            "background": "双非本科大三，均分85，GPA 3.2",
            "language": "托福没考，四级刚过",
            "dest": ["美国", "香港"],
            "school": "TOP 50",
            "major": "CS",
            "budget_foreign": "30万人民币",
            "budget_amount": 30,
            "budget_period": "TOTAL",
            "readiness": None,
            "parent_word": "孩子",
        },
        {
            "stage": "本科",
            "background": "211本科金融，均分88",
            "language": "雅思6.5",
            "dest": ["新加坡", "香港"],
            "school": "新二或港前三",
            "major": "金融",
            "budget_foreign": "5000镑",
            "budget_amount": 5,
            "budget_period": "TOTAL",
            "readiness": None,
            "parent_word": "孩子",
        },
        {
            "stage": "研究生",
            "background": "国内硕一在读，做过两段科研",
            "language": "托福102，GRE 323",
            "dest": ["美国"],
            "school": "藤校或专排前10",
            "major": "数据科学",
            "budget_foreign": "8万欧元",
            "budget_amount": 62,
            "budget_period": "TOTAL",
            "readiness": None,
            "parent_word": "孩子",
        },
    ]

    for idx, variant in enumerate(variant_bank, start=1):
        group_offset = (idx - 1) * 25

        mixed_major = ["计算机", "金融", "传媒", "生物统计"][idx - 1]
        own_major = ["传媒", "CS", "心理学", "交互设计"][idx - 1]
        mom_major = ["金融", "会计", "法学", "经济学"][idx - 1]
        anxious_sparse_user = [
            "救命，我家孩子高二还能去哪？",
            "二本孩子还能申吗？",
            "本科都快毕业了还有救吗？",
            "读研读一半想出去，还有路吗？",
        ][idx - 1]
        anxious_sparse_expected = [
            profile(user_role="家长", educationStage="高中", academic_background="高二"),
            profile(user_role="家长", educationStage="本科", academic_background="二本"),
            profile(user_role="家长", educationStage="本科", academic_background="本科快毕业"),
            profile(user_role="家长", educationStage="研究生", academic_background="读研读一半"),
        ][idx - 1]
        current_dest = [["美国"], ["英国"], ["香港"], ["美国", "加拿大"]][idx - 1]
        new_dest = [["英国"], ["香港"], ["新加坡"], ["英国"]][idx - 1]
        expected_dest = unique_items(current_dest + new_dest)

        cases.extend(
            [
                {
                    "case_id": f"case_{group_offset + 1:03d}",
                    "tags": ["anxious_parent", "unit_confusion"],
                    "input": {
                        "last_ai_msg": "您先别急，把孩子现在的学历、成绩、预算和想去的地方告诉我。",
                        "last_user_msg": (
                            f"老师我真的快急死了，我家{variant['parent_word']}现在是{variant['stage']}，"
                            f"别的我说得有点乱，反正{variant['background']}，想去{'或'.join(variant['dest'])}，"
                            f"预算大概{variant['budget_foreign']}，学校最好{variant['school']}。"
                        ),
                    },
                    "expected": profile(
                        user_role="家长",
                        educationStage=variant["stage"],
                        budget=budget(variant["budget_amount"], variant["budget_period"]),
                        destination_preference=variant["dest"],
                        target_school=variant["school"],
                        academic_background=variant["background"],
                    ),
                },
                {
                    "case_id": f"case_{group_offset + 2:03d}",
                    "tags": ["anxious_parent", "mixed_scores"],
                    "input": {
                        "last_ai_msg": "成绩和语言先发我，我先判断有没有基本申请空间。",
                        "last_user_msg": (
                            f"我说话啰嗦您别嫌弃啊，{variant['parent_word']}目前{variant['stage']}，{variant['background']}，"
                            f"均分85，绩点3.2，托福没考，四级刚过，但又老说以后想学{mixed_major}。"
                        ),
                    },
                    "expected": profile(
                        user_role="家长",
                        educationStage=variant["stage"],
                        target_major=mixed_major,
                        academic_background=f"{variant['background']}；均分85；绩点3.2",
                        language_level="托福没考；四级刚过",
                    ),
                },
                {
                    "case_id": f"case_{group_offset + 3:03d}",
                    "tags": ["anxious_parent", "target_conflict"],
                    "input": {
                        "last_ai_msg": "主要想读什么方向？我需要分清楚孩子自己和家长的想法。",
                        "last_user_msg": (
                            f"说实话我想让他读{mom_major}更稳，但他自己一直念叨{own_major}，"
                            f"所以国家先看{'、'.join(variant['dest'])}吧。"
                        ),
                    },
                    "expected": profile(
                        user_role="家长",
                        destination_preference=variant["dest"],
                        target_major=own_major,
                    ),
                },
                {
                    "case_id": f"case_{group_offset + 4:03d}",
                    "tags": ["anxious_parent", "all_missing"],
                    "input": {
                        "last_ai_msg": "先说下基本情况。",
                        "last_user_msg": anxious_sparse_user,
                    },
                    "expected": anxious_sparse_expected,
                },
                {
                    "case_id": f"case_{group_offset + 5:03d}",
                    "tags": ["anxious_parent", "implicit_confirmation", "profile_merge"],
                    "input": {
                        "last_ai_msg": (
                            f"我确认一下，您家孩子现在{variant['stage']}，{variant['background']}，"
                            f"预算总共{variant['budget_amount']}万，先看{'、'.join(new_dest)}，对吗？"
                        ),
                        "last_user_msg": "对，先按这个理解。",
                        "current_profile": profile(
                            user_role="家长",
                            destination_preference=current_dest,
                            target_school="先保底",
                        ),
                    },
                    "expected": profile(
                        user_role="家长",
                        educationStage=variant["stage"],
                        budget=budget(variant["budget_amount"], "TOTAL"),
                        destination_preference=expected_dest,
                        target_school="先保底",
                        academic_background=variant["background"],
                    ),
                },
            ]
        )

        top_offset = 100 + (idx - 1) * 25
        study_major = ["CS", "DS", "ECE", "BA"][idx - 1]
        school_target = ["Top30 CS", "G5", "CMU/UC系", "专排前10"][idx - 1]
        self_choice = ["人工智能", "数据科学", "交互媒体", "金融工程"][idx - 1]
        parent_choice = ["金融", "会计", "统计", "MBA"][idx - 1]
        top_sparse_user = [
            "双非高二还能冲吗？",
            "二本码农还能申？",
            "211商科转码还有戏？",
            "硕一想转申美博还行吗？",
        ][idx - 1]
        top_sparse_expected = [
            profile(user_role="学生", educationStage="高中", academic_background="高二"),
            profile(user_role="学生", educationStage="本科", academic_background="二本码农"),
            profile(user_role="学生", educationStage="本科", academic_background="211商科转码"),
            profile(user_role="学生", educationStage="研究生", academic_background="硕一"),
        ][idx - 1]

        cases.extend(
            [
                {
                    "case_id": f"case_{top_offset + 1:03d}",
                    "tags": ["top_student", "unit_confusion"],
                    "input": {
                        "last_ai_msg": "把你的背景、预算、目标国家和项目缩写都直接扔给我。",
                        "last_user_msg": (
                            f"我本人，{variant['stage']}，{variant['background']}，TBD 但预算 {variant['budget_foreign']} all-in，"
                            f"目标 {' / '.join(variant['dest'])} 的 {study_major}，学校最好 {school_target}。"
                        ),
                    },
                    "expected": profile(
                        user_role="学生",
                        educationStage=variant["stage"],
                        budget=budget(variant["budget_amount"], "TOTAL"),
                        destination_preference=variant["dest"],
                        target_school=school_target,
                        target_major=study_major,
                        academic_background=variant["background"],
                    ),
                },
                {
                    "case_id": f"case_{top_offset + 2:03d}",
                    "tags": ["top_student", "mixed_scores"],
                    "input": {
                        "last_ai_msg": "把均分、GPA、标化和语言一起说全。",
                        "last_user_msg": (
                            f"我{variant['stage']}，CV 还行，均分85，GPA 3.2，托福没考，四级刚过，"
                            f"另外有两段 RA，想申 {study_major}。"
                        ),
                    },
                    "expected": profile(
                        user_role="学生",
                        educationStage=variant["stage"],
                        target_major=study_major,
                        academic_background=f"{variant['stage']}；均分85；GPA 3.2；两段 RA",
                        language_level="托福没考；四级刚过",
                    ),
                },
                {
                    "case_id": f"case_{top_offset + 3:03d}",
                    "tags": ["top_student", "target_conflict"],
                    "input": {
                        "last_ai_msg": "你自己真正想读什么，不要给我家长版本。",
                        "last_user_msg": (
                            f"我妈想让我读{parent_choice}，但我自己更偏{self_choice}，"
                            f"国家还是优先{'、'.join(variant['dest'])}。"
                        ),
                    },
                    "expected": profile(
                        user_role="学生",
                        destination_preference=variant["dest"],
                        target_major=self_choice,
                    ),
                },
                {
                    "case_id": f"case_{top_offset + 4:03d}",
                    "tags": ["top_student", "all_missing"],
                    "input": {
                        "last_ai_msg": "一句话给我背景。",
                        "last_user_msg": top_sparse_user,
                    },
                    "expected": top_sparse_expected,
                },
                {
                    "case_id": f"case_{top_offset + 5:03d}",
                    "tags": ["top_student", "implicit_confirmation", "profile_merge"],
                    "input": {
                        "last_ai_msg": (
                            f"所以你现在{variant['stage']}，{variant['background']}，目标{'、'.join(variant['dest'])}的"
                            f"{study_major}，预算总共{variant['budget_amount']}万，对吧？"
                        ),
                        "last_user_msg": "yep，按这个版本记。",
                        "current_profile": profile(
                            user_role="学生",
                            target_school="先看综排",
                            destination_preference=[variant["dest"][0]],
                        ),
                    },
                    "expected": profile(
                        user_role="学生",
                        educationStage=variant["stage"],
                        budget=budget(variant["budget_amount"], "TOTAL"),
                        destination_preference=variant["dest"],
                        target_school="先看综排",
                        target_major=study_major,
                        academic_background=variant["background"],
                    ),
                },
            ]
        )

        cross_offset = 200 + (idx - 1) * 25
        new_major = ["心理学", "传媒", "公共政策", "工业设计"][idx - 1]
        old_major = ["土木", "会计", "生物", "机械"][idx - 1]
        alt_major = ["金融", "法学", "CS", "建筑"][idx - 1]
        cross_sparse_user = [
            "土木转心理还有戏吗？",
            "会计不想读了能转传媒吗？",
            "生物背景还来得及换公政吗？",
            "机械狗转工业设计晚不晚？",
        ][idx - 1]
        cross_sparse_expected = [
            profile(user_role="学生", target_major="心理学", academic_background="土木转心理"),
            profile(user_role="学生", target_major="传媒", academic_background="会计不想读了"),
            profile(user_role="学生", target_major="公共政策", academic_background="生物背景"),
            profile(user_role="学生", target_major="工业设计", academic_background="机械狗"),
        ][idx - 1]

        cases.extend(
            [
                {
                    "case_id": f"case_{cross_offset + 1:03d}",
                    "tags": ["cross_major", "unit_confusion"],
                    "input": {
                        "last_ai_msg": "你现在学什么，想转什么，预算怎么说，按顺序告诉我。",
                        "last_user_msg": (
                            f"我现在是{old_major}背景，{variant['background']}，但其实想跨到{new_major}，"
                            f"预算卡在{variant['budget_foreign']}，国家可能是{'或者'.join(variant['dest'])}。"
                        ),
                    },
                    "expected": profile(
                        user_role="学生",
                        educationStage=variant["stage"],
                        budget=budget(variant["budget_amount"], "TOTAL"),
                        destination_preference=variant["dest"],
                        target_major=new_major,
                        academic_background=f"{old_major}背景；{variant['background']}",
                    ),
                },
                {
                    "case_id": f"case_{cross_offset + 2:03d}",
                    "tags": ["cross_major", "mixed_scores"],
                    "input": {
                        "last_ai_msg": "先把成绩和语言说一下，再谈跨专业。",
                        "last_user_msg": (
                            f"我脑子有点乱，我现在{variant['stage']}，原来学{old_major}，均分85，绩点3.2，"
                            "托福没考，四级刚过，想申"
                            f"{new_major}。"
                        ),
                    },
                    "expected": profile(
                        user_role="学生",
                        educationStage=variant["stage"],
                        target_major=new_major,
                        academic_background=f"原来学{old_major}；均分85；绩点3.2",
                        language_level="托福没考；四级刚过",
                    ),
                },
                {
                    "case_id": f"case_{cross_offset + 3:03d}",
                    "tags": ["cross_major", "target_conflict"],
                    "input": {
                        "last_ai_msg": "你自己最终想读的到底是哪一个？",
                        "last_user_msg": (
                            f"我一开始说想转{alt_major}，后来家里又让我回到{old_major}，"
                            f"但我自己现在最想读的还是{new_major}。"
                        ),
                    },
                    "expected": profile(
                        user_role="学生",
                        target_major=new_major,
                    ),
                },
                {
                    "case_id": f"case_{cross_offset + 4:03d}",
                    "tags": ["cross_major", "all_missing"],
                    "input": {
                        "last_ai_msg": "一句话说你现在和目标。",
                        "last_user_msg": cross_sparse_user,
                    },
                    "expected": cross_sparse_expected,
                },
                {
                    "case_id": f"case_{cross_offset + 5:03d}",
                    "tags": ["cross_major", "implicit_confirmation", "profile_merge"],
                    "input": {
                        "last_ai_msg": (
                            f"我确认一下，你目前{variant['background']}，原专业{old_major}，"
                            f"想转{new_major}，先看{'、'.join(variant['dest'])}，对吗？"
                        ),
                        "last_user_msg": "嗯，方向没错。",
                        "current_profile": profile(
                            user_role="学生",
                            destination_preference=[variant["dest"][0]],
                            target_school="先不设限",
                        ),
                    },
                    "expected": profile(
                        user_role="学生",
                        educationStage=variant["stage"],
                        destination_preference=variant["dest"],
                        target_school="先不设限",
                        target_major=new_major,
                        academic_background=f"原专业{old_major}；{variant['background']}",
                    ),
                },
            ]
        )

        budget_offset = 300 + (idx - 1) * 25
        cheap_major = ["护理", "商科", "酒店管理", "教育学"][idx - 1]
        budget_sparse_user = [
            "预算很低，高二还能去哪？",
            "二本没钱还能出吗？",
            "家里抠门，211还能申吗？",
            "读研但真没钱还有路吗？",
        ][idx - 1]
        budget_sparse_expected = [
            profile(educationStage="高中", academic_background="高二"),
            profile(educationStage="本科", academic_background="二本"),
            profile(educationStage="本科", academic_background="211"),
            profile(educationStage="研究生", academic_background="读研"),
        ][idx - 1]
        debt_background = [
            "家里还在还房贷",
            "我自己有助学贷款",
            "家里背着生意欠款",
            "刚创业亏了一笔还在还债",
        ][idx - 1]

        cases.extend(
            [
                {
                    "case_id": f"case_{budget_offset + 1:03d}",
                    "tags": ["budget_sensitive", "unit_confusion"],
                    "input": {
                        "last_ai_msg": "预算能说具体点吗，最好带上总预算还是年预算。",
                        "last_user_msg": (
                            f"我就想捡便宜的，{variant['stage']}，{variant['background']}，预算真就"
                            f"{variant['budget_foreign']}，而且是全部花销一起算，想去{'、'.join(variant['dest'])}读{cheap_major}。"
                        ),
                    },
                    "expected": profile(
                        user_role="学生",
                        educationStage=variant["stage"],
                        budget=budget(variant["budget_amount"], "TOTAL"),
                        destination_preference=variant["dest"],
                        target_major=cheap_major,
                        academic_background=variant["background"],
                    ),
                },
                {
                    "case_id": f"case_{budget_offset + 2:03d}",
                    "tags": ["budget_sensitive", "mixed_scores"],
                    "input": {
                        "last_ai_msg": "我先看你的硬条件，再判断能不能走低成本路线。",
                        "last_user_msg": (
                            f"{variant['stage']}，均分85，绩点3.2，托福没考，四级刚过，预算不多，"
                            f"想申最省钱的{cheap_major}。"
                        ),
                    },
                    "expected": profile(
                        user_role="学生",
                        educationStage=variant["stage"],
                        target_major=cheap_major,
                        academic_background="均分85；绩点3.2",
                        language_level="托福没考；四级刚过",
                    ),
                },
                {
                    "case_id": f"case_{budget_offset + 3:03d}",
                    "tags": ["budget_sensitive", "target_conflict"],
                    "input": {
                        "last_ai_msg": "你是因为喜欢还是因为便宜在选专业？",
                        "last_user_msg": (
                            f"我妈让我读金融说好找工作，但我其实只想找便宜一点的{cheap_major}，"
                            f"国家先看{'、'.join(variant['dest'])}。"
                        ),
                    },
                    "expected": profile(
                        user_role="学生",
                        destination_preference=variant["dest"],
                        target_major=cheap_major,
                    ),
                },
                {
                    "case_id": f"case_{budget_offset + 4:03d}",
                    "tags": ["budget_sensitive", "all_missing"],
                    "input": {
                        "last_ai_msg": "先说阶段和预算。",
                        "last_user_msg": budget_sparse_user,
                    },
                    "expected": budget_sparse_expected,
                },
                {
                    "case_id": f"case_{budget_offset + 5:03d}",
                    "tags": ["budget_sensitive", "debt_not_budget", "implicit_confirmation"],
                    "input": {
                        "last_ai_msg": (
                            f"我理解你现在{variant['stage']}，{variant['background']}，手上没有明确预算，"
                            f"只是{debt_background}，对吗？"
                        ),
                        "last_user_msg": "对，所以别给我太贵的。",
                        "current_profile": profile(user_role="学生", destination_preference=variant["dest"]),
                    },
                    "expected": profile(
                        user_role="学生",
                        educationStage=variant["stage"],
                        budget=budget(None, "UNKNOWN"),
                        destination_preference=variant["dest"],
                        academic_background=variant["background"],
                    ),
                },
            ]
        )

        low_offset = 400 + (idx - 1) * 25
        plain_major = ["幼教", "传媒", "商科", "设计"][idx - 1]
        low_sparse_user = [
            "救命，我高二还能去哪？",
            "二本毕业还能去哪？",
            "211毕业还能去哪？",
            "硕士读着读着还能去哪？",
        ][idx - 1]
        low_sparse_expected = [
            profile(educationStage="高中", academic_background="高二"),
            profile(educationStage="本科", academic_background="二本毕业"),
            profile(educationStage="本科", academic_background="211毕业"),
            profile(educationStage="研究生", academic_background="硕士在读"),
        ][idx - 1]
        readiness = ["需要过渡/暂缓", None, None, None][idx - 1]

        cases.extend(
            [
                {
                    "case_id": f"case_{low_offset + 1:03d}",
                    "tags": ["low_cognition", "unit_confusion"],
                    "input": {
                        "last_ai_msg": "你把能记住的信息慢慢发给我。",
                        "last_user_msg": (
                            f"我也不太懂，就知道自己现在{variant['stage']}，想去{'或者'.join(variant['dest'])}，"
                            f"兜里最多{variant['budget_foreign']}，学个{plain_major}差不多。"
                        ),
                    },
                    "expected": profile(
                        educationStage=variant["stage"],
                        budget=budget(variant["budget_amount"], "TOTAL"),
                        destination_preference=variant["dest"],
                        target_major=plain_major,
                    ),
                },
                {
                    "case_id": f"case_{low_offset + 2:03d}",
                    "tags": ["low_cognition", "mixed_scores"],
                    "input": {
                        "last_ai_msg": "成绩有多少就说多少，没有就直接说没有。",
                        "last_user_msg": (
                            "我就知道均分85，绩点3.2，托福没考，四级刚过，"
                            f"别的真说不上来，应该还想读{plain_major}。"
                        ),
                    },
                    "expected": profile(
                        target_major=plain_major,
                        academic_background="均分85；绩点3.2",
                        language_level="托福没考；四级刚过",
                    ),
                },
                {
                    "case_id": f"case_{low_offset + 3:03d}",
                    "tags": ["low_cognition", "target_conflict"],
                    "input": {
                        "last_ai_msg": "如果家里意见不一致，你先说你自己想学什么。",
                        "last_user_msg": (
                            f"家里让我随便读个金融，我自己觉得{plain_major}也许更适合，反正别太难。"
                        ),
                    },
                    "expected": profile(target_major=plain_major),
                },
                {
                    "case_id": f"case_{low_offset + 4:03d}",
                    "tags": ["low_cognition", "all_missing"],
                    "input": {
                        "last_ai_msg": "先别急，一句话说你现在在哪个阶段。",
                        "last_user_msg": low_sparse_user,
                    },
                    "expected": low_sparse_expected,
                },
                {
                    "case_id": f"case_{low_offset + 5:03d}",
                    "tags": ["low_cognition", "implicit_confirmation", "profile_merge"],
                    "input": {
                        "last_ai_msg": (
                            f"我确认一下，你现在{variant['stage']}，{variant['background']}，"
                            f"先看{'、'.join(variant['dest'])}，"
                            f"{'想先在国内过渡一下再出去' if readiness else '暂时不设过渡'}，对吗？"
                        ),
                        "last_user_msg": "嗯，你这么记吧。",
                        "current_profile": profile(
                            destination_preference=[variant["dest"][0]],
                            target_school="别太难进",
                        ),
                    },
                    "expected": profile(
                        educationStage=variant["stage"],
                        destination_preference=variant["dest"],
                        abroad_readiness=readiness,
                        target_school="别太难进",
                        academic_background=variant["background"],
                    ),
                },
            ]
        )

    cases = cases[:80] + build_special_cases()

    for index, case in enumerate(cases, start=1):
        case["case_id"] = f"case_{index:03d}"

    overrides = {
        "case_004": {
            "expected": profile(
                user_role="家长",
                educationStage="高中",
                academic_background=None,
            ),
        },
        "case_009": {
            "input": {
                "last_ai_msg": "一句话给我背景。",
                "last_user_msg": "高二还能冲英国Top30吗？",
            },
            "expected": profile(
                user_role="学生",
                educationStage="高中",
                academic_background=None,
            ),
        },
        "case_011": {
            "tags": ["cross_major", "unit_confusion", "currency_conversion"],
            "input": {
                "last_ai_msg": "你现在学什么，想转什么，预算怎么说，按顺序告诉我。",
                "last_user_msg": "我现在本科土木，均分82，但其实想跨到心理学，预算卡在5万美金，国家可能是英国。",
            },
            "expected": profile(
                user_role="学生",
                educationStage="本科",
                budget=budget(36, "TOTAL"),
                destination_preference=["英国"],
                target_major="心理学",
                academic_background="本科土木；均分82",
            ),
        },
        "case_012": {
            "input": {
                "last_ai_msg": "先把成绩和语言说一下，再谈跨专业。",
                "last_user_msg": "我脑子有点乱，我现在本科土木，均分85，绩点3.2，托福没考，四级刚过，想申心理学。",
            },
            "expected": profile(
                user_role="学生",
                educationStage="本科",
                target_major="心理学",
                academic_background="本科土木；均分85；绩点3.2",
                language_level="托福没考；四级刚过",
            ),
        },
        "case_014": {
            "expected": profile(
                user_role="学生",
                target_major="心理学",
                academic_background=None,
            ),
        },
    }
    for case in cases:
        patch = overrides.get(case["case_id"])
        if not patch:
            continue
        if "tags" in patch:
            case["tags"] = patch["tags"]
        if "input" in patch:
            case["input"] = patch["input"]
        if "expected" in patch:
            case["expected"] = patch["expected"]

    if len(cases) != 100:
        raise ValueError(f"Expected 100 cases, got {len(cases)}")

    return cases


def main() -> None:
    cases = build_cases()
    OUTPUT_PATH.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(cases)} cases to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
