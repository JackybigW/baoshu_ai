# Execution Eval Dataset Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand `execution_eval` from a 14-case smoke set into a production-grade benchmark where each target execution node has at least 20-50 hand-authored golden cases, with isolated per-node authoring and stronger benchmark contracts for node-specific failure modes.

**Architecture:** Split the execution golden dataset into per-node shard files so six worker agents can write cases in parallel without touching the same JSON file. Keep `run_eval.py` consuming one combined dataset, but generate that combined file from shard files through a deterministic builder/validator. Strengthen the benchmark only where current contracts are too weak for the requested coverage: `consultant` knowledge-base grounding, `interviewer` follow-up quality, and `chit_chat` non-human bracketed-emotion patterns.

**Tech Stack:** Python 3.10, pytest, Pydantic models, existing `nodes_eval/execution_eval` pipeline, LangChain message schema, GitHub Flow, subagents for parallel dataset authoring.

---

## File Structure

**Create**
- `docs/superpowers/plans/2026-04-07-execution-eval-dataset-expansion.md`
- `nodes_eval/execution_eval/datasets/README.md`
- `nodes_eval/execution_eval/datasets/consultant_cases.json`
- `nodes_eval/execution_eval/datasets/interviewer_cases.json`
- `nodes_eval/execution_eval/datasets/high_value_cases.json`
- `nodes_eval/execution_eval/datasets/low_budget_cases.json`
- `nodes_eval/execution_eval/datasets/art_cases.json`
- `nodes_eval/execution_eval/datasets/chit_chat_cases.json`
- `nodes_eval/execution_eval/build_dataset.py`
- `tests/test_execution_eval_dataset_builder.py`

**Modify**
- `nodes_eval/execution_eval/golden_dataset.json`
- `nodes_eval/execution_eval/run_eval.py`
- `nodes_eval/execution_eval/benchmark.py`
- `nodes_eval/execution_eval/README.md`
- `tests/test_execution_eval.py`

**Responsibilities**
- `datasets/*.json`: one node per file, hand-authored golden cases, disjoint ownership for subagents
- `build_dataset.py`: schema validation, duplicate `case_id` detection, shard merge, per-node count reporting
- `golden_dataset.json`: built artifact consumed by `run_eval.py`
- `benchmark.py`: minimal contract extensions only where current scoring cannot reliably catch requested node-specific failures
- `tests/test_execution_eval.py`: benchmark and loader behavior
- `tests/test_execution_eval_dataset_builder.py`: shard merge and minimum-count guarantees

---

### Task 1: Lock The Dataset Sharding Contract

**Files:**
- Create: `nodes_eval/execution_eval/datasets/README.md`
- Create: `nodes_eval/execution_eval/build_dataset.py`
- Modify: `nodes_eval/execution_eval/run_eval.py`
- Test: `tests/test_execution_eval_dataset_builder.py`

- [ ] **Step 1: Write the failing test for shard merge and per-node minimums**

```python
from pathlib import Path

from nodes_eval.execution_eval.build_dataset import build_execution_dataset


def test_build_execution_dataset_merges_all_required_node_shards(tmp_path: Path):
    payload = build_execution_dataset(output_path=tmp_path / "golden_dataset.json")

    node_counts = payload["node_counts"]

    assert node_counts["consultant"] >= 20
    assert node_counts["interviewer"] >= 20
    assert node_counts["high_value"] >= 20
    assert node_counts["low_budget"] >= 20
    assert node_counts["art"] >= 20
    assert node_counts["chit_chat"] >= 20
    assert (tmp_path / "golden_dataset.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
conda run -n agent pytest tests/test_execution_eval_dataset_builder.py::test_build_execution_dataset_merges_all_required_node_shards -v
```

Expected: FAIL with `ModuleNotFoundError` for `nodes_eval.execution_eval.build_dataset` or missing builder function.

- [ ] **Step 3: Write the dataset builder with explicit shard registry**

```python
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


DATASET_DIR = Path(__file__).resolve().parent / "datasets"
SHARD_FILES = (
    ("consultant", DATASET_DIR / "consultant_cases.json"),
    ("interviewer", DATASET_DIR / "interviewer_cases.json"),
    ("high_value", DATASET_DIR / "high_value_cases.json"),
    ("low_budget", DATASET_DIR / "low_budget_cases.json"),
    ("art", DATASET_DIR / "art_cases.json"),
    ("chit_chat", DATASET_DIR / "chit_chat_cases.json"),
)
MIN_CASES_PER_NODE = 20


def build_execution_dataset(*, output_path: Path) -> dict:
    merged = []
    node_counts = Counter()
    seen_case_ids = set()

    for expected_node, shard_path in SHARD_FILES:
        cases = json.loads(shard_path.read_text(encoding="utf-8"))
        for item in cases:
            case_id = item["case_id"]
            if case_id in seen_case_ids:
                raise ValueError(f"duplicate case_id: {case_id}")
            if item["node_name"] != expected_node:
                raise ValueError(f"{case_id} has wrong node_name: {item['node_name']}")
            seen_case_ids.add(case_id)
            node_counts[expected_node] += 1
            merged.append(item)

    for node_name, count in node_counts.items():
        if count < MIN_CASES_PER_NODE:
            raise ValueError(f"{node_name} only has {count} cases")

    merged.sort(key=lambda item: item["case_id"])
    output_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"case_count": len(merged), "node_counts": dict(node_counts)}
```

- [ ] **Step 4: Wire `run_eval.py` to keep consuming `golden_dataset.json` without changing CLI**

```python
DEFAULT_DATASET_PATH = Path(__file__).resolve().parent / "golden_dataset.json"


def load_cases(dataset_path: Path) -> List[EvalCase]:
    return [EvalCase.model_validate(item) for item in load_json(dataset_path)]
```

Keep this behavior unchanged. The builder is a separate pre-step so runtime semantics stay stable.

- [ ] **Step 5: Add shard authoring rules for workers**

Write `nodes_eval/execution_eval/datasets/README.md` with these non-negotiables:

```md
# Execution Eval Dataset Shards

- One file per node. Do not edit another node's shard in the same task.
- Every case must use a unique `case_id` prefix:
  - `consultant`: `exe_cons_###`
  - `interviewer`: `exe_int_###`
  - `high_value`: `exe_vip_###`
  - `low_budget`: `exe_low_###`
  - `art`: `exe_art_###`
  - `chit_chat`: `exe_chat_###`
- `expected` must express observable behavior only.
- Do not encode ideal prose; encode contract, failure risks, and node goal.
```

- [ ] **Step 6: Run the new test**

Run:

```bash
conda run -n agent pytest tests/test_execution_eval_dataset_builder.py::test_build_execution_dataset_merges_all_required_node_shards -v
```

Expected: PASS once the shard files exist and each node reaches minimum count.

- [ ] **Step 7: Commit**

```bash
git add nodes_eval/execution_eval/datasets/README.md \
        nodes_eval/execution_eval/build_dataset.py \
        tests/test_execution_eval_dataset_builder.py
git commit -m "feat(eval): add execution dataset shard builder"
```

### Task 2: Strengthen Benchmark Contracts For Special Nodes

**Files:**
- Modify: `nodes_eval/execution_eval/benchmark.py`
- Modify: `tests/test_execution_eval.py`

- [ ] **Step 1: Write failing tests for new contract fields**

```python
from langchain_core.messages import AIMessage

from nodes_eval.execution_eval.benchmark import score_execution_output


def test_score_execution_output_penalizes_parenthetical_emotion_stage_directions():
    breakdown = score_execution_output(
        node_name="chit_chat",
        contract={
            "min_segments": 1,
            "max_segments": 3,
            "forbidden_regexes": [r"[（(][^）)]{1,12}[）)]"],
            "skip_rubric": True,
        },
        case_input={},
        output_messages=[AIMessage(content="哈哈（挠头）你也挺会聊")],
        actual_status=None,
        judge=None,
    )

    assert "格式违规" in breakdown.failure_tags
    assert breakdown.overall_score < 100


def test_score_execution_output_penalizes_missing_knowledge_reference_when_required():
    breakdown = score_execution_output(
        node_name="consultant",
        contract={
            "min_segments": 1,
            "max_segments": 3,
            "required_keyword_groups": [["方案"]],
            "required_context_terms": [["预科", "中外合办", "马来西亚"]],
            "skip_rubric": True,
        },
        case_input={},
        output_messages=[AIMessage(content="我先给你一个路径，你看看能不能接受")],
        actual_status=None,
        judge=None,
    )

    assert "关键要点缺失" in breakdown.failure_tags
```

- [ ] **Step 2: Run the targeted tests to verify failure**

Run:

```bash
conda run -n agent pytest \
  tests/test_execution_eval.py::test_score_execution_output_penalizes_parenthetical_emotion_stage_directions \
  tests/test_execution_eval.py::test_score_execution_output_penalizes_missing_knowledge_reference_when_required -v
```

Expected: FAIL because `forbidden_regexes` and `required_context_terms` are not supported yet.

- [ ] **Step 3: Add the smallest benchmark extension needed**

```python
import re


def _regex_penalty(text: str, forbidden_regexes: Sequence[str]) -> tuple[float, list[str]]:
    if not forbidden_regexes:
        return 1.0, []
    for pattern in forbidden_regexes:
        if re.search(pattern, text):
            return 0.5, ["格式违规"]
    return 1.0, []


def _required_context_score(text: str, required_context_terms: Sequence[Sequence[str]]) -> tuple[float, list[str]]:
    if not required_context_terms:
        return 1.0, []
    lowered = text.lower()
    matched = sum(
        1
        for group in required_context_terms
        if any(token.lower() in lowered for token in group)
    )
    score = matched / len(required_context_terms)
    return score, ([] if score == 1.0 else ["关键要点缺失"])
```

Integrate these only as extensions to the existing deterministic scoring path. Do not rewrite the benchmark.

- [ ] **Step 4: Keep weights explicit and conservative**

Use this integration:

```python
context_score, context_failures = _required_context_score(
    output_text,
    contract.get("required_context_terms") or [],
)
regex_score, regex_failures = _regex_penalty(
    output_text,
    contract.get("forbidden_regexes") or [],
)
```

Apply them by adjusting existing keyword/format components, not by introducing a brand-new heavyweight dimension.

- [ ] **Step 5: Run focused tests**

Run:

```bash
conda run -n agent pytest tests/test_execution_eval.py -v
```

Expected: PASS, with old benchmark tests still green.

- [ ] **Step 6: Commit**

```bash
git add nodes_eval/execution_eval/benchmark.py tests/test_execution_eval.py
git commit -m "feat(eval): extend execution benchmark contracts"
```

### Task 3: Define The Six Worker Briefs Before Parallel Dataset Authoring

**Files:**
- Modify: `nodes_eval/execution_eval/datasets/README.md`

- [ ] **Step 1: Add worker ownership table**

Write this into the README:

```md
## Worker Ownership

- Worker A: `consultant_cases.json`
- Worker B: `interviewer_cases.json`
- Worker C: `high_value_cases.json`
- Worker D: `low_budget_cases.json`
- Worker E: `art_cases.json`
- Worker F: `chit_chat_cases.json`

Workers are not alone in the codebase. Do not revert others' edits. Only touch your owned shard file unless explicitly asked.
```

- [ ] **Step 2: Add node-specific authoring guardrails**

Use this exact content:

```md
### consultant
- Must cover `NEED_CONSULTING`, `SALES_READY`, and `DECISION_SUPPORT`
- Must include cases where answer should reference Excel-backed product paths from `nodes/tools.py::search_products`
- Must include cases where user asks a concrete program question before asking for a plan

### interviewer
- Focus on missing-field follow-up quality, not final方案输出
- Cover `academic_background`, `language_level`, `budget`, `destination_preference`, `educationStage`, `abroad_readiness`
- At least 5 anti-loop cases where user says “不知道/不懂/没想好”

### high_value
- Cover early trust-building, tough objection handling, explicit request for真人, and long-dialog forced handoff
- Include gray-area requests that should be smoothly escalated

### low_budget
- Cover frugal pathing, budget shame avoidance, concrete low-cost routes, and handoff timing

### art
- Cover作品集、审美方向、跨专业艺术申请、语言短板、用户已有作品时的转人工

### chit_chat
- Cover pure闲聊、夸赞、玩笑、轻情绪、边聊边试探业务
- At least 10 cases must explicitly guard against bracketed mood text like `（笑）`, `(捂脸)`, `（摊手）`
```

- [ ] **Step 3: Commit**

```bash
git add nodes_eval/execution_eval/datasets/README.md
git commit -m "docs(eval): define execution dataset worker briefs"
```

### Task 4: Worker A Builds `consultant` Cases

**Files:**
- Create: `nodes_eval/execution_eval/datasets/consultant_cases.json`
- Modify: `tests/test_execution_eval_dataset_builder.py`

- [ ] **Step 1: Write the failing minimum-count test**

```python
def test_consultant_shard_has_at_least_20_cases(dataset_summary):
    assert dataset_summary["node_counts"]["consultant"] >= 20
```

- [ ] **Step 2: Run the test to verify failure**

Run:

```bash
conda run -n agent pytest tests/test_execution_eval_dataset_builder.py::test_consultant_shard_has_at_least_20_cases -v
```

Expected: FAIL because the shard is empty or below 20.

- [ ] **Step 3: Author 20-30 consultant cases**

Seed the file with this shape:

```json
[
  {
    "case_id": "exe_cons_001",
    "tags": ["consultant", "need_consulting", "excel_rag", "malaysia_path"],
    "node_name": "consultant",
    "input": {
      "messages": [
        {"type": "human", "content": "预算一年12万，别太冒险，给我个稳一点的路径"}
      ],
      "profile": {
        "user_role": "家长",
        "educationStage": "高中",
        "budget": {"amount": 12, "period": "YEAR"},
        "abroad_readiness": "需要过渡/暂缓"
      },
      "last_intent": "NEED_CONSULTING",
      "dialog_status": "CONSULTING"
    },
    "expected": {
      "must_call_tool": false,
      "min_segments": 1,
      "max_segments": 3,
      "max_chars_per_segment": 40,
      "required_keyword_groups": [["方案", "路径"]],
      "required_context_terms": [["预科", "中外合办", "马来西亚", "港澳"]],
      "expected_status": "PERSUADING",
      "node_goal": "给出知识库支持的初步路径并继续推进",
      "rubric_notes": "先回答，再问接受度；不要空泛鸡汤"
    }
  }
]
```

- [ ] **Step 4: Cover these sub-buckets explicitly**

Author at least:
- 8 `NEED_CONSULTING`
- 6 `SALES_READY`
- 6 `DECISION_SUPPORT`
- 5 Excel/RAG-heavy path cases
- 3 long-dialog handoff cases

- [ ] **Step 5: Rebuild the combined dataset**

Run:

```bash
conda run -n agent python nodes_eval/execution_eval/build_dataset.py
```

Expected: `consultant` count reported at `>=20`.

- [ ] **Step 6: Commit**

```bash
git add nodes_eval/execution_eval/datasets/consultant_cases.json \
        nodes_eval/execution_eval/golden_dataset.json \
        tests/test_execution_eval_dataset_builder.py
git commit -m "test(eval): expand consultant execution cases"
```

### Task 5: Worker B Builds `interviewer` Cases

**Files:**
- Create: `nodes_eval/execution_eval/datasets/interviewer_cases.json`
- Modify: `tests/test_execution_eval_dataset_builder.py`

- [ ] **Step 1: Write the failing minimum-count test**

```python
def test_interviewer_shard_has_at_least_20_cases(dataset_summary):
    assert dataset_summary["node_counts"]["interviewer"] >= 20
```

- [ ] **Step 2: Run the test to verify failure**

Run:

```bash
conda run -n agent pytest tests/test_execution_eval_dataset_builder.py::test_interviewer_shard_has_at_least_20_cases -v
```

- [ ] **Step 3: Author 20-30 interviewer cases**

Use this shape:

```json
[
  {
    "case_id": "exe_int_001",
    "tags": ["interviewer", "academic_background", "anti_loop"],
    "node_name": "interviewer",
    "input": {
      "messages": [{"type": "human", "content": "我先看看还有没有机会，别上来就推销"}],
      "profile": {
        "user_role": "学生",
        "educationStage": "本科",
        "academic_background": "本科在读，传媒方向"
      },
      "last_intent": "NEED_CONSULTING",
      "dialog_status": "CONSULTING"
    },
    "expected": {
      "must_call_tool": false,
      "min_segments": 1,
      "max_segments": 3,
      "max_chars_per_segment": 40,
      "required_keyword_groups": [["成绩", "均分", "GPA"], ["语言", "雅思", "托福"]],
      "node_goal": "先接住用户，再继续补齐背景",
      "rubric_notes": "必须是问问题，不是给方案；别机械重复"
    }
  }
]
```

- [ ] **Step 4: Cover these sub-buckets explicitly**

Author at least:
- 4 `academic_background`
- 4 `language_level`
- 4 `budget`
- 3 `destination_preference`
- 3 `educationStage`
- 4 `abroad_readiness`
- 5 anti-loop reformulation cases

- [ ] **Step 5: Rebuild and verify**

Run:

```bash
conda run -n agent python nodes_eval/execution_eval/build_dataset.py
conda run -n agent pytest tests/test_execution_eval_dataset_builder.py::test_interviewer_shard_has_at_least_20_cases -v
```

- [ ] **Step 6: Commit**

```bash
git add nodes_eval/execution_eval/datasets/interviewer_cases.json \
        nodes_eval/execution_eval/golden_dataset.json \
        tests/test_execution_eval_dataset_builder.py
git commit -m "test(eval): expand interviewer execution cases"
```

### Task 6: Worker C Builds `high_value` Cases

**Files:**
- Create: `nodes_eval/execution_eval/datasets/high_value_cases.json`
- Modify: `tests/test_execution_eval_dataset_builder.py`

- [ ] **Step 1: Add the failing count test**

```python
def test_high_value_shard_has_at_least_20_cases(dataset_summary):
    assert dataset_summary["node_counts"]["high_value"] >= 20
```

- [ ] **Step 2: Run it and watch it fail**

```bash
conda run -n agent pytest tests/test_execution_eval_dataset_builder.py::test_high_value_shard_has_at_least_20_cases -v
```

- [ ] **Step 3: Author 20-25 high-value cases**

Required buckets:
- early trust-building
- elite target objections
- user asks for voice/phone
- explicit真人 request
- gray-area or “能不能搞定” escalation
- long-dialog forced handoff

Seed case:

```json
{
  "case_id": "exe_vip_001",
  "tags": ["high_value", "early_stage", "elite_target"],
  "node_name": "high_value",
  "input": {
    "messages": [{"type": "human", "content": "预算一年70万，想冲美本前二十，先说路径"}],
    "profile": {
      "educationStage": "高中",
      "budget": {"amount": 70, "period": "YEAR"},
      "destination_preference": ["美国"]
    },
    "last_intent": "HIGH_VALUE",
    "dialog_status": "VIP_SERVICE"
  },
  "expected": {
    "must_call_tool": false,
    "min_segments": 1,
    "max_segments": 3,
    "max_chars_per_segment": 40,
    "required_keyword_groups": [["美国", "规划", "申请"]],
    "expected_status": "VIP_SERVICE",
    "node_goal": "先建立信任和专业感，不急着拉群",
    "rubric_notes": "高净值客户要像真人老手，不要客服腔"
  }
}
```

- [ ] **Step 4: Rebuild and verify**

```bash
conda run -n agent python nodes_eval/execution_eval/build_dataset.py
conda run -n agent pytest tests/test_execution_eval_dataset_builder.py::test_high_value_shard_has_at_least_20_cases -v
```

- [ ] **Step 5: Commit**

```bash
git add nodes_eval/execution_eval/datasets/high_value_cases.json \
        nodes_eval/execution_eval/golden_dataset.json \
        tests/test_execution_eval_dataset_builder.py
git commit -m "test(eval): expand high-value execution cases"
```

### Task 7: Worker D Builds `low_budget` Cases

**Files:**
- Create: `nodes_eval/execution_eval/datasets/low_budget_cases.json`
- Modify: `tests/test_execution_eval_dataset_builder.py`

- [ ] **Step 1: Add failing count test**

```python
def test_low_budget_shard_has_at_least_20_cases(dataset_summary):
    assert dataset_summary["node_counts"]["low_budget"] >= 20
```

- [ ] **Step 2: Run it**

```bash
conda run -n agent pytest tests/test_execution_eval_dataset_builder.py::test_low_budget_shard_has_at_least_20_cases -v
```

- [ ] **Step 3: Author 20-25 low-budget cases**

Required buckets:
- total budget under 10w
- annual budget under 10w
- family anxiety without shaming
- concrete low-cost pathing
- direct ask for真人
- overlong dialogue handoff

Seed case:

```json
{
  "case_id": "exe_low_001",
  "tags": ["low_budget", "frugal_path", "budget_anxiety"],
  "node_name": "low_budget",
  "input": {
    "messages": [{"type": "human", "content": "总共就8万，别给我画大饼，能走什么路"}],
    "profile": {
      "educationStage": "本科",
      "budget": {"amount": 8, "period": "TOTAL"}
    },
    "last_intent": "LOW_BUDGET",
    "dialog_status": "CONSULTING"
  },
  "expected": {
    "must_call_tool": false,
    "min_segments": 1,
    "max_segments": 3,
    "max_chars_per_segment": 40,
    "required_keyword_groups": [["性价比", "省", "预算"], ["路径", "方案"]],
    "expected_status": "PERSUADING",
    "node_goal": "给出务实路径，不羞辱用户预算",
    "rubric_notes": "要直给，但不能高高在上"
  }
}
```

- [ ] **Step 4: Rebuild and verify**

```bash
conda run -n agent python nodes_eval/execution_eval/build_dataset.py
conda run -n agent pytest tests/test_execution_eval_dataset_builder.py::test_low_budget_shard_has_at_least_20_cases -v
```

- [ ] **Step 5: Commit**

```bash
git add nodes_eval/execution_eval/datasets/low_budget_cases.json \
        nodes_eval/execution_eval/golden_dataset.json \
        tests/test_execution_eval_dataset_builder.py
git commit -m "test(eval): expand low-budget execution cases"
```

### Task 8: Worker E Builds `art` Cases

**Files:**
- Create: `nodes_eval/execution_eval/datasets/art_cases.json`
- Modify: `tests/test_execution_eval_dataset_builder.py`

- [ ] **Step 1: Add failing count test**

```python
def test_art_shard_has_at_least_20_cases(dataset_summary):
    assert dataset_summary["node_counts"]["art"] >= 20
```

- [ ] **Step 2: Run it**

```bash
conda run -n agent pytest tests/test_execution_eval_dataset_builder.py::test_art_shard_has_at_least_20_cases -v
```

- [ ] **Step 3: Author 20-25 art cases**

Required buckets:
- portfolio chaos
- strong作品 but weak language
- weak作品 but strong school desire
- cross-discipline art switch
- user already has works and needs expert review
- human escalation for nuanced art judging

Seed case:

```json
{
  "case_id": "exe_art_001",
  "tags": ["art", "portfolio", "early_stage"],
  "node_name": "art",
  "input": {
    "messages": [{"type": "human", "content": "我想申交互，作品集方向很散，先说我该怎么收"}],
    "profile": {
      "educationStage": "本科",
      "target_major": "交互设计"
    },
    "last_intent": "ART_CONSULTING",
    "dialog_status": "CONSULTING"
  },
  "expected": {
    "must_call_tool": false,
    "min_segments": 1,
    "max_segments": 3,
    "max_chars_per_segment": 40,
    "required_keyword_groups": [["作品集", "portfolio", "项目集"], ["方向", "专业", "学校"]],
    "expected_status": "VIP_SERVICE",
    "node_goal": "围绕作品集和方向展开，不把艺术生当普通申请",
    "rubric_notes": "必须体现艺术申请特性"
  }
}
```

- [ ] **Step 4: Rebuild and verify**

```bash
conda run -n agent python nodes_eval/execution_eval/build_dataset.py
conda run -n agent pytest tests/test_execution_eval_dataset_builder.py::test_art_shard_has_at_least_20_cases -v
```

- [ ] **Step 5: Commit**

```bash
git add nodes_eval/execution_eval/datasets/art_cases.json \
        nodes_eval/execution_eval/golden_dataset.json \
        tests/test_execution_eval_dataset_builder.py
git commit -m "test(eval): expand art execution cases"
```

### Task 9: Worker F Builds `chit_chat` Cases

**Files:**
- Create: `nodes_eval/execution_eval/datasets/chit_chat_cases.json`
- Modify: `tests/test_execution_eval_dataset_builder.py`

- [ ] **Step 1: Add failing count test**

```python
def test_chit_chat_shard_has_at_least_20_cases(dataset_summary):
    assert dataset_summary["node_counts"]["chit_chat"] >= 20
```

- [ ] **Step 2: Run it**

```bash
conda run -n agent pytest tests/test_execution_eval_dataset_builder.py::test_chit_chat_shard_has_at_least_20_cases -v
```

- [ ] **Step 3: Author 20-25 chit-chat cases**

Required buckets:
- compliment
- harmless teasing
- emotional but non-business
- “你像机器人吗” style probes
- light follow-up into study topic without hard sell
- bracketed mood anti-patterns

Seed case:

```json
{
  "case_id": "exe_chat_001",
  "tags": ["chit_chat", "compliment", "anti_parenthetical_emotion"],
  "node_name": "chit_chat",
  "input": {
    "messages": [{"type": "human", "content": "你这说话还挺逗"}],
    "profile": {},
    "last_intent": "CHIT_CHAT",
    "dialog_status": "CONSULTING"
  },
  "expected": {
    "must_call_tool": false,
    "min_segments": 1,
    "max_segments": 3,
    "max_chars_per_segment": 30,
    "forbidden_keywords": ["summon_specialist_tool", "tool"],
    "forbidden_regexes": ["[（(][^）)]{1,12}[）)]"],
    "node_goal": "轻松回应闲聊，不要演戏式括号情绪，不要突然卖方案",
    "rubric_notes": "口语化、像真人，不用括号标注动作或心情"
  }
}
```

- [ ] **Step 4: Rebuild and verify**

```bash
conda run -n agent python nodes_eval/execution_eval/build_dataset.py
conda run -n agent pytest tests/test_execution_eval_dataset_builder.py::test_chit_chat_shard_has_at_least_20_cases -v
```

- [ ] **Step 5: Commit**

```bash
git add nodes_eval/execution_eval/datasets/chit_chat_cases.json \
        nodes_eval/execution_eval/golden_dataset.json \
        tests/test_execution_eval_dataset_builder.py
git commit -m "test(eval): expand chit-chat execution cases"
```

### Task 10: Rebuild The Combined Dataset And Tighten Top-Level Tests

**Files:**
- Modify: `nodes_eval/execution_eval/golden_dataset.json`
- Modify: `tests/test_execution_eval.py`
- Modify: `tests/test_execution_eval_dataset_builder.py`

- [ ] **Step 1: Replace the obsolete 14-case assertion**

Change this:

```python
def test_execution_golden_dataset_has_14_cases():
    cases = load_cases(DATASET_PATH)
    assert len(cases) == 14
```

To this:

```python
def test_execution_golden_dataset_has_large_node_coverage():
    cases = load_cases(DATASET_PATH)
    assert len(cases) >= 120
```

- [ ] **Step 2: Add explicit per-node coverage assertions**

```python
from collections import Counter


def test_execution_golden_dataset_has_minimum_cases_per_target_node():
    cases = load_cases(DATASET_PATH)
    counts = Counter(case.node_name for case in cases)

    assert counts["consultant"] >= 20
    assert counts["interviewer"] >= 20
    assert counts["high_value"] >= 20
    assert counts["low_budget"] >= 20
    assert counts["art"] >= 20
    assert counts["chit_chat"] >= 20
```

- [ ] **Step 3: Add one golden-case smoke assertion for the new special contracts**

```python
def test_chit_chat_dataset_contains_parenthetical_emotion_guardrails():
    cases = load_cases(DATASET_PATH)
    chat_cases = [case for case in cases if case.node_name == "chit_chat"]
    assert any("forbidden_regexes" in case.expected.model_dump(exclude_none=True) for case in chat_cases)
```

- [ ] **Step 4: Rebuild the combined dataset**

Run:

```bash
conda run -n agent python nodes_eval/execution_eval/build_dataset.py
```

Expected: combined `golden_dataset.json` is regenerated in sorted `case_id` order.

- [ ] **Step 5: Run focused dataset tests**

Run:

```bash
conda run -n agent pytest \
  tests/test_execution_eval.py \
  tests/test_execution_eval_dataset_builder.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add nodes_eval/execution_eval/golden_dataset.json \
        tests/test_execution_eval.py \
        tests/test_execution_eval_dataset_builder.py
git commit -m "test(eval): enforce expanded execution dataset coverage"
```

### Task 11: Smoke The Expanded Eval Pipeline

**Files:**
- Modify: `nodes_eval/execution_eval/README.md`

- [ ] **Step 1: Update README for shard build + targeted eval loops**

Add these commands:

```bash
python nodes_eval/execution_eval/build_dataset.py
python nodes_eval/execution_eval/run_eval.py --tag consultant --limit 5 --llm deepseek --no-judge
python nodes_eval/execution_eval/run_eval.py --tag chit_chat --limit 5 --llm qwen
python nodes_eval/execution_eval/run_eval.py --tag interviewer --limit 5 --llm gemini_flash
```

- [ ] **Step 2: Run a deterministic smoke without judge**

Run:

```bash
conda run -n agent env PYTHONPATH=. python nodes_eval/execution_eval/run_eval.py --tag consultant --limit 5 --llm deepseek --no-judge --output-json /tmp/execution_consultant_smoke.json
```

Expected: command completes, writes JSON, no schema errors.

- [ ] **Step 3: Run a judge-enabled smoke for one special node**

Run:

```bash
conda run -n agent env PYTHONPATH=. python nodes_eval/execution_eval/run_eval.py --tag chit_chat --limit 5 --llm deepseek --output-json /tmp/execution_chitchat_smoke.json
```

Expected: command completes, no protocol errors, failure tags visible in output JSON.

- [ ] **Step 4: Run the full execution eval test suite**

Run:

```bash
conda run -n agent pytest tests/test_execution_eval.py tests/test_eval_entrypoints.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nodes_eval/execution_eval/README.md
git commit -m "docs(eval): document expanded execution dataset workflow"
```

### Task 12: Final Review And Landing

**Files:**
- Modify: only files touched above

- [ ] **Step 1: Run the final verification batch**

```bash
conda run -n agent pytest \
  tests/test_execution_eval.py \
  tests/test_execution_eval_dataset_builder.py \
  tests/test_eval_entrypoints.py -v
```

Expected: PASS.

- [ ] **Step 2: Generate a quick dataset summary for the PR description**

```bash
conda run -n agent python - <<'PY'
import json
from collections import Counter
from pathlib import Path
cases = json.loads(Path('nodes_eval/execution_eval/golden_dataset.json').read_text())
print('total_cases=', len(cases))
print('node_counts=', dict(Counter(item['node_name'] for item in cases)))
PY
```

- [ ] **Step 3: Request code review before merge**

```bash
BASE_SHA=$(git rev-parse origin/main)
HEAD_SHA=$(git rev-parse HEAD)
echo "$BASE_SHA $HEAD_SHA"
```

Then run the `requesting-code-review` skill against that range.

- [ ] **Step 4: Commit any review fixes**

```bash
git add -A
git commit -m "fix(eval): address execution dataset review feedback"
```

- [ ] **Step 5: Push the branch**

```bash
git push -u origin HEAD
```

---

## Self-Review

**Spec coverage**
- Need 20-50 cases per execution node: covered by shard-per-node tasks for `consultant`, `interviewer`, `high_value`, `low_budget`, `art`, `chit_chat`
- Need six subagents with isolated context: covered by shard split, worker ownership table, and per-node task decomposition
- `consultant` special because it uses Excel knowledge base: covered by `required_context_terms` and consultant-specific buckets
- `interviewer` special because it asks questions: covered by missing-field buckets and anti-loop cases
- `chit_chat` special because models output bracketed emotions: covered by `forbidden_regexes` benchmark extension and dedicated dataset cases

**Placeholder scan**
- No `TODO`/`TBD`
- Every code-changing task includes concrete code blocks or commands
- All test commands are explicit and runnable in conda env `agent`

**Type consistency**
- Dataset builder emits the existing `golden_dataset.json` shape
- `required_context_terms` and `forbidden_regexes` are consistently named across benchmark, tests, and dataset examples
- Node shard filenames match the worker ownership table and per-node minimum assertions

Plan complete and saved to `docs/superpowers/plans/2026-04-07-execution-eval-dataset-expansion.md`. Two execution options:

1. Subagent-Driven (recommended) - I dispatch a fresh subagent per task, review between tasks, fast iteration
2. Inline Execution - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
