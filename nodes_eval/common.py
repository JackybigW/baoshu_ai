from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.llm_factory import (
    get_backend_llm,
    get_frontend_llm,
    get_llm,
    get_llm_descriptor,
    list_supported_llms,
    normalize_message_content,
    resolve_llm_key,
)


@dataclass(frozen=True)
class EvalModelConfig:
    requested_id: str
    canonical_id: str
    label: str
    provider: str
    resolved_model: str
    llm: Any

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requested_id": self.requested_id,
            "canonical_id": self.canonical_id,
            "label": self.label,
            "provider": self.provider,
            "resolved_model": self.resolved_model,
        }


def slugify(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip())
    text = text.strip("._").lower()
    return text or "unknown_llm"


def _build_model_configs(
    model_ids: Sequence[str],
    *,
    default_aliases: Iterable[str],
    default_label: str,
    default_provider: str,
    default_resolved_model: str,
    default_builder: Callable[[float], Any],
    explicit_temperature: float,
) -> List[EvalModelConfig]:
    requested_models = list(model_ids) or [default_label]
    configs: List[EvalModelConfig] = []
    seen_labels: set[str] = set()
    default_alias_set = {alias.strip().lower().replace("-", "_") for alias in default_aliases}

    for requested_id in requested_models:
        normalized = requested_id.strip()
        lowered = normalized.lower().replace("-", "_")

        if lowered in default_alias_set:
            llm = default_builder(explicit_temperature)
            if llm is None:
                raise SystemExit(
                    f"{default_label} 无法初始化。请检查本地环境变量是否完整。"
                )
            if default_label in seen_labels:
                continue
            seen_labels.add(default_label)
            configs.append(
                EvalModelConfig(
                    requested_id=normalized,
                    canonical_id=default_label,
                    label=default_label,
                    provider=default_provider,
                    resolved_model=default_resolved_model,
                    llm=llm,
                )
            )
            continue

        try:
            descriptor = get_llm_descriptor(normalized)
        except ValueError as exc:
            raise SystemExit(
                f"{exc}. 当前支持: {', '.join(list_supported_llms())}。"
            ) from exc

        llm = get_llm(normalized, temperature=explicit_temperature, allow_missing=True)
        if llm is None:
            canonical = descriptor["canonical_id"]
            raise SystemExit(
                f"LLM `{normalized}` 无法初始化。请检查相关环境变量是否完整。"
                f" 当前支持: {', '.join(list_supported_llms())}。"
                f" 需要的 key/base_url 可参考 `{canonical}` 对应配置。"
            )

        label = slugify(descriptor["canonical_id"])
        if label in seen_labels:
            continue
        seen_labels.add(label)
        configs.append(
            EvalModelConfig(
                requested_id=normalized,
                canonical_id=resolve_llm_key(normalized),
                label=label,
                provider=descriptor["provider"],
                resolved_model=descriptor["resolved_model"],
                llm=llm,
            )
        )
    return configs


def _resolved_chain_model_names(model_ids: Sequence[str]) -> str:
    parts: List[str] = []
    for model_id in model_ids:
        descriptor = get_llm_descriptor(model_id)
        parts.append(descriptor["resolved_model"])
    return " -> ".join(parts)


def build_backend_model_configs(model_ids: Sequence[str], *, temperature: float = 0.0) -> List[EvalModelConfig]:
    return _build_model_configs(
        model_ids,
        default_aliases=("backend", "default", "backend_default"),
        default_label="backend_default",
        default_provider="fallback_chain",
        default_resolved_model=_resolved_chain_model_names(("deepseek", "glm", "doubao")),
        default_builder=get_backend_llm,
        explicit_temperature=temperature,
    )


def build_frontend_model_configs(model_ids: Sequence[str], *, temperature: float = 0.7) -> List[EvalModelConfig]:
    return _build_model_configs(
        model_ids,
        default_aliases=("frontend", "default", "frontend_default"),
        default_label="frontend_default",
        default_provider="fallback_chain",
        default_resolved_model=_resolved_chain_model_names(("deepseek", "glm", "doubao")),
        default_builder=get_frontend_llm,
        explicit_temperature=temperature,
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_key_value_log(
    *,
    title: str,
    kv_pairs: Sequence[tuple[str, Any]],
    log_paths: Sequence[Path],
) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"[{timestamp}] {title}"]
    for key, value in kv_pairs:
        if isinstance(value, (dict, list)):
            rendered = json.dumps(value, ensure_ascii=False)
        else:
            rendered = value
        lines.append(f"{key}={rendered}")
    lines.append("")

    content = "\n".join(str(line) for line in lines)
    for log_path in log_paths:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(content)


def message_from_dict(item: Dict[str, Any]) -> BaseMessage:
    msg_type = str(item.get("type", "human")).strip().lower()
    content = item.get("content", "")
    if msg_type == "human":
        return HumanMessage(content=content)
    if msg_type == "system":
        return SystemMessage(content=content)
    if msg_type == "tool":
        return ToolMessage(
            content=content,
            tool_call_id=item.get("tool_call_id", "tool_call_eval"),
        )
    if msg_type == "ai":
        message = AIMessage(content=content, tool_calls=item.get("tool_calls", []))
        if item.get("id"):
            message.id = item["id"]
        return message
    raise ValueError(f"Unsupported message type: {msg_type}")


def messages_from_dicts(items: Sequence[Dict[str, Any]]) -> List[BaseMessage]:
    return [message_from_dict(item) for item in items]


def dump_message(message: BaseMessage) -> Dict[str, Any]:
    if isinstance(message, HumanMessage):
        return {"type": "human", "content": normalize_message_content(message.content)}
    if isinstance(message, SystemMessage):
        return {"type": "system", "content": normalize_message_content(message.content)}
    if isinstance(message, ToolMessage):
        return {
            "type": "tool",
            "content": normalize_message_content(message.content),
            "tool_call_id": getattr(message, "tool_call_id", ""),
        }
    payload: Dict[str, Any] = {"type": "ai", "content": normalize_message_content(message.content)}
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        payload["tool_calls"] = tool_calls
    if getattr(message, "id", None):
        payload["id"] = message.id
    return payload


def dump_messages(messages: Sequence[BaseMessage]) -> List[Dict[str, Any]]:
    return [dump_message(message) for message in messages]


def join_message_text(messages: Sequence[BaseMessage]) -> str:
    parts = [normalize_message_content(getattr(message, "content", "")).strip() for message in messages]
    return " ||| ".join(part for part in parts if part)


def has_tool_call(messages: Sequence[BaseMessage]) -> bool:
    return any(bool(getattr(message, "tool_calls", None)) for message in messages)
