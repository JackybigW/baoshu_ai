import os
import builtins
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Type, Union

from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

# LangChain fallback 在部分场景会走到 get_type_hints，这里继续保留已有补丁。
builtins.BaseModel = BaseModel
builtins.BaseTool = BaseTool
builtins.BaseMessage = BaseMessage
builtins.Sequence = Sequence
builtins.Callable = Callable
builtins.Iterable = Iterable
builtins.Any = Any
builtins.Union = Union
builtins.Dict = Dict
builtins.List = List
builtins.Optional = Optional
builtins.Type = Type
builtins.Literal = Literal

_ = load_dotenv(find_dotenv())


DEFAULT_MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "deepseek": {
        "aliases": ["deepseek", "ds", "backend_primary", "deepseek_volc", "volc_deepseek"],
        "provider": "openai",
        "api_key_env": "DOUBAO_API_KEY",
        "base_url_env": "DOUBAO_BASE_URL",
        "model_env": "DEEPSEEK_MODEL",
        "default_model": "deepseek-v3-2-251201",
    },
    "deepseek_official": {
        "aliases": ["deepseek_official", "official_deepseek", "deepseek_native"],
        "provider": "deepseek",
        "api_key_env": "DEEPSEEK_API_KEY",
        "model_env": "DEEPSEEK_OFFICIAL_MODEL",
        "default_model": "deepseek-chat",
    },
    "gemini_flash": {
        "aliases": ["gemini", "gemini_flash", "flash", "google_flash", "frontend_backup", "backend_backup"],
        "provider": "google_genai",
        "api_key_env": "GOOGLE_API_KEY",
        "model_env": "GEMINI_FLASH_MODEL",
        "default_model": "gemini-3.1-flash-lite-preview",
    },
    "gemini_pro": {
        "aliases": ["gemini_pro", "pro", "google_pro"],
        "provider": "google_genai",
        "api_key_env": "GOOGLE_API_KEY",
        "model_env": "GEMINI_PRO_MODEL",
        "default_model": "gemini-3.1-pro-preview",
    },
    "qwen": {
        "aliases": ["qwen", "tongyi", "qwen_plus"],
        "provider": "openai",
        "api_key_env": "QWEN_API_KEY",
        "api_key_env_fallbacks": ["DASHSCOPE_API_KEY"],
        "base_url_env": "QWEN_BASE_URL",
        "model_env": "QWEN_MODEL",
        "default_model": "qwen-plus",
    },
    "glm": {
        "aliases": ["glm", "zhipu", "zhipu_glm"],
        "provider": "openai",
        "api_key_env": "DOUBAO_API_KEY",
        "base_url_env": "DOUBAO_BASE_URL",
        "model_env": "GLM_MODEL",
        "default_model": "glm-4-7-251222",
    },
    "doubao": {
        "aliases": ["doubao", "dou_bao", "ark", "volcengine"],
        "provider": "openai",
        "api_key_env": "DOUBAO_API_KEY",
        "base_url_env": "DOUBAO_BASE_URL",
        "model_env": "DOUBAO_MODEL",
        "default_model": "doubao-seed-2-0-pro-260215",
    },
    "doubao_lite": {
        "aliases": ["doubao_lite", "doubao-lite", "doubao_light", "lite"],
        "provider": "openai",
        "api_key_env": "DOUBAO_API_KEY",
        "base_url_env": "DOUBAO_BASE_URL",
        "model_env": "DOUBAO_LITE_MODEL",
        "default_model": "doubao-seed-2-0-lite-260215",
    },
}


DEEPSEEK_MODEL = "deepseek"
GEMINI_PRO = "gemini_pro"
GEMINI_FLASH = "gemini_flash"


def _normalize_model_key(model_id: str) -> str:
    normalized = model_id.strip().lower().replace("-", "_")
    for canonical_name, spec in DEFAULT_MODEL_SPECS.items():
        aliases = {canonical_name, *(alias.lower().replace("-", "_") for alias in spec.get("aliases", []))}
        if normalized in aliases:
            return canonical_name
    raise ValueError(f"没配置这个模型: {model_id}")


def _resolve_model_name(spec: Dict[str, Any], explicit_model: Optional[str] = None) -> str:
    if explicit_model:
        return explicit_model
    model_env = spec.get("model_env")
    if model_env:
        configured = os.getenv(model_env, "").strip()
        if configured:
            return configured
    return spec["default_model"]


def _get_first_configured_env(primary_env: str, fallback_envs: Optional[List[str]] = None) -> str:
    env_names = [primary_env, *(fallback_envs or [])]
    for env_name in env_names:
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    return ""


def _create_llm(
    model_id: str,
    *,
    temperature: float = 0,
    model_name: Optional[str] = None,
    allow_missing: bool = True,
    **kwargs: Any,
) -> Optional[BaseChatModel]:
    canonical_name = _normalize_model_key(model_id)
    return _create_llm_from_canonical(
        canonical_name,
        temperature=temperature,
        model_name=model_name,
        allow_missing=allow_missing,
        **kwargs,
    )


def _create_llm_from_canonical(
    canonical_name: str,
    *,
    temperature: float = 0,
    model_name: Optional[str] = None,
    allow_missing: bool = True,
    **kwargs: Any,
) -> Optional[BaseChatModel]:
    spec = DEFAULT_MODEL_SPECS[canonical_name]

    provider = spec["provider"]
    api_key = _get_first_configured_env(
        spec["api_key_env"],
        spec.get("api_key_env_fallbacks"),
    )

    if not api_key:
        if allow_missing:
            return None
        raise RuntimeError(f"{canonical_name} 未配置 API Key: {spec['api_key_env']}")

    init_kwargs: Dict[str, Any] = {
        "model": _resolve_model_name(spec, explicit_model=model_name),
        "model_provider": provider,
        "api_key": api_key,
        "temperature": temperature,
        "configurable_fields": ("model", "model_provider", "api_key", "temperature"),
    }

    base_url_env = spec.get("base_url_env")
    if base_url_env:
        base_url = _get_first_configured_env(
            base_url_env,
            spec.get("base_url_env_fallbacks"),
        )
        if not base_url:
            if allow_missing:
                return None
            raise RuntimeError(f"{canonical_name} 未配置 Base URL: {base_url_env}")
        init_kwargs["base_url"] = base_url

    init_kwargs.update(kwargs)
    return init_chat_model(**init_kwargs)


def create_base_llm(model: str, temperature: float) -> Optional[BaseChatModel]:
    """兼容旧接口：按模型别名创建基础实例，缺少配置时返回 None。"""
    return _create_llm(model, temperature=temperature, allow_missing=True)


def get_llm(
    model_id: str,
    temperature: float = 0,
    *,
    model_name: Optional[str] = None,
    allow_missing: bool = True,
    **kwargs: Any,
) -> Optional[BaseChatModel]:
    """
    通用 LLM 工厂。

    用法示例:
    - get_llm("deepseek")
    - get_llm("qwen")
    - get_llm("glm", model_name="glm-4.5")
    - get_llm("doubao", temperature=0.7)
    """
    canonical_name = _normalize_model_key(model_id)
    if canonical_name == "deepseek":
        return get_deepseek_llm(
            temperature=temperature,
            model_name=model_name,
            allow_missing=allow_missing,
            **kwargs,
        )
    return _create_llm_from_canonical(
        canonical_name,
        temperature=temperature,
        model_name=model_name,
        allow_missing=allow_missing,
        **kwargs,
    )


def _combine_with_fallbacks(
    primary: Optional[BaseChatModel],
    backups: List[Optional[BaseChatModel]],
) -> Optional[BaseChatModel]:
    valid_backups = [llm for llm in backups if llm]
    if primary and valid_backups:
        return primary.with_fallbacks(valid_backups)
    if primary:
        return primary
    if valid_backups:
        return valid_backups[0]
    return None


def get_deepseek_llm(
    temperature: float = 0,
    *,
    model_name: Optional[str] = None,
    allow_missing: bool = True,
    **kwargs: Any,
) -> Optional[BaseChatModel]:
    primary = _create_llm_from_canonical(
        "deepseek",
        temperature=temperature,
        model_name=model_name,
        allow_missing=allow_missing,
        **kwargs,
    )
    backup = _create_llm_from_canonical(
        "deepseek_official",
        temperature=temperature,
        allow_missing=allow_missing,
        **kwargs,
    )
    return _combine_with_fallbacks(primary, [backup])


def get_backend_llm(temperature: float = 0) -> Optional[BaseChatModel]:
    primary = get_deepseek_llm(temperature=temperature)
    backup = get_llm(GEMINI_FLASH, temperature=temperature)
    backup_2 = get_llm("doubao", temperature=temperature)
    return _combine_with_fallbacks(primary, [backup, backup_2])


def get_frontend_llm(temperature: float = 0.7) -> Optional[BaseChatModel]:
    primary = get_deepseek_llm(temperature=temperature)
    backup = get_llm(GEMINI_PRO, temperature=temperature)
    backup_2 = get_llm("doubao", temperature=temperature)
    return _combine_with_fallbacks(primary, [backup, backup_2])
