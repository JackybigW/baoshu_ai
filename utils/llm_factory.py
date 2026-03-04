import os
import typing
import builtins
from typing import Optional, List, Any, Union, Dict, Sequence, Literal, Callable, Iterable, Type
from pydantic import BaseModel
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage

# 🛠️ 终极补丁：强制污染全局 builtins 命名空间
# 这是由于 LangChain 的 RunnableWithFallbacks 在委托 getattr 时会调用 get_type_hints，
# 而 get_type_hints 在评估 ForwardRef 时对作用域极其敏感。
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

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

DEEPSEEK_MODEL = "deepseek-chat"
GEMINI_PRO = "gemini-3.1-pro-preview"
GEMINI_FLASH = "gemini-3.1-flash-lite-preview"

def create_base_llm(model: str, temperature: float) -> Optional[BaseChatModel]:
    """创建基础模型实例，若缺少 Key 则返回 None"""
    provider = "google_genai" if "gemini" in model else "deepseek"
    api_key = os.getenv("GOOGLE_API_KEY") if provider == "google_genai" else os.getenv("DEEPSEEK_API_KEY")
    
    # 如果没有 Key，我们不在这里崩溃，而是返回 None 供后面过滤
    if not api_key:
        return None
        
    return init_chat_model(
        model,
        model_provider=provider,
        api_key=api_key,
        temperature=temperature,
        configurable_fields=("model", "model_provider", "api_key", "temperature")
    )

def get_backend_llm(temperature: float = 0) -> BaseChatModel:
    primary = create_base_llm(DEEPSEEK_MODEL, temperature=temperature)
    backup = create_base_llm(GEMINI_FLASH, temperature=temperature)
    
    # 🔥 核心修正：仅当两者都存在时才使用 Fallback，
    # 否则直接返回单个模型，避免 RunnableWithFallbacks 在 getattr 时触发 get_type_hints 导致的作用域 NameError
    if primary and backup:
        return primary.with_fallbacks([backup])
    return primary or backup

def get_frontend_llm(temperature: float = 0.7) -> BaseChatModel:
    primary = create_base_llm(DEEPSEEK_MODEL, temperature=temperature)
    backup = create_base_llm(GEMINI_PRO, temperature=temperature)
    
    if primary and backup:
        return primary.with_fallbacks([backup])
    return primary or backup
