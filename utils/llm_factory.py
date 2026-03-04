import os
from typing import Optional
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
    
    if primary and backup:
        return primary.with_fallbacks([backup])
    return primary or backup # 哪个有 Key 就用哪个

def get_frontend_llm(temperature: float = 0.7) -> BaseChatModel:
    primary = create_base_llm(DEEPSEEK_MODEL, temperature=temperature)
    backup = create_base_llm(GEMINI_PRO, temperature=temperature)
    
    if primary and backup:
        return primary.with_fallbacks([backup])
    return primary or backup
