"""
Runtime feature flags with Redis-backed overrides and settings fallback.
"""

from typing import Dict, Optional
from app.core.cache import cache_service
from app.core.config import settings

_FLAG_KEYS = {
    "knowledge_graph_enabled": "feature:knowledge_graph_enabled",
    "summarization_enabled": "feature:summarization_enabled",
    "auto_summarize_on_process": "feature:auto_summarize_on_process",
    "unsafe_code_execution_enabled": "feature:unsafe_code_execution_enabled",
}

_STR_KEYS = {
    "llm_default_model": "feature:llm_default_model",
    "llm_provider_fast": "feature:llm_provider_fast",
    "llm_model_fast": "feature:llm_model_fast",
    "llm_provider_balanced": "feature:llm_provider_balanced",
    "llm_model_balanced": "feature:llm_model_balanced",
    "llm_provider_deep": "feature:llm_provider_deep",
    "llm_model_deep": "feature:llm_model_deep",
    "ai_hub_enabled_eval_templates": "feature:ai_hub_enabled_eval_templates",
    "ai_hub_enabled_dataset_presets": "feature:ai_hub_enabled_dataset_presets",
    "ai_hub_customer_profile": "feature:ai_hub_customer_profile",
    "unsafe_code_exec_backend": "feature:unsafe_code_exec_backend",
    "unsafe_code_exec_docker_image": "feature:unsafe_code_exec_docker_image",
}


async def get_flag(name: str) -> Optional[bool]:
    key = _FLAG_KEYS.get(name)
    if not key:
        return None
    val = await cache_service.get(key)
    if isinstance(val, bool):
        return val
    # Fallback to settings
    if name == "knowledge_graph_enabled":
        return bool(getattr(settings, "KNOWLEDGE_GRAPH_ENABLED", True))
    if name == "summarization_enabled":
        return bool(getattr(settings, "SUMMARIZATION_ENABLED", True))
    if name == "auto_summarize_on_process":
        return bool(getattr(settings, "AUTO_SUMMARIZE_ON_PROCESS", False))
    if name == "unsafe_code_execution_enabled":
        return bool(getattr(settings, "ENABLE_UNSAFE_CODE_EXECUTION", False))
    return None


async def set_flag(name: str, value: bool) -> bool:
    key = _FLAG_KEYS.get(name)
    if not key:
        return False
    return await cache_service.set(key, bool(value))


async def get_flags() -> Dict[str, bool]:
    return {
        "knowledge_graph_enabled": bool(await get_flag("knowledge_graph_enabled")),
        "summarization_enabled": bool(await get_flag("summarization_enabled")),
        "auto_summarize_on_process": bool(await get_flag("auto_summarize_on_process")),
        "unsafe_code_execution_enabled": bool(await get_flag("unsafe_code_execution_enabled")),
    }


async def get_str(name: str) -> Optional[str]:
    key = _STR_KEYS.get(name)
    if not key:
        return None
    val = await cache_service.get(key)
    if isinstance(val, (str, bytes)):
        return val.decode() if isinstance(val, bytes) else val
    return None


async def set_str(name: str, value: str) -> bool:
    key = _STR_KEYS.get(name)
    if not key:
        return False
    return await cache_service.set(key, value)
