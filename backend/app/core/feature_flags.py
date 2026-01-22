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
}

_STR_KEYS = {
    "llm_default_model": "feature:llm_default_model",
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
