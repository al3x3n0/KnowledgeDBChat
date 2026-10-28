"""Shared LLM routing resolution helpers.

This module centralizes routing config coercion + tier attempt planning so that:
- runtime (`LLMService.generate_response`) and
- preview endpoints
use the exact same precedence/logic.

The core runtime behavior lives in `LLMService._generate_response_once`; this module
implements the same selection logic for provider/model/api_url without making requests.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from app.core.config import settings

FeatureGetStr = Callable[[str], Awaitable[Any]]


def prefers_deepseek(
    prefer: bool, *, effective_provider: str, provider_chosen: bool
) -> bool:
    """Whether a caller's ``prefer_deepseek`` sends this call to DeepSeek.

    Heavy work (long summaries, context compression) asked for DeepSeek
    whenever a DeepSeek key existed, whatever provider was configured, so a
    deployment pointed at a local endpoint still sent those calls out. The
    preference now applies only where it was meant to: the deployment's own
    Ollama, with nobody having chosen a provider.
    """
    return (
        bool(prefer)
        and not provider_chosen
        and effective_provider == "ollama"
        and bool(getattr(settings, "DEEPSEEK_API_KEY", None))
    )


def _to_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, (bytes, bytearray)):
        try:
            v = v.decode()
        except Exception:
            v = str(v)
    s = str(v).strip()
    return s or None


def _clamp_int(
    v: Any, *, min_v: int, max_v: int, default: Optional[int]
) -> Optional[int]:
    if v is None:
        return default
    try:
        iv = int(v)
    except Exception:
        return default
    return max(min_v, min(iv, max_v))


def coerce_routing_config(routing: Any) -> Dict[str, Any]:
    """Coerce routing config into canonical keys + validated values.

    Accepts canonical keys:
      - tier, fallback_tiers, timeout_seconds, max_tokens_cap, cooldown_seconds
    and legacy/alternate keys:
      - llm_tier, llm_fallback_tiers, llm_timeout_seconds, llm_max_tokens_cap,
        llm_unhealthy_cooldown_seconds
    """
    cfg = routing if isinstance(routing, dict) else {}

    tier = _to_str(cfg.get("tier") or cfg.get("llm_tier"))
    tier = tier.lower() if tier else None

    fb = cfg.get("fallback_tiers")
    if fb is None:
        fb = cfg.get("llm_fallback_tiers")
    if not isinstance(fb, list):
        fb = []
    fallback_tiers = [(_to_str(x) or "").lower() for x in fb]
    fallback_tiers = [t for t in fallback_tiers if t]

    timeout_seconds = cfg.get("timeout_seconds")
    if timeout_seconds is None:
        timeout_seconds = cfg.get("llm_timeout_seconds")
    timeout_seconds = _clamp_int(timeout_seconds, min_v=2, max_v=600, default=None)

    max_tokens_cap = cfg.get("max_tokens_cap")
    if max_tokens_cap is None:
        max_tokens_cap = cfg.get("llm_max_tokens_cap")
    max_tokens_cap = _clamp_int(max_tokens_cap, min_v=64, max_v=20000, default=None)

    cooldown_seconds = cfg.get("cooldown_seconds")
    if cooldown_seconds is None:
        cooldown_seconds = cfg.get("llm_unhealthy_cooldown_seconds")
    cooldown_seconds = _clamp_int(cooldown_seconds, min_v=5, max_v=3600, default=60)

    return {
        "tier": tier,
        "fallback_tiers": fallback_tiers,
        "timeout_seconds": timeout_seconds,
        "max_tokens_cap": max_tokens_cap,
        "cooldown_seconds": cooldown_seconds,
    }


def compute_attempt_tiers(
    *, tier: Optional[str], fallback_tiers: List[str]
) -> List[Optional[str]]:
    """Compute tier attempt order.

    Matches `LLMService.generate_response` logic: requested tier first, then fallback tiers
    excluding duplicates; if nothing is set, attempt a single default (None) tier.
    """
    if tier:
        return [tier] + [t for t in fallback_tiers if t and t != tier]
    if fallback_tiers:
        return list(fallback_tiers)
    return [None]


async def resolve_tier_overrides(
    get_feature_str: FeatureGetStr, tier: Optional[str]
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve provider/model for a tier from feature flags."""
    if not tier:
        return None, None
    t = str(tier).strip().lower()
    if t not in {"fast", "balanced", "deep"}:
        return None, None

    try:
        p = await get_feature_str(f"llm_provider_{t}")
        m = await get_feature_str(f"llm_model_{t}")
        p = _to_str(p)
        m = _to_str(m)
        return (p.lower() if p else None), m
    except Exception:
        return None, None


async def resolve_feature_default_model(
    get_feature_str: FeatureGetStr,
) -> Optional[str]:
    try:
        m = await get_feature_str("llm_default_model")
        return _to_str(m)
    except Exception:
        return None


def resolve_effective_provider_model(
    *,
    system_provider: str,
    system_default_model: str,
    feature_default_model: Optional[str],
    user_settings: Any,
    task_type: str,
    model: Optional[str],
    provider_override: Optional[str],
    api_url_override: Optional[str],
    api_key_override: Optional[str] = None,
    prefer_deepseek: bool = False,
) -> Dict[str, Any]:
    """Resolve effective provider/model/api_url exactly like runtime.

    Mirrors `LLMService._generate_response_once` selection logic up to the point where
    it chooses which provider branch to call.
    """
    effective_provider = (system_provider or "deepseek").strip().lower()
    effective_api_url: Optional[str] = None
    effective_api_key: Optional[str] = None

    provider_chosen = False

    # User settings apply first.
    try:
        if user_settings and getattr(user_settings, "has_custom_settings")():
            conn = user_settings.connection_for_task(task_type)
            if conn.provider:
                effective_provider = str(conn.provider).strip().lower()
            provider_chosen = bool(conn.chosen_by_user)
            if conn.model:
                model = str(conn.model)
            if conn.api_url:
                effective_api_url = str(conn.api_url).strip() or None
            if conn.api_key:
                effective_api_key = str(conn.api_key)
    except Exception:
        pass

    # Explicit overrides (per-agent routing / per-request).
    if provider_override:
        effective_provider = str(provider_override).strip().lower()
        provider_chosen = True
    if api_url_override:
        effective_api_url = str(api_url_override).strip() or None
    if api_key_override:
        effective_api_key = str(api_key_override)

    sdk_names = {"anthropic", "qwen", "kimi", "glm", "openai_compatible"}
    if effective_provider == "openai" and not effective_api_url:
        sdk_names.add("openai")

    # The system default model is for Ollama and a bare custom URL; every
    # other provider falls back to its own <PROVIDER>_MODEL.
    if not model and (
        effective_provider == "ollama"
        or (effective_api_url and effective_provider not in sdk_names)
    ):
        model = feature_default_model or system_default_model or None
    model = (model or "").strip() or None

    # Determine which provider branch runtime will use.
    if effective_provider in sdk_names:
        provider_used = effective_provider
    elif effective_api_url:
        provider_used = "custom"
    elif effective_provider == "deepseek" or prefers_deepseek(
        prefer_deepseek,
        effective_provider=effective_provider,
        provider_chosen=provider_chosen,
    ):
        provider_used = "deepseek"
    else:
        provider_used = effective_provider
    model_used = model

    return {
        "effective_provider": effective_provider,
        "provider_used": provider_used,
        "effective_model": model_used,
        "api_url": effective_api_url,
        "api_key": effective_api_key,
    }
