"""
LLM service for generating responses using local or external LLMs.

Supported providers:
- Ollama (local)
- DeepSeek (external, OpenAI-compatible chat API)
- OpenAI (external, official SDK)
- Anthropic (external, official SDK)
- Qwen via DashScope compatible mode (external)
- Kimi / Moonshot AI (external, OpenAI-compatible)
- Custom OpenAI-compatible APIs (user-configured)

Two generation paths:
- ``generate_response``: legacy prompted-text completion (returns str)
- ``generate_structured``: native tool calling / schema-constrained output
  via ``app.services.llm_providers`` (returns ``LLMCompletion``)
"""

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import httpx
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings
from app.models.llm_usage import LLMUsageEvent
from app.services.llm_routing import (
    coerce_routing_config,
    compute_attempt_tiers,
    resolve_tier_overrides,
)
from app.utils.exceptions import LLMServiceError
from app.services import llm_truncation

_LLM_SEMAPHORE = asyncio.Semaphore(settings.LLM_MAX_CONCURRENCY)

#: The last completion's reasoning, per asyncio task.
#:
#: The text is wanted by the snapshot recorder, which sits several frames above
#: the client that receives it, and _generate_response_once returns only a
#: string. A ContextVar carries it without threading a new return type through
#: every provider path, and without the cross-talk an instance attribute would
#: have under LLM_MAX_CONCURRENCY: each task gets its own copy.
_LAST_REASONING: ContextVar[Optional[Tuple[str, Optional[int]]]] = ContextVar(
    "llm_last_reasoning", default=None
)

#: Cache accounting from the most recent completion, carried the same way the
#: reasoning is: the recorder runs a layer above the provider call and has no
#: other way to see the raw usage block.
_LAST_CACHE: ContextVar[Optional[Tuple[Optional[int], Optional[int]]]] = ContextVar(
    "llm_last_cache", default=None
)


# Supported task types for per-task model configuration.
# Keep this list in sync with `backend/app/api/endpoints/users.py`.
LLM_TASK_TYPES = [
    "chat",
    "title_generation",
    "summarization",
    "query_expansion",
    "memory_extraction",
    "workflow_synthesis",
    # Agent / jobs
    "code_agent",
    "research_engineer_scientist",
    "latex_reviewer_critic",
    # Knowledge graph / extraction
    "knowledge_extraction",
    # Presentation generation
    "presentation_outline",
    "presentation_diagram",
    "presentation_slide",
]


def _usage_user_id(value: Any) -> Optional[UUID]:
    """A user id as the UUID columns in this module actually type it.

    Callers hand this in as a string -- the agent loop passes
    ``str(job.user_id)`` -- while ``LLMUsageEvent.user_id`` and
    ``LLMCallSnapshot.user_id`` are both ``UUID(as_uuid=True)``. PostgreSQL
    tolerates the string, so this ran for a long time without complaint;
    SQLAlchemy's non-native-UUID path calls ``value.hex`` and raises
    AttributeError, which is what an agent loop running against SQLite hits on
    its first real LLM call. The suite never saw it because it never makes one.

    Used by both writers. Fixing only the usage event left the snapshot to fail
    the same way the moment snapshots were switched on -- the same defect twice,
    found the second time by turning on a feature rather than by reading.
    """
    if isinstance(value, UUID):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return UUID(text)
    except (ValueError, AttributeError, TypeError):
        return None


@dataclass
class UserLLMSettings:
    """User-specific LLM settings that override system defaults."""

    provider: Optional[str] = None  # "ollama", "deepseek", "openai", or custom
    model: Optional[str] = None
    api_url: Optional[str] = None
    api_key: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    task_models: Optional[Dict[str, str]] = None  # Per-task model overrides
    task_providers: Optional[Dict[str, str]] = None  # Per-task provider overrides

    @classmethod
    def from_preferences(cls, prefs) -> "UserLLMSettings":
        """Create UserLLMSettings from a UserPreferences model instance."""
        if prefs is None:
            return cls()
        return cls(
            provider=getattr(prefs, "llm_provider", None),
            model=getattr(prefs, "llm_model", None),
            api_url=getattr(prefs, "llm_api_url", None),
            api_key=getattr(prefs, "llm_api_key", None),
            temperature=getattr(prefs, "llm_temperature", None),
            max_tokens=getattr(prefs, "llm_max_tokens", None),
            task_models=getattr(prefs, "llm_task_models", None),
            task_providers=getattr(prefs, "llm_task_providers", None),
        )

    def has_custom_settings(self) -> bool:
        """Check if any custom settings are configured."""
        return any(
            [
                self.provider,
                self.model,
                self.api_url,
                self.api_key,
                self.temperature is not None,
                self.max_tokens is not None,
                self.task_models,
                self.task_providers,
            ]
        )

    def get_model_for_task(self, task_type: str) -> Optional[str]:
        """
        Get the model to use for a specific task type.

        Args:
            task_type: One of LLM_TASK_TYPES (chat, title_generation, etc.)

        Returns:
            Task-specific model if configured, otherwise falls back to default model
        """
        if self.task_models and task_type in self.task_models:
            return self.task_models[task_type]
        return self.model  # Fall back to default model

    def get_provider_for_task(self, task_type: str) -> Optional[str]:
        """
        Get the provider to use for a specific task type.
        """
        if self.task_providers and task_type in self.task_providers:
            return self.task_providers[task_type]
        return self.provider


def _meta_from_completion(
    data: Dict[str, Any], fallback_model: Optional[str] = None
) -> Dict[str, Any]:
    """Response metadata, with the reasoning set aside where it belongs.

    These models return the chain of thought apart from the answer and charge
    it against max_tokens, so it is paid for whether or not anything reads it.
    Dropping it left an agent's decisions replayable while the thinking behind
    them was not, and made a call that spent its whole budget reasoning look
    simply empty.

    The text goes to a context variable for the snapshot recorder; only its
    size goes into the returned meta, which is written to a usage row on every
    call and would otherwise be dwarfed by it.
    """
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    usage = data.get("usage") or {}
    details = usage.get("completion_tokens_details")
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    reasoning_tokens = (
        details.get("reasoning_tokens") if isinstance(details, dict) else None
    )
    _LAST_REASONING.set((reasoning, reasoning_tokens) if reasoning else None)
    cache_hit, cache_miss = _cache_tokens(usage)
    _LAST_CACHE.set((cache_hit, cache_miss))
    return {
        "id": data.get("id"),
        "model": data.get("model") or fallback_model,
        "usage": data.get("usage"),
        "finish_reason": choice.get("finish_reason"),
        "reasoning_tokens": reasoning_tokens,
        "reasoning_chars": len(reasoning) if reasoning else 0,
        "cache_hit_tokens": cache_hit,
        "cache_miss_tokens": cache_miss,
    }


def _cache_tokens(usage: Dict[str, Any]) -> tuple[Optional[int], Optional[int]]:
    """Prompt tokens served from the provider's cache, and those that were not.

    The prompt is split into a byte-stable prefix and a volatile tail precisely
    so the prefix can be cached, and Anthropic requests carry cache_control
    breakpoints for the same reason -- but nothing ever read back whether any
    of it worked. A cache mechanism whose hit rate is never measured is a
    mechanism nobody can tell apart from a comment.

    Providers spell it three ways and none of them agree:
      DeepSeek    prompt_cache_hit_tokens / prompt_cache_miss_tokens
      OpenAI      prompt_tokens_details.cached_tokens (hits only)
      Anthropic   cache_read_input_tokens / cache_creation_input_tokens

    Returns (hit, miss), either of which is None when the provider said
    nothing. None is not zero here: zero is a measured miss, None is silence,
    and averaging silence as zero would report a healthy cache as broken.
    """
    if not isinstance(usage, dict):
        return None, None

    hit = usage.get("prompt_cache_hit_tokens")
    miss = usage.get("prompt_cache_miss_tokens")
    if hit is not None or miss is not None:
        return _as_int_or_none(hit), _as_int_or_none(miss)

    read = usage.get("cache_read_input_tokens")
    created = usage.get("cache_creation_input_tokens")
    if read is not None or created is not None:
        return _as_int_or_none(read), _as_int_or_none(created)

    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict) and details.get("cached_tokens") is not None:
        cached = _as_int_or_none(details.get("cached_tokens"))
        total = _as_int_or_none(usage.get("prompt_tokens"))
        # The miss is inferred, and only when both halves are known: reporting
        # a miss of "everything" against an unknown total would invent a rate.
        return cached, (
            max(total - cached, 0) if total is not None and cached is not None else None
        )
    return None, None


def _as_int_or_none(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class LLMService:
    """Service for interacting with configured LLM provider."""

    def __init__(self):
        self.provider = (settings.LLM_PROVIDER or "deepseek").lower()
        self.base_url = settings.OLLAMA_BASE_URL
        self.default_model = settings.DEFAULT_MODEL
        # A single client is enough; per-request overrides set timeouts/headers
        self.client = httpx.AsyncClient(timeout=120.0)
        self._unhealthy_until: Dict[str, float] = {}
        self._unhealthy_reason: Dict[str, str] = {}
        self._unhealthy_lock = asyncio.Lock()

    async def _is_healthy(self, key: str) -> bool:
        try:
            now = asyncio.get_event_loop().time()
        except Exception:
            now = 0.0
        async with self._unhealthy_lock:
            until = self._unhealthy_until.get(key)
            if until is None:
                return True
            if now >= float(until):
                self._unhealthy_until.pop(key, None)
                self._unhealthy_reason.pop(key, None)
                return True
            return False

    async def _mark_unhealthy(
        self, key: str, *, cooldown_seconds: int, reason: str
    ) -> None:
        cooldown_seconds = max(5, min(int(cooldown_seconds or 60), 3600))
        try:
            now = asyncio.get_event_loop().time()
        except Exception:
            now = 0.0
        async with self._unhealthy_lock:
            self._unhealthy_until[key] = float(now) + float(cooldown_seconds)
            self._unhealthy_reason[key] = str(reason or "")[:200]

    def _health_key(self, *, provider: str, api_url: Optional[str]) -> str:
        p = (provider or "").strip().lower() or "unknown"
        u = (api_url or "").strip()
        return f"{p}:{u}" if u else p

    async def _tier_overrides_for(
        self, attempt_tier: Optional[str]
    ) -> tuple[Optional[str], Optional[str]]:
        try:
            from app.core.feature_flags import get_str as _get_str

            return await resolve_tier_overrides(_get_str, attempt_tier)
        except Exception:
            return None, None

    @staticmethod
    def _clip_snapshot(text: Optional[str]) -> Optional[str]:
        if text is None:
            return None
        limit = int(getattr(settings, "LLM_CALL_SNAPSHOT_MAX_CHARS", 20000) or 20000)
        text = str(text)
        if len(text) > limit:
            return text[:limit] + " ... [truncated]"
        return text

    def _record_call_snapshot(
        self,
        db: Optional[AsyncSession],
        *,
        request: Dict[str, Any],
        provider: Optional[str] = None,
        model: Optional[str] = None,
        task_type: Optional[str] = None,
        user_id: Optional[UUID] = None,
        response_text: Optional[str] = None,
        tool_calls: Optional[Any] = None,
        structured: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        latency_ms: Optional[int] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        snapshot_context: Optional[Dict[str, Any]] = None,
        reasoning_text: Optional[str] = None,
        reasoning_tokens: Optional[int] = None,
        cache_hit_tokens: Optional[int] = None,
        cache_miss_tokens: Optional[int] = None,
    ) -> None:
        """Persist a full request/response snapshot for replay debugging.

        Best-effort and opt-in (LLM_CALL_SNAPSHOT_ENABLED); never raises.

        The reasoning defaults to whatever the call just produced rather than
        to nothing. Passing it in was tried first and three call sites became
        two that remembered and one that did not -- and the one that did not
        was the structured path, which is where the agent's decisions are made.
        Reading it here means a new call site cannot forget.
        """
        if db is None or not getattr(settings, "LLM_CALL_SNAPSHOT_ENABLED", False):
            return
        if reasoning_text is None:
            carried = _LAST_REASONING.get()
            if carried:
                reasoning_text, reasoning_tokens = carried
        try:
            from uuid import UUID as _UUID

            from app.models.llm_call_snapshot import LLMCallSnapshot

            ctx = snapshot_context if isinstance(snapshot_context, dict) else {}
            job_id = ctx.get("job_id")
            if job_id is not None and not isinstance(job_id, _UUID):
                try:
                    job_id = _UUID(str(job_id))
                except Exception:
                    job_id = None
            iteration = ctx.get("iteration")
            snapshot = LLMCallSnapshot(
                user_id=_usage_user_id(user_id),
                job_id=job_id,
                iteration=int(iteration) if isinstance(iteration, int) else None,
                phase=(str(ctx.get("phase"))[:50] if ctx.get("phase") else None),
                provider=(str(provider)[:50] if provider else None),
                model=(str(model)[:200] if model else None),
                task_type=(str(task_type)[:50] if task_type else None),
                request=request,
                response_text=self._clip_snapshot(response_text),
                tool_calls=tool_calls,
                structured=structured,
                reasoning_text=self._clip_snapshot(reasoning_text)
                if reasoning_text
                else None,
                reasoning_tokens=(
                    int(reasoning_tokens) if isinstance(reasoning_tokens, int) else None
                ),
                error=(str(error)[:2000] if error else None),
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cache_hit_tokens=cache_hit_tokens,
                cache_miss_tokens=cache_miss_tokens,
            )
            db.add(snapshot)
        except Exception:
            pass

    async def generate_response(
        self,
        query: Optional[str] = None,
        context: Optional[str] = None,
        conversation_history: Optional[str] = None,
        memory_context: Optional[str] = None,
        kg_context: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        prefer_deepseek: bool = False,
        user_settings: Optional[UserLLMSettings] = None,
        task_type: str = "chat",
        user_id: Optional[UUID] = None,
        db: Optional[AsyncSession] = None,
        *,
        routing: Optional[Dict[str, Any]] = None,
        provider: Optional[str] = None,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        snapshot_context: Optional[Dict[str, Any]] = None,
        # Back-compat aliases used in parts of the codebase.
        # Prefer `query=` + `context=` + `task_type=`.
        system_prompt: Optional[str] = None,
        user_message: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> str:
        """Generate a response using the configured LLM.

        `routing` enables per-request tier routing + fallback, using feature flags:
          - llm_provider_fast / llm_model_fast
          - llm_provider_balanced / llm_model_balanced
          - llm_provider_deep / llm_model_deep

        If a tier attempt fails with `LLMServiceError`, the next tier is tried.
        """

        # Normalize legacy call styles.
        if query is None:
            query = prompt or user_message
        if query is None:
            raise TypeError(
                "generate_response requires `query` (or `prompt` / `user_message`)."
            )

        # Best-effort: if a DB session and user_id are provided, auto-load per-user LLM preferences.
        # This keeps user settings applied even when call sites don't explicitly pass `user_settings=`.
        if user_settings is None and user_id is not None and db is not None:
            try:
                from uuid import UUID as _UUID

                from sqlalchemy import select as _select

                from app.models.memory import UserPreferences as _UserPreferences

                uid = user_id
                if isinstance(uid, str):
                    uid = _UUID(uid)
                prefs_res = await db.execute(
                    _select(_UserPreferences).where(_UserPreferences.user_id == uid)
                )
                prefs = prefs_res.scalar_one_or_none()
                if prefs is not None:
                    user_settings = UserLLMSettings.from_preferences(prefs)
            except Exception:
                pass

        routing_origin = None
        if isinstance(routing, dict):
            routing_origin = (
                routing.get("_origin")
                if isinstance(routing.get("_origin"), dict)
                else None
            )

        routing_cfg = coerce_routing_config(routing)
        tier = routing_cfg.get("tier")
        fallback_tiers = (
            routing_cfg.get("fallback_tiers")
            if isinstance(routing_cfg.get("fallback_tiers"), list)
            else []
        )

        timeout_seconds = routing_cfg.get("timeout_seconds")
        max_tokens_cap = routing_cfg.get("max_tokens_cap")
        cooldown_seconds = routing_cfg.get("cooldown_seconds")

        tiers = compute_attempt_tiers(tier=tier, fallback_tiers=fallback_tiers)

        last_err: Optional[Exception] = None

        async def _tier_overrides(
            t: Optional[str],
        ) -> tuple[Optional[str], Optional[str]]:
            try:
                from app.core.feature_flags import get_str as _get_str

                return await resolve_tier_overrides(_get_str, t)
            except Exception:
                return None, None

        for idx, attempt_tier in enumerate(tiers):
            tier_provider, tier_model = await _tier_overrides(attempt_tier)

            attempt_provider = provider or tier_provider
            # If this provider/api_url recently failed, skip it and try the next tier.
            try:
                hk = self._health_key(
                    provider=str(attempt_provider or ""), api_url=api_url
                )
                if not await self._is_healthy(hk):
                    continue
            except Exception:
                pass

            attempt_model = model or tier_model

            attempt_provider_source = None
            if provider:
                attempt_provider_source = "call_provider_override"
            elif tier_provider:
                attempt_provider_source = "tier_feature_flag"

            attempt_model_source = None
            if model:
                attempt_model_source = "call_model_override"
            elif tier_model:
                attempt_model_source = "tier_feature_flag"

            attempt_started = asyncio.get_event_loop().time()
            try:
                result_text = await self._generate_response_once(
                    query=query,
                    context=context,
                    conversation_history=conversation_history,
                    memory_context=memory_context,
                    kg_context=kg_context,
                    model=attempt_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    prefer_deepseek=prefer_deepseek,
                    user_settings=user_settings,
                    task_type=task_type,
                    user_id=user_id,
                    db=db,
                    provider_override=attempt_provider,
                    api_url_override=api_url,
                    api_key_override=api_key,
                    system_prompt=system_prompt,
                    routing_meta={
                        "tier": attempt_tier,
                        "attempt": idx + 1,
                        "attempts": len(tiers),
                        "requested_tier": tier,
                        "fallback_tiers": fallback_tiers,
                        "tier_provider": tier_provider,
                        "tier_model": tier_model,
                        "attempt_provider_source": attempt_provider_source,
                        "attempt_model_source": attempt_model_source,
                        "origin": routing_origin,
                        "agent_id": (
                            routing_origin.get("agent_id")
                            if isinstance(routing_origin, dict)
                            else None
                        ),
                        "experiment_id": (
                            routing_origin.get("experiment_id")
                            if isinstance(routing_origin, dict)
                            else None
                        ),
                        "experiment_variant_id": (
                            routing_origin.get("experiment_variant_id")
                            if isinstance(routing_origin, dict)
                            else None
                        ),
                    },
                    timeout_seconds=timeout_seconds,
                    max_tokens_cap=max_tokens_cap,
                )
                self._record_call_snapshot(
                    db,
                    request={
                        "system_prompt": self._clip_snapshot(system_prompt),
                        "query": self._clip_snapshot(query),
                        "context": self._clip_snapshot(context),
                        "conversation_history": self._clip_snapshot(
                            conversation_history
                        ),
                        "tier": attempt_tier,
                    },
                    provider=attempt_provider,
                    model=attempt_model,
                    task_type=task_type,
                    user_id=user_id,
                    response_text=result_text,
                    latency_ms=int(
                        (asyncio.get_event_loop().time() - attempt_started) * 1000
                    ),
                    snapshot_context=snapshot_context,
                    reasoning_text=(_LAST_REASONING.get() or (None, None))[0],
                    reasoning_tokens=(_LAST_REASONING.get() or (None, None))[1],
                    cache_hit_tokens=(_LAST_CACHE.get() or (None, None))[0],
                    cache_miss_tokens=(_LAST_CACHE.get() or (None, None))[1],
                )
                return result_text
            except LLMServiceError as e:
                self._record_call_snapshot(
                    db,
                    request={
                        "system_prompt": self._clip_snapshot(system_prompt),
                        "query": self._clip_snapshot(query),
                        "tier": attempt_tier,
                    },
                    provider=attempt_provider,
                    model=attempt_model,
                    task_type=task_type,
                    user_id=user_id,
                    error=str(e),
                    latency_ms=int(
                        (asyncio.get_event_loop().time() - attempt_started) * 1000
                    ),
                    snapshot_context=snapshot_context,
                )
                try:
                    k = self._health_key(
                        provider=str(attempt_provider or ""), api_url=api_url
                    )
                    await self._mark_unhealthy(
                        k, cooldown_seconds=cooldown_seconds, reason=str(e)
                    )
                except Exception:
                    pass
                last_err = e

        if isinstance(last_err, LLMServiceError):
            raise last_err
        raise LLMServiceError("Failed to generate response")

    async def generate_structured(
        self,
        *,
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        user_message: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        user_settings: Optional[UserLLMSettings] = None,
        task_type: str = "chat",
        user_id: Optional[UUID] = None,
        db: Optional[AsyncSession] = None,
        routing: Optional[Dict[str, Any]] = None,
        provider: Optional[str] = None,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        snapshot_context: Optional[Dict[str, Any]] = None,
    ):
        """Native completion with tool calling and/or structured output.

        Unlike ``generate_response`` (prompted text), this path uses each
        provider's native APIs: tool/function calling (``tools``) and
        schema-constrained JSON output (``response_schema``). Returns an
        ``LLMCompletion`` with ``text``, ``tool_calls``, and ``structured``.

        Provider/model resolution, tier routing with fallback, provider
        health cooldowns, the concurrency semaphore, and usage-event logging
        all match ``generate_response`` semantics.
        """
        from app.services.llm_providers import build_provider

        if messages is None:
            if user_message is None:
                raise TypeError(
                    "generate_structured requires `messages` or `user_message`."
                )
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_message})

        routing_cfg = coerce_routing_config(routing)
        tier = routing_cfg.get("tier")
        fallback_tiers = (
            routing_cfg.get("fallback_tiers")
            if isinstance(routing_cfg.get("fallback_tiers"), list)
            else []
        )
        timeout_seconds = routing_cfg.get("timeout_seconds")
        max_tokens_cap = routing_cfg.get("max_tokens_cap")
        cooldown_seconds = routing_cfg.get("cooldown_seconds")
        tiers = compute_attempt_tiers(tier=tier, fallback_tiers=fallback_tiers)

        last_err: Optional[Exception] = None

        for idx, attempt_tier in enumerate(tiers):
            tier_provider, tier_model = await self._tier_overrides_for(attempt_tier)

            # Resolution mirrors _generate_response_once: user settings first,
            # then call/tier overrides on top.
            effective_provider = self.provider
            effective_api_url: Optional[str] = None
            effective_api_key: Optional[str] = None
            effective_model = model
            effective_temperature = temperature
            effective_max_tokens = max_tokens

            if user_settings and user_settings.has_custom_settings():
                task_provider = user_settings.get_provider_for_task(task_type)
                if task_provider:
                    effective_provider = task_provider.lower()
                if effective_model is None:
                    effective_model = (
                        user_settings.get_model_for_task(task_type)
                        or user_settings.model
                    )
                if user_settings.api_url:
                    effective_api_url = user_settings.api_url
                if user_settings.api_key:
                    effective_api_key = user_settings.api_key
                if effective_temperature is None:
                    effective_temperature = user_settings.temperature
                if effective_max_tokens is None:
                    effective_max_tokens = user_settings.max_tokens

            attempt_provider = provider or tier_provider
            if attempt_provider:
                effective_provider = str(attempt_provider).strip().lower()
            if api_url:
                effective_api_url = str(api_url).strip() or None
            if api_key:
                effective_api_key = str(api_key)
            if effective_model is None and tier_model:
                effective_model = tier_model

            # Only Ollama falls back to the system default model; external
            # providers use their own configured defaults.
            if effective_model is None and effective_provider == "ollama":
                try:
                    from app.core.feature_flags import get_str as _get_str

                    effective_model = await _get_str("llm_default_model")
                except Exception:
                    effective_model = None
                effective_model = effective_model or self.default_model

            if max_tokens_cap:
                effective_max_tokens = min(
                    int(effective_max_tokens or max_tokens_cap), int(max_tokens_cap)
                )

            hk = self._health_key(
                provider=effective_provider, api_url=effective_api_url
            )
            try:
                if not await self._is_healthy(hk):
                    continue
            except Exception:
                pass

            start_time = asyncio.get_event_loop().time()
            completion = None
            error_text: Optional[str] = None
            try:
                await _LLM_SEMAPHORE.acquire()
                try:
                    llm_provider = build_provider(
                        effective_provider,
                        api_url=effective_api_url,
                        api_key=effective_api_key,
                        http_client=self.client,
                    )
                    completion = await llm_provider.complete(
                        messages,
                        model=effective_model,
                        tools=tools,
                        response_schema=response_schema,
                        temperature=effective_temperature,
                        max_tokens=effective_max_tokens,
                        timeout_seconds=timeout_seconds,
                    )
                finally:
                    _LLM_SEMAPHORE.release()
                return completion
            except LLMServiceError as e:
                error_text = str(e)
                try:
                    await self._mark_unhealthy(
                        hk, cooldown_seconds=cooldown_seconds, reason=str(e)
                    )
                except Exception:
                    pass
                last_err = e
            except Exception as e:
                error_text = str(e)
                logger.error(f"Error in structured LLM generation: {e}")
                last_err = LLMServiceError(
                    f"Failed to generate structured response: {e}"
                )
            finally:
                if db is not None:
                    try:
                        latency_ms = int(
                            (asyncio.get_event_loop().time() - start_time) * 1000
                        )
                        event = LLMUsageEvent(
                            user_id=_usage_user_id(user_id),
                            provider=(
                                completion.provider
                                if completion
                                else effective_provider
                            ),
                            model=(completion.model if completion else effective_model),
                            task_type=task_type,
                            prompt_tokens=(
                                completion.prompt_tokens if completion else None
                            ),
                            completion_tokens=(
                                completion.completion_tokens if completion else None
                            ),
                            total_tokens=(
                                completion.total_tokens if completion else None
                            ),
                            input_chars=sum(
                                len(str(m.get("content") or "")) for m in messages
                            ),
                            output_chars=(
                                len(completion.text or "") if completion else None
                            ),
                            latency_ms=latency_ms,
                            error=(error_text[:255] if error_text else None),
                            extra={
                                "structured": True,
                                "has_schema": bool(response_schema),
                                "tool_count": len(tools or []),
                                "tool_call_count": (
                                    len(completion.tool_calls) if completion else None
                                ),
                                "cache_read_input_tokens": (
                                    (completion.raw or {}).get(
                                        "cache_read_input_tokens"
                                    )
                                    if completion
                                    else None
                                ),
                                "cache_creation_input_tokens": (
                                    (completion.raw or {}).get(
                                        "cache_creation_input_tokens"
                                    )
                                    if completion
                                    else None
                                ),
                                "routing": {
                                    "tier": attempt_tier,
                                    "attempt": idx + 1,
                                    "attempts": len(tiers),
                                    "requested_tier": tier,
                                },
                            },
                        )
                        db.add(event)
                    except Exception:
                        pass
                self._record_call_snapshot(
                    db,
                    request={
                        "messages": [
                            {
                                "role": m.get("role"),
                                "content": self._clip_snapshot(
                                    str(m.get("content") or "")
                                ),
                            }
                            for m in messages
                        ],
                        "tool_names": [str(t.get("name") or "") for t in (tools or [])],
                        "has_schema": bool(response_schema),
                        "tier": attempt_tier,
                    },
                    provider=(
                        completion.provider if completion else effective_provider
                    ),
                    model=(completion.model if completion else effective_model),
                    task_type=task_type,
                    user_id=user_id,
                    response_text=(completion.text if completion else None),
                    tool_calls=(
                        [
                            {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                            for tc in completion.tool_calls
                        ]
                        if completion and completion.tool_calls
                        else None
                    ),
                    structured=(completion.structured if completion else None),
                    error=error_text,
                    latency_ms=int(
                        (asyncio.get_event_loop().time() - start_time) * 1000
                    ),
                    prompt_tokens=(completion.prompt_tokens if completion else None),
                    completion_tokens=(
                        completion.completion_tokens if completion else None
                    ),
                    snapshot_context=snapshot_context,
                )

        if isinstance(last_err, LLMServiceError):
            raise last_err
        raise LLMServiceError("Failed to generate structured response")

    async def _generate_response_once(
        self,
        *,
        query: str,
        context: Optional[str] = None,
        conversation_history: Optional[str] = None,
        memory_context: Optional[str] = None,
        kg_context: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        prefer_deepseek: bool = False,
        user_settings: Optional[UserLLMSettings] = None,
        task_type: str = "chat",
        user_id: Optional[UUID] = None,
        db: Optional[AsyncSession] = None,
        provider_override: Optional[str] = None,
        api_url_override: Optional[str] = None,
        api_key_override: Optional[str] = None,
        system_prompt: Optional[str] = None,
        routing_meta: Optional[Dict[str, Any]] = None,
        timeout_seconds: Optional[int] = None,
        max_tokens_cap: Optional[int] = None,
    ) -> str:
        start_time = asyncio.get_event_loop().time()
        provider_used: Optional[str] = None
        model_used: Optional[str] = None
        prompt_tokens: Optional[int] = None
        completion_tokens: Optional[int] = None
        total_tokens: Optional[int] = None
        input_chars: Optional[int] = None
        output_chars: Optional[int] = None
        extra: Optional[Dict[str, Any]] = None
        error_text: Optional[str] = None

        try:
            await _LLM_SEMAPHORE.acquire()
            try:
                # Apply user settings if provided (they take priority)
                effective_provider = self.provider
                effective_api_url = None
                effective_api_key = None

                if user_settings and user_settings.has_custom_settings():
                    task_provider = user_settings.get_provider_for_task(task_type)
                    if task_provider:
                        effective_provider = task_provider.lower()
                    task_model = user_settings.get_model_for_task(task_type)
                    if task_model:
                        model = task_model
                    elif user_settings.model:
                        model = user_settings.model
                    if user_settings.api_url:
                        effective_api_url = user_settings.api_url
                    if user_settings.api_key:
                        effective_api_key = user_settings.api_key
                    if user_settings.temperature is not None:
                        temperature = user_settings.temperature
                    if user_settings.max_tokens is not None:
                        max_tokens = user_settings.max_tokens

                # Explicit overrides (per-agent routing)
                if provider_override:
                    effective_provider = str(provider_override).strip().lower()
                if api_url_override:
                    effective_api_url = str(api_url_override).strip() or None
                if api_key_override:
                    effective_api_key = str(api_key_override)

                # Fall back to system defaults for model
                if not model:
                    try:
                        from app.core.feature_flags import get_str as _get_str

                        model = await _get_str("llm_default_model")
                    except Exception:
                        model = None
                model = model or self.default_model
                temperature = (
                    temperature if temperature is not None else settings.TEMPERATURE
                )
                # Deliberately NOT defaulted here. Each provider applies its
                # own cap below when the caller named none, and coercing to the
                # generic one first made those unreachable: a request with no
                # max_tokens arrived at DeepSeek as 1000 rather than
                # DEEPSEEK_MAX_RESPONSE_TOKENS, and its reasoning models spend
                # the budget thinking before they answer, so the call returned
                # empty. The agent's decision path passes no budget, which is
                # how that setting came to be dead exactly where it mattered.
                requested_max_tokens = max_tokens
                max_tokens = max_tokens or settings.MAX_RESPONSE_LENGTH

                # Routing decision provenance (best-effort)
                routing_decision: Dict[str, Any] = {}
                try:
                    if routing_meta and isinstance(routing_meta, dict):
                        aps = routing_meta.get("attempt_provider_source")
                        ams = routing_meta.get("attempt_model_source")
                    else:
                        aps = None
                        ams = None

                    # api_url source determines provider_used=custom
                    if api_url_override:
                        routing_decision["api_url_source"] = "api_url_override"
                    elif user_settings and getattr(user_settings, "api_url", None):
                        routing_decision["api_url_source"] = "user_api_url"

                    # provider source
                    if provider_override:
                        routing_decision["provider_source"] = aps or "provider_override"
                    elif (
                        user_settings
                        and getattr(user_settings, "has_custom_settings")()
                        and user_settings.get_provider_for_task(task_type)
                    ):
                        routing_decision["provider_source"] = "user_task_provider"
                    elif (
                        user_settings
                        and getattr(user_settings, "has_custom_settings")()
                        and getattr(user_settings, "provider", None)
                    ):
                        routing_decision["provider_source"] = "user_provider"
                    else:
                        routing_decision["provider_source"] = "system_default_provider"

                    # model source
                    if (
                        user_settings
                        and getattr(user_settings, "has_custom_settings")()
                        and user_settings.get_model_for_task(task_type)
                    ):
                        routing_decision["model_source"] = "user_task_model"
                    elif (
                        user_settings
                        and getattr(user_settings, "has_custom_settings")()
                        and getattr(user_settings, "model", None)
                    ):
                        routing_decision["model_source"] = "user_model"
                    elif model and ams:
                        routing_decision["model_source"] = ams
                    else:
                        # If we had to fall back to feature/system defaults, infer based on prior steps.
                        routing_decision["model_source"] = "default"
                except Exception:
                    pass

                if routing_meta and isinstance(routing_meta, dict):
                    routing_meta["decision"] = routing_decision

                # Determine which provider to use.
                # SDK-module providers route through llm_providers; the model
                # prefixes guard against the generic (Ollama) model default
                # leaking into a provider that can't serve it.
                _sdk_provider_model_prefixes = {
                    "anthropic": ("claude",),
                    "qwen": ("qwen",),
                    "kimi": ("kimi", "moonshot"),
                }
                if effective_provider in _sdk_provider_model_prefixes:
                    from app.services.llm_providers import (
                        build_provider as _build_provider,
                    )

                    provider_used = effective_provider
                    chat_messages = self._build_chat_messages(
                        query=query,
                        context=context,
                        conversation_history=conversation_history,
                        memory_context=memory_context,
                        kg_context=kg_context,
                        system_prompt=system_prompt,
                    )
                    input_chars = sum(
                        len(m.get("content") or "") for m in chat_messages
                    )
                    prefixes = _sdk_provider_model_prefixes[effective_provider]
                    native_model = (
                        model
                        if model and str(model).lower().startswith(prefixes)
                        else None
                    )
                    llm_provider = _build_provider(
                        effective_provider,
                        api_url=effective_api_url,
                        api_key=effective_api_key,
                    )
                    completion = await llm_provider.complete(
                        chat_messages,
                        model=native_model,
                        # AnthropicProvider never sends temperature; the
                        # OpenAI-compatible providers honor it.
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout_seconds=timeout_seconds,
                    )
                    model_used = completion.model or native_model
                    prompt_tokens = completion.prompt_tokens
                    completion_tokens = completion.completion_tokens
                    total_tokens = completion.total_tokens
                    output_chars = len(completion.text or "")
                    extra = completion.raw
                    return completion.text

                if effective_api_url:
                    provider_used = "custom"
                    model_used = model
                    messages = self._build_chat_messages(
                        query=query,
                        context=context,
                        conversation_history=conversation_history,
                        memory_context=memory_context,
                        kg_context=kg_context,
                        system_prompt=system_prompt,
                    )
                    input_chars = sum(len(m.get("content") or "") for m in messages)
                    result, meta = await self._make_openai_compatible_request(
                        api_url=effective_api_url,
                        api_key=effective_api_key,
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout_seconds=timeout_seconds,
                    )
                    output_chars = len(result or "")
                    if isinstance(meta, dict):
                        model_used = meta.get("model") or model_used
                        usage = meta.get("usage")
                        if isinstance(usage, dict):
                            prompt_tokens = usage.get("prompt_tokens")
                            completion_tokens = usage.get("completion_tokens")
                            total_tokens = usage.get("total_tokens")
                        extra = meta
                    return result

                use_deepseek = (
                    effective_provider == "deepseek"
                    or effective_provider == "openai"
                    or (
                        prefer_deepseek
                        and bool(getattr(settings, "DEEPSEEK_API_KEY", None))
                    )
                )

                if use_deepseek:
                    provider_used = (
                        "deepseek" if effective_provider == "deepseek" else "openai"
                    )
                    messages = self._build_chat_messages(
                        query=query,
                        context=context,
                        conversation_history=conversation_history,
                        memory_context=memory_context,
                        kg_context=kg_context,
                        system_prompt=system_prompt,
                    )
                    input_chars = sum(len(m.get("content") or "") for m in messages)
                    api_key = effective_api_key or settings.DEEPSEEK_API_KEY
                    api_base = effective_api_url or settings.DEEPSEEK_API_BASE

                    model_used = model or settings.DEEPSEEK_MODEL
                    result, meta = await self._make_deepseek_chat_request(
                        model=model_used,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=(
                            requested_max_tokens
                            or settings.DEEPSEEK_MAX_RESPONSE_TOKENS
                        ),
                        api_key_override=api_key,
                        api_base_override=api_base,
                        timeout_seconds=timeout_seconds,
                    )
                    output_chars = len(result or "")
                    if isinstance(meta, dict):
                        model_used = meta.get("model") or model_used
                        usage = meta.get("usage")
                        if isinstance(usage, dict):
                            prompt_tokens = usage.get("prompt_tokens")
                            completion_tokens = usage.get("completion_tokens")
                            total_tokens = usage.get("total_tokens")
                        extra = meta
                    return result

                # Default to Ollama
                provider_used = "ollama"
                model_used = model
                prompt = self._build_prompt(
                    query,
                    context,
                    conversation_history,
                    memory_context,
                    kg_context,
                    system_prompt=system_prompt,
                )
                input_chars = len(prompt or "")
                ollama_url = effective_api_url or self.base_url
                result, meta = await self._make_ollama_request(
                    model=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    base_url_override=ollama_url,
                    timeout_seconds=timeout_seconds,
                )
                output_chars = len(result or "")
                if isinstance(meta, dict):
                    model_used = meta.get("model") or model_used
                    prompt_tokens = meta.get("prompt_eval_count") or meta.get(
                        "prompt_tokens"
                    )
                    completion_tokens = meta.get("eval_count") or meta.get(
                        "completion_tokens"
                    )
                    if isinstance(prompt_tokens, int) and isinstance(
                        completion_tokens, int
                    ):
                        total_tokens = prompt_tokens + completion_tokens
                    extra = meta
                return result
            finally:
                _LLM_SEMAPHORE.release()

        except Exception as e:
            error_text = str(e)
            logger.error(f"Error generating LLM response: {e}")
            raise LLMServiceError(f"Failed to generate response: {str(e)}")
        finally:
            if db is not None and provider_used is not None:
                try:
                    latency_ms = int(
                        (asyncio.get_event_loop().time() - start_time) * 1000
                    )
                    event_extra: Optional[Dict[str, Any]]
                    if isinstance(extra, dict):
                        event_extra = dict(extra)
                    else:
                        event_extra = None
                    if routing_meta:
                        if event_extra is None:
                            event_extra = {}
                        event_extra["routing"] = routing_meta

                    event = LLMUsageEvent(
                        user_id=_usage_user_id(user_id),
                        provider=provider_used,
                        model=model_used,
                        task_type=task_type,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        input_chars=input_chars,
                        output_chars=output_chars,
                        latency_ms=latency_ms,
                        error=(error_text[:255] if error_text else None),
                        extra=event_extra,
                    )
                    db.add(event)
                except Exception:
                    pass

    def _build_prompt(
        self,
        query: str,
        context: Optional[str] = None,
        conversation_history: Optional[str] = None,
        memory_context: Optional[str] = None,
        kg_context: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Build the complete prompt for the LLM.

        Args:
            query: User's question or query
            context: Optional context from knowledge base
            conversation_history: Optional previous conversation history
            memory_context: Optional context from user's conversation memories
            kg_context: Optional knowledge graph context (entities and relationships)

        Returns:
            Complete formatted prompt string
        """
        prompt_parts = []

        # System instruction
        system_instruction = (
            system_prompt
            or """You are a helpful AI assistant for an organizational knowledge base. Your role is to answer questions based on the provided context from internal documents and previous conversation history.

Guidelines:
1. Answer questions accurately based on the provided context
2. If the context doesn't contain enough information, clearly state this
3. Always cite your sources when referencing specific documents
4. Be concise but thorough in your explanations
5. If asked about something not in the context, politely explain that you don't have that information
6. Maintain a professional and helpful tone
7. Use relevant memories from past conversations to provide personalized responses
8. Use knowledge graph context to understand entity relationships and provide more connected answers

Citation format:
- The context includes entries labeled “Source 1”, “Source 2”, etc.
- When you use a source, add an inline citation like [1] or [2] matching the source number.
- If you quote or rely on a specific claim, include a short evidence excerpt and cite it (e.g., “…excerpt…” [3])."""
        )

        prompt_parts.append(system_instruction)

        # Add memory context if provided (most relevant for personalization)
        if memory_context:
            prompt_parts.append(
                f"\nRelevant memories from past conversations:\n{memory_context}"
            )

        # Add context if provided
        if context:
            prompt_parts.append(f"\nContext from knowledge base:\n{context}")

        # Add knowledge graph context if provided
        if kg_context:
            prompt_parts.append(kg_context)

        # Add conversation history if provided
        if conversation_history:
            prompt_parts.append(f"\nPrevious conversation:\n{conversation_history}")

        # Add the current query
        prompt_parts.append(f"\nUser question: {query}")
        prompt_parts.append("\nAssistant response:")

        return "\n".join(prompt_parts)

    def _build_chat_messages(
        self,
        query: str,
        context: Optional[str] = None,
        conversation_history: Optional[str] = None,
        memory_context: Optional[str] = None,
        kg_context: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Build OpenAI-style chat messages for chat completion APIs."""
        system_instruction = system_prompt or (
            "You are a helpful AI assistant for an organizational knowledge base. "
            "Answer based on provided context and prior conversation. Cite sources when relevant. "
            "Use knowledge graph context to understand entity relationships. "
            "Use inline citations like [1], [2] matching 'Source 1', 'Source 2' in the provided context. "
            "Include short evidence excerpts when making factual claims."
        )

        user_parts: List[str] = []
        if memory_context:
            user_parts.append(
                f"Relevant memories from past conversations:\n{memory_context}"
            )
        if context:
            user_parts.append(f"Context from knowledge base:\n{context}")
        if kg_context:
            user_parts.append(kg_context)
        if conversation_history:
            user_parts.append(f"Previous conversation:\n{conversation_history}")
        user_parts.append(f"User question: {query}")

        return [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        reraise=True,
    )
    async def _make_ollama_request(
        self,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        base_url_override: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
    ) -> tuple[str, Dict[str, Any]]:
        """
        Make a request to the Ollama API with retry logic.

        Args:
            model: Model name to use
            prompt: Complete prompt to send
            temperature: Temperature for generation
            max_tokens: Maximum tokens for response
            base_url_override: Optional override for the Ollama base URL

        Returns:
            Generated response text

        Raises:
            LLMServiceError: If request fails after retries
            httpx.HTTPError: If HTTP request fails
        """
        try:
            base_url = base_url_override or self.base_url
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "top_p": settings.TOP_P,
                    "stop": ["Human:", "User:", "\nUser:", "\nHuman:"],
                    # Force CPU usage and limit memory for Mac compatibility
                    "num_gpu": 0,  # Use CPU only (important for Mac)
                    "num_thread": 4,  # Limit CPU threads
                    "numa": False,  # Disable NUMA (not needed on Mac)
                },
            }

            response = await self.client.post(f"{base_url}/api/generate", json=payload)

            response.raise_for_status()
            result = response.json()
            text = (result.get("response", "") or "").strip()
            meta: Dict[str, Any] = {
                "model": result.get("model") or model,
                "created_at": result.get("created_at"),
                "done_reason": result.get("done_reason"),
                "prompt_eval_count": result.get("prompt_eval_count"),
                "eval_count": result.get("eval_count"),
                "total_duration": result.get("total_duration"),
                "load_duration": result.get("load_duration"),
                "prompt_eval_duration": result.get("prompt_eval_duration"),
                "eval_duration": result.get("eval_duration"),
            }
            return text, meta

        except httpx.HTTPStatusError as e:
            logger.error(
                f"Ollama API error: {e.response.status_code} - {e.response.text}"
            )
            raise LLMServiceError(f"Ollama API error: {e.response.status_code}")
        except httpx.TimeoutException:
            logger.error("LLM request timed out")
            raise LLMServiceError("Request timed out")
        except httpx.RequestError as e:
            logger.error(f"LLM request error: {e}")
            raise LLMServiceError(f"Request error: {str(e)}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        reraise=True,
    )
    async def _make_deepseek_chat_request(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        api_key_override: Optional[str] = None,
        api_base_override: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        _budget_retry: bool = False,
    ) -> tuple[str, Dict[str, Any]]:
        """Call DeepSeek's OpenAI-compatible chat completions API.

        `_budget_retry` marks the one automatic retry on a truncated empty
        response, so a second truncation reports rather than recursing.
        """
        api_key = api_key_override or settings.DEEPSEEK_API_KEY
        if not api_key:
            raise LLMServiceError("DEEPSEEK_API_KEY is not set")

        api_base = api_base_override or settings.DEEPSEEK_API_BASE
        url = f"{api_base.rstrip('/')}/chat/completions"
        timeout = (
            int(timeout_seconds)
            if timeout_seconds is not None
            else int(settings.DEEPSEEK_TIMEOUT_SECONDS or 120)
        )

        # The caller sized an answer; the model spends an unknown amount
        # reasoning before it writes one. Raising a cap costs nothing that is
        # not generated -- max_tokens is a ceiling, not a purchase -- and
        # without it every call site that asked for a short reply gets an empty
        # string instead of a short reply.
        effective_max_tokens = max(
            int(max_tokens or 0), int(settings.DEEPSEEK_MIN_COMPLETION_TOKENS or 0)
        )

        payload = {
            "model": model or settings.DEEPSEEK_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": effective_max_tokens,
            "stream": False,
            "top_p": settings.TOP_P,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = await self.client.post(
                url, json=payload, headers=headers, timeout=timeout
            )
            response.raise_for_status()
            data = response.json()
            # OpenAI-compatible shape: choices[0].message.content
            choice = (data.get("choices") or [{}])[0]
            content = ((choice.get("message") or {}).get("content")) or ""
            meta = _meta_from_completion(data, fallback_model=model)

            # DeepSeek's current models reason before they answer, and the
            # reasoning is charged against max_tokens. Ask for too few and the
            # call succeeds, spends the whole budget thinking, and returns an
            # empty string -- which every caller here reports as the model
            # producing unusable output. The decision parser says "No valid
            # JSON object found in response", an error about the model when the
            # cause is a number in the config, and it only bites once the
            # prompt grows enough to make the reasoning long: early iterations
            # of a run parse, later ones do not.
            if not (content or "").strip():
                usage = data.get("usage") or {}
                # Name the value that actually bound, not a plausible one. The
                # budget is max(caller_asked, DEEPSEEK_MIN_COMPLETION_TOKENS),
                # so when the floor wins it is the floor to raise -- this said
                # DEEPSEEK_MAX_RESPONSE_TOKENS, which is a ceiling applied
                # elsewhere and never the constraint here. Following that
                # advice changes nothing and reads as though the fix was tried.
                floor = int(settings.DEEPSEEK_MIN_COMPLETION_TOKENS or 0)
                remedy = (
                    f"Raise DEEPSEEK_MIN_COMPLETION_TOKENS (currently {floor}), "
                    "which is what set this budget."
                    if effective_max_tokens == floor and floor > int(max_tokens or 0)
                    else f"Ask for more than {max_tokens} tokens at the call site."
                )
                # The cause is known exactly -- the budget ended before the
                # answer began -- so the first response is to give it room,
                # not to report. Once: a second truncation means the prompt
                # itself is the problem and doubling again only spends more.
                retry_budget = (
                    None
                    if _budget_retry
                    else llm_truncation.next_budget(effective_max_tokens)
                )
                if retry_budget is not None and llm_truncation.is_truncated(
                    choice.get("finish_reason"), content
                ):
                    logger.warning(
                        f"{model} spent its whole {effective_max_tokens}-token "
                        f"budget reasoning; retrying once at {retry_budget}"
                    )
                    return await self._make_deepseek_chat_request(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=retry_budget,
                        api_key_override=api_key_override,
                        api_base_override=api_base_override,
                        timeout_seconds=timeout_seconds,
                        _budget_retry=True,
                    )
                raise LLMServiceError(
                    f"{model} returned no content "
                    f"(finish_reason={choice.get('finish_reason')!r}, "
                    f"max_tokens={effective_max_tokens} "
                    f"(caller asked {max_tokens}), "
                    f"completion_tokens={usage.get('completion_tokens')}"
                    + (", already retried on a doubled budget" if _budget_retry else "")
                    + "). These models spend max_tokens on reasoning before "
                    f"answering, so a budget that fits the answer may not fit "
                    f"the thinking. {remedy}"
                )
            return (content or "").strip(), meta
        except httpx.HTTPStatusError as e:
            logger.error(
                f"DeepSeek API error: {e.response.status_code} - {e.response.text}"
            )
            raise LLMServiceError(f"DeepSeek API error: {e.response.status_code}")
        except httpx.TimeoutException:
            logger.error("DeepSeek request timed out")
            raise LLMServiceError("Request timed out")
        except httpx.RequestError as e:
            logger.error(f"DeepSeek request error: {e}")
            raise LLMServiceError(f"Request error: {str(e)}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        reraise=True,
    )
    async def _make_openai_compatible_request(
        self,
        api_url: str,
        api_key: Optional[str],
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        timeout_seconds: Optional[int] = None,
    ) -> tuple[str, Dict[str, Any]]:
        """
        Make a request to any OpenAI-compatible chat completions API.

        Args:
            api_url: Full base URL for the API (e.g., "https://api.openai.com/v1")
            api_key: API key for authentication (optional for some local servers)
            model: Model name to use
            messages: Chat messages in OpenAI format
            temperature: Temperature for generation
            max_tokens: Maximum tokens for response

        Returns:
            Generated response text
        """
        # Ensure URL ends with /chat/completions
        url = api_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            if not url.endswith("/v1"):
                url = f"{url}/v1"
            url = f"{url}/chat/completions"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            response = await self.client.post(
                url, json=payload, headers=headers, timeout=120.0
            )
            response.raise_for_status()
            data = response.json()

            # OpenAI-compatible shape: choices[0].message.content
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            meta: Dict[str, Any] = {
                "id": data.get("id"),
                "model": data.get("model") or model,
                "usage": data.get("usage"),
            }
            return (content or "").strip(), meta
        except httpx.HTTPStatusError as e:
            logger.error(
                f"OpenAI-compatible API error: {e.response.status_code} - {e.response.text}"
            )
            raise LLMServiceError(f"API error: {e.response.status_code}")
        except httpx.TimeoutException:
            logger.error("OpenAI-compatible API request timed out")
            raise LLMServiceError("Request timed out")
        except httpx.RequestError as e:
            logger.error(f"OpenAI-compatible API request error: {e}")
            raise LLMServiceError(f"Request error: {str(e)}")

    async def list_available_models(
        self, base_url_override: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all available models in Ollama.

        Returns:
            List of model dictionaries with model information
        """
        try:
            base_url = (base_url_override or self.base_url).rstrip("/")
            response = await self.client.get(f"{base_url}/api/tags")

            if response.status_code == 200:
                result = response.json()
                return result.get("models", [])
            else:
                logger.error(f"Failed to list models: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    async def pull_model(self, model: str) -> bool:
        """
        Pull/download a model in Ollama.

        Args:
            model: Model name to pull

        Returns:
            True if model pull was successful, False otherwise
        """
        try:
            payload = {"name": model}

            response = await self.client.post(f"{self.base_url}/api/pull", json=payload)

            if response.status_code == 200:
                logger.info(f"Successfully pulled model: {model}")
                return True
            else:
                logger.error(f"Failed to pull model {model}: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Error pulling model {model}: {e}")
            return False

    async def health_check(self) -> bool:
        """
        Check if the configured LLM service is healthy.

        Returns:
            True if service is healthy, False otherwise
        """
        try:
            if self.provider == "deepseek":
                # Ping DeepSeek models list (OpenAI-compatible) to verify auth and availability
                url = f"{settings.DEEPSEEK_API_BASE.rstrip('/')}/models"
                headers = {"Authorization": f"Bearer {settings.DEEPSEEK_API_KEY}"}
                response = await self.client.get(url, headers=headers)
                return response.status_code == 200
            else:
                response = await self.client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception as e:
            logger.error(f"LLM health check failed: {e}")
            return False

    def get_active_model(self) -> str:
        """Return the currently active model name based on provider and settings."""
        if self.provider == "deepseek":
            return settings.DEEPSEEK_MODEL
        return self.default_model

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
