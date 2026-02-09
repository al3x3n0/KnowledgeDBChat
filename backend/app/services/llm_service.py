"""
LLM service for generating responses using local or external LLMs.

Supported providers:
- Ollama (local)
- DeepSeek (external, OpenAI-compatible chat API)
- Custom OpenAI-compatible APIs (user-configured)
"""

import asyncio
import httpx
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from loguru import logger
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.llm_usage import LLMUsageEvent
from app.services.llm_routing import (
    coerce_routing_config,
    compute_attempt_tiers,
    resolve_tier_overrides,
)
from app.utils.exceptions import LLMServiceError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

_LLM_SEMAPHORE = asyncio.Semaphore(settings.LLM_MAX_CONCURRENCY)


# Supported task types for per-task model configuration
LLM_TASK_TYPES = [
    "chat",
    "title_generation",
    "summarization",
    "query_expansion",
    "memory_extraction",
    "workflow_synthesis",
]


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
        return any([
            self.provider, self.model, self.api_url,
            self.api_key, self.temperature is not None, self.max_tokens is not None,
            self.task_models, self.task_providers
        ])

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


class LLMService:
    """Service for interacting with configured LLM provider."""

    def __init__(self):
        self.provider = (settings.LLM_PROVIDER or "ollama").lower()
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

    async def _mark_unhealthy(self, key: str, *, cooldown_seconds: int, reason: str) -> None:
        cooldown_seconds = max(5, min(int(cooldown_seconds or 60), 3600))
        try:
            now = asyncio.get_event_loop().time()
        except Exception:
            now = 0.0
        async with self._unhealthy_lock:
            self._unhealthy_until[key] = float(now) + float(cooldown_seconds)
            self._unhealthy_reason[key] = str(reason or '')[:200]

    def _health_key(self, *, provider: str, api_url: Optional[str]) -> str:
        p = (provider or '').strip().lower() or 'unknown'
        u = (api_url or '').strip()
        return f"{p}:{u}" if u else p

    async def generate_response(
        self,
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
        *,
        routing: Optional[Dict[str, Any]] = None,
        provider: Optional[str] = None,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> str:
        """Generate a response using the configured LLM.

        `routing` enables per-request tier routing + fallback, using feature flags:
          - llm_provider_fast / llm_model_fast
          - llm_provider_balanced / llm_model_balanced
          - llm_provider_deep / llm_model_deep

        If a tier attempt fails with `LLMServiceError`, the next tier is tried.
        """

        routing_origin = None
        if isinstance(routing, dict):
            routing_origin = routing.get("_origin") if isinstance(routing.get("_origin"), dict) else None

        routing_cfg = coerce_routing_config(routing)
        tier = routing_cfg.get("tier")
        fallback_tiers = routing_cfg.get("fallback_tiers") if isinstance(routing_cfg.get("fallback_tiers"), list) else []

        timeout_seconds = routing_cfg.get("timeout_seconds")
        max_tokens_cap = routing_cfg.get("max_tokens_cap")
        cooldown_seconds = routing_cfg.get("cooldown_seconds")

        tiers = compute_attempt_tiers(tier=tier, fallback_tiers=fallback_tiers)

        last_err: Optional[Exception] = None

        async def _tier_overrides(t: Optional[str]) -> tuple[Optional[str], Optional[str]]:
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
                hk = self._health_key(provider=str(attempt_provider or ""), api_url=api_url)
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

            try:
                return await self._generate_response_once(
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
                        "agent_id": (routing_origin.get("agent_id") if isinstance(routing_origin, dict) else None),
                        "experiment_id": (routing_origin.get("experiment_id") if isinstance(routing_origin, dict) else None),
                        "experiment_variant_id": (routing_origin.get("experiment_variant_id") if isinstance(routing_origin, dict) else None),
                    },
                    timeout_seconds=timeout_seconds,
                    max_tokens_cap=max_tokens_cap,
                )
            except LLMServiceError as e:
                try:
                    k = self._health_key(provider=str(attempt_provider or ""), api_url=api_url)
                    await self._mark_unhealthy(k, cooldown_seconds=cooldown_seconds, reason=str(e))
                except Exception:
                    pass
                last_err = e

        if isinstance(last_err, LLMServiceError):
            raise last_err
        raise LLMServiceError("Failed to generate response")

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
                temperature = temperature if temperature is not None else settings.TEMPERATURE
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
                    elif user_settings and getattr(user_settings, "has_custom_settings")() and user_settings.get_provider_for_task(task_type):
                        routing_decision["provider_source"] = "user_task_provider"
                    elif user_settings and getattr(user_settings, "has_custom_settings")() and getattr(user_settings, "provider", None):
                        routing_decision["provider_source"] = "user_provider"
                    else:
                        routing_decision["provider_source"] = "system_default_provider"

                    # model source
                    if user_settings and getattr(user_settings, "has_custom_settings")() and user_settings.get_model_for_task(task_type):
                        routing_decision["model_source"] = "user_task_model"
                    elif user_settings and getattr(user_settings, "has_custom_settings")() and getattr(user_settings, "model", None):
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

                # Determine which provider to use
                if effective_api_url:
                    provider_used = "custom"
                    model_used = model
                    messages = self._build_chat_messages(
                        query=query,
                        context=context,
                        conversation_history=conversation_history,
                        memory_context=memory_context,
                        kg_context=kg_context,
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
                    or (prefer_deepseek and bool(getattr(settings, "DEEPSEEK_API_KEY", None)))
                )

                if use_deepseek:
                    provider_used = "deepseek" if effective_provider == "deepseek" else "openai"
                    messages = self._build_chat_messages(
                        query=query,
                        context=context,
                        conversation_history=conversation_history,
                        memory_context=memory_context,
                        kg_context=kg_context,
                    )
                    input_chars = sum(len(m.get("content") or "") for m in messages)
                    api_key = effective_api_key or settings.DEEPSEEK_API_KEY
                    api_base = effective_api_url or settings.DEEPSEEK_API_BASE

                    model_used = model or settings.DEEPSEEK_MODEL
                    result, meta = await self._make_deepseek_chat_request(
                        model=model_used,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=(max_tokens or settings.DEEPSEEK_MAX_RESPONSE_TOKENS),
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
                prompt = self._build_prompt(query, context, conversation_history, memory_context, kg_context)
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
                    prompt_tokens = meta.get("prompt_eval_count") or meta.get("prompt_tokens")
                    completion_tokens = meta.get("eval_count") or meta.get("completion_tokens")
                    if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
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
                    latency_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
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
                        user_id=user_id,
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
        system_instruction = """You are a helpful AI assistant for an organizational knowledge base. Your role is to answer questions based on the provided context from internal documents and previous conversation history.

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

        prompt_parts.append(system_instruction)

        # Add memory context if provided (most relevant for personalization)
        if memory_context:
            prompt_parts.append(f"\nRelevant memories from past conversations:\n{memory_context}")

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
    ) -> List[Dict[str, str]]:
        """Build OpenAI-style chat messages for chat completion APIs."""
        system_instruction = (
            "You are a helpful AI assistant for an organizational knowledge base. "
            "Answer based on provided context and prior conversation. Cite sources when relevant. "
            "Use knowledge graph context to understand entity relationships. "
            "Use inline citations like [1], [2] matching 'Source 1', 'Source 2' in the provided context. "
            "Include short evidence excerpts when making factual claims."
        )

        user_parts: List[str] = []
        if memory_context:
            user_parts.append(f"Relevant memories from past conversations:\n{memory_context}")
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
        reraise=True
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
                    "numa": False  # Disable NUMA (not needed on Mac)
                }
            }

            response = await self.client.post(
                f"{base_url}/api/generate",
                json=payload
            )
            
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
            logger.error(f"Ollama API error: {e.response.status_code} - {e.response.text}")
            raise LLMServiceError(f"Ollama API error: {e.response.status_code}")
        except httpx.TimeoutException as e:
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
    ) -> tuple[str, Dict[str, Any]]:
        """Call DeepSeek's OpenAI-compatible chat completions API."""
        api_key = api_key_override or settings.DEEPSEEK_API_KEY
        if not api_key:
            raise LLMServiceError("DEEPSEEK_API_KEY is not set")

        api_base = api_base_override or settings.DEEPSEEK_API_BASE
        url = f"{api_base.rstrip('/')}/chat/completions"
        timeout = int(timeout_seconds) if timeout_seconds is not None else int(settings.DEEPSEEK_TIMEOUT_SECONDS or 120)

        payload = {
            "model": model or settings.DEEPSEEK_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            "top_p": settings.TOP_P,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = await self.client.post(url, json=payload, headers=headers, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            # OpenAI-compatible shape: choices[0].message.content
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            meta: Dict[str, Any] = {
                "id": data.get("id"),
                "model": data.get("model") or model,
                "usage": data.get("usage"),
            }
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
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
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

    async def check_model_availability(self, model: Optional[str] = None) -> bool:
        """
        Check if a model is available in Ollama.
        
        Args:
            model: Model name to check (uses default if not provided)
            
        Returns:
            True if model is available, False otherwise
        """
        try:
            model = model or self.default_model
            
            response = await self.client.get(f"{self.base_url}/api/tags")
            
            if response.status_code == 200:
                models = response.json()
                available_models = [m["name"] for m in models.get("models", [])]
                
                # Check for exact match or partial match (e.g., "llama2" in "llama2:latest")
                is_available = any(
                    model in available_model or available_model.startswith(model)
                    for available_model in available_models
                )
                
                if not is_available:
                    logger.warning(f"Model {model} not found. Available models: {available_models}")
                
                return is_available
            else:
                logger.error(f"Failed to check model availability: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error checking model availability: {e}")
            return False
    
    async def list_available_models(self, base_url_override: Optional[str] = None) -> List[Dict[str, Any]]:
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
            
            response = await self.client.post(
                f"{self.base_url}/api/pull",
                json=payload
            )
            
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
