"""
LLM service for generating responses using local or external LLMs.

Supported providers:
- Ollama (local)
- DeepSeek (external, OpenAI-compatible chat API)
- Custom OpenAI-compatible APIs (user-configured)
"""

import httpx
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from loguru import logger

from app.core.config import settings
from app.utils.exceptions import LLMServiceError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


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
        )

    def has_custom_settings(self) -> bool:
        """Check if any custom settings are configured."""
        return any([
            self.provider, self.model, self.api_url,
            self.api_key, self.temperature is not None, self.max_tokens is not None,
            self.task_models
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


class LLMService:
    """Service for interacting with configured LLM provider."""

    def __init__(self):
        self.provider = (settings.LLM_PROVIDER or "ollama").lower()
        self.base_url = settings.OLLAMA_BASE_URL
        self.default_model = settings.DEFAULT_MODEL
        # A single client is enough; per-request overrides set timeouts/headers
        self.client = httpx.AsyncClient(timeout=120.0)
    
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
    ) -> str:
        """
        Generate a response using the configured LLM.

        Args:
            query: User's question or query
            context: Optional context from knowledge base documents
            conversation_history: Optional previous conversation history
            memory_context: Optional context from user's conversation memories
            kg_context: Optional knowledge graph context (entities and relationships)
            model: Optional model name (uses default if not provided)
            temperature: Optional temperature for generation (uses default if not provided)
            max_tokens: Optional maximum tokens for response (uses default if not provided)
            prefer_deepseek: Whether to prefer DeepSeek for this request
            user_settings: Optional user-specific LLM settings that override defaults
            task_type: Task type for per-task model selection (chat, title_generation, etc.)

        Returns:
            Generated response string

        Raises:
            LLMServiceError: If LLM service is unavailable or request fails
        """
        try:
            # Apply user settings if provided (they take priority)
            effective_provider = self.provider
            effective_api_url = None
            effective_api_key = None

            if user_settings and user_settings.has_custom_settings():
                if user_settings.provider:
                    effective_provider = user_settings.provider.lower()
                # Use task-specific model if configured, otherwise fall back to default model
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

            # Determine which provider to use
            # Custom API URL means user wants to use their own endpoint
            if effective_api_url:
                messages = self._build_chat_messages(
                    query=query,
                    context=context,
                    conversation_history=conversation_history,
                    memory_context=memory_context,
                    kg_context=kg_context,
                )
                result = await self._make_openai_compatible_request(
                    api_url=effective_api_url,
                    api_key=effective_api_key,
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return result

            # Provider-specific request (auto-route heavy jobs to DeepSeek if available)
            use_deepseek = (
                effective_provider == "deepseek" or
                effective_provider == "openai" or (
                    prefer_deepseek and bool(getattr(settings, "DEEPSEEK_API_KEY", None))
                )
            )

            if use_deepseek:
                messages = self._build_chat_messages(
                    query=query,
                    context=context,
                    conversation_history=conversation_history,
                    memory_context=memory_context,
                    kg_context=kg_context,
                )
                # Use user's API key if provided, otherwise system key
                api_key = effective_api_key or settings.DEEPSEEK_API_KEY
                api_base = effective_api_url or settings.DEEPSEEK_API_BASE

                result = await self._make_deepseek_chat_request(
                    model=(model or settings.DEEPSEEK_MODEL),
                    messages=messages,
                    temperature=temperature,
                    max_tokens=(max_tokens or settings.DEEPSEEK_MAX_RESPONSE_TOKENS),
                    api_key_override=api_key,
                    api_base_override=api_base,
                )
                return result
            else:
                # Default to Ollama
                prompt = self._build_prompt(query, context, conversation_history, memory_context, kg_context)
                # Use custom URL for Ollama if provided
                ollama_url = effective_api_url or self.base_url
                result = await self._make_ollama_request(
                    model=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    base_url_override=ollama_url,
                )
                return result
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            raise LLMServiceError(f"Failed to generate response: {str(e)}")
    
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
8. Use knowledge graph context to understand entity relationships and provide more connected answers"""

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
            "Use knowledge graph context to understand entity relationships."
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
    ) -> str:
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
            return result.get("response", "").strip()
                
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
    ) -> str:
        """Call DeepSeek's OpenAI-compatible chat completions API."""
        api_key = api_key_override or settings.DEEPSEEK_API_KEY
        if not api_key:
            raise LLMServiceError("DEEPSEEK_API_KEY is not set")

        api_base = api_base_override or settings.DEEPSEEK_API_BASE
        url = f"{api_base.rstrip('/')}/chat/completions"
        timeout = settings.DEEPSEEK_TIMEOUT_SECONDS or 120

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
            return (content or "").strip()
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
    ) -> str:
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
            return (content or "").strip()
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
    
    async def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List all available models in Ollama.
        
        Returns:
            List of model dictionaries with model information
        """
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            
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
