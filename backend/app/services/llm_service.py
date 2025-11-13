"""
LLM service for generating responses using local LLM deployment.
"""

import httpx
from typing import Optional, List, Dict, Any
from loguru import logger

from app.core.config import settings
from app.utils.exceptions import LLMServiceError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


class LLMService:
    """Service for interacting with local LLM (Ollama)."""
    
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.default_model = settings.DEFAULT_MODEL
        self.client = httpx.AsyncClient(timeout=120.0)
    
    async def generate_response(
        self,
        query: str,
        context: Optional[str] = None,
        conversation_history: Optional[str] = None,
        memory_context: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate a response using the local LLM.
        
        Args:
            query: User's question or query
            context: Optional context from knowledge base documents
            conversation_history: Optional previous conversation history
            memory_context: Optional context from user's conversation memories
            model: Optional model name (uses default if not provided)
            temperature: Optional temperature for generation (uses default if not provided)
            max_tokens: Optional maximum tokens for response (uses default if not provided)
            
        Returns:
            Generated response string
            
        Raises:
            LLMServiceError: If LLM service is unavailable or request fails
        """
        try:
            model = model or self.default_model
            temperature = temperature or settings.TEMPERATURE
            max_tokens = max_tokens or settings.MAX_RESPONSE_LENGTH
            
            # Build the prompt
            prompt = self._build_prompt(query, context, conversation_history, memory_context)
            
            # Make request to Ollama
            response = await self._make_ollama_request(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            raise LLMServiceError(f"Failed to generate response: {str(e)}")
    
    def _build_prompt(
        self,
        query: str,
        context: Optional[str] = None,
        conversation_history: Optional[str] = None,
        memory_context: Optional[str] = None
    ) -> str:
        """
        Build the complete prompt for the LLM.
        
        Args:
            query: User's question or query
            context: Optional context from knowledge base
            conversation_history: Optional previous conversation history
            memory_context: Optional context from user's conversation memories
            
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
7. Use relevant memories from past conversations to provide personalized responses"""
        
        prompt_parts.append(system_instruction)
        
        # Add memory context if provided (most relevant for personalization)
        if memory_context:
            prompt_parts.append(f"\nRelevant memories from past conversations:\n{memory_context}")
        
        # Add context if provided
        if context:
            prompt_parts.append(f"\nContext from knowledge base:\n{context}")
        
        # Add conversation history if provided
        if conversation_history:
            prompt_parts.append(f"\nPrevious conversation:\n{conversation_history}")
        
        # Add the current query
        prompt_parts.append(f"\nUser question: {query}")
        prompt_parts.append("\nAssistant response:")
        
        return "\n".join(prompt_parts)
    
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
        max_tokens: int
    ) -> str:
        """
        Make a request to the Ollama API with retry logic.
        
        Args:
            model: Model name to use
            prompt: Complete prompt to send
            temperature: Temperature for generation
            max_tokens: Maximum tokens for response
            
        Returns:
            Generated response text
            
        Raises:
            LLMServiceError: If request fails after retries
            httpx.HTTPError: If HTTP request fails
        """
        try:
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
                f"{self.base_url}/api/generate",
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
        Check if the Ollama service is healthy.
        
        Returns:
            True if service is healthy, False otherwise
        """
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()


