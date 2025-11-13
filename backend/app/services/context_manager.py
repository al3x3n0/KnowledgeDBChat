"""
Context management service for RAG system.
Handles context compression, summarization, and token-aware window management.
"""

from typing import List, Dict, Any, Optional
from loguru import logger

from app.core.config import settings


class ContextManager:
    """Service for managing and optimizing context for LLM prompts."""
    
    def __init__(self):
        # Rough estimation: 1 token ≈ 4 characters for English text
        self.chars_per_token = 4
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate number of tokens in text.
        
        Args:
            text: Text to estimate
            
        Returns:
            Estimated token count
        """
        return len(text) // self.chars_per_token
    
    def truncate_context(
        self,
        results: List[Dict[str, Any]],
        max_tokens: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Truncate context to fit within token limit, prioritizing higher-scoring results.
        
        Args:
            results: List of search results with scores
            max_tokens: Maximum tokens allowed (uses config if None)
            
        Returns:
            Truncated list of results
        """
        max_tokens = max_tokens or settings.RAG_MAX_CONTEXT_TOKENS
        
        # Sort by score (highest first)
        sorted_results = sorted(results, key=lambda x: x.get("score", 0.0), reverse=True)
        
        truncated = []
        total_tokens = 0
        
        for result in sorted_results:
            content = result.get("content", result.get("page_content", ""))
            content_tokens = self.estimate_tokens(content)
            
            if total_tokens + content_tokens <= max_tokens:
                truncated.append(result)
                total_tokens += content_tokens
            else:
                # Try to fit partial content if there's remaining space
                remaining_tokens = max_tokens - total_tokens
                if remaining_tokens > 50:  # Only if meaningful space remains
                    remaining_chars = remaining_tokens * self.chars_per_token
                    partial_result = result.copy()
                    partial_result["content"] = content[:remaining_chars] + "..."
                    partial_result["page_content"] = partial_result["content"]
                    truncated.append(partial_result)
                break
        
        logger.debug(f"Truncated context from {len(results)} to {len(truncated)} results ({total_tokens} tokens)")
        return truncated
    
    def filter_by_relevance(
        self,
        results: List[Dict[str, Any]],
        min_score: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Filter results by relevance score.
        
        Args:
            results: List of search results
            min_score: Minimum relevance score (uses config if None)
            
        Returns:
            Filtered list of results
        """
        min_score = min_score or settings.RAG_MIN_RELEVANCE_SCORE
        
        filtered = [r for r in results if r.get("score", 0.0) >= min_score]
        
        logger.debug(f"Filtered {len(results)} results to {len(filtered)} by relevance (min_score={min_score})")
        return filtered
    
    async def compress_context(
        self,
        results: List[Dict[str, Any]],
        llm_service: Optional[Any] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Compress context using LLM summarization if available, otherwise truncate.
        
        Args:
            results: List of search results
            llm_service: Optional LLM service for summarization
            max_tokens: Maximum tokens for compressed context
            
        Returns:
            Compressed context string
        """
        max_tokens = max_tokens or settings.RAG_MAX_CONTEXT_TOKENS
        
        if not results:
            return ""
        
        # Build full context first
        full_context = self.build_context_string(results)
        full_tokens = self.estimate_tokens(full_context)
        
        # If within limit, return as-is
        if full_tokens <= max_tokens:
            return full_context
        
        # Try LLM compression if available
        if llm_service:
            try:
                summary_prompt = f"""Summarize the following context while preserving key information and facts:

{full_context}

Provide a concise summary that captures the essential information:"""
                
                compressed = await llm_service.generate_response(
                    query=summary_prompt,
                    context=None,
                    conversation_history=None
                )
                
                compressed_tokens = self.estimate_tokens(compressed)
                if compressed_tokens <= max_tokens:
                    logger.info(f"Compressed context from {full_tokens} to {compressed_tokens} tokens using LLM")
                    return compressed
                else:
                    logger.warning(f"LLM compression still exceeds limit ({compressed_tokens} > {max_tokens}), truncating")
            except Exception as e:
                logger.warning(f"LLM compression failed: {e}, using truncation instead")
        
        # Fall back to truncation
        truncated_results = self.truncate_context(results, max_tokens)
        return self.build_context_string(truncated_results)
    
    def build_context_string(self, results: List[Dict[str, Any]]) -> str:
        """
        Build context string from search results.
        
        Args:
            results: List of search results
            
        Returns:
            Formatted context string
        """
        if not results:
            return ""
        
        context_parts = []
        for i, result in enumerate(results, 1):
            content = result.get("content", result.get("page_content", ""))
            metadata = result.get("metadata", {})
            source = metadata.get("source", "Unknown")
            title = metadata.get("title", "Unknown Document")
            score = result.get("score", 0.0)
            
            context_parts.append(
                f"Source {i} ({source} - {title}, relevance: {score:.2f}):\n{content}"
            )
        
        return "\n\n".join(context_parts)
    
    def select_relevant_parts(
        self,
        results: List[Dict[str, Any]],
        query: str,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Select most relevant parts from results based on query.
        
        Args:
            results: List of search results
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            Selected results
        """
        # Results are already sorted by score, so just take top N
        selected = results[:max_results]
        
        logger.debug(f"Selected {len(selected)} most relevant results from {len(results)} total")
        return selected
    
    def get_context_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get metrics about context quality.
        
        Args:
            results: List of search results
            
        Returns:
            Dictionary with context metrics
        """
        if not results:
            return {
                "total_results": 0,
                "total_tokens": 0,
                "avg_score": 0.0,
                "coverage": 0.0
            }
        
        total_tokens = sum(
            self.estimate_tokens(r.get("content", r.get("page_content", "")))
            for r in results
        )
        
        scores = [r.get("score", 0.0) for r in results]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        # Coverage: percentage of results above minimum relevance
        min_score = settings.RAG_MIN_RELEVANCE_SCORE
        above_threshold = sum(1 for s in scores if s >= min_score)
        coverage = (above_threshold / len(scores)) * 100 if scores else 0.0
        
        return {
            "total_results": len(results),
            "total_tokens": total_tokens,
            "avg_score": avg_score,
            "coverage": coverage,
            "min_score": min(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0
        }

