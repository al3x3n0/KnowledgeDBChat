"""
Context management service for RAG system.
Handles context compression, summarization, and token-aware window management.
"""

from typing import List, Dict, Any, Optional, TYPE_CHECKING
from loguru import logger

from app.core.config import settings

if TYPE_CHECKING:
    from app.services.llm_service import UserLLMSettings


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
        max_tokens: Optional[int] = None,
        user_settings: Optional["UserLLMSettings"] = None,
    ) -> str:
        """
        Compress context using LLM summarization if available, otherwise truncate.

        Args:
            results: List of search results
            llm_service: Optional LLM service for summarization
            max_tokens: Maximum tokens for compressed context
            user_settings: Optional user LLM settings for provider preference

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
                    conversation_history=None,
                    prefer_deepseek=True,  # Route heavy compression to external provider if available
                    user_settings=user_settings,
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

    def build_kg_context(
        self,
        entities: List[Any],
        relationships: List[Any],
        max_entities: Optional[int] = None,
        max_relationships: Optional[int] = None
    ) -> str:
        """
        Build knowledge graph context string for LLM prompt.

        Creates a structured text representation of entities and their
        relationships to inject into the chat context.

        Args:
            entities: List of Entity objects from knowledge graph
            relationships: List of Relationship objects
            max_entities: Maximum entities to include (uses config if None)
            max_relationships: Maximum relationships to include

        Returns:
            Formatted knowledge graph context string
        """
        if not entities:
            return ""

        max_entities = max_entities or settings.RAG_KG_MAX_ENTITIES
        max_relationships = max_relationships or settings.RAG_KG_MAX_RELATIONSHIPS

        parts = ["\n--- Knowledge Graph Context ---"]

        # Build entity map for quick lookup
        entity_map = {}
        for e in entities[:max_entities]:
            entity_map[str(e.id)] = e
            entity_info = f"• {e.canonical_name} ({e.entity_type})"
            if hasattr(e, 'description') and e.description:
                entity_info += f": {e.description[:100]}"
            parts.append(entity_info)

        # Add relationships section if present
        if relationships:
            parts.append("\nRelationships:")
            rel_count = 0
            for r in relationships:
                if rel_count >= max_relationships:
                    break

                # Get entity names from map or use IDs
                source_id = str(r.source_entity_id)
                target_id = str(r.target_entity_id)

                source_name = entity_map.get(source_id)
                target_name = entity_map.get(target_id)

                if source_name:
                    source_name = source_name.canonical_name
                else:
                    source_name = f"[Entity:{source_id[:8]}]"

                if target_name:
                    target_name = target_name.canonical_name
                else:
                    target_name = f"[Entity:{target_id[:8]}]"

                rel_line = f"  {source_name} --[{r.relation_type}]--> {target_name}"
                if hasattr(r, 'confidence') and r.confidence:
                    rel_line += f" (confidence: {r.confidence:.2f})"
                parts.append(rel_line)
                rel_count += 1

        kg_context = "\n".join(parts)
        logger.debug(f"Built KG context with {len(entities)} entities, {len(relationships)} relationships")
        return kg_context

    def extract_entity_names_from_results(
        self,
        results: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Extract potential entity names from search results.

        Looks for capitalized phrases and proper nouns that might be entity names.

        Args:
            results: List of search results

        Returns:
            List of potential entity names
        """
        import re

        entity_names = set()

        for result in results:
            content = result.get("content", result.get("page_content", ""))
            if not content:
                continue

            # Find capitalized phrases (potential names)
            # Pattern: Two or more capitalized words together
            pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
            matches = re.findall(pattern, content)
            for match in matches:
                if len(match) > 3:  # Filter out very short matches
                    entity_names.add(match)

            # Also get metadata entities if present
            metadata = result.get("metadata", {})
            if "entities" in metadata:
                for ent in metadata["entities"]:
                    if isinstance(ent, str):
                        entity_names.add(ent)
                    elif isinstance(ent, dict) and "name" in ent:
                        entity_names.add(ent["name"])

        # Return as sorted list, limited
        return sorted(list(entity_names))[:50]
