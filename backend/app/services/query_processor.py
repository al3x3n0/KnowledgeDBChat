"""
Query processing service for RAG system.
Handles query normalization, expansion, rewriting, and multi-query generation.
"""

import re
from typing import List, Optional, Dict, Any
from loguru import logger

from app.core.config import settings


class QueryProcessor:
    """Service for processing and enhancing queries for better retrieval."""
    
    def __init__(self):
        self.stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
            'had', 'what', 'said', 'each', 'which', 'their', 'time', 'if',
            'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some', 'her',
            'would', 'make', 'like', 'into', 'him', 'has', 'two', 'more', 'very',
            'after', 'words', 'long', 'than', 'first', 'been', 'call', 'who',
            'oil', 'sit', 'now', 'find', 'down', 'day', 'did', 'get', 'come',
            'made', 'may', 'part'
        }
    
    def normalize_query(self, query: str) -> str:
        """
        Normalize query text for consistent processing.
        
        Args:
            query: Raw query string
            
        Returns:
            Normalized query string
        """
        if not query:
            return ""
        
        # Convert to lowercase
        normalized = query.lower()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove leading/trailing whitespace
        normalized = normalized.strip()
        
        return normalized
    
    def clean_query(self, query: str, remove_stopwords: bool = False) -> str:
        """
        Clean query by removing special characters and optionally stopwords.
        
        Args:
            query: Query string to clean
            remove_stopwords: Whether to remove stopwords
            
        Returns:
            Cleaned query string
        """
        if not query:
            return ""
        
        # Remove special characters except spaces and basic punctuation
        cleaned = re.sub(r'[^\w\s\-]', ' ', query)
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        if remove_stopwords:
            # Split into words and filter stopwords
            words = cleaned.split()
            cleaned = ' '.join([w for w in words if w.lower() not in self.stopwords])
        
        return cleaned
    
    def extract_key_terms(self, query: str) -> List[str]:
        """
        Extract key terms from query.
        
        Args:
            query: Query string
            
        Returns:
            List of key terms
        """
        # Clean query
        cleaned = self.clean_query(query, remove_stopwords=True)
        
        # Split into terms
        terms = [term for term in cleaned.split() if len(term) > 2]
        
        return terms
    
    def expand_query_synonyms(self, query: str, synonyms: Optional[Dict[str, List[str]]] = None) -> str:
        """
        Expand query with synonyms.
        
        Args:
            query: Original query
            synonyms: Optional dictionary of term -> synonyms mapping
            
        Returns:
            Expanded query with synonyms
        """
        if not synonyms:
            # Basic synonym expansion could be added here
            # For now, return original query
            return query
        
        words = query.split()
        expanded_words = []
        
        for word in words:
            expanded_words.append(word)
            if word.lower() in synonyms:
                expanded_words.extend(synonyms[word.lower()])
        
        return ' '.join(expanded_words)
    
    def rewrite_query(self, query: str) -> str:
        """
        Rewrite query for better retrieval.
        Currently implements basic rewriting rules.
        
        Args:
            query: Original query
            
        Returns:
            Rewritten query
        """
        # Normalize first
        rewritten = self.normalize_query(query)
        
        # Expand common abbreviations
        abbreviations = {
            'how to': 'how do I',
            'how do': 'how to',
            'what is': 'what are',
            'explain': 'describe',
        }
        
        for abbrev, expansion in abbreviations.items():
            if abbrev in rewritten:
                rewritten = rewritten.replace(abbrev, expansion)
        
        return rewritten
    
    async def generate_query_variations(
        self,
        query: str,
        llm_service: Optional[Any] = None,
        max_variations: int = 3
    ) -> List[str]:
        """
        Generate multiple query variations using LLM or rule-based approach.
        
        Args:
            query: Original query
            llm_service: Optional LLM service for generating variations
            max_variations: Maximum number of variations to generate
            
        Returns:
            List of query variations including original
        """
        variations = [query]  # Always include original
        
        if llm_service:
            try:
                # Use LLM to generate variations
                prompt = f"""Generate {max_variations} different ways to ask this question:
"{query}"

Return only the variations, one per line, without numbering or bullets."""

                response = await llm_service.generate_response(
                    query=prompt,
                    context=None,
                    conversation_history=None,
                    task_type="query_expansion",
                )
                
                # Parse variations from response
                lines = response.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    # Remove numbering/bullets
                    line = re.sub(r'^[\d\.\-\*]\s*', '', line)
                    if line and line != query and len(variations) < max_variations + 1:
                        variations.append(line)
            except Exception as e:
                logger.warning(f"Failed to generate LLM query variations: {e}")
        
        # Add rule-based variations
        if len(variations) < max_variations + 1:
            # Add question form variations
            if not query.endswith('?'):
                variations.append(query + '?')
            
            # Add "what is" prefix if not present
            if not query.lower().startswith(('what', 'how', 'why', 'when', 'where', 'who')):
                variations.append(f"what is {query}")
            
            # Add alternative question words
            query_lower = query.lower()
            if query_lower.startswith('what'):
                variations.append(query.replace('what', 'how', 1))
            elif query_lower.startswith('how'):
                variations.append(query.replace('how', 'what', 1))
        
        # Limit to max_variations + 1 (including original)
        return variations[:max_variations + 1]
    
    async def generate_multi_queries(
        self,
        query: str,
        llm_service: Optional[Any] = None,
        num_queries: int = 3
    ) -> List[str]:
        """
        Generate multiple query variations for improved recall.
        This is an enhanced version that focuses on different aspects of the query.
        
        Args:
            query: Original query
            llm_service: Optional LLM service
            num_queries: Number of query variations to generate
            
        Returns:
            List of query variations
        """
        variations = await self.generate_query_variations(query, llm_service, num_queries - 1)
        
        # Ensure we have the requested number
        while len(variations) < num_queries:
            # Add more rule-based variations
            query_lower = query.lower()
            
            # Add "explain" variation
            if 'explain' not in query_lower:
                variations.append(f"explain {query}")
            
            # Add "describe" variation
            if 'describe' not in query_lower:
                variations.append(f"describe {query}")
            
            # Break if we can't generate more
            if len(variations) >= num_queries:
                break
        
        return variations[:num_queries]
    
    def classify_query_intent(self, query: str) -> str:
        """
        Classify query by intent.
        
        Args:
            query: Query string
            
        Returns:
            Intent classification: 'factual', 'analytical', 'procedural', 'conversational'
        """
        query_lower = query.lower()
        
        # Factual queries (what, who, when, where)
        if any(word in query_lower for word in ['what is', 'who is', 'when', 'where', 'what are']):
            return 'factual'
        
        # Procedural queries (how to, steps, process)
        if any(word in query_lower for word in ['how to', 'how do', 'steps', 'process', 'procedure']):
            return 'procedural'
        
        # Analytical queries (why, compare, analyze, difference)
        if any(word in query_lower for word in ['why', 'compare', 'analyze', 'difference', 'similar', 'different']):
            return 'analytical'
        
        # Default to conversational
        return 'conversational'
    
    def process_query(
        self,
        query: str,
        expand: bool = True,
        rewrite: bool = True,
        normalize: bool = True
    ) -> Dict[str, Any]:
        """
        Process query with all available techniques.
        
        Args:
            query: Original query
            expand: Whether to expand query with synonyms
            rewrite: Whether to rewrite query
            normalize: Whether to normalize query
            
        Returns:
            Dictionary with processed query and metadata
        """
        original_query = query
        
        # Normalize
        if normalize:
            query = self.normalize_query(query)
        
        # Rewrite
        if rewrite:
            query = self.rewrite_query(query)
        
        # Extract key terms
        key_terms = self.extract_key_terms(query)
        
        # Classify intent
        intent = self.classify_query_intent(query)
        
        # Expand (if enabled)
        expanded_query = query
        if expand:
            expanded_query = self.expand_query_synonyms(query)
        
        return {
            'original': original_query,
            'processed': query,
            'expanded': expanded_query,
            'key_terms': key_terms,
            'intent': intent,
            'cleaned': self.clean_query(query)
        }

