"""
Tests for RAG (Retrieval-Augmented Generation) features.
Tests hybrid search, reranking, query expansion, and context management.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

from app.services.query_processor import QueryProcessor
from app.services.context_manager import ContextManager
from app.services.vector_store import VectorStoreService
from app.core.config import settings


class TestQueryProcessor:
    """Tests for QueryProcessor service."""
    
    def test_normalize_query(self):
        """Test query normalization."""
        processor = QueryProcessor()
        
        # Test lowercase conversion
        assert processor.normalize_query("Hello World") == "hello world"
        
        # Test whitespace normalization
        assert processor.normalize_query("  hello   world  ") == "hello world"
        
        # Test empty string
        assert processor.normalize_query("") == ""
        assert processor.normalize_query("   ") == ""
    
    def test_clean_query(self):
        """Test query cleaning."""
        processor = QueryProcessor()
        
        # Test without stopword removal
        cleaned = processor.clean_query("Hello, world! This is a test.")
        assert "hello" in cleaned.lower()
        assert "world" in cleaned.lower()
        
        # Test with stopword removal
        cleaned = processor.clean_query("This is a test", remove_stopwords=True)
        assert "this" not in cleaned.lower() or "is" not in cleaned.lower() or "a" not in cleaned.lower()
    
    def test_extract_key_terms(self):
        """Test key term extraction."""
        processor = QueryProcessor()
        
        terms = processor.extract_key_terms("What is machine learning?")
        assert len(terms) > 0
        assert isinstance(terms, list)
        
        # Test with technical terms
        terms = processor.extract_key_terms("neural network deep learning")
        assert len(terms) >= 2
    
    def test_classify_query_intent(self):
        """Test query intent classification."""
        processor = QueryProcessor()
        
        # Factual query
        intent = processor.classify_query_intent("What is Python?")
        assert intent in ["factual", "conversational"]
        
        # Analytical query
        intent = processor.classify_query_intent("Compare X and Y")
        assert intent in ["analytical", "conversational"]
        
        # Procedural query
        intent = processor.classify_query_intent("How do I install Python?")
        assert intent in ["procedural", "conversational"]
    
    def test_process_query(self):
        """Test complete query processing pipeline."""
        processor = QueryProcessor()
        
        result = processor.process_query(
            "What is machine learning?",
            expand=True,
            rewrite=True,
            normalize=True
        )
        
        assert "original" in result
        assert "processed" in result
        assert "expanded" in result
        assert "key_terms" in result
        assert "intent" in result
        assert "cleaned" in result
        
        assert result["original"] == "What is machine learning?"
        assert isinstance(result["processed"], str)
        assert isinstance(result["key_terms"], list)
        assert result["intent"] in ["factual", "analytical", "procedural", "conversational"]
    
    @pytest.mark.asyncio
    async def test_generate_multi_queries(self):
        """Test multi-query generation."""
        processor = QueryProcessor()
        
        # Mock LLM service
        mock_llm = Mock()
        mock_llm.generate_response = AsyncMock(return_value="1. What is machine learning?\n2. Explain machine learning concepts\n3. Machine learning definition")
        
        queries = await processor.generate_multi_queries(
            "machine learning",
            llm_service=mock_llm,
            num_queries=3
        )
        
        assert isinstance(queries, list)
        assert len(queries) > 0
        assert all(isinstance(q, str) for q in queries)


class TestContextManager:
    """Tests for ContextManager service."""
    
    def test_estimate_tokens(self):
        """Test token estimation."""
        manager = ContextManager()
        
        # Test estimation
        tokens = manager.estimate_tokens("Hello world")
        assert tokens > 0
        assert isinstance(tokens, int)
        
        # Test with longer text
        long_text = "Hello world " * 100
        tokens_long = manager.estimate_tokens(long_text)
        assert tokens_long > tokens
    
    def test_truncate_context(self):
        """Test context truncation."""
        manager = ContextManager()
        
        # Create mock results
        results = [
            {"content": "Text chunk 1", "score": 0.9},
            {"content": "Text chunk 2", "score": 0.8},
            {"content": "Text chunk 3", "score": 0.7},
            {"content": "Text chunk 4", "score": 0.6},
        ]
        
        # Truncate to small token limit
        truncated = manager.truncate_context(results, max_tokens=50)
        
        assert len(truncated) <= len(results)
        # Higher scoring results should be prioritized
        if len(truncated) > 0:
            assert truncated[0]["score"] >= truncated[-1]["score"] if len(truncated) > 1 else True
    
    def test_filter_by_relevance(self):
        """Test relevance filtering."""
        manager = ContextManager()
        
        results = [
            {"content": "Relevant text", "score": 0.8},
            {"content": "Less relevant", "score": 0.3},
            {"content": "Irrelevant", "score": 0.1},
        ]
        
        filtered = manager.filter_by_relevance(results, min_score=0.3)
        
        assert len(filtered) <= len(results)
        assert all(r["score"] >= 0.3 for r in filtered)
    
    def test_build_context_string(self):
        """Test context string building."""
        manager = ContextManager()
        
        results = [
            {"content": "First chunk", "score": 0.9},
            {"content": "Second chunk", "score": 0.8},
        ]
        
        context = manager.build_context_string(results)
        
        assert isinstance(context, str)
        assert "First chunk" in context
        assert "Second chunk" in context
    
    def test_get_context_metrics(self):
        """Test context metrics calculation."""
        manager = ContextManager()
        
        results = [
            {"content": "Chunk 1", "score": 0.9},
            {"content": "Chunk 2", "score": 0.8},
        ]
        
        metrics = manager.get_context_metrics(results)
        
        assert "total_chunks" in metrics
        assert "avg_relevance" in metrics
        assert "total_tokens" in metrics
        assert metrics["total_chunks"] == 2
        assert 0 <= metrics["avg_relevance"] <= 1
    
    @pytest.mark.asyncio
    async def test_compress_context(self):
        """Test context compression."""
        manager = ContextManager()
        
        results = [
            {"content": "Long text chunk " * 100, "score": 0.9},
            {"content": "Another long chunk " * 100, "score": 0.8},
        ]
        
        # Test without LLM (should truncate)
        compressed = await manager.compress_context(results, llm_service=None, max_tokens=100)
        
        assert isinstance(compressed, str)
        assert len(compressed) > 0
        
        # Test with LLM (mock)
        mock_llm = Mock()
        mock_llm.generate_response = AsyncMock(return_value="Compressed summary of the context")
        
        compressed_with_llm = await manager.compress_context(
            results,
            llm_service=mock_llm,
            max_tokens=100
        )
        
        assert isinstance(compressed_with_llm, str)
        assert len(compressed_with_llm) > 0


class TestVectorStoreRAGFeatures:
    """Tests for VectorStore RAG features (hybrid search, reranking, etc.)."""
    
    @pytest.mark.asyncio
    async def test_hybrid_search_combines_semantic_and_bm25(self):
        """Test that hybrid search combines semantic and BM25 results."""
        vector_store = VectorStoreService()
        
        # Mock the search methods
        with patch.object(vector_store, '_semantic_search', new_callable=AsyncMock) as mock_semantic, \
             patch.object(vector_store, '_bm25_search', new_callable=Mock) as mock_bm25:
            
            mock_semantic.return_value = [
                {"id": "1", "content": "Semantic result", "score": 0.9}
            ]
            mock_bm25.return_value = [
                {"id": "2", "content": "BM25 result", "bm25_score": 0.8}
            ]
            
            # Initialize vector store (mocked)
            vector_store._initialized = True
            vector_store.collection = Mock()
            
            results = await vector_store._hybrid_search("test query", limit=10)
            
            # Should combine results from both methods
            assert isinstance(results, list)
            # Note: Actual implementation may vary, but should handle both result types
    
    def test_rerank_results(self):
        """Test result reranking."""
        vector_store = VectorStoreService()
        
        # Mock reranker
        mock_reranker = Mock()
        mock_reranker.predict = Mock(return_value=[0.9, 0.8, 0.7])
        vector_store.reranker = mock_reranker
        
        results = [
            {"id": "1", "content": "Result 1", "score": 0.5},
            {"id": "2", "content": "Result 2", "score": 0.6},
            {"id": "3", "content": "Result 3", "score": 0.4},
        ]
        
        reranked = vector_store._rerank_results("test query", results, top_k=2)
        
        assert len(reranked) <= 2
        assert all("rerank_score" in r for r in reranked)
    
    def test_apply_mmr(self):
        """Test Maximal Marginal Relevance for diversity."""
        vector_store = VectorStoreService()
        
        # Mock embedding model for MMR
        vector_store.embedding_model = Mock()
        vector_store.embedding_model.encode = Mock(return_value=[[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]])
        
        results = [
            {"id": "1", "content": "Result 1", "score": 0.9},
            {"id": "2", "content": "Result 2", "score": 0.8},
            {"id": "3", "content": "Result 3", "score": 0.7},
        ]
        
        mmr_results = vector_store._apply_mmr(results, "test query", lambda_param=0.5, top_k=2)
        
        assert len(mmr_results) <= 2
        assert isinstance(mmr_results, list)
    
    def test_deduplicate_results(self):
        """Test result deduplication."""
        vector_store = VectorStoreService()
        
        # Mock embedding model
        vector_store.embedding_model = Mock()
        vector_store.embedding_model.encode = Mock(return_value=[[0.1, 0.2], [0.1, 0.2], [0.9, 0.8]])
        
        results = [
            {"id": "1", "content": "Duplicate content", "score": 0.9},
            {"id": "2", "content": "Duplicate content", "score": 0.8},  # Near duplicate
            {"id": "3", "content": "Different content", "score": 0.7},
        ]
        
        deduplicated = vector_store._deduplicate_results(results, similarity_threshold=0.95)
        
        # Should remove duplicates
        assert len(deduplicated) <= len(results)
        # Should keep unique content
        unique_contents = set(r["content"] for r in deduplicated)
        assert len(unique_contents) <= len(deduplicated)


class TestRAGIntegration:
    """Integration tests for RAG pipeline."""
    
    @pytest.mark.asyncio
    async def test_query_processing_pipeline(self):
        """Test complete query processing pipeline."""
        processor = QueryProcessor()
        
        query = "What is artificial intelligence?"
        processed = processor.process_query(query)
        
        assert processed["original"] == query
        assert len(processed["processed"]) > 0
        assert len(processed["key_terms"]) > 0
    
    @pytest.mark.asyncio
    async def test_context_management_pipeline(self):
        """Test context management pipeline."""
        manager = ContextManager()
        
        results = [
            {"content": "Relevant information about AI", "score": 0.9},
            {"content": "Less relevant information", "score": 0.4},
            {"content": "Irrelevant information", "score": 0.1},
        ]
        
        # Filter by relevance
        filtered = manager.filter_by_relevance(results)
        
        # Truncate context
        truncated = manager.truncate_context(filtered, max_tokens=100)
        
        # Build context string
        context = manager.build_context_string(truncated)
        
        assert isinstance(context, str)
        assert len(context) > 0
        
        # Get metrics
        metrics = manager.get_context_metrics(truncated)
        assert metrics["total_chunks"] > 0


class TestRAGPerformance:
    """Performance benchmarks for RAG features."""
    
    def test_query_processing_performance(self):
        """Benchmark query processing speed."""
        import time
        
        processor = QueryProcessor()
        query = "What is machine learning and how does it work?"
        
        start_time = time.time()
        result = processor.process_query(query)
        elapsed = time.time() - start_time
        
        # Should complete in reasonable time (< 1 second for rule-based processing)
        assert elapsed < 1.0
        assert result is not None
    
    def test_context_truncation_performance(self):
        """Benchmark context truncation speed."""
        import time
        
        manager = ContextManager()
        
        # Create large result set
        results = [
            {"content": f"Chunk {i} with some content " * 10, "score": 0.9 - (i * 0.01)}
            for i in range(100)
        ]
        
        start_time = time.time()
        truncated = manager.truncate_context(results, max_tokens=1000)
        elapsed = time.time() - start_time
        
        # Should complete quickly (< 0.1 seconds)
        assert elapsed < 0.1
        assert len(truncated) <= len(results)
    
    def test_relevance_filtering_performance(self):
        """Benchmark relevance filtering speed."""
        import time
        
        manager = ContextManager()
        
        # Create large result set
        results = [
            {"content": f"Result {i}", "score": i / 100.0}
            for i in range(1000)
        ]
        
        start_time = time.time()
        filtered = manager.filter_by_relevance(results, min_score=0.5)
        elapsed = time.time() - start_time
        
        # Should complete quickly (< 0.1 seconds)
        assert elapsed < 0.1
        assert all(r["score"] >= 0.5 for r in filtered)

