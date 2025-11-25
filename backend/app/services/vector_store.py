"""
Vector store service for semantic search using ChromaDB.
"""

import os
import hashlib
from typing import List, Optional, Dict, Any
from uuid import UUID
import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from loguru import logger

from app.core.config import settings
from app.models.document import Document, DocumentChunk


class VectorStoreService:
    """Service for managing vector embeddings and similarity search."""
    
    def __init__(self):
        self.client = None
        self.collection = None
        self.embedding_model = None
        self.current_model_name = None
        self.reranker = None
        self.bm25_index = None
        self.bm25_documents = []
        self.bm25_doc_ids = []
        self._initialized = False
    
    async def initialize(self, embedding_model: Optional[str] = None):
        """
        Initialize ChromaDB client and collection.
        
        Args:
            embedding_model: Optional embedding model name (uses config if None)
        """
        try:
            # Ensure the persist directory exists
            os.makedirs(settings.CHROMA_PERSIST_DIRECTORY, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIRECTORY,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(
                    name=settings.CHROMA_COLLECTION_NAME
                )
                logger.info(f"Loaded existing collection: {settings.CHROMA_COLLECTION_NAME}")
            except Exception:
                self.collection = self.client.create_collection(
                    name=settings.CHROMA_COLLECTION_NAME,
                    metadata={"description": "Knowledge base document chunks"}
                )
                logger.info(f"Created new collection: {settings.CHROMA_COLLECTION_NAME}")
            
            # Initialize embedding model
            model_name = embedding_model or settings.EMBEDDING_MODEL
            self.embedding_model = SentenceTransformer(model_name)
            self.current_model_name = model_name
            logger.info(f"Loaded embedding model: {model_name}")
            
            # Initialize BM25 index for hybrid search if enabled
            if settings.RAG_HYBRID_SEARCH_ENABLED:
                self._build_bm25_index()
            
            # Initialize reranker if enabled
            if settings.RAG_RERANKING_ENABLED:
                try:
                    self.reranker = CrossEncoder(settings.RAG_RERANKING_MODEL)
                    # Test the reranker to catch CPU compatibility issues early
                    test_scores = self.reranker.predict([["test query", "test document"]])
                    logger.info(f"Loaded reranking model: {settings.RAG_RERANKING_MODEL}")
                except Exception as e:
                    error_msg = str(e)
                    if "primitive descriptor" in error_msg.lower() or "matmul" in error_msg.lower():
                        logger.warning(
                            f"Reranking model has CPU compatibility issues ({error_msg}). "
                            "Reranking will be disabled. This is often due to Intel MKL/oneDNN compatibility. "
                            "Consider using a different reranking model or disabling reranking in settings."
                        )
                    else:
                        logger.warning(f"Failed to load reranking model: {e}. Reranking disabled.")
                    self.reranker = None
                    settings.RAG_RERANKING_ENABLED = False
            
            self._initialized = True
            logger.info("Vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    def _ensure_initialized(self):
        """Ensure the vector store is initialized."""
        if not self._initialized:
            raise RuntimeError("Vector store not initialized. Call initialize() first.")
    
    def _generate_embedding_id(self, document_id: UUID, chunk_index: int) -> str:
        """Generate a unique embedding ID for a document chunk."""
        return f"doc_{document_id}_chunk_{chunk_index}"
    
    def _generate_embedding_hash(self, content: str) -> str:
        """Generate a hash for embedding content to detect changes."""
        model_name = self.current_model_name or settings.EMBEDDING_MODEL
        model_info = f"{model_name}_{content}"
        return hashlib.sha256(model_info.encode()).hexdigest()
    
    async def switch_embedding_model(self, new_model: str) -> bool:
        """
        Switch to a different embedding model.
        
        Args:
            new_model: Name of the new embedding model
            
        Returns:
            True if switch was successful, False otherwise
            
        Note:
            Changing models requires reprocessing all documents for consistency.
            This method only switches the model for new embeddings.
        """
        if new_model not in settings.EMBEDDING_MODEL_OPTIONS:
            logger.warning(f"Model {new_model} not in allowed options: {settings.EMBEDDING_MODEL_OPTIONS}")
            return False
        
        try:
            # Load new model
            new_embedding_model = SentenceTransformer(new_model)
            
            # Switch models
            old_model = self.current_model_name
            self.embedding_model = new_embedding_model
            self.current_model_name = new_model
            
            logger.info(f"Switched embedding model from {old_model} to {new_model}")
            logger.warning(
                "Note: Existing embeddings were created with the old model. "
                "For best results, reprocess all documents with the new model."
            )
            
            return True
        except Exception as e:
            logger.error(f"Failed to switch embedding model: {e}")
            return False
    
    def get_current_model(self) -> Optional[str]:
        """Get the name of the currently loaded embedding model."""
        return self.current_model_name
    
    def _build_bm25_index(self):
        """Build BM25 index from all documents in collection."""
        try:
            # Get all documents from collection
            all_docs = self.collection.get(include=["documents", "metadatas"])
            
            if not all_docs["ids"]:
                logger.info("No documents in collection for BM25 index")
                return
            
            # Tokenize documents for BM25
            tokenized_docs = []
            for doc in all_docs["documents"]:
                # Simple tokenization (split on whitespace and lowercase)
                tokens = doc.lower().split()
                tokenized_docs.append(tokens)
            
            # Build BM25 index
            self.bm25_index = BM25Okapi(tokenized_docs)
            self.bm25_documents = all_docs["documents"]
            self.bm25_doc_ids = all_docs["ids"]
            
            logger.info(f"Built BM25 index with {len(tokenized_docs)} documents")
        except Exception as e:
            logger.warning(f"Failed to build BM25 index: {e}")
            self.bm25_index = None
    
    def _bm25_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform BM25 keyword search.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of search results with BM25 scores
        """
        if not self.bm25_index or not self.bm25_doc_ids:
            return []
        
        try:
            # Tokenize query
            query_tokens = query.lower().split()
            
            # Get BM25 scores
            scores = self.bm25_index.get_scores(query_tokens)
            
            # Get top results
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:limit]
            
            # Build results
            results = []
            for idx in top_indices:
                if scores[idx] > 0:  # Only include results with positive scores
                    # Get metadata from collection
                    doc_id = self.bm25_doc_ids[idx]
                    doc_metadata = self.collection.get(
                        ids=[doc_id],
                        include=["metadatas"]
                    )
                    
                    result = {
                        "id": doc_id,
                        "content": self.bm25_documents[idx],
                        "metadata": doc_metadata["metadatas"][0] if doc_metadata["metadatas"] else {},
                        "bm25_score": float(scores[idx]),
                        "page_content": self.bm25_documents[idx]
                    }
                    results.append(result)
            
            return results
        except Exception as e:
            logger.error(f"Error in BM25 search: {e}")
            return []
    
    async def add_document_chunks(
        self,
        document: Document,
        chunks: List[DocumentChunk]
    ) -> List[str]:
        """Add document chunks to the vector store."""
        # Auto-initialize if not already initialized
        if not self._initialized:
            await self.initialize()
        
        self._ensure_initialized()
        
        if not chunks:
            logger.warning(f"No chunks provided for document {document.id}")
            return []
        
        try:
            # Prepare data for ChromaDB
            ids = []
            documents = []
            metadatas = []
            embeddings = []
            
            for chunk in chunks:
                embedding_id = self._generate_embedding_id(document.id, chunk.chunk_index)
                embedding_hash = self._generate_embedding_hash(chunk.content)
                
                # Check if embedding already exists and is up to date
                if chunk.embedding_id and chunk.embedding_hash == embedding_hash:
                    logger.debug(f"Skipping unchanged chunk {embedding_id}")
                    continue
                
                # Generate embedding
                embedding = self.embedding_model.encode(chunk.content).tolist()
                
                ids.append(embedding_id)
                documents.append(chunk.content)
                embeddings.append(embedding)
                
                # Build metadata, filtering out None values (ChromaDB doesn't allow None)
                # Avoid triggering lazy-loads on async ORM relationships inside this async context
                # Access only fields that are already loaded on the instance
                source_name = "Unknown"
                source_type = "Unknown"
                if "source" in document.__dict__ and document.__dict__["source"] is not None:
                    try:
                        src = document.__dict__["source"]
                        source_name = getattr(src, "name", source_name)
                        source_type = getattr(src, "source_type", source_type)
                    except Exception:
                        pass

                metadata_dict = {
                    "document_id": str(document.id),
                    "chunk_id": str(chunk.id),
                    "chunk_index": chunk.chunk_index,
                    "title": document.title,
                    "source": source_name,
                    "source_type": source_type,
                }
                
                # Add optional fields only if they're not None
                if document.url:
                    metadata_dict["url"] = document.url
                if document.file_type:
                    metadata_dict["file_type"] = document.file_type
                if document.author:
                    metadata_dict["author"] = document.author
                
                # Add timestamps
                metadata_dict["created_at"] = document.created_at.isoformat()
                metadata_dict["updated_at"] = document.updated_at.isoformat()
                
                metadatas.append(metadata_dict)
                
                # Update chunk with embedding info
                chunk.embedding_id = embedding_id
                chunk.embedding_hash = embedding_hash
            
            if ids:
                # Add to ChromaDB
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
                
                # Rebuild BM25 index if hybrid search is enabled
                if settings.RAG_HYBRID_SEARCH_ENABLED:
                    self._build_bm25_index()
                
                logger.info(f"Added {len(ids)} chunks to vector store for document {document.id}")
            
            return ids
            
        except Exception as e:
            logger.error(f"Error adding document chunks to vector store: {e}")
            raise
    
    async def search(
        self,
        query: str,
        limit: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using semantic similarity or hybrid search.
        
        Args:
            query: Search query text
            limit: Maximum number of results to return (default: 10)
            filter_metadata: Optional metadata filters (e.g., {"source_type": "file"})
            
        Returns:
            List of search result dictionaries, each containing:
            - id: Embedding ID
            - content: Document chunk content
            - metadata: Document metadata
            - score: Similarity score (0-1, higher is more similar)
            - page_content: Content (for LangChain compatibility)
        """
        # Auto-initialize if not already initialized
        if not self._initialized:
            await self.initialize()
        
        self._ensure_initialized()
        
        try:
            # Use hybrid search if enabled
            if settings.RAG_HYBRID_SEARCH_ENABLED:
                return await self._hybrid_search(query, limit, filter_metadata)
            else:
                return await self._semantic_search(query, limit, filter_metadata)
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []
    
    async def _semantic_search(
        self,
        query: str,
        limit: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform semantic search only."""
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=filter_metadata,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        search_results = []
        for i in range(len(results["ids"][0])):
            result = {
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1.0 - results["distances"][0][i],  # Convert distance to similarity score
                "page_content": results["documents"][0][i]  # For LangChain compatibility
            }
            
            # Add score to metadata for easier access
            result["metadata"]["score"] = result["score"]
            
            search_results.append(result)
        
        logger.info(f"Found {len(search_results)} semantic results for query: {query[:50]}...")
        
        # Apply reranking if enabled
        if settings.RAG_RERANKING_ENABLED and self.reranker and search_results:
            search_results = self._rerank_results(query, search_results)
        
        # Apply deduplication if enabled
        if settings.RAG_DEDUPLICATION_ENABLED and search_results:
            search_results = self._deduplicate_results(
                search_results,
                similarity_threshold=settings.RAG_DEDUPLICATION_THRESHOLD
            )
        
        # Apply MMR for diversity if enabled
        if settings.RAG_MMR_ENABLED and search_results:
            search_results = self._apply_mmr(
                search_results,
                query,
                lambda_param=settings.RAG_MMR_LAMBDA,
                top_k=limit
            )
        
        return search_results
    
    async def _hybrid_search(
        self,
        query: str,
        limit: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic and keyword (BM25) search.
        
        Args:
            query: Search query
            limit: Maximum number of results
            filter_metadata: Optional metadata filters
            
        Returns:
            Combined and ranked search results
        """
        # Perform semantic search (get more results for reranking)
        semantic_results = await self._semantic_search(query, limit * 2, filter_metadata)
        
        # Perform BM25 search
        bm25_results = self._bm25_search(query, limit * 2)
        
        # Combine results
        combined_results = {}
        
        # Add semantic results
        for result in semantic_results:
            doc_id = result["id"]
            combined_results[doc_id] = {
                **result,
                "semantic_score": result["score"],
                "bm25_score": 0.0
            }
        
        # Add/update with BM25 scores
        for result in bm25_results:
            doc_id = result["id"]
            if doc_id in combined_results:
                combined_results[doc_id]["bm25_score"] = result["bm25_score"]
            else:
                combined_results[doc_id] = {
                    **result,
                    "semantic_score": 0.0,
                    "bm25_score": result["bm25_score"]
                }
        
        # Normalize scores to 0-1 range
        if combined_results:
            max_semantic = max(r["semantic_score"] for r in combined_results.values()) or 1.0
            max_bm25 = max(r["bm25_score"] for r in combined_results.values()) or 1.0
            
            for result in combined_results.values():
                # Normalize
                norm_semantic = result["semantic_score"] / max_semantic if max_semantic > 0 else 0
                norm_bm25 = result["bm25_score"] / max_bm25 if max_bm25 > 0 else 0
                
                # Weighted combination
                alpha = settings.RAG_HYBRID_SEARCH_ALPHA
                result["score"] = alpha * norm_semantic + (1 - alpha) * norm_bm25
                result["metadata"]["score"] = result["score"]
        
        # Sort by combined score and return top results
        sorted_results = sorted(
            combined_results.values(),
            key=lambda x: x["score"],
            reverse=True
        )[:limit]
        
        # Filter by minimum relevance score
        min_score = settings.RAG_MIN_RELEVANCE_SCORE
        filtered_results = [r for r in sorted_results if r["score"] >= min_score]
        
        # Apply reranking if enabled
        if settings.RAG_RERANKING_ENABLED and self.reranker and filtered_results:
            filtered_results = self._rerank_results(query, filtered_results)
        
        # Apply deduplication if enabled
        if settings.RAG_DEDUPLICATION_ENABLED and filtered_results:
            filtered_results = self._deduplicate_results(
                filtered_results,
                similarity_threshold=settings.RAG_DEDUPLICATION_THRESHOLD
            )
        
        # Apply MMR for diversity if enabled
        if settings.RAG_MMR_ENABLED and filtered_results:
            filtered_results = self._apply_mmr(
                filtered_results,
                query,
                lambda_param=settings.RAG_MMR_LAMBDA,
                top_k=limit
            )
        
        logger.info(f"Found {len(filtered_results)} hybrid search results for query: {query[:50]}...")
        return filtered_results
    
    def _rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank search results using cross-encoder model.
        
        Args:
            query: Search query
            results: List of search results to rerank
            top_k: Number of top results to return (uses config if None)
            
        Returns:
            Reranked list of results
        """
        if not self.reranker or not results:
            return results
        
        try:
            top_k = top_k or settings.RAG_RERANKING_TOP_K
            
            # Prepare pairs for reranking
            pairs = [[query, result["content"]] for result in results]
            
            # Get reranking scores
            rerank_scores = self.reranker.predict(pairs)
            
            # Update results with reranking scores
            for i, result in enumerate(results):
                result["rerank_score"] = float(rerank_scores[i])
                # Combine original score with rerank score (weighted)
                original_score = result.get("score", 0.0)
                result["score"] = 0.7 * original_score + 0.3 * result["rerank_score"]
                result["metadata"]["score"] = result["score"]
            
            # Sort by reranking score
            reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
            
            # Return top k
            return reranked[:top_k]
            
        except Exception as e:
            error_msg = str(e)
            if "primitive descriptor" in error_msg.lower() or "matmul" in error_msg.lower():
                logger.warning(
                    f"Reranking failed due to CPU compatibility issue: {e}. "
                    "Disabling reranking for this session. Returning original results."
                )
                # Disable reranking to prevent repeated errors
                self.reranker = None
                settings.RAG_RERANKING_ENABLED = False
            else:
                logger.error(f"Error in reranking: {e}")
            return results
    
    def _apply_mmr(
        self,
        results: List[Dict[str, Any]],
        query: str,
        lambda_param: float = 0.5,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Apply Maximal Marginal Relevance (MMR) for diverse results.
        
        Args:
            results: List of search results
            query: Search query
            lambda_param: Balance between relevance (1.0) and diversity (0.0)
            top_k: Number of diverse results to return
            
        Returns:
            Diverse list of results
        """
        if not results or top_k <= 0:
            return []
        
        # Generate query embedding for similarity calculation
        query_embedding = self.embedding_model.encode(query)
        
        selected = []
        remaining = results.copy()
        
        # Select first result (highest relevance)
        if remaining:
            selected.append(remaining.pop(0))
        
        # Select remaining results using MMR
        while len(selected) < top_k and remaining:
            best_score = -float('inf')
            best_idx = 0
            
            for i, candidate in enumerate(remaining):
                # Relevance score
                relevance = candidate.get("score", 0.0)
                
                # Diversity: max similarity to already selected
                max_similarity = 0.0
                if selected:
                    candidate_embedding = self.embedding_model.encode(candidate.get("content", ""))
                    for selected_result in selected:
                        selected_embedding = self.embedding_model.encode(selected_result.get("content", ""))
                        # Cosine similarity
                        similarity = np.dot(candidate_embedding, selected_embedding) / (
                            np.linalg.norm(candidate_embedding) * np.linalg.norm(selected_embedding)
                        )
                        max_similarity = max(max_similarity, float(similarity))
                
                # MMR score: balance relevance and diversity
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            selected.append(remaining.pop(best_idx))
        
        return selected
    
    def _deduplicate_results(
        self,
        results: List[Dict[str, Any]],
        similarity_threshold: float = 0.95
    ) -> List[Dict[str, Any]]:
        """
        Remove near-duplicate results.
        
        Args:
            results: List of search results
            similarity_threshold: Similarity threshold for considering duplicates
            
        Returns:
            Deduplicated list of results
        """
        if not results:
            return []
        
        deduplicated = []
        seen_embeddings = []
        
        for result in results:
            content = result.get("content", result.get("page_content", ""))
            embedding = self.embedding_model.encode(content)
            
            # Check similarity with already selected results
            is_duplicate = False
            for seen_emb in seen_embeddings:
                similarity = np.dot(embedding, seen_emb) / (
                    np.linalg.norm(embedding) * np.linalg.norm(seen_emb)
                )
                if float(similarity) >= similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(result)
                seen_embeddings.append(embedding)
        
        if len(deduplicated) < len(results):
            logger.debug(f"Deduplicated {len(results)} results to {len(deduplicated)}")
        
        return deduplicated
    
    async def delete_document_chunks(self, document_id: UUID) -> bool:
        """
        Delete all chunks for a document from the vector store.
        
        Args:
            document_id: UUID of the document whose chunks should be deleted
            
        Returns:
            True if chunks were deleted, False if no chunks found
        """
        self._ensure_initialized()
        
        try:
            # Find all chunk IDs for this document
            results = self.collection.get(
                where={"document_id": str(document_id)},
                include=["metadatas"]
            )
            
            if results["ids"]:
                # Delete the chunks
                self.collection.delete(ids=results["ids"])
                
                # Rebuild BM25 index if hybrid search is enabled
                if settings.RAG_HYBRID_SEARCH_ENABLED:
                    self._build_bm25_index()
                
                logger.info(f"Deleted {len(results['ids'])} chunks for document {document_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting document chunks: {e}")
            return False
    
    async def update_document_chunks(
        self,
        document: Document,
        chunks: List[DocumentChunk]
    ) -> List[str]:
        """
        Update document chunks in the vector store.
        
        This method deletes existing chunks and adds new ones.
        
        Args:
            document: Document object
            chunks: List of document chunks to add
            
        Returns:
            List of embedding IDs for the added chunks
        """
        # Delete existing chunks first
        await self.delete_document_chunks(document.id)
        
        # Add new chunks
        return await self.add_document_chunks(document, chunks)
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store collection.
        
        Returns:
            Dictionary containing:
            - total_chunks: Total number of chunks in the collection
            - collection_name: Name of the ChromaDB collection
            - embedding_model: Currently active embedding model
            - available_models: List of available embedding models
        """
        self._ensure_initialized()
        
        try:
            count = self.collection.count()
            
            return {
                "total_chunks": count,
                "collection_name": settings.CHROMA_COLLECTION_NAME,
                "embedding_model": self.current_model_name or settings.EMBEDDING_MODEL,
                "available_models": settings.EMBEDDING_MODEL_OPTIONS,
                "hybrid_search_enabled": settings.RAG_HYBRID_SEARCH_ENABLED,
                "reranking_enabled": settings.RAG_RERANKING_ENABLED
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
    
    async def reset_collection(self):
        """Reset the entire collection (delete all data)."""
        self._ensure_initialized()
        
        try:
            self.client.delete_collection(settings.CHROMA_COLLECTION_NAME)
            self.collection = self.client.create_collection(
                name=settings.CHROMA_COLLECTION_NAME,
                metadata={"description": "Knowledge base document chunks"}
            )
            logger.warning("Vector store collection reset")
            
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            raise

