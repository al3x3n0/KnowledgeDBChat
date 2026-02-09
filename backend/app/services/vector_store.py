"""
Vector store service for semantic search.

Supports:
- ChromaDB (embedded, persistent)
- Qdrant (service-based)
"""

import os
import hashlib
import asyncio
import time
from typing import List, Optional, Dict, Any
from uuid import UUID
import numpy as np
"""
Ensure Chroma telemetry is disabled at runtime to avoid noisy warnings
from PostHog/telemetry inside the worker logs.
This must be set before importing chromadb so its telemetry picks it up.
"""
os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "false")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("POSTHOG_DISABLED", "true")

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from loguru import logger

from app.core.config import settings
from app.models.document import Document, DocumentChunk

try:
    from qdrant_client import QdrantClient  # type: ignore
    from qdrant_client.models import (  # type: ignore
        Distance,
        FieldCondition,
        Filter,
        FilterSelector,
        MatchAny,
        MatchValue,
        PointStruct,
        VectorParams,
    )
except Exception:  # pragma: no cover
    QdrantClient = None
    Distance = None
    FieldCondition = None
    Filter = None
    FilterSelector = None
    MatchAny = None
    MatchValue = None
    PointStruct = None
    VectorParams = None


class VectorStoreService:
    """Service for managing vector embeddings and similarity search."""
    
    def __init__(self):
        self.provider = str(getattr(settings, "VECTOR_STORE_PROVIDER", "chroma")).strip().lower()
        self.client = None
        self.collection = None
        self.qdrant_client = None
        self.embedding_model = None
        self.current_model_name = None
        self.reranker = None
        self.bm25_index = None
        self.bm25_documents = []
        self.bm25_doc_ids = []
        self.bm25_metadatas: list[dict] = []
        self._chroma_ready = False
        self._models_ready = False
        self._init_error: Optional[str] = None
        self._init_task: asyncio.Task | None = None
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self._embedding_dim: Optional[int] = None
    
    async def initialize(self, embedding_model: Optional[str] = None, background: bool = False):
        """
        Initialize the vector store backend and embedding/reranking models.
        
        Args:
            embedding_model: Optional embedding model name (uses config if None)
            background: If True, return after Chroma is ready and load models in background.
        """
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return

            try:
                if self.provider == "chroma":
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

                    self._chroma_ready = True
                    self._init_error = None

                    # Initialize BM25 index for hybrid search if enabled
                    if settings.RAG_HYBRID_SEARCH_ENABLED:
                        self._build_bm25_index()

                elif self.provider == "qdrant":
                    if QdrantClient is None:
                        raise RuntimeError("qdrant-client is not installed. Install backend dependencies with qdrant support.")

                    def _init_qdrant() -> Any:
                        return QdrantClient(
                            url=settings.QDRANT_URL,
                            api_key=settings.QDRANT_API_KEY,
                        )

                    self.qdrant_client = await asyncio.to_thread(_init_qdrant)
                    # Mark backend as "ready" (collection is ensured after embeddings load).
                    self._chroma_ready = True
                    self._init_error = None

                    # BM25 index can be built once a collection exists; skip here if empty/uninitialized.
                    if settings.RAG_HYBRID_SEARCH_ENABLED:
                        try:
                            self._build_bm25_index()
                        except Exception:
                            pass

                else:
                    raise RuntimeError(f"Unknown VECTOR_STORE_PROVIDER: {self.provider}")

                model_name = embedding_model or settings.EMBEDDING_MODEL
                self.current_model_name = model_name

                if background:
                    if self._init_task is None or self._init_task.done():
                        self._init_task = asyncio.create_task(self._load_models(model_name))
                    logger.info(f"Vector store backend '{self.provider}' ready; model load running in background")
                    return

                await self._load_models(model_name)
                logger.info("Vector store initialized successfully")

            except Exception as e:
                logger.error(f"Failed to initialize vector store: {e}")
                self._init_error = str(e)
                raise

    async def _load_models(self, model_name: str) -> None:
        """Load embedding model and optional reranker without blocking the event loop."""
        try:
            def _load_embedding():
                from sentence_transformers import SentenceTransformer  # type: ignore
                return SentenceTransformer(model_name)

            self.embedding_model = await asyncio.to_thread(_load_embedding)
            self.current_model_name = model_name
            logger.info(f"Loaded embedding model: {model_name}")
            try:
                self._embedding_dim = int(self.embedding_model.get_sentence_embedding_dimension())
            except Exception:
                self._embedding_dim = None

            if self.provider == "qdrant" and self.qdrant_client is not None:
                await self._ensure_qdrant_collection()

            if settings.RAG_RERANKING_ENABLED:
                try:
                    def _load_reranker():
                        from sentence_transformers import CrossEncoder  # type: ignore
                        return CrossEncoder(settings.RAG_RERANKING_MODEL)

                    self.reranker = await asyncio.to_thread(_load_reranker)
                    await asyncio.to_thread(self.reranker.predict, [["test query", "test document"]])
                    logger.info(f"Loaded reranking model: {settings.RAG_RERANKING_MODEL}")
                except Exception as e:
                    error_msg = str(e)
                    if "primitive descriptor" in error_msg.lower() or "matmul" in error_msg.lower():
                        logger.warning(
                            f"Reranking model has CPU compatibility issues ({error_msg}). "
                            "Reranking will be disabled."
                        )
                    else:
                        logger.warning(f"Failed to load reranking model: {e}. Reranking disabled.")
                    self.reranker = None
                    settings.RAG_RERANKING_ENABLED = False

            self._models_ready = True
            self._initialized = True
            self._init_error = None
        except Exception as e:
            self._init_error = str(e)
            logger.warning(f"Vector store model load failed: {e}")
            # Keep API up; keyword mode can still work if Chroma is ready.

    @property
    def is_chroma_ready(self) -> bool:
        # Historical name; now means "vector backend ready".
        if self.provider == "chroma":
            return bool(self._chroma_ready and self.collection is not None)
        if self.provider == "qdrant":
            return bool(self._chroma_ready and self.qdrant_client is not None)
        return False

    @property
    def is_embeddings_ready(self) -> bool:
        return bool(self._initialized and self.embedding_model is not None)

    @property
    def init_error(self) -> Optional[str]:
        return self._init_error
    
    def _ensure_initialized(self, require_embeddings: bool = True):
        """Ensure the vector store is initialized."""
        if require_embeddings:
            if not self.is_embeddings_ready:
                raise RuntimeError("Vector store embeddings not initialized. Call initialize() first.")
        else:
            if not self.is_chroma_ready:
                raise RuntimeError("Vector store not initialized. Call initialize() first.")

    async def _ensure_qdrant_collection(self) -> None:
        """Ensure the Qdrant collection exists and matches the embedding dimension."""
        if self.provider != "qdrant" or self.qdrant_client is None:
            return
        if self._embedding_dim is None:
            return
        if VectorParams is None or Distance is None:
            return

        collection_name = settings.QDRANT_COLLECTION_NAME
        vector_size = int(self._embedding_dim)

        def _ensure() -> None:
            try:
                self.qdrant_client.get_collection(collection_name=collection_name)
                return
            except Exception:
                pass

            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

        await asyncio.to_thread(_ensure)

    def _qdrant_filter_from_metadata(self, filter_metadata: Optional[Dict[str, Any]]) -> Any:
        """Convert a Chroma-style `where` filter into a Qdrant Filter (best-effort)."""
        if self.provider != "qdrant" or filter_metadata is None:
            return None
        if Filter is None or FieldCondition is None or MatchValue is None or MatchAny is None:
            return None

        def _as_scalar(v: Any) -> Any:
            if isinstance(v, UUID):
                return str(v)
            return v

        def _conds(obj: Dict[str, Any]) -> list[Any]:
            if "$and" in obj and isinstance(obj.get("$and"), list):
                out: list[Any] = []
                for part in obj["$and"]:
                    if isinstance(part, dict):
                        out.extend(_conds(part))
                return out

            out: list[Any] = []
            for key, val in obj.items():
                if key == "$and":
                    continue
                if isinstance(val, dict) and "$in" in val:
                    values = [str(_as_scalar(x)) for x in (val.get("$in") or [])]
                    out.append(FieldCondition(key=key, match=MatchAny(any=values)))
                else:
                    out.append(FieldCondition(key=key, match=MatchValue(value=_as_scalar(val))))
            return out

        conditions = _conds(filter_metadata)
        if not conditions:
            return None
        return Filter(must=conditions)

    def _qdrant_query_points(
        self,
        *,
        collection_name: str,
        query_embedding: list[float],
        qfilter: Any,
        limit: int,
    ) -> list[Any]:
        """
        Version-tolerant Qdrant search.

        Newer qdrant-client versions expose `query_points(...)` and remove `search(...)`.
        Older versions use `search(...)`.
        """
        if self.qdrant_client is None:
            raise RuntimeError("Qdrant client is not initialized")

        client = self.qdrant_client

        # New API (qdrant-client >= ~1.8/1.9+): query_points returns QueryResponse(points=[...])
        query_points = getattr(client, "query_points", None)
        if callable(query_points):
            resp = query_points(
                collection_name=collection_name,
                query=query_embedding,
                query_filter=qfilter,
                limit=int(limit),
                with_payload=True,
                with_vectors=False,
            )
            points = getattr(resp, "points", None)
            if isinstance(points, list):
                return points
            # Some clients may return the points list directly.
            if isinstance(resp, list):
                return resp
            return []

        # Legacy API: search returns list[ScoredPoint]
        search = getattr(client, "search", None)
        if callable(search):
            return search(
                collection_name=collection_name,
                query_vector=query_embedding,
                query_filter=qfilter,
                limit=int(limit),
                with_payload=True,
                with_vectors=False,
            )

        raise RuntimeError("Unsupported qdrant-client: missing query_points/search methods")
    
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
            def _load_embedding():
                from sentence_transformers import SentenceTransformer  # type: ignore
                return SentenceTransformer(new_model)

            new_embedding_model = await asyncio.to_thread(_load_embedding)
            
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
            ids: list[str] = []
            docs: list[str] = []
            metas: list[dict] = []

            if self.provider == "chroma":
                if self.collection is None:
                    logger.info("BM25 index skipped: Chroma collection not ready")
                    return

                all_docs = self.collection.get(include=["documents", "metadatas"])
                ids = list(all_docs.get("ids") or [])
                docs = list(all_docs.get("documents") or [])
                metas = list(all_docs.get("metadatas") or [])

            elif self.provider == "qdrant":
                if self.qdrant_client is None:
                    logger.info("BM25 index skipped: Qdrant client not ready")
                    return

                collection_name = settings.QDRANT_COLLECTION_NAME

                def _scroll_all() -> tuple[list[str], list[str], list[dict]]:
                    out_ids: list[str] = []
                    out_docs: list[str] = []
                    out_metas: list[dict] = []

                    offset = None
                    while True:
                        points, next_page = self.qdrant_client.scroll(
                            collection_name=collection_name,
                            limit=512,
                            offset=offset,
                            with_payload=True,
                            with_vectors=False,
                        )
                        for p in points:
                            payload = dict(p.payload or {})
                            text = payload.pop("_text", None) or payload.pop("text", None) or payload.pop("content", None) or ""
                            out_ids.append(str(p.id))
                            out_docs.append(str(text or ""))
                            out_metas.append(payload)
                        if next_page is None:
                            break
                        offset = next_page
                    return out_ids, out_docs, out_metas

                ids, docs, metas = _scroll_all()

            else:
                logger.warning(f"BM25 index skipped: unknown provider {self.provider}")
                return

            if not ids:
                logger.info("No documents in collection for BM25 index")
                return
            
            # Tokenize documents for BM25
            tokenized_docs = []
            for doc in docs:
                # Simple tokenization (split on whitespace and lowercase)
                tokens = doc.lower().split()
                tokenized_docs.append(tokens)
            
            # Build BM25 index
            self.bm25_index = BM25Okapi(tokenized_docs)
            self.bm25_documents = docs
            self.bm25_doc_ids = ids
            self.bm25_metadatas = metas
            
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
                    doc_id = self.bm25_doc_ids[idx]
                    metadata = {}
                    try:
                        if idx < len(self.bm25_metadatas):
                            metadata = self.bm25_metadatas[idx] or {}
                    except Exception:
                        metadata = {}

                    result = {
                        "id": doc_id,
                        "content": self.bm25_documents[idx],
                        "metadata": metadata,
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
                embedding = self.embedding_model.encode(chunk.content, show_progress_bar=False).tolist()
                
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
                if self.provider == "chroma":
                    # Add to ChromaDB
                    if self.collection is None:
                        raise RuntimeError("Chroma collection is not initialized")
                    self.collection.add(
                        ids=ids,
                        documents=documents,
                        embeddings=embeddings,
                        metadatas=metadatas
                    )
                elif self.provider == "qdrant":
                    if self.qdrant_client is None or PointStruct is None:
                        raise RuntimeError("Qdrant client is not initialized")
                    await self._ensure_qdrant_collection()

                    collection_name = settings.QDRANT_COLLECTION_NAME

                    # Store chunk text in payload for BM25 + citation snippets
                    points = []
                    for i, pid in enumerate(ids):
                        payload = dict(metadatas[i] or {})
                        payload["_text"] = documents[i]
                        points.append(PointStruct(id=pid, vector=embeddings[i], payload=payload))

                    await asyncio.to_thread(
                        self.qdrant_client.upsert,
                        collection_name=collection_name,
                        points=points,
                    )
                else:
                    raise RuntimeError(f"Unknown VECTOR_STORE_PROVIDER: {self.provider}")
                
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
        filter_metadata: Optional[Dict[str, Any]] = None,
        document_ids: Optional[List[str]] = None,
        apply_postprocessing: bool = True,
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
        # Merge document_ids filter (legacy callers) into Chroma where clause
        if document_ids:
            doc_filter: Dict[str, Any] = {"document_id": {"$in": [str(d) for d in document_ids]}}
            if filter_metadata:
                filter_metadata = {"$and": [filter_metadata, doc_filter]}
            else:
                filter_metadata = doc_filter

        # Auto-initialize Chroma quickly and kick off model load in background.
        if not self.is_chroma_ready:
            await self.initialize(background=True)

        self._ensure_initialized(require_embeddings=False)

        # If embeddings aren't ready yet, fall back to keyword search (BM25) to avoid blocking.
        if not self.is_embeddings_ready:
            if self.bm25_index is None:
                try:
                    self._build_bm25_index()
                except Exception:
                    pass
            bm25_results = self._bm25_search(query, limit=limit)
            if filter_metadata:
                filtered = []
                for r in bm25_results:
                    md = r.get("metadata") or {}
                    def _match(k: str, v: Any) -> bool:
                        if isinstance(v, dict) and "$in" in v:
                            return md.get(k) in (v.get("$in") or [])
                        return md.get(k) == v

                    # Support both {"field": value} and {"$and":[...]} shapes
                    if "$and" in filter_metadata and isinstance(filter_metadata.get("$and"), list):
                        ok = True
                        for part in filter_metadata["$and"]:
                            if not isinstance(part, dict):
                                continue
                            for k2, v2 in part.items():
                                if not _match(k2, v2):
                                    ok = False
                                    break
                            if not ok:
                                break
                        if ok:
                            filtered.append(r)
                    elif all(_match(k, v) for k, v in filter_metadata.items()):
                        filtered.append(r)
                bm25_results = filtered
            # Add a normalized "score" field for downstream compatibility
            max_score = max((r.get("bm25_score", 0.0) for r in bm25_results), default=0.0) or 1.0
            for r in bm25_results:
                r["score"] = float(r.get("bm25_score", 0.0)) / max_score
                r.setdefault("metadata", {})["score"] = r["score"]
                r.setdefault("metadata", {})["degraded"] = True
                r.setdefault("metadata", {})["degraded_reason"] = "embeddings_loading"
            return bm25_results
        
        try:
            # Use hybrid search if enabled
            if settings.RAG_HYBRID_SEARCH_ENABLED:
                self._ensure_initialized(require_embeddings=True)
                return await self._hybrid_search(query, limit, filter_metadata, apply_postprocessing=apply_postprocessing)
            else:
                self._ensure_initialized(require_embeddings=True)
                return await self._semantic_search(query, limit, filter_metadata, apply_postprocessing=apply_postprocessing)
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []

    def _pack_result_for_trace(self, result: Dict[str, Any]) -> Dict[str, Any]:
        md = (result.get("metadata") or {}) if isinstance(result, dict) else {}
        def _f(x: Any) -> Optional[float]:
            try:
                if x is None:
                    return None
                return float(x)
            except Exception:
                return None

        return {
            "id": str(result.get("id") or ""),
            "document_id": str(md.get("document_id") or ""),
            "chunk_id": str(md.get("chunk_id") or ""),
            "chunk_index": md.get("chunk_index"),
            "title": md.get("title"),
            "source": md.get("source") or md.get("source_type"),
            "score": _f(result.get("score") if isinstance(result, dict) else None),
            "semantic_score": _f(result.get("semantic_score") if isinstance(result, dict) else None),
            "bm25_score": _f(result.get("bm25_score") if isinstance(result, dict) else None),
            "rerank_score": _f(result.get("rerank_score") if isinstance(result, dict) else None),
        }

    async def search_with_trace(
        self,
        query: str,
        limit: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
        document_ids: Optional[List[str]] = None,
        apply_postprocessing: bool = True,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Like `search`, but also returns a trace payload describing retrieval stages.

        The trace is designed to be persisted in `retrieval_traces.trace`.
        """
        started_at = time.time()

        # Reuse the same filter merge behavior as `search`.
        if document_ids:
            doc_filter: Dict[str, Any] = {"document_id": {"$in": [str(d) for d in document_ids]}}
            if filter_metadata:
                filter_metadata = {"$and": [filter_metadata, doc_filter]}
            else:
                filter_metadata = doc_filter

        trace: Dict[str, Any] = {
            "started_at": started_at,
            "query": query,
            "limit": int(limit),
            "provider": getattr(self, "provider", None),
            "hybrid_enabled": bool(getattr(settings, "RAG_HYBRID_SEARCH_ENABLED", False)),
            "rerank_enabled": bool(getattr(settings, "RAG_RERANKING_ENABLED", False)),
            "mmr_enabled": bool(getattr(settings, "RAG_MMR_ENABLED", False)),
            "dedup_enabled": bool(getattr(settings, "RAG_DEDUPLICATION_ENABLED", False)),
            "alpha": float(getattr(settings, "RAG_HYBRID_SEARCH_ALPHA", 0.0)),
            "filter_metadata": filter_metadata,
        }

        # Auto-initialize quickly and kick off model load in background.
        if not self.is_chroma_ready:
            await self.initialize(background=True)

        self._ensure_initialized(require_embeddings=False)

        # If embeddings aren't ready yet, fall back to BM25.
        if not self.is_embeddings_ready:
            trace["mode"] = "bm25_degraded"
            if self.bm25_index is None:
                try:
                    self._build_bm25_index()
                except Exception:
                    pass
            bm25_results = self._bm25_search(query, limit=limit)
            trace["bm25_raw"] = [self._pack_result_for_trace(r) for r in bm25_results[:50]]

            # Apply filter_metadata in the same way as `search`.
            if filter_metadata:
                filtered = []
                for r in bm25_results:
                    md = r.get("metadata") or {}

                    def _match(k: str, v: Any) -> bool:
                        if isinstance(v, dict) and "$in" in v:
                            return md.get(k) in (v.get("$in") or [])
                        return md.get(k) == v

                    if "$and" in filter_metadata and isinstance(filter_metadata.get("$and"), list):
                        ok = True
                        for part in filter_metadata["$and"]:
                            if not isinstance(part, dict):
                                continue
                            for k2, v2 in part.items():
                                if not _match(k2, v2):
                                    ok = False
                                    break
                            if not ok:
                                break
                        if ok:
                            filtered.append(r)
                    elif all(_match(k, v) for k, v in filter_metadata.items()):
                        filtered.append(r)
                bm25_results = filtered

            max_score = max((r.get("bm25_score", 0.0) for r in bm25_results), default=0.0) or 1.0
            for r in bm25_results:
                r["score"] = float(r.get("bm25_score", 0.0)) / max_score
                r.setdefault("metadata", {})["score"] = r["score"]
                r.setdefault("metadata", {})["degraded"] = True
                r.setdefault("metadata", {})["degraded_reason"] = "embeddings_loading"

            trace["results_final"] = [self._pack_result_for_trace(r) for r in bm25_results[:limit]]
            trace["elapsed_ms"] = int((time.time() - started_at) * 1000)
            return bm25_results, trace

        try:
            if settings.RAG_HYBRID_SEARCH_ENABLED:
                self._ensure_initialized(require_embeddings=True)
                results, hybrid_trace = await self._hybrid_search_with_trace(
                    query,
                    limit,
                    filter_metadata,
                    apply_postprocessing=apply_postprocessing,
                )
                trace.update(hybrid_trace)
                trace["elapsed_ms"] = int((time.time() - started_at) * 1000)
                return results, trace

            self._ensure_initialized(require_embeddings=True)
            results, sem_trace = await self._semantic_search_with_trace(
                query,
                limit,
                filter_metadata,
                apply_postprocessing=apply_postprocessing,
            )
            trace.update(sem_trace)
            trace["elapsed_ms"] = int((time.time() - started_at) * 1000)
            return results, trace
        except Exception as e:
            logger.error(f"Error searching vector store (with trace): {e}")
            trace["error"] = str(e)
            trace["elapsed_ms"] = int((time.time() - started_at) * 1000)
            return [], trace
    
    async def _semantic_search(
        self,
        query: str,
        limit: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
        apply_postprocessing: bool = True,
    ) -> List[Dict[str, Any]]:
        """Perform semantic search only."""
        # Generate query embedding (SentenceTransformer is CPU-bound and synchronous)
        query_embedding = await asyncio.to_thread(
            lambda: self.embedding_model.encode(query, show_progress_bar=False).tolist()
        )

        search_results: list[dict[str, Any]] = []

        if self.provider == "chroma":
            if self.collection is None:
                raise RuntimeError("Chroma collection is not initialized")

            # Search in ChromaDB
            results = await asyncio.to_thread(
                self.collection.query,
                query_embeddings=[query_embedding],
                n_results=limit,
                where=filter_metadata,
                include=["documents", "metadatas", "distances"],
            )

            # Format results
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

        elif self.provider == "qdrant":
            if self.qdrant_client is None:
                raise RuntimeError("Qdrant client is not initialized")
            await self._ensure_qdrant_collection()

            qfilter = self._qdrant_filter_from_metadata(filter_metadata)
            collection_name = settings.QDRANT_COLLECTION_NAME

            hits = await asyncio.to_thread(
                self._qdrant_query_points,
                collection_name=collection_name,
                query_embedding=query_embedding,
                qfilter=qfilter,
                limit=limit,
            )

            for hit in hits:
                payload = dict(getattr(hit, "payload", None) or {})
                content = payload.pop("_text", None) or payload.pop("text", None) or payload.pop("content", None) or ""
                score = float(getattr(hit, "score", 0.0) or 0.0)
                # Clamp for consistency with Chroma's 0..1-ish scores
                if score < 0.0:
                    score = 0.0
                if score > 1.0:
                    score = 1.0

                payload["score"] = score
                search_results.append(
                    {
                        "id": str(getattr(hit, "id", "")),
                        "content": str(content),
                        "metadata": payload,
                        "score": score,
                        "page_content": str(content),
                    }
                )

        else:
            raise RuntimeError(f"Unknown VECTOR_STORE_PROVIDER: {self.provider}")
         
        logger.info(f"Found {len(search_results)} semantic results for query: {query[:50]}...")

        if apply_postprocessing:
            # Apply reranking if enabled
            if settings.RAG_RERANKING_ENABLED and self.reranker and search_results:
                search_results = await asyncio.to_thread(self._rerank_results, query, search_results)

            # Apply deduplication if enabled
            if settings.RAG_DEDUPLICATION_ENABLED and search_results:
                search_results = await asyncio.to_thread(
                    self._deduplicate_results,
                    search_results,
                    similarity_threshold=settings.RAG_DEDUPLICATION_THRESHOLD,
                )

            # Apply MMR for diversity if enabled
            if settings.RAG_MMR_ENABLED and search_results:
                search_results = await asyncio.to_thread(
                    self._apply_mmr,
                    search_results,
                    query,
                    lambda_param=settings.RAG_MMR_LAMBDA,
                    top_k=limit,
                )
        
        return search_results

    async def _semantic_search_with_trace(
        self,
        query: str,
        limit: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
        apply_postprocessing: bool = True,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        trace: Dict[str, Any] = {"mode": "semantic"}

        query_embedding = await asyncio.to_thread(
            lambda: self.embedding_model.encode(query, show_progress_bar=False).tolist()
        )

        search_results: list[dict[str, Any]] = []

        if self.provider == "chroma":
            if self.collection is None:
                raise RuntimeError("Chroma collection is not initialized")

            results = await asyncio.to_thread(
                self.collection.query,
                query_embeddings=[query_embedding],
                n_results=limit,
                where=filter_metadata,
                include=["documents", "metadatas", "distances"],
            )

            for i in range(len(results["ids"][0])):
                result = {
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": 1.0 - results["distances"][0][i],
                    "page_content": results["documents"][0][i],
                }
                result["metadata"]["score"] = result["score"]
                search_results.append(result)

        elif self.provider == "qdrant":
            if self.qdrant_client is None:
                raise RuntimeError("Qdrant client is not initialized")
            await self._ensure_qdrant_collection()

            qfilter = self._qdrant_filter_from_metadata(filter_metadata)
            collection_name = settings.QDRANT_COLLECTION_NAME

            hits = await asyncio.to_thread(
                self._qdrant_query_points,
                collection_name=collection_name,
                query_embedding=query_embedding,
                qfilter=qfilter,
                limit=limit,
            )

            for hit in hits:
                payload = dict(getattr(hit, "payload", None) or {})
                content = payload.pop("_text", None) or payload.pop("text", None) or payload.pop("content", None) or ""
                score = float(getattr(hit, "score", 0.0) or 0.0)
                if score < 0.0:
                    score = 0.0
                if score > 1.0:
                    score = 1.0

                payload["score"] = score
                search_results.append(
                    {
                        "id": str(getattr(hit, "id", "")),
                        "content": str(content),
                        "metadata": payload,
                        "score": score,
                        "page_content": str(content),
                    }
                )
        else:
            raise RuntimeError(f"Unknown VECTOR_STORE_PROVIDER: {self.provider}")

        trace["semantic_raw"] = [self._pack_result_for_trace(r) for r in search_results[:50]]

        results = search_results
        if apply_postprocessing:
            if settings.RAG_RERANKING_ENABLED and self.reranker and results:
                results = await asyncio.to_thread(self._rerank_results, query, results)
                trace["semantic_reranked"] = [self._pack_result_for_trace(r) for r in results[:50]]

            if settings.RAG_DEDUPLICATION_ENABLED and results:
                results = await asyncio.to_thread(
                    self._deduplicate_results,
                    results,
                    similarity_threshold=settings.RAG_DEDUPLICATION_THRESHOLD,
                )
                trace["semantic_deduped"] = [self._pack_result_for_trace(r) for r in results[:50]]

            if settings.RAG_MMR_ENABLED and results:
                results = await asyncio.to_thread(
                    self._apply_mmr,
                    results,
                    query,
                    lambda_param=settings.RAG_MMR_LAMBDA,
                    top_k=limit,
                )
                trace["semantic_mmr"] = [self._pack_result_for_trace(r) for r in results[:50]]

        trace["results_final"] = [self._pack_result_for_trace(r) for r in results[:limit]]
        return results, trace
    
    async def _hybrid_search(
        self,
        query: str,
        limit: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
        apply_postprocessing: bool = True,
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
        semantic_results = await self._semantic_search(
            query,
            limit * 2,
            filter_metadata,
            apply_postprocessing=False,
        )
        
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
        
        if apply_postprocessing:
            # Apply reranking if enabled
            if settings.RAG_RERANKING_ENABLED and self.reranker and filtered_results:
                filtered_results = await asyncio.to_thread(self._rerank_results, query, filtered_results)

            # Apply deduplication if enabled
            if settings.RAG_DEDUPLICATION_ENABLED and filtered_results:
                filtered_results = await asyncio.to_thread(
                    self._deduplicate_results,
                    filtered_results,
                    similarity_threshold=settings.RAG_DEDUPLICATION_THRESHOLD,
                )

            # Apply MMR for diversity if enabled
            if settings.RAG_MMR_ENABLED and filtered_results:
                filtered_results = await asyncio.to_thread(
                    self._apply_mmr,
                    filtered_results,
                    query,
                    lambda_param=settings.RAG_MMR_LAMBDA,
                    top_k=limit,
                )
        
        logger.info(f"Found {len(filtered_results)} hybrid search results for query: {query[:50]}...")
        return filtered_results

    async def _hybrid_search_with_trace(
        self,
        query: str,
        limit: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
        apply_postprocessing: bool = True,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        trace: Dict[str, Any] = {"mode": "hybrid"}

        semantic_results = await self._semantic_search(
            query,
            limit * 2,
            filter_metadata,
            apply_postprocessing=False,
        )
        bm25_results = self._bm25_search(query, limit * 2)

        trace["semantic_raw"] = [self._pack_result_for_trace(r) for r in semantic_results[:50]]
        trace["bm25_raw"] = [self._pack_result_for_trace(r) for r in bm25_results[:50]]

        combined_results: Dict[str, Dict[str, Any]] = {}

        for result in semantic_results:
            doc_id = result["id"]
            combined_results[doc_id] = {**result, "semantic_score": result["score"], "bm25_score": 0.0}

        for result in bm25_results:
            doc_id = result["id"]
            if doc_id in combined_results:
                combined_results[doc_id]["bm25_score"] = result["bm25_score"]
            else:
                combined_results[doc_id] = {**result, "semantic_score": 0.0, "bm25_score": result["bm25_score"]}

        if combined_results:
            max_semantic = max(r["semantic_score"] for r in combined_results.values()) or 1.0
            max_bm25 = max(r["bm25_score"] for r in combined_results.values()) or 1.0

            for result in combined_results.values():
                norm_semantic = result["semantic_score"] / max_semantic if max_semantic > 0 else 0
                norm_bm25 = result["bm25_score"] / max_bm25 if max_bm25 > 0 else 0
                alpha = settings.RAG_HYBRID_SEARCH_ALPHA
                result["score"] = alpha * norm_semantic + (1 - alpha) * norm_bm25
                result["metadata"]["score"] = result["score"]

        sorted_results = sorted(combined_results.values(), key=lambda x: x["score"], reverse=True)[:limit]
        trace["hybrid_sorted"] = [self._pack_result_for_trace(r) for r in sorted_results[:50]]

        min_score = settings.RAG_MIN_RELEVANCE_SCORE
        filtered_results = [r for r in sorted_results if r["score"] >= min_score]
        trace["hybrid_filtered"] = [self._pack_result_for_trace(r) for r in filtered_results[:50]]

        results = filtered_results
        if apply_postprocessing:
            if settings.RAG_RERANKING_ENABLED and self.reranker and results:
                results = await asyncio.to_thread(self._rerank_results, query, results)
                trace["hybrid_reranked"] = [self._pack_result_for_trace(r) for r in results[:50]]

            if settings.RAG_DEDUPLICATION_ENABLED and results:
                results = await asyncio.to_thread(
                    self._deduplicate_results,
                    results,
                    similarity_threshold=settings.RAG_DEDUPLICATION_THRESHOLD,
                )
                trace["hybrid_deduped"] = [self._pack_result_for_trace(r) for r in results[:50]]

            if settings.RAG_MMR_ENABLED and results:
                results = await asyncio.to_thread(
                    self._apply_mmr,
                    results,
                    query,
                    lambda_param=settings.RAG_MMR_LAMBDA,
                    top_k=limit,
                )
                trace["hybrid_mmr"] = [self._pack_result_for_trace(r) for r in results[:50]]

        trace["results_final"] = [self._pack_result_for_trace(r) for r in results[:limit]]
        return results, trace
    
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
        query_embedding = self.embedding_model.encode(query, show_progress_bar=False)
        
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
                    candidate_embedding = self.embedding_model.encode(candidate.get("content", ""), show_progress_bar=False)
                    for selected_result in selected:
                        selected_embedding = self.embedding_model.encode(selected_result.get("content", ""), show_progress_bar=False)
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
            embedding = self.embedding_model.encode(content, show_progress_bar=False)
            
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
            if self.provider == "chroma":
                if self.collection is None:
                    raise RuntimeError("Chroma collection is not initialized")

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

            if self.provider == "qdrant":
                if self.qdrant_client is None or FilterSelector is None:
                    raise RuntimeError("Qdrant client is not initialized")

                await self._ensure_qdrant_collection()

                collection_name = settings.QDRANT_COLLECTION_NAME
                doc_filter = Filter(
                    must=[FieldCondition(key="document_id", match=MatchValue(value=str(document_id)))]
                )

                cnt = await asyncio.to_thread(
                    self.qdrant_client.count,
                    collection_name=collection_name,
                    count_filter=doc_filter,
                    exact=True,
                )
                total = int(getattr(cnt, "count", 0) or 0)
                if total <= 0:
                    return False

                await asyncio.to_thread(
                    self.qdrant_client.delete,
                    collection_name=collection_name,
                    points_selector=FilterSelector(filter=doc_filter),
                )

                if settings.RAG_HYBRID_SEARCH_ENABLED:
                    self._build_bm25_index()

                logger.info(f"Deleted {total} chunks for document {document_id}")
                return True

            raise RuntimeError(f"Unknown VECTOR_STORE_PROVIDER: {self.provider}")
             
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
            if self.provider == "chroma":
                if self.collection is None:
                    raise RuntimeError("Chroma collection is not initialized")
                count = self.collection.count()
                collection_name = settings.CHROMA_COLLECTION_NAME
            elif self.provider == "qdrant":
                if self.qdrant_client is None:
                    raise RuntimeError("Qdrant client is not initialized")
                await self._ensure_qdrant_collection()
                collection_name = settings.QDRANT_COLLECTION_NAME
                cnt = await asyncio.to_thread(
                    self.qdrant_client.count,
                    collection_name=collection_name,
                    exact=True,
                )
                count = int(getattr(cnt, "count", 0) or 0)
            else:
                raise RuntimeError(f"Unknown VECTOR_STORE_PROVIDER: {self.provider}")

            return {
                "provider": self.provider,
                "total_chunks": count,
                "collection_name": collection_name,
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
            if self.provider == "chroma":
                if self.client is None:
                    raise RuntimeError("Chroma client is not initialized")
                self.client.delete_collection(settings.CHROMA_COLLECTION_NAME)
                self.collection = self.client.create_collection(
                    name=settings.CHROMA_COLLECTION_NAME,
                    metadata={"description": "Knowledge base document chunks"}
                )
            elif self.provider == "qdrant":
                if self.qdrant_client is None:
                    raise RuntimeError("Qdrant client is not initialized")
                collection_name = settings.QDRANT_COLLECTION_NAME
                await asyncio.to_thread(self.qdrant_client.delete_collection, collection_name=collection_name)
                await self._ensure_qdrant_collection()
            else:
                raise RuntimeError(f"Unknown VECTOR_STORE_PROVIDER: {self.provider}")

            logger.warning("Vector store collection reset")
            
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            raise


# Shared singleton to avoid loading embedding models multiple times per process.
vector_store_service = VectorStoreService()


# Backwards-compatible alias (older code imports/instantiates `VectorStore`).
class VectorStore(VectorStoreService):
    pass
