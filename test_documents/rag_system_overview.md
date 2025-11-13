# RAG System Overview

## What is RAG?

Retrieval-Augmented Generation (RAG) is a technique that enhances Large Language Models (LLMs) by providing them with relevant context from a knowledge base before generating responses.

## System Architecture

### Components

1. **Document Ingestion**
   - Upload documents in various formats (PDF, Markdown, Text, etc.)
   - Extract text content
   - Split into semantic chunks
   - Generate embeddings

2. **Vector Store**
   - Store document embeddings in ChromaDB
   - Enable semantic similarity search
   - Support hybrid search (semantic + BM25)

3. **Query Processing**
   - Normalize user queries
   - Expand queries with synonyms
   - Generate multiple query variations
   - Classify query intent

4. **Retrieval**
   - Semantic search using embeddings
   - Keyword search using BM25
   - Hybrid search combining both methods
   - Reranking with cross-encoder models

5. **Context Management**
   - Filter retrieved chunks by relevance
   - Compress context to fit token limits
   - Prioritize important information
   - Remove duplicates

6. **Generation**
   - Build prompt with retrieved context
   - Send to LLM (Ollama)
   - Stream response to user
   - Store conversation history

## Key Features

### Hybrid Search
Combines semantic similarity search with keyword-based BM25 search for better retrieval accuracy.

### Query Expansion
Automatically expands user queries with related terms and synonyms to improve search results.

### Reranking
Uses cross-encoder models to rerank retrieved documents by relevance to the query.

### Context Compression
Intelligently compresses retrieved context to fit within token limits while preserving important information.

### Multi-Query Generation
Generates multiple query variations to retrieve more comprehensive results.

## Workflow

1. User submits a query
2. System processes and expands the query
3. Retrieves relevant document chunks using hybrid search
4. Reranks results by relevance
5. Filters and compresses context
6. Builds prompt with context and query
7. Sends to LLM for generation
8. Returns response to user
9. Stores conversation in memory

## Performance Optimizations

- Caching of embeddings
- Batch processing of documents
- Async document processing
- Connection pooling
- Index optimization

## Future Enhancements

- Multi-modal support (images, tables)
- Advanced query understanding
- Personalized retrieval
- Feedback loop for improvement
- Multi-language support


