# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KnowledgeDBChat is a full-stack enterprise knowledge management system with an LLM-powered chat interface. It aggregates data from multiple sources (GitLab, GitHub, Confluence, Web, ArXiv, file uploads) and provides semantic search with RAG (Retrieval-Augmented Generation) capabilities.

## Common Commands

### Docker Development (Recommended)
```bash
make setup              # Initial setup (creates directories and .env files)
make build              # Build Docker containers
make start              # Start all services
make stop               # Stop all services
make logs-backend       # View backend logs
make logs-celery        # View Celery worker logs
make test-backend       # Run backend tests
make test-frontend      # Run frontend tests
make db-migrate         # Run database migrations
make shell-backend      # Access backend container shell
make health             # Check health of all services
```

### Manual Development
```bash
# Backend
cd backend && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend
cd frontend && npm install && npm start

# Celery worker
cd backend && celery -A app.core.celery worker --loglevel=info
```

### Testing
```bash
# Backend (with pytest)
cd backend && pytest                      # Run all tests
pytest tests/test_chat.py -v              # Single test file
pytest --cov=app tests/                   # With coverage

# Frontend
cd frontend && npm test
npm run test:ci                           # CI mode with coverage
```

### Database Migrations (Alembic)
```bash
cd backend
alembic revision --autogenerate -m "description"   # Create migration
alembic upgrade head                                # Apply migrations
```

## Architecture

### Tech Stack
- **Backend**: FastAPI + SQLAlchemy 2.0 (async) + PostgreSQL + Redis + Celery
- **Frontend**: React 18 + TypeScript + Tailwind CSS + React Router
- **Vector Store**: Qdrant (default) or ChromaDB, with sentence-transformers embeddings
- **LLM**: Ollama (local) or DeepSeek (external API)
- **Storage**: MinIO (S3-compatible object storage)
- **Transcription**: OpenAI Whisper for video/audio

### Backend Structure (`backend/app/`)
- `api/endpoints/` - FastAPI route handlers (auth, chat, documents, upload, admin, etc.)
- `api/routes.py` - Main router that assembles all endpoints
- `core/` - Configuration (`config.py`), database setup, Celery, middleware, rate limiting
- `models/` - SQLAlchemy models (Document, User, ChatSession, Persona, etc.)
- `schemas/` - Pydantic request/response models
- `services/` - Business logic (chat_service, document_service, vector_store, llm_service, etc.)
- `services/connectors/` - Data source integrations (GitLab, GitHub, Confluence, Web, ArXiv)
- `tasks/` - Celery background tasks (ingestion, sync, transcription, summarization)
- `alembic/versions/` - Database migrations

### Frontend Structure (`frontend/src/`)
- `pages/` - Page components (ChatPage, DocumentsPage, AdminPage, etc.)
- `components/` - Reusable UI components
- `services/api.ts` - Centralized Axios API client
- `contexts/AuthContext.tsx` - Authentication state management
- `types/index.ts` - TypeScript interfaces

### Key Patterns
- **Authentication**: JWT tokens with role-based access (admin, user, viewer)
- **RAG Pipeline**: Query → Vector search (Qdrant/Chroma) + BM25 → Reranking → LLM response
- **Background Jobs**: Celery with Redis broker; scheduled tasks via Celery Beat
- **WebSocket**: Real-time chat with typing indicators
- **Data Connectors**: Abstract `BaseConnector` class; implement `sync()` for new sources

## API Versioning

All API endpoints are prefixed with `/api/v1/`. Key endpoint groups:
- `/auth` - Authentication (login, register, token refresh)
- `/chat` - Chat sessions and messages
- `/documents` - Document CRUD and upload
- `/admin` - System admin operations and data source management
- `/kg` - Knowledge graph operations
- `/personas` - AI persona management

## Environment Configuration

Backend configuration is in `backend/.env` (copy from `env.example`). Key variables:
- `DATABASE_URL`, `REDIS_URL` - Database connections
- `LLM_PROVIDER` - `ollama` or `deepseek`
- `OLLAMA_BASE_URL`, `DEFAULT_MODEL` - Local LLM settings
- `DEEPSEEK_API_KEY` - External LLM (if using DeepSeek)
- `RAG_*` - RAG pipeline settings (hybrid search, reranking, chunking)
- `MINIO_*` - Object storage configuration

## Docker Services

Main services in `docker-compose.yml`:
- `postgres` (5432), `redis` (6379), `ollama` (11434), `minio` (9000/9001)
- `backend` (8000), `frontend` via `nginx` (3000), `celery` worker
- `video-streamer` - Go microservice for video streaming (8080)

Access points:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- MinIO Console: http://localhost:9001
