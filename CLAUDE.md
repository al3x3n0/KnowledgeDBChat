# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KnowledgeDBChat is a full-stack autonomous R&D and knowledge-management platform. At its core it aggregates data from multiple sources (GitLab, GitHub, Confluence, Web, ArXiv, file uploads) and provides semantic search with RAG chat. On top of that core it has grown a large set of subsystems: autonomous agent jobs with a control plane, a multi-agent coding swarm that proposes code patches and PRs, a research suite (papers, notes, portfolios, inbox, monitoring), document generation (LaTeX, DOCX, PPTX, presentations), model fine-tuning with a model registry, scientific validation in Docker sandboxes, and tool governance with policies and audit logs.

Scale reference: ~57 API endpoint groups, 50+ SQLAlchemy models, 130+ services, 27 Celery task modules, 70+ Alembic migrations, 35 frontend pages, ~107 backend test modules.

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
make test-backend-coverage  # Backend tests with CI-style 70% coverage gate
make test-frontend      # Run frontend tests
make typecheck-frontend # Frontend TypeScript typecheck
make db-migrate         # Run database migrations
make db-shell           # PostgreSQL shell
make shell-backend      # Access backend container shell
make health             # Check health of all services
make fmt                # Format backend code (black + isort)
make lint               # Lint backend code (flake8)
make doctor             # Validate env + health checks
make download-models    # Download Ollama + embedding + reranking models
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
pytest tests/test_chat.py::test_name -v   # Single test
pytest --cov=app tests/                   # With coverage
pytest -m unit                            # Markers: unit, integration, slow

# Frontend
cd frontend && npm test
npm run test:ci                           # CI mode with coverage (60% thresholds)
```

### Database Migrations (Alembic)
```bash
cd backend
alembic revision --autogenerate -m "description"   # Create migration
alembic upgrade head                                # Apply migrations
```
Migrations use sequential numeric prefixes (`0001_...` ... `0072_...`); follow that naming.

## Architecture

### Tech Stack
- **Backend**: FastAPI + SQLAlchemy 2.0 (async) + PostgreSQL + Redis + Celery
- **Frontend**: React 18 + TypeScript (CRA) + Tailwind CSS + React Router 6; Zustand (workflow editor state), React Query (server cache), ReactFlow + Dagre (graph/workflow canvases), Tiptap (rich text), react-hook-form, react-hot-toast
- **Vector Store**: Qdrant (default, runs as a service) or ChromaDB (embedded), sentence-transformers embeddings
- **LLM**: Ollama (local), DeepSeek, OpenAI, Anthropic, Qwen (DashScope), or Kimi (Moonshot), selected by `LLM_PROVIDER`; per-request routing via `services/llm_routing.py` (fast/balanced/deep tiers). Native tool calling and schema-constrained output live in `services/llm_providers/` (used by `LLMService.generate_structured()`); `generate_response()` is the legacy prompted-text path
- **Storage**: MinIO (S3-compatible object storage)
- **Transcription**: OpenAI Whisper (optional speaker diarization via pyannote)
- **Diagrams**: Kroki service (Mermaid/PlantUML) with external fallback

### Backend Structure (`backend/app/`)
- `api/endpoints/` - ~57 FastAPI route modules; `api/routes.py` assembles them all
- `core/` - Configuration (`config.py`, 150+ settings), database setup, Celery, middleware, rate limiting, feature flags
- `models/` - 50+ SQLAlchemy models
- `schemas/` - Pydantic request/response models
- `services/` - Business logic layer (130+ modules; see Subsystems below)
- `services/connectors/` - Data source integrations (GitLab, GitHub, Confluence, Web, ArXiv)
- `services/trainers/` - Fine-tuning backends (local, simulated)
- `services/transcription/` - Whisper orchestration
- `tasks/` - 27 Celery task modules (ingestion, sync, transcription, summarization, agent jobs, training, LaTeX, paper enrichment/extraction/KG, synthesis, repo reports, workflows, ...)
- `mcp/` - MCP server exposing tools (search, documents, chat, generation, web_scrape, docker_execute) to external agents via API keys
- `alembic/versions/` - Database migrations

### Frontend Structure (`frontend/src/`)
- `pages/` - 35 page components (ChatPage, DocumentsPage, AdminPage, AutonomousAgentsPage, AgentControlPlanePage, AgentBuilderPage, LatexStudioPage, PapersPage, ResearchNotesPage, ReadingListsPage, PatchPRsPage, RepoReportsPage, PresentationsPage, SynthesisPage, WorkflowsPage/WorkflowEditorPage, AIHubPage, KGAdminPage, GlobalGraphPage, ToolsPage, UsagePage, RoutingExperimentsPage, etc.)
- `components/` - Reusable UI, grouped by domain (`agent/`, `common/`, `docx/`, `kg/`, `notifications/`, `presentations/`, `search/`, `workflows/`)
- `services/api.ts` - Single `ApiClient` class (~4700 lines, 460+ methods) wrapping Axios with `/api/v1` base, token interceptor, and toast-based error handling — add new endpoints here
- `contexts/` - `AuthContext.tsx`, `NotificationContext.tsx`
- `hooks/` - `useWebSocket`, `useKeyboardShortcuts`, `useElementSize`
- `types/index.ts` - TypeScript interfaces (very large; mirror backend schemas here)
- Tailwind uses an inverted terminal/dark palette (gray-50 = dark, gray-900 = light) — check `tailwind.config.js` before assuming standard shades

## Major Subsystems

Beyond RAG chat, these are the main functional areas. When touching one, its endpoint module, model(s), and service(s) usually share a name prefix.

- **Autonomous agents & control plane** — observe→think→act→evaluate loop in `services/autonomous_agent_executor.py` (the largest service), decomposed into runtime services (`agent_observation_service`, `agent_thinking_service`, `agent_action_service`, `agent_progress_evaluation_service`, `agent_checkpoint_service`, `agent_runtime_*`). Job chaining/swarm orchestration in `agent_chain_orchestration_service.py`; autonomy policies and decision events (`models/autonomy_decision_event.py`, `agent_tool_prior.py`) surface in the control-plane UI. Specialized deterministic runners: coding, research, experiment, LaTeX, scientific validation (`agent_*_runner_service.py`, registered in `agent_deterministic_runner_registry.py`).
- **Coding swarm** — backlog items, swarm profiles, code patch proposals, and PRs (`coding_backlog`, `coding_swarm_profiles`, `code_patches`, `patch_prs`); git operations via `git_service.py`, workspaces via `coding_workspace_manager.py`, symbol indexing via `repo_symbol_index_service.py`. KB patch application is gated by `AGENT_KB_PATCH_APPLY_ENABLED`.
- **Research suite** — papers (arXiv ingestion, enrichment, extraction, KG building: `paper_*_service.py`), research notes, portfolios, inbox with follow-up automation, monitor profiles, domain research profiles, reading lists. The research runner (`agent_research_runner_service.py`) orchestrates end-to-end workflows.
- **Document generation** — LaTeX projects with server-side compilation (dedicated `celery_latex` worker, disabled/admin-only by default via `LATEX_COMPILER_*`), DOCX editor, PPTX/presentation generation, PDF export, artifact drafts staged for review before publishing.
- **Training / AI Hub** — datasets, fine-tuning jobs (`services/trainers/`, backends: local/modal/runpod), model registry, eval templates and benchmark harness. Gated by `TRAINING_ENABLED`.
- **Experiments & scientific validation** — experiment plans/runs, Docker-sandboxed validation with image allowlists and resource caps (`SCIENTIFIC_VALIDATION_*`, `UNSAFE_CODE_EXEC_*` settings).
- **Tool governance** — `tool_registry.py` + `tool_policy_engine.py` + `models/tool_audit.py`; per-user tool policies, approval gates for dangerous tools (`AGENT_REQUIRE_TOOL_APPROVAL`, `AGENT_DANGEROUS_TOOLS`), full execution audit log, user-defined custom tools (optionally Docker-executed). Tool dispatch lives in `agent_tool_dispatch.py` / `agent_tools.py`.
- **Workflows** — visual workflow builder (ReactFlow frontend, Zustand store), `workflow_engine.py` execution, workflow→synthesis conversion, LangGraph-based issue/PR graphs (`langgraph_issue_pr_service.py`).
- **Synthesis & reporting** — multi-document synthesis jobs, repo analysis reports and presentations, retrieval traces for RAG observability.

## Key Architectural Patterns

### Database Session Management
- Async sessions via `get_db()` dependency in `core/database.py`
- **Semaphore-based concurrency limiting** prevents pool exhaustion; returns HTTP 503 with Retry-After when saturated
- Celery tasks create **fresh async engines per invocation** (workers fork, old event loops are incompatible); Celery-specific pool settings (`CELERY_DB_USE_NULLPOOL`, etc.)
- Config: `DB_POOL_SIZE`, `DB_MAX_OVERFLOW`, `DB_SESSION_CONCURRENCY_LIMIT` in `core/config.py`

### Multi-Tenancy / User Scoping
- All resources filtered by `user_id` foreign key in database queries
- Auth chain: `get_current_user` (validates JWT) → `get_current_active_user` (checks is_active)
- Admin users have broader access; non-admins cannot access other users' resources
- Optional LDAP/AD auth with group-based role mapping (`LDAP_*` settings)
- MCP API keys are tied to users; tool policies evaluated per user context

### Service Layer Pattern
- Services are singleton-style classes with composition (not inheritance)
- **User settings passed per-call** (not stored in service state) for multi-tenant isolation
- Late async initialization with double-checked locking (`_ensure_vector_store_initialized()`)
- Key dependency chain: `AgentService` → `LLMService`, `DocumentService`, `VectorStoreService`, `MemoryService`
- The agent runtime is intentionally split into many small `agent_*` services around `autonomous_agent_executor.py` — prefer extending the relevant sub-service over growing the executor

### Feature Flags (`core/feature_flags.py`)
- Two-tier resolution: Redis cache → Settings fallback
- Boolean flags (e.g., `knowledge_graph_enabled`) and string config flags (e.g., `llm_default_model`)
- Runtime-updatable without restart via Redis

### Agent Job System
- Goal-driven autonomous execution with observe→think→act→evaluate loop
- Optional native tool-calling loop in the think phase (`services/agent_native_tool_loop.py`, gated by `AGENT_NATIVE_TOOL_LOOP_ENABLED` or job config `native_tool_loop`): the model calls read-safe tools natively via `generate_structured` before emitting its decision; approval-gated/dangerous tools are deferred back to the act phase
- LLM call snapshots for replay debugging (opt-in via `LLM_CALL_SNAPSHOT_ENABLED`): `LLMService` records full prompts/responses to `llm_call_snapshots`, correlated by job/iteration/phase via the `snapshot_context` kwarg; read via `GET /api/v1/llm-snapshots?job_id=...` (owner/admin). New LLM call sites in the agent loop should pass `snapshot_context`
- Automatic context compaction (`services/agent_context_compaction.py`, on by default via `AGENT_AUTO_COMPACTION_ENABLED`): when serialized iteration state crosses a size threshold, older actions are summarized into `state["compressed_history"]` (same contract as the agent-invoked `compress_history` tool) at the start of the think phase; falls back to a deterministic digest if the summary LLM call fails
- The thinking prompt is split for prompt caching: `_build_thinking_prompt_stable` (per-job, byte-stable — keep it that way; it keys provider prompt caches) is the system prompt, `_build_thinking_prompt_volatile` (plan/critic/focus/history) rides in the user message. New per-iteration context belongs in the volatile part. Anthropic requests add `cache_control` breakpoints automatically (`ANTHROPIC_PROMPT_CACHE_ENABLED`)
- Job chaining: parent/child jobs with configurable trigger conditions (`on_complete`, `on_fail`, `on_findings`)
- Tool fallback policies per job type; per-job overrides via `config.tool_fallback_map`
- Memory integration: jobs extract and store memories for future retrieval (`agent_job_memory_service.py`)
- Execution tracked via `execution_log` JSON array of timestamped iterations; checkpoints allow resume
- Background execution via Celery tasks with progress callbacks; operator interventions for human approval

### Knowledge Graph
- Three-entity model: `Entity` → `EntityMention` → `Relationship` (in `models/knowledge_graph.py`)
- All KG data links back to source document + chunk for provenance
- Relationships have confidence scores and inferred flags
- Cascading deletes: removing a document removes all associated KG data
- Optional LLM-based extraction (`KG_LLM_EXTRACTION_ENABLED`); KG context can be injected into RAG (`RAG_KG_CONTEXT_ENABLED`)

### MCP Server (`mcp/server.py`)
- Stateless tool wrappers with authentication (API key via header or query param) and policy enforcement
- Tools filtered by API key capabilities; high-risk tools require approval
- Usage logging for rate limiting and quota tracking

### RAG Pipeline
- Query → hybrid search (vector + BM25) → reranking (cross-encoder) → MMR / deduplication / query expansion → optional KG context → LLM response
- Retrieval traces recorded for observability (`models/retrieval_trace.py`, `/retrieval-traces` endpoints)
- Configured via `RAG_*` environment variables

## API Versioning

All API endpoints are prefixed with `/api/v1/`. Endpoint groups by domain (see `api/routes.py` for the authoritative list):
- **Core**: `/auth`, `/users`, `/chat`, `/documents`, `/upload`, `/memory`, `/admin`, `/system`, `/kg`, `/api-keys`, `/personas`, `/notifications`, `/searches`, `/dashboard`
- **Agents**: `/agent`, `/agent-jobs`, `/agent-control-plane`, `/workflows`, `/templates`
- **Coding**: `/git`, `/code-patches`, `/patch-prs`, `/coding-backlog`, `/coding-swarm-profiles`, `/langgraph`, `/repo-reports`
- **Research**: `/research`, `/research/papers`, `/research/inbox`, `/research/monitor-profiles`, `/research-portfolios`, `/research-notes`, `/reading-lists`, `/domain-research-profiles`, `/scientific-sandbox-profiles`, `/experiments`, `/synthesis`
- **Generation**: `/presentations`, `/latex`, `/docx-editor`, `/artifact-drafts`, `/export`, `/content-generation`
- **Training**: `/training/datasets`, `/training/jobs`, `/training/models`, `/training/evals`
- **Governance**: `/tools`, `/admin-tools`, `/audit`, `/secrets`, `/user-tools`, `/mcp-config`, `/usage`, `/analytics`, `/retrieval-traces`

### Golden-Task Agent Regression Suite
- `tests/test_golden_agent_tasks.py` runs the REAL `_run_autonomous_loop` end-to-end (in-memory SQLite) with only two seams scripted: `ScriptedLLM` (serves queued decision JSON only to decision-shaped prompts) and `ScriptedActionService` (canned tool results via the `act()` seam)
- Covers: goal completion, iteration-budget stop, malformed-LLM-output recovery, goal-contract false-completion blocking, tool-failure resilience
- Run it after any change to the executor, thinking service, decision parser, or prompt builders; assertions are behavioral (subsequence/count-based) because the loop interleaves its own support actions

## Testing Patterns

- Backend tests use **in-memory SQLite** with `aiosqlite` (configured in `tests/conftest.py`)
- Heavy optional dependencies (pptx, sentence_transformers, bs4, croniter, mammoth, jsonpath_ng) are stubbed in conftest — don't import them at module top-level in code paths tests touch without checking the stubs
- FastAPI dependency overrides replace `get_db` with test session
- User fixtures: `test_user` (regular) and `admin_user` with real password hashing; `auth_headers` / `admin_headers` via live token creation
- Async tests use `pytest-asyncio` (auto mode); markers: `unit`, `integration`, `slow`
- CI-style coverage gate: 70% backend (`make test-backend-coverage`), 60% frontend (`npm run test:ci`)

## Commit Style

Follow Conventional Commits as seen in history: `fix(ui): ...`, `feat(admin): ...`, `style(ui): ...`

## Environment Configuration

Backend configuration is in `backend/.env` (copy from `env.example`). `core/config.py` has 150+ settings; major groups:
- `DATABASE_URL`, `REDIS_URL`, `DB_*` - Database connections and pooling
- `LLM_PROVIDER` - `ollama`, `deepseek`, `openai`, `anthropic`, `qwen`, or `kimi`; `OLLAMA_BASE_URL`, `DEFAULT_MODEL`, `DEEPSEEK_*`, `OPENAI_*`, `ANTHROPIC_*`, `QWEN_*`, `KIMI_*`
- `RAG_*` - RAG pipeline (hybrid search, reranking, MMR, dedup, KG context, chunking)
- `VECTOR_STORE_PROVIDER` + `QDRANT_*` / `CHROMA_*`
- `MINIO_*` - Object storage
- `WHISPER_*`, `TRANSCRIPTION_*` - Transcription and diarization
- `LDAP_*` - Optional LDAP/AD authentication
- `LATEX_COMPILER_*` - LaTeX compilation (disabled/admin-only by default)
- `UNSAFE_CODE_EXEC_*`, `SCIENTIFIC_VALIDATION_*` - Sandboxed code execution limits (subprocess or Docker)
- `TRAINING_*`, `AI_HUB_*`, `DATASET_MAX_*` - Fine-tuning and evals
- `AGENT_REQUIRE_TOOL_APPROVAL`, `AGENT_DANGEROUS_TOOLS`, `AGENT_KB_PATCH_APPLY_ENABLED` - Agent governance
- `SECRETS_ENCRYPTION_KEY` - Fernet key for the encrypted secrets store
- `KROKI_URL` - Diagram rendering; `GITLAB_*`, `CONFLUENCE_*` - Data sources

Security-sensitive features (code execution, LaTeX compilation, Docker custom tools, KB patch apply) default to **disabled** — keep new dangerous capabilities behind similar flags.

## Docker Services

Main services in `docker-compose.yml`:
- `postgres` (5432), `redis` (6379), `qdrant` (6333), `minio` (9000/9001), `ollama` (11434)
- `backend` (8000), `frontend` via `nginx` (3000)
- `celery` worker + `celery_latex` (dedicated LaTeX compilation queue)
- `kroki` (8001) - diagram rendering
- `video-streamer` - Go microservice for video streaming (in `video-streamer/`)

Variants: `docker-compose.prod.yml` (gunicorn, healthchecks, adds `celery_beat` scheduler), `docker-compose.test.yml` (isolated test stack on shifted ports), `docker-compose.docker-tools.yml` (mounts Docker socket for Docker-based tool execution).

Access points:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- MinIO Console: http://localhost:9001

There is no CI pipeline (no `.github/workflows`); quality gates are the Makefile targets (`make lint`, `make fmt`, `make test-backend-coverage`, `make typecheck-frontend`).

## Other Documentation

Root-level docs worth checking before larger changes: `BUILD_AND_RUN.md`, `QUICK_START.md`, `DOCKER_SETUP.md`, `AGENTS.md`, `AUTONOMOUS_RND_AGENTS.md`, plus `docs/` for architecture guides. Current visual architecture map (deployment, subsystems, agent runtime, LLM stack): `docs/ARCHITECTURE_DIAGRAMS.md`.
