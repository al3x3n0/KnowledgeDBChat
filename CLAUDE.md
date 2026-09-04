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
make test-backend-coverage  # Backend tests with CI-style 48% coverage gate
make test-frontend      # Run frontend tests
make typecheck-frontend # Frontend TypeScript typecheck
make db-migrate         # Run database migrations
make db-shell           # PostgreSQL shell
make shell-backend      # Access backend container shell
make health             # Check health of all services
make fmt                # Format backend code (black + isort)
make lint               # Lint backend code (flake8)
make doctor             # Validate env + health checks
make download-models    # Download embedding + reranking models
```

### Kubernetes / Helm
```bash
make minikube-up        # Start minikube, build images into it, install the chart
make minikube-reinstall # Reinstall on the running cluster without rebuilding
make helm-lint          # Lint the chart against every values profile
make helm-validate      # Render + kubeconform against the Kubernetes API schemas
make helm-smoke         # Install on the current cluster and assert its wiring
make k8s-status         # Pods and services of the release
make k8s-logs-migrate   # Alembic migration Job output
make k8s-test           # In-cluster smoke test (helm test)
```
The chart is `deploy/helm/knowledgedbchat`; see `deploy/README.md`. It mirrors
`docker-compose.prod.yml`, with one structural difference: Alembic runs in a
hook Job (`pre-upgrade` always; `post-install` with the in-chart Postgres, which
does not exist yet during pre-install) and every long-running pod sets
`RUN_ALEMBIC_MIGRATIONS=false`, so replicas never race the schema. App containers
override the image `ENTRYPOINT` for the same reason. A failed migration aborts
the upgrade before any pod rolls. New backend settings need no
chart change — anything in `config.py` can go under `config.extra` (ConfigMap) or
`secrets.extra` (Secret).

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
Migrations use sequential numeric prefixes (`0001_...` ... `0082_...`); follow that naming.

**Alembic is the only source of schema truth.** Never create tables from model
metadata and never add hand-written DDL at startup — that is what produced the
drift `0082_reconcile_schema_with_models` had to repair (12 tables, 46 columns,
42 indexes existed nowhere in the migration history). `create_tables()` remains
for tests and throwaway databases only. The `schema-truth` CI job applies every
migration to an empty Postgres and fails if the result differs from the models;
run it locally with `make db-check-drift`.

A database created before this change (by `create_all`, with no `alembic_version`
table) cannot simply be stamped: Alembic would create that table at its default
`VARCHAR(32)` and this repo's revision ids are longer. Use `make db-stamp-legacy`,
which creates the table at the right width and stamps head; it refuses to touch a
database that already has a revision recorded.

## Architecture

### Tech Stack
- **Backend**: FastAPI + SQLAlchemy 2.0 (async) + PostgreSQL + Redis + Celery
- **Frontend**: React 18 + TypeScript (CRA) + Tailwind CSS + React Router 6; Zustand (workflow editor state), React Query (server cache), ReactFlow + Dagre (graph/workflow canvases), Tiptap (rich text), react-hook-form, react-hot-toast
- **Vector Store**: Qdrant (default, runs as a service); ChromaDB (embedded) is still supported by the code but is no longer installed by default — it brings 173 MB of transitive dependencies for a backend this project does not use, so `pip install chromadb==0.4.18` first. **Embeddings and reranking run under ONNX Runtime** (`services/onnx_embeddings.py`), not torch: each model's own `onnx/model.onnx` is loaded from the same HF repo sentence-transformers used, so an existing index stays valid — measured per-vector cosine 1.000000 against the torch pipeline with identical top-5 rankings. That removed torch, transformers, scipy and scikit-learn (578 MB) for 53 MB, and it runs the cross-encoder torch fails on aarch64 ("could not create a primitive descriptor for a matmul primitive"), so reranking works where it used to disable itself. `EMBEDDING_BACKEND=sentence-transformers` restores the old path after `pip install sentence-transformers`
- **LLM**: DeepSeek (default), OpenAI, Anthropic, Qwen (DashScope), Kimi (Moonshot), or Ollama, selected by `LLM_PROVIDER`. The stack no longer bundles Ollama — that provider still works against an instance you run yourself via `OLLAMA_BASE_URL`. `DEFAULT_MODEL` must name a model the chosen provider serves, since it reaches the request as `model or <PROVIDER>_MODEL`; per-request routing via `services/llm_routing.py` (fast/balanced/deep tiers). Native tool calling and schema-constrained output live in `services/llm_providers/` (used by `LLMService.generate_structured()`); `generate_response()` is the legacy prompted-text path
- **Storage**: MinIO (S3-compatible object storage)
- **Transcription**: OpenAI Whisper, on a dedicated `celery_transcription` worker. Whisper, librosa, speechbrain and resemblyzer (and numba/llvmlite under them) live only in `Dockerfile.transcription-worker`, which builds FROM the backend image; the API, general worker and beat images do not carry them. `transcribe_document` is routed to the `transcription` queue (`TRANSCRIPTION_CELERY_QUEUE`), so with that worker stopped the task waits rather than fails. Speaker diarization (speechbrain first, then resemblyzer + KMeans) is optional and off by default
- **Diagrams**: Mermaid, rendered by `mermaid-renderer/` — a first-party Node service holding one headless Chromium, speaking the Kroki companion protocol (the full Kroki gateway was 3.76 GB to proxy to it, and its mermaid companion 1.54 GB); falls back to kroki.io

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
- **Tool governance** — `tool_registry.py` + `tool_policy_engine.py` + `models/tool_audit.py`; per-user tool policies, approval gates for dangerous tools (`AGENT_REQUIRE_TOOL_APPROVAL`, `AGENT_DANGEROUS_TOOLS`), full execution audit log, user-defined custom tools (optionally Docker-executed). Tool dispatch lives in `agent_tool_dispatch.py`. Every tool is **declared once** in `app/agent_core/tool_specs/` (one module per domain): the schema a model reads, the governance classification, which job types may call it, and — for measurement tools — what evidence it produces. `agent_tools.AGENT_TOOLS`, the catalog, the job-type policy and the evidence map are all views of those specs, so adding a tool is a handler plus a `ToolSpec`, not four files kept in step by hand. `tests/test_tool_specs.py` enforces it.
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
- Goal contracts are deterministic stopping rules in `config.goal_contract`. Besides the counting requirements (`min_findings`, `required_finding_types`, `required_result_keys`, ...) a contract may declare a `validity` block, checked by `services/agent_measurement_validity.py`: `predictions_measured` (every `record_prediction` in the run was settled by `record_measurement`), `require_uncertainty` (findings of these types must carry a spread or sample count), and `bounds` (per finding type, a numeric field and the range it must physically fall in). Counting requirements cannot tell a measurement from an artifact of the harness that produced it — a throughput benchmark short of independent chains returns exactly `latency/ways` and satisfies any count. Validity requirements also go into the *stable* thinking prompt, because they change how the work is done rather than only when it may stop
- Repeated tool failures escalate (`services/agent_failure_diagnosis.py`): the same tool failing with the same arguments and the same error class is called out on the second attempt and given a diagnostic protocol on the third (run a trivial control through the same tool; if it also fails the tool is broken and no edit to the input helps; if it succeeds, bisect the input one element at a time). Attached to the failing result so it travels in the history the model reads, and projected into `results.actions` as `repeat_attempt`/`failure_class`/`diagnosis_escalated`. Varying the call resets the count — changing the input is the wanted behaviour
- Methods are first-class knowledge (`services/agent_method_record.py`, tool `record_method`): a run records *how* to investigate something — procedure, what it prevents, and the finding types in this run that establish it — stored as a `pattern` job memory so later jobs recall it. Evidence is checked the same way `record_prediction` checks `derived_from`: citing a finding the run never produced is refused, and a method may only be stored without evidence by passing `['none']`, which marks it unvalidated. Construct these as `ConversationMemory` directly — `MemoryCreate` rejects the `pattern` type, and a method stored under a type the job-memory filter does not inject is written but never recalled. A contract can require one with `validity.records_method`
- Implementing an algorithm and measuring it is one chain, and the middle link is a correctness gate: `check_implementation` (`services/agent_implementation_check.py`) runs the code against reference cases from the paper before `benchmark_c_snippet` times it, because the fastest implementation of any algorithm is one that returns garbage. It is `perishable`, so a looping implement stage cannot inherit a verdict it earned before the edit, and a check with no cases reports unverified rather than passing vacuously. `compare_to_claim` (`services/agent_claim_comparison.py`) scores the result against the paper's number with three verdicts, not two — `incomparable` is the honest answer when units, hardware or input size make the two numbers untestable against each other. Build recipes live in `services/agent_toolchains.py`: **C and Rust**, both compiled from a single self-contained file (there is no network in the sandbox, so Rust is std-only, no crates). Both the checker and the benchmark take `language` and share that table, so the binary that was verified is the binary that was timed. rustc rejects `-O2` — its flags default to `-O`, and its own default is an unoptimised build that times the debug binary
- Wall-clock measurements report the machine they were taken on: `benchmark_c_snippet` samples `/proc/loadavg` and `nproc` in the same container as the trials and returns `load_per_cpu`, `measurement_environment` (quiet/busy/saturated) and `trial_spread`, warning when the host was busy or the trials unstable. Carried on the finding as well as the result, so `validity.bounds` can refuse a run whose numbers were taken on a machine too busy to measure anything. Only the wall-clock tool needs this — simulated cycles are the same on a busy host as a quiet one
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
- Heavy optional dependencies (pptx, sentence_transformers, bs4, croniter, mammoth, jsonpath_ng) are stubbed in conftest — sentence_transformers is no longer installed at all, so that stub is now the only thing that module means in tests — don't import them at module top-level in code paths tests touch without checking the stubs
- FastAPI dependency overrides replace `get_db` with test session
- User fixtures: `test_user` (regular) and `admin_user` with real password hashing; `auth_headers` / `admin_headers` via live token creation
- Async tests use `pytest-asyncio` (auto mode); markers: `unit`, `integration`, `slow`
- Coverage gate: 48% backend (`make test-backend-coverage`) — that is the measured floor, meant to ratchet upward; the suite currently reports 49.55%. The frontend has no `coverageThreshold` configured, so `npm run test:ci` collects coverage without enforcing it

## Commit Style

Follow Conventional Commits as seen in history: `fix(ui): ...`, `feat(admin): ...`, `style(ui): ...`

## Environment Configuration

Backend configuration is in `backend/.env` (copy from `env.example`). `core/config.py` has 150+ settings; major groups:
- `DATABASE_URL`, `REDIS_URL`, `DB_*` - Database connections and pooling
- `LLM_PROVIDER` - `deepseek` (default), `openai`, `anthropic`, `qwen`, `kimi`, or `ollama`; `OLLAMA_BASE_URL`, `DEFAULT_MODEL`, `DEEPSEEK_*`, `OPENAI_*`, `ANTHROPIC_*`, `QWEN_*`, `KIMI_*`
- `RAG_*` - RAG pipeline (hybrid search, reranking, MMR, dedup, KG context, chunking)
- `VECTOR_STORE_PROVIDER` + `QDRANT_*` / `CHROMA_*`
- `MINIO_*` - Object storage
- `WHISPER_*`, `TRANSCRIPTION_*` - Transcription and diarization; `TRANSCRIPTION_CELERY_QUEUE` names the queue the dedicated worker consumes
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
- `postgres` (5432), `redis` (6379), `qdrant` (6333), `minio` (9000/9001)
- `backend` (8000), `frontend` via `nginx` (3000)
- `celery` worker + `celery_latex` (dedicated LaTeX compilation queue) + `celery_transcription` (dedicated Whisper queue; its image derives from the backend image, so `make build` builds the backend first — compose does not infer build order from a `FROM`)
- `kroki-mermaid` (8001) - Mermaid rendering, built from `mermaid-renderer/` (this repo's own: Alpine + Chromium + mermaid-cli, 1.08 GB against `yuzutech/kroki-mermaid`'s 1.54 GB, with mesa and libLLVM deleted because a headless browser never opens them). Speaks the Kroki companion protocol: POST the raw diagram to `/svg` or `/png`
- `video-streamer` - Go microservice for video streaming (in `video-streamer/`)

Variants: `docker-compose.prod.yml` (gunicorn, healthchecks) — `celery_beat` runs in the dev stack too, and it is what makes `check_stalled_agent_jobs` fire there: with no beat, a job whose worker died (typically from restarting the celery container) stays `running` for ever until that task is invoked by hand, which requeues it to resume from its last checkpoint rather than failing it), `docker-compose.test.yml` (isolated test stack on shifted ports), `docker-compose.docker-tools.yml` (mounts Docker socket for Docker-based tool execution).

Access points:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- MinIO Console: http://localhost:9001

CI runs in `.github/workflows/ci.yml` on pull requests and pushes to `main`: backend lint/format, backend tests with the coverage gate, a single-alembic-head check, frontend typecheck plus tests, a `helm-chart` job that lints every values profile, validates the rendered manifests with kubeconform, and parses the generated nginx configs, and a `helm-smoke` job that installs the chart on an ephemeral kind cluster and asserts the wiring rendering cannot check (hook ordering, Secret-to-URL assembly, gateway routing, migration-gated upgrades). Lint is gated on `app/` and `tests/` only — `alembic/`, `scripts/`, and `seed_data/` carry pre-existing formatting and flake8 debt that is reported but not enforced. The same checks are available locally as Makefile targets (`make lint`, `make fmt`, `make test-backend-coverage`, `make typecheck-frontend`), which shell into a running Docker stack.

## Other Documentation

Root-level docs worth checking before larger changes: `BUILD_AND_RUN.md`, `QUICK_START.md`, `DOCKER_SETUP.md`, `deploy/README.md` (Kubernetes/Helm), `AGENTS.md`, `AUTONOMOUS_RND_AGENTS.md`, plus `docs/` for architecture guides. Current visual architecture map (deployment, subsystems, agent runtime, LLM stack): `docs/ARCHITECTURE_DIAGRAMS.md`.
