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

### Sandbox images
```bash
make sandbox-images     # every image this repo can build (base, compiler, polyglot, profiling, microarch)
make sandbox-polyglot   # C + Rust + Python and nothing else; no crates, so not the default
make sandbox-check      # which exist locally, plus the compiler image's toolchains and crate count
make sandbox-gem5       # arm64 only; --platform is not optional
make sandbox-axis AXIS_PATH=/path/to/axis   # context is the AXIS repo, not this one
```
**The agent resolves images by their registry-qualified name**
(`SCIENTIFIC_VALIDATION_ALLOWED_DOCKER_IMAGES` lists
`ghcr.io/al3x3n0/kdbc-*:latest`), so `docker build -t kdbc-compiler-research .`
produces an image the runtime never uses — `docker.io/library/...` is a
different image with the same Dockerfile. Build through `make sandbox-*`, which
tags `$(SANDBOX_REGISTRY)/...`, or `docker tag` afterwards. The symptom is a
fix that is demonstrably present when you run the image by hand and demonstrably
absent in the run: a tool kept reporting `unknown mnemonic 'uaddw'` two rebuilds
after that mnemonic was added. When the Makefile path is blocked (it rebuilds
`sandbox-base`, which needs apt), retagging the already-built image is
equivalent and needs no network.

**Without `docker-compose.docker-tools.yml` in the stack, none of these
images are reachable and every sandbox-backed tool fails** — gem5, compiler,
profiling and microarch alike — with `Cannot connect to the Docker daemon`,
because the backend and celery containers have no socket. `.env` can say
`UNSAFE_CODE_EXEC_BACKEND=docker` and be simultaneously true and useless. Bring
it up with:

```bash
docker compose -f docker-compose.yml -f docker-compose.override.yml \
  -f docker-compose.docker-tools.yml up -d backend celery
```

Add `--build` only if `docker` is absent from the image (`WITH_DOCKER_CLI`);
where the CLI is already present, `--no-build` avoids needing the package
mirrors, which this network intercepts. The socket grants the container
root-equivalent control of the host, so this is for a development machine or an
isolated runner. The symptom of forgetting it is silence in the evidence: a
capability reads as 0 findings, indistinguishable from one nobody has used.

The images agent tools run submitted code in (`deploy/sandbox-images/`). They
are coupled to the code: Rust support and the pinned crate set only work
against a compiler-research image built after they were added, and an older one
fails in ways that read as the model's mistake rather than a stale image —
`make sandbox-check` is the quickest way to tell.

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
- **LLM**: DeepSeek (default), OpenAI, Anthropic, Qwen (DashScope), Kimi (Moonshot), GLM (Zhipu), or Ollama, selected by `LLM_PROVIDER`. The stack no longer bundles Ollama — that provider still works against an instance you run yourself via `OLLAMA_BASE_URL`. `DEFAULT_MODEL` must name a model the chosen provider serves, since it reaches the request as `model or <PROVIDER>_MODEL`; per-request routing via `services/llm_routing.py` (fast/balanced/deep tiers). Native tool calling and schema-constrained output live in `services/llm_providers/` (used by `LLMService.generate_structured()`); `generate_response()` is the legacy prompted-text path
- **Storage**: MinIO (S3-compatible object storage)
- **Transcription**: OpenAI Whisper, on a dedicated `celery_transcription` worker. Whisper, librosa, speechbrain and resemblyzer (and numba/llvmlite under them) live only in `Dockerfile.transcription-worker`, which builds FROM the backend image; the API, general worker and beat images do not carry them. `transcribe_document` is routed to the `transcription` queue (`TRANSCRIPTION_CELERY_QUEUE`), so with that worker stopped the task waits rather than fails. Speaker diarization (speechbrain first, then resemblyzer + KMeans) is optional and off by default
- **Code symbols**: every language through its own parser
  (`services/repo_symbol_parsers.py`): Python `ast`, JS/TS tree-sitter, and
  **C/C++ libclang, which is essential**, not optional. A missing libclang
  raises `LibclangMissing` where it is first needed, instead of quietly
  treating C files as unreadable. The backend image's runtime stage asserts
  that libclang parses, so a broken install fails the build, and
  `test_libclang_is_installed_and_parses` fails CI. No regex fallback for any
  language: a pattern answered "not found" for a function defined in the file
  it searched
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
- `pages/` - 41 page components (ChatPage, DocumentsPage, AdminPage, AutonomousAgentsPage, AgentControlPlanePage, AgentBuilderPage, LatexStudioPage, PapersPage, ResearchNotesPage, ReadingListsPage, ResearchInboxPage, CodingBacklogPage, ResearchFleetPage, DomainProfilesPage, PatchPRsPage, RepoReportsPage, PresentationsPage, SynthesisPage, WorkflowsPage/WorkflowEditorPage, AIHubPage, KGAdminPage, GlobalGraphPage, ToolsPage, UsagePage, RoutingExperimentsPage, etc.)
- `components/` - Reusable UI, grouped by domain (`agent/`, `common/`, `docx/`, `kg/`, `notifications/`, `presentations/`, `search/`, `workflows/`)
- **The agent surface** was one 16,270-line `AutonomousAgentsPage` with thirteen
  tabs; it is now ~4,700 lines and a set of destinations. Each tab lives in
  `components/agent/tabs/`, and four moved out of Runs entirely because they are
  a different *noun*: Research Inbox (`/research/inbox`, Library — it is triage
  of papers, not a view of a run), Coding Backlog (`/coding-backlog`, beside
  Patch PRs — an item here becomes a patch there), Research Fleet
  (`/research/fleet`) and Domain Profiles (`/settings/domain-profiles`). Old
  `?tab=` links redirect, carrying their parameters.
  Shared machinery: `agent/useOpportunitySurface.tsx` (everything a domain
  profile and a research portfolio have in common — they render the same
  controls but are different nouns), `agent/autonomyShared.tsx` (their
  presentational half), `agent/agentJobMutations.ts` (job mutations both
  surfaces need), `agent/drilldowns.ts` (a URL parameter's parser and its
  printer, kept together), `agent/propTypes.ts` (prop types the tabs share).
  **Two rules, each learned from a bug that type-checked and passed tests.**
  State lives where its *writers* are, not its readers — splitting it yields two
  `useState` calls sharing a name, where one side's edits silently vanish;
  `tabs/__tests__/splitState.test.ts` fails on that. And never type a lifted
  prop `any`: it disables inference *inside* the component, which is how a badge
  got typed as a ReactNode when it is an object, and how two invented response
  shapes silently dropped fields the code reads.
- `services/api.ts` - Single `ApiClient` class (~4700 lines, 460+ methods) wrapping Axios with `/api/v1` base, token interceptor, and toast-based error handling — add new endpoints here
- `contexts/` - `AuthContext.tsx`, `NotificationContext.tsx`
- `hooks/` - `useWebSocket`, `useKeyboardShortcuts`, `useElementSize`
- `types/index.ts` - TypeScript interfaces (very large; mirror backend schemas here)
- Tailwind uses an inverted terminal/dark palette (gray-50 = dark, gray-900 = light) — check `tailwind.config.js` before assuming standard shades

## Major Subsystems

Beyond RAG chat, these are the main functional areas. When touching one, its endpoint module, model(s), and service(s) usually share a name prefix.

- **Autonomous agents & control plane** — observe→think→act→evaluate loop in `services/autonomous_agent_executor.py` (the largest service), decomposed into runtime services (`agent_observation_service`, `agent_thinking_service`, `agent_action_service`, `agent_progress_evaluation_service`, `agent_checkpoint_service`, `agent_runtime_*`). Job chaining/swarm orchestration in `agent_chain_orchestration_service.py`; autonomy policies and decision events (`models/autonomy_decision_event.py`, `agent_tool_prior.py`) surface in the control-plane UI. Specialized deterministic runners: coding, research, experiment, LaTeX, scientific validation (`agent_*_runner_service.py`, registered in `agent_deterministic_runner_registry.py`).
- **A tool that could not run says so on the first failure.** The usual
  silence at attempt 1 is right when a tool ran and refused the input — its own
  message is the remedy — and wrong when the tool never got that far, because
  no edit to the call can help. `agent_failure_diagnosis.could_not_run`
  recognises a daemon that is not listening, an upstream answering with a
  status, or a binary missing from the image, and escalates immediately with
  the control-run protocol. Measured: 19 iterations rewriting arXiv calls
  against a 406, 8 against an unmounted Docker socket. The predicate is narrow
  on purpose — `unknown mnemonic 'uaddw'` is also a `not_found` error, and
  there the tool ran and judged the input, which is a different situation with
  a different remedy — and a bare status only counts with its standard phrase,
  so a run reporting "503 cycles" is not mistaken for an outage.
- **A blocked run names what would end the stall.** "3 consecutive rounds
  produced no new findings" is honest and unanswerable: it describes the stall,
  not the thing a person could supply. `services/agent_unblock_request.py`
  derives a typed `needs` from the run's own history — a tool that never ran
  ("Is X available? It reported…", answerable only by a platform change), a
  tool that refused the call ("What does X accept here? It refused with…",
  answerable by an operator), or required evidence no tool this job type may
  call can produce. Derived rather than asked of the model, because a run that
  could reliably phrase its own blocker would not be stuck. A tool that never
  ran outranks a refusal seen earlier, since it has nothing to say about what
  it accepts; the run's own broken code raises no question at all; and a stall
  with no nameable blocker returns `None` rather than dressing itself up as a
  question. The `blocked_run` queue row shows it, and says when typing an
  answer would not help.

- **What a run learns about *calling* a tool is kept.** 415 methods existed and
  none was about tool usage, so one gem5 study was refused for passing a
  mechanism at the top level of a config, corrected itself, finished — and the
  next study made the identical mistake and spent five iterations on it.
  `services/agent_tool_usage_methods.py` records a method when a tool refuses a
  call and a later call to that same tool succeeds: the refusal was the lesson
  and the run has proved the correction works. Not recorded: the arguments that
  worked, since they routinely contain whole programs and the reusable part is
  the shape the tool described; and a refusal never recovered from, since an
  undemonstrated correction is a guess.

- **A run blocked by the platform files the platform's problem.** A run that
  blocks on its own bad input is the development loop; a run that blocks
  because a *tool* cannot do what it was asked is different in kind, and every
  later run meets the same wall until someone edits the platform. Those
  blockers arrive fully specified — `unknown mnemonic 'uaddw': add it to
  operand_arity` — so `services/agent_blocked_to_backlog.py` files them as
  coding backlog items with the tool's own words in `failure_symptom` and
  `error_output`. The discriminator is
  `agent_failure_diagnosis.blames_the_submitted_code`, reused rather than
  restated so the two cannot disagree about whose fault a failure is; a failure
  must repeat before it is filed (once is a flake, twice is a wall) and one
  item stays open per `(tool, error class)`. Filing is unconditional because
  recording a blocker costs nothing; **starting** work on it spends model
  budget, so that is what `AGENT_BLOCKER_AUTO_CODING_ENABLED` gates (default
  off). Nothing lands unattended either way: auto-filed items carry
  `auto_apply_enabled=False`, which the coding runner resolves to
  `proposal_only`.

- **Coding swarm** — backlog items, swarm profiles, code patch proposals, and PRs (`coding_backlog`, `coding_swarm_profiles`, `code_patches`, `patch_prs`); git operations via `git_service.py`, workspaces via `coding_workspace_manager.py`, symbol indexing via `repo_symbol_index_service.py`. KB patch application is gated by `AGENT_KB_PATCH_APPLY_ENABLED`.
- **Research suite** — papers (arXiv ingestion, enrichment, extraction, KG building: `paper_*_service.py`), research notes, portfolios, inbox with follow-up automation, monitor profiles, domain research profiles, reading lists. The research runner (`agent_research_runner_service.py`) orchestrates end-to-end workflows.

  **The inbox is a loop, and both halves of it are legible.** The monitor
  profile learns from triage and the runner scores candidates against it;
  `services/research_discovery_signals.py` keeps the terms that produced a
  score so an item can say *matches "sparse attention", a phrase from items you
  kept* instead of the old `token_bias`. Its tokenizer is asserted equal to
  `ResearchMonitorProfileService`'s rather than copied — drift there names a
  term that was never scored. The weight is a signed net over **occurrences**
  (`Counter.update(tokens)`), not a count of items, so the wording never says
  "you accepted this N times"; the number goes to a tooltip.
  Going the other way, a rejection carries a reason
  (`services/research_rejection_reasons.py`) and the reason decides what may be
  learned: only `off_topic` moves the token and phrase counters. Rejecting a
  weak paper on your own subject used to teach the profile to hide that
  subject, most strongly for the topics you see most. `low_quality` teaches
  **nothing** on purpose — the honest lesson is about a venue or a group, which
  this profile does not model, and downweighting every arXiv paper would be a
  worse error than silence; the UI says so beside the choice. A NULL reason
  keeps the old behaviour, because rows triaged before the column existed were
  learned from under those rules.
- **Document generation** — LaTeX projects with server-side compilation (dedicated `celery_latex` worker, disabled/admin-only by default via `LATEX_COMPILER_*`), DOCX editor, PPTX/presentation generation, PDF export, artifact drafts staged for review before publishing.
- **Training / AI Hub** — datasets, fine-tuning jobs (`services/trainers/`, backends: local/modal/runpod), model registry, eval templates and benchmark harness. Gated by `TRAINING_ENABLED`.
- **Experiments & scientific validation** — experiment plans/runs, Docker-sandboxed validation with image allowlists and resource caps (`SCIENTIFIC_VALIDATION_*`, `UNSAFE_CODE_EXEC_*` settings).
- **Navigation & UI customization** — the nav was a literal inside
  `Layout.tsx`: the same five doors for everyone, changeable only by editing
  the component. It is now `frontend/src/navigation/`: `catalog.ts` declares
  every destination as data (a `visibility` flag rather than an inline
  ternary, so settings can explain why an entry is absent), `preferences.ts`
  is a **pure** function applying one person's arrangement to it, and
  `useNavigation.ts` is the single resolved answer the sidebar, the settings
  editor and the `/` redirect all read — they used to disagree by
  construction, since the landing page was a hardcoded `<Navigate to="/chat">`
  in `App.tsx`. A destination's identity is its **route** (including the query
  string, which is the only thing separating the two Admin tabs), so renaming
  an entry never orphans the preference that hid it. Three rules are decisions
  rather than details: an order is a preference and not a whitelist (entries it
  does not mention keep their catalog position, or every newly shipped page
  would be invisible to anyone who had customized); hiding never hides the page
  you are standing on; and a door disappears when its last section does.
  Storage is `user_preferences.ui` (JSON), **normalized on write** by
  `services/ui_preferences.py` — it is the one field whose value is a document
  the client composes, so unknown keys are dropped, lists capped and labels
  trimmed, and a non-string key is rejected rather than coerced. Null means
  "never customized", which is deliberately distinct from `{}`. Per-panel
  collapse toggles stay in `localStorage`: those are per-device conveniences,
  while hiding a destination is a decision about your work.
- **An agent definition can be written from a description**
  (`services/agent_definition_author_service.py`, `POST /agent/agents/draft`,
  the Draft box in Agent Builder). The prose is the easy part; the two closed
  sets either side of it are not, and both fail **silently**. A capability
  outside `CAPABILITY_KEYWORDS` is never matched, so the router never reaches
  the agent; a `tool_whitelist` naming something that is not a tool is a filter
  matching nothing, so the agent ends up with fewer tools than its author
  believes. Neither raises and neither logs. So the drafter checks against the
  real things: `AgentDefinitionCreate` (the schema the create endpoint uses),
  the router's own capability vocabulary, and the tool catalog — none of them
  restated — and hands a refusal back verbatim, because a refusal that names
  what is wrong is what the next attempt needs. An empty `tool_whitelist` is
  refused separately from an absent one: `null` means every tool and `[]` means
  none, and the difference is total. Drafting **creates nothing**; `notes` says
  what had to be repaired, which is the part worth reading.
  Passing `current` makes it a **revision** — "restrict it to the coding
  tools" applied to what the author has, rather than a fresh invention. Three
  choices there are deliberate: the revision reads from **the form**, not from
  the model's own last answer, because a hand edit between passes would
  otherwise be discarded silently; a revision goes through the same `check()`
  as a first draft, since "make it narrower" is no reason to accept an agent
  nothing can route to; and empty fields are not shown as current state,
  because offering `tool_whitelist: []` invites the model to preserve "no tools
  at all" when an absent whitelist means the opposite. A revision keeping its
  own name skips the collision check — colliding with yourself is not a
  collision.

- **Three drafters share one repair loop** (`services/draft_repair_loop.py`).
  Plugins, agent definitions and sandbox skills each wrote the same thing:
  ask for JSON, hand it to the real validator, and when it refuses give the
  refusal back and ask again. What differs is the judge, so that is all a
  drafter supplies — a function returning a `Verdict` (the candidate to keep,
  the complaint for the model, the note for the person, `stop` when the
  failure is not in the draft, `discard` to forget an earlier candidate). The
  loop owns the rest: the model call, an unreachable model, a reply that is
  not JSON, the growing message, the notes and the progress callback. The
  pipeline drafter is deliberately not on it: one repair round, the whole
  draft resent, the less-broken version kept — a different procedure.

- **Plugins** — a plugin is one installable unit that contributes tools (and, in
  later slices, flows and UI). `models/plugin.py` holds the manifest;
  `PluginInstallation` holds one user's decision to run it, so a builtin bundle
  can belong to the deployment and still be opted into per user. Two sources,
  one loader (`services/plugin_registry.py`): bundles shipped under
  `backend/plugins/<name>/plugin.json` are synced at startup, user-authored
  plugins are rows. `services/plugin_manifest.py` validates at install and
  **names what is wrong**. A contributed tool is promoted to a real `ToolSpec`
  (`agent_core/plugin_specs.py`) and offered to the model by name under the
  reserved `p_<slug>_<tool>` namespace, resolved **once at job start** because
  the tool menu keys the provider prompt cache. Two things are never taken from
  the author: the **classification**, which is derived from the executor type
  (a webhook declaring `effects: read` is overridden, not believed), and the
  **name**, which is checked against what a provider accepts — `UserTool.name`
  has no charset constraint, so a tool called `my tool!` is skipped with a
  warning rather than breaking a run. Execution goes through
  `CustomToolService`, so the policy engine, approval gates and audit apply
  unchanged. `agent_core/tool_specs.ToolCatalog` is what makes this possible:
  the module-level views now delegate to `STATIC_CATALOG` (built-ins only), and
  a caller that knows whose tools it wants builds its own with
  `extended_with()`, which refuses anything shadowing a built-in. Chat offers
  contributed tools too (`AgentService._contributed_tool_schemas`, resolved
  per call because the service is a singleton).

  **Chat offers only the built-ins it can run** (`AgentService._chat_tools`).
  `AGENT_TOOLS` is every declared tool and most are answered only by the
  autonomous-job providers; chat described all 250 in 158,600 characters of
  every planning prompt, and 180 of them came back as "Unknown tool". The menu
  is asked of the registry rather than listed, so a provider that starts
  answering in chat is offered there without anyone remembering to say so.
  **There is one chat turn** (`AgentService.process_message`). The streaming
  socket the chat widget uses had its own planner and its own responder, so
  the chat window had no agent routing, no whitelists, no memory and no tool
  rounds while `POST /agent/chat` had all four — every fix made to one had to
  be made twice or it reached the endpoint and not the window. The duplicates
  are deleted; the socket handler calls `process_message` with an `on_event`
  sink and forwards what it reports (`planning`, `tool_start`,
  `tool_progress`, `tool_complete`/`tool_error`, `generating`). A sink that
  fails is ignored: progress is a courtesy. The widget now sends its
  conversation id, which memory and skill working directories are keyed on.
  **A chat turn plans in rounds** (`AgentService._run_tool_rounds`). It used
  to plan every call before any result existed, which cannot do anything whose
  second step depends on the first: asked to use a sandbox skill, chat planned
  `list` and `load` and stopped, because the command is in the procedure
  `load` had not yet returned. The planner is now asked again with what came
  back until it plans nothing. The loop also ends at
  `AGENT_CHAT_MAX_TOOL_ROUNDS` (4; 1 restores single-shot), at
  `AGENT_CHAT_MAX_TOOL_CALLS` (12), on a round that only repeats calls already
  made, and on anything that needs the person (an approval, a file). Run live:
  list → load → run in one turn, 28 s, with the judged result.
  **A specialist's whitelist decides what it can do in chat.** The router sent
  that same request to `compiler_optimization_expert`, whose whitelist
  predates the skill tools, so it planned nothing — and the reply then
  described what the skill *would* have printed. Migration `0104` grants that
  one agent the skill tools, and the response prompt now forbids presenting an
  expected output for a run that did not happen. Migration `0105` then grants
  them to every *system* specialist, because a skill working or silently not
  depending on which agent the router picked is not a behaviour anyone chose;
  agents a user authored keep the whitelist their author wrote.

  **Skills are reachable from every surface, through one set of handlers.**
  Autonomous jobs and pipelines (the `autonomous` registry), chat and
  workflow `tool` nodes (the chat registry, which
  `AgentService.execute_tool` also serves), and the MCP server
  (`mcp/tools/sandbox_skills.py`), whose input schemas are read from the tool
  specs rather than restated. What differs per surface is only what the
  working directory is keyed on — job, conversation, user, or API key — and,
  on MCP, the scope a call needs: `read` to list or load, `write` to run or
  propose. MCP does not force approval for a skill run the way it does for
  `docker_execute`: the image is allowlisted, there is no network, and only a
  skill whose control has passed can run at all.

  **`llm_json` is the only place JSON is dug out of a model's reply.** About
  twenty sites did it by hand (`find("{")`..`rfind("}")`, or a greedy
  `\{.*\}`), and they disagreed: a reply holding two objects parsed in one
  service and failed in another, some handled a fenced reply and some did
  not, and exactly one accepted raw newlines inside a string because it had
  been bitten. Four drafters each carried a copy of `_payload`. All of them
  now call `extract_json_object`, `extract_json_array`, `require_json_object`
  (raises) or `completion_object` (takes an `LLMCompletion`, a mapping or a
  string; returns `{}`), with `strict=False` for replies that carry source
  code. `test_nothing_else_digs_json_out_of_a_reply_by_hand` refuses the
  idiom anywhere else.

  **`llm_json` must define every helper a caller uses.** `extract_json_array`
  was deleted on 2026-08-06 while the chat planner and the presentation
  generator still called it. Both wrap the call in `except Exception`, so
  nothing raised: chat discarded every tool call the model made for eight
  weeks and answered as though none had been needed.
  `test_every_helper_a_caller_uses_exists` reads the callers, since a missing
  attribute is invisible to a test that exercises only the helper.

  A plugin also contributes **UI, declaratively** (`services/plugin_ui.py`,
  rendered by `frontend/src/plugins/`): `contributes.views` (kinds `table`,
  `detail`, `stats`, `markdown`), `contributes.nav` (an entry appended to a
  door, reachable at `/p/<slug>/<viewId>` — a manifest cannot choose its own
  path, or an installed plugin could claim `/documents`), and
  `contributes.panels` (into a named slot in an existing page). **No plugin
  JavaScript runs in the app's origin**; first-party React renders plugin
  *data*, markdown is rendered without raw-HTML support, and every value goes
  through JSX text. Two rules are security boundaries checked against the
  executor type rather than the manifest: a view may only be backed by a
  **read-only** tool (rendering a page calls its source, so a write-backed
  view would perform a write on every visit), and a view may only call **its
  own plugin's** tools. `GET /plugins/me/ui` serves the contributions;
  `POST /plugins/me/views/{slug}/{view}/data` runs one view's declared source
  with arguments from the manifest — deliberately not a general "run any tool
  from the browser" endpoint. Every name in `PANEL_SLOTS`, `NAV_DOORS` and
  `ICONS` must exist on both sides;
  `frontend/src/plugins/__tests__/contract.test.ts` reads the Python directly
  and fails on drift, including a slot no page actually hosts.

  A plugin can also be **written from a description**
  (`services/plugin_author_service.py`, `POST /plugins/draft`, the Draft box in
  the Plugins panel). It does two things a single model call cannot. It
  *repairs against the real validator* — a refused draft goes back with the
  refusal, and nothing re-implements a rule, so the thing that decides at
  install is the thing that decides while drafting. And it *runs the tool it
  wrote*: a drafted `transform` is executed and the view's declared `path`
  resolved against the real output, because a path that misses renders an
  empty table indistinguishable from a tool with nothing to say. Only
  `transform` is dry-run — it touches nothing, whereas a webhook would reach
  the network and an `llm_prompt` would spend a model call, neither of which
  should happen because someone typed a sentence. The prompt's vocabularies
  are read from `plugin_ui.py` and `custom_tool_types.py` rather than
  restated. Drafting never installs: the manifest comes back for review, with
  `notes` saying what had to be repaired, because one that validates is not
  one that does what was meant.

  A plugin can also contribute **flows**. A workflow's `tool` node may name a
  contributed tool (`workflow_engine._execute_contributed_tool` goes through
  the same `PluginToolProvider` an autonomous job uses, so the two surfaces
  cannot disagree about what a tool is), and `contributes.workflows`
  (`services/plugin_flows.py`) ships whole graphs. Unlike tools and views a
  shipped workflow is **materialized** — real `workflows`/`workflow_nodes`/
  `workflow_edges` rows owned by the installing user — because it is an
  editable object with executions attached. That decides the update rule:
  installing creates, re-installing **keeps**, because somebody's edits are
  worth more than a version bump. Identity is `origin_flow_id` (the manifest's
  id) and never the name: matching on name meant renaming a flow made the next
  install create a duplicate. Node types are read from
  `workflow_engine.NODE_TYPES` rather than restated, and node ids are
  **refused** when longer than the `String(50)` column rather than truncated —
  shortening an identifier changes which node an edge names.
- **Sandbox skills** — a sandboxed capability written as *data*. Every
  sandbox tool used to be a handler plus a `ToolSpec`, so a new kind of
  sandboxed work was a code change and a pipeline could only require evidence
  one of those handlers produced. A skill (`models/sandbox_skill.py`,
  validated by `services/sandbox_skill_manifest.py`) carries a procedure the
  agent reads, helper files, the image it runs in, the fields its
  `result.json` must have, and a control. Four built-in tools
  (`agent_core/tool_specs/skills.py`, handlers in
  `services/agent_sandbox_skill_tools.py`) let a run list, load, run and
  propose one. **The agent chooses the commands; the platform decides what
  counts.** A result is read from the sandbox and checked against the declared
  fields, a stale `result.json` is deleted before every run, and a skill may
  name a `judge_command` whose output replaces whatever the run wrote — the
  finding records `judged_by` because those are not equally strong.
  **One path to active**: every skill arrives as a draft (hand-written,
  drafted, or proposed by a run), and `activate` refuses until the control has
  passed against the *current* content hash. Editing an active skill returns
  it to draft; a control that starts failing deactivates it; a daemon that is
  merely unreachable revokes nothing, since it says nothing about the skill.
  Evidence is named `skill_<id>`, a namespace no built-in occupies
  (`test_no_builtin_evidence_lives_in_the_skill_namespace`).
  `agent_evidence_map` treats the namespace as produced by
  `run_sandbox_skill`, so contracts, chains and prices work unchanged — but
  that static check has no user, so it accepts a skill nobody has. The half
  that knows is `agent_pipeline_draft.skill_problems`, applied in the pipeline
  `check`/`bind`/`launch` endpoints and in the drafter's repair loop.
  Drafting (`services/sandbox_skill_author_service.py`,
  `POST /sandbox-skills/draft`) repairs against the real validator and **runs
  the control it wrote**; when the sandbox cannot run at all it returns the
  draft unverified rather than spending model calls on a failure no edit can
  fix. The drafter also **asks each image what it contains**
  (`sandbox_skill_runtime.probe_tools`, cached per process): the first live
  draft called `gcc` in an image that ships clang, was shown the error, and
  called `gcc` again; with the tool list in the prompt the same request was
  right first time. An image that cannot be asked is "unknown", never "empty".
  Images: a skill may only name an allowlisted image or a *built*
  authored one. Authored images (`services/sandbox_skill_image_service.py`)
  are proposed by anyone and built only by an admin, only with
  `SANDBOX_SKILL_IMAGE_BUILD_ENABLED` (default off); a Dockerfile must have
  exactly one `FROM` naming an allowlisted image, no `COPY`/`ADD` (there is no
  build context) and no `ENTRYPOINT` (the sandbox runs `/bin/sh -lc`). The UI
  is the Sandbox Skills panel on the Tools page.
  **Files pass between pipeline stages by copy.** A job has one working
  directory, shared by every skill it uses; a job with a parent starts with a
  *copy* of the parent's (minus `skill/` and `result.json`), made once when
  the directory is first created. Copied rather than shared because sibling
  stages run at the same time and each rewrites `skill/` and deletes
  `result.json`. `load_sandbox_skill` and every run return the directory
  listing, since a stage is only spared rebuilding what it was handed if it
  is told the files exist; a stage that inherits nothing (parent used no
  skill, directory pruned, or over 256 MB) is told it starts empty. Run live:
  a build stage left `kernel.o` and an inspect stage read its symbols from it.
  **A skill says whether its result goes stale** (`perishable: true`). The
  evidence map is fixed at import and cannot know one user's skills, so the
  *finding* carries the flag to the two places that act on it:
  `_inherit_assumed_findings` skips it, and `validity.bounds` reads only the
  latest. Omitted from the manifest when false, so adding the option did not
  change any existing skill's content hash.
  **Chat can use skills too.** Every spec is advertised to chat whatever its
  provider's modes, so an autonomous-only provider is a tool chat is offered
  and then told is unknown; the skill provider answers in both. Chat has no
  job, so the working directory is keyed on the conversation ("build it" and
  "now measure it" two messages apart find the same files) and proposals are
  bounded by the review queue (10 unreviewed) rather than per job. A finding
  recorded in chat satisfies no contract -- there is none.
  Not done yet: run directories
  under `$TMPDIR/kdbc-skills` are pruned after a day rather than at job end,
  so a stage waiting longer than that on a checkpoint inherits nothing; and a
  stage inherits from its chain parent only, not from every `depends_on`.
- **There is one confined `docker run`, and it is `agent_sandbox_runtime.docker_command`.**
  Five places built that command by hand: the runtime, the experiment runner,
  the ingestion demo runner (twice) and the admin sandbox check. They agreed
  on the posture — no network, no capabilities, uid 65534 — so nothing looked
  wrong. But only the runtime named its container, so a timeout in the other
  four killed the `docker run` client and left the container running on the
  daemon: the orphan leak that module's docstring describes, fixed in one copy
  out of five. Checked on the real daemon: after a timed-out `subprocess.run`
  the container was still listed until removed by name. All four now call
  `docker_command(..., name=new_container_name())` and remove the container
  on timeout (`remove_container_sync` for code running in a thread).
  `tests/test_sandbox_container_cleanup.py` fails if `--cap-drop` appears in
  any other module. `docker_tool_executor` is a different thing — a custom
  tool chooses its own image, user and network, and is gated by
  `CUSTOM_TOOL_DOCKER_ENABLED` and approvals — so it keeps its own builder,
  but it had the same leak and now also sets `no-new-privileges` and a pids
  limit. It still does not drop capabilities: that could break a tool that
  legitimately runs as root, and is a decision rather than a cleanup.
- **Tool governance** — `tool_registry.py` + `tool_policy_engine.py` + `models/tool_audit.py`; per-user tool policies, approval gates for dangerous tools (`AGENT_REQUIRE_TOOL_APPROVAL`, `AGENT_DANGEROUS_TOOLS`), full execution audit log, user-defined custom tools (optionally Docker-executed). Tool dispatch lives in `agent_tool_dispatch.py`. Every tool is **declared once** in `app/agent_core/tool_specs/` (one module per domain): the schema a model reads, the governance classification, which job types may call it, and — for measurement tools — what evidence it produces. `agent_tools.AGENT_TOOLS`, the catalog, the job-type policy and the evidence map are all views of those specs, so adding a tool is a handler plus a `ToolSpec`, not four files kept in step by hand. `tests/test_tool_specs.py` enforces it.
- **Chains are retired as an authoring concept.** `POST`/`PATCH
  /agent-jobs/chains` are marked `deprecated` in the OpenAPI schema and the
  Chains tab points at Pipeline Studio's import, but both still serve and every
  saved chain still runs: this is deprecated as a way to *write work down*, not
  as a runtime, and removing it before someone has a working pipeline in its
  place would take away the thing that works. The three shipped chains were
  converted and saved as pipelines that validate. A chain and a pipeline
  produce the same runtime -- chained jobs linked by `chain_config`, created by
  `create_chained_job` -- and that runtime is staying; pipelines run on it. What
  is going is `AgentJobChainDefinition` as a *second way to write work down*. A
  chain says when the next step fires; a pipeline says what must be true when a
  stage is done. `services/agent_chain_to_pipeline.py` converts one to the
  other, surfaced in the Pipeline Studio, and **refuses rather than
  approximates**: three of six `trigger_condition` values map (`on_complete` to
  `depends_on`, `on_approval` to `checkpoint`, `on_findings` to `spawn_on`), and
  `on_fail` / `on_any_end` / `on_progress` do not, because a DAG edge cannot
  branch on an outcome or fire partway through. A converted pipeline arrives
  with empty contracts and so does not validate: that is the point, since
  `validate` then names per stage the contract someone has to write.
- **`spawn_on` is the one place a pipeline edge does not mean "after".** Every
  other dependency waits for a stage to finish, which cannot describe a stage
  that never does -- a continuous monitor is still monitoring, and a successor
  waiting on it waits forever. A stage may instead release its successors on
  evidence: `spawn_on: {findings: N}`, which the binding emits as the
  `on_findings` trigger the chain runtime already handles, so nothing in the
  runtime changed. It is refused alongside `checkpoint` (one waits for a person,
  the other does not wait at all), on a stage nothing depends on, and on a
  contract requiring no finding types, since there would be nothing to count.
  Known limitation: `agent_pipeline_restart` will not restart a stage whose
  parent has not completed, which a spawning parent may never do.
- **A contract may only require evidence some callable tool produces.** A tool
  spec's `job_types` distinguishes `None` (every job type) from `()` (**no**
  autonomous job type — 58 tools are reachable only from chat or MCP), and
  reading the empty tuple as "unrestricted" let a stage validate, plan and start
  with a contract nothing it could call would satisfy. `validate()` now refuses
  a required evidence type when *every* producer is out of reach for the
  stage's job type — per evidence, not per tool, because several types have
  alternatives and one barred producer proves nothing (`papers_ingested` names
  `ingest_arxiv_papers`, which no job may call, and `ingest_paper_by_id`, which
  research may). `tests/test_contract_producers_are_reachable.py` pins the stranded
  list at empty, so a tool withdrawn from autonomous jobs surfaces there rather
  than in a stuck run. `literature_review` was the last stranded type and was
  fixed both ways: `literature_review_arxiv` gained the job types its
  same-shaped neighbour `ingest_paper_by_id` has (it searches arXiv and
  ingests), while `generate_literature_review_for_source` **stopped declaring
  `produces`** — it queues a Celery task and returns, so the review is written
  after the run that asked for it has moved on, and `produces` feeds machinery
  that is strictly in-run. It is still callable; it just no longer advertises
  evidence a caller cannot observe.
  `agent_contract_suggestions` filters the same way — a suggestion that cannot
  be satisfied is worse than none, because it looks like an answer.
  A third way a contract can be unsatisfiable is invisible to both checks: the
  tool is reachable and the guidance names it correctly, but its handler records
  the evidence on the wrong channel. Contracts count **findings**;
  `create_synthesis_document` emitted `synthesis_document` only under
  `artifacts`, so the one tool meant to satisfy that contract never could — a
  pipeline's writeup stage called it, succeeded, and still ended
  `completed_contract_unmet`, and in 364 jobs no finding of that type had ever
  been recorded. A tool declaring `produces=(...)` must return a `findings`
  entry carrying that type.

  The same rule governs the *prompt* and the *price*:
  `agent_evidence_map.chain_for` takes the job type and picks a producer that
  job type may call, so `describe_chain` (the prompt) and `plan()` (the estimate
  a person acknowledges before launching) derive the same tool instead of
  deriving it separately and disagreeing. Two saved pipelines were priced
  against `ingest_arxiv_papers`, which no job may call, while the run would have
  used `ingest_paper_by_id` — one estimate was out by a factor of four.
  A `papers_ingested` stage was told "ingest_arxiv_papers (or
  ingest_paper_by_id) yields papers_ingested" when it could call only the
  second; it searched, found 18 papers and spent its remaining rounds on web
  search and progress reports without ingesting one. Naming a tool the runtime
  will refuse is worse than naming none, because the run plans around it.

- **One idealisation/mechanism pairing crashes gem5: `l1d_capacity` with a
  prefetcher on L2.** It dies inside
  `BaseCache::CacheReqPacketQueue::sendDeferredPacket`, printing a libc
  backtrace and *no* diagnostic — no `panic:`, no `fatal:`, nothing — which is
  why it was first recorded as the unreadable "A simulation failed." and
  diagnosed far too broadly, as idealisation being incompatible with mechanism
  configs generally. `measure_headroom` now refuses this one pairing up front,
  since the baseline arm runs first and a doomed run should not spend a full
  simulation to discover it.

  Measured one factor at a time, and the rule is exactly this narrow:
  idealised l1d + L2 prefetcher **crashes**; default l1d + L2 prefetcher is
  fine; idealised l1d with the same prefetcher on **l1d** is fine; idealised
  l1d alone is fine (73.77% headroom on the kernel that first hit it);
  idealised **l1i** or **l2** with an L2 prefetcher are fine. So the natural
  progression — measure a mechanism, attribute what now limits it, bound the
  remaining headroom of that same machine — is available; it is one square of
  the grid that is not. An earlier hypothesis that the cause was the inverted
  hierarchy (a 16MiB L1 above a 2MiB L2) was tested and refuted: widening L2
  to 64MiB still crashes, so the prefetcher is the necessary ingredient.

  A crash with no diagnostic is also why `_gem5_failure_line` falls back to
  the top gem5 stack frame, demangled: when gem5 dies without a word, where it
  died is the only account of what happened that exists.

- **A failed tool call records why it failed.** `compact_action_ledger` keeps
  raw tool output out of `results.actions` deliberately, but it was dropping
  the error *message* too, so a failed call was indistinguishable from one that
  returned nothing. That is how an upstream outage read as an agent refusing to
  work: arXiv's API began answering `export.arxiv.org/api/query` with **HTTP
  406** for this host (while `arxiv.org` itself serves normally and egress is
  fine), and three discovery stages spent forty-odd iterations able to report
  only "no new findings". The distinction is visible in the ledger now, and it
  is sharper than "arXiv is down": `search_arxiv` succeeded while
  `ingest_arxiv_papers` and `literature_review_arxiv` failed, so a pipeline
  whose discovery stage required `papers_ingested` could not pass, while ISE
  fusion's research stage kept working because it draws on the corpus
  (`related_paper_set`) rather than the API.

  **That outage cleared on 2026-09-21** — `export.arxiv.org/api/query` answers
  200 again, with any User-Agent or none, and `literature_review_arxiv` returns
  papers and creates its document source through the real tool path. It was
  never a request the code could fix: the same binary that got 406 for days now
  gets 200 unchanged. Worth remembering when the next discovery stage reports
  nothing, because the shape recurs — check the upstream by hand before
  editing anything, and read the ledger's error, which is what makes an outage
  distinguishable from an agent with nothing to say.

- **A model proposes application-specific optimisations; it never judges
  them.** Scanning and `build_llvm_pass` stay inside what a compiler may
  assume about *any* program. `propose_restructurings`
  (`services/agent_restructure_proposer.py`) asks for the other kind of change:
  one that holds because of something true of *this* application, such as a
  parameter that doesn't change within a call, a bounded value range, or work
  that gets repeated. Every proposal goes through `agent_restructure`, which
  compiles a caller-owned driver once. A candidate replaces only the kernel,
  so it can't change what is measured. The checks run in order: build, then
  identical output on **every** input (a diverging candidate is never timed),
  then interleaved timing against the original **and** against the original
  at -O3 (plus -ffast-math if results may change). A win must beat the noise
  on both the fastest and the median trial. `compiler_already_can` requires
  the ceiling to *show* the gain. When the ceiling comparison is inside the
  noise, the verdict is `faster_than_original`: a sincos rewrite 1.31x over
  -O3 was first mislabelled because the host noise was 34%. Two things the
  first live run got wrong are now structural. Proposals are requested **one
  per call**, each told which ideas are taken; three whole files in one reply
  exhausted a reasoning model's 32k budget before it wrote a character. And
  when several proposals win, each is re-measured **against the best**. Three
  "different" ideas all carried the same lookup table, so "branchless
  masking, 6.3x" was really the table's 7.2x under another name.
  `propose_binary_rewrites` / `evaluate_binary_rewrite`
  (`services/agent_binary_rewrite.py`) apply the same judgement to one
  function of a relocatable object. The model sees only the disassembly;
  given C, it is compiled and withheld. The replacement is spliced in by
  `llvm-objcopy --weaken-symbol`, and the build refuses a replacement that
  does not export the symbol globally. Without that check the link succeeds
  against the weakened original, and the "rewrite" is the original measured
  twice. That check uses `false`, not `exit 1`: inside the shared script's
  brace group, `exit` ended the whole run and hid the log. Equivalence is
  checked on the given inputs, not proved, so the invariant a proposal relies
  on is recorded on its finding.

- **A winning rewrite can become a pass, and the pass is judged on four
  separate questions** (`services/agent_pass_from_rewrite.py`,
  `synthesize_pass_from_rewrite` and `evaluate_pass_on_kernel`). Does it
  *fire* under `clang -fpass-plugin` (`did_not_fire` otherwise, never timed)?
  Is the output identical on every input? How much of the hand rewrite's gain
  does it *recover*? Both are timed in one interleaved run. And does it leave
  a `must_decline` case alone? A pass that changes that case gets
  `overreaches` whatever its speed. One live example matched any integer width
  and indexed past its 256-entry table on 16-bit input. The kernel's inputs
  could not catch that. The decline case is **frozen after the first
  attempt**, or the cheapest repair is to move the test. Each source also goes
  through `opt -passes=<name>,verify` right after the pass. Release clang does
  not verify IR, and later passes papered over a fastmod pass's
  `lshr i128 %x, i64 64`: the final IR verified clean, two repairs were told
  only "diverged", and the verifier named the instruction at once
  (`invalid_ir`). `not_expressible` is a result, not a failure. The model is
  asked to say when a rewrite relies on a fact the IR does not carry. Asked to
  drop stores the driver never reads, it named exactly that fact. Live: the
  trig-table pass was 8.0x, with 116% recovered, since the table is computed
  at compile time. The fastmod pass was 2.48x on the median, 2.44x over -O3.
  An `unresolved` verdict is re-measured once at 15 trials, because the Docker
  VM's load average cannot see load on the macOS host. It read "quiet" while
  that host was saturated.

- **The optimisation chain has been run on a real codebase (raylib 5.0),
  and three things only real code could show are fixed.**
  First, `scan_for_optimizations` could not take a repository at all. It
  accepted only bare `.c` text, while raylib's translation units include
  11 MB of headers under `external/`. It now takes `paths` + `include_dirs`
  from a `clone_and_index_repo` workspace. The workspace is copied into the
  sandbox directory, because the daemon resolves mounts on the host, where
  only that directory is shared. Each failed file is reported with its own
  first error. Over 7 of 7 translation units, 227 of 374 divisions inside
  loops were loop-invariant.
  Second, `propose_restructurings` did not re-measure. Three
  `ImageBlurGaussian` rewrites were all bit-identical to raylib and all
  `unresolved` at 7 trials. The proposer now does what the pass path does: an
  unresolved candidate whose fastest trial hints at a gain is re-timed once at
  15 trials. The one that held was 1.43x, and 1.44x over -O3: it stores the
  blur's intermediate as bytes, since the vertical pass truncates to
  `unsigned char` anyway.
  Third, verdicts are judged on user+sys time from a tiny `wait4` helper
  built in the sandbox. That helper resolves microseconds where `times` and
  `/usr/bin/time` give 10 ms, and it enforces its own timeout, since
  `timeout` around it would orphan the program. **This does not remove noise
  under Docker Desktop.** The VM cannot see host preemption of its vCPUs, so
  that time counts as running. The same null control read 0.159 on both
  bases. Wall time is still used whenever CPU time exceeds it, because then
  the candidate is multithreaded.
  Turning the blur rewrite into a pass failed honestly: two compile errors,
  then a pass that never fired. A data-layout change is a hard case, and
  nothing broken was reported as working.
  Found on the way: raylib's `ImageBlurGaussian` reads past its pixel buffer
  when `blurSize` exceeds the image width or height. It is still present on
  master `6ecf21f` (2026-09-28), shown with guard-page allocations since the
  image ships no ASan runtime. It has not been reported upstream.

- **Linked executables are optimised with BOLT, and judged like everything
  else** (`services/agent_bolt.py`, image `kdbc-bolt-research` =
  sandbox-base + Debian's `bolt-19`, `make sandbox-bolt`). A linked binary
  has no linker left to prefer a strong symbol, so the function-swap used for
  object files is impossible. What can change is *layout*. The tools build the
  program **non-PIE with `--emit-relocs`**. Both are required: BOLT cannot
  move functions without relocations, and instrumenting a PIE Lua failed on
  `luaV_execute`, its hottest function. Profiles come from BOLT's own
  instrumentation, because `perf` cannot run under `--cap-drop ALL`. By
  default the profile is taken on **every input except the timed one**. The
  measured binary is the profiled binary's own bytes, not a rebuild, because
  BOLT maps a profile by address. There are three arms: the original, the
  configuration under test, and the **standard recipe** as the ceiling, so
  matching the recipe reads as `compiler_already_can`.
  `propose_bolt_configurations` asks a model for configurations one at a
  time, from the hot-function profile and what the recipe achieved.
  `optimize_executable_with_bolt` judges one configuration you choose.
  Options must come from an **allowlist**: they are interpolated into a
  shell, and several BOLT options write files. BOLT's branch statistics are
  reported as the mechanism (Lua: taken branches -89% to -93%), but the
  verdict rests on timing. On Lua that effect is about 4% (fastest run 250 to
  243 ms in a quiet prototype). With the host at load 58 to 72, every
  configuration came back `unresolved`, which is correct.
  Also fixed in the shared comparison: a timed run that *fails* is no longer
  recorded as a time. A prototype "timed" a binary that was never built at
  3 ms, beside the real one at 250 ms. A candidate that passes every
  differential run and then fails a timed one is `crashed`.

- **A speed verdict is paired, rotated, and checked against a control that
  runs inside the same measurement.** This lives in the comparison
  (`agent_restructure.judge_speed` / `run_comparison`), which every
  restructuring, binary-rewrite, pass and BOLT tool shares. Three steps, each
  forced by a measurement that lied without it.
  *Paired*: arms are interleaved trial by trial, so the verdict uses the
  median of per-trial ratios, with a distribution-free 95% interval from
  order statistics (x(4)..x(12) at 15 pairs; the unpaired rule below 6).
  Comparing each arm's fastest and median separately had to clear the whole
  host's spread, and a ~4% BOLT effect sat under a 15-80% band.
  *Rotated*: trial t starts at arm t mod k. With the baseline always first,
  the same held-out BOLT recipe read "slower" (CI [0.776, 0.995]) and then
  neutral on an identical run.
  *Controlled*: a byte-identical copy of the baseline is timed alongside the
  other arms, and a candidate's interval must clear the control's, not just
  1. With the host at load 59, one of six A/A runs of identical binaries
  still read "faster" at 1.037x, because bursty load breaks the independence
  the interval assumes. With the control, six of six A/A runs were
  unresolved at load 44-95. Real wins survived: raylib blur 1.31x
  (CI [1.21, 1.48]), fastmod 2.10x. A win still has to clear the 3%
  code-placement floor. Under this host's load, BOLT on Lua **never
  resolved**: an in-sample "8%" did not reproduce, and a held-out profile
  never showed a gain. That is the finding, not a failure to find one.

- **A layout claim is a claim about a predictor, so BOLT can also be judged
  in simulated cycles on a named core** (`measure="cycles"` on
  `optimize_executable_with_bolt`, default `NeoverseV2`). Wall time could not
  resolve BOLT on Lua on this host. Cycles are deterministic and could, and
  the generic model told a very different story from the named one:

  | core (conditional predictor) | recipe speedup | mispredicts |
  |---|---|---|
  | `O3CPU` (TournamentBP) | 1.307x | -72% |
  | `NeoverseV2` (TAGE-SC-L 64KB) | 1.026x | -0.4% |

  Nearly all of the generic core's gain was a weak predictor being rescued
  by layout. What remains on a modern predictor is 19% fewer i-cache misses,
  worth about 2.6%, below this host's wall-clock noise, which is why nothing
  resolved natively.
  Three traps are handled in code:
  - BOLT cannot flush an instrumentation profile from a static binary at
    exit, and a BOLT-ed static glibc binary aborts with "Unexpected reloc
    type in static binary" (IRELATIVE). So the dynamic non-PIE build is
    simulated; the gem5 and BOLT images carry the same glibc 2.36.
  - NeoverseV2 cannot execute scalar `fmadd`, and Lua contains some. The
    tool counts the fmadd family in every arm and refuses with the remedy
    (`-ffp-contract=off`) before spending a simulation that would never end.
  - `profile_run_args` sizes profiling separately from the measured run.
    Simulation wants a small run and a profile wants a large one. Profiling
    at the simulated size (n=300) turned the 2.6% gain into a 6.8% loss, with
    mispredicts +134%.

- **A mechanism that never engaged is not a measurement of that mechanism.**
  `evaluate_across_kernels` reported `geomean 1.0000x over 4 kernels` and
  recorded it as a `mechanism_evaluation` finding a contract accepted. The
  cause: an `IrregularStreamBufferPrefetcher` on L2 reports
  `pfIdentified = 0, pfIssued = 0` -- it is instantiated, it has counters, and
  it prefetches nothing -- so its arm matched the no-prefetcher run *to the
  cycle* on every kernel. A `StridePrefetcher` on the same kernel issues
  63,127 with 4,171 useful.

  The dangerous form is not the 1.0000. Run the same inert mechanism against a
  baseline that *does* prefetch and the null becomes the baseline's own gain,
  inverted and attributed to the mechanism: a study concluded "ISB is a worse
  general-purpose accelerator, geomean 0.78x, worst 0.50x" when 1/2.1028 =
  0.4756 is simply Stride's measured 2.1x seen from the other side. Every
  number was real and the conclusion was about a mechanism that never ran.

  `inert_prefetchers()` reads the mechanism's **own counters**, not the
  cycles: two arms can coincide honestly, but a prefetcher reporting zero
  identified candidates has said itself that it never engaged. A mechanism
  inert on every kernel is refused rather than reported. A build that does not
  publish the counters accuses nobody -- silence is not evidence of idleness.

- **A contract is read under every spelling it is written in.** A contract may
  name its evidence as `required_finding_types` (a list, or a mapping of
  counts) or as the normalised `required_finding_type_counts` — which is the
  key `agent_pipeline_spec._required_types()` reads *first* and the one the
  pipeline authoring path emits. The runtime normaliser
  (`_get_goal_contract_config`) read only the former, so a contract written the
  other way survived validation, reached the job config intact, and was then
  normalised to **nothing required while still `enabled`**. An enabled contract
  that requires nothing is satisfied on the spot: the stage autocompleted at
  progress 100 having produced none of its evidence, and the log said
  "deterministic goal contract satisfied". Measured on a live stage contracted
  for three `fusion_candidate` findings that completed with zero. The evaluator
  was right throughout — it was asked the wrong question, which is why
  `tests/test_goal_contract_key_spellings.py` pins the normaliser rather than
  the evaluator.

- **Pipelines** — a DAG of stages in `services/agent_pipeline_spec.py`, each stage a goal contract; tools are *derived* from the contract rather than named. A stage must declare a `job_type` its tools are allowed to run under: every coding tool is restricted to `analysis`/`coding` and the default is `research`, so a coding stage left at the default is planned with `clone_and_index_repo, apply_patch, run_repo_tests` and then cannot see one of them at runtime — measured, eight iterations of `search_documents` while the plan promised a repository fix. `validate()` now refuses that and names the job types that would work. Contract `validity.bounds` are checked on the **latest** finding of a *perishable* type, which is what makes "end with the tests passing" expressible: bounding every `test_result` at `failed == 0` is unsatisfiable, since the baseline run that finds the bug is red by definition, while requiring only that a `test_result` exists is satisfied by a red one. `"latest": false` restores the check-every-occurrence behaviour
- **Workflows** — visual workflow builder (ReactFlow frontend, Zustand store), `workflow_engine.py` execution, workflow→synthesis conversion, LangGraph-based issue/PR graphs (`langgraph_issue_pr_service.py`).
- **Campaigns** — a line of enquiry that outlives any one job
  (`services/research_campaign_service.py`, beat task every 5 minutes, page at
  `/campaigns`). A campaign holds a goal, a list of questions and a job budget;
  each tick launches at most one job, because the reason to have a campaign
  rather than a batch is that each result changes what is asked next. Findings
  of the declared types spawn further questions (`origin='discovered'`), and a
  line that twice produces nothing is abandoned. It **concludes at both
  endings** using the same machinery a single run uses
  (`agent_run_synthesis_service.conclude`): if what was collected does not
  answer the goal, saying so is the output rather than a sentence that sounds
  like an answer. Running out of budget is recorded as a gap, because "we did
  not find out" differs from "we found out that no". `conclusion` is the line a
  list shows; `conclusion_detail` keeps the evidence, gaps and confidence
  beside it. The detail endpoint returns the campaign's items, the list does
  not — loading every question of every campaign to render rows that show none
  of them is a query per campaign for nothing.
- **The operator queue shows three different things.** An `approval_checkpoint`
  is a proposal awaiting sign-off; a `blocked_run` is a run that stopped
  because it needs a person and has nothing to propose
  (`current_phase='blocked_needs_input'`, set by `agent_runtime_finalizer` when
  a loop ends with its contract unmet). Only the first had a projector, and the
  recovery branch beside it requires `schedule_type` to be recurring, so a
  one-shot blocked run appeared in no queue at all — six were found waiting 8 to
  11 days, one of them a pipeline stage, which is the whole DAG behind it
  stopped with nobody told. The row carries the sentence the run recorded about
  why it gave up and the `missing` list it could not satisfy, because that is
  what the person answering it needs; Resume is offered only when the run
  recorded `resumable`, since an action that would fail is worse than none.

  A `contract_unmet` row is the third: a run that **completed** without
  satisfying its contract. A run that gives up early pauses and is visible; one
  that exhausts its iteration budget with the contract unmet is marked
  `completed`, so the worse outcome carried the better-looking status and
  nothing surfaced it — 28 such runs in a fortnight, each reporting success
  while delivering nothing the contract asked for. It ranks below the other two
  (`priority` 60) because nobody is waiting on it: it is a quality signal about
  work already reported as done. It offers `restart` (which resets `iteration`
  and `progress`, so the run gets its budget back instead of re-hitting the cap
  it just hit) and `relaunch`.

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
- **Whether a user is an admin is decided in one place:**
  `auth_service.is_admin(user)` (None-safe), with `ensure_admin(user, detail)`
  for a handler that has one admin-only branch and `Depends(require_admin)`
  for a route that is admin-only throughout. It used to be decided in about a
  dozen ways — the model's method, raw role-string comparisons, three private
  helpers, a second `require_admin` — and one variant named the method
  without calling it, which is always truthy and left three pipeline routes
  open to any signed-in user. `tests/test_one_admin_check.py` refuses the
  other forms, including the uncalled method. `get_current_user` is imported
  from `auth_service`; `get_current_active_user` (in `endpoints/auth.py`) is
  a documented alias of it and is left alone.
- **A route with no auth dependency is public, and nothing fails.** Twelve
  knowledge-graph routes and the document editor's read *and write* declared
  no user: `GET /kg/stats` answered anyone, and `PUT /documents/{id}/edit`
  would overwrite a document for whoever knew its id. Three progress streams
  (presentations, repo reports, workflow executions) accepted any socket that
  knew a job id. `tests/test_every_route_requires_a_caller.py` tests the
  absence: an HTTP route with no auth dependency must be listed in `PUBLIC`
  with its reason, and a WebSocket — which cannot use the HTTP dependency —
  must authenticate in its handler (`websocket_auth.authorize_owner` for a
  stream about one user's job; a stranger gets 4004, not "forbidden"). This
  covers *whether* a route knows its caller, not whether it checks the row
  belongs to them.
- **A job's progress reaches a socket through one loop**
  (`utils/websocket_progress.forward_progress`). Five handlers each had a
  copy, and each got a different part wrong under conditions a manual test
  does not create: two released their Redis connection *after* the loop
  rather than in a `finally`, so a client leaving or a job already finished
  leaked one; three waited on `pubsub.listen()` alone and could not notice a
  client leaving, so a socket on a job that never finishes held its handler
  until restart; and the template stream polled the **blocking** Redis client
  inside the event loop, stalling every other request for up to a second at a
  time while anyone watched. The template stream also checked the token and
  not the job's owner. A handler now decides who may watch and what the first
  message says; the loop subscribes before sending it, forwards until a
  terminal message, checks on the client between polls, and releases
  everything on every exit. The agent-job stream (already built this way,
  with injected edges) and the per-user notification feed keep their own.
- **A job task reports and fails through `tasks/job_support.py`.** Some
  thirty publishers across twelve task modules each opened a Redis client per
  message; the synchronous ones never closed it and the async ones closed it
  on the line after `publish` rather than in a `finally`. `publish_sync` and
  `publish_message` take the message whole (a module keeps its `_publish_*`
  wrappers, which only name the channel and payload); `publish_progress` is
  the plain progress shape; `mark_job_failed` records a dead task without
  rewriting a job that already ended. The agent-job task keeps its own
  failure write, because that one must respect the execution lease.
  A user's LLM preferences are likewise read in one place,
  `llm_service.load_user_llm_settings(db, user_id)` (never raises; UUID or
  string). The inline lookups left are counted in
  `tests/test_load_user_llm_settings.py` and may only decrease.
  **A shared helper does not mean a shared constant**: `_canonical_params` is
  one function, but a failure and a repeated success ignore different params,
  so each caller passes its own set. Sharing the set made two successful
  calls differing only in `title` count as a repeat.
  The same reason keeps `agent_decision_parser.coerce_bool` (a model's
  output: true/yes/1/y) apart from `config_values.coerce_bool` (a person's
  config, which also accepts on/off), and `workflow_tasks.run_async` (a fresh
  loop per call, closed after) apart from `job_support.run_async`.
- **A backlog item's `decomposition` is edited through
  `services/coding_backlog_decomposition.py`.** The operator endpoint and the
  backlog orchestrator write the same JSON document, and the orchestrator
  carried its own nested copies of the eight helpers that append to it — so
  the history caps (100 backlog events, 60 per slice, 40 artifacts, 12
  decisions) were stated twice. **The two `_normalize_decomposition` functions
  are still separate and do differ**: the runner's rebuilds every slice and
  reads the legacy `slices_planned` key, the endpoint's keeps unknown slice
  fields. Merging them is a behaviour decision, not a cleanup.
  `services/config_values.py` holds the clamped config readers the executor
  and two policy modules each redefined as closures over `cfg`
  (`clamped_int`, `clamped_float`, `string_list`) plus the never-raising
  number readers a dozen services each had (`safe_float`, `safe_int`,
  `positive_int`, `as_number`, `uuid_list`), and
  `connectors/repo_tree.RepoTreeMixin` what the GitHub and GitLab connectors
  both do with a tree. Three autonomy route modules shared a copied ownership
  query; it is `modules/autonomy/api/owned_job.get_owned_job` now, since that
  query is the whole of those routes' ownership check.
  `tests/test_shared_helpers_have_one_definition.py`
  names the one module allowed to define each.
- **An import inside a function, under `except Exception`, can name
  nothing and nobody finds out.** Four did: synthesis output files imported
  three builder singletons that never existed, so every DOCX/PDF/PPTX
  synthesis completed with no file; bulk summarisation queued a task under a
  name it never had; and the GitLab architecture tool imported a model module
  that does not exist from a service that could not itself be imported, which
  also took four `/git` routes with it. `tests/test_imports_resolve.py` checks
  every `from app.x import name` against the source of `app.x` — from source
  rather than by importing, so a missing optional dependency cannot hide a
  module. The same file checks one step further on — that a method called on
  an imported singleton, or on a service held as `self.x = X()`, exists on
  that class — which found agent delegation calling
  `LLMService.generate_chat_response` (never defined; every
  `delegate_to_agent` answered "Delegation failed") and job finalisation
  calling `DataSandboxManager.cleanup` (never defined; no data sandbox was
  released at job end). It asserts it examined more than 500 uses, since a
  check that examines nothing passes for ever.
  `tests/test_calls_fit_signatures.py` is the third of the kind: a call passes
  arguments its callee can take. Python checks that when the call happens,
  which is too late inside `except Exception` and too late for `.delay()`,
  where the TypeError is raised in a worker after the caller answered 200.
  `POST /agent/agents` passed a keyword its validator lacks, so no agent could
  be created through the API (and `routing_defaults` was never stored); the
  research presentation route queued its task without `user_id`, leaving the
  job pending for ever. All three tests skip any target that is rebound,
  shadowed or decorated rather than guess.
  `tests/test_model_fields_exist.py` is the fourth, and its trap is the
  quietest: **`Model(metadata={...})` never raises.** Every mapped class has a
  `metadata` attribute (the table registry), so SQLAlchemy accepts the
  keyword, sets a plain instance attribute and writes NULL. Two models name
  their JSON column "metadata" in the database and map it as
  `extra_metadata` / `proposal_metadata`; every code patch proposal lost what
  it recorded about itself and every document chunk its section title.
  `format_as_report` with `persist` was dead three ways at once (a service
  method that does not exist, a keyword `Document` lacks, two required
  columns missing) and reported success with `document_id: None`.
  Two things found while
  repairing those: the GitLab service is a
  singleton that cached one HTTP client **with the first caller's token** (now
  one client per token), and the agent tool picked the first active GitLab
  source for anyone (now the `/git` endpoints' rule: an admin, or whoever
  requested the source). The synthesis PPTX path is repaired but only
  shape-tested, since `pptx` is stubbed in tests.
  A task polls Redis for "cancel" through `job_support.flag_is_set`, and clears
  its keys through `delete_keys`; ingestion opened a client per document and
  training one per step, closing none.
- **A test must call the thing it is named after.** About 730 of 4,700
  backend tests never touched application code: whole `tests/test_*_tools.py`
  files restated a handler inline (`title = str(params.get("title",
  "")).strip(); assert not title`) and asserted on the restatement, so they
  passed whatever the tool did. Two files have been rewritten against the real
  handlers (`test_output_formatting_tools.py`,
  `test_agent_code_execution_tools.py`), and the second paid for itself at
  once: `write_and_run_script` handed its script to the container as stdin and
  then ran `python /workspace/<name>`, a file nothing had written; chained
  `pip install` in front with `&&` in a container with no network; and pasted
  input JSON and arguments into the shell line. Underneath it,
  `docker_tool_executor` never passed `-i`, so **no custom Docker tool ever
  received its stdin** — the default input mode — and wrote the input file as
  `input.txt` whatever `input_file_path` said. To reach a real handler:
  `build_*_provider(executor)._handlers[name](params, ctx)`. The Docker fixes
  are tested at the command and file level only; nothing was run against a
  daemon. Eight more files followed (knowledge graph, snapshots, media,
  scheduling, notifications, document authoring, batch, analytics) and every
  one but output formatting found something. Tools that had **never worked**:
  `create_kg_relationship` (and `link_entities`, and `POST /kg/relationship`)
  inserted a NULL into a NOT NULL column — migration `0106` makes a
  relationship nobody extracted from a document legal; `schedule_job` built
  its job without the required `name`; `transcribe_document`, `analyze_image`
  and `get_media_info` filtered on `Document.user_id`, a column that does not
  exist. Tools that worked and lied: the analytics tools reported every tool
  at a 100% success rate, because the execution log recorded a call the same
  way whether it failed or not (the iteration entry now carries `success` and
  `error`) and counted an operator's "approve" as a tool named `approve`;
  snapshots were all stamped iteration 0, since the iteration is the job's and
  not in runtime state. A run may now leave at most
  `MAX_SCHEDULED_JOBS_PER_RUN` (10) scheduled jobs and
  `MAX_NOTIFICATIONS_PER_RUN` (20) notifications behind it. The last thirteen
  files followed and all found something too. The shapes worth remembering:
  - **A check made after the act.** The web fetcher let its HTTP client follow
    redirects and validated only the final address, once the body had been
    read: a public URL redirecting to `169.254.169.254` withheld the content
    and still sent the request. Redirects are followed one hop at a time and
    each is checked first.
  - **State written "for future reference" that nothing reads.** Reflections,
    hypotheses, the evidence ledger and plan critiques are now rendered into
    the volatile prompt (`agent_prompt_sections.format_reasoning_notes`);
    findings shared with a child went to `inherited_findings`, a key no code
    read, instead of `inherited_data.parent_findings`, which the child's
    prompt does.
  - **A snapshot nobody invalidates.** `assembled_markdown` survived a
    revision, so an export shipped the text from before it.
  - **A search that ignores its query.** Memory search sorted by importance
    alone; it is lexical now and says so (`relevance` on each result), not
    semantic.
  - **`x or default` eats zero.** Importance 0.0 became 0.5, confidence 0.0
    became 0.8, threshold 0 became 1, `keep_last` 0 became 5.
  - **A parameter declared and never read** is removed rather than left:
    `write_section.search_query`, `export_document.latex_project_id`,
    `request_review.reviewer_job_id`, `transcribe_document.language`.
  `export_document` now stores the file it builds (it used to measure it and
  drop it), derives its slides and its LaTeX from the same parsed items as the
  DOCX and PDF (`services/markdown_latex.py`), and honours
  `LATEX_COMPILER_ENABLED`, which that branch had never consulted.
  Following up what those reads had only suspected found four more.
  **No workflow execution stored its node outputs**: the engine writes into
  the dict it captured from `execution.context` at the start, and the
  cancellation check refreshes the row between nodes, giving it a new dict;
  later nodes read the captured one, so workflows ran correctly and persisted
  only their initial context (`WorkflowEngine._context_changed` now stores
  what the run wrote; a parallel branch's copy is left alone until merged).
  `MemoryService.update_memory` took no owner, so `PUT /memory/{id}` rewrote
  anyone's memory. `generate_memory_summary` treated `(memories, total)` as
  the list. The finaliser replaced `job.results` wholesale and with it the
  messages and shared findings other jobs had delivered. And
  `request_review` with `human` answered "paused" after setting a flag the
  executor clears before its next action: it notifies the owner now and says
  the run continues, because a real pause belongs in the executor's action
  path and is not built.
  `summarize_findings` with `consolidate` folds only **untyped** findings: a
  finding with a `type` is evidence a goal contract counts in that same list,
  and replacing them all with one synthesis let a run un-satisfy a contract it
  had met. A child job is **committed before it is queued** — delegation,
  peer review and handoff flushed the row and called `.delay`, and a worker
  in another process cannot see a row that has only been flushed.
- **A setting must be read, or admitted inert.** `TRAINING_ENABLED` was the
  documented gate on training and nothing read it; the concurrency limit and
  two dataset limits beside it were the same. They are enforced now (the gate
  as a router-level dependency, counted across users for concurrency since
  the machine is what is rationed). A field cannot simply be deleted — a
  deployed `.env` still naming it makes `Settings` refuse to load — so the
  remaining eight are marked `NOT READ` in `config.py` and listed in
  `tests/test_settings_are_read.py`, a list that may only shrink.
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
- Implementing an algorithm and measuring it is one chain, and the middle link is a correctness gate: `check_implementation` (`services/agent_implementation_check.py`) runs the code against reference cases from the paper before `benchmark_c_snippet` times it, because the fastest implementation of any algorithm is one that returns garbage. It is `perishable`, so a looping implement stage cannot inherit a verdict it earned before the edit, and a check with no cases reports unverified rather than passing vacuously. `compare_to_claim` (`services/agent_claim_comparison.py`) scores the result against the paper's number with three verdicts, not two — `incomparable` is the honest answer when units, hardware or input size make the two numbers untestable against each other. Build recipes live in `services/agent_toolchains.py`: **C, Rust and Python**, each from a single self-contained file; the `language` enum in the tool schemas is read from that table rather than restated, so a language it can build is never one the model is refused for naming. Python's "compile" step is `py_compile`, which is what makes a syntax error report as a build failure instead of as every reference case failing — and because `./prog` is timed as a whole process, an interpreted language reports `interpreter_startup_ms` measured in the same container, with a warning when startup dominates (measured: a 15 ms Python run was 8 ms of CPython booting). Both the checker and the benchmark take `language` and share that table, so the binary that was verified is the binary that was timed. Rust gets **crates without a network**: `RUST_CRATES` (rand, rand_chacha, rayon, ndarray, num-traits, num-complex, itertools, pinned exactly) is built into the compiler-research image at *build* time and linked as prebuilt rlibs named by `/opt/rust-deps/externs.txt`, which the compile line reads — so the image stays the authority on what exists and older images degrade to no-crates rather than failing to compile. The image derives its Cargo manifest by importing `agent_toolchains.py` (`deploy/sandbox-images/compiler-research/gen_crate_manifest.py`), so the list the model is told about and the list the image builds cannot drift; keep that module stdlib-only or the image build breaks. Three rustc traps are handled in code because each fails in a way that blames the wrong thing: it defaults to **edition 2015**, where `use some_crate::X` does not resolve at all (added via `enforced_flags`, only when the caller names no edition, since rustc rejects a repeated `--edition`); it invokes `cc`, which the sandbox lacks, so the linker is pinned to clang in the template where overriding flags cannot drop it; and it rejects `-O2`, defaulting to an unoptimised build that times the debug binary
- Wall-clock measurements report the machine they were taken on: `benchmark_c_snippet` samples `/proc/loadavg` and `nproc` in the same container as the trials and returns `load_per_cpu`, `measurement_environment` (quiet/busy/saturated) and `trial_spread`, warning when the host was busy or the trials unstable. Carried on the finding as well as the result, so `validity.bounds` can refuse a run whose numbers were taken on a machine too busy to measure anything. Only the wall-clock tool needs this — simulated cycles are the same on a busy host as a quiet one
- Job chaining: parent/child jobs with configurable trigger conditions (`on_complete`, `on_fail`, `on_findings`)
- Swarm consensus is typed, not textual (`services/agent_swarm_consensus.py`): roles' findings are grouped by finding type and subject, and compared on their numbers. Four verdicts rather than two, for the same reason `compare_to_claim` has three — **contested** (they disagreed), **corroborated** (they agreed within a tolerance narrow enough to mean something), **inconclusive** (they measured the same thing through an instrument too imprecise to resolve it) and **uncorroborated** (only one role spoke to it). Tolerance comes from the claims' own reported spread, which is right until the spread is enormous: the first swarm whose roles both benchmarked reported `agreement 1.0` over two measurements taken on a `saturated` host, where the 142% tolerance would have admitted any two numbers. Above `MAX_USEFUL_TOLERANCE` (50%) a pair that falls *inside* the window is `inconclusive`; a gap that exceeds it is still `contested`, because a disagreement surviving a generous window is the most confident verdict there is. `inconclusive` is out of the `agreement` denominator — it is not evidence either way
- A swarm's merged verdict can stop for a person (`services/agent_swarm_review_gate.py`, `AGENT_SWARM_REVIEW_GATE`, per-job `config.swarm_review_gate`): `never`, `on_dispute` (default) or `always`. `on_dispute` holds on contested, inconclusive, an incomplete swarm, or a merge that cross-checked nothing, and lets a clean corroboration through — a gate met on every clean run is one people approve without reading. The hold writes the ordinary `approval_checkpoint` payload, so the queue, the panel and the approve/reject actions apply unchanged. Both this gate and the older chain-approval gate are applied on the **deterministic-runner path too**, which returns before `finalize_job` and so saw neither: the fan-in aggregator is itself a deterministic runner, making the swarm verdict the one job the gate could not see, and an `on_approval` chain on any deterministic stage was accepted, stored and silently dropped. Releasing it fires the chain with `approval` **and then** `complete`: the gate fires on the verdict, so it may be holding a job whose chain answers either event, and sending only one would strand the other's stages behind a parent marked done
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
- Heavy optional dependencies (sentence_transformers, bs4, croniter, mammoth, jsonpath_ng) are stubbed in conftest; `pptx` is stubbed **only when it is not installed** — it used to be stubbed whenever nothing had imported it yet, which was always, so no test built a real presentation and two broken PPTX paths passed — — sentence_transformers is no longer installed at all, so that stub is now the only thing that module means in tests — don't import them at module top-level in code paths tests touch without checking the stubs
- FastAPI dependency overrides replace `get_db` with test session
- User fixtures: `test_user` (regular) and `admin_user` with real password hashing; `auth_headers` / `admin_headers` via live token creation
- Async tests use `pytest-asyncio` (auto mode); markers: `unit`, `integration`, `slow`
- Coverage gate: 48% backend (`make test-backend-coverage`) — that is the measured floor, meant to ratchet upward; the suite currently reports 49.55%. The frontend has no `coverageThreshold` configured, so `npm run test:ci` collects coverage without enforcing it

## Commit Style

Follow Conventional Commits as seen in history: `fix(ui): ...`, `feat(admin): ...`, `style(ui): ...`

## Environment Configuration

Backend configuration is in `backend/.env` (copy from `env.example`). `core/config.py` has 150+ settings; major groups:
- `DATABASE_URL`, `REDIS_URL`, `DB_*` - Database connections and pooling
- `LLM_PROVIDER` - `deepseek` (default), `openai`, `anthropic`, `qwen`, `kimi`, `glm`, or `ollama`; `OLLAMA_BASE_URL`, `DEFAULT_MODEL`, `DEEPSEEK_*`, `OPENAI_*`, `ANTHROPIC_*`, `QWEN_*`, `KIMI_*`, `GLM_*`
- `RAG_*` - RAG pipeline (hybrid search, reranking, MMR, dedup, KG context, chunking)
- `VECTOR_STORE_PROVIDER` + `QDRANT_*` / `CHROMA_*`
- `MINIO_*` - Object storage
- `WHISPER_*`, `TRANSCRIPTION_*` - Transcription and diarization; `TRANSCRIPTION_CELERY_QUEUE` names the queue the dedicated worker consumes
- `LDAP_*` - Optional LDAP/AD authentication
- `LATEX_COMPILER_*` - LaTeX compilation (disabled/admin-only by default)
- `UNSAFE_CODE_EXEC_*`, `SCIENTIFIC_VALIDATION_*` - Sandboxed code execution limits (subprocess or Docker)
- `TRAINING_*`, `AI_HUB_*`, `DATASET_MAX_*` - Fine-tuning and evals
- `AGENT_REQUIRE_TOOL_APPROVAL`, `AGENT_DANGEROUS_TOOLS`, `AGENT_KB_PATCH_APPLY_ENABLED` - Agent governance
- `PLUGIN_BUILTIN_DIR`, `PLUGINS_USER_AUTHORING_ENABLED` - Plugin bundles shipped
  with the repo, and whether users may author their own
- `SANDBOX_SKILLS_AUTHORING_ENABLED`, `SANDBOX_SKILL_IMAGE_BUILD_ENABLED` - Whether
  users may author sandbox skills, and whether an admin may build images for them
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

Variants: `docker-compose.prod.yml` (gunicorn, healthchecks) — `celery_beat` runs in the dev stack too, and it is what makes `check_stalled_agent_jobs` fire there: with no beat, a job whose worker died (typically from restarting the celery container) stays `running` for ever until that task is invoked by hand, which requeues it to resume from its last checkpoint rather than failing it. The same sweep also covers the opposite failure: a job whose row was committed but whose Celery task never arrived — a broker restart, a purged queue, an enqueue that failed after the commit — sits in `pending` with no task id and no activity, which the running-job query could not see, so two were found in a live database 14 and 3 days old. Re-delivery is safe because the execution lease decides who runs: a duplicate returns `lease_conflict` without executing. A job holding a live lease, one already claimed, and one with a `schedule_type` are all left alone — firing a recurring job early is a wrong run, not a recovery), `docker-compose.test.yml` (isolated test stack on shifted ports), `docker-compose.docker-tools.yml` (mounts Docker socket for Docker-based tool execution).

Access points:
- Frontend: http://localhost:23000
- Backend API: http://localhost:28000
- API Docs: http://localhost:28000/docs
- MinIO Console: http://localhost:29001

CI runs in `.github/workflows/ci.yml` on pull requests and pushes to `main`: backend lint/format, backend tests with the coverage gate, a single-alembic-head check, frontend typecheck plus tests, a `helm-chart` job that lints every values profile, validates the rendered manifests with kubeconform, and parses the generated nginx configs, and a `helm-smoke` job that installs the chart on an ephemeral kind cluster and asserts the wiring rendering cannot check (hook ordering, Secret-to-URL assembly, gateway routing, migration-gated upgrades). Lint is gated on `app/` and `tests/` only — `alembic/`, `scripts/`, and `seed_data/` carry pre-existing formatting and flake8 debt that is reported but not enforced. The same checks are available locally as Makefile targets (`make lint`, `make fmt`, `make test-backend-coverage`, `make typecheck-frontend`), which shell into a running Docker stack.

## Other Documentation

Root-level docs worth checking before larger changes: `BUILD_AND_RUN.md`, `QUICK_START.md`, `DOCKER_SETUP.md`, `deploy/README.md` (Kubernetes/Helm), `AGENTS.md`, `AUTONOMOUS_RND_AGENTS.md`, plus `docs/` for architecture guides. Current visual architecture map (deployment, subsystems, agent runtime, LLM stack): `docs/ARCHITECTURE_DIAGRAMS.md`.
