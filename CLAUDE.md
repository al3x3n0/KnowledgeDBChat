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
make logs-celery-agents # View the agent-job worker's logs
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
make sandbox-images     # every image this repo can build (base, compiler, polyglot, microarch)
make sandbox-polyglot   # C + Rust + Python and nothing else; no crates, so not the default
make sandbox-load       # copy what was built into the stack's sandbox daemon (see below)
make sandbox-check      # which exist locally, the compiler image's toolchains, and what the app's daemon lacks
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

**Sandboxes run in the stack's own Docker daemon, `sandbox-docker`**
(`docker:27-dind`, privileged), not the host's. Backend and celery reach it over
TLS through `DOCKER_HOST=tcp://sandbox-docker:2376`, with client certs from the
daemon's own volume, and never see the host socket. Every sandbox path shells out
to the `docker` CLI, so `DOCKER_HOST` redirects all of them. Two consequences
follow, each found by running it:

- **Its image store is its own.** An image `make sandbox-*` built on the host is
  invisible to it, and the `ghcr.io/al3x3n0/kdbc-*` images are not published to
  pull. `make sandbox-load` copies each locally built image that differs. It
  compares layers and creation time, not image ID: a host on the containerd
  image store reports the manifest digest as the ID and dind the config digest,
  so one image's IDs never match. A missing image is what the overlay era's
  missing socket used to be: silence in the evidence, a capability reading as
  0 findings. `make sandbox-check` names what the app's daemon lacks.
- **The work directory is a volume mounted at `/sandbox-work` in all three
  containers**, and `TMPDIR` points there, because `-v <dir>:/work` is resolved
  by the daemon's filesystem. It must not be under `/tmp`: the dind entrypoint
  mounts a tmpfs over its `/tmp`, which hid a volume there and gave every
  sandbox an empty, unwritable `/work`.

**Production has the daemon as an opt-in, in both shapes.** The prod compose
file and the chart had none, so every sandbox tool answered "could not run"
there. `docker-compose.sandbox.yml` overlays `docker-compose.prod.yml` with the
dev stack's wiring. The chart's `sandbox.enabled` adds the daemon as a
**sidecar** to the backend, `celery` and `celery-agents` pods, reached over a
unix socket in an `emptyDir`: a daemon shared between pods would need the work
directory on ReadWriteMany storage, since `-v` resolves on the daemon's
filesystem. So the image store and work directory are per pod — a restarted
pod pulls again, and a stage on another pod than its parent inherits no files.
Off by default because the container is privileged. The sidecar has a
liveness probe and no readiness probe: a dead daemon must not take the API
out of service. `tests/test_sandbox_daemon_in_every_deployment.py` checks the
shared paths; the sidecar is rendered and schema-validated, **not run on a
cluster**.

`docker-compose.docker-tools.yml` is now the opt-in alternative: it points the
same CLI at the host socket (clearing `DOCKER_HOST`), binds a host temp path
at the same location inside and out, and enables `CUSTOM_TOOL_DOCKER_ENABLED`.
The socket grants root-equivalent control of the host, so that is for a
development machine or an isolated runner. Recreate with `--no-build` where the
CLI is already in the image; this network intercepts the package mirrors.

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
  **What `diagram_service` generates is checked against the renderers**
  (`tests/test_diagrams_render.py`). Its builders pasted a model's labels and
  ids into Mermaid and DOT as they came, and 10 of 16 ordinary diagrams did
  not render: a parenthesis in a label broke every flowchart, an id with a
  space or a hyphen (or the word `end`) broke the graph, "dotted" was written
  as `...`, a quote broke DOT. Labels are quoted strings now, unsafe ids are
  replaced consistently, and each form was chosen by trying it in the real
  renderer, which found one more the first list had missed (a colon in a
  state label). `dot` runs wherever Graphviz is installed; the Mermaid cases
  run only with `MERMAID_RENDERER_URL` set (start the image on a spare port),
  so **CI checks the generated text, not that Mermaid accepts it**. The
  Graphviz layout program and output format were taken from config and
  reached a command line; both are chosen from a list.

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
- `types/index.ts` - TypeScript interfaces (very large; mirror backend schemas here).
  **They are checked against the API, not trusted.** `backend/scripts/export_openapi.py`
  writes the schema FastAPI builds to `src/api/openapi.json`, `npm run gen:api`
  turns it into `src/api/schema.ts`, and CI fails if either is stale (backend
  job: `export_openapi.py --check`; frontend job: `npm run check:api`). Both are
  generated -- never edit them by hand. `src/api/__typechecks__/` holds
  type-level assertions that a hand-written response type declares no field the
  matching backend schema lacks; `tsc` fails on drift. It covers the 55
  `AgentJob*` types that have a backend counterpart, and its first run found
  one: `AgentJob.execution_log`, which only `GET /agent-jobs/{id}` returns, so
  `JobDetailPanel` -- usually handed a job from the list -- showed no launch
  log. That field now lives on `AgentJobDetail`, `getAgentJob` returns it, and
  the panel fetches the detail when it was given a list row.
  `__typechecks__/byName.ts` extends it to every type whose name matches a
  schema (290 more). That run found the same shape from the other side: the
  admin dashboard reads `documents.without_summary`, the stats task computes
  it, and `DocumentStatsResponse` did not declare it, so **the response model
  dropped the field** and the line never rendered. A backend test now reads
  the task for the keys it writes and requires the schema to name each.
  Extend the checks area by area; prefer aliasing a type to `components['schemas'][...]` where
  the two already agree.
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
  **Deploying an adapter says what Ollama did** (`model_registry_service`,
  `tests/test_model_registry.py`). Every error from Ollama was logged and
  ignored "for demo purposes", so an adapter was recorded as deployed with
  Ollama unreachable; a failure is `DeploymentFailed` (503) now and leaves the
  adapter `ready`, since `can_deploy` never accepts `failed`. No undeploy
  ever removed its model: `AsyncClient.delete(..., json=...)` is a TypeError,
  which the surrounding `except` swallowed. Deploy and undeploy are the
  owner's only (they went through the lookup that also returns other
  people's public adapters), and a deployment may not take the name of a
  model Ollama already has, because `ollama create` replaces it and
  undeploying deletes it. **The adapter's weights are still not applied**: the
  Modelfile names the base model only, so a "deployed adapter" answers as the
  base model. Not built.
  **An eval counts only what the judge scored** (`ai_hub_eval_service`). A
  judge reply that was not a score from 1 to 5 was entered as a 3 and
  averaged in; it is `judged: false` now and `num_judged` says how many
  counted. The "last eval" summary was edited into the dict the JSON column
  already held and never saved.
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
- **Research programs** — domain research code that lives **outside this
  repository** (`app/agent_core/research_programs.py`; decided 2026-10-19).
  Core is the lab: the run loop, contracts, the sandbox, review, the knowledge
  base. A line of research's own tools (a simulator driver, a binary rewriter,
  the images they run in) are a *program*: a Python package in its own repo,
  pip-installed into the backend and worker images, found through the
  `kdbc.research_programs` entry point. It declares `ToolSpec`s exactly as a
  built-in does, provider factories for their handlers, and its sandbox
  images. **Installed means available**: its tools are appended to
  `TOOL_SPECS` at import, so the model's menu, governance, job-type policy and
  the evidence map see them unchanged (`TOOL_DOMAINS` lists them under the
  program's name); the executor registers its providers after the built-ins;
  its images join `allowed_images()`, since installing the program is the
  decision to run them. A program that will not import, or that declares a
  tool a built-in or another program already has, is refused **whole**,
  logged, and shown by `GET /admin/research-programs` -- it does not stop the
  API. `RESEARCH_PROGRAMS_DISABLED` leaves one out by name. Plugins and
  sandbox skills stay the UI-authored, data-only extension points; a program
  is code and is trusted like the rest of the image. **Core never imports a
  program.** New domain-specific tools belong in a program, not in
  `app/services`.
- **Program identity travels with evidence** — each evidence manifest entry
  from a loaded program records `research_program` (name and declared version)
  at call time, including failed calls. Resuming after an upgrade appends the
  new version without rewriting old entries. Bundle summaries and their README
  read recorded versions rather than the current catalog, so disabling a
  program cannot erase its history. Core tools and legacy entries omit this
  field. Declared versions do not pin Python code bytes; sandbox image IDs and
  artifact hashes retain their separate roles.
- **Evidence across research stages** — inherited findings keep the original
  producer's job id, including across intermediate stages and checkpoint
  resumes. Inheritance is idempotent by producer and the loop policy's claim
  identity. Campaigns count each stage's own distinct claims; inherited copies
  remain available to downstream tools but do not inflate campaign summaries,
  conclusions, or discovery. Independent stages may establish the same claim
  and both still count.
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
  **A fan-in stage starts with every branch's files.** It is created as a
  child of whichever stage it waits on finished last, so the binding records
  `pipeline_depends_on` and the skill tools find the others among its
  parent's siblings (the newest job per stage, so a restart counts). A name
  two branches left is kept from the chain parent; the other copy goes under
  `from-<stage>/`, and the inheritance note says so. One 256 MB budget covers
  all of them.
  **A run's directories live as long as the run.** Pruning used to go by age
  alone, so a finished stage's files were deleted while the next waited more
  than a day on a checkpoint. Now nothing in a pipeline run (every job sharing
  a root) is pruned while any job in it is pending, running or paused; age
  decides only after that. If the database cannot say what is live, nothing
  is pruned.
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
- **A dataset expression a model writes is parsed before pandas sees it**
  (`services/data_expression.check_expression`). `query_data` and
  `transform_data` passed their string to `DataFrame.query` / `.eval` under a
  comment calling it safe. It is not: `@name` resolves against the caller's
  Python scope (`@pd.io.common.os...` runs a command in the worker, which
  holds the database credentials), and an attribute chain reaches the object
  graph and the filesystem with no `@` at all (`a.__class__...`,
  `a.to_csv(path)`). Each evaluated on the real library. The check allows
  columns, constants, arithmetic, comparisons, boolean logic and fixed lists
  of functions and column methods; the call is also made with empty scopes.
  Aggregation names are a fixed list for the same reason: the name is looked
  up on the column. **"Sandbox" in `data_sandbox_service` means per-job
  datasets in the worker's memory, not isolation**; code that must be
  confined runs through `agent_sandbox_runtime`.
- **Tool governance** — `tool_registry.py` + `tool_policy_engine.py` + `models/tool_audit.py`; per-user tool policies, approval gates for dangerous tools (`AGENT_REQUIRE_TOOL_APPROVAL`, `AGENT_DANGEROUS_TOOLS`), full execution audit log, user-defined custom tools (optionally Docker-executed). Tool handlers live one module per provider in `services/agent_tool_providers/` (the registry and provider types in `base`, helpers several providers share in `common`); `agent_tool_dispatch.py` was a 12,000-line module holding all of them and is now a facade re-exporting every name. A test that reads handler source must read the provider modules, and one that patches a module global must patch it where the provider looks it up: three source checks (LLM snapshot coverage, tool effects, the removed 12,000-character cap) look for the *absence* of something and would have passed against the facade while checking nothing. Every tool is **declared once** in `app/agent_core/tool_specs/` (one module per domain): the schema a model reads, the governance classification, which job types may call it, and — for measurement tools — what evidence it produces. `agent_tools.AGENT_TOOLS`, the catalog, the job-type policy and the evidence map are all views of those specs, so adding a tool is a handler plus a `ToolSpec`, not four files kept in step by hand. `tests/test_tool_specs.py` enforces it.
- **A custom Python tool runs in the sandbox container, in the caller's
  working directory** (`services/python_tool_container.py`; decided
  2026-10-13). `python` is a tool type anyone may author, an agent included.
  Its code ran in the server's own process under RestrictedPython: the "safe"
  modules were handed over whole and led to `os`
  (`re.enum.sys.modules.get("os")`), and a loop that never ended kept its
  thread after the timeout stopped waiting. Hardening that was a list of
  doors found so far, so the in-process path is **removed, not kept as a
  fallback**: where the sandbox is off or its image is missing, a Python tool
  says so and does not run. It goes through `run_in_sandbox` like every other
  piece of submitted code (no network, no capabilities, uid 65534, a timeout
  that removes the container), in `CUSTOM_TOOL_PYTHON_IMAGE`, which must be
  allowlisted. Inside it the code is ordinary Python with the standard
  library; `allowed_imports` is no longer read. The contract is unchanged:
  `input` in, `output` or `result` out, through files in `.pytool/<call>/`
  that are removed after the call. **`/work` is the directory sandbox skills
  already use** for that job, conversation or user
  (`agent_sandbox_skill_tools.working_directory`), so a file a skill built is
  there for the tool and what the tool writes is there for the next call.
  `execute_python` and the in-process branch of `execute_data_pipeline` go
  the same way. Tests replace `run_in_sandbox` with a stand-in that runs the
  same command line against the same files, and one test runs a real
  container (uid 65534, no network). **Not run against the stack's own
  `sandbox-docker` daemon**, only the host's.
  `execute_python` takes `include_workspace`: the run's repository is copied
  to `./workspace` first, as the scan tools copy it (the daemon cannot mount
  where workspaces live). It is a copy of the repository as it is, without
  `.git`, replaced on each call, and **nothing written there goes back**: a
  run changes code through `write_file` and `apply_patch`, which a patch
  proposal is built from.
- **A repository copied into the sandbox leaves its symbolic links behind**
  (`utils/tree_copy.without_git_or_links`). A cloned repository can hold a
  link to any path on this machine, and `shutil.copytree` follows links: a
  `notes.c` pointing at the server's `.env` arrived in the sandbox directory
  as a copy of that file, for the tool about to run to read and quote back.
  The scan, restructuring and BOLT tools all copied that way. Keeping the
  links instead is no better, since a tool that writes a file of that name
  writes through it. A test reads every `copytree` call in `app`.
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
- **A run can propose the pipeline that should follow it**
  (`propose_pipeline`, `services/agent_self_planned_pipeline.py`). The step
  from one job to a plan was always a person's: a run could spawn a child
  and a person could draft a pipeline, but nothing let a run say "the next
  thing is these stages, each done when this evidence exists". Three things
  stay out of the model's hands. **It is checked, not trusted**: the draft
  goes through `agent_pipeline_draft.problems_with`, the same validator,
  skill check and chain-expressibility check a person's pipeline meets, and a
  refused plan is handed back with the reasons and nothing attached. **It is
  priced and bounded** (`AGENT_SELF_PLANNED_PIPELINE_MAX_STAGES`, 6;
  `..._MAX_SECONDS`, 4 h), and a plan over either is refused with the figure.
  **It starts nothing**: the bound pipeline becomes the run's `chain_config`
  with trigger `on_approval`, the trigger a checkpoint stage uses, so the
  finished run is held by `hold_for_chain_approval` like any other, shows in
  the queue and on the Review page with its stages and estimate
  (`approval_checkpoint.proposed_pipeline`), and the chain runtime starts the
  first stage only when someone approves. No new approval machinery. Only a
  run that is nobody's stage may propose one (a stage planning its own
  pipeline is a plan that grows a plan), a run may replace its own proposal
  but never a chain someone else configured, and it may ask three times.
  Only a run that ends `completed` offers its plan. **The drafting half has
  been run with a real model; the rest has not.** Four ordinary follow-ups
  were sent through `propose` with the real model and the real checker: two
  of four produced a plan. The raw drafts showed why, and neither cause was
  judgement. A stage listed a stage *and* that stage's predecessor in
  `depends_on` (to say "this uses the paper too"), which the binding reads
  as a fan-in between stages that are not siblings; and `job_type` and
  `assumes` were written inside `contract`, where nothing reads them, so a
  coding stage was planned as research. `agent_pipeline_draft.tidy` now
  drops an implied dependency and moves those two fields, as it already
  moved a misplaced goal; after that the same requests attached five of
  five before the run was cut short. **A job has now proposed a plan on its own, a person approved it, and
  it ran** (2026-10-13, real model, the stack's own worker). Three runs of
  one scoping goal that asked for the follow-up to be left for a person: the
  first called `create_handoff`, which started the work at once; the second
  wrote the plan into a finding and stopped; both had `propose_pipeline` in
  their list of 250. The stable prompt and the critic now carry
  `agent_self_planned_pipeline.prompt_note` (what each way of "planning"
  does), the two spawning tools say they start at once, and the third run
  proposed a four-stage plan while a control with no follow-up proposed
  none. The note depends only on the setting and on being nobody's stage,
  since it is in the prompt that keys the cache. **Approving it was a 500**:
  both gate decisions (chain and swarm review) called
  `append_approval_event` with three positional arguments where it takes
  two and keywords, and their tests passed a stand-in that accepted
  anything, so no checkpoint or swarm verdict could be approved through the
  API since each gate was added. The tests use the real function now. **And no gate decision was ever recorded on the job**: both gates cleared
  the checkpoint, which places the approval payload in `job.results`, and
  then appended the decision to it, but `results` is a `NestedMutableDict`,
  which stores a converted *copy*, so the event went to a dict nothing
  saved. Every other decision records first; the gates do too now, and the
  test reads `job.results` with the real sync instead of a no-op that hid
  the copy. Found by the campaign self-approval tests. After
  the fix the first stage started, spent its ten iterations orienting
  (findings, spec, cloning a repository, browsing it), wrote no file, and
  ended `completed` with its contract unmet; **the three stages behind it
  never started**. A stage that fails still stops the plan. Also seen: a run
  reads `completed` for the minutes its conclusion and memories are written
  and only then `paused`, so look again before deciding a plan was not held;
  and the run's conclusion synthesis returned no content (the model spent
  its 8,000 tokens reasoning), which is not fixed.
  **The tool list says what a required parameter accepts**
  (`_required_choices`): it named parameters and not their values, and four
  runs of four lost their first `save_research_finding` to a category the
  tool does not have. Required parameters with at most eight fixed strings
  are listed as `category*=key_insight|methodology|...`, 548 characters over
  the whole catalog.
- **A stage that fails still stops the plan, but it no longer does so
  quietly or for the wrong reason.** Found by restarting the stage the first
  live plan stopped on (2026-10-14), each on the real stack:
  - **A proposed plan's stages have a stage's budget.** A chained job takes
    any limit its config does not name from its parent, and a proposed
    plan's parent is the run that proposed it: a scoping run allowed ten
    iterations gave ten to a stage that had to clone, write and propose a
    patch. `_with_a_stage_budget` writes a launched pipeline's limits on the
    first stage (100 iterations, and the job model's defaults); later stages
    inherit from it.
  - **The queue says a plan is stopped.** A `contract_unmet` row for a job
    whose chain has not fired names the stages behind it
    (`checkpoint.waiting_stages`, and the summary) and ranks at 80, with a
    blocked run, instead of 60: somebody is waiting after all.
  - **An operator action queues its job after it commits**
    (`job_action_dispatch.queue_job`). Restart, resume, approving a mid-run
    checkpoint and the swarm relaunches changed the row and called `.delay`
    with the commit still to come. The worker took the restart 45 ms later,
    read the row as still `completed`, called it a duplicate delivery and
    dropped it; the job then sat `pending` with a task id and nothing
    coming. The dependency's task is still what is called, so stand-ins
    work as before.
  - **A restarted run forgets that it was stuck** (`agent_loop_policy.
    forget_stall`). Restart keeps the run's findings and kept its round
    counts with them, so the stage was stopped after one round as "3
    consecutive rounds produced no new findings", two of them from its
    previous life. A checkpoint ahead of the job's own iteration is what a
    restart leaves, and is where the counters are dropped.
  With all four, the restarted stage ran seven iterations and then blocked
  honestly: it cloned a repository again (a restart does not keep the
  coding workspace) and read it without writing a file. **No proposed plan
  has yet run past its first stage.** Not built: a stage retrying itself,
  or handing back to the run that planned it.

- **A plan a run proposed for itself is held to an evidence standard**
  (`services/agent_evidence_standards.py`). The first self-approved
  campaign's benchmark stage met a contract asking for one
  `benchmark_measurement` with one trial of one program, and the campaign
  rightly concluded nothing. Self-approval checked size and cost; the
  standard was whatever the drafting model wrote. Before a proposal is
  checked, priced and shown for approval, its contracts gain
  `require_uncertainty` on `benchmark_measurement` and a bound of
  `generated_cases >= 50` on `implementation_verified` (met only by
  `check_implementation` with a reference and a generator). A stronger
  rule the plan wrote is kept, a weaker one raised; one on another field
  is kept and said, since a contract holds one bound per type. The drafter
  is told the standard first, the proposal records what was added
  (`evidence_standards`), and the approval message says what the plan is
  held to. Person-authored pipelines are untouched. **A bound must say what
  its field counts**: told `generated_cases must lie in [50, None]`, the
  next live stage wrote 70 cases by hand in `execute_python`, which that
  field does not count, and stalled. `agent_measurement_validity.
  FIELD_MEANINGS` now explains it where the run reads its contract, and the
  `cases` description no longer recommends a self-checking program that
  prints PASS. **A spread is judged on what was last measured**:
  `require_uncertainty` checked every occurrence, and a run cannot retract
  a finding, so a stage that took one trial and then seven was blocked by
  the one. It now takes the latest finding per subject (so a comparison
  cannot pass with a spread on one side only) and names the bare subjects.
  Run live (2026-10-16, second self-approved campaign): the drafter wrote
  the oracle into the implement stage's goal; its first oracle call had a
  generator that ignored its index (100 inputs, 1 distinct) and was not
  counted; the next compared 100 distinct inputs with 0 disagreements and
  met the contract. The benchmark stage took 1 trial then 7. Re-measuring
  the first under its label, it could not find the source it had timed
  (nothing records it), wrote a harness that did not compile
  (`CLOCK_MONOTONIC` under `-std=c11`) and stalled -- left there.
- **A stage can read what it inherited** (`get_research_findings`). It
  served only `executor._job_findings`, the in-memory list
  `save_research_finding` appends to; findings a tool produced and those
  inherited from earlier stages live in the run's `state["findings"]`, and
  the in-memory list is empty after a restart. The first self-approved
  campaign's report stage held both measurements its parent made, was told
  "no findings" four times, and blocked. It now reads the run's findings
  (saved ones included, once), filters by `finding_type` and the `tags` it
  declared and ignored, lets a measurement with no stated confidence through
  a confidence floor, and returns a count by type. **And a resume that would
  stop at once is refused**: resumed without a note, that stage restored the
  dry rounds it stopped on and stopped again six seconds later while the
  action answered 200. A no-note resume is kept as it was decided (no
  correction, no reason to think the stop was wrong), but when
  `agent_loop_policy.should_stop` already says the restored state stops, it
  answers 409 asking for a note -- which clears that state and reaches the
  run as a clue. Resumed with one, the stage read its findings, wrote the
  report and met its contract in two iterations, and the campaign concluded
  on its own (2026-10-16): four jobs, and an honest "the findings do not
  establish whether fastmod is worth adopting" -- the benchmark stage had
  taken one trial with no `%` baseline, which its contract allowed.
- **Dry-round limits count completed iterations.** The runtime keeps one
  initial finding-count baseline and records each round after its work, before
  checkpointing. Recording only at iteration start hid the latest round from
  the stop check and spent one extra action. A completion that fails after
  recording is not counted again by the error path. The runtime regression
  tests cover empty rounds, repeated claims, and failed finalization.
- **A resumable blocked pause checkpoints its final state.** Tool-boundary
  checkpoints precede evaluation, and a loop can stop before the five-round
  periodic checkpoint. Finalization saves the findings and stop history at
  `blocked_needs_input`, so a no-note resume can detect that it would stop
  again, while an operator correction clears the stall and retains the work.
- **Contract finding counts mean distinct eligible claims.** Both
  `min_findings` and per-type thresholds use the loop policy's finding
  identity. Repeating a tool result cannot satisfy a multi-result contract;
  recalled evidence, reports of absence, and malformed entries cannot fill
  the minimum. Filtering happens before deduplication so a recalled copy
  cannot hide a fresh result. Raw findings stay intact for review and validity
  checks, including checks that deliberately inspect successive measurements.
- **The pipeline drafter is told what each evidence type is**
  (`agent_pipeline_draft._what_makes`). Its list gave a name, a duration and
  job types, so "implement fastmod in C, check it, benchmark it" became a
  stage requiring `code_patch_proposal`: the live stage cloned
  lemire/fastmod, found the function already there, and had nothing to
  change. Each entry now names the tool that makes it, in that tool's own
  first sentence (10.5k characters for the list), and rule 7 says code
  belonging to no repository is written inside the stage that requires
  `implementation_verified`. On the live job's own follow-up: before, every
  draft began with a repository patch stage; after, three of three were one
  stage requiring `implementation_verified` and `benchmark_measurement`. A
  genuine repository fix drafted as a control still required
  `repo_workspace`, `test_result`, `patch_applied` and
  `code_patch_proposal`. That control also found three more misplacements
  for `tidy`: a stage named by `name` with no `id`,
  `required_finding_types` beside the goal instead of in `contract`, and
  `loop: true` with `max_iterations` beside it. Before, two of its three
  drafts were refused even after repair; after, three of three passed on
  the first call. **Then a proposed plan ran to its end** (2026-10-14,
  real model, the stack's worker, host load ~300): the same scoping job
  proposed two stages, implement-and-check then benchmark, with the plan
  approved once. Stage one failed its check twice honestly (no `main`,
  then a harness that printed PASS whatever its input) and passed 3 of 3
  reference cases on the third; only that verified finding satisfied its
  contract. Stage two measured fastmod at 2.25x plain `%` (spread 0.45%
  over three replications; the Docker VM reported its own load as quiet,
  which says nothing about the host's). One defect: the run's goal said
  "once a person agrees", and the drafter made the first stage a
  checkpoint, so the approved plan stopped again for the same approval.
  `ALREADY_APPROVED` now tells the drafter the proposal is that approval.
  A second: **the last stage then started three "Subgoal Follow-up" jobs
  nobody had approved.** A job with an empty chain builds follow-ups from
  its subgoals unless `auto_subgoal_child_jobs_enabled` is off, and a
  pipeline's last stage has an empty chain on purpose. A pipeline stage
  never gets them now (`_ensure_subgoal_chain_config`), whoever launched
  the pipeline; its successors are the pipeline's.

- **`spawn_on` is the one place a pipeline edge does not mean "after".** Every
  other dependency waits for a stage to finish, which cannot describe a stage
  that never does -- a continuous monitor is still monitoring, and a successor
  waiting on it waits forever. A stage may instead release its successors on
  evidence: `spawn_on: {findings: N}`, which the binding emits as the
  `on_findings` trigger the chain runtime already handles, so nothing in the
  runtime changed. It is refused alongside `checkpoint` (one waits for a person,
  the other does not wait at all), on a stage nothing depends on, and on a
  contract requiring no finding types, since there would be nothing to count.
  A stage after a spawning one **can be restarted** while its parent still
  runs: the parent must have reached its threshold (`chain_triggered`), not
  completed, and its flag is left set, or it would release every successor
  again at its next finding. **The threshold used to be ignored**: the
  binding nested it under `trigger_thresholds`, which only chain definitions
  flatten, while `should_trigger_chain` reads the top of `chain_config`, so
  every pipeline spawn released at the default of 10.
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

- **Compiler and microarchitecture research lives in its own repository,
  `kdbc-uarch-lab`** (the `uarch_lab` research program): gem5 simulation and
  mechanism studies, BOLT, application-specific restructuring and its search,
  pass synthesis, binary rewriting, AXIS, fusion mining, profiling and
  predictability -- their tools, handlers, images and tests. Its CLAUDE.md
  carries what was learned building them (the gem5 crash rule, the paired and
  controlled speed verdict, BOLT on a named core, inert mechanisms). The
  speed-judging machinery (`judge_speed`, `run_comparison`) went with it,
  since every caller is there.
- **A deterministic sandbox run is replayed rather than repeated**
  (`services/sandbox_result_cache.py`; adapted from ucb-bar/chia's caching
  and bypass, and its golden results computed once). The gem5 simulations
  (`simulate_c_workload`, and `run_configs` under every mechanism study) go
  through `sandbox_result_cache.run`, keyed on the image's **content id**
  (a rebuilt image is a different simulator; a tag is not enough), the
  script, the limits and the hash of every file in the working directory.
  The files the run wrote come back with it. Only an exit-0 run is stored,
  since a timeout or a missing daemon replayed would make an outage
  permanent; inputs over 64 MB or outputs over 256 MB are not cached, nor is
  anything when the daemon cannot name the image. **Never a wall-clock tool**:
  a test pins the modules allowed to use it, because two honest timings
  disagree. An evidence-bundle replay runs under `bypassed()`, since it
  exists to show the simulator reproduces the number, and
  `replayed_from_cache` is a volatile field so a replay hashes like the
  original. Shared through the work volume, trimmed least-recently-used to
  `SANDBOX_RESULT_CACHE_MAX_MB` (2048; `SANDBOX_RESULT_CACHE_ENABLED` turns
  it off). Run live on the stack: the same simulation twice gave the same
  4,114,705 cycles, `replayed_from_cache` true, 27.4 s then 9.1 s -- the rest
  is preflight outside the simulation, so the saving grows with its length.
  The mark is on `simulate_c_workload`'s data and on each run `run_configs`
  returns; the five studies built on `run_configs` do not yet repeat it in
  their own output. A study whose arms differ shares nothing: the key is the
  whole script, not each arm.
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
- **Workflows** — visual workflow builder (ReactFlow frontend, Zustand store), `workflow_engine.py` execution, CRUD and execution queries in `modules/workflows/application/workflow_store.py` (the endpoint only maps them to HTTP; **a reload after a write passes `fresh=True`** -- sessions do not expire on commit, and `PUT /workflows/{id}` answered with the graph it had replaced, so the editor got its own save undone), workflow→synthesis conversion, LangGraph-based issue/PR graphs (`langgraph_issue_pr_service.py`).
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
  **A campaign follows the plan its own job proposed.** A campaign's job is
  nobody's stage, so it may call `propose_pipeline`, and the campaign read
  only that one job: paused for approval it looked "still running" for ever,
  and once approved the stages did the work while the campaign settled the
  question from the job that had merely planned it. An item is now settled
  from its whole run (`_run_of`: the job and every job sharing its root) and
  only when all of it has ended; the stages' findings count towards the
  outcome, the next questions and the conclusion; and each stage is a job
  out of the budget, counted when the run ends. A plan with more stages than
  the campaign has jobs left is refused when proposed, with the figure.
  `waiting_on` (`job`, `approval`, `person`) says why nothing moved, on the
  step and in the summary, and the page says "waiting for a plan to be
  approved" with a link to Review. A stage that has just completed is given
  five minutes to start its successor before its run counts as ended,
  because completing and starting the next are two writes. A campaign
  still asks one question at a time.
  **A campaign may approve its own plans, within limits it is given**
  (`plan_approval: {max_stages, max_minutes}` on create, migration `0110`;
  unset, a person approves every plan, as before). On its tick, an item
  waiting on approval is passed only when what waits is the gate in front
  of a plan the run proposed itself (a tool's mid-run approval and a
  checkpoint the plan declares between its stages are a person's), the plan
  is within the limits, the budget left covers every stage, and the owner
  is active. Otherwise the reason is kept on the item and shown as
  `summary.plan_held`. Limits over `CAMPAIGN_PLAN_APPROVAL_MAX_STAGES` (4)
  or `..._MAX_MINUTES` (120) are refused at create with the figure, and a
  ceiling lowered later binds existing campaigns. The approval goes through
  `perform_checkpoint_decision`, the function a person's approval uses, as
  the campaign's owner with a note naming the campaign and its limits
  (`modules/autonomy/application/plan_approval.py` builds what a decision
  needs without the HTTP layer, which nothing below the API may import).
  The campaign service is handed that function by the scheduled task
  (`advance(..., approve=...)`) rather than importing it: the decision
  machinery reaches the finaliser, the planner and back to the campaign's
  budget, and importing it closed a 13-module cycle `test_layering` refused.
- **Everything waiting on a person is one page** (`/review`, first in the
  R&D door; `GET /review`, `modules/review/application/review_inbox.py`). The
  decisions a person owes were on five pages under three doors: runs waiting
  for sign-off, drafts in review, proposed patches, skills a run proposed,
  the research inbox -- and tool approvals on the Admin page only, so anyone
  who was not an admin had nowhere to approve an action waiting on *them*.
  The page composes them and owns none: each section is read the way its own
  page reads it (for runs, the operator queue's composer itself, so the two
  cannot disagree about what a queue row is) and links there. Three choices
  are deliberate. A section that cannot be read says so, with a savepoint
  around each so one failure does not poison the rest, and its count is
  "unknown", not zero. An approval waits on each person who still owes one
  (`approval_waits_on`): the owner is not asked again while the admin's half
  is outstanding. And approving from this page **runs** the action once
  nobody else's approval is owed, because approval is consent for it to
  happen; the Admin page still has the two as separate buttons. The queue's
  monitor and portfolio rows are standing configuration and are not counted.
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
- **A pool is per process, and the budget is checked** (`core/db_budget.py`).
  The API may open `(DB_POOL_SIZE + DB_MAX_OVERFLOW) x replicas x workers`
  connections, and nothing added that up: at the old 20 + 40 the chart's
  default 2 x 4 processes could ask for 480 of Postgres's default 100.
  Defaults are now 10 + 20, every deployment sets `max_connections=300`
  explicitly, the API compares its budget with the server's real limit at
  startup (`DB_EXPECTED_API_PROCESSES`; a warning, never a failed start), and
  the Helm chart **refuses to render** a configuration that cannot fit
  (`backend.dbPool`, `postgres.maxConnections`, or
  `postgres.external.maxConnections` for a database it does not run; 40
  connections are left for Celery and tooling). Turning on backend
  autoscaling therefore needs a smaller pool or a larger limit, which is the
  point.
- **Rate limits are counted in Redis** (`RATE_LIMIT_STORAGE_URL`, unset means
  `REDIS_URL`). The limiter's default was process memory, so each of N API
  processes allowed the full limit -- a `5/minute` login limit was `5 x N`
  and reset on restart. With Redis unreachable it falls back to in-process
  counting rather than failing requests. Tests use `memory://`.

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
- **An access token is issued and read in one place** (`core/tokens.py`:
  `create_access_token`, `decode_access_token`). Seven places decoded the JWT
  by hand — the HTTP dependency, the WebSocket helpers, the rate limiter's key
  function, the refresh route, the optional-user document download — and they
  agreed on the signature alone: none checked the token's `type`, so anything
  signed with the key and carrying a `sub` was a login. The decoder returns
  the subject or `None` and never raises; the caller decides what a missing
  user means. `tests/test_one_token_decoder.py` refuses `jwt.decode` or
  `jwt.encode` anywhere else. Two things found on the way: `POST
  /auth/refresh` renewed a **deactivated** user's token (the active check
  lived in the dependency, which that route does not use), and it raised its
  own 401s inside a `try` ending in `except Exception`, so a bad token came
  back as a 500.
  **`SECRET_KEY` is validated, not warned about.** An empty key is refused
  always — an unset variable in a compose file arrives as `""`, and a JWT
  signed with the empty string is one anybody can mint. The placeholders this
  repository publishes (the code default and the one in `env.example`) are
  refused unless `DEBUG` is on. The Helm chart's own default
  (`change-me-please-...`) is **not** refused: `make minikube-up` runs on it
  with `DEBUG` off, and the chart's NOTES already warn. A deployment that was
  running on either now fails at startup, which is the point.
- **A URL from data is not a link until it is checked**
  (`frontend/src/utils/safeUrl.ts`, `safeExternalUrl`). React escapes text
  and does nothing about `href="javascript:..."`; paper URLs, tool-result
  links and knowledge-graph entities of type `url` all reach an `<a>` from
  ingested content. Only http(s) and same-origin paths pass (`//host` does
  not). `safeUrl.test.ts` also reads every page: a `target="_blank"` link
  must carry `noopener`, and no `href` may take a data field directly.
  Mermaid is the same hazard by another route — the diagram source is model
  output over ingested documents — and it ran with `securityLevel: 'loose'`
  (click handlers and raw HTML labels) from an unpinned CDN URL. It is
  `'strict'` now, loaded at an exact version with an SRI hash; bumping the
  version means recomputing `MERMAID_INTEGRITY` — no test compares the two
  (that needs the network), and a stale hash shows up as the browser refusing
  the script, so every diagram falls back to its source text.
- **An exemption that states a fact has to be true.** The auth guard's
  socket list exempted `/dashboard/ws` as "waits for an auth message and
  verifies its token". It took `user_id` from the client's first message and
  verified nothing: anyone who could reach the server was sent the overview,
  document titles and source health. It also held one request database
  session for as long as it stayed open. It authenticates now (a token in the
  query, a header, or a first `auth` message) and opens a session per read.
  `tests/test_dashboard.py` connects without a token. The dashboard service
  beside it had no test and failed with rows present: the activity feed read
  `doc.source` (a lazy relationship) and `execution.updated_at` (no such
  column), so one document made `/dashboard/activity` and `/dashboard/all` a
  500; the route walk meets it with an empty database.
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
  fields. Decided 2026-10-04: **they stay separate** -- they serve different
  writers, and merging would change one of them.
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
  **Copies that differ only in a string are invisible to an exact-hash scan.**
  A later pass hashed function bodies with their string and number constants
  normalised, and found more:
  - 24 identical phase-report closures in the deterministic runners, now
    `agent_runner_progress.phase_reporter(job, action)`.
  - Three job-cleanup tasks, now `job_support.prune_finished_jobs`. All three
    deleted the row even when `StorageService.delete_file` answered False,
    which it does instead of raising, so the file was orphaned with nothing
    pointing at it. The row is now kept for the next sweep.
  - Three entity get-or-creates, now `kg_entities.get_or_create_entity`. The
    paper-KG copy looked a name up whole and stored it cut to 512 characters,
    so a long name never found its own row and made another each time; with
    no unique constraint the copies piled up until a lookup raised
    `MultipleResultsFound`.
  - Four document progress sockets with one relay loop.
  - Two admin "enabled ids" route pairs.
  - Four built-in document sources.
  - Two experiment run-config setters.
  - The private-network decision both scrapers made (and each queried twice),
    now `web_scraper_service.private_network_access`.
  - Five private `_as_int` copies of `safe_int` / `bounded_int`.

  `config_values.string_list` no longer turns a null item into `"None"`, and
  `split=False` reads a JSON list without splitting a bare string.

  Left apart on purpose:
  - `_slugify` ×3: different separators, caps and fallbacks.
  - `_text` ×11: a one-line idiom in five variants.
  - `_job_to_response` ×6: six models into six schemas.
  - Connector `test_connection`s: one interface, different endpoints.
  - Per-class Pydantic validators.
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
- **Tools nobody had tested at all.** 19 of 64 chat tool methods and 55 of
  220 autonomous handlers were named in no test. Four new files cover 62 of
  them (`test_chat_document_read_tools.py`, `test_chat_document_write_tools.py`,
  `test_chat_tags_stats_kg_tools.py`, `test_generation_and_template_tools.py`),
  and the defects they found are fixed. Three `xfail(strict)` tests remain
  there, and each is a decision rather than a bug (below, and source
  ownership for `generate_slides_for_source`). What they found, by shape:
  - `search_by_tags` called `.overlap()` on a plain JSON column and had never
    returned a document; it and `list_documents_by_tag` now share
    `services/document_tags.py`. `generate_diagram` passed `messages=` to
    `generate_response` and failed every time. `export_data` read a lazy
    relationship on an async session.
  - **`confirm` was tested for truthiness**: the strings `"false"` and `"no"`
    deleted a document or a whole batch. Only `True` confirms.
  - **A limit is a number a model typed.** `min(params.get("limit", N), cap)`
    raises on a string or a null and has no floor, so a negative limit was
    `LIMIT -1` (every row) or a slice that dropped the tail.
    `config_values.bounded_int` is the one way to read one; `parse_uuid` and
    `parse_date` likewise (`UUID(123)` raises AttributeError, which
    `except (ValueError, TypeError)` does not catch).
  - **Chunks are fetched, documents are returned.** A window of `limit * 2`
    chunks let one long document hide every other match, and a reference
    document be "similar to itself".
  - **An allowance decided once and applied to every hop.** A non-admin
    scraping an allowlisted internal page was carried by a link on it to any
    private host; the allowance is per host now (`private_hosts`), and the
    allowlist lookup, which existed twice, is `internal_scrape_allowed_hosts`.
  - A text shorter than 50 characters produced no chunks, was marked
    processed and reported created: saved and unfindable.
  Tags are compared **without regard to case** everywhere (search and both
  counters): the two counters disagreed, one lower-casing and one not.
  Faceted search takes its results, total and facets from one sample of
  documents and applies all five declared filters; it says when the sample
  was full (`sampled`).
  Three decisions, now made: **`POST /agent/confirm-delete/{id}` goes through
  `_execute_tool`**, the same approval gate and audit row a chat turn gets —
  with `AGENT_REQUIRE_TOOL_APPROVAL` on (the default) it answers 409 with an
  `approval_id` instead of deleting, and the widget says the deletion is
  waiting; **a delete reports what it left behind** (`DocumentService.
  delete_document(..., warnings=[...])`: a file or vectors that could not be
  removed) rather than a bare success; and **document sources are shared**,
  so no source tool checks who requested the source.
  Decided 2026-10-04, following from those: the chat widget's delete confirm
  **keeps the approval gate** (with `AGENT_REQUIRE_TOOL_APPROVAL` on it
  creates an approval request rather than deleting), and source-ingestion
  progress is open to **any signed-in user** (it was requester-only, which
  contradicted "sources are shared").
  - **The scraper connects only to an address it checked**
    (`web_scraper_service.PinnedNetworkBackend`). The URL check resolved a
    name and httpx resolved it again to connect, so a name answering public
    then private (DNS rebinding) reached the private address. The connection
    now resolves once and checks every address; TLS still verifies the
    hostname. **Not behind a proxy**: there the proxy resolves, and httpx
    ignores proxy settings whenever a transport is given, so pinning would
    silently route around the proxy. macOS reports system proxies to
    `urllib.request.getproxies()` with no variable set (this machine has one
    at 127.0.0.1:1082). That made the first control test pass for the wrong
    reason. So did anyio resolving the IDNA-encoded name as *bytes*, which a
    fake DNS comparing against a `str` misses.
  - **A webhook tool goes through the same rule**
    (`web_scraper_service.OutboundGuard`). A custom webhook tool's URL is
    written by whoever authored the tool, an agent included, and may be
    templated from the call's inputs; it went straight to the HTTP client,
    so a tool could read the cloud metadata address or the deployment's own
    vector store and object storage, which answer without credentials on
    that network. It now reaches public addresses only, over a pinned
    connection, with no redirects followed. Private hosts are opt-in by name
    (`CUSTOM_TOOL_WEBHOOK_PRIVATE_HOSTS`, plus the hosts of active web
    sources); loopback and link-local are never reachable, listed or not.
    The inbox's repo lookup fetched an item's stored URL wherever it
    pointed; it fetches arXiv pages only, over https (arXiv's API returns
    `http://` ids, so both are accepted).
  - `execute_workflow` **queues** (`WorkflowEngine.queue_workflow`, shared
    with `POST /workflows/{id}/execute/async`); it used to run the whole graph
    inside the tool call. What can be checked without running (owned, active,
    one start node) is checked first, and a refusal is recorded as a failed
    execution.
  - `merge_documents` names every id it did not merge and why, merges a
    repeated id once, and warns when the result could not be indexed.
  - Nine tools were labelled `effects: read` or `network: none` while their
    handlers queued work, persisted documents, searched arXiv or (for
    `analyze_image`) posted to Ollama;
    `tests/test_tool_effects_match_handlers.py` reads the handlers' source
    and refuses that. The labels feed the policy UI and plugin
    classification, not the approval gate.
  - `capture_snapshot` no longer requires `clone_and_index_repo`: it records
    run state, and the false prerequisite planned and priced a clone.
  - `analyze_image` stays Ollama-only (see `VISION_MODEL`); with no Ollama
    reachable it now says that instead of a bare connection error.
  A second round (`test_patch_bundle_review_handlers.py`,
  `test_compiler_tool_handlers.py`, five of whose tests run live against
  Docker) found:
  - **Docker's exit 125 is not the code's failure.** Nine compiler, binary,
    BOLT and gem5 tools ignored the return code. So "the image is not built"
    read as "the pass did not compile", or "work is not defined", or
    `baseline_broken`, and a pass synthesis spent three model calls repairing
    it. `agent_sandbox_runtime.could_not_run(rc, stderr, image)` is the
    check, and it fires on 125 alone: a script may exit 126/127 itself.
  - **An autonomous context carries the job's owner.** `agent_action_service`
    left `user_id` unset, so every model-calling handler ran without the
    owner's LLM settings.
  - **A workspace belongs to its job.** `get_or_default(..., job=ctx.job)`
    refuses a workspace another job owns; naming its id used to propose that
    job's diff as your own.
  - **A workspace built from knowledge-base documents starts with a git
    commit** (`coding_workspace_manager._commit_baseline`). It kept hashes of
    its original files and not the files, so nothing could say what an edit
    changed and `propose_code_patch` answered "Not a git repository" on every
    one. `git add -f`, so a `.gitignore` among the documents hides nothing
    from the baseline. Best effort: without git the workspace is still made.
  - **`propose_code_patch` writes a `CodePatchProposal` row.** It used to stay
    in run state, which no review page reads. It also includes files the run
    created (bare `git diff` omits untracked files), lists deleted files, and
    reports a failed `git diff` instead of "no changes".
  - `literature_review_arxiv` had no autonomous handler, so the one producer
    of `literature_review` was unreachable from the jobs it names. An empty
    search no longer yields that finding. An old-style arXiv id
    (`hep-th/9901001v1`) keeps its archive prefix.
  - Replaying your own evidence bundle no longer appends the replay to it
    (`extra["replaying_bundle"]`). gem5 runs have a time limit, and none had
    one by default. A BOLT option is refused before the build, not after it.
    A pass that crashes is `pass_crashed`, not "unregistered".
- **A task the worker never imports is a message thrown away.** The worker
  loads only `celery_app`'s `include` list. `workflow_tasks` (since January),
  `export_tasks` and `synthesis_tasks` (since February) were not on it, so the
  API queued their tasks happily and the worker discarded each one as
  unregistered: no async workflow, export or synthesis job ever ran, and the
  rows just stayed pending. `tests/test_every_task_module_is_registered.py`
  fails on a module defining a task that is not included. Decided
  2026-10-05: scheduled workflows are **on** (a per-minute beat entry; it
  fires only for a tick in the last minute, so a dormant schedule does not
  replay), and the export, presentation, repo-report and synthesis cleanups
  stay **off** -- nothing is deleted on a timer.
  Running those task bodies for the first time
  (`test_export_and_synthesis_tasks_run.py`, `test_workflow_tasks_run.py`)
  found what code that has never run usually has:
  - **No export could complete.** `upload_file(file_bytes, file_path)` was
    called against `upload_file(document_id, filename, content)`. Chat
    exports also read `session.topic` and `msg.sources`, which do not exist.
    A partly-filled custom theme laid `None` over the defaults, and the UI's
    `heading_size` matched no builder key.
  - **Synthesis:** the theme map was never produced (wrong keys both ways).
    An output file that failed to upload left the job `completed` with no
    file. Cited agent runs never reached the seven document job types, and
    a run's findings were hidden behind its goal used as a summary. Papers
    were loaded by id with no owner check. The cleanup dropped rows whose
    file survived.
  - **Workflows:** the documented `{{context.X}}` input syntax (77 template
    uses) resolved to None, and a top-level run had no `trigger_data` in its
    context. Nothing published workflow events: document upload, processing
    and a user's delete now do.
  `test_calls_fit_signatures` now also checks methods called on a held
  service (`self.storage.upload_file`) and on an imported singleton
  (`storage_service.download_file`). It found the DOCX editor and training
  calling `download_file(path)` for bytes it never returns (the editor could
  open and save nothing, and every training job failed reading its dataset).
  A second pass ran the user-triggered tasks no test named
  (`test_training_template_report_tasks_run.py`,
  `test_document_and_paper_tasks_run.py`):
  - **Training:** metrics were edited in place on a plain JSON column, so only
    step 1 was stored. Dataset export called `upload_file` wrongly, so a
    dataset that was never exported could not be trained on. A job cancelled
    mid-run was recorded `failed`. Checkpoint cleanup orphaned weights, and a
    failed run leaked its dataset copy.
  - **Template fill:** a model failure was written into the delivered
    document as text, and the job completed. A heading with no placeholder
    under it got nothing. The section search ignored the source documents.
  - **Repo reports:** diagrams went straight to public kroki.io, now through
    `mermaid_renderer`. That renderer cached its HTTP client across Celery
    tasks' event loops, so in a worker every render after the first failed;
    it is now per loop. DOCX and PDF ignored the chosen theme.
  - **Documents:** `process_document` counted chunks through a lazy
    relationship (MissingGreenlet), then marked a well-indexed document
    failed. Both processing tasks reported success when indexing had failed.
    A re-synced document kept its old chunks.
  - **Connectors:** `initialize()` returning False was ignored. They now keep
    `last_error`, and dry runs and git compares report it.
  - **Smaller ones:** GitLab projects with an id could not be matched by
    name, and their diffs were counted +0/-0. Any title starting "Chat " was
    treated as the default and overwritten. arXiv BibTeX (plain text, not
    `<pre>`) was never parsed. Enrichment stamped `enriched_at` during
    outages. A literature review read earlier reviews in as papers.
  `test_calls_fit_signatures` also follows `self.x = <imported singleton>`,
  which is how the dataset export call had escaped it.
  A third pass ran the beat-scheduled and remaining tasks
  (`test_sync_and_maintenance_tasks_run.py`,
  `test_monitoring_latex_transcode_tasks_run.py`). **Timestamps from Postgres
  are aware, `utcnow()` is not, and SQLite hides the difference.**
  `document_sources.last_sync` and `notifications.created_at` are
  `TIMESTAMP WITH TIME ZONE`. Compared with a naive `utcnow()`, they raised a
  TypeError that a per-item `except` swallowed. As a result an auto-sync
  source that had synced once was never scanned again, a user who had one
  overdue queue alert got no alert after it, and the admin next-run times came
  back None. Those comparisons now run in aware UTC. A test reproducing this
  has to feed rows as Postgres returns them, since SQLite returns naive values.
  `tests/test_naive_now_meets_aware_columns.py` now refuses the shape from
  source: it reads the models for `DateTime(timezone=True)` columns and fails
  on a `utcnow()` (or a local derived from one) compared with or subtracted
  from such a value, leaving SQL expressions alone. Its first run found the
  decision-trace overdue count and the validation backoff cooldown. **The
  other ~700 `utcnow()` calls are not swapped wholesale**: an aware value
  written to one of the ~22 naive columns fails under asyncpg, and
  `isoformat()` output stored in JSON would change shape. Use
  `utils.datetimes` (`utc_now`, `is_past`, `age`, `as_aware_utc`) where a
  stored value is involved.
  Also:
  - The LLM health check pinged Ollama for every provider except DeepSeek, so
    OpenAI and Anthropic deployments read "degraded".
  - The citation lint's own write bumped `updated_at` (`onupdate`), so notes
    were re-linted and their owners re-notified indefinitely.
  - `blocked_run` queue rows were never alerted on.
  - When only the PDF upload failed, LaTeX reported "server error" and lost
    the compiler log. The source-based BibTeX trigger never matched (the
    regex was double-escaped in a raw string).
  - Transcoding ran ffmpeg on an empty file when the original was missing,
    crashed with UnboundLocalError on an early failure (`subprocess`
    imported inside the `try`), and left `is_transcribing` set when queueing
    failed.
  - A real sync ignored a failed `initialize()`, just as the dry run had.
  Decided 2026-10-06:
  - **Per-source sync settings beat the sweeps.** The hourly and daily sweeps
    skip a source that is already syncing, one whose Auto Sync is off, and one
    the 5-minute scan schedules (interval or cron). A source whose config
    never set `auto_sync` is still swept. That covers every source created
    from the Documents page, where the request's `auto_sync` means "ingest
    now" and is not stored. (`sync_tasks._sweep_skips`)
  - **Memories outlive the chat they came from.** The year-old session
    cleanup detaches their memories (`session_id` set to NULL) before
    deleting sessions. Previously the delete-orphan cascade took them too.
- **A setting must be read, or admitted inert.** `TRAINING_ENABLED` was the
  documented gate on training and nothing read it; the concurrency limit and
  two dataset limits beside it were the same. They are enforced now (the gate
  as a router-level dependency, counted across users for concurrency since
  the machine is what is rationed). A field cannot simply be deleted — a
  deployed `.env` still naming it makes `Settings` refuse to load — so the
  remaining eight are marked `NOT READ` in `config.py` and listed in
  `tests/test_settings_are_read.py`, a list that may only shrink.
- Admin users have broader access; non-admins cannot access other users' resources
- **LDAP / Active Directory sign-in** (`services/ldap_service.py`, the login
  flow in `auth_service.authenticate_user`, Admin → Directory (LDAP)). It
  shipped with no tests, and each of these was wrong:
  - **The server's certificate is verified.** `ldap3` validates nothing unless
    given a `Tls` object that says to; it was given none, so `ldaps://`
    accepted any certificate with `LDAP_INSECURE_SKIP_TLS_VERIFY` at false.
    `LDAP_CA_CERT_FILE` names an internal CA. Plain `ldap://` without StartTLS
    is refused unless `LDAP_ALLOW_PLAINTEXT`.
  - **The username is escaped** into the filter and the DN template (it was
    `str.format`), and a name matching two entries is not a login.
  - **"Wrong password" is None; "could not ask" raises `LdapUnavailable`.**
    They were both None, so the stored-password fallback meant for an outage
    ran after every rejection: an account disabled in the directory kept
    working. The fallback now needs an outage *and*
    `LDAP_LOCAL_FALLBACK_WHEN_UNAVAILABLE`.
  - **A deactivated user stays deactivated.** An LDAP login used to set
    `is_active` back to true.
  - **Group DNs are separated by `;`.** `LDAP_ADMIN_GROUP_DNS` was split on
    commas, which cut every DN into its RDNs: role mapping could never match.
    DNs are compared normalised (`normalize_dn`), groups can be found by
    member search (`LDAP_GROUP_SEARCH_BASE`) where there is no `memberOf`, and
    with no mapping configured `map_role` returns None and the role is left
    alone (it was overwritten with "user", demoting local admins).
  - **A shared username does not take over a local account**: linked only
    when the emails match, by import, or with
    `LDAP_LINK_LOCAL_USERS_BY_USERNAME`.
  - It is synchronous I/O with connect and receive timeouts; the event loop
    uses the `*_async` wrappers.
  `ldap_service.diagnose()` (`POST /admin/ldap/test`) walks the setup the way
  a login would and reports the failing step, without any user's password.
  `tests/test_ldap_integration.py` runs against ldap3's in-memory server, so
  binds, searches and filters are real.
  **Login, import and sync turn a directory entry into an account by one set
  of rules** (`modules/identity/application/directory_accounts.py`). The
  import route carried its own: it matched by username and linked whatever
  the email, so a directory entry `uid=admin` became the local `admin`,
  which login refused to do. Accounts are matched by the entry's stable id
  first (`LDAP_UNIQUE_ID_ATTRIBUTE`: `entryUUID`, or `objectGUID` on AD,
  whose 16 bytes are read as a UUID), then by DN for rows from before, then
  by username and email; a move or rename in the directory no longer
  changes who someone is. Bulk searches now apply member-search groups
  (`LDAP_GROUP_SEARCH_BASE`) as login does, so an import no longer maps
  someone whose admin group is visible only that way to "user"; the paging
  cookie is read before those searches, which replace the connection's
  result (a test fails if it is not). The import writes each person under a
  savepoint, so one bad entry does not poison the rest.
  **Accounts are reconciled between logins** (`directory_sync.py`,
  `POST /admin/ldap/sync`, the Sync card; on a schedule with
  `LDAP_SYNC_INTERVAL_MINUTES`, default 0 = off). Before, someone who left
  or lost the admin group kept the account and the role until they signed
  in again. Those `LDAP_IMPORT_FILTER` finds are the people who should have
  access: the rest are deactivated with `deactivated_by: directory_sync`,
  which is the only deactivation the sync will undo; an admin's never is,
  and local accounts are not touched. A result that looks like a broken
  query deactivates nobody and says why: an empty answer, or more than
  `LDAP_SYNC_MAX_DEACTIVATE_FRACTION` (20%) of managed accounts missing.
  An unreachable directory changes nothing. The page previews by default;
  the last report is kept in Redis for the status card.
  **A directory account cannot set a local password** (`PUT
  /users/me/password` answers 400): it is never checked except by the
  outage fallback, where it would outlive the account's being disabled in
  the directory. Hashing moved to `core/passwords.py` so provisioning needs
  no import of the auth service.
  **The sign-in page, where accounts come from a directory.** Anyone could
  register a local account with an email nothing checks, and linking
  trusted a matching email: a stranger could register a directory person's
  name and email before their first sign-in, use the account, and have the
  directory hand it to them (account pre-hijacking). The register route now
  records `auth_metadata.origin = "self_registration"`, and `may_link`
  never links such an account, by email or by name; an account an admin
  made, or one from before the marker, links as before (nothing marks any
  account verified, so "verified" could not be the rule).
  `AUTH_SELF_REGISTRATION` unset means "allowed unless LDAP is on", the
  page reads `GET /auth/options` and hides "Sign up" when it is off, and
  says "Sign in with your directory account" when there is one. An
  unreachable directory answers 503 "could not be reached" with
  `Retry-After`: `authenticate_user` returned None for it, which the page
  showed as "Incorrect username or password".
  **A viewer is read-only** (`services/viewer_access.py`, decided
  2026-10-14). The group mapping assigned `viewer` and nothing enforced it,
  so a viewer could upload, delete and start jobs. Each way in is checked
  where every request passes: the authentication dependency refuses any
  non-GET request except `VIEWER_MAY` (signing in and out, the viewer's own
  conversations, settings and notifications, and POST routes that only
  compute); the tool policy engine (chat, workflow nodes, MCP, custom
  tools) allows only tools declared `effects: read`, and no user or plugin
  tool, since the engine cannot see what those do; and a viewer's MCP key
  keeps only the `read` and `chat` scopes whatever it was granted. The
  optional-user dependency applies the rule outside its `try`, which turns
  every HTTP error into "anonymous"; MCP key validation checks scopes after
  loading the owner, which it used to do before.
  `tests/test_viewer_is_read_only.py` sends every write route that needs a
  user as a viewer and requires the refusal unless the route is allowed.
  **Run against a real slapd** (`tests/test_ldap_real_server.py`, skipped
  unless `TEST_LDAP_URI` names one; its docstring has the commands). Over
  LDAPS and over StartTLS, 10 of 10: the certificate is refused without the
  CA and for an address it was not issued for, `entryUUID` and the
  `memberOf` overlay are the server's own, paging works with the real
  control and member searches between pages, an entry moved to another OU
  keeps its account, and the sync deactivates a deleted entry and takes a
  lost admin role. The first run failed on the image's bundled CA, which
  had expired: the client refusing it is the verification working. Not
  run against Active Directory; `objectGUID` is tested as bytes only.
- MCP API keys are tied to users; tool policies evaluated per user context

### Service Layer Pattern
- Services are singleton-style classes with composition (not inheritance)
- **User settings passed per-call** (not stored in service state) for multi-tenant isolation
- Late async initialization with double-checked locking (`_ensure_vector_store_initialized()`)
- Key dependency chain: `AgentService` → `LLMService`, `DocumentService`, `VectorStoreService`, `MemoryService`
- The agent runtime is intentionally split into many small `agent_*` services around `autonomous_agent_executor.py` — prefer extending the relevant sub-service over growing the executor

- **The coding backlog's reads live in `modules/coding_backlog/application/backlog_store.py`**
  (the visibility rule, `get_visible_item`, `list_visible_items`), on the
  pattern `modules/workflows` set: the endpoint maps to HTTP. The list route
  used to fetch every item shared with anyone, decide in Python which the
  caller could see, filter in Python and slice a page; a person's own items
  are now selected, counted and paged by the database, and for other
  people's the database narrows and the rule decides (the grant is a JSON
  list). The first test to run that route found it **had answered 500 for
  every unassigned item since July**: `getattr(item, "assigned_user_id", "")`
  returns None when the attribute exists and is None, and `str(None)` went to
  the response model as the id "None". All three items in the dev database
  failed. `getattr(obj, name, default)` is not `getattr(...) or default`.
  **The action handler followed, tests first.** It was thirteen branches in
  one 650-line route with no test. `tests/test_coding_backlog_actions.py` was
  written against it as it stood -- through the real route, with jobs really
  created and only the task's `.delay` recorded -- and then the code moved:
  `backlog_actions.py` (one function per action; `ACTIONS` says in one place
  who may do each and whether it is about a slice), `backlog_jobs.py` (the
  repair and apply jobs an action starts), `backlog_slices.py` (slice state
  as plain data). The endpoint went from 1,526 lines to 478. The application
  layer raises `ActionRefused(kind, detail)` and imports nothing from
  FastAPI; `_REFUSAL_STATUS` in the endpoint is the only place that knows the
  status codes. Writing the tests found two defects before anything moved: a
  second operator note never replaced the first in `latest_summary` (the
  stored dict was edited in place and assigned back, which a plain JSON
  column does not see as a change), and relaunching a slice recorded the job
  it had **just started** as the one it retried from, because it read the
  slice's child job after overwriting it. Characterise before moving: the
  tests are what make a 650-line move checkable, and they find things first.
- The operator queue presents a job only where it becomes a row. The full
  presenter ran for every job a user had ever run on every poll, to show the
  handful waiting. **And it loads only the jobs that could become a row**
  (`checkpoint_queue_candidates.load_queue_jobs`, used by the queue route
  and the monitoring task). Two of the four reasons a job becomes a row are
  visible to the database (paused on a person; a recurring job that failed,
  paused or went quiet). The other two are inside JSON, so two flags carry
  them, `has_approval_checkpoint` and `has_contract_unmet_entry` (migration
  `0108`), each kept by a mapper event **from its own source alone** -- one
  combined column could not be kept right for a row loaded without its log,
  which is how the job list loads them. The query may select too many jobs
  and never too few: the projector still decides. Asking for too few would
  hide a row with nobody told, so `tests/test_checkpoint_queue_candidates.py`
  builds the queue both ways over every combination of status, phase,
  schedule, results, log and age (15,750 jobs) and requires the same rows.
  On a copy of the dev data: 305 jobs, 54 candidates, the same 53 rows. The
  job behind each accepted inbox item is loaded whatever its state, since
  the monitor snapshot looks it up. `has_approval_checkpoint` restates
  `extract_approval_checkpoint` because a model cannot import a service; a
  test holds the two equal on every shape.

- **One inline arXiv ingest, `services/arxiv_instant_ingest.ingest_paper`.**
  `POST /documents/ingest-arxiv-instant` and `POST
  /presentations/from-research` each carried a copy, and neither copy could
  store a new paper. The document was built without `source_id` (NOT NULL),
  `TextProcessor.split_text` -- a coroutine function -- was called without
  `await`, and the chunks had no `content_hash` (NOT NULL): three failures,
  the first refused by the database so the others never ran. The documents
  route answered 500; the presentations route caught the error and then
  failed on the session it had poisoned. The shared function files the
  document under a built-in, **inactive** source ("ArXiv Instant") and hands
  it to `DocumentService._process_document_async`, the pipeline an upload
  uses, so one place knows what a chunk needs. It looks a paper up without
  assuming one row per arXiv id (an ordinary source may hold the same paper).
  The presentations route ingests **one paper per `try`** -- one `try` around
  the loop lost every later paper to the first failure -- and reads
  `current_user.id` **before** any rollback, because a rollback expires every
  object in the session, the caller among them (the borrowed-session trap).
  `tests/test_coroutines_are_awaited.py` refuses an async method of a
  first-party class called without `await`; it resolves the receiver to the
  class, so a third-party object sharing a method name is not accused.

- **A download's filename goes through `utils/http_headers.content_disposition`.**
  Thirteen routes wrote a title or an upload's own name into
  `Content-Disposition` with an f-string. A header is latin-1, so a name in
  any other script (`отчёт.pdf`) raised `UnicodeEncodeError` when the
  response was built and the download was a 500; a double quote ended the
  value early; a line break was a second header. The helper sends an ASCII
  fallback and, when the name needs it, the real one as `filename*=UTF-8''...`
  (RFC 5987). `tests/test_download_filenames.py` refuses a hand-built one.
- **A call on an object built a line earlier is checked too**
  (`tests/test_locally_built_objects.py`). The import and signature guards
  cover singletons and `self.x = X()`; an object built inside the function
  slipped past both, twice, each inside `except Exception`: the chat tool
  `run_workflow` built `WorkflowEngine()` with no arguments and called an
  `execute` it does not have (it queues through `queue_workflow` now, as the
  workflows page does), and the PPTX template upload passed
  `StorageService.upload_file` a `prefix` it does not take, so **no template
  was ever uploaded** (it uses `upload_to_path` with the path it records).
  The guard follows first-party base classes, reads only a function's body
  (a decorator's `router` is the module's, not the local one), and only
  judges a name assigned once.
- **A number that sizes a query has a floor and a ceiling**
  (`tests/test_paging_parameters_are_bounded.py`). `limit: int = 20` on a
  route accepts -1, which is "no limit" to the database, and 0, which is a
  divisor in every page count: `GET /export?page_size=0` was a 500, API-key
  usage overflowed a timedelta on a large `days`, and `GET /admin/logs`
  returned the whole file for `lines=0` (and read it whole on the event
  loop; it keeps a tail in a thread now). The test requires `ge` and `le` on
  such a parameter; a route that clamps by hand is in `CLAMPED_BY_HAND`,
  which may only shrink. Writing its MCP half found that **MCP
  `list_sources` had never returned a source** (it read four attributes the
  model lacks, inside `except Exception`) and `get_document` returned
  `doc.metadata`, the table registry, where `extra_metadata` was meant.
- The presentation routes have tests (`tests/test_presentation_routes.py`);
  the list's `limit` and `offset` are bounded, since a negative limit is "no
  limit" to the database. Their queries and writes are in
  `modules/presentations/application/presentation_store.py`, moved under
  those tests; it raises `Refused(kind, detail)` and the endpoint maps the
  kind to a status, queues the task and builds the responses.

- **Every route is sent a request, and none may answer 500**
  (`tests/test_every_route_answers.py`). About four in five route handlers
  are named in no test, and each one first run by hand had never worked. The
  test does not check what a route returns: as an admin it sends every GET,
  and every DELETE/PUT/PATCH/POST that has a path parameter, an id naming
  nothing and an empty body, and requires an answer that is not a 500. 503 is
  allowed (a dependency is away); `NEEDS_A_SERVICE` lists what waits on
  another process. Its first run found ten routes, pinned in
  `tests/test_weak_spots_found_by_running_every_route.py`:
  - `GET /workflows/tools/builtin` was a 500, so the workflow editor had no
    built-in tools: JSON Schema lets `type` be a list (`["array", "string"]`)
    and the response model wants a string. `_parameter_type` picks the first
    that is not `"null"`.
  - **The campaign models were never imported by `app/models/__init__`.**
    That import is how the schema drift check and every test database learn
    which tables exist, so the drift check had never looked at the campaign
    tables and no test could store a campaign. A guard requires every model
    module to be registered. (Built from the migrations afterwards, the
    schema does match.)
  - Knowledge-graph path ids were typed `str`; a malformed one reached the
    database as text. They are `UUID`, so the framework answers 422.
  - `GET /admin/vector-store/stats` answers 503, not 500, when the store is
    away.
- **POST routes are walked too, with a body built from their own schema.**
  The same test builds, for each POST route with no path parameter, the
  smallest body its declared schema accepts, so the handler really runs. The
  network is closed for that walk (`socket.connect` refuses), the model
  raises `LLMServiceError`, and task dispatch is recorded, so a route that
  needs a dependency must answer 503. Nine more routes answered 500:
  - **No AI Hub customer profile could be saved.** The route assigns the
    profile a UUID and then called `json.dumps(profile.model_dump())`, which
    cannot write one. `model_dump(mode="json")`.
  - **The agent router never consulted the model.** `agent_core.contracts.
    LLMClient` names `generate_response` and `generate_text`; `LLMService`
    had only the first, so intent analysis raised AttributeError inside
    `except Exception` and every message was routed by keywords. The service
    has `generate_text` now, and a test requires it to satisfy the protocol.
  - **"A dependency did not answer" is 503, decided in one place**
    (`core/exceptions.failure_status`, `is_dependency_failure`). The typed
    errors already map to 503 in the global handler, but a route that wraps
    its body in `except Exception` and raises its own 500 never lets them
    reach it: workflow synthesis, arXiv query translation, the three LaTeX
    copilots, document upload and the vector-store reset all said "server
    error" when the model or storage was merely away. Causes are followed,
    since services wrap what they catch. A vector store that is not ready
    raises `VectorStoreNotReady`, still a `RuntimeError` for callers that
    catch one.
  - Synthesis job and document ids are `UUID` in the routes and the quick
    request; "nothing to summarise" is a 400.
- **Waiting on Celery from a route goes to a thread.** `task.get(timeout=60)`
  in `POST /admin/sources/{id}/dry-run` was a blocking read on the event
  loop: for up to a minute that API process served nobody else. The admin
  health and stats routes did the same for two seconds a poll, and the task
  status route ran three `control.inspect()` broadcasts inline. The
  blocking-call guard now knows these shapes: `.get(timeout=...)` with no
  positional argument that is not awaited (which is what tells it from
  `client.get(url, timeout=...)`), and `control.inspect/purge/ping`.

- **Lower layers do not import higher ones** (`tests/test_layering.py`, read
  from source including imports inside functions). Nothing outside `app/api`
  imports `app.api`; services, models, schemas, core, agent_core, utils and mcp
  do not import a module that defines Celery tasks (a helper under
  `app/tasks` that defines none, such as `job_support`, is not the worker);
  and in `app/modules/<domain>`, `application` and `domain` do not import
  `api`. 891 function-local imports had let 51 modules (services, tasks,
  endpoints and `modules/autonomy`) form one import cycle Python never
  reports. The test's `ALLOWED` set started at 19 entries and is empty; it may
  only shrink, and a stale entry fails too. The last one was the monitoring
  task importing the operator-queue composer from its endpoint: that composer
  and the job presenter it binds now live in
  `modules/autonomy/application/checkpoint_queue_builder.py`. With these gone
  the cross-layer cycle is broken. **Import a name from the module that owns
  it, not from a facade**: six modules took `AgentToolExecutionContext` from
  `agent_tool_dispatch`, which imports every provider, and providers call
  back into `agent_service` -- a 34-module cycle that pointing those imports
  at `agent_tool_providers.base` reduced to 7. `test_no_import_cycle_grows`
  caps the largest cycle at 7 (`LARGEST_CYCLE`, which may only shrink). Those
  7 are mutual recursion between peers (a workflow runs tools; a custom tool
  can run a workflow; chat runs workflows, whose tool nodes call
  `AgentService`), which needs an interface to break, not a moved import.
- **Work is queued through `services/job_dispatch`**: `enqueue(db,
  "app.tasks.<module>.<task>", *args)`, `enqueue_agent_job(db, job_id,
  user_id)`, and `send_now(...)` for a caller with no session to wait on.
  Tasks are named by path and resolved when sent, through the task's own
  `.delay`, so a test patching it still intercepts and a misspelt name fails
  at the call (`test_every_task_named_in_the_code_exists`). The rule is
  **"has this transaction written"**, tracked by `core/session_writes.py` from
  ORM flushes and ORM DML: if it has, the message goes when it commits and is
  dropped on rollback; if not, it goes now and a broker error reaches the
  caller (the chain orchestrator records `chain_dispatch_failed` from it).
  Not "is a transaction open": any SELECT opens one, and a read-only caller's
  message deferred to a commit nobody makes is lost. `enqueue` returns the
  `AsyncResult` only when it sent now, so a caller needing the task id commits
  first or uses `send_now` -- a tool running inside another session's
  transaction (chat, an autonomous job) cannot know it is clean. Fixed on the
  way, each queuing work its worker could not yet see or overwriting the
  worker: the coding backlog's three orchestrator starts and its repair and
  apply spawns, the coding runner's spawns, the opportunity reprioritiser, and
  training's `start_job`, which sent the task and then committed `QUEUED`
  over a worker that had already marked the job running.

- **Nothing blocks the event loop** (`tests/test_no_blocking_calls_in_async.py`).
  A synchronous subprocess, sleep, HTTP call or bcrypt hash inside `async def`
  stops every other coroutine in the process -- in the API, every request
  being served. MCP `docker_execute` ran `docker pull` (up to ten minutes) and
  `docker info` inline, an upload ran ffprobe, each login ran bcrypt, and
  agent tools ran graphviz, ffprobe, a LaTeX build and a sandboxed program
  run. Each now goes through `asyncio.to_thread`; `docker_executor` offers
  `await docker_available()` beside the sync `is_docker_available()`. The
  test reads source for known blocking calls made directly in an `async def`
  (a nested `def`/`lambda`, or a function passed uncalled to `to_thread`, is
  how work is handed to a thread and is not counted); a blocking helper of
  this codebase's own belongs in its `BLOCKING_HELPERS`.

- **The Redis cache holds JSON** (`core/cache.py`). Values were pickled, and
  reading a pickle runs what it describes: anything able to write to Redis
  could run code in every API and worker process. Values the old code wrote
  are still read through an unpickler that refuses every class and function
  (feature flags have no expiry, so a format change would otherwise reset
  them). A value JSON cannot hold is refused, not coerced. Two things found
  there: the `document:{id}` cache pickled whole rows on every upload and
  **nothing ever read it** (removed); and the scheduled sync wrote
  `ingestion:task:{id}` through that cache while every reader took the raw
  string, so a scheduler-started sync could not be cancelled -- it now uses
  `utils/ingestion_state`, as the other writers do. The test fake had hidden
  it by keeping Python objects for writer and reader alike.
- **A 500 does not carry the exception's text** (`core/exceptions.
  internal_error_detail(exc, "what failed")`). Sixty-seven routes and the
  handler for unhandled errors returned `str(e)` -- SQL, paths, upstream
  replies -- to the caller. The client now gets a reference and the exception
  is logged under it; `EXPOSE_ERROR_DETAILS` (off by default, on in the dev
  stack and tests) restores the text. `tests/test_error_details_are_hidden.py`
  refuses a 500 whose `detail` is built from the exception. 4xx responses
  still carry their message: that text is for the caller.
- **`except Exception: pass` is capped** (`tests/
  test_silent_exception_swallows.py`, 226 and only shrinking). Not a demand
  to rewrite them: a new best-effort step catches what it expects or logs
  what it ignores.
- **A time limit on waiting is not a time limit on the work.**
  `await asyncio.wait_for(asyncio.to_thread(run), timeout)` stops waiting and
  leaves the thread blocked on a child nobody will kill. The system-status
  route did that for `docker version`: with the daemon hung, each request
  kept one of the default executor's threads for ever, and every
  `asyncio.to_thread` in the process shares that executor. Nine
  `subprocess.run` calls had no `timeout` of their own (the two Docker
  probes, the admin image pull and sandbox check, three diagram renders, the
  custom-tool image inspect, the Whisper weights download). Each has one now,
  and `tests/test_subprocesses_have_a_time_limit.py` refuses a call without.
- **A task nobody awaits is started with `utils.background.spawn`.** The loop
  refers to its tasks weakly, so `asyncio.create_task(...)` on a line by
  itself can be collected while it runs -- it just stops, with nothing
  raised -- and its error is reported only at collection. Four places did
  that: the two admin downloads (a status stuck at "downloading"), export
  progress, and chat's memory extraction. `spawn` keeps the task until it
  ends and logs what it raised; `tests/test_background_tasks_are_kept.py`
  refuses the bare form. The fourth was also **handed the request's database
  session**, which is closed when the reply is sent while the extraction
  runs on; it opens its own now. Work that outlives a request never borrows
  the request's session.
- **Object storage does not run on the event loop, and it has a time limit.**
  The MinIO SDK is synchronous and `storage_service` called it straight from
  its `async` methods -- 23 calls -- on the SDK's default client, which waits
  five minutes for each of five attempts. A MinIO that accepted the connection
  and said nothing (a hung container) therefore froze the whole API process;
  it is how a test hung for over ten minutes on a developer machine. Each call
  goes through `asyncio.to_thread`, the client has `MINIO_CONNECT_TIMEOUT_SECONDS`
  (5), `MINIO_READ_TIMEOUT_SECONDS` (60) and `MINIO_MAX_RETRIES` (2), and the
  blocking-call guard knows the SDK's method names. Checked against exactly
  that half-dead port: the upload failed in 3.1 s with the loop still running.
  Tests pin `MINIO_ENDPOINT` to an address that refuses at once: settings read
  `backend/.env` from the working directory, so the suite had been talking to
  the dev stack's MinIO.

- **Model calls are capped per process and across the deployment**
  (`services/llm_concurrency.llm_slot`). `LLM_MAX_CONCURRENCY` (4) was one
  module-level semaphore, so the real ceiling was the setting times the number
  of processes — 64 on the chart's defaults — and rose with every replica.
  `LLM_GLOBAL_MAX_CONCURRENCY` (16; 0 turns it off) is counted in a Redis
  sorted set shared by every API and worker process. It is **soft** by
  decision: Redis unreachable disables it for 30 s rather than failing calls,
  a caller that has waited `LLM_GLOBAL_ACQUIRE_TIMEOUT_SECONDS` proceeds, and
  a slot expires after `LLM_GLOBAL_SLOT_TTL_SECONDS` so a dead process does
  not hold one for ever. Overshooting a quota costs a 429 the caller already
  handles; a limiter that can stop every model call costs the platform. The
  local semaphore is also kept **per event loop** now: Celery runs each task
  in a fresh loop, and the shared one raised "bound to a different event
  loop" on its first contended wait. The counting is a Lua script, which a
  fake cannot check, so three tests run against a real Redis in the Postgres
  CI job (`TEST_REDIS_URL`) and skip without one.
  **Sandbox runs are capped the same way** (`SANDBOX_MAX_CONCURRENT_RUNS`,
  4; `services/shared_slots.py` is the counter both use). Nothing limited
  them: each worker process could start a container per tool call, two CPUs
  apiece, on a daemon whose load none of them could see -- and a run timed
  beside too many others measures the others. The count is **per daemon**
  (`agent_sandbox_runtime.sandbox_slots_key`): a TCP `DOCKER_HOST` is one
  daemon every client shares, a unix socket is the chart's per-pod sidecar
  and is keyed by host name. Waiting for a slot is not charged to the run's
  time limit, and a run still waiting after `SANDBOX_SLOT_WAIT_SECONDS`
  proceeds. Only `run_in_sandbox` is capped; the four callers that build the
  command for `subprocess.run` (experiments, the ingestion demo, the admin
  check) and custom Docker tools are not.

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
- Methods are first-class knowledge (`services/agent_method_record.py`, tool `record_method`): a run records *how* to investigate something — procedure, what it prevents, and the finding types in this run that establish it — stored as a `pattern` job memory so later jobs recall it. Evidence is checked the same way `record_prediction` checks `derived_from`: citing a finding the run never produced is refused, and a method may only be stored without evidence by passing `['none']`, which marks it unvalidated. Construct these as `ConversationMemory` directly — `MemoryCreate` rejects the `pattern` type, and a method stored under a type the job-memory filter does not inject is written but never recalled. A contract can require one with `validity.records_method`. The run also keeps it in `state["recorded_methods"]`, which the finaliser copies to `results["methods"]` -- the key the run's own library record reads. Nothing wrote that key before, so a run's methods reached later jobs' memory and never its record; the record's test had set the key by hand
- Implementing an algorithm and measuring it is one chain, and the middle link is a correctness gate: `check_implementation` (`services/agent_implementation_check.py`) runs the code against reference cases from the paper before `benchmark_c_snippet` times it, because the fastest implementation of any algorithm is one that returns garbage. It is `perishable`, so a looping implement stage cannot inherit a verdict it earned before the edit, and a check with no cases reports unverified rather than passing vacuously. `compare_to_claim` (`services/agent_claim_comparison.py`) scores the result against the paper's number with three verdicts, not two — `incomparable` is the honest answer when units, hardware or input size make the two numbers untestable against each other. Build recipes live in `services/agent_toolchains.py`: **C, Rust and Python**, each from a single self-contained file; the `language` enum in the tool schemas is read from that table rather than restated, so a language it can build is never one the model is refused for naming. Python's "compile" step is `py_compile`, which is what makes a syntax error report as a build failure instead of as every reference case failing — and because `./prog` is timed as a whole process, an interpreted language reports `interpreter_startup_ms` measured in the same container, with a warning when startup dominates (measured: a 15 ms Python run was 8 ms of CPython booting). Both the checker and the benchmark take `language` and share that table, so the binary that was verified is the binary that was timed. Rust gets **crates without a network**: `RUST_CRATES` (rand, rand_chacha, rayon, ndarray, num-traits, num-complex, itertools, pinned exactly) is built into the compiler-research image at *build* time and linked as prebuilt rlibs named by `/opt/rust-deps/externs.txt`, which the compile line reads — so the image stays the authority on what exists and older images degrade to no-crates rather than failing to compile. The image derives its Cargo manifest by importing `agent_toolchains.py` (`deploy/sandbox-images/compiler-research/gen_crate_manifest.py`), so the list the model is told about and the list the image builds cannot drift; keep that module stdlib-only or the image build breaks. Three rustc traps are handled in code because each fails in a way that blames the wrong thing: it defaults to **edition 2015**, where `use some_crate::X` does not resolve at all (added via `enforced_flags`, only when the caller names no edition, since rustc rejects a repeated `--edition`); it invokes `cc`, which the sandbox lacks, so the linker is pinned to clang in the template where overriding flags cannot drop it; and it rejects `-O2`, defaulting to an unoptimised build that times the debug binary
- **A check can name an oracle instead of hand-written cases** (`check_implementation` with `reference` and `generator`; adapted from ucb-bar/chia, whose RISC-V flow runs Spike inside the simulator so "the golden model *is* the check"). The reference is a second program written the obvious way, however slow (plain `%`, a scalar loop); the generator prints one input per index, edge cases first. Every input goes through both, in the one container the cases use, and the outputs must agree under the same comparison the cases get. No expected output is written by hand, so none can be written to match the code -- a live stage's harness printed PASS whatever its input, which this fails on every input. Three ways it could still check nothing are refused: a reference that is the candidate (refused unrun, as an error so repeats escalate), a generator ignoring its index (one distinct input is one case: `distinct_inputs >= 2`), and a reference that fails everywhere (its failures are counted apart, and nothing compared is not a pass). Directed cases still count when given. `generated_cases` and `oracle_checked` are on the finding so `validity.bounds` can require a size. Run live on the stack's daemon: a C fastmod against a Python `%` agreed on 200 inputs; with the reciprocal off by one (`/ d` for `/ d + 1`) it disagreed on 4, and the two shown were generator edge cases while 196 agreed -- a handful of random cases would most likely have passed it.
- Wall-clock measurements report the machine they were taken on: `benchmark_c_snippet` samples `/proc/loadavg` and `nproc` in the same container as the trials and returns `load_per_cpu`, `measurement_environment` (quiet/busy/saturated) and `trial_spread`, warning when the host was busy or the trials unstable. Carried on the finding as well as the result, so `validity.bounds` can refuse a run whose numbers were taken on a machine too busy to measure anything. Only the wall-clock tool needs this — simulated cycles are the same on a busy host as a quiet one
- One user's running jobs are capped (`AGENT_JOBS_MAX_RUNNING_PER_USER`, 4; 0 turns it off; `services/agent_job_user_cap.py`). A running job holds a worker process for hours and nothing bounded one user's share of them. Applied **where a worker claims a job, not where one is created**: pipelines, swarms and campaigns create jobs on a user's behalf, and refusing one breaks the run that asked. "Running" means holding a live lease, since a dead worker's job still says `running`. A held job stays `pending` with `queue_reason: user_job_cap` in its scheduler state and is sent again as a **new message** (`self.retry` would spend the retry budget of a job that has not failed). Three things keep "held" from turning into something else: only a job this delivery could claim is held (an ended or already-held job answers `lease_conflict`, or it would be redelivered for ever); a held job is set back to `pending` with no task id; and the stalled-job sweep skips a job whose redelivery is still due, because old, pending and unclaimed is also what "never dispatched" looks like, and three of those fail the job. Count and claim are two statements, so on Postgres a row lock on the user (`lock_user_claims`, FOR NO KEY UPDATE) makes one user's claims take turns: run live without it, six jobs delivered together each counted none running and all six started on a cap of four; with it, eight delivered together ran four at a time. A job whose worker was killed counts until its lease lapses. Not FIFO among one user's held jobs
- The job list reads the page it returns (`modules/autonomy/api/job_queries.py`). It loaded every job the caller had ever run, whole, and sliced twenty out in Python; the execution log alone is 163 KB after 80 iterations. The caller's own jobs in the default order are now paged and counted by the database, and the log column is deferred on every path -- so **nothing on the list path may read `job.execution_log`**, which on an async session is a MissingGreenlet, not a lazy load. The swarm sorts rank on each row's JSON in Python and still read every candidate. `collaboration_service.list_collaboration_user_ids` runs on every list request and did the same thing; it now selects the two columns it reads. Who a swarm job is shared with is a column now, `agent_jobs.collaborator_index` (`|id|id|`, migration `0107`): the lookup used to cast every job's `results` to text and pattern-match it across the whole table. **Mapper events on `AgentJob` keep it** from `results["swarm_collaboration"]`, not the functions that edit sharing, because `results` is also replaced whole by the executor and the finaliser; a Core `update(AgentJob).values(results=...)` would bypass them, and there is none. The partial index behind it is on `user_id`, since 200 ids outgrow a btree entry. The shared and all views use the column to narrow candidates and `is_job_visible` still decides; an admin's views are paged by the database. Measured on the way: a run rewrites its row about 10 MB in total over 80 iterations (quadratic in iterations), which is not worth moving the log to its own table at the default budget of 100
- Several agent jobs can share one worker process (`celeryAgents.pool: threads` in the chart, `AGENT_WORKER_POOL=threads` in compose; prefork is still the default). Each job runs on its own event loop in its own thread, so a job that blocks its loop stalls its own lease heartbeat and nobody else's. Measured on this stack with real jobs: prefork idles at 1.0 GiB for six slots and held 2.6 GiB after six jobs; threads idles at 0.3 GiB and ran **twelve at once in 1.8 GiB**, all completing, with the embedding models loaded once. Two things had to change. **State that assumed one live loop per process**: three per-loop tables (Redis client, model-call semaphore, diagram client) dropped every entry but the caller's on each access, which is right only while loops run one after another -- with two alive, each access evicted the other's live value. `utils/per_loop.PerLoop` is the one holder now, dropping an entry when its loop closes; `LLMService.client` and the init locks in `vector_store` and `agent_service` use it. A new process-wide singleton holding an asyncio lock, semaphore or client must too. **The wall clock**: Celery's soft limit is a signal only prefork can deliver, so the job task enforces `AGENT_JOB_SOFT_TIME_LIMIT_SECONDS` itself. A thread cannot be killed: a wedged job holds its slot until the worker restarts, while its lease lapses and the sweep moves the job on. **Run live on every kind of job**: eight at once in one worker (sandbox, gem5, coding and research, two users), all completing with their contracted evidence, and ten threads of the real compiler, binary-rewrite, BOLT and gem5 tools in one process. What that found, none of it visible with one job per process (`tests/test_jobs_share_a_worker_process.py`):
  - **A network client is closed on its own loop, before the loop ends.** A holder given a `close` is closed by `per_loop.close_running_loop_values`, the job task's last act. The provider SDK clients were one table for the process (now per loop), and an SDK client nobody closed schedules its own close on whatever loop is running when it is collected -- another job's.
  - **anyio's backend is imported in `core/celery.py`.** anyio looks for it in `sys.modules` without waiting on the import lock; jobs starting together met it half initialised and lost a model call.
  - **A vector-store write waits for a model load already under way** (`_wait_for_models`). `initialize(background=True)` returns while models load, and a write straight after was refused -- the first reprocess in any fresh worker.
  - **No `preexec_fn`** (`utils/limited_process.with_limits`): a fork hook is unsafe in a process with threads. Limits are set by a launcher that then `exec`s the command.
  - A tree-sitter parser is made per parse, libclang parses one at a time, and the text splitter is built per call instead of kept on the shared service.
  - **`get_document_details` read `doc.source`, a lazy relationship**: MissingGreenlet unless something earlier in the run had loaded that source, which is why it failed in 18 runs and never on demand.
- **The sandbox's user owns its work directory** (`agent_sandbox_runtime.open_to_sandbox_user`, called where the directory is mounted). `tempfile` makes a directory only root may enter and the container runs as 65534, so on a Linux daemon -- the stack's own -- every sandbox tool but skills failed alone: "cannot create compile_err.txt: Permission denied", reported as a kernel that did not compile. Docker Desktop's file sharing, which the first live runs went through, does not carry ownership. Opening the permission bits was not enough (BOLT's script runs `chmod +x` on a file the server wrote, which only an owner may), so as root the tree is given to the sandbox's uid.
- Agent workers can autoscale (`celeryAgents.autoscaling`, off by default, needs KEDA on the cluster). **The signal is demand, not CPU and not queue depth**: a job waiting on a model shows no CPU, and a running job has left the queue, so an empty queue would scale away the pods doing the work. Demand is the jobs a worker holds (a live lease, which includes one finished but still writing up) plus the pending jobs that could start now -- not one waiting for its schedule, and not one held by its owner's cap, which no number of pods would let run. One pod per `concurrency`. The query is `deploy/helm/knowledgedbchat/files/agent-worker-demand.sql`, and `tests/test_agent_worker_demand_query.py` runs it on Postgres (in the slice) and ties its literals to the code, because nothing else type-checks SQL kept in a chart. KEDA reads the password from the worker's own environment (`passwordFromEnv`); `DATABASE_URL` cannot be used because it is assembled with `$(POSTGRES_PASSWORD)`. **Scaling down can interrupt a job**: the pod removed is not chosen by whether it is busy, the job gets `terminationGracePeriodSeconds` and is then killed, and the sweep resumes it from its last checkpoint, which can take half an hour to notice -- so scale-down waits 30 minutes of low demand and removes one pod per 10. Rendered and validated against KEDA's CRD schema; **not run on a cluster**
- **A run says where its time went** (`GET /agent-jobs/{id}/profile`,
  `services/agent_run_profile.py`, "Where the time went" in the job panel;
  adapted from ucb-bar/chia's profiler, which draws a loop as a timeline and
  an aggregated table). A run is the job and every job sharing its root. Per
  job: queued, running, and the gap after the job it follows ended (a
  chain's handoff plus any approval waited for); per iteration its span
  from the log; and the observe/think/act/evaluate split, which
  `AgentRuntimeRunner` now times and the executor writes on each
  `iteration_complete` entry as `phase_ms` (iterations logged before have
  spans only, and the profile counts them). Totals group act time by tool.
  A job stopped without completing (paused, blocked) ran until its last log
  entry, and the time since is `paused_seconds`: measured to now, the first
  profiled run's blocked stage read 9.6 hours running and the run's wall
  time kept growing. That run, the self-approved fastmod campaign: 54 min
  wall, 46 running, 7 between jobs -- five of them the campaign's 5-minute
  tick approving the plan -- and under a minute queued.
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

- The search, chat and generation tools had no test
  (`tests/test_mcp_search_chat_generation_tools.py` calls them as the server
  does). Search loaded the id of every document, capped at 10,000, and dropped
  results outside that list, so past the cap a document could not be found;
  everyone may read every document, so it no longer builds the list. A
  **source filter is handed to the store**: search and chat applied it to a
  window of the best chunks from anywhere, which other sources' matches could
  fill. Chat read a filter matching no document as no filter. A failed
  presentation's status read `error_message`, which the model lacks.

- **A key's own rate limits are enforced** (`mcp/auth.enforce_rate_limits`,
  called from the `get_mcp_auth` dependency). Every key has a per-minute and a
  per-day limit, set in the form and shown in the list, and nothing read
  either. They are counted from the usage log, which every MCP request now
  writes (only tool calls did), so the count is shared by every API process;
  a refusal is a 429 with `Retry-After` and is not itself counted.
  **`api_keys.key_prefix` was VARCHAR(8) while the service stores 12
  characters**: Postgres refuses that, so no key could be created outside
  SQLite. Migration `0109` widens it, and `tests/test_api_keys.py` is in the
  Postgres slice, since SQLite does not enforce a width.

- **A chat turn's test must tell the model's reply from an apology**
  (`tests/test_chat_turn.py`). `ChatService.generate_response` answers every
  failure with "I apologize..." stored as an assistant message, and the old
  test accepted any non-empty reply, so it passed whether or not a turn
  worked. Run with a model that answers: `POST
  /chat/sessions/{id}/messages` stored the user's message twice (the route
  and the service each did), and the copy became the model's conversation
  history. The socket path never did.

### RAG Pipeline
- Query → hybrid search (vector + BM25) → reranking (cross-encoder) → MMR / deduplication / query expansion → optional KG context → LLM response
- Retrieval traces recorded for observability (`models/retrieval_trace.py`, `/retrieval-traces` endpoints)
- Configured via `RAG_*` environment variables
- **One retrieval for every answer written from the knowledge base**
  (`services/rag_retrieval.py`, `retrieve`). The pipeline above was inline in
  `ChatService.generate_response`, so the chat page was the only place that
  had it. The agent's `answer_question` tool and the MCP chat tool each
  searched once and pasted in whatever came back -- no query rewriting, no
  relevance floor, no knowledge-graph context -- and answered with the
  default model whatever the person had chosen. The block was **moved, not
  rewritten** (by script, text for text), and all three call it;
  `document_ids` restricts the search in the store itself and `max_results`
  is the caller's passage budget. This is the first step towards one chat:
  the chat page still has its own turn (sessions, sources, traces, feedback)
  and the widget has the agent turn, but they no longer retrieve differently.
  Each of the three also **records a trace** (`record_trace`; `trace_type`
  is `chat`, `tool` or `mcp`) and hands back its id: only chat-page answers
  could be inspected afterwards. The agent tool flushes and does not commit,
  because it runs inside the turn's transaction.

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

- Backend tests use **in-memory SQLite** with `aiosqlite` (configured in `tests/conftest.py`).
  `TEST_DATABASE_URL=postgresql+asyncpg://...` runs them on Postgres instead,
  and the `backend-postgres` CI job does that for the files in
  `tests/postgres_slice.txt` -- code whose behaviour depends on the database.
  On Postgres the schema is created once per run and the tables that hold
  rows are emptied between tests (`DELETE` with foreign-key triggers
  suspended; `TRUNCATE ... CASCADE` reached every table through `users`).
  Tests using the HTTP `client` fixture skip themselves there: TestClient runs
  the app on its own event loop, and an asyncpg connection belongs to one.
  Its first run found two things SQLite hides: tests inserting rows for users
  that do not exist (no foreign keys), and **asyncpg reading a naive datetime
  in the process's local zone** -- the app writes naive `utcnow()` into
  `TIMESTAMP WITH TIME ZONE` columns, so on a UTC+3 host every timestamp was
  stored three hours early. `app/core/database.py` now pins the process to
  UTC (`_pin_process_to_utc`, with a warning if `TZ` said otherwise);
  `tests/test_naive_datetimes_are_utc.py` fails without it.
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
- `celery` worker (default queue) + `celery_agents` (agent jobs only: the `agents` queue, `AGENT_JOB_CELERY_QUEUE`. They run for hours and mostly wait on a model; on the shared four-process pool they held every slot while uploads, summaries and workflows -- and the ingestion an agent's own tool waited for -- queued behind them. Nothing else consumes `agents`, so `tests/test_agent_job_queue.py` checks every stack has a consumer, and that each overlay configuring `celery` gives `celery_agents` the same through a YAML anchor: `extends` copies the base file's service, not an overlay's changes to it) + `celery_latex` (dedicated LaTeX compilation queue) + `celery_transcription` (dedicated Whisper queue; its image derives from the backend image, so `make build` builds the backend first — compose does not infer build order from a `FROM`)
- `kroki-mermaid` (8001) - Mermaid rendering, built from `mermaid-renderer/` (this repo's own: Alpine + Chromium + mermaid-cli, 1.08 GB against `yuzutech/kroki-mermaid`'s 1.54 GB, with mesa and libLLVM deleted because a headless browser never opens them). Speaks the Kroki companion protocol: POST the raw diagram to `/svg` or `/png`
- `sandbox-docker` - private Docker-in-Docker daemon (privileged) that every sandboxed tool runs in; backend and celery reach it over TLS. Load images with `make sandbox-load`
- `video-streamer` - Go microservice for video streaming (in `video-streamer/`)

Variants: `docker-compose.prod.yml` (gunicorn, healthchecks) — `celery_beat` runs in the dev stack too, and it is what makes `check_stalled_agent_jobs` fire there: with no beat, a job whose worker died (typically from restarting the celery container) stays `running` for ever until that task is invoked by hand, which requeues it to resume from its last checkpoint rather than failing it. The same sweep also covers the opposite failure: a job whose row was committed but whose Celery task never arrived — a broker restart, a purged queue, an enqueue that failed after the commit — sits in `pending` with no task id and no activity, which the running-job query could not see, so two were found in a live database 14 and 3 days old. Re-delivery is safe because the execution lease decides who runs: a duplicate returns `lease_conflict` without executing. A job holding a live lease, one already claimed, and one with a `schedule_type` are all left alone — firing a recurring job early is a wrong run, not a recovery), `docker-compose.test.yml` (isolated test stack on shifted ports), `docker-compose.docker-tools.yml` (uses the host's Docker socket instead of the `sandbox-docker` sidecar, and enables Docker custom tools).

Access points:
- Frontend: http://localhost:23000
- Backend API: http://localhost:28000
- API Docs: http://localhost:28000/docs
- MinIO Console: http://localhost:29001

CI runs in `.github/workflows/ci.yml` on pull requests and pushes to `main`: backend lint/format, backend tests with the coverage gate, a single-alembic-head check, frontend typecheck plus tests, a `helm-chart` job that lints every values profile, validates the rendered manifests with kubeconform, and parses the generated nginx configs, and a `helm-smoke` job that installs the chart on an ephemeral kind cluster and asserts the wiring rendering cannot check (hook ordering, Secret-to-URL assembly, gateway routing, migration-gated upgrades). Lint is gated on `app/` and `tests/` only — `alembic/`, `scripts/`, and `seed_data/` carry pre-existing formatting and flake8 debt that is reported but not enforced. The same checks are available locally as Makefile targets (`make lint`, `make fmt`, `make test-backend-coverage`, `make typecheck-frontend`), which shell into a running Docker stack.

## Other Documentation

Root-level docs worth checking before larger changes: `BUILD_AND_RUN.md`, `QUICK_START.md`, `DOCKER_SETUP.md`, `deploy/README.md` (Kubernetes/Helm), `AGENTS.md`, `AUTONOMOUS_RND_AGENTS.md`, plus `docs/` for architecture guides. Current visual architecture map (deployment, subsystems, agent runtime, LLM stack): `docs/ARCHITECTURE_DIAGRAMS.md`.
