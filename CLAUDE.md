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
  `extended_with()`, which refuses anything shadowing a built-in. **Chat is not
  wired yet** — it builds its menu from the global `AGENT_TOOLS` at three
  sites; plugin tools reach autonomous jobs only.

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

- **`measure_headroom` crashes gem5 when a mechanism config is attached.**
  Idealising `l1d_capacity` with no config works on the same kernel; adding
  `{"caches": {"l2": {"prefetcher": "StridePrefetcher"}}}` aborts the arm.
  Reproduced four times — three inside a run, once by hand. This is the
  combination the natural research progression leads to (measure a mechanism,
  attribute what now limits it, bound the remaining headroom *of that same
  machine*), so it is worth knowing before designing the study. Until it is
  fixed, bound the headroom of the baseline machine and treat the mechanism's
  effect as separately measured.

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
- `PLUGIN_BUILTIN_DIR`, `PLUGINS_USER_AUTHORING_ENABLED` - Plugin bundles shipped
  with the repo, and whether users may author their own
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
