# KnowledgeOps Lab

KnowledgeOps Lab is a self-hosted platform for knowledge operations and
autonomous research and development. It connects organizational knowledge,
research agents, coding harnesses, scientific verification, and external
engineering systems in one governed workspace.

The platform can ingest and search technical material, run long-lived research
programs, execute bounded coding workflows, collect experimental evidence, and
produce auditable reports. Local models are supported through Ollama, while
external model and research-system access remains policy controlled.

> Status: active development. The product name is KnowledgeOps Lab; some
> internal package, database, container, and repository identifiers still use
> the historical `KnowledgeDBChat` or `knowledge_db` names for compatibility.

## What it does

| Area | Capabilities |
| --- | --- |
| Knowledge operations | Document ingestion, semantic and lexical retrieval, RAG chat, citations, knowledge graphs, notes, reading lists, synthesis, and exports |
| Autonomous R&D | Goal decomposition, plans, recurring monitors, research fleets, hypotheses, experiments, verification, checkpoints, approvals, and outcome grading |
| Coding agents | Isolated workspaces, file and command tools, role-based swarms, verification gates, candidate snapshots, durable checkpoints, and recovery |
| Research evidence | Evidence ledgers, provenance, external-system links, bounded response storage, verification plans, trajectory evaluation, and signed audit snapshots |
| External systems | MCP, GitHub, GitLab, Confluence, web and arXiv connectors, CompOps compiler research, and MLflow experiment evidence |
| Authoring | Research reports, presentations, diagrams, LaTeX projects, citation synchronization, and document generation |
| Governance | Authentication, scoped secrets, tool policies, approval gates, execution budgets, audit trails, leases, fencing, and idempotency |

## System at a glance

```text
Researchers and operators
          |
          v
React operator workspace
          |
          v
FastAPI control plane ----------------------------------+
  |                 |                 |                 |
  v                 v                 v                 v
Knowledge       Autonomous R&D    Coding harness    MCP / external
services        agent runtime     and swarms        system gateway
  |                 |                 |                 |
  +-----------------+-----------------+-----------------+
                            |
                            v
               Durable execution and evidence
        checkpoints | journal | leases | outbox | audit
                            |
          +-----------------+------------------+
          |                 |                  |
          v                 v                  v
      PostgreSQL       Qdrant/MinIO       Redis/Celery
          |
          +----> CompOps | MLflow | repositories | model providers
```

See [Architecture](docs/ARCHITECTURE_ASCII.md) for the canonical text map and
[Architecture diagrams](docs/ARCHITECTURE_DIAGRAMS.md) for Mermaid views.

## Autonomous execution model

An autonomous job is a durable state machine, not a single model request:

```text
goal and policy
      |
      v
observe -> plan -> choose tool -> act -> verify -> summarize
  ^                                              |
  +--------- checkpoint / replan / resume -------+
```

The runtime separates model reasoning from authority:

- Models propose actions.
- The tool catalog and policy engine determine what is available.
- Scope guards, budgets, and approvals determine what may execute.
- The action service journals intent and results.
- Checkpoints make jobs restartable.
- Execution leases and fencing prevent concurrent workers from owning the same
  job.
- Transactional outbox rows make asynchronous external calls recoverable.
- External results return as untrusted evidence and can trigger local
  verification before they influence an accepted outcome.

The canonical autonomy configuration contract is:

- `automation_profile`
- `automation_policy`
- `effective_policy`

Legacy fields such as `validation_policy` and `follow_up_autonomy` are
compatibility mirrors, not the preferred API.

## Coding harness and swarms

Coding agents operate through bounded workspace sessions. Depending on policy,
they can inspect and create files, apply patches, run allowlisted commands, call
tools, checkpoint work, and hand candidates to verifier or reviewer roles.

The modern harness includes:

- Planner, implementer, verifier, reviewer, and specialist roles
- Explicit task and scope contracts
- Workspace-local file and command tools
- Candidate snapshots and recovery handoffs
- Durable workspace checkpoints
- Plan/act/verify/summarize execution graphs
- Tool-result journals and stable idempotency keys
- Database-backed execution leases with heartbeat and fencing
- Fan-out/fan-in swarm collaboration

Agents do not receive unrestricted host or network access. Effective capability
is always the intersection of the agent definition, runtime mode, configured
tools, tool policy, workspace boundary, and operator approvals.

## External research systems

### CompOps

CompOps remains the system of record for compiler and microarchitecture
experiments. KnowledgeOps Lab can submit allowlisted operations, follow bounded
study/run/artifact evidence, receive signed refresh webhooks, and correlate
results with a waiting agent plan.

See [CompOps integration](docs/COMPOPS_INTEGRATION.md).

### MLflow

MLflow is supported as a typed, read-only evidence provider. Agents and
operators can import experiment, run, metric, and artifact metadata without
granting model-registry mutation authority.

See [MLflow integration](docs/MLFLOW_INTEGRATION.md).

### MCP and custom tools

The backend exposes an MCP-compatible tool surface for authenticated external
clients. User tools can also represent external agents, webhooks, transforms,
containers, prompts, and workflow runners. Every tool call passes through the
same policy and audit boundary.

See [External Agent Gateway](docs/EXTERNAL_AGENT_GATEWAY.md).

## Quick start

### Prerequisites

- Docker Desktop or Docker Engine with Compose
- At least 8 GB RAM for the default local stack; more for larger local models
- Git and Make

### Start the development stack

```bash
git clone <repository-url>
cd KnowledgeDBChat
make setup
make build
make start
```

Pull the default local model:

```bash
make pull-model
```

Apply or verify database migrations:

```bash
docker compose exec backend alembic upgrade head
docker compose exec backend alembic current
```

Open:

- Web application: <http://localhost:3000>
- Backend API: <http://localhost:8000>
- OpenAPI documentation: <http://localhost:8000/docs>
- MinIO console: <http://localhost:9001>
- Kroki diagram service: <http://localhost:8001>

Check the environment and service health:

```bash
make doctor
make health
```

### Run on Kubernetes

A Helm chart covers the same stack on Kubernetes, with minikube tooling for
local clusters:

```bash
make minikube-up          # start minikube, build images into it, install the chart
make k8s-status
make k8s-test             # in-cluster smoke test
```

The chart is `deploy/helm/knowledgedbchat`. It ships every dependency
(PostgreSQL, Redis, Qdrant, MinIO, Kroki, optional Ollama) so a release is
self-contained, and each one can be switched off in favour of a managed service.
Alembic runs in a hook Job rather than in the app containers, so replicas never
race the schema and a failed migration aborts the upgrade before any pod rolls.
See [deploy/README.md](deploy/README.md) for the production checklist.

### Configuration

`make setup` creates local environment files from:

- `backend/env.example`
- `frontend/.env.example`

Important backend settings include:

```dotenv
DATABASE_URL=postgresql://user:password@postgres:5432/knowledge_db
REDIS_URL=redis://redis:6379/0
VECTOR_STORE_PROVIDER=qdrant
QDRANT_URL=http://qdrant:6333
MINIO_ENDPOINT=minio:9000

LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://ollama:11434
DEFAULT_MODEL=llama3.2:1b

SECRET_KEY=replace-me
CUSTOM_TOOL_DOCKER_ENABLED=false
AGENT_KB_PATCH_APPLY_ENABLED=false
```

Never commit credentials. Store external-system credentials through the scoped
secret APIs and reference secret IDs from tool configurations.

## Development

### Repository layout

```text
KnowledgeDBChat/
├── backend/
│   ├── app/
│   │   ├── agent_core/       runtime contracts and graph primitives
│   │   ├── api/              FastAPI routes and endpoints
│   │   ├── core/             configuration, database, Celery, middleware
│   │   ├── mcp/              MCP server and tools
│   │   ├── modules/          Domain-oriented modular-monolith boundaries
│   │   ├── models/           SQLAlchemy persistence models
│   │   ├── schemas/          API and service contracts
│   │   ├── services/         domain logic and agent runtime services
│   │   └── tasks/            Celery workers and scheduled jobs
│   ├── alembic/              database migrations
│   └── tests/                backend regression and integration tests
├── frontend/
│   └── src/
│       ├── components/       operator and domain components
│       ├── hooks/            reusable UI/runtime hooks
│       ├── pages/            product workspaces
│       ├── services/         typed API client
│       ├── types/            TypeScript contracts
│       └── utils/            shared frontend utilities
├── video-streamer/           Go media-streaming service
├── scripts/                  setup, validation, evaluation, and operations
├── docs/                     architecture and operator documentation
├── backend/evals/            autonomous R&D evaluation suites
├── data/                     local runtime data; not source code
└── test_documents/           test fixtures and sample inputs
```

### Common commands

```bash
make setup                 # Create local directories and environment files
make build                 # Build the Docker stack
make start                 # Start services
make stop                  # Stop services
make logs-backend          # Follow API logs
make logs-celery           # Follow agent/task logs

make test-backend          # Backend suite
make test-frontend         # Frontend suite
make test-rnd-evals        # Autonomous R&D evaluation regressions
make test-external-agents  # External gateway regressions
make typecheck-frontend    # TypeScript validation
make fmt                   # Python formatting
make lint                  # Python lint
```

For manual development, use `make dev-backend`, `make dev-frontend`, and
`make dev-celery`.

## Safety and trust boundaries

KnowledgeOps Lab is designed for controlled autonomy:

- Network and write-capable tools can require approval.
- External responses are evidence, not trusted instructions.
- Private-network and domain constraints can be applied to network tools.
- Coding commands run inside a scoped workspace and safety policy.
- Scientific validation uses explicit sandbox profiles.
- Destructive or high-impact actions should remain approval gated.
- Raw compiler artifacts and complete MLflow payloads remain in their source
  systems unless explicitly imported.

Local deployment does not by itself guarantee that data stays local. Enabling an
external LLM, connector, agent, webhook, or research provider sends bounded data
to that configured system.

## Documentation

- [Architecture](docs/ARCHITECTURE_ASCII.md)
- [Kubernetes / Helm deployment](deploy/README.md)
- [Architecture diagrams](docs/ARCHITECTURE_DIAGRAMS.md)
- [Autonomous R&D agents](docs/AUTONOMOUS_RND_AGENTS.md)
- [CompOps integration](docs/COMPOPS_INTEGRATION.md)
- [MLflow integration](docs/MLFLOW_INTEGRATION.md)
- [External Agent Gateway](docs/EXTERNAL_AGENT_GATEWAY.md)
- [Data ingestion](docs/INGESTION_GUIDE.md)
- [Knowledge graph](docs/KNOWLEDGE_GRAPH.md)
- [LaTeX Studio](docs/LATEX_STUDIO.md)
- [Research lab pilot](docs/pilots/research_lab_pilot.md)
