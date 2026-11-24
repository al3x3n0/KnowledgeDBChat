# KnowledgeOps Lab

KnowledgeOps Lab is a self-hosted platform for organizational knowledge retrieval
and autonomous research and development. It provides context to external AI
agents through an MCP-compatible tool API. Authenticated agents can search the
knowledge base, retrieve documents, and ask questions grounded in that material.

Research agents can run scheduled campaigns, use findings to prioritize
follow-up work, and track validation experiments within configured budgets and
approval rules. The platform can also execute bounded coding workflows, collect
experimental evidence, and produce auditable reports. Local models are
supported through Ollama, and policies govern access to external models and
research systems.

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
| Kubernetes execution | Helm deployment, optional KEDA worker scaling, isolated experiment Jobs, shared artifacts, and journal-based recovery |
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

### MCP context and custom tools

Open **MCP Profiles → New MCP profile** to name a profile, pick installed
research programs, and choose knowledge tools. Use the generated URL
(`/api/v1/mcp`) with a Streamable HTTP client and send the one-time key in the
`Authorization: Bearer <key>` header. The key works on MCP routes and can be
revoked under **API Keys**.

In a **Context Pack**, choose **Save as Research Note** to keep the retrieved
context and its source citations. Review the note and choose **Publish as MCP
profile** to open the builder with that note selected.
From **Admin → Research Programs**, use the program’s **Create MCP** link.
In **Documents**, select an active source and choose **Create MCP from source**
to start with search and document browsing scoped to that source.
Expand **Access preview** in the builder to review the selected tools,
resources, and document scope before saving.

`get_research_programs` returns the selected programs' descriptions, versions,
and tool schemas. Each selected program also appears as a JSON resource in
clients with resource browsing. Choose semantic search, document
retrieval, source listing, or knowledge-base chat. Use **Limit to selected
sources** to restrict those tools to specific document sources. The profile's
**Sources** tab lets you adjust read, search, and chat permissions after creation.
Use **Edit profile** in the profile's **General** tab to update the name,
programs, research notes, tools, and source selection. The **Connection** tab has
the endpoint, client setup, and key rotation.
Clients can call `get_research_notes` to list the selected notes or pass a
`note_id` to read that note's current Markdown. Clients with resource browsing
can also attach selected notes as Markdown resources. [MCP resources specification](https://modelcontextprotocol.io/specification/2025-11-25/server/resources).
Clients with prompt support can select **Research brief** to work from the
profile’s available resources or **Compare research notes** when at least two
notes are available. [MCP prompts specification](https://modelcontextprotocol.io/specification/2025-11-25/server/prompts).
In the builder’s **Custom prompts** section, add a title, client-facing name,
and instructions to publish your own prompt. Each prompt can optionally
reference the resources the profile is allowed to share.
Turn on **Enable prompt inputs** and use placeholders such as `{{topic}}` in the
instructions to ask for new values each time the prompt is used.

Open **Client setup** after creating a profile or in its **General** tab to
copy a configuration for GitHub Copilot or Cursor. Set the displayed environment
variable to the profile key on the host running the client. The copied JSON
references that variable. [GitHub Copilot configuration](https://code.visualstudio.com/docs/agents/reference/mcp-configuration), [Cursor configuration](https://cursor.com/docs/mcp).
Use **Test connection** in the same panel to check the endpoint from the browser
and see the tools, resources, and prompts the key can discover.
Use **Rotate key** in the profile’s **General** tab to replace its credential.
Save the replacement when it appears and update each connected client. The old
key stops authenticating after rotation.
The profile’s **Usage** tab shows request counts and frequent MCP operations,
including tool calls, resource discovery, and resource reads.

HTTP integrations can also discover tools at `GET /api/v1/mcp/tools` and invoke
them at `POST /api/v1/mcp/tools/call`.

For autonomous research campaigns, recurring monitoring, and validation, see
the [Autonomous R&D guide](docs/AUTONOMOUS_RND_AGENTS.md).

The platform can also invoke registered external agents, webhooks, transforms,
containers, prompts, and workflows as user tools. See the
[External Agent Gateway](docs/EXTERNAL_AGENT_GATEWAY.md) for agent registration
and invocation.

### Kubernetes

Kubernetes hosts the application and can run sandbox experiments as independent
Jobs. Celery workers submit bounded experiments, then supported journaled agent
actions release their worker slot while waiting. Celery Beat polls completion
and queues a continuation that restores the result and workspace. Optional KEDA
scaling adjusts worker pools independently of experiment Jobs.

See [Kubernetes integration](docs/KUBERNETES_INTEGRATION.md) for configuration,
execution boundaries, recovery, and troubleshooting.

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

- Web application: <http://localhost:23000>
- Backend API: <http://localhost:28000>
- OpenAPI documentation: <http://localhost:28000/docs>
- MinIO console: <http://localhost:29001>
- Kroki diagram service: <http://localhost:28001>

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
See [deploy/README.md](deploy/README.md) for the production checklist and
[Kubernetes integration](docs/KUBERNETES_INTEGRATION.md) to enable independent
experiment Jobs and optional worker autoscaling. The minikube profile alone
does not enable those features; experiment Jobs require shared RWX storage and
NetworkPolicy enforcement.

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

LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=replace-me
# Must name a model the chosen provider serves: it reaches the request as
# `model or <PROVIDER>_MODEL`. For LLM_PROVIDER=ollama, point OLLAMA_BASE_URL
# at an instance you run yourself — the stack does not bundle one.
DEFAULT_MODEL=deepseek-v4-pro

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

- [LDAP / Active Directory configuration](BUILD_AND_RUN.md#ldap--active-directory-configuration-optional)
- [Architecture](docs/ARCHITECTURE_ASCII.md)
- [Kubernetes / Helm deployment](deploy/README.md)
- [Kubernetes integration and experiment Jobs](docs/KUBERNETES_INTEGRATION.md)
- [Architecture diagrams](docs/ARCHITECTURE_DIAGRAMS.md)
- [Autonomous R&D agents](docs/AUTONOMOUS_RND_AGENTS.md)
- [CompOps integration](docs/COMPOPS_INTEGRATION.md)
- [MLflow integration](docs/MLFLOW_INTEGRATION.md)
- [External Agent Gateway](docs/EXTERNAL_AGENT_GATEWAY.md)
- [Data ingestion](docs/INGESTION_GUIDE.md)
- [Knowledge graph](docs/KNOWLEDGE_GRAPH.md)
- [LaTeX Studio](docs/LATEX_STUDIO.md)
- [Research lab pilot](docs/pilots/research_lab_pilot.md)
