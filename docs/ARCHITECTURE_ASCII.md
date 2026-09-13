# KnowledgeOps Lab Architecture

This is the canonical implementation-oriented architecture map for
KnowledgeOps Lab. It describes the current product boundaries, autonomous
runtime, persistence model, coding harness, and external-system interfaces.

Historical identifiers such as `KnowledgeDBChat`, `knowledge_db`, and
`knowledge_db_*` remain in repository paths, database names, and container names
for compatibility.

## Architectural intent

KnowledgeOps Lab combines four systems that share governance and evidence:

```text
┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐
│ Knowledge platform │  │ Autonomous R&D     │  │ Engineering agents │
│ ingest/search/RAG  │  │ plans/experiments  │  │ code/test/review   │
└─────────┬──────────┘  └─────────┬──────────┘  └─────────┬──────────┘
          │                       │                       │
          └───────────────────────┼───────────────────────┘
                                  v
                     ┌────────────────────────┐
                     │ Evidence + governance  │
                     │ provenance, policy,    │
                     │ checkpoints, audit     │
                     └───────────┬────────────┘
                                 │
                                 v
                     ┌────────────────────────┐
                     │ External R&D systems   │
                     │ CompOps, MLflow, MCP,  │
                     │ repositories, models   │
                     └────────────────────────┘
```

The central design rule is that reasoning and authority are separate. A model
may propose an action, but configured tools, policies, scopes, budgets, leases,
and approvals decide whether and how that action executes.

## System context

```text
                         ┌───────────────────────────────┐
                         │ Researchers / operators      │
                         │ reviewers / administrators   │
                         └───────────────┬───────────────┘
                                         │ HTTPS / WebSocket
                                         v
┌─────────────────────────────────────────────────────────────────────────────┐
│ React + TypeScript operator workspace                                      │
│ chat | documents | research | agents | swarms | experiments | tools | ops │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │ /api/v1
                                   v
┌─────────────────────────────────────────────────────────────────────────────┐
│ FastAPI control plane                                                       │
│ authentication | schemas | endpoints | policy | orchestration | streaming │
└───────┬──────────────────┬──────────────────┬───────────────────┬───────────┘
        │                  │                  │                   │
        v                  v                  v                   v
┌──────────────┐  ┌──────────────────┐  ┌────────────────┐  ┌───────────────┐
│ Knowledge    │  │ Autonomous agent │  │ Authoring and  │  │ MCP / custom  │
│ services     │  │ control plane    │  │ media services │  │ tool gateway  │
└──────┬───────┘  └─────────┬────────┘  └───────┬────────┘  └──────┬────────┘
       │                    │                   │                  │
       └────────────────────┴───────────┬───────┴──────────────────┘
                                        v
┌─────────────────────────────────────────────────────────────────────────────┐
│ Durable platform services                                                   │
│ PostgreSQL | Redis/Celery | Qdrant | MinIO | execution journal | outbox    │
└───────┬──────────────────┬──────────────────┬───────────────────┬───────────┘
        │                  │                  │                   │
        v                  v                  v                   v
  repositories       model providers      CompOps             MLflow
  web / arXiv        local / external      compiler R&D        experiment data
```

## Logical layers

```text
frontend/src/
├── pages/                  Product workspaces and operator control planes
├── components/             Domain panels, editors, inspectors, status views
├── hooks/                  WebSocket, polling, and reusable UI state
├── services/api.ts         Typed HTTP client boundary
├── types/                  Shared frontend contracts
└── utils/                  Presentation and client-side helpers

backend/app/
├── api/
│   ├── routes.py           /api/v1 composition root
│   └── endpoints/          HTTP and WebSocket transport boundary
├── schemas/                Pydantic input/output contracts
├── services/               Domain logic and orchestration
├── modules/                Incremental domain-oriented ownership boundaries
├── agent_core/             Runtime contracts, planning, routing, graph types
├── models/                 SQLAlchemy persistence entities
├── tasks/                  Celery jobs and recurring workers
├── mcp/                    Authenticated MCP server and tool adapters
├── core/                   Configuration, DB, Celery, middleware, logging
└── utils/                  Shared infrastructure helpers
```

Endpoint modules should translate transport contracts and delegate. Services
own domain behavior. Models own persisted state. Celery tasks provide execution
entry points but should reuse the same services as synchronous routes.

New extractions use `app/modules/<domain>/` with optional `api`, `application`,
`domain`, and `infrastructure` layers. Legacy aggregate modules compose these
routers and retain temporary compatibility exports while callers migrate.

## Primary domain subsystems

```text
Knowledge operations
├── sources and ingestion
├── document processing and object storage
├── embeddings, vector search, and lexical retrieval
├── RAG chat and source attribution
├── knowledge graph and memory
├── research notes, papers, reading lists, and synthesis
└── exports, presentations, diagrams, and LaTeX

Autonomous R&D
├── agent definitions, jobs, templates, and quick starts
├── domain research profiles and recurring monitors
├── research portfolios and fleets
├── inbox opportunities and follow-up jobs
├── hypotheses, experiment plans, and sandbox runs
├── evidence verification and outcome grading
└── control-plane status, interventions, and audit

Engineering agents
├── coding backlog and task decomposition
├── coding workspace sessions
├── coding harness and command/file tools
├── role-based coding swarms
├── patch proposals and PR workflows
├── workspace/candidate checkpoints
└── verifier and reviewer handoffs

External systems
├── generic external-agent gateway
├── transactional external-call outbox
├── CompOps provider, evidence sync, and signed webhooks
├── MLflow read-only evidence provider
├── MCP server and clients
└── repository, web, and enterprise connectors
```

## Autonomous job lifecycle

```text
Create job
   |
   v
Resolve effective policy, tools, scope, budget, and execution mode
   |
   v
Acquire execution lease + fencing token
   |
   v
Load latest checkpoint and reconcile incomplete journal entries
   |
   v
┌──────────────────────────────────────────────────────────────────────┐
│ Observe -> Plan -> Select action -> Policy/scope check -> Act        │
│                                      |                    |          │
│                                      | denied/approval    v          │
│                                      +--------------> Pause          │
│                                                           |          │
│                                       Verify <- tool result          │
│                                          |                           │
│                                       Summarize                       │
│                                          |                           │
│                              Evaluate progress / replan / finish      │
└──────────────────────────────────────────┬───────────────────────────┘
                                           |
                          checkpoint + journal + heartbeat
                                           |
               +---------------------------+--------------------------+
               |                           |                          |
               v                           v                          v
            complete                    paused                     failed
```

### Runtime responsibilities

| Component | Responsibility |
| --- | --- |
| `autonomous_agent_executor.py` | Job-level orchestration and runtime integration |
| `agent_core/runtime.py` | Phase runner and runtime contracts |
| `agent_execution_planner.py` | Plan construction and step contracts |
| `agent_action_service.py` | Tool-call execution boundary and journal integration |
| `agent_tool_dispatch.py` | Modular tool providers and handler dispatch |
| `agent_checkpoint_service.py` | Durable runtime-state snapshots |
| `agent_execution_journal_service.py` | Hash-chained intent/result history and recovery |
| `agent_execution_lease_service.py` | Single-owner execution, heartbeat, and fencing |
| `agent_runtime_finalizer.py` | Terminal state and result finalization |
| `agent_job_tasks.py` | Celery execution and progress publication |

## Durable execution model

The runtime uses complementary mechanisms rather than one generic “state” blob:

```text
AgentJob
├── status, phase, progress, limits, and current result summary
├── execution lease owner / expiry / fence
└── current runtime identity

AgentJobCheckpoint
├── normalized runtime state
├── plan and active step
├── findings and artifacts
├── tool statistics and execution graph
└── journal cursor

Execution journal
├── tool intent
├── stable invocation and idempotency identity
├── tool result
├── hash linkage
└── reconciliation metadata

External-call outbox
├── capability-scoped request
├── delivery claim and retry state
├── response and correlation metadata
└── resume claim and enqueue marker
```

The lease prevents simultaneous owners. Fencing prevents a stale owner from
writing after losing its lease. The journal makes partial tool calls
reconcilable. Checkpoints make the runtime restartable. The outbox bridges the
database transaction and asynchronous external side effects.

## Coding harness architecture

```text
Coding job / backlog slice
          |
          v
Harness policy + workspace session
          |
          +--> Planner role ------> task graph and acceptance criteria
          |
          +--> Implementer role --> inspect / patch / create / command
          |                              |
          |                              v
          |                         candidate snapshot
          |
          +--> Verifier role ------> tests / lint / build / evidence
          |
          +--> Reviewer role ------> accept / reject / request repair
          |
          v
Verified artifact, patch proposal, PR handoff, or recoverable checkpoint
```

Workspace sessions provide a bounded filesystem identity. Candidate snapshots
record reviewable states. Durable workspace checkpoints survive process
restarts. Swarm collaboration metadata supports role assignment, fan-out, and
fan-in without giving every role the same authority.

## External-call and automatic-resume lifecycle

```text
Agent selects external capability
          |
          v
Policy, connection, capability, and payload validation
          |
          v
Atomically persist:
  journal tool result + outbox request + waiting plan state
          |
          v
Job status = paused / phase = awaiting_external
          |
          v
Celery claims outbox row -> invokes external gateway -> stores response
          |
          v
Response correlator claims successful response
          |
          +--> merge bounded response into latest checkpoint
          +--> complete waiting plan step
          +--> activate next plan step
          +--> set job pending
          +--> enqueue one resume task
          |
          v
Execution lease arbitrates duplicate task delivery
```

Raw or oversized source artifacts remain in the external system. The local
checkpoint stores bounded response data and stable remote references.

## Evidence and verification boundary

```text
Observation / tool result / external response
                    |
                    v
             Evidence ledger
                    |
          provenance + confidence
                    |
                    v
       Verification planner and tasks
          |          |           |
          v          v           v
       local test  sandbox run  independent grader
          |          |           |
          +----------+-----------+
                     |
                     v
         accepted / disputed / rejected
                     |
                     v
          outcome grade + audit snapshot
```

External responses enter as unverified evidence. Verification policies decide
whether local tests, scientific sandboxes, graders, or operator approval are
required before an outcome is accepted.

## External-system ownership

| System | KnowledgeOps Lab owns | External system owns |
| --- | --- | --- |
| CompOps | Connection policy, request identity, bounded provenance, evidence links, verification state | Compiler execution, raw IR, logs, artifacts, study state |
| MLflow | Read-only connection, selected metadata, evidence links, local verification | Experiments, runs, metrics, artifacts, model registry |
| GitHub/GitLab | Indexed knowledge, bounded repo analysis, proposed patches/PR metadata | Repository history, branches, issues, merge authority |
| MCP client/server | Authentication, exposed capability contract, audit records | Client reasoning and downstream use |
| Model provider | Routing policy, prompt construction, usage and snapshot metadata | Inference infrastructure and provider-side processing |

## Deployment topology

```text
nginx
├── React frontend
├── FastAPI backend
└── MinIO proxy routes

FastAPI backend
├── PostgreSQL
├── Redis
├── Qdrant
├── MinIO
├── Ollama and optional external model providers
├── Kroki
└── video-streamer

Redis broker
├── general Celery worker
├── isolated LaTeX worker
└── Celery Beat schedules in production
```

Development Compose bind-mounts the backend and frontend source trees. Database,
object, vector, cache, and model data live under `data/` and are runtime state,
not source logic.

## Trust boundaries

```text
Untrusted or conditionally trusted
├── user prompts and uploaded files
├── retrieved web/repository content
├── model output
├── external-agent responses
├── CompOps and MLflow metadata
└── generated code and commands

Enforcement boundaries
├── authentication and ownership checks
├── schema and payload validation
├── secret indirection
├── tool registry and policy engine
├── approval checkpoints
├── domain/private-network restrictions
├── coding workspace and command policy
├── scientific sandbox profile
├── execution budgets and leases
└── evidence verification and audit
```

## Operator surfaces

The frontend exposes domain-specific workspaces rather than a single agent chat:

```text
Knowledge
├── Chat, Documents, Search, Knowledge Graph
├── Papers, Notes, Reading Lists, Synthesis
└── Presentations, LaTeX, Templates, Exports

Autonomous R&D
├── Jobs, Quick Starts, Templates, Job Chains
├── Checkpoint Queue and interventions
├── Domain Profiles, Research Fleet, Research Inbox
├── Experiments and scientific sandboxes
└── Evaluation, verification, and audit views

Engineering
├── Coding Backlog
├── Swarm Profiles, Review, and Outcomes
├── Repository Reports
└── Patch and PR workflows

Operations
├── Tools, CompOps, and MLflow connections
├── Tool policies and approvals
├── Routing and control-plane observability
├── Models, usage, and secrets
└── Health and administration
```

## Related documentation

- [Architecture diagrams](ARCHITECTURE_DIAGRAMS.md)
- [Single-slide Mermaid map](knowledgeops_architecture_slide.mmd)
- [Autonomous R&D agents](AUTONOMOUS_RND_AGENTS.md)
- [CompOps integration](COMPOPS_INTEGRATION.md)
- [MLflow integration](MLFLOW_INTEGRATION.md)
- [External Agent Gateway](EXTERNAL_AGENT_GATEWAY.md)
- [Data ingestion](INGESTION_GUIDE.md)
- [Knowledge graph](KNOWLEDGE_GRAPH.md)
- [LaTeX Studio](LATEX_STUDIO.md)
