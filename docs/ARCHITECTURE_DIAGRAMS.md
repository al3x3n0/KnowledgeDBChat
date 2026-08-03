# KnowledgeOps Lab Architecture Diagrams

These Mermaid diagrams complement the canonical
[text architecture](ARCHITECTURE_ASCII.md). They intentionally show stable
subsystem boundaries instead of file, endpoint, model, or migration counts.

GitHub renders Mermaid inline. The local development stack also includes Kroki
at `http://localhost:8001`.

## 1. Product context

```mermaid
flowchart LR
    PEOPLE["Researchers, engineers,<br/>reviewers, operators"]

    subgraph KOL["KnowledgeOps Lab"]
        UI["Operator workspace<br/>React + TypeScript"]
        API["Control plane<br/>FastAPI"]
        KNOW["Knowledge operations<br/>ingest, retrieve, RAG, graph"]
        RND["Autonomous R&D<br/>plan, investigate, experiment"]
        CODE["Engineering agents<br/>workspace, code, test, review"]
        EVID["Evidence and governance<br/>policy, provenance, verification, audit"]
    end

    subgraph EXT["External systems"]
        REPOS["GitHub, GitLab,<br/>Confluence, web, arXiv"]
        MODELS["Ollama and<br/>external model providers"]
        COMPOPS["CompOps<br/>compiler research"]
        MLFLOW["MLflow<br/>experiment tracking"]
        MCP["MCP clients<br/>and external agents"]
    end

    PEOPLE --> UI --> API
    API --> KNOW
    API --> RND
    API --> CODE
    KNOW --> EVID
    RND --> EVID
    CODE --> EVID
    KNOW <--> REPOS
    API <--> MODELS
    RND <--> COMPOPS
    RND <--> MLFLOW
    API <--> MCP
```

## 2. Deployment topology

```mermaid
flowchart TB
    CLIENT["Browser / MCP client"]

    subgraph EDGE["Application edge"]
        NGINX["nginx"]
        FRONT["React frontend"]
        API["FastAPI backend"]
        VIDEO["Go video-streamer"]
    end

    subgraph WORKERS["Asynchronous execution"]
        REDIS[("Redis<br/>broker, cache, pub/sub")]
        CELERY["General Celery worker"]
        LATEX["Isolated LaTeX worker"]
        BEAT["Celery Beat<br/>production schedules"]
    end

    subgraph STATE["Durable state"]
        PG[("PostgreSQL")]
        QDRANT[("Qdrant")]
        MINIO[("MinIO")]
    end

    subgraph LOCAL["Local auxiliary services"]
        OLLAMA["Ollama"]
        KROKI["Kroki"]
    end

    PROVIDERS["External models,<br/>repositories, CompOps, MLflow"]

    CLIENT --> NGINX
    NGINX --> FRONT
    NGINX --> API
    NGINX --> MINIO
    FRONT --> API
    FRONT --> VIDEO
    API --> PG
    API --> QDRANT
    API --> MINIO
    API --> REDIS
    API --> OLLAMA
    API --> KROKI
    API <--> PROVIDERS
    BEAT --> REDIS
    REDIS --> CELERY
    REDIS --> LATEX
    CELERY --> PG
    CELERY --> QDRANT
    CELERY --> MINIO
    CELERY --> OLLAMA
    CELERY <--> PROVIDERS
    LATEX --> MINIO
    VIDEO --> MINIO
```

## 3. Backend subsystem map

```mermaid
flowchart TB
    ROUTES["API endpoints and WebSockets"]
    SCHEMAS["Pydantic contracts"]

    subgraph DOMAINS["Domain services"]
        KNOW["Knowledge<br/>documents, retrieval, graph, memory"]
        RESEARCH["Research<br/>papers, notes, portfolios, experiments"]
        AGENTS["Autonomy<br/>jobs, planning, tools, runtime"]
        CODING["Coding<br/>workspaces, swarms, patches"]
        AUTHOR["Authoring<br/>reports, slides, diagrams, LaTeX"]
        EXTERNAL["External systems<br/>gateway, CompOps, MLflow, MCP"]
        GOVERN["Governance<br/>policy, approvals, secrets, audit"]
    end

    subgraph INFRA["Infrastructure services"]
        TASKS["Celery tasks"]
        MODELS["SQLAlchemy models"]
        VECTOR["Vector and lexical search"]
        OBJECTS["Object storage"]
        LLM["LLM routing"]
    end

    ROUTES --> SCHEMAS
    SCHEMAS --> KNOW
    SCHEMAS --> RESEARCH
    SCHEMAS --> AGENTS
    SCHEMAS --> CODING
    SCHEMAS --> AUTHOR
    SCHEMAS --> EXTERNAL
    KNOW --> GOVERN
    RESEARCH --> GOVERN
    AGENTS --> GOVERN
    CODING --> GOVERN
    EXTERNAL --> GOVERN
    KNOW --> MODELS
    RESEARCH --> MODELS
    AGENTS --> MODELS
    CODING --> MODELS
    AUTHOR --> MODELS
    EXTERNAL --> MODELS
    KNOW --> VECTOR
    KNOW --> OBJECTS
    AGENTS --> LLM
    AUTHOR --> LLM
    AGENTS --> TASKS
    RESEARCH --> TASKS
    EXTERNAL --> TASKS
```

## 4. Autonomous R&D runtime

```mermaid
flowchart TD
    CREATE["Create job<br/>goal + configuration"]
    POLICY["Resolve effective policy<br/>scope + tools + budget + mode"]
    LEASE["Acquire execution lease<br/>and fencing token"]
    RESTORE["Load checkpoint<br/>reconcile journal"]
    OBSERVE["Observe"]
    PLAN["Plan / replan"]
    SELECT["Select action"]
    GUARD{"Policy, scope,<br/>budget allowed?"}
    APPROVAL["Pause for approval<br/>or intervention"]
    ACT["Execute tool"]
    DEFER{"Deferred<br/>external call?"}
    WAIT["Checkpoint waiting state<br/>pause job"]
    VERIFY["Verify"]
    SUMMARY["Summarize"]
    PROGRESS{"Goal complete,<br/>blocked, or continue?"}
    CHECKPOINT["Save checkpoint<br/>renew lease"]
    FINISH["Finalize outcome<br/>evidence + audit"]

    CREATE --> POLICY --> LEASE --> RESTORE --> OBSERVE --> PLAN --> SELECT --> GUARD
    GUARD -- "denied or approval" --> APPROVAL
    APPROVAL -- "approved / resumed" --> RESTORE
    GUARD -- "allowed" --> ACT --> DEFER
    DEFER -- "yes" --> WAIT
    WAIT -- "response correlated" --> RESTORE
    DEFER -- "no" --> VERIFY --> SUMMARY --> PROGRESS
    PROGRESS -- "continue" --> CHECKPOINT --> OBSERVE
    PROGRESS -- "replan" --> PLAN
    PROGRESS -- "complete or terminal" --> FINISH
```

## 5. Durable execution and recovery

```mermaid
flowchart LR
    WORKER["Celery execution worker"]
    LEASE{"Execution lease<br/>owner + expiry + fence"}
    JOB[("AgentJob<br/>status, phase, budgets")]
    CHECKPOINT[("Checkpoint<br/>normalized runtime state")]
    JOURNAL[("Hash-chained journal<br/>intent and result")]
    OUTBOX[("External-call outbox<br/>delivery and resume claims")]
    TOOL["Tool or external side effect"]

    WORKER --> LEASE
    LEASE -- "acquired" --> JOB
    LEASE -- "conflict" --> STOP["Skip duplicate delivery"]
    WORKER --> CHECKPOINT
    WORKER --> JOURNAL
    JOURNAL -- "persist intent" --> TOOL
    TOOL -- "result" --> JOURNAL
    JOURNAL --> CHECKPOINT
    TOOL -- "asynchronous request" --> OUTBOX
    OUTBOX -- "response" --> CHECKPOINT
    CHECKPOINT -- "restart / resume" --> WORKER
    LEASE -- "heartbeat and fenced writes" --> WORKER
```

The mechanisms have different jobs: leases arbitrate ownership, fencing rejects
stale owners, journal entries reconcile partial calls, checkpoints restore
runtime state, and the outbox bridges transactions with asynchronous systems.

## 6. Coding harness and swarm

```mermaid
flowchart TD
    REQUEST["Coding goal or backlog slice"]
    SCOPE["Task contract<br/>workspace + policy + acceptance criteria"]
    PLAN["Planner role<br/>task graph"]

    subgraph SWARM["Bounded role swarm"]
        IMPLEMENT["Implementer<br/>inspect, patch, create, command"]
        SPECIALIST["Specialist roles<br/>frontend, backend, tests, research"]
        VERIFY["Verifier<br/>tests, lint, build, evidence"]
        REVIEW["Reviewer<br/>accept, reject, request repair"]
    end

    SESSION[("Workspace session")]
    SNAPSHOT[("Candidate snapshot")]
    DURABLE[("Durable checkpoint")]
    OUTPUT["Verified patch, artifact,<br/>PR handoff, or recovery state"]

    REQUEST --> SCOPE --> PLAN
    PLAN --> IMPLEMENT
    PLAN --> SPECIALIST
    IMPLEMENT --> SESSION
    SPECIALIST --> SESSION
    SESSION --> SNAPSHOT
    SESSION --> DURABLE
    SNAPSHOT --> VERIFY --> REVIEW
    REVIEW -- "repair" --> IMPLEMENT
    REVIEW -- "accepted" --> OUTPUT
    DURABLE -- "restart recovery" --> SESSION
```

## 7. External call, correlation, and resume

```mermaid
sequenceDiagram
    participant A as Agent runtime
    participant DB as PostgreSQL
    participant W as Outbox worker
    participant X as External system
    participant R as Response correlator
    participant Q as Agent task queue

    A->>A: Validate tool policy, connection, capability, payload
    A->>DB: Commit journal result + outbox row + waiting plan state
    A->>DB: Set job paused / awaiting_external
    W->>DB: Claim due outbox row
    W->>X: Invoke with stable request identity
    X-->>W: Return bounded response
    W->>DB: Fence acknowledgement and store response
    R->>DB: Claim successful uncorrelated response
    R->>DB: Merge response into checkpoint
    R->>DB: Complete waiting step and set job pending
    R->>Q: Enqueue resume task
    R->>DB: Mark resume enqueued
    Q->>DB: Acquire execution lease
    Q->>A: Restore checkpoint and continue next step
```

Delivery is retryable and idempotent. The task queue can provide at-least-once
delivery; the resume marker and execution lease prevent duplicate active
execution.

## 8. Evidence verification

```mermaid
flowchart TD
    SOURCE["Finding, artifact,<br/>tool result, external response"]
    LEDGER[("Evidence ledger<br/>provenance + confidence")]
    PLANNER["Verification planner"]

    subgraph CHECKS["Verification work"]
        LOCAL["Local deterministic checks"]
        SANDBOX["Scientific sandbox run"]
        EXTERNAL["External-system attestation"]
        GRADER["Independent grader"]
        HUMAN["Operator approval"]
    end

    RECONCILE["Verification reconciliation"]
    DECISION{"Evidence status"}
    ACCEPT["Accepted outcome"]
    DISPUTE["Disputed / needs work"]
    REJECT["Rejected"]
    AUDIT[("Trajectory grade<br/>signed audit snapshot")]

    SOURCE --> LEDGER --> PLANNER
    PLANNER --> LOCAL
    PLANNER --> SANDBOX
    PLANNER --> EXTERNAL
    PLANNER --> GRADER
    PLANNER --> HUMAN
    LOCAL --> RECONCILE
    SANDBOX --> RECONCILE
    EXTERNAL --> RECONCILE
    GRADER --> RECONCILE
    HUMAN --> RECONCILE
    RECONCILE --> DECISION
    DECISION --> ACCEPT
    DECISION --> DISPUTE
    DECISION --> REJECT
    ACCEPT --> AUDIT
    DISPUTE --> AUDIT
    REJECT --> AUDIT
```

## 9. External-system ownership

```mermaid
flowchart LR
    subgraph KOL["KnowledgeOps Lab owns"]
        POLICY["Policy and approval"]
        IDENTITY["Request identity<br/>and correlation"]
        PROV["Bounded provenance<br/>and evidence links"]
        VERIFY["Local verification<br/>and outcome state"]
    end

    subgraph SOURCE["Source systems own"]
        COMPILER["CompOps<br/>runs, raw IR, logs, artifacts"]
        TRACKING["MLflow<br/>experiments, metrics, artifacts, registry"]
        REPO["Repositories<br/>history, branches, merge authority"]
        PROVIDER["Model providers<br/>inference infrastructure"]
    end

    POLICY --> IDENTITY --> PROV --> VERIFY
    IDENTITY <--> COMPILER
    IDENTITY <--> TRACKING
    IDENTITY <--> REPO
    IDENTITY <--> PROVIDER
```

KnowledgeOps Lab stores enough local state to reproduce decisions and audit
provenance without silently becoming the system of record for every remote
artifact.

For a compact presentation view, use
[the single-slide Mermaid map](knowledgeops_architecture_slide.mmd).
