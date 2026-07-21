# KnowledgeDBChat Architecture

This is the current in-repo architecture map for the project. It is intended to be implementation-faithful to the codebase as it exists today, with a special focus on the autonomous RnD runtime slice.

## Top-Level System

```text
KnowledgeDBChat
├── frontend/                React + TypeScript operator UI
├── backend/                 FastAPI + Celery + DB-backed platform
├── video-streamer/          Go media service
├── scripts/                 Ops / validation helpers
├── data/                    Runtime data
└── test_documents/          Test assets
```

## Runtime Architecture

```text
[Operator / User]
        |
        v
[Frontend React UI]
        |
        v
[FastAPI endpoints]
        |
        +--> [Schemas]
        +--> [Services]
        |       |
        |       +--> [SQLAlchemy models + Postgres]
        |       +--> [Vector store / retrieval / documents]
        |       +--> [LLM routing + agent orchestration]
        |       +--> [Connectors: GitHub, GitLab, Confluence, Web, arXiv]
        |       +--> [Scientific validation + sandbox policy]
        |
        +--> [Celery tasks]
        +--> [MCP server + tools]
        +--> [Video streamer integration]
```

## Backend Layout

```text
backend/app/
├── api/endpoints/           HTTP route layer
├── schemas/                 Request / response contracts
├── models/                  Persisted DB entities
├── services/                Core product logic
│   ├── autonomy_service.py
│   ├── autonomous_agent_executor.py
│   ├── scientific_validation_service.py
│   ├── research_monitor_profile_service.py
│   ├── research_opportunity_service.py
│   ├── agent_job_templates.py
│   ├── llm_service.py / llm_routing.py
│   ├── document_service.py / vector_store.py / search_service.py
│   └── connectors/
├── tasks/                   Celery task execution
├── mcp/                     MCP server + tools
├── core/                    Config, DB, celery, middleware
├── utils/                   Shared helpers
└── plugins/ai_hub/          Built-in presets and eval templates
```

## Frontend Layout

```text
frontend/src/
├── pages/
│   └── AutonomousAgentsPage.tsx
├── components/
├── services/
├── types/
├── hooks/
├── contexts/
└── utils/
```

## Autonomous RnD Runtime Slice

The canonical autonomy contract is:

- `automation_profile`
- `automation_policy`
- `effective_policy`

Legacy fields such as `validation_policy` and monitor `follow_up_autonomy` are compatibility mirrors, not the primary architecture.

```text
AutonomousAgentsPage
   |
   +--> agent_jobs endpoints
   +--> domain_research_profiles endpoints
   +--> research_portfolios endpoints
   +--> research_monitor_profiles endpoints
   +--> experiments endpoints
            |
            v
   autonomy_service.py
            |
            +--> autonomous_agent_executor.py
            +--> scientific_validation_service.py
            +--> research_monitor_profile_service.py
            +--> research_opportunity_service.py
            |
            v
   Core persisted entities
   - AgentJob
   - DomainResearchProfile
   - ResearchPortfolio
   - ResearchMonitorProfile
   - ResearchInboxItem
   - ResearchNote
   - ExperimentPlan / ExperimentRun
```

## Operator Surface Map

The main operator control plane is `frontend/src/pages/AutonomousAgentsPage.tsx`.

```text
Autonomous Agents
├── Checkpoint Queue      -> approvals, follow-up decisions, recovery, policy/budget review
├── Autonomy Health       -> monitor analytics, policy tuning, budget clamps, rebalancing
├── My Jobs               -> running and completed agent jobs
├── Domain Profiles       -> recurring domain research monitors
├── Research Fleet        -> portfolio orchestration across profiles
├── Research Inbox        -> accepted/rejected discoveries and follow-up launches
├── Swarm Review          -> coding swarm review workflows
├── Swarm Outcomes        -> swarm analytics and verified outcomes
├── Swarm Profiles        -> saved coding swarm configs
├── Coding Backlog        -> coding decomposition / slice workflow
├── Templates             -> bounded launch templates
└── Job Chains            -> chain-based launch orchestration
```

## Operator Flow

```text
Quick Start / Template / Monitor
        |
        v
AgentJob or recurring profile/portfolio run
        |
        v
Discovery / ranking / recommendation
        |
        +--> Queue item if approval or review is needed
        |
        +--> Follow-up job if policy allows
        |
        +--> Experiment plan + validation run if readiness allows
        |
        v
Research notes / inbox / experiment outcomes / summaries
```

## Owning Subsystems

- Queue and follow-up actions: `backend/app/api/endpoints/agent_jobs.py`
- Domain monitors: `backend/app/api/endpoints/domain_research_profiles.py`
- Research fleets: `backend/app/api/endpoints/research_portfolios.py`
- Monitor analytics and policy controls: `backend/app/api/endpoints/research_monitor_profiles.py`
- Validation runs and interventions: `backend/app/api/endpoints/experiments.py`
- Canonical autonomy contract helpers: `backend/app/services/autonomy_service.py`

## Related Docs

- `docs/AUTONOMOUS_RND_AGENTS.md`
- `docs/INGESTION_GUIDE.md`
- `docs/KNOWLEDGE_GRAPH.md`
- `docs/LATEX_STUDIO.md`
- `docs/knowledgedb_architecture_slide.mmd`
