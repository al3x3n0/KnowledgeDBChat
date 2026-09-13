# Backend module boundaries

`app/modules` is the incremental home for domain-oriented backend modules. The
application remains one deployable process and one transactional database while
HTTP, application, domain, and infrastructure ownership become explicit.

## Package shape

```text
modules/<domain>/
├── api/              FastAPI routers and transport translation
├── application/      Use cases and orchestration
├── domain/           Domain rules and interfaces
└── infrastructure/   SQLAlchemy, Celery, and external adapters
```

Only create layers a domain currently needs. Small modules should remain small.

## Dependency rules

1. `api` may depend on its application/domain contracts and shared
   authentication or database dependencies.
2. `application` may coordinate domain interfaces but must not import FastAPI
   endpoint modules.
3. `domain` must not import FastAPI, Celery tasks, or another module's private
   implementation.
4. `infrastructure` implements domain interfaces and may depend on SQLAlchemy,
   Celery, or external SDKs.
5. Cross-module calls use an explicitly exported application service or
   contract, not another module's models or private helpers.
6. Existing URLs and compatibility imports remain stable while routes move out
   of legacy aggregate modules.

The first extracted slices are `modules/autonomy/api/chain_definitions.py`,
`chain_execution.py`, and `quick_starts.py`. The legacy
`api/endpoints/agent_jobs.py` module composes their routers and temporarily
re-exports their handlers for compatibility.

`chain_execution.py` uses a router factory for dependencies that still belong
to the legacy aggregate, such as the complete job response presenter. This
keeps the dependency direction explicit and prevents the new module from
importing the legacy endpoint.

`quick_starts.py` owns transport orchestration and repository-source validation.
Its launch configuration builders live in
`autonomy/application/quick_start_builders.py` and have no FastAPI or task
dependencies. General, domain-research, and role-workflow relaunch
reconstruction lives in `autonomy/application/quick_start_relaunch.py`.
Coding-swarm relaunch and retry recovery interpretation lives in
`autonomy/application/coding_swarm_relaunch.py`. The legacy endpoint only
retains compatibility aliases while callers migrate to these application
modules.

`autonomy/application/relaunch_dispatcher.py` maps persisted launch modes to
their request builders and injected launchers. The action endpoint retains
authorization, status validation, audit logging, and transaction ownership.
Relaunch graph traversal lives in
`autonomy/application/relaunch_lineage.py`; its user-scoped HTTP query is
owned by `autonomy/api/relaunch_lineage.py`.
Job-memory response normalization lives in
`autonomy/application/memory_presenters.py`, keeping malformed service payload
handling independent from the HTTP and persistence layers.
Memory CRUD, graph, statistics, and search routes live in
`autonomy/api/job_memories.py`. Human-feedback learning routes remain separate
because they also consume customer-profile policy.
Human-feedback normalization lives in
`autonomy/application/feedback_presenters.py`; feedback persistence, scope
resolution, and graph linking live in `autonomy/api/job_feedback.py`.
AI Hub recommendation feedback list/upsert routes live in
`autonomy/api/ai_hub_feedback.py`.
Core job creation and record CRUD are composed in
`autonomy/api/job_crud.py`. Creation routes mount before quick starts, while
the one-segment detail route mounts after static list/template routes to
preserve FastAPI route precedence.
Job listing, visibility filtering, optional swarm projection, and aggregate
statistics are composed in `autonomy/api/job_queries.py`.
Coding-swarm aggregate metrics are composed in
`autonomy/api/swarm_analytics.py`; terminal outcome reporting remains a
separate `autonomy/api/swarm_outcomes.py` boundary.
