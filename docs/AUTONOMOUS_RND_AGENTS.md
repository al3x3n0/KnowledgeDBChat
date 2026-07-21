# Autonomous RnD AI Agents

This repo already contains the main building blocks for autonomous research and engineering agents. The operator surface is centered around the `Autonomous Jobs` UI, but the full loop also spans research notes, experiment plans, validation sandboxes, research portfolios, and checkpoint-based approvals.

This document is the practical runbook for using that stack as an autonomous RnD system rather than as isolated features.

For the current system map and subsystem boundaries, start with `docs/ARCHITECTURE_ASCII.md`.

## What exists today

The current product supports these autonomous RnD primitives:

- Quick-start autonomous jobs for domain research, repo bug triage, coding swarms, frontend regression swarms, and role workflows.
- Persistent `DomainResearchProfile` records for recurring domain scans and opportunity tracking.
- `ResearchPortfolio` orchestration for running multiple domain profiles as a ranked research fleet.
- `ScientificSandboxProfile` records for bounded validation environments and experiment execution policy.
- Research note persistence, experiment-plan generation, experiment-run tracking, and research inbox follow-up handling.
- Approval checkpoints and queue management for operator review before risky or high-impact actions continue.

The main code paths are:

- Frontend orchestration UI: `frontend/src/pages/AutonomousAgentsPage.tsx`
- Quick-start payload builders: `frontend/src/pages/autonomousAgentQuickStarts.ts`
- Agent job quick starts and checkpoint queue APIs: `backend/app/api/endpoints/agent_jobs.py`
- Recurring domain research profiles: `backend/app/api/endpoints/domain_research_profiles.py`
- Research fleet orchestration: `backend/app/api/endpoints/research_portfolios.py`
- Scientific validation environments: `backend/app/api/endpoints/scientific_sandbox_profiles.py`

The runtime architecture for how those pieces connect is documented in `docs/ARCHITECTURE_ASCII.md`.

## Operator workflow

### 1. Launch a bounded domain-research agent

Use `Autonomous Jobs` and start with the domain research quick start when the goal is discovery rather than code repair.

Recommended setup for the first production-worthy path:

- `domain`: start with compiler optimization, code generation, or regression analysis before broader tracks.
- `objective`: what the agent should prove or disprove.
- `source_scope`: default to `kb_plus_arxiv_plus_repo` when repo evidence is available.
- `track_type`: default to `compiler` for the first end-to-end deployment.
- `monitor_queries` and `benchmark_queries`: compact, high-signal query lists.
- `sandbox_profile_id`: set this when you want downstream validation runs to stay inside a known resource and tool policy boundary.
- `automation_profile`: default to `balanced`, which prepares follow-ups automatically but queues risky launches for approval.

The quick-start request shape is defined in `frontend/src/types/index.ts` as `AgentJobQuickStartDomainResearchRequest`, and the payload is constructed in `frontend/src/pages/autonomousAgentQuickStarts.ts`.

Use the canonical autonomy contract for new integrations:

- `automation_profile`
- `automation_policy`
- `effective_policy`

Older compatibility fields such as `validation_policy` and monitor `follow_up_autonomy` are still accepted by the backend, but they are legacy mirrors and should not be the primary integration shape for new clients.

### 2. Convert one-off research into a persistent monitor

When a domain scan is useful enough to repeat, create a `DomainResearchProfile` instead of relaunching ad hoc jobs.

Use profiles for:

- Daily or scheduled domain monitoring
- Keeping a stable objective and source mix
- Tracking opportunities and linked research notes over time
- Carrying validation policy and sandbox defaults forward

Relevant API surface:

- `GET /api/v1/domain-research-profiles`
- `POST /api/v1/domain-research-profiles`
- `PATCH /api/v1/domain-research-profiles/{profile_id}`
- `POST /api/v1/domain-research-profiles/{profile_id}/action`
- `POST /api/v1/domain-research-profiles/{profile_id}/opportunities/{opportunity_id}/action`

### 3. Aggregate profiles into a research fleet

Use a `ResearchPortfolio` when you want multiple domain profiles to compete for attention and budget as a single autonomous RnD program.

A portfolio is the right abstraction when you need:

- Multiple monitors covering adjacent domains
- Ranked opportunities across those monitors
- Shared automation policy for follow-up and validation
- A single operating view of the current research pipeline

Relevant API surface:

- `GET /api/v1/research-portfolios`
- `POST /api/v1/research-portfolios`
- `PATCH /api/v1/research-portfolios/{portfolio_id}`
- `POST /api/v1/research-portfolios/{portfolio_id}/action`
- `POST /api/v1/research-portfolios/{portfolio_id}/opportunities/{opportunity_id}/action`

### 4. Turn opportunities into experiments

The autonomous loop is most useful when it produces testable artifacts, not just summaries. In this repo, that means:

- Persisting findings as `ResearchNote` records
- Creating `ExperimentPlan` records from opportunities
- Starting `ExperimentRun` records for validation
- Appending outcomes back to notes or the research inbox

Relevant API surface:

- `GET /api/v1/research-notes`
- `POST /api/v1/research-notes`
- `POST /api/v1/experiments/plans/generate`
- `POST /api/v1/experiments/plans/{plan_id}/runs`
- `POST /api/v1/experiments/runs/{run_id}/start`
- `POST /api/v1/experiments/runs/{run_id}/action`
- `POST /api/v1/experiments/runs/{run_id}/append-to-note`

### 5. Keep validation bounded with sandbox profiles

Do not treat autonomous RnD as unrestricted execution. `ScientificSandboxProfile` is the contract for what a validation run is allowed to use.

Use sandbox profiles to define:

- Track type compatibility
- Execution backend
- Docker image or toolchain requirements
- Timeout and resource caps
- Allowed benchmark families and collectors
- Default budget limits

Relevant API surface:

- `GET /api/v1/scientific-sandbox-profiles`
- `GET /api/v1/scientific-sandbox-profiles/{profile_id}`
- `POST /api/v1/scientific-sandbox-profiles`
- `PATCH /api/v1/scientific-sandbox-profiles/{profile_id}`
- `DELETE /api/v1/scientific-sandbox-profiles/{profile_id}`

### 6. Operate through checkpoints, not blind autonomy

The backend already has approval-checkpoint support and a checkpoint queue. Use that as the main control plane for human-in-the-loop review.

Relevant API surface:

- `GET /api/v1/agent-jobs/checkpoint-queue`
- `POST /api/v1/agent-jobs/checkpoint-queue/bulk-action`
- `POST /api/v1/agent-jobs/checkpoint-queue/follow-up-action`

This is the right place to review:

- tool calls that need approval
- follow-up launches
- recovery actions after partial failure
- high-impact plan steps before execution

## Recommended operating model

For autonomous RnD, the most defensible rollout in this codebase is:

1. Start with one narrow compiler domain research quick start.
2. Promote successful runs into `DomainResearchProfile` monitors.
3. Group stable monitors into one `ResearchPortfolio`.
4. Require sandbox-backed validation for anything that claims performance or correctness impact.
5. Use checkpoint approvals for costly, risky, or irreversible actions.
6. Capture every accepted opportunity as a research note or experiment artifact.

This keeps the loop evidence-driven:

- discover
- rank
- validate
- record
- relaunch

## Suggested first deployment

If the goal is to "proceed with autonomous RnD AI agents" in a controlled way, start with one of these tracks:

- Compiler research: repo-aware scans for optimization opportunities, regressions, and benchmark ideas.
- Microarchitecture research: cache, branch, SIMD, and throughput hypotheses tied to benchmark evidence.
- Retrieval quality research: document ingestion gaps, citation quality, ranking failures, and follow-up experiments.

For each track:

- create one sandbox profile
- create one domain research profile
- run it manually first
- review checkpoint and experiment behavior
- only then switch to recurring or portfolio mode

## Related docs

- `docs/pilots/research_lab_pilot.md`
- `docs/DOCUMENT_SUMMARIZATION.md`
- `docs/KNOWLEDGE_GRAPH.md`
- `docs/INGESTION_GUIDE.md`
