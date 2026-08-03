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
- Agent-job composition root: `backend/app/api/endpoints/agent_jobs.py`
- Operator checkpoint queue querying and presentation:
  `backend/app/modules/autonomy/api/checkpoint_queue.py`
- Decision-trace querying, fallback merging, filtering, and aggregation:
  `backend/app/modules/autonomy/api/decision_trace_queries.py`
- Decision-trace exports and persisted-event analytics:
  `backend/app/modules/autonomy/api/decision_trace_reporting.py`
- Decision-trace operator mutations:
  `backend/app/modules/autonomy/api/decision_trace_actions.py`
- Ownership-scoped decision-trace saved views:
  `backend/app/modules/autonomy/api/decision_trace_views.py`
- Job-template catalog querying and presentation:
  `backend/app/modules/autonomy/api/job_templates.py`
- Pure template recommendation policy:
  `backend/app/modules/autonomy/application/template_recommendations.py`
- Promotion from exploratory jobs into durable profiles and research fleets:
  `backend/app/modules/autonomy/api/domain_research_promotion.py`
- Domain-research job promotion into profile and portfolio creation seeds:
  `backend/app/modules/autonomy/application/domain_research_promotion_seed.py`
- Single-job operator action HTTP boundary:
  `backend/app/modules/autonomy/api/job_actions.py`
- Operator action state machine:
  `backend/app/modules/autonomy/application/job_action_state_machine.py`
- Shared job-action contracts:
  `backend/app/modules/autonomy/application/job_action_contracts.py`
- Swarm assignment, review, tie-breaker, and promotion actions:
  `backend/app/modules/autonomy/application/job_action_swarm.py`
- Restart and relaunch recovery actions:
  `backend/app/modules/autonomy/application/job_action_recovery.py`
- Paused-job and approval-checkpoint resume handling:
  `backend/app/modules/autonomy/application/job_action_checkpoint_resume.py`
- Explicit approval-checkpoint decisions (`approve`, `edit`, `skip`, `reject`):
  `backend/app/modules/autonomy/application/job_action_checkpoint_decisions.py`
- Basic pause, cancel, and summary lifecycle actions:
  `backend/app/modules/autonomy/application/job_action_lifecycle.py`
- Structured, bounded operator-intervention history mutation:
  `backend/app/modules/autonomy/application/job_action_interventions.py`
- Checkpoint action normalization, audit events, and execution-state synchronization:
  `backend/app/modules/autonomy/application/job_action_checkpoints.py`
- Operator checkpoint execution-plan step mutation and advancement:
  `backend/app/modules/autonomy/application/job_action_plan_state.py`
- Job operator-action decision-event normalization and persistence:
  `backend/app/modules/autonomy/application/job_operator_events.py`
- Job approval and recurring-recovery checkpoint queue projection:
  `backend/app/modules/autonomy/application/checkpoint_queue_jobs.py`
- Shared checkpoint priority, SLA, and escalation policy:
  `backend/app/modules/autonomy/application/checkpoint_queue_priority.py`
- Bound five-source checkpoint queue composition and stable priority ordering:
  `backend/app/modules/autonomy/application/checkpoint_queue_composer.py`
- Monitor policy and budget checkpoint queue projection:
  `backend/app/modules/autonomy/application/checkpoint_queue_monitors.py`
- Accepted research-inbox follow-up checkpoint queue projection:
  `backend/app/modules/autonomy/application/checkpoint_queue_inbox.py`
- Research-portfolio opportunity checkpoint queue projection:
  `backend/app/modules/autonomy/application/checkpoint_queue_portfolios.py`
- Domain-research-profile opportunity checkpoint queue projection:
  `backend/app/modules/autonomy/application/checkpoint_queue_profiles.py`
- Research-inbox follow-up approval and rejection actions:
  `backend/app/modules/autonomy/application/follow_up_queue_inbox.py`
- Failed or cancelled research-inbox follow-up relaunch workflow:
  `backend/app/modules/autonomy/application/follow_up_inbox_relaunch.py`
- Follow-up learning-profile loading and stable score normalization:
  `backend/app/modules/autonomy/application/follow_up_learning_profiles.py`
- Inbox, portfolio, and profile follow-up action target dispatch:
  `backend/app/modules/autonomy/application/follow_up_queue_dispatcher.py`
- Follow-up queue operator-decision event normalization and persistence:
  `backend/app/modules/autonomy/application/follow_up_queue_events.py`
- Research-portfolio follow-up approval, rejection, and child-job launch actions:
  `backend/app/modules/autonomy/application/follow_up_queue_portfolios.py`
- Portfolio opportunity and linked-artifact state synchronization:
  `backend/app/modules/autonomy/application/portfolio_queue_state.py`
- Domain-profile follow-up approval, rejection, and child-job launch actions:
  `backend/app/modules/autonomy/application/follow_up_queue_profiles.py`
- Follow-up autonomy policy, budget throttling, and launch-state application:
  `backend/app/modules/autonomy/application/follow_up_policy.py`
- Research-inbox follow-up recommendation construction and learned ranking:
  `backend/app/modules/autonomy/application/follow_up_recommendations.py`
- Autonomous-job response presentation and contract normalization:
  `backend/app/modules/autonomy/application/job_presenters.py`
- Derived decision-trace source loading and event-stream aggregation:
  `backend/app/modules/autonomy/application/decision_trace_loader.py`
- Deterministic decision-trace event IDs and response-contract construction:
  `backend/app/modules/autonomy/application/decision_trace_events.py`
- Job operator-intervention and scheduler-recovery decision-trace projection:
  `backend/app/modules/autonomy/application/decision_trace_jobs.py`
- Checkpoint-queue classification and decision-trace projection:
  `backend/app/modules/autonomy/application/decision_trace_queue.py`
- Persisted decision-trace follow-up approval target resolution:
  `backend/app/modules/autonomy/application/decision_trace_follow_up_targets.py`
- Operator queue and decision-trace context normalization:
  `backend/app/modules/autonomy/application/operator_queue_context.py`
- Monitor policy, budget, and customer-rebalance decision-trace projection:
  `backend/app/modules/autonomy/application/decision_trace_monitors.py`
- Scientific-validation run decision-trace projection:
  `backend/app/modules/autonomy/application/decision_trace_validation.py`
- Opportunity decision-state classification and bound trace projection:
  `backend/app/modules/autonomy/application/decision_trace_opportunities.py`
- Coding-swarm terminal outcome case projection:
  `backend/app/modules/autonomy/application/swarm_outcome_cases.py`
- Repair verification evidence precedence and lifecycle status derivation:
  `backend/app/modules/autonomy/application/repair_verification.py`
- Coding-swarm execution, fan-in, and collaboration summary projection:
  `backend/app/modules/autonomy/application/swarm_summaries.py`
- Job-backed bulk checkpoint eligibility and actions:
  `backend/app/modules/autonomy/api/checkpoint_job_actions.py`
- Individual and bulk follow-up checkpoint decisions:
  `backend/app/modules/autonomy/api/checkpoint_follow_up_actions.py`
- Follow-up queue application-to-HTTP error adapter:
  `backend/app/modules/autonomy/api/follow_up_queue_actions.py`
- Job-results document and presentation export boundary:
  `backend/app/modules/autonomy/api/job_exports.py`
- Owned-job checkpoint history and HTTP boundary:
  `backend/app/modules/autonomy/api/job_checkpoints.py`
- Owned-job execution-log pagination and HTTP boundary:
  `backend/app/modules/autonomy/api/job_logs.py`
- Authenticated Redis-backed job-progress WebSocket boundary:
  `backend/app/modules/autonomy/api/job_progress.py`
- Owned-job step-event source selection and paginated HTTP boundary:
  `backend/app/modules/autonomy/api/job_step_events.py`
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

## Capability evaluation

Use the autonomous R&D evaluation harness to measure completed research outcomes,
not job activity or self-reported completion. The initial compiler suite covers:

- reproducing a regression with preserved compiler, code-generation, and benchmark
  artifacts
- rejecting a null-control regression instead of inventing a cause
- reconciling misleading evidence without executing untrusted instructions

Each task is run multiple times. `pass_at_k` shows whether at least one trial
succeeded; `pass_pow_k` shows whether every trial succeeded and is the more useful
reliability gate for unattended operation.

The grader consumes replayable JSON outcomes keyed by task ID:

```json
{
  "compiler_regression_reproduce": [
    {
      "status": "completed",
      "claims": [{"id": "claim-1", "evidence_ids": ["evidence-1"]}],
      "evidence": [{"id": "evidence-1", "kind": "benchmark_output"}],
      "artifacts": [
        {"kind": "compiler_logs"},
        {"kind": "ir_or_codegen_artifacts"},
        {"kind": "benchmark_output"}
      ],
      "experiment": {"repeat_count": 3, "all_commands_ok": true},
      "actions": []
    }
  ]
}
```

Run it from `backend/`:

```bash
python scripts/run_autonomous_rnd_evals.py \
  --suite evals/autonomous_rnd/compiler_research_v1.json \
  --outcomes /path/to/outcomes.json \
  --report /tmp/compiler-research-eval.json \
  --fail-below 0.95
```

The runner callback in
`app/services/autonomous_rnd_eval_service.py` can also execute live trials against
any local or hosted agent. Keep recorded outcomes as regression fixtures so model,
prompt, tool, and orchestration changes can be compared on identical graders.

Completed `AgentJob` trajectories can be graded directly without exporting replay
JSON. Bind persisted job IDs to evaluation tasks:

```http
POST /api/v1/autonomous-rnd-evals/grade-jobs
Content-Type: application/json
Authorization: Bearer <token>

{
  "suite_id": "compiler_research_v1",
  "trials": [
    {
      "task_id": "compiler_regression_reproduce",
      "job_ids": ["<trial-1-job-id>", "<trial-2-job-id>", "<trial-3-job-id>"]
    }
  ]
}
```

The adapter reads persisted status, structured claims and evidence, output
artifacts, step events, and linked `ExperimentRun` measurements. Narrative summaries
are not treated as proof. Missing task trials therefore fail `pass_pow_k`, even if
all supplied trials pass.

### Run history and baselines

A graded report is ephemeral unless it is stored, and a score with nothing to
compare against cannot tell a real regression from noise. Add `"persist": true`
(optionally with a `"label"`) to the `grade-jobs` request and the response also
carries a `run_id` for the stored run.

Stored runs are owner-scoped and support one baseline per suite:

```http
GET  /api/v1/autonomous-rnd-evals/runs?suite_id=compiler_research_v1&limit=20
GET  /api/v1/autonomous-rnd-evals/runs/{run_id}
POST /api/v1/autonomous-rnd-evals/runs/{run_id}/baseline
GET  /api/v1/autonomous-rnd-evals/runs/{run_id}/comparison
```

Promoting a run to the baseline demotes the previous one; migration
`0080_add_autonomous_rnd_eval_runs` also enforces a single baseline per owner and
suite at the database level. The comparison endpoint diffs a candidate against
the suite baseline by default, or against an explicit `baseline_run_id`.

A comparison reports `mean_score`, `pass_at_k`, and `pass_pow_k` deltas plus a
per-task status of `regressed`, `improved`, `unchanged`, `added`, or `removed`.
`has_regression` is driven by `pass_pow_k`, the reliability gate for unattended
operation: it is set when any task loses all-trial reliability, or when aggregate
`pass_pow_k` drops without an individual task flipping. Tasks that only appear in
one of the two runs — a suite gaining or losing coverage — are surfaced as
`added` or `removed` and never counted as regressions, so suite edits do not
masquerade as capability loss. Compare `suite_version_changed` before reading too
much into a delta.

Full trial detail, including each graded check, is stored inline on the run, so a
persisted run replays exactly. Reports grow with trial count and outcome size;
prune old runs if a suite is graded on a tight schedule.

Runtime finalization also stores a versioned `evaluation_outcome` and a compact
action ledger on every finalized job. The ledger deliberately excludes tool
parameters and raw outputs. External-agent responses contribute only allowlisted
provenance (agent identity, capability, request ID, response hash, size, and
timing) and enter the evidence set with `verification_status: unverified`; they
never create a claim by themselves.

External evidence follows a conservative verification state machine:

```text
unverified ── explicit local evidence/artifact link ──> corroborated
     │                                                   │
     └──── grounded contradiction ───────> rejected      │
                                                         │
       independent evidence + replayable artifact +      │
       successful repeated experiment <──────────────────┘
                             │
                             └──────────────────────> verified
```

Promotion requires a structured verification link, for example:

```json
{
  "verification_links": [
    {
      "external_evidence_id": "external-agent:request-17",
      "verdict": "supports",
      "local_evidence_ids": ["benchmark-result-4"],
      "artifact_kinds": ["compiler_logs"],
      "min_repeat_count": 3
    }
  ]
}
```

References must resolve to evidence and artifacts actually recorded in the
outcome. `verified` additionally requires a successful experiment meeting the
repeat threshold (two runs by default). A grounded `contradicts` link moves the
external evidence to `rejected`. The traceability grader does not allow
`unverified` or `rejected` evidence to support a verified claim. Only
runtime-observed findings count as independent local evidence for promotion;
agent-authored structured output cannot designate itself as a trusted source.

### Verification planning

Schema-v3 outcomes include a deterministic `verification_plan`. Every unresolved
external response (`unverified` or `corroborated`) receives a bounded task with:

- claim-aware priority
- missing verification checks
- a local-sandbox experiment specification
- minimum repeat count and required artifact kinds
- success and stop conditions
- explicit prohibition on using another external agent as the verifier

Compiler capabilities default to compiler logs, code-generation/IR artifacts, and
benchmark output. Retrieval capabilities default to retrieval traces and
evaluation metrics. Other capabilities require an experiment log and local
observation.

Plans are `proposal_only`, capped at 50 tasks, require approval, and are never
launched during finalization. This keeps finishing a job free of hidden compute or
external side effects. The `verification_plan_coverage` grader can require every
unresolved external evidence record to have a corresponding task.

An operator can materialize one proposed task through the authenticated launch
boundary:

```http
POST /api/v1/autonomous-rnd-evals/jobs/{job_id}/verification-tasks/{task_id}/launch
Authorization: Bearer <token>
Content-Type: application/json

{
  "approval_confirmed": true,
  "approval_note": "Approved bounded local reproduction",
  "research_note_id": "<note-id>",
  "source_id": "<git-source-id>",
  "sandbox_profile_id": "scientific-compiler-sandbox",
  "commands": ["pytest -q tests/compiler"],
  "repeat_count": 3,
  "timeout_seconds": 120,
  "max_runtime_minutes": 10,
  "budget_limit": 5.0,
  "start_immediately": false
}
```

The endpoint verifies job and note ownership, resolves an enabled sandbox
profile, filters commands through the selected scientific recipe, enforces the
profile budget and worst-case runtime, and records an approval audit. It creates
deterministic IDs for the `ExperimentPlan`, `ExperimentRun`, and sandboxed
`AgentJob`, so retries are idempotent and concurrent duplicates converge on one
launch. Command contents are required in the execution records but are omitted
from the approval audit.

`repeat_count` represents complete repetitions of the command set. Individual
command count no longer masquerades as experimental replication.

When the deterministic verification job finishes, reconciliation writes only a
sanitized local evidence record back to the parent job: run/job identifiers,
command and repeat counts, status, and whether all commands succeeded. Raw
stdout, stderr, and command output remain on the child experiment record.

Successful controlled repetitions create an explicit support link scoped to that
exact `ExperimentRun`; they can promote the external evidence to `verified`.
Failed or blocked execution is recorded as `inconclusive`, never automatically
as a contradiction. Verification and planning cannot borrow success from an
unrelated experiment run. Reconciliation is idempotent and is also retried when
an operator synchronizes the experiment run.

Operators can retrieve the canonical outcome and its verification lifecycle with:

```http
GET /api/v1/autonomous-rnd-evals/jobs/{job_id}/outcome
Authorization: Bearer <token>
```

The response combines current evidence status with unresolved and historical
verification tasks. Each task reports launch, job, approval, and reconciliation
status; deterministic plan/run/job/audit identifiers; and the approved numeric
budget. It never returns approval-audit command contents or raw experiment
stdout/stderr.

Jobs with a canonical R&D outcome expose the same lifecycle in the Autonomous
Agents job detail panel. Operators can review required checks, configure a
bounded local recipe, explicitly approve it, and optionally queue the verifier
immediately. Research notes and document sources are selected from searchable
owned-resource lists. Active verifier jobs push progress through the existing
agent-job WebSocket, with a low-frequency lifecycle refresh as fallback.
Completed launches link to the verifier job. Jobs without R&D verification
tasks do not render the panel or make the lifecycle request.

Reconciliation emits one idempotent `autonomous_rnd_verification_update`
notification per task/run pair, using the existing experiment-update user
preference. The notification contains only safe identifiers, counts, and
verification state; its action opens the parent job with the exact evidence task
highlighted. Verifier job details provide the reverse link back to that parent
task.

The lifecycle response also includes a normalized `timeline` ordered by
timestamp. Events cover proposal creation, explicit approval, bounded experiment
creation, execution start/completion when available, and reconciliation.
Timeline entries expose actors, statuses, and safe entity identifiers only.
Reconciliation timestamps are stable across idempotent sync retries.
Operators can filter the timeline by verification task or event status and
export the current view as a versioned JSON audit report. Export construction
uses an explicit allowlist; commands, approval notes, raw output, and unexpected
API fields are excluded.

Browser-generated exports include a SHA-256 digest over compact JSON with
recursively sorted object keys. For stronger provenance, an authenticated
operator can request `POST
/api/v1/autonomous-rnd-evals/jobs/{job_id}/verification-audit-snapshot`. The
server reconstructs the allowlisted snapshot from the owned job and lifecycle,
persists it in the append-only audit registry, then adds the same canonical
SHA-256 digest and an Ed25519 signature. It never signs arbitrary client-provided
report content. Signed envelopes carry the raw public key and key identifier, so
they can be verified independently offline against a trusted key obtained from
`GET /api/v1/autonomous-rnd-evals/verification-audit-keys`. They can also be
checked with `POST
/api/v1/autonomous-rnd-evals/verification-audit-snapshots/verify`; valid
envelopes remain ownership-scoped, registry-backed, and request-size bounded.
Historical snapshot records retain their public key after
`AUTONOMOUS_RND_AUDIT_SIGNING_KEY_ID` and
`AUTONOMOUS_RND_AUDIT_SIGNING_PRIVATE_KEY` rotate, while new snapshots use the
active key. A private-key rotation must always use a new key ID; the signing
endpoint rejects reuse of an existing ID with different public material.
Snapshot rows cannot be updated; deletion remains possible only
through database lifecycle/privacy erasure. Migration
`0074_add_rnd_verification_audit_snapshots` creates the registry and its
database-level update guard.

For offline verification, save the key-list response beside the exported
snapshot and run:

```bash
cd backend
python scripts/verify_autonomous_rnd_audit.py \
  /path/to/verification-audit-signed.json \
  --keys /path/to/trusted-verification-audit-keys.json
```

The verifier requires a trusted key registry instead of trusting the public key
embedded in the envelope.

Run the evaluation regression gate with:

```bash
make test-rnd-evals
```

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

- `docs/EXTERNAL_AGENT_GATEWAY.md`
- `docs/pilots/research_lab_pilot.md`
- `docs/DOCUMENT_SUMMARIZATION.md`
- `docs/KNOWLEDGE_GRAPH.md`
- `docs/INGESTION_GUIDE.md`
