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

## Tool access in Pipeline Studio

Select a stage in `/pipelines` and use **Tool access** to choose between the
normal menu of all permitted tools and **Evidence tools + selected extras**.
The latter offers a searchable catalog of built-in tools, installed research
program tools, and the current user's enabled plugin tools. Each entry names
its source and execution type; the choices are filtered by the stage's job
type. These can be file tools, data processors, compilers, solvers, API calls,
or other capabilities, not only simulators.

Expand **Inputs, results, and execution** on a tool to inspect its declared
arguments (including required fields and the full JSON Schema), evidence types,
prerequisite tools, and effects/network/cost classification. Container plugin
tools show their configured image. Research programs show their declared image
set, which is not a per-tool image binding or a readiness check. Inspection is
available in both tool-access modes; the agent supplies arguments at call time.

In the specification, omit `tools` (or use `null`) for the normal menu. An
explicit list such as `"tools": ["search_documents"]` limits the menu to those
extras plus the tools and prerequisites derived from the stage's evidence
contract. Required result keys also keep their writers available:
`structured_output` permits `set_output_schema`, and `formatted_outputs`
permits `format_as_table` and `format_as_report`. A stage with `may_revisit`
keeps `request_stage_rerun` available. These appear in the stage's plan,
including a notice when their execution cost is unknown. Successors receive
only their own declared backward edges.
`"tools": []` means contract tools only. Selection controls what the
agent may call; it does not force a call or supply its arguments. Job and role
restrictions still apply. Each stage has its own selection, including after a
restricted parent, and unavailable selections are refused before launch.
The action executor rechecks that policy before dispatch, including for direct
calls and external harnesses; a default or result binding cannot grant access
to an excluded tool.
Studio ties each check to the exact specification and budget. Editing either
disables Launch immediately until that version passes its check, and late
responses from older checks cannot replace the current verdict or estimate.

In a tool's inspection panel, **Stage argument defaults** accepts a JSON object.
Click **Apply defaults** to store it; unapplied edits do not change the stage.
In restricted mode, applying defaults also selects that tool as an extra.
The specification stores these under `tool_defaults`, keyed by tool name, for
example `"tool_defaults": {"search_web": {"query": "compiler optimization"}}`.
Explicit call arguments override defaults by top-level field; nested objects
are replaced as a whole. **Clear defaults** removes the stage's settings.
Opening another pipeline or a worked example resets the stage editor and
discards unapplied defaults and binding edits, even when stage IDs match.

Before launch, defaults are checked for available tools and declared argument
names, types, enum values, and UUID requirements using the runtime parameter
checker. Required arguments can be omitted from defaults because the agent
supplies the rest at call time. Defaults reach scope checks, approval, and the
execution journal before dispatch, and are cleared rather than inherited by
the next stage. They configure arguments only; selecting a Docker image still
follows the tool or Sandbox Skill's own configuration.

For a Docker-backed procedure, configure and validate a Sandbox Skill under
**Tools**, activate it, then select its `skill_<slug>` evidence under **Must
produce**. Its image, files, result schema, and control stay with the skill.
The stage automatically receives both the skill loader and runner. Installing
a research program remains a deployment operation; attaching one of its tools
does not build an image or install a package.

Selected skills appear under **Sandbox Skills used by this stage**, with their
configured image, command timeout, result fields, and whether their evidence
goes stale when working files change. These details come from the current
owner's active, verified skills; they describe configuration, not live Docker
availability. The evidence filter also searches skill names, descriptions,
and images. Edit and validate a skill in **Tools**, then choose **Refresh tools
and skills** in Studio. Refresh preserves the specification, selected stage,
and argument edits for tools that remain in the catalog. A failed refresh keeps
the last catalog visible and offers a retry. Launch waits for the refresh and a
fresh check of the current configuration. Requiring a skill makes its runner
available; the agent still chooses commands from the loaded procedure and
collects the declared result when ready.

Tools with saved defaults or bindings remain visible for repair even if they
are no longer selected or compatible with the stage's job type. Unavailable
tools show their saved settings, including in normal access mode. **Remove
tool and settings** explicitly removes that tool's extra selection, defaults,
and bindings in one edit, preserving other tools and the stage's access mode.
It does not remove evidence requirements from the stage contract.

Refreshing the catalog also clears binding previews and waits for the refresh
before allowing another preview request. Run **Preview applied bindings** again
to check values against the current tool schema. Older requests cannot restore
preview values or errors after a refresh or specification change.

### Passing results between tools

Inside tool inspection, use **Bind arguments to stage results**. Select an
argument, a declared dependency stage, and a JSON Pointer into that stage's
results, then **Apply bindings**. For example, `/structured_output/code` reads
the code field, while `/findings/0/value` reads a finding's value. An empty
pointer selects the complete results object. `~1` escapes a slash in a key and
`~0` escapes a tilde. Expressions and arbitrary job IDs are not supported.

The spec stores these as
`"tool_bindings": {"check_implementation": {"code": {"stage": "produce", "path": "/structured_output/code"}}}`.
The source must be in this stage's `depends_on`. Bound values override defaults
and explicit call arguments, preserve JSON types, and are resolved before
approval and execution. Missing sources or paths fail the call without invoking
the tool; there is no fallback to a model-supplied value. Argument types are
checked by the normal runtime validator after resolution.
For contributed tools, runtime type, enum, required-field, and UUID checks use
the owner's plugin schema resolved for that job, before reaching the plugin
handler. A malformed bound value is refused instead of falling back to a
model argument or starting a container.

Child creation snapshots only its declared dependency results. Fan-in stages
receive each dependency's results; new and restarted children rebuild these
inputs from their actual parents rather than retaining an older stage's inputs.
Each attempt also receives independent copies of its argument settings,
inherited evidence, and downstream chain definition. Editing or restarting one
attempt does not mutate an earlier attempt or the template used for later runs.

Stage renames update dependencies, revisit edges, and binding sources together.
Enter or leave the stage-id field to apply a rename; duplicate or empty ids are
refused without changing the specification.

Deleting a stage removes dependencies and revisit edges naming it, through
either the inspector or the graph keyboard shortcut. Explicit result bindings
remain in the specification for repair; Studio does not silently replace them
with defaults or model-supplied arguments. The inspector names missing binding
sources even with tool inspection collapsed. Removing a dependency or changing
a tool's argument schema also highlights invalid rows in the binding editor.
Restore the dependency, choose a new source/argument, or remove the binding,
then **Apply bindings**. Until repaired, the pipeline checker refuses launch.

Restarting a stage uses the latest attempt of its predecessor and refuses to
fall back to older evidence while that attempt is unfinished or its contract
is unmet. A fan-in restart also requires completed dependencies from the same
branching attempt. The run panel uses those same checks to offer Restart.
Inserted stages receive their own tool settings; they do not
inherit the parent's tool selection, defaults, or bindings. Explicit bindings
downstream keep naming their original source, carried through the inserted
stages as frozen snapshots even across repeated insertions. Forwarded results
remain separate from each inserted stage's declared tool inputs and are cleared
when no successor needs them. Insertion inside a fan-in branch is refused because
it would break the runner's sibling-based join; edit that graph in Studio instead.

For a saved pipeline with a previous run, select a stage with bindings and
choose **Preview applied bindings**. The preview resolves the edited spec
against the latest recorded attempt of each dependency in that run. It shows
values and types, missing paths, empty bound source scopes, parameter-check
errors, and the source job's status and contract result. Known defaults and
bindings go through the normal tool argument coercion before preview validation.
Values and the number of returned bindings are capped, with truncation and
omitted entries shown explicitly.

Preview is read-only and does not call tools or change the pipeline. Editing
the spec clears the displayed preview. Recorded results can differ from a
running stage's frozen inputs and from what a new run will produce; a resolved
value is not evidence that its source contract passed.

UI examples captured with local fixture data: [stage tool defaults](screenshots/pipeline-tool-defaults.png)
and [resolved binding preview](screenshots/pipeline-binding-preview.png). They show
the authoring interface, not a live tool execution or a deployment health check.

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

### Launching a suite as real trials

Binding job ids to tasks by hand only works if an operator already launched
every trial. A launch does the fan-out instead: it creates `trials` agent jobs
per suite task, tracks them as one unit, and grades the whole set once every
trial reaches a terminal state.

```http
POST /api/v1/autonomous-rnd-evals/launches
{
  "suite_id": "compiler_research_v1",
  "trials_per_task": 3,
  "label": "nightly",
  "start_immediately": true
}

GET /api/v1/autonomous-rnd-evals/launches
GET /api/v1/autonomous-rnd-evals/launches/{launch_id}
```

Fan-out is unattended agent execution that costs real tokens and sandbox time,
so it follows the same default-off policy as the other dangerous capabilities:

- `AUTONOMOUS_RND_EVAL_LAUNCH_ENABLED` (default `false`) gates the endpoint
- `AUTONOMOUS_RND_EVAL_MAX_TRIAL_JOBS` (default `30`) caps one launch; a request
  above the cap is rejected before any job is created
- `AUTONOMOUS_RND_EVAL_TRIAL_MAX_ITERATIONS` and
  `AUTONOMOUS_RND_EVAL_TRIAL_MAX_RUNTIME_MINUTES` bound each trial

Each trial job carries an `autonomous_rnd_eval` block in its config recording
the launch, suite, task, trial index, and seed, so a trial is identifiable from
the job alone. `finalize_autonomous_rnd_eval_launch` polls every 60 seconds and
grades the launch once the last trial settles, writing a run with
`source: "launch"` that compares against a baseline like any other run. It gives
up after roughly four hours and leaves the launch marked failed.

A trial job that was deleted before grading counts as a failed trial rather than
a skipped one — dropping it would inflate `pass_pow_k` for its task.

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

## Model-call accounting

Progress scoring is deterministic and does not consume `max_llm_calls` or
increment `llm_calls_used`. This also applies when scoring inherited findings or
completing a contract after the last available model call. Historical stored
usage totals are retained; they are not recalculated by this change.

For the optional native tool loop, `native_tool_loop.max_llm_calls` includes
the final schema-formatting request. That request runs only if a slot remains,
and a failed formatting request still counts as an attempt. If the native loop
uses its allowance without producing a valid decision, it returns a budget-stop
decision instead of making a prompted fallback or parse-repair request. Already
gathered findings remain in the run state; this stop does not claim that the
goal was achieved. Planning that spends the job's final slot also prevents a
subsequent thinking request in the same iteration.

Observation rechecks the remaining job allowance before causal planning,
execution planning, criticism, and replanning. A request that falls back to a
deterministic plan or yields no critic note still consumes its attempted call;
disabled or already-completed planning does not. Existing plans and findings
remain available when later model steps are skipped for lack of allowance.

Reloading job control state keeps the higher of the in-memory and persisted
model/tool usage counts. A refresh cannot refund calls already made, while
higher stored usage, tightened limits, and cancellation still take effect before
the next iteration. This preserves usage across refreshes; it does not reconstruct
calls lost before the refresh boundary.

Automatic history compaction counts its summary request against the same job
allowance, including failed requests. When only the decision's last slot remains,
compaction uses its deterministic digest without a model request. Disabled,
below-threshold, and digest-only compaction do not consume model-call allowance.

Final conclusion synthesis also shares the job allowance. Structured output and
its prompted JSON fallback each count as a request, including failures; fallback
runs only if another slot remains. If no allowance remains, the optional
conclusion has `generated_by: budget_exhausted` and no invented answer. Findings
and normal chain completion remain available. Campaign-level synthesis, which
has no single job allowance, is unchanged.

## Evidence bundle completion contracts

For a readable evidence bundle, `verify_run_bundle` returns a
`bundle_verified` integrity report. A broken
bundle remains visible for diagnosis but does not count toward a finding-type
completion contract, including historical reports marked `intact: false`. A
manually saved finding with a verification type label is not runtime proof and
cannot satisfy either verification contract.

To require reproduction rather than integrity alone, use this job configuration:

```json
{
  "goal_contract": {
    "enabled": true,
    "required_finding_types": ["bundle_reproduced"]
  }
}
```

For a pipeline stage, put the inner contract under the stage's `contract` key.
Call `verify_run_bundle` with `replay: true`; pass `job_id` to check an earlier
job belonging to the same owner, or omit it to check the current run. When a
stage has an allowed-tools list, include both `verify_run_bundle` and the recorded
tools it must replay; replay keeps the stage's tool policy.

`bundle_reproduced` is emitted only when artifacts are intact and a successful
replay compared every eligible original successful call, with at least one
comparison. Integrity-only checks, changed results,
execution errors, all-skipped replays, and partial legacy comparisons do not
satisfy this contract.
Wall-clock results and other skipped calls remain reported; the finding proves
the comparisons that ran, not that every recorded call was comparable. Its
`replay_summary` records expected, judged, and skipped counts. Timing-dependent
calls and originally failed calls are excluded from expected comparisons; a
legacy successful call without a canonical hash remains expected and prevents
the stronger reproduction proof.

Both outputs retain source job IDs, program diagnostics, and stable context IDs
when passed to downstream stages. A reproduced result may carry a program-version
mismatch warning: matching outputs do not establish identical installed code. A stage may
assume `bundle_reproduced` to consume an upstream proof; that is not a fresh
replay by the receiving stage.

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
