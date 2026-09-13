# CompOps integration

KnowledgeOps Lab treats [CompOps](https://github.com/al3x3n0/CompOps) as a
first-class external compiler-research system. It is not modeled as an
unrestricted remote shell and it is not trusted as an agent that can validate
its own claims.

## Responsibility boundary

```text
KnowledgeOps Lab                          CompOps / CloudCompiler
────────────────────────────────────      ─────────────────────────────────
hypotheses and research portfolios        typed LLVM operator catalog
agent planning and checkpoints            workflow, batch, and study execution
cross-domain knowledge and synthesis      worker leases and runtime isolation
approval and tool policy                   immutable IR and metric artifacts
evidence state and verification plans     provenance, lineage, attestations
signed audit snapshots                    noise-aware study gates
              │                                      │
              └──── typed HTTPS capability calls ────┘
                     scoped bearer identity
```

CompOps remains the system of record for compiler executions and their
artifacts. KnowledgeOps Lab stores only bounded invocation provenance in the
agent trajectory: connection ID, capability, request ID, response digest,
timing, and allowlisted remote entity references. Raw IR, logs, and full CompOps
responses are not copied into the compact action ledger.

## Register a CompOps connection

Create a project-scoped researcher token in CompOps, store it through the
KnowledgeOps Lab secrets API, then register the HTTPS control-plane base URL:

The Tools page provides the same workflow under **CompOps research systems**:
store or select a vault credential, register the base URL and capability
manifest, then run an audited health check or operator discovery. The default
selection is read-only; write capabilities are visually identified and should
be paired with an explicit tool-policy approval.

```http
POST /api/v1/external-agents
Authorization: Bearer <knowledgeops-token>
Content-Type: application/json

{
  "name": "CompOps Compiler Research",
  "provider_type": "compops",
  "endpoint_url": "https://compops.example.com",
  "capabilities": [
    "compops.operators.list",
    "compops.runs.get",
    "compops.runs.submit",
    "compops.artifacts.get",
    "compops.artifacts.lineage",
    "compops.studies.get",
    "compops.studies.report",
    "compops.studies.gates.evaluate",
    "compops.batches.create"
  ],
  "auth_type": "bearer",
  "secret_id": "<compops-researcher-token-secret-id>",
  "timeout_seconds": 60
}
```

The provider adapter accepts only known CompOps capability names and constructs
the REST method and path itself. Callers cannot supply arbitrary URLs or route
fragments.

## Capability mapping

| KnowledgeOps capability | CompOps REST operation | Effect |
|---|---|---|
| `compops.health` | `GET /healthz` | read |
| `compops.operators.list` | `GET /v1/operators` | read |
| `compops.runs.get` | `GET /v1/runs/{run_id}` | read |
| `compops.runs.submit` | `POST /v1/runs` | write |
| `compops.artifacts.get` | `GET /v1/artifacts/{artifact_id}` | read |
| `compops.artifacts.lineage` | `GET /v1/artifacts/{artifact_id}/lineage` | read |
| `compops.studies.get` | `GET /v1/studies/{study_id}` | read |
| `compops.studies.report` | `GET /v1/studies/{study_id}/report` | read |
| `compops.studies.gates.evaluate` | `GET /v1/studies/{study_id}/gates/evaluate` | read |
| `compops.batches.create` | `POST /v1/batches` | write |
| `compops.actions.get` | `GET /v1/actions/{action_id}` | read |
| `compops.actions.approve` | `POST /v1/actions/{action_id}/approve` | write |
| `compops.actions.reject` | `POST /v1/actions/{action_id}/reject` | write |

Write capabilities take the exact CompOps request object under `payload.request`.
KnowledgeOps Lab adds `knowledgeops_request_id` for correlation and idempotency
tracing. Identifier fields are validated before being inserted into a route.

## Evidence and trust

A successful call becomes `external_system_response` evidence with
`external_system_type: compops`, initially `unverified`. This is intentionally
different from an external agent response while retaining the same conservative
promotion rules:

1. The response digest proves which bounded response KnowledgeOps Lab received.
2. Remote run, study, batch, or artifact IDs preserve the cross-system link.
3. CompOps artifact lineage, integrity/attestation results, and study gates can
   corroborate the response.
4. A KnowledgeOps verification link and replayable evidence are still required
   before a research claim becomes verified.

CompOps administration, worker provisioning, access-token management, raw
artifact upload, lease completion, retention purge, and arbitrary paths are not
exposed by this adapter.

### Import into an R&D job

The autonomous R&D job view can import a CompOps study report, gate evaluation,
run result, artifact record, or artifact lineage. The invocation includes the
KnowledgeOps agent-job ID, so a successful audited call is reconciled directly
into that job's canonical outcome.

Only allowlisted provenance is appended to the job: the connection and
capability, correlation request ID, response digest and size, timing, audit ID,
and stable CompOps entity IDs. The remote response remains in the tool audit and
is never copied into the job result or rendered by the import panel. Imported
evidence starts as `unverified` and immediately produces a verification-plan
task.

If tool policy requires approval, the initial request returns the pending audit
ID. Approving that audit executes the same captured call and performs the same
job linkage, preserving the original scope and correlation data.

### Keep evidence synchronized

Study reports, gate evaluations, run records, artifact metadata, and artifact
lineage can be registered as bounded synchronization subscriptions from the same
R&D job panel. A subscription fixes the CompOps connection, capability, remote
entity ID, query options, job ID, and polling interval at creation time.

Celery Beat scans due subscriptions every five minutes. Each actual request:

1. re-checks the current connection state and tool policy;
2. executes only the stored typed read capability;
3. writes a complete tool audit containing the remote response;
4. compares the response digest with the previous observation; and
5. replaces the job's sanitized evidence projection only when the digest changed.

The evidence ID remains stable across updates, so a long-running CompOps run
does not create duplicate evidence or verification tasks on every poll.
Unchanged observations still receive an audit record and advance the next poll.
If policy is denied or begins requiring per-call approval, the subscription is
disabled and marked `policy_blocked` or `approval_required`; it must be
explicitly re-enabled after policy is corrected. Operators can also pause,
resume, or synchronize a subscription immediately from the job panel.

### Signed push refreshes

Each subscription can optionally create or rotate a dedicated webhook signing
secret. KnowledgeOps returns the secret only in that setup response; it is then
stored encrypted in the user vault. Configure CompOps (or a project-scoped
event relay) to send:

```http
POST /api/v1/external-agents/compops-webhooks/<subscription-id>
Content-Type: application/json
X-CompOps-Timestamp: <unix-seconds>
X-CompOps-Event-ID: <stable-unique-event-id>
X-CompOps-Event-Type: run.completed
X-CompOps-Signature: v1=<hex-hmac-sha256>
```

The signed bytes are:

```text
<timestamp>.<event-id>.<exact raw request body>
```

KnowledgeOps accepts timestamps within five minutes, caps bodies at 64 KiB,
uses constant-time HMAC comparison, and stores only the event type and body
digest in its replay-protection ledger. Reusing an event ID with a different
body is rejected.

An accepted event is only a refresh signal. Its JSON body is never treated as
evidence. A Celery task rereads the subscription's fixed typed REST resource
using the scoped CompOps credential, current tool policy, response-size limits,
and normal audit trail. The polling deadline is also advanced immediately, so
Celery Beat provides recovery if event dispatch is temporarily unavailable.

## Recommended deployment

- Give each KnowledgeOps agent or service identity its own CompOps researcher
  token, scoped to one CompOps project.
- Apply owner/admin approval policies to run submission, batch creation, and
  action approval/rejection.
- Keep both systems behind private ingress. For a Docker/internal CompOps
  hostname, add that exact hostname to
  `EXTERNAL_GATEWAY_PRIVATE_HOST_ALLOWLIST`; wildcards and CIDR ranges are not
  accepted.
- Use the same correlation request ID in KnowledgeOps audit records and CompOps
  audit events.
- Import only hashes, stable identifiers, metrics, gate decisions, and artifact
  lineage into KnowledgeOps evidence; retrieve raw IR from CompOps on demand.
