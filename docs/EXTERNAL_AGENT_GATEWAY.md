# External Agent Gateway

KnowledgeOps Lab can register an external agent as a capability-scoped user tool.
The gateway is intended for explicit collaboration with known services, not
automatic discovery or trust of arbitrary internet agents.

It also supports typed external-system providers. The first is CompOps, the
domain-specific compiler research control plane. Its adapter maps a fixed
capability set onto known CompOps REST routes and records results as external
system evidence. See `docs/COMPOPS_INTEGRATION.md`.

## Security model

Each registration has:

- a fixed HTTPS endpoint
- an explicit capability manifest
- a bounded timeout
- an optional reference to an encrypted `UserSecret`
- a namespaced tool-policy identity (`user_tool:<agent-id>`)
- a `ToolExecutionAudit` record for every API invocation

The gateway rejects redirects, invalid JSON, oversized responses, non-public
network destinations, unregistered capabilities, embedded URL credentials, and
plaintext HTTP endpoints. Secrets are resolved at invocation time and are never
written into the registration response, tool input, or audit record.

DNS validation substantially reduces server-side request forgery risk. Production
deployments should additionally enforce an outbound network allowlist at the
container or infrastructure layer.

## Register an agent

First store any credential through `/api/v1/secrets`, then reference its ID:

```http
POST /api/v1/external-agents
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "Compiler Research Reviewer",
  "endpoint_url": "https://agents.example.com/compiler-review",
  "capabilities": ["compiler.review", "research.critique"],
  "auth_type": "bearer",
  "secret_id": "<user-secret-id>",
  "timeout_seconds": 30
}
```

For unauthenticated endpoints, use `"auth_type": "none"` and omit `secret_id`.

## Invoke an agent

```http
POST /api/v1/external-agents/<agent-id>/invoke
Authorization: Bearer <token>
Content-Type: application/json

{
  "capability": "compiler.review",
  "request_id": "compiler-review-trial-17",
  "payload": {
    "hypothesis": "The missed vectorization is caused by an aliasing decision.",
    "evidence": [{"id": "remark-diff-1", "kind": "compiler_remarks"}]
  }
}
```

The external endpoint receives:

```json
{
  "request_id": "compiler-review-trial-17",
  "capability": "compiler.review",
  "input": {
    "hypothesis": "The missed vectorization is caused by an aliasing decision.",
    "evidence": [{"id": "remark-diff-1", "kind": "compiler_remarks"}]
  }
}
```

Successful responses include a provenance block containing the registered agent,
capability, endpoint origin, response hash, byte count, and elapsed time.
When an autonomous job is finalized, this allowlisted provenance is retained in
the compact action ledger and projected into its canonical evaluation outcome as
`external_agent_response` evidence with `verification_status: unverified`. Request
payloads and raw response bodies are not copied into that ledger.

External-agent evidence cannot promote itself. The R&D verification layer resets
self-asserted status and requires an explicit link to locally recorded evidence
or artifacts. Such a link yields `corroborated`; reaching `verified` additionally
requires independent evidence, a replayable artifact, and a successful repeated
experiment. Independent evidence must originate from a runtime-observed local
finding, not agent-authored structured output. A locally grounded contradiction
yields `rejected`.

Unresolved responses produce proposal-only local verification tasks in the
canonical outcome. These tasks prohibit external-agent verification, declare
required artifacts and repeat counts, and require approval before launch.
Approved launches create an idempotent local `ExperimentPlan`, `ExperimentRun`,
and deterministic sandbox job through the autonomous R&D verification API.
Completed local runs reconcile a sanitized, run-scoped result into the parent
outcome. Successful controlled repetitions may verify the external response;
failed or blocked execution remains inconclusive and never becomes an automatic
contradiction.
The authenticated R&D job-outcome endpoint exposes the resulting lifecycle state
without returning raw external or local execution payloads.

## Approval and agent-runtime use

External agents use the existing user-tool path. Tool policies can deny them or
require owner-and-admin approval. Approved audit records can be executed through
the existing tool-audit API.

Internal autonomous agents can invoke a registered external agent through its
`user_tool:<agent-id>` identity. Returned claims should still be treated as
untrusted evidence until local validation and autonomous R&D graders accept them.
