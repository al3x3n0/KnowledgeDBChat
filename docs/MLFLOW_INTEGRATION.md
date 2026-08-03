# MLflow integration

KnowledgeOps Lab treats MLflow as a typed, read-only research evidence provider.
It does not expose arbitrary tracking-server paths. Every request must match a
registered capability, pass the tool-policy engine, and use the hardened
external-system gateway.

## Supported capabilities

| Capability | MLflow REST operation |
| --- | --- |
| `mlflow.experiments.search` | `POST /api/2.0/mlflow/experiments/search` |
| `mlflow.experiments.get` | `GET /api/2.0/mlflow/experiments/get` |
| `mlflow.runs.search` | `POST /api/2.0/mlflow/runs/search` |
| `mlflow.runs.get` | `GET /api/2.0/mlflow/runs/get` |
| `mlflow.artifacts.list` | `GET /api/2.0/mlflow/artifacts/list` |
| `mlflow.registered_models.get` | `GET /api/2.0/mlflow/registered-models/get` |
| `mlflow.model_versions.get` | `GET /api/2.0/mlflow/model-versions/get` |

The mapping follows the official
[MLflow REST API](https://mlflow.org/docs/latest/api_reference/rest-api.html).
Search requests are bounded to 100 results and 20 experiment IDs.

## Authentication

Connections support:

- no authentication;
- bearer tokens;
- configurable API-key headers;
- HTTP Basic authentication using a vault secret in `username:password` form.

Credentials are stored in the encrypted user-secret vault. Connection records
contain only the secret identifier. MLflow documents bearer and Basic
authentication for remote tracking servers in its
[tracking-server guide](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/).

## Evidence boundary

When an MLflow request is attached to an autonomous R&D job:

1. the full response is retained in the tool audit record;
2. the research trajectory receives only allowlisted provenance;
3. the provenance contains the provider, capability, remote identifiers,
   response digest, response size, request ID, timing, and audit ID;
4. the imported evidence starts as `unverified`;
5. the normal verification planner can propose an independent reproduction.

Metrics, parameters, tags, artifact contents, and model descriptions are not
copied into the canonical research outcome.

## Operational setup

1. Open **Tools → MLflow research tracking**.
2. Store or select the tracking credential.
3. Register the HTTPS tracking-server base URL.
4. Select the minimum read capabilities required.
5. Run **Test tracking API**.
6. Open an autonomous R&D job and use **Import MLflow evidence**.

Private tracking servers remain blocked by default. Operators may add an exact
hostname to `EXTERNAL_GATEWAY_PRIVATE_HOST_ALLOWLIST` when the deployment has a
controlled network path to that host.

## Deliberate omissions

This first integration does not create runs, log metrics, upload artifacts,
modify registered models, or subscribe to MLflow webhooks. Those operations
require separate write capabilities and policy defaults rather than being
implicitly granted by a read connection.
