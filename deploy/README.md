# Kubernetes deployment

Helm chart and minikube tooling for KnowledgeDBChat.

```
deploy/
  helm/knowledgedbchat/          # the chart
    values.yaml                  # defaults (self-contained, production-shaped)
    values-minikube.yaml         # local single-node profile
    values-prod.example.yaml     # copy/edit for a real cluster
  minikube/bootstrap.sh          # start cluster -> build images -> install
```

## Quick start (minikube)

```bash
./deploy/minikube/bootstrap.sh
```

That starts a `knowledgedbchat` minikube profile, builds the backend, frontend
and video-streamer images **inside minikube's Docker daemon** (they are not
published to any registry), installs the chart, waits for rollout, and prints
the NodePort URL.

Useful knobs:

| Variable | Default | Effect |
| --- | --- | --- |
| `SKIP_START=1` | – | Use the running cluster as-is |
| `SKIP_BUILD=1` | – | Reinstall without rebuilding images |
| `BUILD_LATEX=1` | – | Also build the TeX Live worker and enable `celeryLatex` |
| `MEMORY` / `CPUS` / `DISK` | `8192` / `4` / `40g` | minikube sizing |
| `NAMESPACE` / `RELEASE` | `knowledgedbchat` / `kdbc` | Install target |
| `APP_URL` | `http://localhost:8080` | Browser-facing origin (see below) |

Then:

```bash
helm test kdbc -n knowledgedbchat                    # in-cluster smoke test
kubectl -n knowledgedbchat logs job/kdbc-knowledgedbchat-migrate   # Alembic output
kubectl -n knowledgedbchat get pods -w
```

## What gets deployed

| Workload | Kind | Notes |
| --- | --- | --- |
| `backend` | Deployment | gunicorn + uvicorn workers, `/health` probes, optional HPA/PDB |
| `celery` | Deployment | default queue; optional HPA; 120s graceful drain |
| `celery-latex` | Deployment | off by default; read-only rootfs, all caps dropped |
| `celery-beat` | Deployment | pinned to 1 replica with `Recreate` — two schedulers double-fire tasks |
| `frontend` | Deployment | static CRA build, SPA fallback |
| `gateway` | Deployment | nginx: `/api` → backend, `/minio` → MinIO, `/video` → streamer, `/` → frontend |
| `video-streamer` | Deployment | Go service, shares the backend's JWT secret |
| `postgres` `redis` `qdrant` `minio` | StatefulSet | single replica, PVC-backed |
| `ollama` `kroki` | StatefulSet / Deployment | Ollama off by default |
| `<release>-migrate` | Job | Hook running `alembic upgrade head` — see below for the phase |
| `<release>-minio-bucket` | Job | `post-install,post-upgrade` hook, `mc mb --ignore-existing` |

### Migrations

Alembic is the only source of schema truth, so the chart gives it a single
owner: the migration hook Job. Every long-running pod gets
`RUN_ALEMBIC_MIGRATIONS=false`, and the app containers override the image's
`ENTRYPOINT` so `entrypoint.sh` never runs migrations in parallel across
replicas.

**On upgrade it is always a `pre-upgrade` hook** — the schema lands before the
new pods roll, and a failed migration aborts the release with the old pods still
serving. (Verified: a deliberately failing migration leaves the previous
ReplicaSet untouched and does not even apply the new ConfigMap.)

**On install the phase depends on who owns the database.** Helm creates normal
resources only after pre-install hooks finish, so with the in-chart Postgres a
pre-install migration would wait for a database that cannot exist yet and hang
until its deadline. The chart therefore uses `post-install` when
`postgres.enabled=true`, and `pre-install` when you point it at an external
database that already exists. With the in-chart Postgres the API is briefly up
against an empty schema on a *first* install only; `/health` does not touch the
database, and there is no data to serve yet.

To run migrations from a dedicated image, override both:

```yaml
migrations:
  image: {repository: registry.example.com/kdbc/migrations, tag: "1.0.0"}
  command: ["alembic", "-c", "/app/alembic.ini", "upgrade", "head"]
```

The Job deliberately does **not** reference the chart's ServiceAccount: as a
pre-install hook it can run before that account exists, and naming a missing one
makes pod creation forbidden so the Job hangs instead of failing fast. For cloud
IAM auth set `serviceAccount.create=false` and `serviceAccount.name` to an
account you manage — the Job then uses it.

### The `/app/data` volume

`backend`, `celery` and `celery-beat` all mount the same `<release>-data` PVC at
`/app/data` (documents, ChromaDB, logs, HuggingFace/Whisper/torch caches), so it
defaults to `ReadWriteMany`. On minikube the single-node hostPath provisioner
serves RWX fine. **On a real cluster you need a genuine RWX class** (EFS,
Filestore, CephFS, Azure Files) — set `backend.persistence.storageClass`. If you
only have RWO, run `backend.replicaCount: 1` and pin all `/app/data` consumers to
one node with `nodeSelector`.

The PVC carries `helm.sh/resource-policy: keep`, so `helm uninstall` does not
destroy uploaded documents.

## Configuration

Everything in `backend/app/core/config.py` is reachable:

```yaml
config:
  llmProvider: anthropic
  extra:                        # any setting name, verbatim
    RAG_HYBRID_SEARCH_ENABLED: "true"
    AGENT_REQUIRE_TOOL_APPROVAL: "true"
```

`config.*` lands in a ConfigMap; `secrets.*` lands in a Secret. Both are
`envFrom`-imported by the backend and every worker, and connection URLs
(`DATABASE_URL`, `REDIS_URL`, `CELERY_*`) are assembled from Secret values via
Kubernetes `$(VAR)` expansion — no password is ever written into a ConfigMap.

### Secrets

For anything beyond local development, manage the Secret yourself:

```yaml
secrets:
  create: false
  existingSecret: knowledgedbchat-secrets
  redisPassword: "managed-externally"   # non-empty ⇒ build redis:// URLs with auth
```

The Secret must contain: `SECRET_KEY`, `POSTGRES_PASSWORD`, `REDIS_PASSWORD`,
`MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, plus any provider keys
(`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, …) and `SECRETS_ENCRYPTION_KEY`. Every
key is a valid environment-variable name so pods can `envFrom` it directly.

`SECRET_KEY` is also injected into video-streamer as `JWT_SECRET`; the two must
match or video playback rejects every token.

### External data services

Each dependency can be swapped for a managed one:

```yaml
postgres: {enabled: false, external: {host: ..., port: 5432, database: ..., username: ...}}
redis:    {enabled: false, external: {host: ..., port: 6379}}
qdrant:   {enabled: false, external: {url: "https://..."}}
minio:    {enabled: false, external: {endpoint: "s3.eu-west-1.amazonaws.com", useSSL: true}}
ollama:   {enabled: false, external: {url: "http://ollama.internal:11434"}}
```

The in-chart StatefulSets are single-replica with no backup story — fine for dev,
not a substitute for a managed database.

### The browser-facing origin

Two things must agree with how a user actually opens the app:

1. **The frontend bundle.** CRA inlines `process.env.REACT_APP_*` at build time,
   so the origin is fixed when the image is built, not when it runs. Three build
   args matter:

   ```bash
   docker build -f frontend/Dockerfile frontend \
     --build-arg REACT_APP_API_URL=https://kdbc.example.com \
     --build-arg REACT_APP_WS_URL=wss://kdbc.example.com \
     --build-arg REACT_APP_VIDEO_STREAM_URL=https://kdbc.example.com/video
   ```

   Unset, the bundle falls back to `http://localhost:8000` and every request from
   a deployed page misses the gateway. `REACT_APP_VIDEO_STREAM_URL` is what makes
   video work off-compose: `getDocumentDownloadUrl` otherwise infers the streamer
   route by testing the API base URL for port 3000, which only matches the
   docker-compose layout, and falls through to `http://localhost:8080/stream/...`.
   `bootstrap.sh` sets all three from `APP_URL`. **An origin change means
   rebuilding the frontend image** — don't combine a new `APP_URL` with
   `SKIP_BUILD=1`.
2. **`config.minioProxyBaseUrl`**, the origin plus `/minio`. Get it wrong and
   downloads 404 even though the API is healthy.

For ingress installs both are `https://<your-host>`. The minikube default is
`http://localhost:8080`, matching the port-forward `bootstrap.sh` prints; the
chart also exposes a NodePort (30080), but that is not routable from the host on
minikube's Docker driver, which is why it isn't the default.

### Ingress

Off by default. When enabled it fronts the gateway rather than routing paths
itself, which keeps the `/video/{id}` → `/stream/{id}` rewrite, WebSocket
upgrade and 2GB body limits identical to docker-compose and independent of which
ingress controller is installed.

```bash
minikube addons enable ingress
helm upgrade kdbc deploy/helm/knowledgedbchat -n knowledgedbchat --reuse-values \
  --set ingress.enabled=true --set ingress.hosts[0].host=kdbc.local
echo "$(minikube ip) kdbc.local" | sudo tee -a /etc/hosts
```

### NetworkPolicy

`networkPolicy.enabled=true` restricts the datastores to in-release traffic.
Needs a CNI that enforces policy — on minikube start with `--cni=calico`,
otherwise the objects are accepted and silently ignored.

## Production checklist

1. Push images to a registry; set `image.*.repository`/`tag` and `global.imagePullSecrets`.
2. `secrets.create: false` + `existingSecret` from sealed-secrets/external-secrets/SOPS.
3. Managed Postgres/Redis/Qdrant/object storage (`<dep>.enabled: false`).
4. RWX `backend.persistence.storageClass`, or single-node pinning.
5. `ingress.enabled: true` with TLS; set `config.minioProxyBaseUrl` to the public host.
6. `backend.autoscaling` + `backend.podDisruptionBudget`, `celery.autoscaling`.
7. `networkPolicy.enabled: true`.

Start from `values-prod.example.yaml`.

## Local checks

```bash
make helm-lint        # lint default + minikube + prod profiles
make helm-template    # render every profile
make helm-validate    # render + kubeconform against the Kubernetes API schemas
make helm-smoke       # install on the current cluster and assert its wiring
```

`make helm-smoke` (`deploy/smoke-test.sh`) is the one that catches what rendering
cannot. It installs into a throwaway namespace using lightweight stand-ins for
the application images — real Postgres/Redis/Qdrant/MinIO/gateway, stubbed
Python — and asserts 21 behaviours, each corresponding to something that broke
during development:

- the migration hook completes on install (it used to hang: as a `pre-install`
  hook it ran before its ServiceAccount and before the in-chart Postgres existed)
- `DATABASE_URL`/`REDIS_URL` carry the Secret password via `$(VAR)` expansion,
  and the datastores actually accept those credentials
- app pods carry `RUN_ALEMBIC_MIGRATIONS=false`
- the MinIO bucket hook ran
- gateway routing, including that `/minio/` preserves the object path
- **a failing migration aborts an upgrade** with no pods rolled and no config applied
- a successful upgrade rolls pods via the config checksum annotation

It runs on any cluster kubectl points at:

```bash
kind create cluster --image kindest/node:v1.31.0
SMOKE_STORAGE_CLASS=standard make helm-smoke
KEEP=1 ./deploy/smoke-test.sh          # leave the release up to poke at
```

CI runs it on an ephemeral kind cluster in the `helm-smoke` job.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| Pods `ImagePullBackOff` | Images were not built into minikube's daemon — rerun `bootstrap.sh` without `SKIP_BUILD`, or check `eval $(minikube docker-env)` |
| Release fails at `pre-upgrade` | Alembic failed; `kubectl logs job/<release>-migrate` |
| Install fails: `PVC is not Bound` for `<release>-data` | The default StorageClass cannot provision `ReadWriteMany`. local-path provisioners (kind, k3s) log `Only support ReadWriteOnce access mode`; minikube's hostPath provisioner ignores access modes and works. Point `backend.persistence.storageClass` at an RWX class, or drop to `accessMode: ReadWriteOnce` with `backend.replicaCount: 1` and the workers pinned to the same node — see the `/app/data` section |
| 502 from `/api/` | Backend not ready yet; the gateway re-resolves DNS every 10s and recovers on its own |
| Downloads 404 but the API is fine | `config.minioProxyBaseUrl` does not match how the browser reaches the gateway |
| LLM calls fail | `config.llmProvider: ollama` with `ollama.enabled: false` and no `ollama.external.url` |
