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

## OpenAI-compatible LLM

Any OpenAI-compatible endpoint (OpenAI, DeepSeek, Qwen, Kimi, GLM, a local vLLM/Ollama-compatible server, …) is configured through `config.llmProvider: openai` plus the three settings below. The key is stored live in the cluster so nothing sensitive lands in git or in the values file; the base URL and model name are live config, kept out of the chart the same way:

- `OPENAI_API_KEY` → key in the chart-managed `Opaque` Secret `<fullname>-secrets`.
- `OPENAI_API_BASE` → entry in the chart-managed ConfigMap `<fullname>-config`.
- `DEFAULT_MODEL` → entry in the chart-managed ConfigMap `<fullname>-config`.

`<fullname>` is what the chart actually renders (see `_helpers.tpl`, `kdbc.fullname`): release `kdbc` plus chart name `knowledgedbchat` is **`kdbc-knowledgedbchat`**, so the objects are `kdbc-knowledgedbchat-secrets` / `-config`. Derive them dynamically rather than typing them, so a release rename can't break the patch:

```bash
NS=knowledgedbchat               # adjust to your namespace
REL=kdbc                         # the helm release name
FULL=$(kubectl -n $NS get secret --sort-by=.metadata.name \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' \
  | grep -E "^${REL}-.*-secrets$" | head -1 | sed -E "s/-secrets$//")
# -> kdbc-knowledgedbchat

# 1) the key, into the chart-managed Secret (idempotent: --type merge upserts the key)
kubectl -n $NS patch secret "${FULL}-secrets" --type merge \
  -p '{"stringData":{"OPENAI_API_KEY":"sk-...your-key..."}}'
# 2) the base URL and model, into the chart-managed ConfigMap
kubectl -n $NS patch configmap "${FULL}-config" \
  --type merge -p '{"data":{"OPENAI_API_BASE":"<endpoint>/v1","DEFAULT_MODEL":"<model-name>"}}'
# 3) restart so the running pods pick up the new env (the chart's checksum
#    annotation only covers the chart-rendered secret, not a live patch)
kubectl -n $NS rollout restart deployment "${FULL}-backend"
kubectl -n $NS rollout restart deployment "${FULL}-celery"
```

Both objects exist only **after** `helm install`/`upgrade`, so run the patch after the release is up. `helm upgrade --reuse-values` does not touch these, because the values file leaves `openaiApiKey` empty and `config.extra` unset — the patch is out-of-band.

Verify:

```bash
kubectl -n $NS get secret "${FULL}-secrets" -o jsonpath='{.data.OPENAI_API_KEY}' | base64 -d; echo
kubectl -n $NS get configmap "${FULL}-config" -o jsonpath='{.data.OPENAI_API_BASE}'; echo
kubectl -n $NS get configmap "${FULL}-config" -o jsonpath='{.data.DEFAULT_MODEL}'; echo
```

All three are out-of-band; `config.defaultModel` in the values file only supplies a fallback for non-OpenAI providers.

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
| `celery` | Deployment | default queue; optional CPU HPA or KEDA Redis queue scaler; configurable graceful drain (31 minutes by default). Takes the `agents` queue too when `celeryAgents` is disabled |
| `celery-agents` | Deployment | autonomous agent jobs (`agents` queue, `AGENT_JOB_CELERY_QUEUE`): hours long and mostly waiting on a model, so they no longer hold the general worker's processes. `celeryAgents.pool: threads` runs every job in one process, each on its own event loop: about a third of the idle memory, and run with sandbox, gem5, coding and research jobs together. A job that wedges holds its slot until the worker restarts, which is why prefork stays the default. `celeryAgents.autoscaling` (off; needs [KEDA](https://keda.sh)) scales on jobs running or able to start, one pod per `concurrency` — not CPU, since an idle-waiting worker shows none, and not queue depth, since a running job has left the queue. Scale-down is slow on purpose: removing a pod can interrupt a job, which then resumes from its last checkpoint |
| `celery-latex` | Deployment | off by default; read-only rootfs, all caps dropped |
| `celery-transcription` | Deployment | off by default; the only pod carrying Whisper and the audio stack (~250 MB the backend and celery pods no longer replicate). Its image builds `FROM` the backend image, so build that first (`BUILD_TRANSCRIPTION=1 deploy/minikube/bootstrap.sh`) |
| `celery-beat` | Deployment | pinned to 1 replica with `Recreate` — two schedulers double-fire tasks |
| `frontend` | Deployment | static CRA build, SPA fallback |
| `gateway` | Deployment | nginx: `/api` → backend, `/minio` → MinIO, `/video` → streamer, `/` → frontend |
| `video-streamer` | Deployment | Go service, shares the backend's JWT secret |
| `postgres` `redis` `qdrant` `minio` | StatefulSet | single replica, PVC-backed |
| `ollama` `kroki` | StatefulSet / Deployment | Ollama off by default. `kroki` runs this repo's own Mermaid renderer (`mermaid-renderer/`, 1.08 GB) on the Kroki companion protocol — the chart used to deploy the 3.76 GB `yuzutech/kroki` gateway on port 8000 while the backend defaulted to POSTing companion-style, so it rendered nothing locally |
| `<release>-migrate` | Job | Hook running `alembic upgrade head` — see below for the phase |
| `<release>-minio-bucket` | Job | `post-install,post-upgrade` hook, `mc mb --ignore-existing` |

### Scale Celery runners in production

The existing Celery worker Deployments are the Kubernetes job runners. Enable
[KEDA](https://keda.sh/docs/2.18/deploy/) in the cluster, including its CRDs and
metrics server, then apply the opt-in overlay **after your production values**:

```bash
helm upgrade --install kdbc deploy/helm/knowledgedbchat \
  -n knowledgedbchat --create-namespace \
  -f my-production-values.yaml \
  -f deploy/helm/knowledgedbchat/values-keda.yaml
```

If the release currently uses `celery.autoscaling.enabled: true` with `mode: cpu`,
first remove that CPU HPA with a separate upgrade using your current production
values plus `--set celery.autoscaling.enabled=false`. Wait for that upgrade to
complete before applying the overlay, so the CPU HPA and KEDA never control the
same Deployment together. Switching back requires the same intermediate step
to remove the ScaledObject before enabling the CPU HPA.

The overlay preserves replica bounds from your environment values and enables
autoscaling for general workers and any enabled dedicated
workers; it does not enable disabled LaTeX/transcription workers or change their
images. Keep registry, image tags, shared RWX storage and service endpoints in
your production values. Agent runners use the existing Postgres demand query;
the other runners use [Redis list length](https://keda.sh/docs/2.18/scalers/redis-lists/).
Celery Beat remains a single scheduler. Producers continue to enqueue Celery
tasks normally; no application Kubernetes API credentials are needed. These
are persistent worker pools: a normal Celery worker does not exit after one
task, so it is deployed as a Deployment rather than a batch Job.

Tune each worker's `autoscaling.minReplicas`, `maxReplicas`, `queueLength`,
`pollingIntervalSeconds`, `scaleDownStabilizationSeconds` and
`scaleDownPeriodSeconds` in a final values file after the overlay. `queueLength: 0`
uses worker concurrency as the target backlog per pod. General workers select
KEDA with `celery.autoscaling.mode: redis`; `cpu` keeps the previous CPU HPA.
The general scaler measures every queue in `celery.queues` (default `celery`),
including the agent queue when the dedicated agent worker is disabled. For
multiple queues the HPA uses the largest per-queue recommendation, rather than
summing their backlogs. LaTeX and transcription use the same configured queue
names for both their command and scaler (`LATEX_COMPILER_CELERY_QUEUE` and
`TRANSCRIPTION_CELERY_QUEUE` in `config.extra`).

Redis scalers use the chart's `redis` endpoint and database 0, matching the
generated `CELERY_BROKER_URL`; overriding that URL through extra environment
variables requires a matching scaler configuration. This profile assumes the
chart's ordinary, non-priority Redis lists and plaintext Redis connection.
Redis Cluster/Sentinel, TLS brokers and priority bucket lists need corresponding
KEDA trigger configuration before enabling this profile. When Redis uses auth,
set `secrets.redisPassword` to a nonempty placeholder for an existing Secret;
KEDA reads the real `REDIS_PASSWORD` via the worker's Secret reference. Set
`celeryQueueAutoscaling.kedaNamespace` if the operator runs outside `keda`;
with chart NetworkPolicies enabled it receives Redis access on port 6379.
The agent scaler has its own `celeryAgents.autoscaling.kedaNamespace` setting.
For managed Redis/Postgres, allow the KEDA operator through their external
firewall/network rules too.

Queue depth excludes running and prefetched tasks, so Redis scaling requires
at least one replica. Scale-down waits for stable demand and removes at most
one pod per configured period. Celery receives SIGTERM and drains tasks within
`terminationGracePeriodSeconds`: 1860 seconds for general/LaTeX and 21660 for
transcription by default. Increase this if task limits are raised or general
workers also run hours-long agent jobs. A force-killed or OOM-killed task still
follows its existing Celery acknowledgement/retry behavior; this does not add
exactly-once execution. Draining pods can temporarily keep resource usage above
the autoscaler's replica ceiling. Budget cluster capacity and database
connections for all worker pools at their maximum concurrency. Long drains also
extend rollouts; allow enough time in Helm's `--timeout` and maintenance windows.

Check readiness and observe scaling:

```bash
kubectl -n knowledgedbchat get scaledobjects,hpa
kubectl -n knowledgedbchat describe scaledobject kdbc-knowledgedbchat-celery
kubectl -n knowledgedbchat get pods -l app.kubernetes.io/component=celery -w
```

`ScaledObject` conditions should show `Ready=True`; inspect events for broker
connectivity/authentication errors. In staging, enqueue a representative burst,
observe replicas increase, verify task completion, and let the queue drain to
check delayed scale-down. Local checks are `make helm-test` (Python PyYAML +
Helm) and `make helm-validate` (also requires kubeconform and schema access).

### Independent experiment Jobs for CHIA-style loops

Apply `values-kubernetes-jobs.yaml` to move common sandbox steps onto independent
Kubernetes Jobs. It can be used with static Celery workers or with KEDA:

```bash
helm upgrade --install kdbc deploy/helm/knowledgedbchat \
  -n knowledgedbchat --create-namespace \
  -f my-production-values.yaml \
  -f deploy/helm/knowledgedbchat/values-keda.yaml \
  -f deploy/helm/knowledgedbchat/values-kubernetes-jobs.yaml
```

This covers callers of `agent_sandbox_runtime.run_in_sandbox`: compiler and
benchmark tools, research-program tools using that interface, sandbox skills,
and custom Python tools. Direct Docker integrations (including generic Docker
tools, image builds, and older experiment/demo runners) retain their existing
backend. The API's own sandbox backend is unchanged; the chart configures the
general and agent Celery workers to use `SANDBOX_EXECUTION_BACKEND=kubernetes`.
Those workers no longer need the privileged Docker sidecar for common sandbox
steps. For mixed workloads, set `kubernetesSandbox.keepDockerSidecar: true`
alongside `sandbox.enabled: true` to retain Docker for those direct integrations.

Prerequisites: a real RWX data PVC, a CNI enforcing NetworkPolicy, and tool
images in a registry reachable by the cluster. Every tool image must include
`python3` and `/bin/sh`. Execution still requires
`config.extra.ENABLE_UNSAFE_CODE_EXECUTION: "true"` and allowlisted images;
enabling the runner alone does not enable code execution. The launcher has a
per-UID process limit of 256; configure the kubelet's `podPidsLimit` on sandbox
nodes as the cgroup process bound. A sandbox RuntimeClass/user namespace can
also isolate process accounting between experiments sharing uid 65534.

Each journaled action receives a stable step identity. Inputs, the submitted
script, Job/pod identity, node, resolved image ID, terminal result, logs and
output files live under
`/app/data/kubernetes-sandbox/<job-name>/`; original inputs are retained beside
the writable work directory. A lost Celery waiter leaves the bounded Job
running. Retrying the same journaled action and sandbox-call order reattaches
to that Job or restores its recorded result, including files. Changed inputs
or settings are rejected rather than silently submitting another experiment.
Keep execution journaling enabled (the default) for recovery. Calls outside a
journaled action receive a fresh identity and do not provide retry deduplication.
Skill workspaces and deterministic caches also move to shared data directories
(`sandbox-workspaces` and `sandbox-result-cache`). Pin image digests to enable
deterministic caching across nodes; mutable tags bypass that cache.
Cache keys include file contents and permission bits, and replay applies file
deletions and executable modes before a later step consumes the workspace.
Older cache entries are bypassed after this format change. Missing or damaged
archives fall back to execution without first changing the input files.

For journaled primary actions in the native agent loop, the worker now returns
after submission, releasing its slot and execution lease. A short general-queue
task scheduled by Celery Beat every 30 seconds checks experiments and queues
ready continuations. The saved action resumes within its original iteration;
it does not make another observation/model decision or charge another tool
invocation. Original runtime deadlines persist across these continuations and
bound new experiment submissions. A completed tool result is checkpointed
before its remaining bookkeeping, so replay can collect it without calling
the tool again. Manual approval/reconciliation checkpoints still require their
existing decisions.

For a skill with a separate judge Job, replay can recognize the judge's
recorded inputs/outputs if they were restored before the worker lost its
checkpoint. The saved continuation must match the action identity, namespace,
step, request fingerprint, Job UID and workspace. This lets the earlier
command restore its own result before collecting the judge again, without
submitting either experiment twice. Changes outside those recorded snapshots
are rejected. Cache hits reserve both step and workspace order.

Output restoration stages and checks the recorded snapshot before changing
the caller's workspace. It installs complete files with atomic renames and
then removes obsolete files; interrupted restores can replay known input and
output versions without accepting partial file contents. Workspace locks
serialize restores and input snapshots, including across steps of one action.
The worker needs a writable parent directory on the workspace filesystem for
sibling staging directories. A process killed during restore may leave a
staging directory for the existing workspace/OS retention cleanup.

This is enabled by `kubernetesSandbox.defer.enabled` (default true with this
backend). Celery Beat and a general worker must be running; deployments using
an external scheduler must schedule `poll_sandbox_continuations` themselves.
Compiler, cycle analysis, benchmark, comparison, sandbox-skill and custom-tool
primary actions are supported. Wall-clock benchmarks and comparisons keep
their Celery waiter while holding the measurement lock, so suspension cannot
release the claim before the remote Job finishes. If the waiter is lost or
submission fails with an unknown outcome, that claim remains until its bounded
TTL expires. When serialization is disabled or the lock cannot be acquired,
the measurement reports that limitation and can use continuation waiting.
Verification/recovery side actions, externally
driven harnesses and calls outside the native journaled action path retain
synchronous waiting. Add other replay-safe tool names to
`kubernetesSandbox.defer.tools`; tools with other side effects must make those
effects idempotent before opting in. Redacted journal parameters retain
synchronous execution rather than being stored for automatic replay. Set
`defer.enabled: false` to use synchronous waiting for new calls.

Waiting experiments continue to count toward their owner's running-job cap,
including cancelled experiments until deletion is confirmed. They do not
inflate the agent worker's KEDA demand while paused; ready continuations do.
Already counted continuations can resume after a cap reduction, allowing them
to finish and release capacity.
KEDA counts a pending continuation even if it still carries an older cap-wait
reason, so it can wake workers from zero replicas. When an ordinary cap-waiting
job loses its delayed delivery, maintenance requeues it after capacity is
available and clears that old reason before publishing. Cap-blocked jobs do
not spend the never-started retry budget while their owner remains full.
The poller retries lost broker dispatches after a two-minute grace window,
but leaves a ready continuation alone if Redis still reports its queued or
reserved message. The broker check runs once per eligible batch, outside row
locks, in pages of at most 256 list entries with a 10,000-message/two-second
scan budget (plus a final socket operation). Incomplete scans preserve positive
sightings; unknown or missing messages retain the recovery retry behavior.
The poller uses row locks to avoid concurrent claims, and scans at most `defer.pollBatchSize`
(default 20) per tick with a short batch time bound. API errors are retained
on `results.sandbox_wait` and retried fairly; changed identities require
operator review. The generic paused-job sweep and recurring scheduler leave
these waiting actions alone. Drain/cancel pending experiments before disabling
the Kubernetes backend or its polling scheduler.
Restart/relaunch and deletion return a conflict while an experiment reference
still needs reconciliation, so those actions cannot lose the cancellation
handle. Once cleanup finishes they work normally. Fresh runs get fresh
journal identities; explicit retries retain their original identity.

The stalled-job sweep also leaves ready continuations to the sandbox poller,
instead of treating them as jobs that never started. If a continuation worker
dies, its expired lease returns the action to experiment waiting, including at
the iteration limit: finishing the saved action does not start another
iteration. The existing three-attempt worker-crash limit still applies.
Retention cleanup keeps experiment references/checkpoints until cleanup is
confirmed, and keeps terminal jobs while a worker still holds a live lease.
Maintenance claims use row locks, and recovery messages are sent after commit.

Cancellation verifies the request fingerprint and Job UID, and sends a UID
precondition with foreground deletion so a replacement Job cannot be deleted
by an old execution. On agent timeout, the timeout decision is persisted before
deletion and the action remains waiting until deletion is confirmed, preserving
the owner's capacity limit while pods terminate. A late experiment success
does not override an already requested timeout.

Jobs use one pod, no automatic Job retries (`backoffLimit: 0`), a deadline and
explicit CPU/memory requests and limits. Timeout includes scheduling/startup
time. The default step ceiling is six hours; existing tool-specific limits may
be shorter. The original agent runtime deadline can shorten it further.
Cancellation is observed by the live waiter or the continuation poller; losing
the waiter leaves the Job bounded by its deadline. Completed Kubernetes
objects expire after one day by default. Persistent records/artifacts are
retained for recovery and audit: provision storage quotas, monitor free space,
and remove terminal records only after their owning runs no longer need replay.
Per-file output limits and bounded input/output collection do not impose a
total live PVC quota. Kubernetes may still replace or duplicate a Pod under
failure, so experiment code should be idempotent; submission deduplication is
not an exactly-once execution guarantee.

Job pods mount only their own workspace, read-only launcher/control directory,
and output directory from the data claim. They receive no app environment,
application Secrets, API token, or worker owner-reference. They run as uid
65534 with a read-only root filesystem, dropped capabilities, seccomp and a
deny-all ingress/egress policy. The trusted Celery service account can
create/get/delete Jobs and list pod statuses in the release namespace. The
experiment service account has no RoleBinding and disables token mounting.
Ensure other namespace policies do not grant network access to sandbox pods;
NetworkPolicy allow rules are additive.
The experiment deny policy carries `helm.sh/resource-policy: keep` so disabling
the runner or uninstalling the release does not open networking for Jobs still
running. Remove that retained policy only after those Jobs have terminated.

Operator-defined resource profiles select scheduling and extended resources
for exact image names; CPU/memory stay bounded by
`SCIENTIFIC_VALIDATION_MAX_CPUS` / `SCIENTIFIC_VALIDATION_MAX_MEMORY_MB`:

```yaml
kubernetesSandbox:
  profiles:
    default:
      runtimeClassName: sandboxed       # must already exist in the cluster
    simulation:
      nodeSelector: {workload: simulation}
    gpu:
      nodeSelector: {workload: gpu}
      extraResources: {nvidia.com/gpu: "1"}
  imageProfiles:
    "registry.example.com/gem5@sha256:<digest>": simulation
    "registry.example.com/evaluator@sha256:<digest>": gpu
```

Profiles inherit the default profile and registry pull-secret names. Check a
staging run with `kubectl -n knowledgedbchat get jobs,pods -l
app.kubernetes.io/component=sandbox-job`, stop its Celery waiter, retry the
journaled action, and verify that the Job identity is reused and its artifact
is restored. The tool result records `sandbox_jobs` plus artifact references
in the existing execution journal. This path is unit/schema tested; live
worker-loss and storage behavior must be checked against the production CNI,
CSI driver and sandbox RuntimeClass before rollout.

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

`backend`, `celery`, `celery-agents` and `celery-beat` all mount the same `<release>-data` PVC at
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

   Unset, the bundle falls back to `http://localhost:28000` and every request from
   a deployed page misses the gateway. `REACT_APP_VIDEO_STREAM_URL` is what makes
   video work off-compose: `getDocumentDownloadUrl` otherwise uses the page origin
   only when the API base URL shares it, which a gateway install does not
   guarantee, and falls through to `http://localhost:28080/stream/...`.
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

## Sandboxed tools

Compiler, gem5, BOLT, sandbox-skill and scientific-validation tools, and MCP
`docker_execute`, run `docker run` against a private daemon. It is **off by
default** in both production shapes, because the daemon is a privileged
container; with it off those tools report that they could not run.

**Compose:** add the overlay.

```bash
docker compose -f docker-compose.prod.yml -f docker-compose.sandbox.yml up -d --build
make sandbox-images
make sandbox-load DC="docker compose -f docker-compose.prod.yml -f docker-compose.sandbox.yml"
```

**Helm:** `sandbox.enabled=true` adds the daemon as a sidecar to the backend,
`celery` and `celery-agents` pods (`sandbox.components` narrows that). Each pod
talks to its own daemon over a unix socket in an `emptyDir`; nothing is exposed
on the network. It needs:

- a backend image built with `--build-arg WITH_DOCKER_CLI=true`;
- the sandbox images pushed to a registry the cluster can reach, and named in
  `config.extra.SCIENTIFIC_VALIDATION_ALLOWED_DOCKER_IMAGES` (they are not
  published; `sandbox.registryAuthSecret` names a `dockerconfigjson` Secret
  for a private registry);
- `config.extra.UNSAFE_CODE_EXEC_BACKEND: "docker"`;
- a namespace whose Pod Security level admits privileged pods, and, with
  `networkPolicy.enabled`, egress from those pods to the registry.

The KEDA overlay also sets `sandbox.workerNativeSidecar: true`. When sandboxing
is enabled, general and agent workers run Docker as a restartable init
container (`restartPolicy: Always`). Kubernetes keeps this daemon running until
the Celery container has stopped, so a scale-down does not stop tools while
tasks drain. This needs Kubernetes >=1.29 with the `SidecarContainers` feature
enabled (on by default since 1.29); see the
[Kubernetes sidecar lifecycle](https://kubernetes.io/docs/concepts/workloads/pods/sidecar-containers/).
For older clusters set `sandbox.workerNativeSidecar: false`; regular sidecars
do not guarantee shutdown order. Without the KEDA overlay this setting defaults
to false for compatibility. Neither setting enables sandboxing itself.

Limits worth knowing: the image store and the work directory are per pod. A
restarted pod pulls its images again, and a pipeline stage that runs on a
different pod from its parent starts without the parent's files (it is told
so). With one `celery-agents` replica neither arises. The chart is rendered
and schema-validated with the sidecar on in CI; it has not been run on a
cluster.

## Production checklist

1. Push images to a registry; set `image.*.repository`/`tag` and `global.imagePullSecrets`.
2. `secrets.create: false` + `existingSecret` from sealed-secrets/external-secrets/SOPS.
3. Managed Postgres/Redis/Qdrant/object storage (`<dep>.enabled: false`).
4. RWX `backend.persistence.storageClass`, or single-node pinning.
5. `ingress.enabled: true` with TLS; set `config.minioProxyBaseUrl` to the public host.
6. `backend.autoscaling` + `backend.podDisruptionBudget`, `celery.autoscaling`.
   Autoscaling the backend multiplies its database connections: each API
   process holds its own pool (`backend.dbPool`), so the chart refuses to
   render unless `(size + maxOverflow) x workers x maxReplicas`, plus 40 for
   workers, fits in `postgres.maxConnections` (or
   `postgres.external.maxConnections`). Shrink the pool, raise the limit, or
   front the database with PgBouncer / RDS Proxy and give its limit.
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

## Directory sign-in (LDAP / Active Directory)

Optional. Put the `LDAP_*` settings under `config.extra` and
`LDAP_BIND_PASSWORD` in the Secret; `values-prod.example.yaml` has a commented
block. The connection must be `ldaps://` or StartTLS, and the server's
certificate is verified, so an internal CA needs `LDAP_CA_CERT_FILE` (mount the
bundle into the backend pod). Once deployed, check it from
Admin → Directory (LDAP) → Run test, which reports the step that fails.
