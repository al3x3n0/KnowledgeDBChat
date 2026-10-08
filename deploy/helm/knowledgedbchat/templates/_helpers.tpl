{{/* vim: set filetype=mustache: */}}

{{- define "kdbc.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "kdbc.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "kdbc.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "kdbc.labels" -}}
helm.sh/chart: {{ include "kdbc.chart" . }}
{{ include "kdbc.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "kdbc.selectorLabels" -}}
app.kubernetes.io/name: {{ include "kdbc.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{/* Per-component labels. Usage: include "kdbc.componentLabels" (dict "ctx" $ "component" "backend") */}}
{{- define "kdbc.componentLabels" -}}
{{ include "kdbc.labels" .ctx }}
app.kubernetes.io/component: {{ .component }}
{{- end -}}

{{- define "kdbc.componentSelectorLabels" -}}
{{ include "kdbc.selectorLabels" .ctx }}
app.kubernetes.io/component: {{ .component }}
{{- end -}}

{{- define "kdbc.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "kdbc.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{- define "kdbc.imagePullSecrets" -}}
{{- with .Values.global.imagePullSecrets }}
imagePullSecrets:
{{- toYaml . | nindent 2 }}
{{- end }}
{{- end -}}

{{/* ---------------------------------------------------------------------- */}}
{{/* Secret / config object names                                            */}}
{{/* ---------------------------------------------------------------------- */}}

{{- define "kdbc.secretName" -}}
{{- if .Values.secrets.existingSecret -}}
{{- .Values.secrets.existingSecret -}}
{{- else -}}
{{- printf "%s-secrets" (include "kdbc.fullname" .) -}}
{{- end -}}
{{- end -}}

{{- define "kdbc.configMapName" -}}
{{- printf "%s-config" (include "kdbc.fullname" .) -}}
{{- end -}}

{{/* ---------------------------------------------------------------------- */}}
{{/* Dependency hostnames                                                    */}}
{{/* ---------------------------------------------------------------------- */}}

{{- define "kdbc.postgres.host" -}}
{{- if .Values.postgres.enabled -}}
{{- printf "%s-postgres" (include "kdbc.fullname" .) -}}
{{- else -}}
{{- required "postgres.enabled=false requires postgres.external.host" .Values.postgres.external.host -}}
{{- end -}}
{{- end -}}

{{- define "kdbc.postgres.port" -}}
{{- if .Values.postgres.enabled -}}{{ .Values.postgres.service.port }}{{- else -}}{{ .Values.postgres.external.port }}{{- end -}}
{{- end -}}

{{- define "kdbc.postgres.database" -}}
{{- if .Values.postgres.enabled -}}{{ .Values.postgres.auth.database }}{{- else -}}{{ .Values.postgres.external.database }}{{- end -}}
{{- end -}}

{{- define "kdbc.postgres.username" -}}
{{- if .Values.postgres.enabled -}}{{ .Values.postgres.auth.username }}{{- else -}}{{ .Values.postgres.external.username }}{{- end -}}
{{- end -}}

{{- define "kdbc.redis.host" -}}
{{- if .Values.redis.enabled -}}
{{- printf "%s-redis" (include "kdbc.fullname" .) -}}
{{- else -}}
{{- required "redis.enabled=false requires redis.external.host" .Values.redis.external.host -}}
{{- end -}}
{{- end -}}

{{- define "kdbc.redis.port" -}}
{{- if .Values.redis.enabled -}}6379{{- else -}}{{ .Values.redis.external.port }}{{- end -}}
{{- end -}}

{{- define "kdbc.qdrant.url" -}}
{{- if .Values.qdrant.enabled -}}
{{- printf "http://%s-qdrant:%v" (include "kdbc.fullname" .) .Values.qdrant.service.httpPort -}}
{{- else -}}
{{- required "qdrant.enabled=false requires qdrant.external.url" .Values.qdrant.external.url -}}
{{- end -}}
{{- end -}}

{{- define "kdbc.minio.endpoint" -}}
{{- if .Values.minio.enabled -}}
{{- printf "%s-minio:%v" (include "kdbc.fullname" .) .Values.minio.service.apiPort -}}
{{- else -}}
{{- required "minio.enabled=false requires minio.external.endpoint" .Values.minio.external.endpoint -}}
{{- end -}}
{{- end -}}

{{- define "kdbc.minio.useSSL" -}}
{{- if .Values.minio.enabled -}}false{{- else -}}{{ .Values.minio.external.useSSL }}{{- end -}}
{{- end -}}

{{- define "kdbc.ollama.url" -}}
{{- if .Values.ollama.enabled -}}
{{- printf "http://%s-ollama:%v" (include "kdbc.fullname" .) .Values.ollama.service.port -}}
{{- else -}}
{{- .Values.ollama.external.url -}}
{{- end -}}
{{- end -}}

{{- define "kdbc.kroki.url" -}}
{{- if .Values.kroki.enabled -}}
{{- printf "http://%s-kroki:%v" (include "kdbc.fullname" .) .Values.kroki.service.port -}}
{{- else -}}
{{- .Values.kroki.external.url -}}
{{- end -}}
{{- end -}}

{{- define "kdbc.backend.serviceName" -}}
{{- printf "%s-backend" (include "kdbc.fullname" .) -}}
{{- end -}}

{{- define "kdbc.frontend.serviceName" -}}
{{- printf "%s-frontend" (include "kdbc.fullname" .) -}}
{{- end -}}

{{- define "kdbc.videoStreamer.serviceName" -}}
{{- printf "%s-video-streamer" (include "kdbc.fullname" .) -}}
{{- end -}}

{{- define "kdbc.minio.serviceName" -}}
{{- printf "%s-minio" (include "kdbc.fullname" .) -}}
{{- end -}}

{{/* ---------------------------------------------------------------------- */}}
{{/* Storage class resolution: component override -> global -> cluster default */}}
{{/* Usage: include "kdbc.storageClass" (dict "ctx" $ "sc" .Values.postgres.persistence.storageClass) */}}
{{/* ---------------------------------------------------------------------- */}}
{{- define "kdbc.storageClass" -}}
{{- $sc := default .ctx.Values.global.storageClass .sc -}}
{{- if $sc }}
storageClassName: {{ $sc | quote }}
{{- end }}
{{- end -}}

{{/* ---------------------------------------------------------------------- */}}
{{/* Connection env for backend + all celery workers.                        */}}
{{/* Credentials come from the Secret and are woven into the connection URLs  */}}
{{/* with Kubernetes $(VAR) expansion, so no password is ever written into a  */}}
{{/* ConfigMap and an externally managed Secret works unchanged.             */}}
{{/* ---------------------------------------------------------------------- */}}
{{/* All Secret keys are valid env var names so `envFrom` imports them directly. */}}
{{/* POSTGRES_PASSWORD/REDIS_PASSWORD are also declared explicitly here because  */}}
{{/* $(VAR) expansion only resolves against `env`, never against `envFrom`.      */}}
{{- define "kdbc.connectionEnv" -}}
- name: POSTGRES_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ include "kdbc.secretName" . }}
      key: POSTGRES_PASSWORD
- name: REDIS_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ include "kdbc.secretName" . }}
      key: REDIS_PASSWORD
- name: DATABASE_URL
  value: "postgresql://{{ include "kdbc.postgres.username" . }}:$(POSTGRES_PASSWORD)@{{ include "kdbc.postgres.host" . }}:{{ include "kdbc.postgres.port" . }}/{{ include "kdbc.postgres.database" . }}"
{{- $redisAuth := "" }}
{{- if .Values.secrets.redisPassword }}{{ $redisAuth = ":$(REDIS_PASSWORD)@" }}{{ end }}
{{- $redisUrl := printf "redis://%s%s:%v/0" $redisAuth (include "kdbc.redis.host" .) (include "kdbc.redis.port" .) }}
- name: REDIS_URL
  value: {{ $redisUrl | quote }}
- name: CELERY_BROKER_URL
  value: {{ $redisUrl | quote }}
- name: CELERY_RESULT_BACKEND
  value: {{ $redisUrl | quote }}
- name: QDRANT_URL
  value: {{ include "kdbc.qdrant.url" . | quote }}
- name: MINIO_ENDPOINT
  value: {{ include "kdbc.minio.endpoint" . | quote }}
- name: MINIO_USE_SSL
  value: {{ include "kdbc.minio.useSSL" . | quote }}
{{- with include "kdbc.ollama.url" . }}
- name: OLLAMA_BASE_URL
  value: {{ . | quote }}
{{- end }}
{{- with include "kdbc.kroki.url" . }}
- name: KROKI_URL
  value: {{ . | quote }}
{{- end }}
{{- end -}}

{{/* envFrom shared by backend + workers */}}
{{- define "kdbc.appEnvFrom" -}}
- configMapRef:
    name: {{ include "kdbc.configMapName" . }}
- secretRef:
    name: {{ include "kdbc.secretName" . }}
{{- with .Values.global.extraEnvFrom }}
{{- toYaml . | nindent 0 }}
{{- end }}
{{- end -}}

{{/* Data volume mount shared by backend + workers */}}
{{- define "kdbc.dataVolume" -}}
- name: data
{{- if .Values.backend.persistence.enabled }}
  persistentVolumeClaim:
    claimName: {{ default (printf "%s-data" (include "kdbc.fullname" .)) .Values.backend.persistence.existingClaim }}
{{- else }}
  emptyDir: {}
{{- end }}
{{- end -}}

{{- define "kdbc.dataVolumeMounts" -}}
- name: data
  mountPath: /app/data
- name: data
  mountPath: /root/.cache/huggingface
  subPath: hf_cache
- name: data
  mountPath: /root/.cache/knowledge_db_transcriber
  subPath: whisper_models
- name: data
  mountPath: /root/.cache/torch
  subPath: torch_cache
{{- end -}}

{{/* Checksum annotations so config/secret changes roll the pods */}}
{{- define "kdbc.configChecksums" -}}
checksum/config: {{ include (print $.Template.BasePath "/configmap-app.yaml") . | sha256sum }}
{{- if .Values.secrets.create }}
checksum/secret: {{ include (print $.Template.BasePath "/secret-app.yaml") . | sha256sum }}
{{- end }}
{{- end -}}

{{/*
API processes sharing the database: gunicorn workers x replicas, counting the
autoscaler's ceiling when it is on.
*/}}
{{- define "kdbc.apiProcesses" -}}
{{- $replicas := .Values.backend.replicaCount -}}
{{- if .Values.backend.autoscaling.enabled -}}
{{- $replicas = .Values.backend.autoscaling.maxReplicas -}}
{{- end -}}
{{- mul (int $replicas) (int .Values.backend.workers) -}}
{{- end -}}

{{/*
Refuse a configuration whose API pools cannot fit in the database's
max_connections. A pool is per process, so the demand is
(size + maxOverflow) x processes; 40 are left for Celery workers and tooling.
Without this the failure arrives under load, as "too many clients already" in
whichever pod asked last.
*/}}
{{- define "kdbc.checkConnectionBudget" -}}
{{- $limit := .Values.postgres.maxConnections -}}
{{- if not .Values.postgres.enabled -}}
{{- $limit = .Values.postgres.external.maxConnections -}}
{{- end -}}
{{- if gt (int $limit) 0 -}}
{{- $perProcess := add (int .Values.backend.dbPool.size) (int .Values.backend.dbPool.maxOverflow) -}}
{{- $processes := include "kdbc.apiProcesses" . | int -}}
{{- $needed := mul $perProcess $processes -}}
{{- $available := sub (int $limit) 40 -}}
{{- if gt (int $needed) (int $available) -}}
{{- fail (printf "database connection budget exceeded: the API may open %d connections (%d per process x %d processes) but max_connections=%d leaves %d after 40 reserved for workers. Lower backend.dbPool, backend.workers or the replica ceiling, or raise postgres.maxConnections (or put PgBouncer in front and set postgres.external.maxConnections to its limit)." $needed $perProcess $processes (int $limit) $available) -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Sandbox daemon (values: sandbox.*).

Sandboxed agent tools shell out to the `docker` CLI. In compose they reach one
shared daemon over TLS; a pod cannot do that, because `-v <dir>:/work` is
resolved on the daemon's filesystem and a volume shared between pods needs
ReadWriteMany storage most clusters lack. So each pod that runs sandbox tools
carries its own daemon as a sidecar: the socket and the work directory are
emptyDirs both containers mount at the same path, and nothing crosses the
network. The sidecar is privileged, which is why this is off by default.

All four take (dict "ctx" . "component" "<values key>").
*/}}
{{- define "kdbc.sandbox.on" -}}
{{- if and .ctx.Values.sandbox.enabled (has .component .ctx.Values.sandbox.components) -}}true{{- end -}}
{{- end }}

{{- define "kdbc.sandbox.env" -}}
{{- if include "kdbc.sandbox.on" . }}
- name: DOCKER_HOST
  value: unix:///run/sandbox/docker.sock
- name: TMPDIR
  value: /sandbox-work
{{- if .ctx.Values.sandbox.registryAuthSecret }}
- name: DOCKER_CONFIG
  value: /sandbox-registry-auth
{{- end }}
{{- end }}
{{- end }}

{{- define "kdbc.sandbox.volumeMounts" -}}
{{- if include "kdbc.sandbox.on" . }}
- name: sandbox-run
  mountPath: /run/sandbox
- name: sandbox-work
  mountPath: /sandbox-work
{{- if .ctx.Values.sandbox.registryAuthSecret }}
- name: sandbox-registry-auth
  mountPath: /sandbox-registry-auth
  readOnly: true
{{- end }}
{{- end }}
{{- end }}

{{- define "kdbc.sandbox.container" -}}
{{- if include "kdbc.sandbox.on" . }}
- name: sandbox-docker
  image: "{{ .ctx.Values.sandbox.image.repository }}:{{ .ctx.Values.sandbox.image.tag }}"
  imagePullPolicy: {{ .ctx.Values.sandbox.image.pullPolicy }}
  # A unix socket in a directory only this pod mounts: no TCP listener, so
  # no TLS to manage. The entrypoint's own TLS setup is switched off.
  args:
    - --host=unix:///run/sandbox/docker.sock
  env:
    - name: DOCKER_TLS_CERTDIR
      value: ""
  securityContext:
    privileged: true
  volumeMounts:
    - name: sandbox-run
      mountPath: /run/sandbox
    # The same path as in the app container: the daemon resolves `-v` here.
    # Not under /tmp, which the dind entrypoint covers with a tmpfs.
    - name: sandbox-work
      mountPath: /sandbox-work
    - name: sandbox-images
      mountPath: /var/lib/docker
  # Liveness, not readiness: a daemon that is down must not take the pod out
  # of service -- the tools that need it say so, and everything else works.
  livenessProbe:
    exec:
      command: ["docker", "--host=unix:///run/sandbox/docker.sock", "info"]
    initialDelaySeconds: 30
    periodSeconds: 30
    timeoutSeconds: 10
    failureThreshold: 4
  resources:
    {{- toYaml .ctx.Values.sandbox.resources | nindent 4 }}
{{- end }}
{{- end }}

{{- define "kdbc.sandbox.volumes" -}}
{{- if include "kdbc.sandbox.on" . }}
- name: sandbox-run
  emptyDir: {}
- name: sandbox-work
  emptyDir:
    sizeLimit: {{ .ctx.Values.sandbox.workSizeLimit }}
# The daemon's image store. An emptyDir, so a restarted pod pulls its images
# again; they must come from a registry the cluster can reach.
- name: sandbox-images
  emptyDir:
    sizeLimit: {{ .ctx.Values.sandbox.imageStoreSizeLimit }}
{{- if .ctx.Values.sandbox.registryAuthSecret }}
- name: sandbox-registry-auth
  secret:
    secretName: {{ .ctx.Values.sandbox.registryAuthSecret }}
    items:
      - key: .dockerconfigjson
        path: config.json
{{- end }}
{{- end }}
{{- end }}
