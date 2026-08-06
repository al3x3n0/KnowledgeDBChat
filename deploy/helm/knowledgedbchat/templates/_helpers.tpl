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
