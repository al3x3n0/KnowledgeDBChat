{{- define "kdbc.workerServiceAccountName" -}}
{{- if .Values.kubernetesSandbox.enabled -}}
{{- printf "%s-sandbox-runner" (include "kdbc.fullname" .) -}}
{{- else -}}
{{- include "kdbc.serviceAccountName" . -}}
{{- end -}}
{{- end -}}

{{- define "kdbc.kubernetesSandbox.env" -}}
{{- if .Values.kubernetesSandbox.enabled }}
{{- $profiles := deepCopy .Values.kubernetesSandbox.profiles }}
{{- if not (hasKey $profiles "default") }}{{- $_ := set $profiles "default" dict }}{{- end }}
{{- $default := index $profiles "default" }}
{{- if and .Values.global.imagePullSecrets (not (hasKey $default "imagePullSecrets")) }}
{{- $_ := set $default "imagePullSecrets" .Values.global.imagePullSecrets }}
{{- end }}
- name: SANDBOX_EXECUTION_BACKEND
  value: kubernetes
- name: KUBERNETES_SANDBOX_DEFER_ENABLED
  value: {{ .Values.kubernetesSandbox.defer.enabled | quote }}
- name: KUBERNETES_SANDBOX_POLL_BATCH_SIZE
  value: {{ .Values.kubernetesSandbox.defer.pollBatchSize | quote }}
- name: KUBERNETES_SANDBOX_DEFER_TOOLS
  value: {{ join "," .Values.kubernetesSandbox.defer.tools | quote }}
- name: KUBERNETES_SANDBOX_NAMESPACE
  value: {{ .Release.Namespace | quote }}
- name: KUBERNETES_SANDBOX_OWNER
  value: {{ include "kdbc.fullname" . | quote }}
- name: KUBERNETES_SANDBOX_PVC
  value: {{ default (printf "%s-data" (include "kdbc.fullname" .)) .Values.backend.persistence.existingClaim | quote }}
- name: KUBERNETES_SANDBOX_ROOT
  value: /app/data/kubernetes-sandbox
- name: KUBERNETES_SANDBOX_PVC_SUBPATH
  value: kubernetes-sandbox
- name: KUBERNETES_SANDBOX_SERVICE_ACCOUNT
  value: {{ printf "%s-sandbox-experiment" (include "kdbc.fullname" .) | quote }}
- name: KUBERNETES_SANDBOX_PROFILES
  value: {{ $profiles | toJson | quote }}
- name: KUBERNETES_SANDBOX_IMAGE_PROFILES
  value: {{ .Values.kubernetesSandbox.imageProfiles | toJson | quote }}
- name: KUBERNETES_SANDBOX_POLL_SECONDS
  value: {{ .Values.kubernetesSandbox.pollSeconds | quote }}
- name: KUBERNETES_SANDBOX_TTL_SECONDS
  value: {{ .Values.kubernetesSandbox.ttlSecondsAfterFinished | quote }}
- name: KUBERNETES_SANDBOX_MAX_TIMEOUT_SECONDS
  value: {{ .Values.kubernetesSandbox.maxTimeoutSeconds | quote }}
- name: KUBERNETES_SANDBOX_MAX_INPUT_MB
  value: {{ .Values.kubernetesSandbox.maxInputMB | quote }}
- name: KUBERNETES_SANDBOX_MAX_OUTPUT_MB
  value: {{ .Values.kubernetesSandbox.maxOutputMB | quote }}
- name: KUBERNETES_SANDBOX_LOG_BYTES
  value: {{ .Values.kubernetesSandbox.logBytes | quote }}
{{- end }}
{{- end -}}
