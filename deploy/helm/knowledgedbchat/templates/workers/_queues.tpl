{{/* Single source of truth for the queues consumed and measured by workers. */}}
{{- define "kdbc.celery.queues" -}}
{{- $queues := .Values.celery.queues | default "celery" -}}
{{- if not .Values.celeryAgents.enabled -}}
{{- $queues = printf "%s,%s" $queues (.Values.config.extra.AGENT_JOB_CELERY_QUEUE | default "agents") -}}
{{- end -}}
{{- $queues -}}
{{- end -}}

{{- define "kdbc.celeryLatex.queues" -}}
{{- .Values.config.extra.LATEX_COMPILER_CELERY_QUEUE | default "latex" -}}
{{- end -}}

{{- define "kdbc.celeryTranscription.queues" -}}
{{- .Values.config.extra.TRANSCRIPTION_CELERY_QUEUE | default "transcription" -}}
{{- end -}}
