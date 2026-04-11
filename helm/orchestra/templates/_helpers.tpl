{{/*
Common labels applied to all resources.
*/}}
{{- define "orchestra.labels" -}}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end }}

{{/*
Namespace shorthand.
*/}}
{{- define "orchestra.namespace" -}}
{{ .Values.namespace | default "axiom-mesh" }}
{{- end }}
