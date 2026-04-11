#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="axiom-mesh"
RELEASE="orchestra"
VALUES_FILE="${VALUES_FILE:-helm/orchestra/values.dev.yaml}"

echo "==> Deploying Orchestra AI to namespace: ${NAMESPACE}..."

helm upgrade --install "${RELEASE}" ./helm/orchestra \
  --namespace "${NAMESPACE}" \
  --create-namespace \
  --values "${VALUES_FILE}" \
  --set secrets.geminiApiKey="${GEMINI_API_KEY:-}" \
  --set secrets.anthropicApiKey="${ANTHROPIC_API_KEY:-}" \
  --set secrets.googleApiKey="${GOOGLE_API_KEY:-}" \
  --wait \
  --timeout 5m

echo "==> Deployment status:"
kubectl get all -n "${NAMESPACE}"
