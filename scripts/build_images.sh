#!/usr/bin/env bash
set -euo pipefail

# Build all three service images and tag them for the Minikube registry.
# Run 'eval $(minikube docker-env)' first so images load directly into Minikube.

TAG="${TAG:-latest}"

echo "==> Building orchestrator..."
docker build -t "orchestra-ai/orchestrator:${TAG}" ./orchestrator

echo "==> Building knowledge-vault..."
docker build -t "orchestra-ai/knowledge-vault:${TAG}" ./knowledge-vault

echo "==> Building executive-sandbox..."
docker build -t "orchestra-ai/executive-sandbox:${TAG}" ./executive-sandbox

echo "==> All images built with tag: ${TAG}"
docker images | grep orchestra-ai
