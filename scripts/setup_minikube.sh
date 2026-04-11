#!/usr/bin/env bash
set -euo pipefail

echo "==> Starting Minikube..."
minikube start --cpus=4 --memory=8192 --driver=docker

echo "==> Enabling addons..."
minikube addons enable metrics-server
minikube addons enable ingress

echo "==> Verifying cluster..."
kubectl cluster-info
kubectl get nodes

echo "==> Minikube ready. Run 'eval \$(minikube docker-env)' before building images."
