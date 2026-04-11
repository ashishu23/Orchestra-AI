#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="axiom-mesh"

echo "==> Port-forwarding all Orchestra AI services..."
echo "    Orchestrator    → http://localhost:8000"
echo "    Knowledge-Vault → http://localhost:8001"
echo "    Exec-Sandbox    → http://localhost:8002"
echo "    Qdrant          → http://localhost:6333"
echo "Press Ctrl+C to stop all forwards."

kubectl port-forward -n "${NAMESPACE}" svc/orchestrator 8000:8000 &
PID1=$!
kubectl port-forward -n "${NAMESPACE}" svc/knowledge-vault 8001:8001 &
PID2=$!
kubectl port-forward -n "${NAMESPACE}" svc/executive-sandbox 8002:8002 &
PID3=$!
kubectl port-forward -n "${NAMESPACE}" svc/qdrant 6333:6333 &
PID4=$!

trap "kill $PID1 $PID2 $PID3 $PID4 2>/dev/null; echo 'Port-forwards stopped.'" EXIT
wait
