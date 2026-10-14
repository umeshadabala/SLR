#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=========================================================="
echo " [EXP-07] SLR Development Launcher: Python Engine + Flutter"
echo "=========================================================="

BACKEND_PID=""

cleanup() {
  echo ""
  echo ">> Shutting down EXP-07 development environment..."
  if [ -n "${BACKEND_PID}" ]; then
    kill "${BACKEND_PID}" 2>/dev/null || true
    echo ">> Stopped Python backend (PID ${BACKEND_PID})."
  fi
  exit 0
}

trap cleanup INT TERM EXIT

echo ">> Starting Python SLR Backend Service on 127.0.0.1:8000..."
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/backend/src:${PYTHONPATH:-}"
"${ROOT_DIR}/backend/.venv/bin/python" -m exp07.main &
BACKEND_PID=$!

echo ">> Waiting for backend health check..."
MAX_RETRIES=20
COUNT=0
HEALTHY=0

while [ $COUNT -lt $MAX_RETRIES ]; do
  if curl -s "http://127.0.0.1:8000/health" > /dev/null 2>&1; then
    HEALTHY=1
    break
  fi
  sleep 0.5
  COUNT=$((COUNT + 1))
done

if [ $HEALTHY -eq 1 ]; then
  echo ">> Backend is healthy! (http://127.0.0.1:8000/health)"
else
  echo ">> Warning: Backend did not respond within 10s. Continuing anyway..."
fi

echo ""
echo ">> Launching Flutter Application..."
cd "${ROOT_DIR}/app"
flutter run -d chrome --web-port 3000
