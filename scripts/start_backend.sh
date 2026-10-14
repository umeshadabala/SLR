#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=== [EXP-07] Starting SLR Backend Engine ==="
cd "${ROOT_DIR}"

VENV_PY="${ROOT_DIR}/backend/.venv/bin/python"

if [ ! -f "${VENV_PY}" ]; then
  echo "Error: Backend virtual environment not found at backend/.venv."
  echo "Please run: uv venv backend/.venv --python 3.12 && uv pip install -e backend"
  exit 1
fi

export PYTHONPATH="${ROOT_DIR}/backend/src:${PYTHONPATH:-}"
exec "${VENV_PY}" -m exp07.main
