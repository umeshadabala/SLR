#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=========================================================="
echo " [EXP-07] Running Comprehensive Automated Test Suite"
echo "=========================================================="

echo ""
echo ">> [1/2] Backend PyTest Suite..."
cd "${ROOT_DIR}"
"${ROOT_DIR}/backend/.venv/bin/python" -m pytest backend/tests -v

echo ""
echo ">> [2/2] Flutter Analysis & Widget Tests..."
cd "${ROOT_DIR}/app"
flutter analyze
flutter test

echo ""
echo "=========================================================="
echo " [EXP-07] ALL BACKEND AND FLUTTER TESTS PASSED SUCCESSFULLY"
echo "=========================================================="
