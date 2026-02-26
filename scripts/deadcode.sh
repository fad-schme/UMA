#!/usr/bin/env bash
set -euo pipefail

python -m pip install -U ruff vulture pip-check-reqs >/dev/null

echo "== Ruff: unused imports/vars =="
ruff check uma extensions --select F401,F841

echo
echo "== Vulture: likely dead code (min confidence 80) =="
vulture uma extensions --min-confidence 80 || true

echo
echo "== pip-check-reqs: extra/missing requirements (imports-based) =="
pip-extra-reqs uma extensions || true
pip-missing-reqs uma extensions || true
