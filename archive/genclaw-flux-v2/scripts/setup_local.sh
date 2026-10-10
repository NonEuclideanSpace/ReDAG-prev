#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
if [ ! -x .venv/bin/python ]; then
  "$PYTHON_BIN" -m venv .venv
fi
.venv/bin/python -m pip install -e '.[test]' -c requirements-tested.txt
export PLAYWRIGHT_BROWSERS_PATH="$ROOT/.browsers"
.venv/bin/python -m playwright install chromium
if [ ! -f config.yaml ]; then cp config.example.yaml config.yaml; fi
printf 'Installed. Configure config.yaml, then run bash scripts/start_local.sh\n'
