#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$ROOT/.browsers}"
export CHAINLIT_NO_TELEMETRY=true
if [ ! -x .venv/bin/python ]; then
  printf 'Run bash scripts/setup_local.sh first.\n' >&2
  exit 1
fi
.venv/bin/python scripts/check_setup.py
exec .venv/bin/chainlit run src/cc_genclaw/ui/chainlit_app.py --host 127.0.0.1 --port "${GENCLAW_PORT:-8000}"
