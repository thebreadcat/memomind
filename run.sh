#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt
fi

# If MEMOMIND_PORT is not set, prefer 7702 but auto-fallback when busy.
if [ -z "${MEMOMIND_PORT:-}" ]; then
  if lsof -iTCP:7702 -sTCP:LISTEN >/dev/null 2>&1; then
    export MEMOMIND_PORT=7712
  else
    export MEMOMIND_PORT=7702
  fi
fi

echo "Starting MemoMind on http://localhost:${MEMOMIND_PORT}"
exec .venv/bin/python memomind.py "$@"
