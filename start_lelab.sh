#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$ROOT/.env" ]]; then
    set -a
    source "$ROOT/.env"
    set +a
fi
PYTHON="${LELAB_PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    echo "Python environment missing. Run bash scripts/setup.sh first." >&2
    exit 1
fi
if [[ ! -f "$ROOT/lelab/src/frontend/dist/index.html" ]]; then
    echo "Frontend missing. Run bash scripts/build_frontend.sh first." >&2
    exit 1
fi
export LELAB_CACHE_ROOT="${LELAB_CACHE_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/lelab}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="$ROOT/lelab/bridge:$ROOT/lelab/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT/lelab/bridge"
exec "$PYTHON" "$ROOT/lelab/bridge/launch.py" "$@"
