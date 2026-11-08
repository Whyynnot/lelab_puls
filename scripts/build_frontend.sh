#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${LELAB_PYTHON:-$ROOT/.venv/bin/python}"
"$PYTHON" - "$ROOT/lelab/src/frontend" <<'PY'
import os, sys
from nodejs_wheel import npm, executable
env = dict(os.environ)
env['PATH'] = str(executable.ROOT_DIR) + '/bin' + os.pathsep + env.get('PATH', '')
for args in (['ci', '--ignore-scripts'], ['run', 'build']):
    code = npm(args, cwd=sys.argv[1], env=env)
    if code:
        raise SystemExit(code)
PY
