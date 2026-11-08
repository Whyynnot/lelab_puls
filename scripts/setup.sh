#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3.12}"
"$PYTHON" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
# --skip-torch preserves a manually installed matching PyTorch stack.
if [[ "${1:-}" != --skip-torch ]]; then
    .venv/bin/python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
fi
.venv/bin/python -m pip install -c requirements/constraints.txt -e './lelab/src[test]'
.venv/bin/python -m pip install -c requirements/constraints.txt 'lerobot[pi,smolvla,diffusion]==0.7.0' nodejs-wheel-binaries==24.19.0
bash scripts/build_frontend.sh
.venv/bin/python scripts/doctor.py
