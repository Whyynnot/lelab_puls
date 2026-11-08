"""Check imports, pinned runtime APIs and cache selection without moving a robot."""
import importlib.metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'lelab/src'), str(ROOT / 'lelab/bridge')]
from lelab.cache import configure_model_cache


def main():
    cache = configure_model_cache()
    errors = []
    versions = {}
    for package in ('LeLab', 'lerobot', 'torch', 'torchvision', 'torchcodec', 'transformers', 'rustypot'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f'Missing dependency: {package}')
    if versions.get('lerobot') != '0.7.0':
        errors.append('This release targets LeRobot 0.7.0')
    try:
        from native_modes import selected_description
        from lerobot.inference.contracts import ExecutionMode
        from lerobot.inference.rtc import validate_local_chunk_policy
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        selected_description({'type': 'smolvla', 'chunk_size': 50, 'n_action_steps': 10}, 'rtc')
        import app  # register routes; no lifespan startup, hardware work or motion
    except Exception as exc:
        errors.append(f'Runtime API/import mismatch: {type(exc).__name__}: {exc}')
    frontend = (ROOT / 'lelab/src/frontend/dist/index.html').is_file()
    if not frontend:
        errors.append('Frontend not built; run bash scripts/build_frontend.sh')
    print(json.dumps({'versions': versions, 'model_cache': str(cache),
                      'robot_data_home': os.environ.get('HF_LEROBOT_HOME'),
                      'frontend_built': frontend, 'errors': errors}, ensure_ascii=False, indent=2))
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
