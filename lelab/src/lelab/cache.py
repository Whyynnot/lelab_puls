"""Set model caches before importing Hugging Face, torch or LeRobot."""
import os
from pathlib import Path

LEGACY_HUB = Path(os.environ.get('LELAB_LEGACY_HUB_CACHE', Path.home() / '.cache/huggingface/hub'))


def configure_model_cache() -> Path:
    old_hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    if (old_hf_home / "token").is_file():
        os.environ.setdefault("HF_TOKEN_PATH", str(old_hf_home / "token"))
    configured = os.environ.get("LELAB_CACHE_ROOT")
    if configured:
        root = Path(configured).expanduser()
    else:
        root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "lelab"
    # LELAB_CACHE_ROOT is authoritative. This runs before
    # libraries capture their constants, and children inherit the same paths.
    for key, path in {
        "HF_HOME": root / "huggingface",
        # New downloads never share files/refs/locks with the legacy migration.
        "HF_HUB_CACHE": root / "huggingface/hub-live",
        "HUGGINGFACE_HUB_CACHE": root / "huggingface/hub-live",
        "HF_DATASETS_CACHE": root / "huggingface/datasets",
        "HF_XET_CACHE": root / "huggingface/xet",
        "TORCH_HOME": root / "torch",
    }.items():
        os.environ[key] = str(path)
    # Keep recordings, robot calibration and job history on their existing path.
    os.environ.setdefault("HF_LEROBOT_HOME", str(Path.home() / '.cache/huggingface/lerobot'))
    os.environ.pop("TRANSFORMERS_CACHE", None)
    return root


def legacy_policy_snapshot(repo_id: str, subfolder: str = "") -> str | None:
    """Reuse a complete old snapshot read-only, pinned to the current Hub commit."""
    legacy = LEGACY_HUB
    if not (legacy / ('models--' + repo_id.replace('/', '--'))).is_dir():
        return None
    from huggingface_hub import HfApi, constants, try_to_load_from_cache
    import json

    try:
        revision = 'main' if constants.HF_HUB_OFFLINE else HfApi().model_info(repo_id).sha
        if not revision:
            return None
        filename = f'{subfolder}/config.json' if subfolder else 'config.json'
        config = try_to_load_from_cache(repo_id, filename, cache_dir=str(legacy), revision=revision)
        if not isinstance(config, str):
            return None
        directory = Path(config).parent
        if not all((directory / name).is_file() for name in (
                'policy_preprocessor.json', 'policy_postprocessor.json')):
            return None
        index = directory / 'model.safetensors.index.json'
        names = set(json.loads(index.read_text())['weight_map'].values()) if index.is_file() else {'model.safetensors'}
        if not names:
            return None
        from safetensors import safe_open
        for name in names:
            path = Path(name)
            if path.is_absolute() or '..' in path.parts or '\\' in name:
                return None
            with safe_open(directory / name, framework='pt') as handle:
                if not list(handle.keys()):
                    return None
        return str(directory)
    except Exception:
        # An uncached/new revision follows the normal live-cache download path.
        return None
