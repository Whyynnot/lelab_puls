import os

from lelab.cache import configure_model_cache


def test_model_cache_replaces_inherited_paths_and_keeps_robot_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(os, 'environ', dict(os.environ))
    monkeypatch.setenv('LELAB_CACHE_ROOT', str(tmp_path / 'models'))
    monkeypatch.setenv('HF_HOME', str(tmp_path / 'old'))
    monkeypatch.setenv('HF_HUB_CACHE', 'D:/old/hub')
    monkeypatch.setenv('TRANSFORMERS_CACHE', 'D:/old/transformers')
    monkeypatch.setenv('TORCH_HOME', 'D:/old/torch')
    monkeypatch.setenv('HF_LEROBOT_HOME', str(tmp_path / 'recordings-and-calibration'))
    # All writes are environment changes. Directory creation belongs to the
    # actual cache consumer, so importing LeLab does not create drives.
    root = configure_model_cache()
    assert os.environ['HF_HOME'] == str(root / 'huggingface')
    assert os.environ['HF_HUB_CACHE'] == str(root / 'huggingface/hub-live')
    assert os.environ['TORCH_HOME'] == str(root / 'torch')
    assert os.environ['HF_LEROBOT_HOME'] == str(tmp_path / 'recordings-and-calibration')
    assert 'TRANSFORMERS_CACHE' not in os.environ
    assert not root.exists()


def _legacy_fixture(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import numpy as np
    from safetensors.numpy import save_file
    import huggingface_hub
    from huggingface_hub import constants
    from lelab import cache

    legacy = tmp_path / 'archive'
    revision = 'a' * 40
    repo = legacy / 'models--owner--policy'
    directory = repo / 'snapshots' / revision
    directory.mkdir(parents=True)
    (repo / 'refs').mkdir()
    (repo / 'refs/main').write_text(revision)
    for name in ['config.json', 'policy_preprocessor.json', 'policy_postprocessor.json']:
        (directory / name).write_text(json.dumps({}))
    save_file({'weight': np.ones(2, dtype=np.float32)}, directory / 'model.safetensors')
    monkeypatch.setattr(cache, 'LEGACY_HUB', legacy)
    monkeypatch.setattr(constants, 'HF_HUB_OFFLINE', False)
    monkeypatch.setattr(huggingface_hub, 'HfApi', lambda: SimpleNamespace(
        model_info=lambda repo_id: SimpleNamespace(sha=revision)))
    return cache, directory


def test_complete_current_legacy_model_is_reused_without_writes(tmp_path, monkeypatch):
    cache, directory = _legacy_fixture(tmp_path, monkeypatch)
    before = {p: p.stat().st_mtime_ns for p in tmp_path.rglob('*')}
    assert cache.legacy_policy_snapshot('owner/policy') == str(directory)
    assert before == {p: p.stat().st_mtime_ns for p in tmp_path.rglob('*')}


def test_new_revision_does_not_silently_load_an_old_model(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import huggingface_hub
    cache, _ = _legacy_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(huggingface_hub, 'HfApi', lambda: SimpleNamespace(
        model_info=lambda repo_id: SimpleNamespace(sha='b' * 40)))
    assert cache.legacy_policy_snapshot('owner/policy') is None


def test_incomplete_legacy_weights_are_not_used(tmp_path, monkeypatch):
    cache, directory = _legacy_fixture(tmp_path, monkeypatch)
    (directory / 'model.safetensors').write_bytes(b'incomplete')
    assert cache.legacy_policy_snapshot('owner/policy') is None
