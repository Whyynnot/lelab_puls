"""Reject cross-model preprocessed samples without touching CUDA or devices."""
import json
import pytest
from adapter import Profile, read_sample_manifest


def save_manifest(tmp_path, profile):
    (tmp_path/'manifest.json').write_text(json.dumps({'artifact':profile.artifact, 'profile':profile.model_dump()}))


def test_profile_specific_sample_manifest_and_legacy_default(tmp_path, monkeypatch):
    import adapter
    monkeypatch.setattr(adapter, 'ROOT', tmp_path)
    profile = Profile(name='new', artifact=str(tmp_path/'new_model'), sample_manifest='manifest.json')
    save_manifest(tmp_path, profile)
    assert read_sample_manifest(profile)['artifact'] == profile.artifact
    assert Profile(name='old', artifact='old').sample_manifest == 'validation/sample_manifest.json'


def test_reject_samples_preprocessed_with_another_checkpoint(tmp_path, monkeypatch):
    import adapter
    monkeypatch.setattr(adapter, 'ROOT', tmp_path)
    profile = Profile(name='new', artifact=str(tmp_path/'new_model'), sample_manifest='manifest.json')
    save_manifest(tmp_path, profile.model_copy(update={'artifact':str(tmp_path/'old_model')}))
    with pytest.raises(ValueError, match='another model'):
        read_sample_manifest(profile)


def test_reject_changed_camera_mapping(tmp_path, monkeypatch):
    import adapter
    monkeypatch.setattr(adapter, 'ROOT', tmp_path)
    profile = Profile(name='new', artifact=str(tmp_path/'new_model'), sample_manifest='manifest.json')
    swapped = {'head':'observation.images.wrist', 'wrist':'observation.images.head'}
    save_manifest(tmp_path, profile.model_copy(update={'camera_map':swapped}))
    with pytest.raises(ValueError, match='camera_map'):
        read_sample_manifest(profile)
