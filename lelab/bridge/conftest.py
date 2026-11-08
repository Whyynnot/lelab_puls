"""Synthetic USB identities for tests; no devices are opened."""
import pytest


@pytest.fixture(autouse=True)
def synthetic_device_identities(monkeypatch):
    import prepare_hardware
    for role in prepare_hardware.ROLES:
        category = 'serial' if role in ('leader', 'follower') else 'v4l'
        suffix = '-if00' if category == 'serial' else '-video-index0'
        monkeypatch.setitem(prepare_hardware.DEVICES, role, f'/dev/{category}/by-id/usb-TEST_{role}{suffix}')
