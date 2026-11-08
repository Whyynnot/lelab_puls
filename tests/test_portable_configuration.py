import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_no_hardware_configuration_does_not_assume_private_devices(monkeypatch, tmp_path):
    import prepare_hardware as hardware
    monkeypatch.delenv('LELAB_HARDWARE_CONFIG', raising=False)
    monkeypatch.delenv('LELAB_USB_RECOVERY_SCRIPT', raising=False)
    monkeypatch.setattr(hardware, 'DEVICES', hardware.load_devices())
    def forbidden(*args, **kwargs):
        raise AssertionError('Unconfigured hardware must not invoke Windows/USB tools')
    monkeypatch.setattr(hardware.subprocess, 'run', forbidden)
    assert hardware.discover() == {}
    report = hardware.prepare(report_path=tmp_path / 'report.json')
    assert report['device_mapping_ready']
    assert not report['config_updated']
    assert hardware.mapped_record({'leader_port': 'chosen-port'}, {})['leader_port'] == 'chosen-port'


def test_partial_hardware_configuration_preserves_other_roles(monkeypatch, tmp_path):
    import prepare_hardware as hardware
    config = tmp_path / 'hardware.json'
    config.write_text(json.dumps({'follower': '/dev/serial/by-id/usb-TEST-if00'}))
    monkeypatch.setenv('LELAB_HARDWARE_CONFIG', str(config))
    monkeypatch.setattr(hardware, 'DEVICES', hardware.load_devices())
    updated = hardware.mapped_record({'leader_port': 'user-choice'}, hardware.discover())
    assert updated['leader_port'] == 'user-choice'
    assert updated['follower_port'] == '/dev/serial/by-id/usb-TEST-if00'


@pytest.mark.parametrize('value', [{'unknown': '/dev/null'}, {'follower': 'relative'}, []])
def test_invalid_hardware_configuration_fails_clearly(monkeypatch, tmp_path, value):
    import prepare_hardware as hardware
    config = tmp_path / 'hardware.json'
    config.write_text(json.dumps(value))
    monkeypatch.setenv('LELAB_HARDWARE_CONFIG', str(config))
    with pytest.raises(ValueError):
        hardware.load_devices()


def test_native_camera_inventory_without_role_mapping(monkeypatch, tmp_path):
    import camera_inventory
    monkeypatch.setattr(camera_inventory, 'DEVICES', {})
    sys_root = tmp_path / 'sys'
    device = sys_root / 'video4'
    device.mkdir(parents=True)
    (device / 'index').write_text('0')
    (device / 'name').write_text('Generic camera')
    rows = camera_inventory.cameras(sys_root, tmp_path / 'missing-by-id')
    assert rows[0]['deviceId'] == 'wsl-camera:4'
    assert rows[0]['role'] is None
