"""Exercise LeLab homing with the installed LeRobot bus and no hardware I/O."""
import json

import pytest


@pytest.mark.parametrize('side', ['leader', 'follower'])
def test_so101_homing_and_save_with_current_lerobot(side, monkeypatch, tmp_path):
    from lelab.calibrate import CalibrationManager

    if side == 'leader':
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
        device = SO101Leader(SO101LeaderConfig(port='/dev/null', id='offline_compat_test'))
    else:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        device = SO101Follower(SO101FollowerConfig(port='/dev/null', id='offline_compat_test'))

    bus = device.bus
    # Keep real model metadata and real set_half_turn_homings/_get_half_turn_homings.
    # Replace every operation which could touch the serial device.
    positions = dict(zip(bus.motors, [0, 4095, 2047, 1000, 3000, 2200], strict=True))
    writes = []
    freed = []
    monkeypatch.setattr(bus, 'disable_torque', lambda *a, **k: freed.append(True))
    monkeypatch.setattr(bus, 'write', lambda register, motor, value, **kwargs: writes.append((register, motor, value)))
    monkeypatch.setattr(bus, 'sync_read', lambda *a, **k: positions.copy())
    device.calibration_fpath = tmp_path / f'{side}.json'

    manager = CalibrationManager()
    manager.device = device
    manager._step_homing()

    # STS offsets subtract from raw position; the +2048 edge clamps to +2047.
    expected = {motor: max(-2047, min(2047, raw - 2047)) for motor, raw in positions.items()}
    assert manager._homing_offsets == expected
    assert freed == [True]
    actual = {motor: value for register, motor, value in writes if register == 'Homing_Offset'}
    assert actual == expected
    assert not any(register in {'Goal_Position', 'Torque_Enable'} for register, _, _ in writes)

    manager._mins = dict.fromkeys(bus.motors, 100)
    manager._maxes = dict.fromkeys(bus.motors, 3900)
    manager._complete_calibration()
    saved = json.loads(device.calibration_fpath.read_text())
    assert set(saved) == set(bus.motors)
    for motor, calibration in saved.items():
        assert calibration['homing_offset'] == expected[motor]
        assert calibration['range_min'] == 100
        assert calibration['range_max'] == 3900
