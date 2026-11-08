import dataclasses
import json
from types import SimpleNamespace
import pytest
from lerobot.motors import MotorCalibration
from lerobot.motors.motors_bus import SerialMotorsBus
from calibration_guard import verify_follower_calibration


class ReadOnlyBus:
    _same_calibration=SerialMotorsBus._same_calibration
    def __init__(self,cached,actual):
        self.calibration=cached;self.actual=actual;self.is_connected=False;self.closed=[]
    def _has_register(self,motor,name):return name!='Drive_Mode'
    def connect(self):self.is_connected=True
    def read_calibration(self):return self.actual
    def disconnect(self,*,disable_torque):
        assert disable_torque is False
        self.closed.append(disable_torque);self.is_connected=False


def calibration(offset=10,drive=0):
    return {'joint':MotorCalibration(id=1,drive_mode=drive,homing_offset=offset,range_min=100,range_max=3900)}


def test_matching_values_close_bus_without_torque_or_camera_access(tmp_path):
    saved=calibration();bus=ReadOnlyBus(saved,calibration())
    result=verify_follower_calibration(SimpleNamespace(bus=bus,calibration=saved),'correct.json',tmp_path)
    assert result['matches'] and result['motors_checked']==1
    assert bus.closed==[False] and not bus.is_connected and bus.calibration is saved


def test_mismatch_suggests_correct_file_and_restores_cached_values(tmp_path):
    wrong=calibration(20);actual=calibration();bus=ReadOnlyBus(wrong,actual)
    (tmp_path/'correct.json').write_text(json.dumps({k:dataclasses.asdict(v) for k,v in actual.items()}))
    (tmp_path/'broken.json').write_text('{')
    with pytest.raises(ValueError,match='correct.json'):
        verify_follower_calibration(SimpleNamespace(bus=bus,calibration=wrong),'wrong.json',tmp_path)
    assert bus.closed==[False] and bus.calibration is wrong


def test_no_matching_file_does_not_rewrite_motor_or_guess(tmp_path):
    saved=calibration(20);bus=ReadOnlyBus(saved,calibration())
    with pytest.raises(ValueError,match='本地没有'):
        verify_follower_calibration(SimpleNamespace(bus=bus,calibration=saved),'wrong.json',tmp_path)
    assert bus.closed==[False] and bus.calibration is saved


def test_read_failure_still_closes_without_changing_torque(tmp_path):
    saved=calibration();bus=ReadOnlyBus(saved,calibration())
    def fail():raise ConnectionError('read failed')
    bus.read_calibration=fail
    with pytest.raises(ConnectionError,match='read failed'):
        verify_follower_calibration(SimpleNamespace(bus=bus,calibration=saved),'correct.json',tmp_path)
    assert bus.closed==[False] and bus.calibration is saved


def test_driver_unsupported_drive_mode_difference_is_not_false_mismatch(tmp_path):
    saved=calibration(drive=1);bus=ReadOnlyBus(saved,calibration(drive=0))
    result=verify_follower_calibration(SimpleNamespace(bus=bus,calibration=saved),'correct.json',tmp_path)
    assert result['matches'] and bus.closed==[False]
