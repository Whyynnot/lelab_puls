from contextlib import contextmanager
from types import SimpleNamespace as NS

import pytest
from native_startup import install_startup_retries
from lerobot.robots.so_follower.so_follower import SOFollower


class Bus:
    motors = {'shoulder_lift': object(), 'gripper': object()}
    def __init__(self):
        self.calls=[]
        self.fail=False
    def enable_torque(self,motors=None,num_retry=0): self.calls.append(('enable',num_retry))
    def disable_torque(self,motors=None,num_retry=0): self.calls.append(('disable',num_retry))
    @contextmanager
    def torque_disabled(self):
        self.disable_torque()
        try: yield
        finally: self.enable_torque()
    def configure_motors(self):
        if self.fail: raise ConnectionError('test-only persistent communication failure')
    def set_operating_mode(self,*args): pass
    def write(self,*args): pass


@pytest.mark.parametrize('fail',[False,True])
def test_retries_are_bounded_native_arguments_and_restored_after_setup(monkeypatch,fail):
    monkeypatch.setattr(SOFollower,'configure',SOFollower.configure)
    install_startup_retries()
    configure=SOFollower.configure
    install_startup_retries()
    assert configure is SOFollower.configure, 'Repeated install must not stack wrappers'
    bus=Bus();bus.fail=fail
    original_enable=bus.enable_torque
    robot=NS(bus=bus,config=NS(position_p_coefficient=16,position_i_coefficient=0,position_d_coefficient=8))
    if fail:
        with pytest.raises(ConnectionError,match='persistent'):
            SOFollower.configure(robot)
    else:
        SOFollower.configure(robot)
    assert bus.calls==[('disable',2),('enable',2)]
    assert bus.enable_torque==original_enable
    bus.enable_torque()
    assert bus.calls[-1]==('enable',0), 'Ordinary calls retain original retry behavior'


def test_checksum_error_keeps_real_motor_detail_with_hint():
    from lelab.rollout import _friendly_hint
    error="ConnectionError: Failed to enable torque after 3 tries on 'shoulder_lift' (motor ID 2): Checksum error."
    assert '串口通信包校验失败' in _friendly_hint(error)
    assert _friendly_hint('ValueError: model checkpoint checksum mismatch') is None


def test_web_entrypoint_installs_retries_for_in_process_device_setup():
    import app
    assert app.teleoperate is __import__('lelab.teleoperate', fromlist=['handle_start_teleoperation'])
    assert getattr(SOFollower.configure, '_native_startup_retries', False)
