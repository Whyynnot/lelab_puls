from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from lerobot.rollout.robot_wrapper import ThreadSafeRobot
from lerobot.rollout.strategies.core import RolloutStrategy
from lerobot.rollout.strategies.base import BaseStrategy
from lerobot.rollout.configs import BaseStrategyConfig
from native_faults import install_fault_cleanup


@pytest.fixture
def guarded(monkeypatch):
    monkeypatch.setattr(ThreadSafeRobot, 'get_observation', ThreadSafeRobot.get_observation)
    normal = Mock()
    monkeypatch.setattr(RolloutStrategy, '_teardown_hardware', normal)
    install_fault_cleanup()
    installed = ThreadSafeRobot.get_observation
    install_fault_cleanup()
    assert ThreadSafeRobot.get_observation is installed
    robot = NS(bus=NS(disconnect=Mock(), port_handler=None),
               cameras={'head': NS(is_connected=True, disconnect=Mock()),
                        'wrist': NS(is_connected=True, disconnect=Mock())},
               disconnect=Mock(), get_observation=Mock(), send_action=Mock(),
               action_features={'shoulder_pan.pos': float})
    wrapper = ThreadSafeRobot(robot)
    hardware = NS(robot_wrapper=wrapper, teleop=None, initial_position={'shoulder_pan.pos': 0})
    strategy = BaseStrategy(BaseStrategyConfig())
    strategy._engine = NS(failed=False, stop=Mock())
    return strategy, hardware, robot, normal


@pytest.mark.parametrize('error', [ConnectionError('motor 1: Operation timed out'),
                                  RuntimeError('wrist camera read thread is not running')])
def test_read_fault_prevents_hold_homing_and_torque_disable(guarded, error):
    strategy, hw, robot, normal = guarded
    robot.get_observation.side_effect = error
    with pytest.raises(type(error), match=str(error)):
        hw.robot_wrapper.get_observation()
    assert 'get_observation' in hw.robot_wrapper.hardware_failure
    hw.robot_wrapper._position_hold_enabled = True
    strategy.hold_control_state(hw)
    strategy._teardown_hardware(hw)
    normal.assert_not_called()
    robot.send_action.assert_not_called()
    robot.disconnect.assert_not_called()
    robot.bus.disconnect.assert_called_once_with(disable_torque=False)
    for camera in robot.cameras.values():
        camera.disconnect.assert_called_once()
    strategy._engine.stop.assert_called_once()


def test_camera_cleanup_still_runs_when_bus_close_raises(guarded):
    strategy, hw, robot, normal = guarded
    hw._lelab_failure = 'ConnectionError: original fault'
    robot.bus.disconnect.side_effect = RuntimeError('port disappeared')
    strategy._teardown_hardware(hw)
    for camera in robot.cameras.values():
        camera.disconnect.assert_called_once()
    normal.assert_not_called()


def test_normal_shutdown_preserves_native_behavior(guarded):
    strategy, hw, robot, normal = guarded
    strategy._teardown_hardware(hw, False)
    normal.assert_called_once_with(strategy, hw, False)
    robot.bus.disconnect.assert_not_called()


def test_partial_setup_failure_releases_resources_without_motion(guarded):
    strategy, hw, robot, normal = guarded
    try:
        raise RuntimeError('camera setup failed')
    except RuntimeError:
        strategy._teardown_hardware(hw)
    normal.assert_not_called()
    robot.bus.disconnect.assert_called_once_with(disable_torque=False)


def test_normal_disconnect_failure_still_releases_cameras(guarded):
    strategy, hw, robot, normal = guarded
    normal.side_effect = ConnectionError('disconnect failed')
    with pytest.raises(ConnectionError, match='disconnect failed'):
        strategy._teardown_hardware(hw)
    robot.bus.disconnect.assert_called_once_with(disable_torque=False)
    for camera in robot.cameras.values():
        camera.disconnect.assert_called_once()
