# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for lelab.teleoperate — request schema and status handlers."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolated_teleoperation_state(monkeypatch):
    from lelab import record, rollout, teleoperate
    monkeypatch.setattr(record, 'recording_active', False)
    monkeypatch.setattr(rollout, 'inference_active', False)
    for name, value in [('teleoperation_active', False), ('teleoperation_thread', None),
                        ('current_robot', None), ('current_teleop', None), ('teleoperation_error', None)]:
        monkeypatch.setattr(teleoperate, name, value)


def test_teleoperate_request_rejects_missing_fields() -> None:
    from pydantic import ValidationError

    from lelab.teleoperate import TeleoperateRequest

    with pytest.raises(ValidationError):
        TeleoperateRequest()


def test_handle_teleoperation_status_returns_dict() -> None:
    from lelab.teleoperate import handle_teleoperation_status

    result = handle_teleoperation_status()
    assert isinstance(result, dict)


def test_handle_get_joint_positions_returns_dict_when_idle() -> None:
    from lelab.teleoperate import handle_get_joint_positions

    result = handle_get_joint_positions()
    assert isinstance(result, dict)


def test_get_joint_positions_from_robot_uses_provided_object() -> None:
    from lelab.teleoperate import get_joint_positions_from_robot
    from tests.mocks import FakeRobot

    robot = FakeRobot()
    robot.connect()
    positions = get_joint_positions_from_robot(robot)
    assert isinstance(positions, dict)


def test_get_joint_positions_maps_degrees_to_radians_without_correction() -> None:
    """Every joint maps its Present_Position (degrees around the calibration
    center) straight to radians, deg * pi/180, with no per-joint offset.
    shoulder_lift/elbow_flex used to be run through a correction table; a
    calibration that would have triggered it is supplied here to prove the
    offset is gone and they now map like every other joint."""
    import math

    from lelab.teleoperate import get_joint_positions_from_robot

    class _Cal:
        # The removed correction derived its offset from range_min/range_max.
        range_min = 0
        range_max = 4095

    class _Robot:
        calibration = {"shoulder_lift": _Cal(), "elbow_flex": _Cal()}

        def get_observation(self):
            return {
                "shoulder_pan.pos": 0.0,
                "shoulder_lift.pos": 90.0,
                "elbow_flex.pos": -45.0,
                "wrist_flex.pos": 30.0,
                "wrist_roll.pos": 12.0,
                "gripper.pos": 50.0,
            }

    positions = get_joint_positions_from_robot(_Robot())

    assert positions["Rotation"] == pytest.approx(0.0)
    assert positions["Pitch"] == pytest.approx(math.radians(90.0))
    assert positions["Elbow"] == pytest.approx(math.radians(-45.0))
    assert positions["Wrist_Pitch"] == pytest.approx(math.radians(30.0))
    assert positions["Wrist_Roll"] == pytest.approx(math.radians(12.0))
    assert positions["Jaw"] == pytest.approx(math.radians(50.0))


def test_start_teleoperation_reports_connection_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A device that fails to connect must make the start handler return
    success=False (so the UI surfaces the error and doesn't navigate to an
    empty teleop screen) and reset state so a retry isn't blocked. Previously
    the connect ran in a worker thread and the handler always claimed success.
    """
    import lelab.teleoperate as teleop

    monkeypatch.setattr(teleop, "teleoperation_active", False)
    monkeypatch.setattr(teleop, "setup_calibration_files", lambda leader, follower: ("leader", "follower"))

    class _Bus:
        def connect(self) -> None:
            raise RuntimeError("serial port unavailable")

    class _Device:
        def __init__(self, config) -> None:
            self.bus = _Bus()
            self.cameras: dict = {}
            self.disconnected = False

        def disconnect(self) -> None:
            self.disconnected = True

    monkeypatch.setattr(teleop, "SO101Follower", _Device)
    monkeypatch.setattr(teleop, "SO101Leader", _Device)

    request = teleop.TeleoperateRequest(
        leader_port="COM_LEADER",
        follower_port="COM_FOLLOWER",
        leader_config="leader",
        follower_config="follower",
    )
    result = teleop.handle_start_teleoperation(request)

    assert result["success"] is False
    # The message must name the arm that failed (the follower connects first).
    assert "follower" in result["message"].lower()
    assert "COM_FOLLOWER" in result["message"]
    # State must be reset so the next attempt isn't blocked by the mutex.
    assert teleop.teleoperation_active is False


def test_start_teleoperation_disconnects_follower_when_leader_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The partial-connect path: if the follower connects but the leader then
    fails, the follower must be disconnected so its serial port is released.
    """
    import lelab.teleoperate as teleop

    monkeypatch.setattr(teleop, "teleoperation_active", False)
    monkeypatch.setattr(teleop, "setup_calibration_files", lambda leader, follower: ("leader", "follower"))

    class _OkBus:
        closed_without_torque = False

        def connect(self) -> None:
            pass

        def disconnect(self, disable_torque=True) -> None:
            self.closed_without_torque = not disable_torque

    class _FailingBus:
        def connect(self) -> None:
            raise RuntimeError("leader offline")

    class _Follower:
        def __init__(self, config) -> None:
            self.bus = _OkBus()
            self.cameras: dict = {}
            self.disconnected = False

        def disconnect(self) -> None:
            self.disconnected = True

    class _Leader:
        def __init__(self, config) -> None:
            self.bus = _FailingBus()
            self.disconnected = False

        def disconnect(self) -> None:
            self.disconnected = True

    created: dict = {}
    monkeypatch.setattr(
        teleop, "SO101Follower", lambda config: created.setdefault("follower", _Follower(config))
    )
    monkeypatch.setattr(teleop, "SO101Leader", lambda config: created.setdefault("leader", _Leader(config)))

    request = teleop.TeleoperateRequest(
        leader_port="COM_LEADER",
        follower_port="COM_FOLLOWER",
        leader_config="leader",
        follower_config="follower",
    )
    result = teleop.handle_start_teleoperation(request)

    assert result["success"] is False
    assert "leader" in result["message"].lower()
    # The already-connected follower must have been cleaned up.
    assert created["follower"].bus.closed_without_torque is True
    assert created["follower"].disconnected is False
    assert teleop.teleoperation_active is False


@pytest.mark.parametrize('failure', ['leader_read', 'follower_write', 'follower_read', None])
def test_worker_fault_stops_targets_and_preserves_follower_torque(monkeypatch, failure):
    from types import SimpleNamespace
    import lelab.teleoperate as teleop

    class Bus:
        def __init__(self):
            self.disconnect_calls = []
        def connect(self):
            pass
        def write_calibration(self, calibration):
            pass
        def disconnect(self, disable_torque=True):
            self.disconnect_calls.append(disable_torque)

    class Device:
        def __init__(self, config):
            self.config = config
            self.bus = Bus()
            self.calibration = {}
            self.cameras = {}
            self.actions = []
            self.reads = 0
        def configure(self):
            pass
        def disconnect(self):
            self.bus.disconnect()
        def get_action(self):
            self.reads += 1
            if failure == 'leader_read' and self.reads > 1:
                raise ConnectionError('leader USB lost')
            if failure is None and self.reads > 1:
                teleop.teleoperation_active = False
            return {'shoulder_pan.pos': 0.0}
        def send_action(self, action):
            self.actions.append(action)
            if failure == 'follower_write':
                raise ConnectionError('follower write failed')
        def get_observation(self):
            if failure == 'follower_read':
                raise ConnectionError('follower feedback lost')
            return {name + '.pos': 0.0 for name in
                    ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']}

    created = {}
    monkeypatch.setattr(teleop, 'setup_calibration_files', lambda *args: ('leader', 'follower'))
    monkeypatch.setattr(teleop, 'SO101Follower', lambda config: created.setdefault('follower', Device(config)))
    monkeypatch.setattr(teleop, 'SO101Leader', lambda config: created.setdefault('leader', Device(config)))
    monkeypatch.setattr(teleop.threading, 'Thread', lambda *, target, **kw: SimpleNamespace(start=target))
    monkeypatch.setattr(teleop.time, 'sleep', lambda seconds: None)

    result = teleop.handle_start_teleoperation(teleop.TeleoperateRequest(
        leader_port='leader', follower_port='follower', leader_config='leader', follower_config='follower'))
    assert result['success'] is True
    assert teleop.teleoperation_active is False
    assert teleop.current_robot is None
    assert created['follower'].bus.disconnect_calls == [failure is None]
    status = teleop.handle_teleoperation_status()
    if failure:
        assert len(created['follower'].actions) == 1
        assert status['last_error']
        assert '未主动关闭从臂扭矩' in status['fault_message']
        # Leaving the page after a fault must not later release the follower.
        teleop.handle_stop_teleoperation()
        assert created['follower'].bus.disconnect_calls == [False]
    else:
        assert status['last_error'] is None
        assert status['fault_message'] is None
