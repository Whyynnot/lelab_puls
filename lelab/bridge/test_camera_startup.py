import logging
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from robot_runner import build_camera_config, disconnect_robot


def test_camera_defaults_to_mjpg_and_keeps_requested_frame_rate():
    settings={'type':'opencv','camera_index':2,'width':640,'height':480,'fps':30,'rotation':0}
    config=build_camera_config(settings)
    assert config.fourcc=='MJPG' and config.fps==30 and config.index_or_path==2
    assert 'fourcc' not in settings


def test_explicit_capture_format_and_rotation_are_preserved():
    config=build_camera_config({'camera_index':2,'width':640,'height':480,'fps':25,
                                'fourcc':'YUYV','rotation':90})
    assert config.fourcc=='YUYV' and config.fps==25
    assert (config.width,config.height)==(480,640)


def test_partial_connect_closes_open_camera_and_bus_without_calling_disconnected_objects(caplog):
    head,wrist,bus=Mock(),Mock(),Mock()
    head.is_connected=True;wrist.is_connected=False;bus.is_connected=True
    device=SimpleNamespace(is_connected=False,cameras={'head':head,'wrist':wrist},bus=bus,
                           config=SimpleNamespace(disable_torque_on_disconnect=True),disconnect=Mock())
    disconnect_robot(device,logging.getLogger(__name__))
    head.disconnect.assert_called_once()
    wrist.disconnect.assert_not_called()
    bus.disconnect.assert_called_once_with(disable_torque=True)
    device.disconnect.assert_not_called()
    assert not caplog.records


def test_cleanup_keeps_closing_bus_when_a_camera_disconnect_fails(caplog):
    camera,bus=Mock(),Mock();camera.is_connected=True;bus.is_connected=True
    camera.disconnect.side_effect=RuntimeError('camera failure')
    device=SimpleNamespace(is_connected=False,cameras={'head':camera},bus=bus,
                           config=SimpleNamespace(disable_torque_on_disconnect=False))
    disconnect_robot(device,logging.getLogger(__name__))
    bus.disconnect.assert_called_once_with(disable_torque=False)
    assert 'camera failure' in caplog.text


def test_blocked_camera_cannot_block_bus_or_other_camera():
    release=threading.Event()
    head,wrist,bus=Mock(),Mock(),Mock()
    head.is_connected=wrist.is_connected=bus.is_connected=True
    head.disconnect.side_effect=lambda:release.wait(2)
    robot=SimpleNamespace(cameras={'head':head,'wrist':wrist},bus=bus,
                          config=SimpleNamespace(disable_torque_on_disconnect=True))
    started=time.monotonic()
    try:
        assert not disconnect_robot(robot,logging.getLogger(__name__),preserve_torque=True,timeout=.05)
        assert time.monotonic()-started < .5
        bus.disconnect.assert_called_once_with(disable_torque=False)
        wrist.disconnect.assert_called_once()
    finally:
        release.set()


def test_already_closed_resources_do_not_produce_cleanup_warnings(caplog):
    camera,bus=Mock(),Mock();camera.is_connected=False;bus.is_connected=False
    device=SimpleNamespace(is_connected=False,cameras={'wrist':camera},bus=bus)
    disconnect_robot(device,logging.getLogger(__name__))
    camera.disconnect.assert_not_called();bus.disconnect.assert_not_called()
    assert not caplog.records


@pytest.mark.parametrize('camera_failed', [False, True])
def test_fault_cleanup_preserves_torque_and_closes_other_resources(camera_failed):
    head, wrist, bus = Mock(), Mock(), Mock()
    head.is_connected = wrist.is_connected = bus.is_connected = True
    if camera_failed:
        wrist.disconnect.side_effect = RuntimeError('USB disconnected')
    robot = SimpleNamespace(is_connected=True, cameras={'head': head, 'wrist': wrist},
                            bus=bus, config=SimpleNamespace(disable_torque_on_disconnect=True),
                            disconnect=Mock())
    disconnect_robot(robot, logging.getLogger(__name__), preserve_torque=True)
    robot.disconnect.assert_not_called()
    head.disconnect.assert_called_once()
    wrist.disconnect.assert_called_once()
    bus.disconnect.assert_called_once_with(disable_torque=False)
