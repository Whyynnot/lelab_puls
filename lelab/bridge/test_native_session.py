"""Regression tests using the real native strategy/controller and simulated I/O."""
import json
import io
import os
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch, Mock

import numpy as np
import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lerobot.inference.sync import SyncInferenceEngine
from lerobot.rollout.configs import BaseStrategyConfig
from lerobot.rollout.controller import LinkedEvent
from lerobot.rollout.robot_wrapper import ThreadSafeRobot
from lerobot.rollout.strategies.base import BaseStrategy
from native_session import NativeMonitor, WebSession, atomic_json, install_observers, read_json

KEYS = [f'{name}.pos' for name in ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']]
FEATURES = {'observation.state': {'dtype': 'float32', 'shape': (6,), 'names': KEYS},
            'observation.images.head': {'dtype': 'image', 'shape': (48, 64, 3)},
            'observation.images.wrist': {'dtype': 'image', 'shape': (48, 64, 3)}}


def test_native_controller_fault_closes_all_io_without_extra_motion(tmp_path, monkeypatch):
    from lerobot.rollout.strategies.core import RolloutStrategy
    from native_faults import install_fault_cleanup
    monkeypatch.setattr(ThreadSafeRobot, 'get_observation', ThreadSafeRobot.get_observation)
    monkeypatch.setattr(RolloutStrategy, '_teardown_hardware', RolloutStrategy._teardown_hardware)
    install_fault_cleanup()
    monkeypatch.setenv('LELAB_NATIVE_SESSION_DIR', str(tmp_path / ('b' * 32)))
    ctx, strategy, robot, _ = context()
    ctx.hardware.teleop = None
    ctx.runtime.cfg.return_to_initial_position = True
    robot.bus = NS(disconnect=Mock(), port_handler=None)
    robot.cameras = {'head': NS(is_connected=True, disconnect=Mock()),
                    'wrist': NS(is_connected=True, disconnect=Mock())}
    robot.disconnect = Mock()
    original_read = robot.get_observation
    fault_count = []
    def fail_after_motion():
        if len(robot.sent) >= 3:
            fault_count.append(len(robot.sent))
            raise ConnectionError('test USB read failure')
        return original_read()
    robot.get_observation = fail_after_motion
    session = WebSession(strategy, ctx)
    thread = threading.Thread(target=session.run)
    thread.start()
    try:
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert session.controller.failed
        assert 'test USB read failure' in ctx.hardware._lelab_failure
        strategy.teardown(ctx)
        assert len(robot.sent) == fault_count[0]
        robot.disconnect.assert_not_called()
        robot.bus.disconnect.assert_called_once_with(disable_torque=False)
        for camera in robot.cameras.values():
            camera.disconnect.assert_called_once()
    finally:
        ctx.runtime.shutdown_event.parent.set()
        thread.join(timeout=5)
        ctx.policy.inference.stop()


@pytest.fixture(autouse=True)
def isolate_worker_observer(monkeypatch):
    from lerobot.inference import rtc
    monkeypatch.setattr(rtc, 'prepare_observation_for_inference', rtc.prepare_observation_for_inference)


def wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError('Timed out waiting for native session state')


class Processor:
    def __call__(self, value):
        return value
    def reset(self):
        pass


class ToyPolicy:
    config = NS(use_amp=False)
    def __init__(self):
        self.queue = deque()
        self.calls = 0
    def reset(self):
        self.queue.clear()
    def drop_queued_actions(self):
        self.queue.clear()
    def predict_action_chunk(self, observation):
        self.calls += 1
        task = observation['task']
        return torch.tensor([[[float(i + (20 if task == 'new instruction' else 1))] * 6
                              for i in range(4)]])
    def select_action(self, observation):
        if not self.queue:
            self.queue.extend(self.predict_action_chunk(observation)[0])
        return self.queue.popleft().unsqueeze(0)


class FakeRobot:
    name = 'simulated'
    robot_type = 'so101_follower'
    supports_position_hold = True
    supports_command_hold = False
    is_connected = True
    action_features = dict.fromkeys(KEYS, float)
    observation_features = action_features
    def __init__(self):
        self.pos = dict.fromkeys(KEYS, 0.0)
        self.sent = []
        self.reads = 0
        self.image = np.zeros((48, 64, 3), dtype=np.uint8)
    def get_observation(self):
        self.reads += 1
        return {**self.pos, 'head': self.image, 'wrist': self.image}
    def send_action(self, action):
        self.sent.append(dict(action))
        self.pos = dict(action)
        return dict(action)


def context(duration=0):
    robot = FakeRobot()
    wrapper = ThreadSafeRobot(robot)
    wrapper.get_observation()
    wrapper.configure_hold()
    policy = ToyPolicy()
    engine = SyncInferenceEngine(policy, Processor(), Processor(), FEATURES, KEYS,
                                 'original instruction', 'cpu', 'so101_follower')
    cfg = NS(fps=100, duration=duration, interpolation_multiplier=1, use_torch_compile=False,
             display_data=False, autosteer_interval_s=10, interactive=True, status_interval_s=0, device='cpu')
    ctx = NS(runtime=NS(cfg=cfg, shutdown_event=LinkedEvent(threading.Event()), cadence_report=None),
             hardware=NS(robot_wrapper=wrapper, initial_position=dict(robot.pos)),
             policy=NS(inference=engine),
             processors=NS(robot_observation_processor=lambda obs: obs,
                           robot_action_processor=lambda value: value[0]),
             data=NS(ordered_action_keys=KEYS, dataset_features=FEATURES))
    strategy = BaseStrategy(BaseStrategyConfig())
    strategy.setup(ctx)
    return ctx, strategy, robot, policy


@pytest.fixture
def session(tmp_path, monkeypatch):
    directory = tmp_path / ('a' * 32)
    monkeypatch.setenv('LELAB_NATIVE_SESSION_DIR', str(directory))
    ctx, strategy, robot, policy = context()
    # Keep native return interpolation; accelerate only sleeps in this test.
    with patch('lerobot.rollout.strategies.core.precise_sleep', lambda _: time.sleep(0.0001)):
        session = WebSession(strategy, ctx)
        thread = threading.Thread(target=session.run)
        thread.start()
        try:
            wait_for(lambda: session.phase == 'running' and len(robot.sent) >= 2)
            yield session, ctx, robot, policy
        finally:
            ctx.runtime.shutdown_event.parent.set()
            thread.join(5)
            assert not thread.is_alive()


def test_reset_pause_resume_and_prompt_retention(session):
    web, ctx, robot, policy = session
    initial = dict(ctx.hardware.initial_position)
    web.command({'session_id': web.directory.name, 'action': 'reset'})
    wait_for(lambda: web.phase == 'paused')
    assert robot.pos == initial
    assert web.controller.task == 'original instruction'
    sends = len(robot.sent)
    time.sleep(0.12)
    assert len(robot.sent) == sends, 'No policy dispatch while paused'
    web.command({'session_id': web.directory.name, 'action': 'resume', 'task': 'new instruction'})
    wait_for(lambda: web.phase == 'running' and len(robot.sent) > sends)
    assert all(value >= 20 for value in robot.sent[sends].values()), 'Old ACT chunk was dropped'
    web.command({'session_id': web.directory.name, 'action': 'reset'})
    wait_for(lambda: web.phase == 'paused')
    assert web.controller.task == 'new instruction', 'Second reset retains latest prompt'
    assert ctx.hardware.initial_position == initial
    assert robot.pos == initial, 'Initial pose never recaptured on resume'


def test_stale_and_early_resume_rejected(session):
    web, *_ = session
    with pytest.raises(ValueError, match='Stale'):
        web.command({'session_id': 'b'*32, 'action': 'reset'})
    with pytest.raises(ValueError, match='Cannot resume'):
        web.command({'session_id': web.directory.name, 'action': 'resume'})


def test_reset_failure_does_not_enable_resume(session):
    web, ctx, robot, _ = session
    ctx.hardware.initial_position = None
    web.command({'session_id': web.directory.name, 'action': 'reset'})
    wait_for(lambda: web.phase == 'error')
    with pytest.raises(ValueError, match='Cannot resume'):
        web.command({'session_id': web.directory.name, 'action': 'resume'})


def test_reset_final_observation_failure_keeps_controller_waiting(session):
    web, ctx, robot, _ = session
    original = ctx.hardware.robot_wrapper.get_observation
    def fail_final_read():
        if web.phase == 'resetting' and robot.pos == ctx.hardware.initial_position:
            raise RuntimeError('test-only final camera read failure')
        return original()
    ctx.hardware.robot_wrapper.get_observation = fail_final_read
    web.command({'session_id': web.directory.name, 'action': 'reset'})
    wait_for(lambda: web.phase == 'error')
    assert not web.controller.stopped
    with pytest.raises(ValueError, match='Cannot resume'):
        web.command({'session_id': web.directory.name, 'action': 'resume'})


def test_stop_during_reset_cancels_remaining_commands(session):
    web, ctx, robot, _ = session
    # The first reset command trips shutdown; further interpolation must not run.
    original = robot.send_action
    def stop_after_send(action):
        value = original(action)
        if web.phase == 'resetting':
            ctx.runtime.shutdown_event.parent.set()
        return value
    robot.send_action = stop_after_send
    web.command({'session_id': web.directory.name, 'action': 'reset'})
    wait_for(lambda: web.cancel_reset.is_set())
    count = len(robot.sent)
    time.sleep(.05)
    assert len(robot.sent) == count
    assert web.phase != 'paused'


def test_observers_preserve_values_order_and_io_count(tmp_path):
    from lerobot.utils.feature_utils import build_dataset_frame
    ctx, _, robot, policy = context()
    monitor = NativeMonitor(tmp_path)
    try:
        engine = ctx.policy.inference
        baseline = []
        for _ in range(9):
            frame = build_dataset_frame(FEATURES, robot.get_observation(), 'observation')
            baseline.append(engine.get_action(frame).clone())
        engine.reset()
        calls = policy.calls
        install_observers(ctx, monitor)
        reads = robot.reads
        for expected in baseline:
            obs = ctx.hardware.robot_wrapper.get_observation()
            frame = build_dataset_frame(FEATURES, obs, 'observation')
            actual = engine.get_action(frame)
            assert torch.equal(actual, expected)
            result = ctx.hardware.robot_wrapper.send_action(dict(zip(KEYS, actual.tolist())))
            assert result == dict(zip(KEYS, actual.tolist()))
        assert robot.reads - reads == 9, 'Monitor must never perform an extra camera/bus read'
        assert monitor.state['inference_count'] == policy.calls - calls == 3
        assert len(monitor.frames) == 2, 'Latest slots are bounded'
        monitor.publish_once()
        assert not list(tmp_path.glob('*.jpg')), 'No viewers, no JPEG encoding'
        atomic_json(tmp_path/'viewer.json', {'enabled': True, 'images': True, 'expires_at': time.time()+4})
        monitor.publish_once()
        assert len(list(tmp_path.glob('*.jpg'))) == 4
        status = read_json(tmp_path/'status.json')
        assert status['image_observations']['input']['task'] == 'original instruction'
        assert len(status['image_observations']['input']['state']) == 6
    finally:
        monitor.close()


def test_api_command_atomicity_and_fp8_isolation(tmp_path, monkeypatch):
    import native_api
    directory = tmp_path / ('c'*32)
    directory.mkdir()
    monkeypatch.setattr(native_api, 'SESSIONS', tmp_path)
    process = NS(args=['python', str(native_api.HERE/'native_runner.py')], poll=lambda: None)
    rollout = NS(_state_lock=threading.Lock(), _inference_proc=process,
                 _inference_meta={'native_session_dir': str(directory)})
    upstream = NS(handle_inference_status=lambda: {'inference_active': True})
    app = FastAPI()
    native_api.install_native_api(app, rollout, upstream)
    client = TestClient(app)
    atomic_json(directory/'status.json', {'phase': 'running'})
    request = {'session_id': directory.name, 'action': 'reset'}
    assert client.post('/api/native-inference/command', json={**request, 'session_id': 'd'*32}).status_code == 409
    result = client.post('/api/native-inference/command', json=request)
    assert result.status_code == 202
    assert client.post('/api/native-inference/command', json=request).status_code == 409
    command = read_json(directory/'command.json')
    atomic_json(directory/'status.json', {'phase': 'paused', 'ack': command['id']})
    assert client.post('/api/native-inference/command', json={**request, 'action':'resume'}).status_code == 202
    (directory/'input-123-0.jpg').write_bytes(b'immutable-frame')
    assert client.get(f'/api/native-inference/image/{directory.name}/input-123-0.jpg').content == b'immutable-frame'
    assert client.get(f'/api/native-inference/image/{directory.name}/status.json').status_code == 404
    process.args = ['python', str(native_api.HERE/'robot_runner.py')]
    assert client.get('/api/native-inference/status').status_code == 409
    assert client.post('/api/native-inference/command', json=request).status_code == 409
    assert 'native_session_id' not in upstream.handle_inference_status()


def test_rtc_input_observer_keeps_raw_frame_before_native_prepare_mutates_it(tmp_path):
    from lerobot.inference import rtc
    from lerobot.utils.feature_utils import build_dataset_frame
    ctx, _, robot, policy = context()
    monitor = NativeMonitor(tmp_path)
    try:
        install_observers(ctx, monitor)
        frame = build_dataset_frame(FEATURES, robot.get_observation(), 'observation')
        prepared = rtc.prepare_observation_for_inference(frame, torch.device('cpu'), 'rtc task', 'so101_follower')
        policy.predict_action_chunk(prepared)
        raw = monitor.frames['input'][1]
        assert isinstance(raw['observation.images.head'], np.ndarray)
        assert raw['observation.images.head'].shape == (48,64,3)
        assert raw['observation.images.head'].dtype == np.uint8
        assert isinstance(prepared['observation.images.head'], torch.Tensor)
    finally:
        monitor.close()


def test_monitor_failure_does_not_change_original_observation(tmp_path, monkeypatch):
    ctx, _, robot, _ = context()
    monitor = NativeMonitor(tmp_path)
    try:
        install_observers(ctx, monitor)
        def broken(_):
            raise RuntimeError('test-only monitor failure')
        monkeypatch.setattr(monitor, 'observation', broken)
        reads = robot.reads
        result = ctx.hardware.robot_wrapper.get_observation()
        assert result['head'] is robot.image
        assert robot.reads == reads+1
    finally:
        monitor.close()


def test_native_duration_still_finishes(tmp_path, monkeypatch):
    monkeypatch.setenv('LELAB_NATIVE_SESSION_DIR', str(tmp_path / ('e'*32)))
    ctx, strategy, robot, _ = context(duration=.08)
    web = WebSession(strategy, ctx)
    thread = threading.Thread(target=web.run)
    thread.start()
    thread.join(5)
    assert not thread.is_alive()
    assert not web.controller.failed
    assert web.controller.stopped
    assert len(robot.sent) >= 1


def test_slow_image_encoder_does_not_block_control_observers(tmp_path, monkeypatch):
    import cv2
    entered, release = threading.Event(), threading.Event()
    original = cv2.imencode
    def slow(*args, **kwargs):
        entered.set()
        release.wait(3)
        return original(*args, **kwargs)
    monkeypatch.setattr(cv2, 'imencode', slow)
    monitor = NativeMonitor(tmp_path)
    atomic_json(tmp_path/'viewer.json', {'enabled': True, 'images': True, 'expires_at': time.time()+4})
    raw = {**dict.fromkeys(KEYS, 1), 'head': np.zeros((480,640,3), np.uint8)}
    monitor.observation(raw)
    try:
        assert entered.wait(2)
        start = time.perf_counter()
        for _ in range(1000):
            monitor.observation(raw)
        assert time.perf_counter()-start < .2
        assert len(monitor.frames) == 1
    finally:
        release.set()
        monitor.close()


@pytest.mark.parametrize('web_enabled', [False, True])
def test_launcher_preserves_native_flags_and_adds_scoped_session(tmp_path, monkeypatch, web_enabled):
    from lelab import rollout, record, teleoperate
    for name, value in {'inference_active': False, '_inference_proc': None,
                        '_inference_meta': {}, '_inference_started_at': None,
                        '_inference_rollout_started_at': None}.items():
        monkeypatch.setattr(rollout, name, value)
    monkeypatch.setattr(record, 'recording_active', False)
    monkeypatch.setattr(teleoperate, 'teleoperation_active', False)
    monkeypatch.setattr(rollout, 'setup_follower_calibration_file', lambda _: 'existing-calibration')
    monkeypatch.setattr(rollout, '_resolve_policy_path', lambda _: str(tmp_path))
    monkeypatch.setattr(rollout, '_detect_device', lambda: 'cuda')
    monkeypatch.setattr(rollout, '_pump_stdout', lambda proc, log: log.close())
    if web_enabled:
        monkeypatch.setenv('LELAB_NATIVE_RUNNER', '/test/native_runner.py')
        monkeypatch.setenv('LELAB_NATIVE_SESSIONS', str(tmp_path/'sessions'))
    else:
        monkeypatch.delenv('LELAB_NATIVE_RUNNER', raising=False)
    captured = {}
    def popen(command, **kwargs):
        captured.update(command=command, **kwargs)
        return NS(pid=1, stdin=io.BytesIO(), poll=lambda: None)
    monkeypatch.setattr(rollout.subprocess, 'Popen', popen)
    request = rollout.InferenceRequest(follower_port='/dev/fake', follower_config='existing.json',
                                      policy_ref=str(tmp_path), task='original task', duration_s=120,
                                      cameras={'head': {'type':'opencv','camera_index':0,'fps':30}})
    result = rollout.handle_start_inference(request)
    assert result['success']
    command = captured['command']
    for flag in ['--strategy.type=base', '--policy.device=cuda', '--robot.type=so101_follower',
                 '--robot.port=/dev/fake', '--robot.id=existing-calibration', '--task=original task', '--duration=120']:
        assert flag in command
    assert any(flag.startswith('--robot.cameras=') for flag in command)
    if web_enabled:
        assert command[1] == '/test/native_runner.py'
        assert '--interactive=true' in command
        assert captured['env']['LELAB_NATIVE_SESSION_DIR'] == rollout._inference_meta['native_session_dir']
    else:
        assert command[1:3] == ['-m', 'lerobot.scripts.lerobot_rollout']
        assert '--interactive=true' not in command


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))


def test_shared_internal_chunk_entry_observes_sync_and_rtc_once_without_changing_actions(tmp_path):
    from lerobot.inference import rtc
    from lerobot.utils.feature_utils import build_dataset_frame

    class SharedChunkPolicy(ToyPolicy):
        # Same dispatch shape as SmolVLA: both entry points call the shared method.
        _get_action_chunk = ToyPolicy.predict_action_chunk

        def predict_action_chunk(self, observation):
            return self._get_action_chunk(observation)

        def select_action(self, observation):
            if not self.queue:
                self.queue.extend(self._get_action_chunk(observation)[0])
            return self.queue.popleft().unsqueeze(0)

    ctx, _, robot, _ = context()
    engine = ctx.policy.inference
    policy = SharedChunkPolicy()
    engine._policy = policy
    frame = build_dataset_frame(FEATURES, robot.get_observation(), 'observation')
    baseline = [engine.get_action(frame).clone() for _ in range(9)]
    engine.reset()
    monitor = NativeMonitor(tmp_path)
    try:
        install_observers(ctx, monitor)
        for expected in baseline:
            assert torch.equal(engine.get_action(frame), expected)
        assert monitor.state['inference_count'] == 3
        assert monitor.state['input_task'] == engine.task
        # RTC prepares the exact task snapshot on its worker, then calls the
        # public entry. Its nested shared call must be counted only once.
        prepared = rtc.prepare_observation_for_inference(dict(frame), torch.device('cpu'),
                                                         'rtc snapshot task', 'so101_follower')
        result = policy.predict_action_chunk(prepared)
        assert result.shape == (1,4,6)
        assert monitor.state['inference_count'] == 4
        assert monitor.state['input_task'] == 'rtc snapshot task'
    finally:
        monitor.close()
