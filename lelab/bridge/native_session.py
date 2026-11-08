"""Browser controls and bounded, read-only telemetry for native LeRobot rollout.

No policy/processor/action loop is implemented here. The native controller owns
segments, resets, task invalidation and holds. Only the publisher encodes images.
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def positions(value: dict) -> dict[str, float]:
    return {k: float(v) for k, v in value.items()
            if k.endswith('.pos') and math.isfinite(float(v))}


class NativeMonitor:
    """Latest slot only; no queues, disk work, camera reads or GPU copies on ticks."""

    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.publish_lock = threading.Lock()
        self.state = {'session_id': directory.name, 'phase': 'setting_up', 'task': '',
                      'ack': None, 'error': None, 'inference_count': 0, 'segment': 0}
        self.frames: dict[str, tuple[int, dict, dict]] = {}
        self.encoded: dict[str, int] = {}
        self.images: dict[str, list[dict]] = {}
        self.generations: dict[str, list[list[str]]] = {}
        self.image_observations: dict[str, dict] = {}
        self.closed = threading.Event()
        self.thread = threading.Thread(target=self._publish, name='native-monitor', daemon=True)
        self.thread.start()

    def update(self, **values):
        with self.lock:
            self.state.update(values)

    def observation(self, obs: dict):
        now = time.time()
        with self.lock:
            self.state.update(actual=positions(obs), observed_at=now)
            self.frames['execution'] = (time.time_ns(), dict(obs), {'at': now})

    def prediction(self, frame: dict, task: str):
        # frame is a fresh CPU dataset frame owned by this tick/RTC snapshot.
        with self.lock:
            self.state['inference_count'] += 1
            self.state.update(input_at=time.time(), input_task=task,
                              input_state=None, target=None)
            self.frames['input'] = (time.time_ns(), dict(frame), {
                'at': self.state['input_at'], 'task': task,
                'inference_count': self.state['inference_count']})

    def selected(self, action, keys):
        # Sync and RTC already return CPU tensors. Never introduce a GPU transfer.
        if action is None or getattr(getattr(action, 'device', 'cpu'), 'type', 'cpu') != 'cpu':
            return
        values = action.detach().reshape(-1).tolist()
        self.update(target={key: value if math.isfinite(value) else None
                            for key, value in zip(keys, values)}, target_at=time.time())

    def _encode(self, group, version, frame, metadata):
        import cv2
        import numpy as np

        files = []
        state = frame.get('observation.state')
        if group == 'input' and state is not None:
            metadata = {**metadata, 'state': state.tolist() if hasattr(state, 'tolist') else list(state)}
        for key, value in frame.items():
            if not hasattr(value, 'shape') or len(value.shape) != 3:
                continue
            # CPU raw images only; prepared GPU tensors are deliberately excluded.
            if getattr(getattr(value, 'device', 'cpu'), 'type', getattr(value, 'device', 'cpu')) != 'cpu':
                continue
            array = np.asarray(value)
            if array.shape[-1] != 3 and array.shape[0] == 3:
                array = np.moveaxis(array, 0, -1)
            if array.shape[-1] != 3:
                continue
            if array.dtype != np.uint8:
                array = np.clip(array * 255, 0, 255).astype(np.uint8)
            height, width = array.shape[:2]
            if width > 480:
                array = cv2.resize(array, (480, max(1, round(height * 480 / width))))
            ok, encoded = cv2.imencode('.jpg', cv2.cvtColor(array, cv2.COLOR_RGB2BGR),
                                       [cv2.IMWRITE_JPEG_QUALITY, 65])
            if not ok:
                continue
            filename = f'{group}-{version}-{len(files)}.jpg'
            path = self.directory / filename
            temporary = path.with_suffix('.tmp')
            temporary.write_bytes(encoded.tobytes())
            temporary.replace(path)
            files.append({'name': key.removeprefix('observation.images.'),
                          'file': filename, 'version': str(version)})
        self.images[group] = files
        self.image_observations[group] = metadata
        self.encoded[group] = version
        generations = self.generations.setdefault(group, [])
        generations.append([item['file'] for item in files])
        while len(generations) > 3:
            for old in generations.pop(0):
                (self.directory / old).unlink(missing_ok=True)

    def publish_once(self):
        with self.publish_lock:
            self._publish_once()

    def _publish_once(self):
        viewer = read_json(self.directory / 'viewer.json')
        watching = viewer.get('expires_at', 0) > time.time() and viewer.get('enabled', False)
        with self.lock:
            frames = dict(self.frames)
            state = dict(self.state)
        if 'input' in frames:
            input_state = frames['input'][1].get('observation.state')
            if input_state is not None:
                state['input_state'] = input_state.tolist()
        if watching and viewer.get('images', True):
            for group, (version, frame, metadata) in frames.items():
                if self.encoded.get(group) != version:
                    self._encode(group, version, frame, metadata)
        # Keep data truthful when paused: timestamps are never refreshed without observations.
        state.update(published_at=time.time(), images=self.images if watching else {},
                     image_observations=self.image_observations, monitor_enabled=watching)
        if state.get('segment_started_at') and state['phase'] == 'running':
            state['segment_elapsed_s'] = time.time() - state['segment_started_at']
        atomic_json(self.directory / 'status.json', state)

    def _publish(self):
        while not self.closed.is_set():
            try:
                self.publish_once()
            except Exception:
                logger.exception('Native monitor publisher failed (control loop continues)')
            self.closed.wait(0.5)

    def close(self):
        self.closed.set()
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            return
        try:
            self.publish_once()
        except Exception:
            logger.exception('Final native monitor publish failed')


def install_observers(ctx, monitor: NativeMonitor, *, hardware=True):
    """Wrap original calls once, returning the exact original values unchanged."""
    wrapper = ctx.hardware.robot_wrapper
    observe, send = wrapper.get_observation, wrapper.send_action
    warning_reported = False

    def notify(callback, *args, **kwargs):
        nonlocal warning_reported
        try:
            callback(*args, **kwargs)
        except Exception:
            # Display failures must never turn successful native I/O into a fault.
            if not warning_reported:
                logger.exception('Native telemetry observer failed; original rollout continues')
                warning_reported = True

    def get_observation(*args, **kwargs):
        result = observe(*args, **kwargs)
        notify(monitor.observation, result)
        return result

    def send_action(action, *args, **kwargs):
        result = send(action, *args, **kwargs)
        applied = result if isinstance(result, dict) else action
        notify(lambda: monitor.update(driver_target=positions(applied), sent_at=time.time()))
        return result

    if hardware:
        wrapper.get_observation, wrapper.send_action = get_observation, send_action
    engine = ctx.policy.inference
    original_get = engine.get_action
    local = threading.local()

    def get_action(frame):
        local.frame = frame
        local.task = engine.task
        result = original_get(frame)
        notify(monitor.selected, result, ctx.data.ordered_action_keys)
        return result

    engine.get_action = get_action
    policy = getattr(engine, '_policy', None)
    # SmolVLA select_action bypasses predict_action_chunk and calls this shared
    # internal entry directly. Hook one entry only to avoid double-counting RTC.
    prediction_entry = ('_get_action_chunk' if callable(getattr(policy, '_get_action_chunk', None))
                        else 'predict_action_chunk')
    predict = getattr(policy, prediction_entry, None)
    predict = getattr(predict, '_native_original', predict)
    if callable(predict):
        def observed_prediction(*args, **kwargs):
            frame = getattr(local, 'frame', None)
            if frame is not None:
                notify(monitor.prediction, frame, getattr(local, 'task', engine.task))
            return predict(*args, **kwargs)
        observed_prediction._native_original = predict
        setattr(policy, prediction_entry, observed_prediction)
    # The native RTC worker snapshots CPU frames before preprocessing. Observe that
    # exact snapshot rather than the latest camera frame or a prepared GPU tensor.
    from lerobot.inference import rtc
    prepare = getattr(rtc.prepare_observation_for_inference, '_native_original', rtc.prepare_observation_for_inference)

    def prepare_observation(frame, device, task, robot_type):
        local.frame = dict(frame)  # native prepare mutates its dict into GPU tensors
        local.task = task
        return prepare(frame, device, task, robot_type)

    prepare_observation._native_original = prepare
    rtc.prepare_observation_for_inference = prepare_observation


class WebSession:
    """Drop-in InteractiveSession, using files instead of an stdin command loop."""

    def __init__(self, strategy, ctx):
        from lerobot.rollout.controller import RolloutController

        self.ctx, self.strategy = ctx, strategy
        self.directory = Path(os.environ['LELAB_NATIVE_SESSION_DIR'])
        self.monitor = NativeMonitor(self.directory)
        from native_parameters import RuntimeParameters
        self.parameters = RuntimeParameters(ctx, json.loads(os.environ.get('LELAB_NATIVE_SOURCE_CONFIG', '{}')))
        self.monitor.update(task=ctx.policy.inference.task,
                            initial_position=ctx.hardware.initial_position,
                            joint_keys=list(ctx.data.ordered_action_keys), parameters=self.parameters.describe())
        self.controller = RolloutController(strategy, ctx, on_event=self._event)
        self.finished = threading.Event()
        self.cancel_reset = threading.Event()
        self.phase = 'setting_up'
        self.last_ack = None
        self.parameter_fault = False
        install_observers(ctx, self.monitor)
        original_return = strategy.return_to_initial_position

        def return_to_initial_position(hw, duration_s=3.0, fps=50):
            # Retain native interpolation and action processing. Only explicit web
            # reset gets cancellation checkpoints; normal teardown stays native.
            if self.cancel_reset.is_set():
                return False
            if self.phase != 'resetting':
                return original_return(hw, duration_s, fps)
            original_send = hw.robot_wrapper.send_action
            def cancellable_send(action):
                if self.ctx.runtime.shutdown_event.parent.is_set() or self.cancel_reset.is_set():
                    self.cancel_reset.set()
                    raise InterruptedError('Reset cancelled by shutdown')
                return original_send(action)
            hw.robot_wrapper.send_action = cancellable_send
            try:
                result = original_return(hw, duration_s, fps)
                if result:
                    measured = hw.robot_wrapper.get_observation()  # once after explicit reset
                    errors = [abs(float(measured[k]) - float(v))
                              for k, v in hw.initial_position.items()]
                    residual = max(errors) if all(math.isfinite(v) for v in errors) else math.inf
                    if not math.isfinite(residual) or residual > 3.0:
                        logger.warning('Reset target not reached (max residual %.3f robot units)', residual)
                        self.monitor.update(reset_residual=residual if math.isfinite(residual) else None)
                        return False
                return result
            except Exception:
                logger.exception('Explicit reset position verification failed')
                return False
            finally:
                hw.robot_wrapper.send_action = original_send
        strategy.return_to_initial_position = return_to_initial_position

    def _set_phase(self, phase, **values):
        self.phase = phase
        self.monitor.update(phase=phase, task=self.controller.task, **values)

    def _event(self, event, _payload):
        name = event.value
        if name == 'segment_started':
            with self.monitor.lock:
                segment = self.monitor.state['segment'] + 1
            self._set_phase('running', segment=segment, segment_started_at=time.time(),
                            segment_elapsed_s=0, error=None)
        elif name == 'segment_ended':
            # Preserve the original one-shot duration limit unless user reset first.
            self._set_phase('completed')
            self.finished.set()
        elif name == 'reset_started':
            self._set_phase('resetting', target=None)
        elif name == 'reset_done':
            self._set_phase('paused', segment_elapsed_s=0, target=None)
        elif name in {'reset_failed', 'reset_skipped', 'engine_failed', 'strategy_failed'}:
            self._set_phase('error', error=name, target=None)
        elif name == 'stopped':
            if self.phase != 'error':
                self._set_phase('stopped')

    def command(self, request):
        if request.get('session_id') != self.directory.name:
            raise ValueError('Stale session')
        if self.parameter_fault:
            raise RuntimeError('Parameter engine unavailable; Stop and restart this session')
        action = request.get('action')
        if action == 'reset' and self.phase in {'running', 'paused', 'error'} and not self.controller.failed:
            task = self.controller.task
            self._set_phase('reset_requested', error=None)
            self.controller.reset()
            if self.controller.task != task:
                self.controller.set_task(task)
        elif action == 'resume' and self.phase == 'paused':
            task = request.get('task', self.controller.task)
            if not isinstance(task, str) or len(task) > 4096:
                raise ValueError('Prompt must be text, at most 4096 characters')
            if task != self.controller.task:
                rejection = self.controller.text_input_error(task, instruction=True)
                if rejection:
                    raise ValueError(rejection)
            requested_parameters = request.get('parameters')
            if requested_parameters is not None:
                from native_parameters import ParameterEngineError
                try:
                    self.parameters.apply(requested_parameters, self.strategy, self.controller, self.monitor)
                except ParameterEngineError as exc:
                    self.parameter_fault = True
                    self._set_phase('error', error=str(exc))
                    raise
                self.monitor.update(parameters=self.parameters.describe())
            if task != self.controller.task:
                if not self.controller.set_task(task):
                    raise RuntimeError('Controller refused prompt update')
            self._set_phase('resuming')
            if not self.controller.start():
                self._set_phase('error', error='Controller refused resume')
                raise RuntimeError('Controller refused resume')
        else:
            raise ValueError(f'Cannot {action} while {self.phase}')

    def _commands(self):
        while not self.controller.stopped:
            if self.finished.is_set():
                self.controller.stop()
                return
            request = read_json(self.directory / 'command.json')
            request_id = request.get('id')
            if request_id and request_id != self.last_ack:
                try:
                    self.command(request)
                    self.monitor.update(command_error=None)
                except Exception as exc:
                    self.monitor.update(command_error=str(exc))
                self.last_ack = request_id
                self.monitor.update(ack=request_id)
            self.finished.wait(0.05)

    def run(self):
        self.controller.start()
        reader = threading.Thread(target=self._commands, name='native-commands', daemon=True)
        reader.start()
        try:
            self.controller.serve()
        finally:
            if self.controller.failed:
                self.ctx.hardware._lelab_failure = self.controller.failure_traceback
            self.finished.set()
            reader.join(timeout=1)
            self.monitor.close()
