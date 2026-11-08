"""Single-owner robot I/O, asynchronous chunk production, live sync/RTC switching."""
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np


@dataclass
class Prediction:
    policy: np.ndarray
    physical: list[dict]
    inference_ms: float
    roundtrip_ms: float


class ControlEngine:
    def __init__(self, robot, predict, settings, monitor, joints, *, fps=30, horizon=10,
                 initial_inference_s=.5):
        self.robot, self.predict = robot, predict
        self.settings, self.monitor, self.joints = settings, monitor, joints
        self.fps, self.horizon = fps, horizon
        self.initial_inference_s = initial_inference_s
        self.policy = np.empty((0, len(joints)), np.float32)
        self.physical, self.index = [], 0
        self.total_sent = self.refill_at = self.epoch = 0
        self.mode = settings.get()['mode']
        self.revision = settings.get().get('revision', 0)
        self.future = None
        self.job = None
        self.prediction = None
        self.latencies = []
        self.bootstrap = True
        self.state, self.target, self.requested_target = {}, {}, {}
        self.state_time = None
        self.frames = None
        self.phase = 'initializing'
        self.rejected = self.overruns = 0
        self.inference_count = 0
        self.offer_us = []

    @property
    def remaining(self):
        return len(self.physical)-self.index

    def budget(self):
        seconds = max([self.initial_inference_s, *self.latencies[-8:]])
        return math.ceil(seconds*self.fps)+2

    def switch(self):
        command = self.settings.get()
        revision = command.get('revision', 0)
        if revision != self.revision or command['mode'] != self.mode:
            self.mode, self.revision = command['mode'], revision
            self.epoch += 1
            self.physical, self.index = [], 0
            self.policy = np.empty((0, len(self.joints)), np.float32)
            self.bootstrap = True
            self.phase = 'mode_transition_hold'
            # The outstanding HTTP call cannot be cancelled safely. Its result
            # is discarded by epoch; no old commands enter the new queue.

    def collect(self):
        if self.future is None or not self.future.done():
            return
        future, job = self.future, self.job
        self.future = self.job = None
        prediction = future.result()
        if job['epoch'] != self.epoch:
            self.rejected += 1
            return
        policy = np.asarray(prediction.policy)
        if (policy.ndim != 2 or policy.shape[1] != len(self.joints) or
            len(prediction.physical) != len(policy) or not np.isfinite(policy).all() or
            any(set(row) != set(self.joints) or not all(math.isfinite(v) for v in row.values())
                for row in prediction.physical)):
            raise ValueError('Invalid predicted physical/policy action chunk')
        elapsed = time.monotonic()-job['observed_at']
        skip = 0
        if job['prefix_length']:
            played = self.total_sent-job['sent_at']
            skip = max(played, math.ceil(elapsed*self.fps))
            if skip >= job['prefix_length'] or skip >= len(policy):
                raise RuntimeError('RTC response exceeded previous playback coverage; stopped')
        self.latencies.append(elapsed)
        self.prediction = prediction
        self.inference_count += 1
        count = len(policy) if self.mode == 'rtc' else min(len(policy), self.horizon)
        self.policy = policy[skip:count].copy()
        self.physical = prediction.physical[skip:count]
        self.index = 0
        if not self.physical:
            raise RuntimeError('No fresh predicted actions remain')
        self.bootstrap = False
        wait_steps = max(1, min(self.horizon, self.remaining-self.budget()-2))
        self.refill_at = self.total_sent+wait_steps
        self.phase = 'executing'

    def submit(self, executor):
        if self.future is not None:
            return
        prefix_length = 0
        if self.mode == 'sync':
            if self.remaining:
                return
        elif not self.bootstrap:
            if self.total_sent < self.refill_at:
                return
            prefix_length = self.remaining
            if prefix_length <= self.budget():
                raise RuntimeError('RTC buffer cannot cover measured inference delay; stopped')
        observation = self.robot.get_observation()
        observed_at = time.monotonic()
        self.state = {name: float(observation[name]) for name in self.joints}
        self.state_time = time.time()
        self.frames = {name: observation[name] for name in ('head', 'wrist') if name in observation}
        previous = self.policy[self.index:].copy() if prefix_length else None
        physical = self.physical[self.index:] if prefix_length else None
        options = {'enabled': self.mode == 'rtc', 'previous': previous, 'physical': physical,
                   'delay': self.budget() if prefix_length else 0,
                   'horizon': prefix_length}
        self.job = {'epoch': self.epoch, 'observed_at': observed_at, 'sent_at': self.total_sent,
                    'prefix_length': prefix_length}
        self.future = executor.submit(self.predict, observation, options)
        self.phase = 'inference_with_playback' if self.remaining else 'inference_hold'
        self.publish(frames=self.frames)

    def publish(self, frames=None, error=None):
        metadata = {'mode': self.mode, 'mode_revision': self.revision, 'phase': self.phase,
                    'state': self.state, 'state_observed_at': self.state_time,
                    'target': self.target, 'requested_target': self.requested_target,
                    'buffer_remaining': self.remaining, 'actions_sent': self.total_sent,
                    'inference_count': self.inference_count, 'control_fps': self.fps,
                    'control_overruns': self.overruns, 'stale_results_discarded': self.rejected,
                    'delay_budget_steps': self.budget(), 'error': error,
                    'monitor_offer_max_us': max(self.offer_us, default=0)}
        if self.prediction:
            metadata.update(inference_ms=self.prediction.inference_ms,
                            roundtrip_ms=self.prediction.roundtrip_ms,
                            predicted_chunk=self.prediction.physical)
        started = time.perf_counter()
        self.monitor.offer(metadata, frames=frames)
        self.offer_us.append(1e6*(time.perf_counter()-started))
        if len(self.offer_us) > 120:
            self.offer_us.pop(0)

    def run(self, stopped, duration):
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='policy-http')
        deadline, tick = time.monotonic()+duration, time.monotonic()
        last_publish = 0.
        try:
            while not stopped.is_set() and time.monotonic() < deadline:
                self.switch()
                self.collect()
                self.submit(executor)
                if self.remaining and not stopped.is_set():
                    action = self.physical[self.index]
                    sent = self.robot.send_action(action)
                    self.requested_target = dict(action)
                    self.target = dict(sent) if isinstance(sent, dict) else dict(action)
                    self.index += 1
                    self.total_sent += 1
                elif self.mode == 'rtc' and not self.bootstrap and self.future is not None:
                    raise RuntimeError('RTC playback buffer exhausted; stopped instead of replaying stale actions')
                now = time.monotonic()
                if now-last_publish >= .1:
                    self.publish()
                    last_publish = now
                tick += 1/self.fps
                if tick < now:
                    self.overruns += 1
                    tick = now+1/self.fps
                stopped.wait(max(0, tick-time.monotonic()))
        except Exception as exc:
            self.phase = 'error_stopped'
            self.publish(error=str(exc))
            raise
        finally:
            if self.phase != 'error_stopped':
                self.phase = 'stopped'
                self.publish()
            executor.shutdown(wait=False, cancel_futures=True)
