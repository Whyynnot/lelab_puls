"""Validated runtime overrides for native policy configs; checkpoint files stay intact."""
from __future__ import annotations

import math

DYNAMIC_CHUNK_POLICIES = {'smolvla', 'pi0', 'pi05'}


class ParameterEngineError(RuntimeError):
    """Old worker could not stop or a replacement could not be restored."""


def config_dict(config):
    if isinstance(config, dict):
        return dict(config)
    return {name: getattr(config, name, None) for name in
            ('type', 'chunk_size', 'n_action_steps', 'temporal_ensemble_coeff',
             'use_relative_actions', 'rtc_training_max_delay')}


def describe_parameters(config, *, fps=30, backend=None, guided=None, trained_chunk_size=None):
    cfg = config_dict(config)
    kind = cfg.get('type')
    backend = backend or ('rtc' if cfg.get('use_relative_actions') or kind == 'groot' else 'sync')
    guided = backend == 'rtc' if guided is None else guided
    chunk = cfg.get('chunk_size')
    maximum = trained_chunk_size or chunk
    return {'policy_type': kind, 'backend': backend, 'guided': bool(guided),
            'effective': {'chunk_size': chunk,
                          'n_action_steps': chunk if guided else cfg.get('n_action_steps'), 'fps': fps},
            'limits': {'max_chunk_size': maximum, 'min_chunk_size': 2 if guided else 1,
                       'chunk_size_editable': kind in DYNAMIC_CHUNK_POLICIES,
                       'n_action_steps_editable': kind in (DYNAMIC_CHUNK_POLICIES | {'act'}) and chunk is not None
                           and not guided and cfg.get('temporal_ensemble_coeff') is None,
                       'min_fps': 1, 'max_fps': 60},
            'note': ('原生 RTC 保留完整预测长度，N 随 M；不改变原生合并规则。' if guided else
                     'ACT 时间集成模式固定每轮执行 1 步。' if cfg.get('temporal_ensemble_coeff') is not None else
                     '同步模式预测 M 步，原生动作队列执行前 N 步后再次预测。')}


def validate_parameters(request, description):
    if not isinstance(request, dict) or set(request) - {'chunk_size', 'n_action_steps', 'fps'}:
        raise ValueError('Only chunk_size, n_action_steps and fps can be overridden')
    current, limits = description['effective'], description['limits']
    values = {**current, **request}
    for name in ('chunk_size', 'n_action_steps'):
        value = values[name]
        if value is None and current[name] is None:
            continue
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    chunk, steps, fps = values['chunk_size'], values['n_action_steps'], values['fps']
    if chunk is not None:
        if not limits['min_chunk_size'] <= chunk <= limits['max_chunk_size']:
            raise ValueError(f"chunk_size must be between {limits['min_chunk_size']} and {limits['max_chunk_size']}")
        if not limits['chunk_size_editable'] and chunk != current['chunk_size']:
            raise ValueError('This model has a fixed trained chunk_size; change n_action_steps instead')
    if chunk is not None and steps is not None and steps > chunk:
        raise ValueError('n_action_steps must not exceed chunk_size (1 <= N <= M)')
    if description['guided'] and steps != chunk:
        raise ValueError('Native guided RTC retains the full chunk; N must equal M')
    if not limits['n_action_steps_editable'] and not description['guided'] and steps != current['n_action_steps']:
        raise ValueError('This native policy mode has a fixed execution step count')
    if type(fps) not in (int, float) or not math.isfinite(fps) or not 1 <= fps <= 60:
        raise ValueError('Action fps must be finite and between 1 and 60')
    return values


class RuntimeParameters:
    def __init__(self, ctx, source=None):
        self.ctx = ctx
        self.policy = getattr(ctx.policy, 'policy', None) or getattr(ctx.policy.inference, '_policy', None)
        self.config = self.policy.config
        self.maximum = (source or {}).get('chunk_size') or getattr(self.config, 'chunk_size', None)
        self.revision = 0

    def describe(self):
        cfg = self.ctx.runtime.cfg
        inference = getattr(cfg, 'inference', None)
        backend = getattr(inference, 'type', 'sync')
        return {**describe_parameters(self.config, fps=cfg.fps, backend=backend,
                    guided=backend == 'rtc' and inference.rtc.enabled,
                    trained_chunk_size=self.maximum), 'revision': self.revision}

    def validate(self, request):
        values = validate_parameters(request, self.describe())
        cfg = self.ctx.runtime.cfg
        if self.describe()['backend'] not in {'sync', 'rtc'}:
            raise ValueError('This remote backend requires restarting its deployment to change parameters')
        if getattr(cfg, 'use_torch_compile', False) and values['chunk_size'] != getattr(self.config, 'chunk_size', None):
            raise ValueError('Changing chunk_size with torch.compile requires a new inference session')
        if self.describe()['guided'] and cfg.inference.rtc.mode == 'trained':
            from lerobot.policies.rtc.configuration_rtc import validate_trained_rtc_horizon
            horizon = min(cfg.inference.rtc.execution_horizon, values['chunk_size'] - 1)
            validate_trained_rtc_horizon(horizon, values['chunk_size'], self.config.rtc_training_max_delay)
        return values

    def _assign(self, values):
        self.ctx.runtime.cfg.fps = values['fps']
        for config in (self.config, getattr(getattr(self.policy, 'model', None), 'config', None)):
            if config is not None:
                for name in ('chunk_size', 'n_action_steps'):
                    if values[name] is not None and hasattr(config, name):
                        setattr(config, name, values[name])

    def apply(self, request, strategy, controller, monitor):
        """Called only while reset-complete and idle, before controller.start()."""
        values = self.validate(request)
        previous = self.describe()['effective']
        if values == previous:
            return
        ctx, cfg = self.ctx, self.ctx.runtime.cfg
        engine = ctx.policy.inference
        if self.describe()['backend'] == 'sync':
            self._assign(values)
        else:
            # Native RTC engines are one-shot. Recreate through the native factory
            # with the same weights/processors/robot after the previous worker exits.
            from lerobot.inference.factory import create_inference_engine
            from native_session import install_observers
            worker = engine._rtc_thread
            engine.stop()
            if worker and worker.is_alive():
                worker.join(timeout=10)
                if worker.is_alive():
                    raise ParameterEngineError('RTC worker is still busy; Stop and restart this session')
            old_horizon = cfg.inference.rtc.execution_horizon

            def build():
                return create_inference_engine(cfg.inference, policy=self.policy,
                    preprocessor=ctx.policy.preprocessor, postprocessor=ctx.policy.postprocessor,
                    robot_wrapper=ctx.hardware.robot_wrapper, dataset_features=ctx.data.dataset_features,
                    ordered_action_keys=ctx.data.ordered_action_keys, task=controller.task, fps=cfg.fps,
                    device=cfg.device, use_torch_compile=cfg.use_torch_compile,
                    compile_warmup_inferences=cfg.compile_warmup_inferences,
                    shutdown_event=ctx.runtime.shutdown_event)

            try:
                self._assign(values)
                if self.describe()['guided']:
                    cfg.inference.rtc.execution_horizon = min(old_horizon, values['chunk_size'] - 1)
                replacement = build()
            except Exception:
                self._assign(previous)
                cfg.inference.rtc.execution_horizon = old_horizon
                try:
                    replacement = build()
                except Exception as exc:
                    raise ParameterEngineError('Cannot restore RTC engine; Stop and restart this session') from exc
                ctx.policy.inference = strategy._engine = replacement
                replacement.set_answer_observer(controller._on_query_answer)
                install_observers(ctx, monitor, hardware=False)
                replacement.start()
                raise
            ctx.policy.inference = strategy._engine = replacement
            replacement.set_answer_observer(controller._on_query_answer)
            install_observers(ctx, monitor, hardware=False)
            replacement.start()
        self.revision += 1
