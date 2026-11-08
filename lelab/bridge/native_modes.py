"""Expose existing LeRobot backends without loading weights or touching hardware."""
from native_parameters import describe_parameters

GUIDED_POLICIES = {'smolvla', 'pi0', 'pi05'}


def rtc_unavailable_reason(config):
    if config.get('type') not in GUIDED_POLICIES:
        return '首版 RTC 前缀引导支持 SmolVLA、PI0 和 PI05。'
    if type(config.get('chunk_size')) is not int or config['chunk_size'] < 2:
        return 'RTC 需要至少 2 步的动作预测块。'
    if config.get('n_obs_steps', 1) != 1 or config.get('temporal_ensemble_coeff') is not None:
        return '该配置使用历史观测或时间集成，不能使用当前 RTC 入口。'
    if config.get('use_visual_memory') or config.get('use_proprioceptive_memory'):
        return '当前 RTC 入口不采样历史记忆。'
    for key in ('observation_delta_indices','image_observation_delta_indices','state_observation_delta_indices'):
        if config.get(key) is not None and config[key] != [0]:
            return '当前 RTC 入口只支持单个当前观测。'
    return ''


def describe_modes(config):
    options = [{'value':'auto','label':'自动（模型默认）','parameters':describe_parameters(config)}]
    if not config.get('use_relative_actions') and config.get('type') != 'groot':
        options.append({'value':'sync','label':'普通（同步）',
                        'parameters':describe_parameters(config,backend='sync',guided=False)})
    reason = rtc_unavailable_reason(config)
    if not reason:
        options.append({'value':'rtc','label':'RTC（异步前缀引导）',
                        'parameters':describe_parameters(config,backend='rtc',guided=True)})
    return {'default':'auto','options':options,'rtc_unavailable_reason':reason}


def selected_description(config, mode):
    option = next((item for item in describe_modes(config)['options'] if item['value'] == mode),None)
    if option is None:
        reason = rtc_unavailable_reason(config) if mode == 'rtc' else '该模型配置不能使用同步模式。'
        raise ValueError(f'不支持推理模式 {mode!r}：{reason}')
    return option['parameters']


def explicit_mode_flags(config, mode, values):
    selected_description(config, mode)
    if mode == 'auto':
        return None  # Caller retains the original automatic selection.
    if mode == 'sync':
        return ['--inference.type=sync', *(['--policy.rtc_config=null'] if config.get('rtc_config') else [])]
    chunk = values['chunk_size']
    return ['--inference.type=rtc','--inference.rtc.enabled=true','--inference.rtc.mode=guided',
            f'--inference.rtc.execution_horizon={chunk-1}',
            f'--inference.queue_threshold={min(30,chunk-1)}']
