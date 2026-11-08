"""LeRobot's inference-time guided RTC for ordinary native PI05 checkpoints."""
import torch
from lerobot.policies.rtc.configuration_rtc import RTCConfig


def predict(policy, batch, noise=None, *, enabled=False, previous=None, delay=0, horizon=0):
    if enabled:
        if policy.config.rtc_config is None:
            policy.config.rtc_config = RTCConfig(enabled=True, mode='guided', debug=False)
            policy.init_rtc_processor()
        policy.config.rtc_config.enabled = True
        kwargs = {}
        if previous is not None:
            if not 0 <= delay < horizon <= previous.shape[1] <= policy.config.chunk_size:
                raise ValueError('RTC requires delay < prefix horizon <= remaining chunk length')
            kwargs = {'prev_chunk_left_over': previous, 'inference_delay': delay,
                      'execution_horizon': horizon}
        # RTC enables action-input gradients internally; inference_mode forbids it.
        # The frozen FP8 language prefix needs no gradient.
        context = torch.no_grad()
    else:
        if policy.config.rtc_config is not None:
            policy.config.rtc_config.enabled = False
        kwargs = {}
        context = torch.inference_mode()
    policy.reset()
    with context:
        return policy.predict_action_chunk(batch, noise=noise, **kwargs)
