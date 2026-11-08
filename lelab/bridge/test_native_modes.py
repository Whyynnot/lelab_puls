from types import SimpleNamespace
import json
import pytest
from native_modes import describe_modes, selected_description, explicit_mode_flags
from native_parameters import validate_parameters


@pytest.mark.parametrize('kind',['smolvla','pi0','pi05'])
def test_guided_mode_uses_full_chunk_and_bounded_prefix(kind):
    config={'type':kind,'chunk_size':50,'n_action_steps':10}
    description=selected_description(config,'rtc')
    assert description['guided'] and description['effective']['n_action_steps']==50
    values=validate_parameters({'chunk_size':20,'n_action_steps':20,'fps':15},description)
    flags=explicit_mode_flags(config,'rtc',values)
    assert '--inference.rtc.mode=guided' in flags
    assert '--inference.rtc.execution_horizon=19' in flags
    assert '--inference.queue_threshold=19' in flags
    with pytest.raises(ValueError,match='N must equal M'):
        validate_parameters({'n_action_steps':10},description)


@pytest.mark.parametrize('extra',[{'type':'act'},{'n_obs_steps':2},{'chunk_size':1},
    {'use_visual_memory':True},{'observation_delta_indices':[-1,0]}, {'temporal_ensemble_coeff':.1}])
def test_unsupported_configs_reject_rtc_before_loading_weights(extra):
    config={'type':'smolvla','chunk_size':50,'n_action_steps':10,**extra}
    assert describe_modes(config)['rtc_unavailable_reason']
    with pytest.raises(ValueError,match='不支持推理模式'):
        selected_description(config,'rtc')


def test_relative_policy_cannot_be_forced_into_sync():
    with pytest.raises(ValueError):
        selected_description({'type':'pi05','chunk_size':50,'n_action_steps':10,'use_relative_actions':True},'sync')


def test_original_automatic_mode_is_retained():
    config={'type':'smolvla','chunk_size':50,'n_action_steps':10}
    assert explicit_mode_flags(config,'auto',{}) is None
    assert selected_description(config,'auto')['effective']['n_action_steps']==10


def test_sync_override_clears_saved_rtc_config_with_native_parser(tmp_path):
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    config=SmolVLAConfig(rtc_config=None)
    config.save_pretrained(tmp_path)
    raw=json.loads((tmp_path/'config.json').read_text())
    raw['rtc_config']={'enabled':True}
    (tmp_path/'config.json').write_text(json.dumps(raw))
    flags=explicit_mode_flags(raw,'sync',{})
    assert '--policy.rtc_config=null' in flags
    loaded=SmolVLAConfig.from_pretrained(tmp_path,cli_overrides=['--rtc_config=null'])
    assert loaded.rtc_config is None


def test_installed_smolvla_declares_real_native_guided_contract():
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.inference.rtc import validate_local_chunk_policy,supports_rtc_inference
    from lerobot.inference.contracts import ExecutionMode
    policy=SmolVLAPolicy.__new__(SmolVLAPolicy)
    torch.nn.Module.__init__(policy)
    policy.config=SimpleNamespace(n_obs_steps=1,chunk_size=50,n_action_steps=50)
    assert supports_rtc_inference(policy)
    assert ExecutionMode.RTC_GUIDED in validate_local_chunk_policy(policy).modes
