"""Import/edit/submit contracts checked against actual LeRobot v0.7 schemas."""
import json

import pytest

from lelab.train import TrainingRequest, build_training_command, parse_training_config, validate_training_config


def flags(request):
    command = build_training_command(request, '/tmp/lelab-test-output')[3:]
    return dict(zip(command[::2], command[1::2], strict=True))


def test_native_yaml_preserves_freezing_scheduler_and_episodes():
    request = parse_training_config('''
dataset:
  repo_id: user/data
  episodes: [0, 2, 4]
policy:
  type: pi05
  device: cpu
  path: lerobot/pi05_base
  freeze_vision_encoder: true
  train_expert_only: true
  dtype: bfloat16
steps: 20
optimizer:
  type: adamw
  lr: 0.00012
scheduler:
  type: constant_with_warmup
  num_warmup_steps: 2
''')
    validate_training_config(request)
    args = flags(request)
    assert request.policy_load_pretrained
    assert args['--policy.path'] == 'lerobot/pi05_base'
    assert '--policy.type' not in args
    assert args['--policy.freeze_vision_encoder'] == 'true'
    assert args['--policy.dtype'] == 'bfloat16'
    assert args['--dataset.episodes'] == '[0, 2, 4]'
    assert args['--scheduler.num_warmup_steps'] == '2'
    assert args['--optimizer.lr'] == '0.00012'
    assert args['--use_policy_training_preset'] == 'false'
    assert len(args) * 2 + 3 == len(build_training_command(request, '/tmp/out'))


def test_round_trip_keeps_expert_and_lora_settings():
    original = TrainingRequest(dataset_repo_id='user/data', policy_type='smolvla',
        policy_load_pretrained=True, policy_pretrained_path='lerobot/smolvla_base',
        extra_config={'policy': {'train_expert_only': True},
                      'peft': {'r': 8, 'target_modules': ['q_proj', 'v_proj']}})
    loaded = parse_training_config(original.model_dump_json())
    assert loaded == original
    validate_training_config(loaded)
    assert json.loads(flags(loaded)['--peft.target_modules']) == ['q_proj', 'v_proj']


def test_scratch_act_disables_backbone_and_has_no_checkpoint_path():
    request = TrainingRequest(dataset_repo_id='user/data', policy_pretrained_path='ignored/model')
    args = flags(request)
    assert '--policy.path' not in args
    assert args['--policy.pretrained_backbone_weights'] == 'null'
    validate_training_config(request)


@pytest.mark.parametrize('policy', ['act', 'pi05', 'smolvla'])
def test_explicit_optimizer_is_not_replaced_by_policy_preset(policy):
    request = TrainingRequest(dataset_repo_id='user/data', policy_type=policy, optimizer_lr=0.000123)
    args = flags(request)
    assert args['--use_policy_training_preset'] == 'false'
    assert args['--optimizer.type'] == 'adamw'
    assert '--scheduler.type' in args
    assert args['--optimizer.lr'] == '0.000123'
    validate_training_config(request)


def test_unknown_native_fields_are_rejected_before_launch():
    request = TrainingRequest(dataset_repo_id='user/data', extra_config={'policy': {'typo_freeze': True}})
    with pytest.raises(ValueError, match='typo_freeze'):
        validate_training_config(request)


def test_lists_maps_null_and_argument_strings_keep_their_types():
    request = TrainingRequest(dataset_repo_id='user/data', policy_load_pretrained=True,
        policy_pretrained_path='some/model',
        extra_config={'rename_map': {'observation.images.left': 'observation.images.camera1'},
                      'policy': {'dtype': None}, 'job_name': 'experiment $(date); tuning'})
    args = flags(request)
    assert args['--policy.dtype'] == 'null'
    assert args['--job_name'] == 'experiment $(date); tuning'
    assert json.loads(args['--rename_map'])['observation.images.left'] == 'observation.images.camera1'
    validate_training_config(request)


@pytest.mark.parametrize('data', ['[]', 'dataset: x', '!!python/object:os.system {}'])
def test_invalid_and_unsafe_config_is_rejected(data):
    with pytest.raises(Exception):
        parse_training_config(data)


def test_saved_legacy_config_migrates_tracking_and_does_not_overwrite_output():
    request = parse_training_config(json.dumps({'dataset': {'repo_id': 'user/data'},
        'policy': {'type': 'act'}, 'wandb': {'enable': True, 'project': 'old', 'entity': 'user'},
        'output_dir': '/do-not-overwrite'}))
    assert request.wandb_project == 'old'
    assert request.wandb_enable
    assert flags(request)['--output_dir'] == '/tmp/lelab-test-output'
    assert not any(key.startswith('--wandb') for key in flags(request))


def test_config_api_validates_without_starting_a_job(monkeypatch):
    from fastapi.testclient import TestClient
    from lelab import server

    def forbidden(*args, **kwargs):
        raise AssertionError('Configuration import must not start a training job')
    monkeypatch.setattr(server.job_registry, 'start', forbidden)
    client = TestClient(server.app)
    response = client.post('/training/config/parse', json={'text': '''
dataset: {repo_id: user/data}
policy: {type: pi05, path: lerobot/pi05_base, freeze_vision_encoder: true}
steps: 12
'''})
    assert response.status_code == 200, response.text
    assert response.json()['extra_config']['policy']['freeze_vision_encoder']
    assert response.json()['steps'] == 12
    invalid = client.post('/training/config/parse', json={'text': '[]'})
    assert invalid.status_code == 400
