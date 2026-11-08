"""Contract tests, no motor connection and no random model execution."""
import json
import io
import sys
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from adapter import JOINTS, ROOT, Profile, RobotAdapter, SentencePieceTokenizer, validate_artifact




def test_joint_order_is_explicit_and_complete():
    with pytest.raises(ValueError,match='six SO101 joints'):
        Profile(name='test',artifact='unused',action_names=JOINTS[:-1]+[JOINTS[0]])


def processor_fixture(tmp_path, statistics=True):
    """A configuration-only fixture; no model weights or motor connection."""
    from lerobot.configs import PolicyFeature, FeatureType
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05 import processor_pi05
    from lerobot.processor import TokenizerProcessorStep
    cfg = PI05Config(input_features={
        'observation.state':PolicyFeature(type=FeatureType.STATE,shape=(6,)),
        'observation.images.head':PolicyFeature(type=FeatureType.VISUAL,shape=(3,224,224)),
        'observation.images.wrist':PolicyFeature(type=FeatureType.VISUAL,shape=(3,224,224))},
        output_features={'action':PolicyFeature(type=FeatureType.ACTION,shape=(6,))},
        use_relative_actions=True, relative_exclude_joints=[],action_feature_names=JOINTS,device='cpu')
    cfg.save_pretrained(tmp_path)
    # There are no model weights in this fixture; it tests processors only.
    stats={'observation.state':{'q01':torch.zeros(6),'q99':torch.ones(6)*100},
           'action':{'q01':torch.ones(6)*-10,'q99':torch.ones(6)*10}}
    if not statistics:
        stats=None
    import sentencepiece as spm
    corpus = tmp_path / 'corpus.txt'
    corpus.write_text('pick red green block put plate test state 0 1 2 3 4 5\n' * 20)
    spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(tmp_path / 'tokenizer'),
                                 vocab_size=48, hard_vocab_limit=False, minloglevel=2)
    tokenizer_path = tmp_path / 'tokenizer.model'
    tokenizer = SentencePieceTokenizer(tokenizer_path)
    def injected_tokenizer(**kwargs):
        return TokenizerProcessorStep(**{**kwargs,'tokenizer':tokenizer})
    with patch.object(processor_pi05,'TokenizerProcessorStep',side_effect=injected_tokenizer):
        pre,post=processor_pi05.make_pi05_pre_post_processors(cfg,stats)
    # Tokenizer is explicitly supplied on restore; this fixture serializes the
    # real training processors/stats without generating a new tokenizer export.
    with patch.object(TokenizerProcessorStep,'save_artifacts',return_value={}):
        pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    profile=Profile(name='fixture',artifact=str(tmp_path),allow_robot=True,
                    tokenizer_model=str(tokenizer_path))
    raw=json.loads((tmp_path/'config.json').read_text())
    return profile,raw


def test_saved_processors_restore_state_and_relative_chunk_anchor(tmp_path):
    profile,raw=processor_fixture(tmp_path)
    with patch('adapter.validate_artifact',return_value=(tmp_path,raw)):
        adapter=RobotAdapter(profile)
    state=np.asarray([10,20,30,40,50,60],dtype=np.float32)
    observation={name:float(x) for name,x in zip(JOINTS,state,strict=True)}
    observation.update(head=np.zeros((480,640,3),np.uint8),wrist=np.zeros((480,640,3),np.uint8))
    batch=adapter.prepare(observation,'pick_red_block')
    np.testing.assert_allclose(batch['observation.state'],(2*state/100-1)[None],atol=1e-6)
    assert batch['observation.images.head'].shape==(1,3,224,224)
    assert batch['observation.language.tokens'].shape==(1,200)
    chunk=adapter.decode(np.zeros((1,50,6),np.float32))
    np.testing.assert_allclose([[row[n] for n in JOINTS] for row in chunk],np.tile(state,(50,1)),atol=1e-5)
    # A new observation must anchor only the new chunk, not re-decode queued actions.
    observation.update({name:float(x+5) for name,x in zip(JOINTS,state,strict=True)})
    adapter.prepare(observation,'pick_red_block')
    new_chunk=adapter.decode(np.zeros((1,50,6),np.float32))
    np.testing.assert_allclose([new_chunk[0][n] for n in JOINTS],state+5,atol=1e-5)
    np.testing.assert_allclose([chunk[0][n] for n in JOINTS],state,atol=1e-5)


def test_missing_training_statistics_are_rejected(tmp_path):
    profile,raw=processor_fixture(tmp_path,statistics=False)
    with patch('adapter.validate_artifact',return_value=(tmp_path,raw)):
        with pytest.raises(ValueError,match='Missing training normalization'):
            RobotAdapter(profile)


def test_relative_rtc_prefix_uses_the_new_state_anchor(tmp_path):
    profile,raw=processor_fixture(tmp_path)
    with patch('adapter.validate_artifact',return_value=(tmp_path,raw)):
        adapter=RobotAdapter(profile)
    state=np.asarray([10,20,30,40,50,60],np.float32)
    observation=dict(zip(JOINTS,state.tolist()))
    observation.update(head=np.zeros((480,640,3),np.uint8),wrist=np.zeros((480,640,3),np.uint8))
    adapter.prepare(observation,'test')
    previous=adapter.decode(np.zeros((1,50,6),np.float32))
    observation.update(dict(zip(JOINTS,(state+5).tolist())))
    adapter.prepare(observation,'test')
    prefix=adapter.rtc_prefix(np.zeros((40,6),np.float32),previous[10:],observation)
    np.testing.assert_allclose(prefix,np.full((1,40,6),-.5,np.float32),atol=1e-6)
    np.testing.assert_allclose([previous[0][n] for n in JOINTS],state)








def test_robot_worker_disconnects_without_actions_on_inference_failure(tmp_path,monkeypatch):
    import robot_runner
    from unittest.mock import MagicMock
    robot=MagicMock()
    robot.calibration={'fixture':'test-only'}
    robot.is_calibrated=True
    robot.is_connected=True
    request={'profile':'fixture','follower_port':'/dev/test-only','follower_config':'test.json',
        'task':'test','duration_s':1,'session':'mock-test','cameras':{name:{'type':'opencv','camera_index':i,
            'width':640,'height':480,'fps':30} for i,name in enumerate(['head','wrist'])}}
    file=tmp_path/'request.json'
    file.write_text(json.dumps(request))
    monkeypatch.setattr(sys,'argv',['robot_runner.py','--request',str(file)])
    monkeypatch.setattr(robot_runner.signal,'signal',lambda *_:None)
    robot.get_observation.return_value=dict.fromkeys(JOINTS,0.)
    monkeypatch.setattr(robot_runner,'ControlReader',MagicMock())
    monkeypatch.setattr(robot_runner,'LatestMonitor',MagicMock())
    profile=Profile(name='fixture',artifact='test-only')
    client=MagicMock()
    client.infer.side_effect=TimeoutError('simulated inference failure')
    with patch.object(robot_runner,'read_profile',return_value=profile), \
         patch.object(robot_runner,'RobotAdapter'), \
         patch.object(robot_runner,'verify_follower_calibration',return_value={'selected':'test.json','motors_checked':6}), \
         patch.object(robot_runner,'ModelClient',return_value=client), \
         patch('lerobot.robots.so_follower.SO101Follower',return_value=robot), \
         patch('lelab.utils.config.setup_follower_calibration_file',return_value='fixture'):
        with pytest.raises(TimeoutError,match='simulated inference failure'):
            robot_runner.main()
    robot.send_action.assert_not_called()
    robot.disconnect.assert_not_called()
    robot.bus.disconnect.assert_called_once_with(disable_torque=False)


def test_offline_adapter_still_validates_contract_but_not_motion_permission(tmp_path):
    profile, raw = processor_fixture(tmp_path)
    profile = profile.model_copy(update={'allow_robot': False})
    with patch('adapter.validate_artifact', return_value=(tmp_path, raw)) as validate:
        RobotAdapter(profile, offline=True)
    validate.assert_called_once_with(profile, robot=True, offline=True)
