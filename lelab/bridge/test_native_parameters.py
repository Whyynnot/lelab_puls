"""Native engines and simulated devices; never opens a real serial port/camera."""
import copy
import io
import json
import threading
import time
from collections import deque
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch

from native_parameters import describe_parameters, validate_parameters, RuntimeParameters
from native_session import WebSession
from test_native_session import context, wait_for, KEYS, FEATURES, Processor


def test_admission_limits_preserve_trained_structure_and_native_rtc():
    act = describe_parameters({'type':'act','chunk_size':100,'n_action_steps':50})
    assert validate_parameters({'n_action_steps':10,'fps':20},act)['chunk_size'] == 100
    with pytest.raises(ValueError,match='fixed trained'):
        validate_parameters({'chunk_size':50},act)
    smol = describe_parameters({'type':'smolvla','chunk_size':50,'n_action_steps':50})
    assert validate_parameters({'chunk_size':20,'n_action_steps':10,'fps':15},smol)['chunk_size'] == 20
    for invalid in ({'chunk_size':51}, {'fps':float('nan')}, {'fps':True}, {'n_action_steps':0}, {'chunk_size':5,'n_action_steps':6}):
        with pytest.raises(ValueError): validate_parameters(invalid,smol)
    rtc = describe_parameters({'type':'pi05','chunk_size':50,'n_action_steps':10,'use_relative_actions':True})
    assert rtc['effective']['n_action_steps'] == 50
    with pytest.raises(ValueError,match='N must equal M'):
        validate_parameters({'n_action_steps':10},rtc)


class ChunkPolicy:
    def __init__(self):
        from lerobot.configs.types import PolicyFeature, FeatureType
        self.config = NS(type='smolvla',use_amp=False,chunk_size=8,n_action_steps=4,n_obs_steps=1,
                         rtc_training_max_delay=0,temporal_ensemble_coeff=None,
                         action_feature=PolicyFeature(type=FeatureType.ACTION,shape=(6,)))
        self.model = NS(config=self.config)
        self.shapes = []
        self.reset()
    def reset(self): self.queue = deque(maxlen=self.config.n_action_steps)
    def drop_queued_actions(self): self.queue.clear()
    def predict_action_chunk(self,batch,**kwargs):
        chunk = torch.arange(1,self.config.chunk_size+1,dtype=torch.float32).reshape(1,-1,1).expand(1,-1,6).clone()
        self.shapes.append(tuple(chunk.shape))
        return chunk
    def select_action(self,batch):
        if not self.queue:
            self.queue.extend(self.predict_action_chunk(batch)[0,:self.config.n_action_steps])
        return self.queue.popleft().unsqueeze(0)
    def supports_rtc(self): return True
    def supports_text_generation(self): return False
    def chunk_inference_spec(self):
        from lerobot.policies.pretrained import PreTrainedPolicy
        return PreTrainedPolicy.chunk_inference_spec(self)


@pytest.mark.parametrize('backend',['sync','rtc_chunk','rtc_guided'])
def test_parameters_take_effect_after_reset_using_native_engines(tmp_path,monkeypatch,backend):
    from lerobot.inference.factory import SyncInferenceConfig, RTCInferenceConfig
    from lerobot.inference.rtc import RTCInferenceEngine
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.rollout.strategies.base import CycleTimer
    from lerobot.inference import rtc
    monkeypatch.setattr(rtc,'prepare_observation_for_inference',rtc.prepare_observation_for_inference)
    monkeypatch.setenv('LELAB_NATIVE_SESSION_DIR',str(tmp_path/('f'*32)))
    monkeypatch.delenv('LELAB_NATIVE_SOURCE_CONFIG',raising=False)
    ctx,strategy,robot,_ = context()
    policy = ChunkPolicy()
    ctx.policy.inference._policy = policy
    ctx.policy.policy = policy
    processor = Processor(); processor.steps = ()
    ctx.policy.preprocessor = ctx.policy.postprocessor = processor
    cfg = ctx.runtime.cfg
    cfg.device='cpu';cfg.fps=30;cfg.compile_warmup_inferences=2
    cfg.inference = SyncInferenceConfig()
    if backend != 'sync':
        cfg.inference=RTCInferenceConfig(rtc=RTCConfig(enabled=backend=='rtc_guided',execution_horizon=2),queue_threshold=0)
        engine=RTCInferenceEngine(policy,processor,processor,ctx.hardware.robot_wrapper,
              cfg.inference.rtc,FEATURES,'original instruction',30,'cpu',rtc_queue_threshold=0)
        ctx.policy.inference=strategy._engine=engine
        engine.start()
    original_engine=ctx.policy.inference
    with patch('lerobot.rollout.strategies.core.precise_sleep',lambda _:time.sleep(.0001)):
        web=WebSession(strategy,ctx)
        thread=threading.Thread(target=web.run);thread.start()
        try:
            wait_for(lambda:web.phase=='running' and len(robot.sent)>2)
            # No runtime mutation while running.
            with pytest.raises(ValueError,match='Cannot resume'):
                web.command({'session_id':web.directory.name,'action':'resume','parameters':{'fps':20}})
            web.command({'session_id':web.directory.name,'action':'reset'})
            wait_for(lambda:web.phase=='paused')
            # Invalid parameters cannot alter the prompt or config.
            with pytest.raises(ValueError):
                web.command({'session_id':web.directory.name,'action':'resume','task':'new instruction',
                             'parameters':{'chunk_size':9}})
            assert web.controller.task=='original instruction' and cfg.fps==30
            if backend=='rtc_chunk':
                from lerobot.inference import factory
                original_factory=factory.create_inference_engine
                attempts=[]
                def fail_first(*args,**kwargs):
                    attempts.append(1)
                    if len(attempts)==1: raise ValueError('test-only replacement error')
                    return original_factory(*args,**kwargs)
                with patch.object(factory,'create_inference_engine',side_effect=fail_first):
                    with pytest.raises(ValueError,match='test-only replacement'):
                        web.command({'session_id':web.directory.name,'action':'resume',
                          'parameters':{'chunk_size':6,'n_action_steps':2,'fps':20}})
                assert cfg.fps==30 and policy.config.chunk_size==8
                assert web.phase=='paused' and ctx.policy.inference._rtc_thread.is_alive()
                original_engine=ctx.policy.inference
            robot.sent.clear()
            with patch('lerobot.rollout.strategies.base.CycleTimer',wraps=CycleTimer) as timer:
                requested={'chunk_size':6,'n_action_steps':6 if backend=='rtc_guided' else 2,'fps':20}
                web.command({'session_id':web.directory.name,'action':'resume','task':'new instruction','parameters':requested})
                wait_for(lambda:len(robot.sent)>=6)
                assert timer.call_args.args[0]==20
            assert policy.shapes[-1]==(1,6,6)
            assert cfg.fps==20 and policy.config.n_action_steps==requested['n_action_steps']
            assert web.monitor.state['parameters']['effective']==requested
            if backend=='sync':
                assert [item[KEYS[0]] for item in robot.sent[:6]]==[1,2,1,2,1,2]
            else:
                assert original_engine is not ctx.policy.inference
                assert ctx.policy.inference._runtime.interval==pytest.approx(.05)
            # A subsequent reset retains effective settings.
            web.command({'session_id':web.directory.name,'action':'reset'})
            wait_for(lambda:web.phase=='paused')
            assert web.parameters.describe()['effective']==requested
        finally:
            ctx.runtime.shutdown_event.parent.set();thread.join(5)
            ctx.policy.inference.stop()
            assert not thread.is_alive()


def test_api_rejects_invalid_parameters_before_enqueuing(tmp_path,monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import native_api
    from native_session import atomic_json,read_json
    directory=tmp_path/('a'*32);directory.mkdir()
    monkeypatch.setattr(native_api,'SESSIONS',tmp_path)
    process=NS(args=['python',str(native_api.HERE/'native_runner.py')],poll=lambda:None)
    rollout=NS(_state_lock=threading.Lock(),_inference_proc=process,_inference_meta={'native_session_dir':str(directory)})
    upstream=NS(handle_inference_status=lambda:{'inference_active':True})
    app=FastAPI();native_api.install_native_api(app,rollout,upstream)
    client=TestClient(app)
    atomic_json(directory/'status.json',{'phase':'paused','parameters':describe_parameters({'type':'smolvla','chunk_size':50,'n_action_steps':50})})
    request={'session_id':directory.name,'action':'resume','parameters':{'chunk_size':20,'n_action_steps':21,'fps':20}}
    assert client.post('/api/native-inference/command',json=request).status_code==400
    assert not (directory/'command.json').exists()
    request['parameters']['n_action_steps']=10
    assert client.post('/api/native-inference/command',json=request).status_code==202
    assert read_json(directory/'command.json')['parameters']==request['parameters']


@pytest.mark.parametrize('mode',['auto','sync','rtc'])
def test_startup_overrides_reach_native_cli_without_editing_checkpoint(tmp_path,monkeypatch,mode):
    from lelab import rollout,record,teleoperate
    for key,value in {'inference_active':False,'_inference_proc':None,'_inference_meta':{},
                       '_inference_started_at':None,'_inference_rollout_started_at':None}.items():
        monkeypatch.setattr(rollout,key,value)
    monkeypatch.setattr(record,'recording_active',False)
    monkeypatch.setattr(teleoperate,'teleoperation_active',False)
    monkeypatch.setattr(rollout,'setup_follower_calibration_file',lambda _:'existing')
    monkeypatch.setattr(rollout,'_prepare_policy_path',lambda *args:str(tmp_path))
    monkeypatch.setattr(rollout,'_detect_device',lambda:'cpu')
    monkeypatch.setattr(rollout,'_pump_stdout',lambda process,log:log.close())
    monkeypatch.setenv('LELAB_NATIVE_RUNNER','/test/native_runner.py')
    monkeypatch.setenv('LELAB_NATIVE_SESSIONS',str(tmp_path/'sessions'))
    config={'type':'smolvla','chunk_size':50,'n_action_steps':50,
            'input_features':{'observation.images.head':{'type':'VISUAL','shape':[3,480,640]}}}
    path=tmp_path/'config.json';path.write_text(json.dumps(config));original=path.read_bytes()
    captured={}
    def popen(command,**kwargs):
        captured.update(command=command,**kwargs)
        return NS(pid=1,stdin=io.BytesIO(),poll=lambda:None)
    monkeypatch.setattr(rollout.subprocess,'Popen',popen)
    result=rollout.handle_start_inference(rollout.InferenceRequest(follower_port='/dev/fake',follower_config='saved.json',
          policy_ref=str(tmp_path),cameras={'head':{'camera_index':0}},inference_mode=mode,
          inference_parameters={'chunk_size':20,'n_action_steps':20 if mode=='rtc' else 10,'fps':15}))
    assert result['success']
    assert all(flag in captured['command'] for flag in ['--fps=15','--policy.chunk_size=20',
        '--policy.n_action_steps=20' if mode=='rtc' else '--policy.n_action_steps=10'])
    if mode=='rtc':
        assert '--inference.type=rtc' in captured['command']
        assert '--inference.rtc.enabled=true' in captured['command']
        assert '--inference.rtc.execution_horizon=19' in captured['command']
    elif mode=='sync':
        assert '--inference.type=sync' in captured['command']
    assert json.loads(captured['env']['LELAB_NATIVE_SOURCE_CONFIG'])==config
    assert path.read_bytes()==original
