from pathlib import Path
import io
import json
from types import SimpleNamespace
import numpy as np
import pytest
from fastapi.testclient import TestClient
from adapter import ROOT, Profile, RobotAdapter, validate_artifact


@pytest.mark.local_artifact
def test_offline_profile_cannot_reach_robot_validation():
    profile=Profile(name='test',artifact=__import__('os').environ.get('LELAB_TEST_ARTIFACT',str(ROOT/'artifacts/my_pi05_fp8')),allow_robot=False)
    with pytest.raises(ValueError,match='offline only'):
        validate_artifact(profile,robot=True)


def test_mode_button_api_changes_control_without_starting_robot(tmp_path,monkeypatch):
    import app as web
    from monitoring import ControlStore
    store=ControlStore(tmp_path/'control.json')
    monkeypatch.setattr(web,'controls',store)
    client=TestClient(web.app)
    assert client.get('/quantized').status_code==200
    assert 'id="rtc"' in client.get('/quantized').text
    for mode in ['rtc','sync']:
        response=client.post('/api/deploy/mode',json={'mode':mode})
        assert response.status_code==200 and response.json()['requested']['mode']==mode
    assert client.post('/api/deploy/mode',json={'mode':'invalid'}).status_code==422
    assert not web.robot_running()


def test_monitor_api_uses_saved_frames_without_opening_camera(tmp_path,monkeypatch):
    import app as web
    from monitoring import ControlStore
    monkeypatch.setattr(web,'controls',ControlStore(tmp_path/'control.json'))
    monkeypatch.setattr(web,'RUNTIME',tmp_path)
    (tmp_path/'monitor.json').write_text(json.dumps({'published_at':0,'images':['head'],'state':{'x':12}}))
    (tmp_path/'head.jpg').write_bytes(b'test-only-saved-image')
    client=TestClient(web.app)
    assert client.post('/api/deploy/monitor',json={'client_id':'test','enabled':True,'images':False}).status_code==200
    assert client.get('/api/deploy/monitor').json()['state']=={'x':12}
    assert client.get('/api/deploy/monitor/head.jpg').content==b'test-only-saved-image'
    assert client.get('/api/deploy/monitor/unknown.jpg').status_code==422


@pytest.mark.local_artifact
def test_service_validates_rtc_prefix_dtype_shape_and_bounds():
    from serve import decode
    adapter=RobotAdapter(Profile(name='test',artifact=__import__('os').environ.get('LELAB_TEST_ARTIFACT',str(ROOT/'artifacts/my_pi05_fp8')),allow_robot=True))
    policy=SimpleNamespace(config=adapter.config)
    manifest=json.loads((Path(__import__('os').environ.get('LELAB_TEST_MANIFEST',str(ROOT/'validation/sample_manifest.json')))).read_text())
    sample=next(s for s in manifest['samples'] if s['split']=='evaluation')
    with np.load(ROOT/sample['file']) as arrays:
        batch={k:arrays[k].copy() for k in arrays.files if k not in {'raw_state','demo_actions'}}
    def payload(data):
        stream=io.BytesIO();np.savez(stream,**data);return stream.getvalue()
    prefix={**batch,'rtc_enabled':np.asarray(True),'prev_chunk_left_over':np.zeros((1,40,6),np.float32),
            'inference_delay':np.asarray(15,np.int64),'execution_horizon':np.asarray(40,np.int64)}
    _,_,rtc=decode(payload(prefix),policy,device='cpu')
    assert rtc['enabled'] and rtc['previous'].shape==(1,40,6)
    with pytest.raises(ValueError,match='out of bounds'):
        decode(payload({**prefix,'inference_delay':np.asarray(40,np.int64)}),policy,device='cpu')
    with pytest.raises(ValueError,match='Invalid normalized RTC'):
        decode(payload({**prefix,'prev_chunk_left_over':np.zeros((1,40,32),np.float32)}),policy,device='cpu')
    with pytest.raises(ValueError,match='Missing policy'):
        decode(payload({k:v for k,v in batch.items() if not k.startswith('observation.images.')}),policy,device='cpu')
