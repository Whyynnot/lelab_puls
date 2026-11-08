from pathlib import Path
import pytest
import io
import json
from types import SimpleNamespace
import numpy as np
from fastapi.testclient import TestClient
from adapter import ROOT,Profile,RobotAdapter,validate_artifact


@pytest.mark.local_artifact
def test_real_fp8_processors_and_http_input_contract():
    from serve import decode
    profile=Profile(name='test',artifact=__import__('os').environ.get('LELAB_TEST_ARTIFACT',str(ROOT/'artifacts/my_pi05_fp8')),allow_robot=True)
    path,cfg=validate_artifact(profile,robot=True)
    adapter=RobotAdapter(profile)
    manifest=json.loads((Path(__import__('os').environ.get('LELAB_TEST_MANIFEST',str(ROOT/'validation/sample_manifest.json')))).read_text())
    sample=next(s for s in manifest['samples'] if s['split']=='evaluation')
    with np.load(ROOT/sample['file']) as data:
        values={k:data[k].copy() for k in data.files if k not in {'raw_state','demo_actions'}}
    model=SimpleNamespace(config=adapter.config)
    payload=io.BytesIO();np.savez(payload,**values)
    batch,noise,rtc=decode(payload.getvalue(),model,device='cpu')
    assert batch['observation.state'].shape==(1,6) and noise is None
    assert set(profile.camera_map.values()).issubset(batch)
    chunk=adapter.decode(np.zeros((1,50,6),np.float32))
    assert len(chunk)==50 and len(chunk[0])==6


def test_fp8_page_and_standard_loader_guard_before_hardware(monkeypatch,tmp_path):
    (tmp_path/"quantization.json").write_text("{}")
    from app import app
    from hardware_runtime import controller
    async def forbidden_repair(*args,**kwargs):
        raise AssertionError('Packed-loader rejection must not touch USB')
    monkeypatch.setattr(controller,'repair',forbidden_repair)
    client=TestClient(app)
    assert client.get('/quantized').status_code==200
    assert 'FP8 W8A8' in client.get('/quantized').text
    response=client.post('/start-inference',json={
        'policy_ref':str(tmp_path),'follower_port':'/dev/not-connected',
        'follower_config':'none.json','task':'test','cameras':{}})
    assert response.status_code==400 and '/quantized' in response.text
