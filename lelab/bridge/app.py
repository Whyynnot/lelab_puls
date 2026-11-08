"""LeLab collection UI plus the local native PI05 FP8 deployment panel."""
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from adapter import ROOT, ModelClient, Profile, RobotAdapter, read_profile, read_sample_manifest, validate_artifact
from monitoring import ControlStore, ControlReader, LatestMonitor, RUNTIME
from robot_stop import RobotStopper, StopBlocked
from lelab.server import app
from lelab import record, rollout, teleoperate
from lelab import server as upstream_server
from native_startup import install_startup_retries

# Teleoperation and recording configure the follower in this web process;
# native rollout installs the same bounded setup-only behavior in its child.
install_startup_retries()

HERE = Path(__file__).resolve().parent
PROFILE_DIR, LOGS = ROOT/'lelab/profiles', ROOT/'lelab/logs'
for directory in [PROFILE_DIR, LOGS, RUNTIME]:
    directory.mkdir(parents=True, exist_ok=True)
controls = ControlStore()
_service = _service_log = None
_service_lock, _job_lock = threading.RLock(), threading.Lock()
_job = {'state': 'idle'}
_session = None
_robot_stopper = None


class Selection(BaseModel):
    profile: str
    steps: int = Field(default=10, ge=1, le=50)
    mode: Literal['sync', 'rtc'] | None = None


class ModeRequest(BaseModel):
    mode: Literal['sync', 'rtc']


class ViewerRequest(BaseModel):
    client_id: str = Field(pattern=r'^[a-zA-Z0-9_-]{1,64}$')
    enabled: bool = True
    images: bool = True


class RobotRequest(BaseModel):
    profile: str
    follower_port: str
    follower_config: str
    cameras: dict[str, dict]
    task: str = Field(min_length=1, max_length=1024)
    duration_s: int = Field(default=30, ge=1, le=3600)
    control_fps: int = Field(default=30, ge=5, le=60)


def fail(exc):
    raise HTTPException(400, detail=str(exc)) from exc


def service_state():
    try:
        return {'running': True, **ModelClient().health()}
    except Exception as exc:
        return {'running': False, 'error': str(exc), 'exit_code': _service.poll() if _service else None}


def robot_running():
    return bool(rollout._inference_proc and rollout._inference_proc.poll() is None)


def tail(path, limit=12000):
    if not path.exists():
        return ''
    with path.open('rb') as file:
        file.seek(0, 2)
        file.seek(max(0, file.tell()-limit))
        return file.read().decode('utf-8', 'replace')


def monitor_snapshot():
    try:
        snapshot = json.loads((RUNTIME/'monitor.json').read_text())
        snapshot['age_s'] = max(0, time.time()-snapshot['published_at'])
        snapshot['stale'] = snapshot['age_s'] > 3
    except (OSError, ValueError, KeyError):
        snapshot = {'phase': 'idle', 'stale': True, 'images': []}
    return {**snapshot, 'control': controls.get(), 'robot_active': robot_running()}


@app.get('/quantized', include_in_schema=False)
def page():
    return FileResponse(HERE/'panel.html')


@app.get('/api/deploy/status')
def status():
    return {'service': service_state(), 'offline_test': dict(_job),
            'rollout': rollout.handle_inference_status(), 'control': controls.get(),
            'robot_stop': _robot_stopper.snapshot() if _robot_stopper else None,
            'service_log': tail(LOGS/'model_service.log'), 'robot_log': tail(LOGS/'robot.log')}


@app.get('/api/deploy/profiles')
def profiles():
    return [Profile.model_validate_json(p.read_text()).model_dump() for p in sorted(PROFILE_DIR.glob('*.json'))]


@app.post('/api/deploy/profiles')
def save_profile(profile: Profile):
    try:
        validate_artifact(profile, robot=profile.allow_robot)
        if profile.allow_robot:
            RobotAdapter(profile)
        target = PROFILE_DIR/(profile.name+'.json')
        temporary = target.with_suffix('.tmp')
        temporary.write_text(profile.model_dump_json(indent=2))
        temporary.replace(target)
        return {'saved': profile.name}
    except Exception as exc:
        fail(exc)


@app.post('/api/deploy/service/start')
def start_service(request: Selection):
    global _service, _service_log
    with _service_lock:
        try:
            artifact, _ = validate_artifact(read_profile(request.profile))
            current = service_state()
            if current['running']:
                if current.get('artifact') == str(artifact) and current.get('steps') == request.steps:
                    return current
                raise ValueError('Unload the current model before changing artifact or steps')
            if _service and _service.poll() is None:
                return {'loading': True}
            if _service_log:
                _service_log.close()
            _service_log = (LOGS/'model_service.log').open('w')
            _service = subprocess.Popen([
                sys.executable, str(ROOT/'serve.py'), '--artifact', str(artifact),
                '--steps', str(request.steps), '--port', '8765'], cwd=ROOT,
                env={**os.environ, 'PYTHONUNBUFFERED': '1'}, stdout=_service_log,
                stderr=subprocess.STDOUT, start_new_session=True)
            return {'loading': True, 'pid': _service.pid}
        except Exception as exc:
            fail(exc)


@app.post('/api/deploy/service/stop')
def stop_service():
    if robot_running() or _job.get('state') == 'running':
        raise HTTPException(409, 'Stop the robot/offline test first')
    with _service_lock:
        if _service and _service.poll() is None:
            _service.terminate()
            try:
                _service.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _service.kill(); _service.wait(timeout=5)
        elif service_state()['running']:
            raise HTTPException(409, 'Service is owned by another terminal; stop it there')
    return {'stopped': True}


@app.post('/api/deploy/mode')
def set_mode(request: ModeRequest):
    return {'requested': controls.set_mode(request.mode),
            'note': 'Live worker discards old in-flight results, holds last target, then starts the new mode.'}


@app.post('/api/deploy/monitor')
def viewer(request: ViewerRequest):
    try:
        return controls.lease(request.client_id, request.enabled, request.images)
    except ValueError as exc:
        fail(exc)


@app.get('/api/deploy/monitor')
def monitor():
    return monitor_snapshot()


@app.get('/api/deploy/monitor/{camera}.jpg')
def monitor_image(camera: Literal['head', 'wrist']):
    path = RUNTIME/(camera+'.jpg')
    if not path.exists():
        raise HTTPException(404, 'No inference frame yet')
    return FileResponse(path, media_type='image/jpeg', headers={'Cache-Control': 'no-store'})


def run_offline(selection):
    global _job
    reader = publisher = None
    try:
        profile = read_profile(selection.profile)
        client = ModelClient()
        health = client.require_artifact(profile.artifact)
        manifest = read_sample_manifest(profile)
        sample = next(s for s in manifest['samples'] if s['split'] == 'evaluation')
        import hashlib
        sample_path = ROOT/sample['file']
        if hashlib.sha256(sample_path.read_bytes()).hexdigest() != sample['sha256']:
            raise ValueError('Offline observation checksum mismatch')
        with np.load(sample_path, allow_pickle=False) as arrays:
            batch = {k: arrays[k].copy() for k in arrays.files if k not in {'raw_state', 'demo_actions'}}
            state = arrays['raw_state'].reshape(-1).tolist()
        batch['noise'] = np.zeros((1, health['chunk_size'], health['max_action_dim']), np.float32)
        frames = {name: (np.moveaxis(batch[key][0], 0, -1)*255).clip(0, 255).astype(np.uint8)
                  for name, key in profile.camera_map.items()}
        adapter = RobotAdapter(profile, offline=True)
        if adapter.config.use_relative_actions:
            observation = {**dict(zip(profile.state_names, state, strict=True)), **frames}
            # Restore the current-state anchor without changing the saved
            # preprocessed tensors used for inference.
            adapter.prepare(observation, sample['task'])
        mode = selection.mode or controls.get()['mode']
        started = time.perf_counter()
        actions, elapsed = client.infer(batch, {'enabled': mode == 'rtc'})
        initial = client.last_result
        if mode == 'rtc':
            actions, elapsed = client.infer(batch, {'enabled': True, 'previous': actions[:, 10:],
                                                    'delay': 15, 'horizon': health['chunk_size']-10})
        physical = adapter.decode(actions)
        reader = ControlReader()
        publisher = LatestMonitor(reader)
        publisher.offer({'phase': 'offline_saved_observation', 'source': 'offline_saved_observation',
                         'profile': profile.name, 'task': sample['task'], 'mode': mode,
                         'state': dict(zip(profile.state_names, state, strict=True)),
                         'state_observed_at': time.time(), 'target': {}, 'predicted_chunk': physical,
                         'inference_ms': elapsed, 'buffer_remaining': 0, 'actions_sent': 0}, frames)
        publisher.publish()
        _job = {'state': 'passed', 'kind': 'saved_observation_no_robot', 'sample': sample['id'],
                'profile': profile.name, 'artifact': profile.artifact, 'dataset': manifest['dataset'],
                'mode': mode, 'action_shape': list(actions.shape), 'finite': bool(np.isfinite(actions).all()),
                'inference_ms': elapsed, 'total_ms': 1000*(time.perf_counter()-started),
                'guidance_applied': client.last_result.get('guidance_applied', False),
                'first_call': initial, 'last_call': client.last_result,
                'steps': health['steps'], 'format': health['format']}
        (LOGS/'offline_test.json').write_text(json.dumps(_job, indent=2))
    except Exception as exc:
        _job = {'state': 'failed', 'error': str(exc)}
    finally:
        if publisher:
            publisher.close()
        if reader:
            reader.close()


@app.post('/api/deploy/offline-test')
def offline_test(request: Selection):
    global _job
    with _job_lock:
        if _job.get('state') == 'running' or robot_running():
            raise HTTPException(409, 'A test or robot rollout is active')
        _job = {'state': 'running'}
        threading.Thread(target=run_offline, args=(request,), daemon=True).start()
    return dict(_job)


def monitor_robot(process, log_handle):
    try:
        rollout._pump_stdout(process, log_handle)
    finally:
        process.wait()
        if not log_handle.closed:
            log_handle.close()
        with rollout._state_lock:
            if rollout._inference_proc is process:
                rollout.inference_active = False


@app.post('/api/deploy/robot/start')
def start_robot(request: RobotRequest):
    global _session, _robot_stopper
    try:
        profile = read_profile(request.profile)
        RobotAdapter(profile)
        ModelClient().require_artifact(profile.artifact)
        if set(request.cameras) != {'head', 'wrist'}:
            raise ValueError('Both head and wrist cameras are required')
        from lelab.camera_config import validate_camera_rotations
        validate_camera_rotations(request.cameras)
        if not Path(request.follower_port).exists():
            raise ValueError('Follower serial device is not visible')
        if not request.follower_config or Path(request.follower_config).name != request.follower_config:
            raise ValueError('Select a calibration filename')
        from lelab.utils.config import FOLLOWER_CONFIG_PATH
        if not (Path(FOLLOWER_CONFIG_PATH)/request.follower_config).is_file():
            raise ValueError('Follower calibration does not exist')
        with record._state_lock, teleoperate._state_lock, rollout._state_lock:
            if record.recording_active or teleoperate.teleoperation_active or rollout.inference_active or robot_running() or _job.get('state') == 'running':
                raise HTTPException(409, 'Stop recording, teleoperation and other inference/tests first')
            _session = uuid.uuid4().hex
            file = LOGS/'robot_request.json'
            file.write_text(json.dumps({**request.model_dump(), 'session': _session}))
            log = (LOGS/'robot.log').open('w')
            process = subprocess.Popen([sys.executable, str(HERE/'robot_runner.py'), '--request', str(file)],
                cwd=HERE, env={**os.environ, 'PYTHONUNBUFFERED': '1'}, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, start_new_session=True)
            rollout._inference_proc, rollout.inference_active = process, True
            rollout._inference_started_at = time.time()
            rollout._inference_rollout_started_at = None
            rollout._inference_meta = {'duration_s': request.duration_s, 'log_path': str(LOGS/'robot.log')}
            _robot_stopper = RobotStopper(process)
            threading.Thread(target=monitor_robot, args=(process, log), daemon=True).start()
            threading.Thread(target=_robot_stopper.watch_fault, args=(LOGS/'robot.log',), daemon=True).start()
            return {'started': True, 'pid': process.pid, 'session': _session}
    except HTTPException:
        raise
    except Exception as exc:
        fail(exc)


@app.post('/api/deploy/robot/stop')
def stop_robot():
    global _robot_stopper
    process = rollout._inference_proc
    if process:
        if _robot_stopper is None or _robot_stopper.process is not process:
            _robot_stopper = RobotStopper(process)
        try:
            result = _robot_stopper.stop()
        except StopBlocked as exc:
            raise HTTPException(503, str(exc)) from exc
        # Reconcile immediately; the stdout monitor thread need not win this race.
        rollout.handle_inference_status()
        return result
    return {'stopped': True}


_native_start = upstream_server.handle_start_inference
def guarded_native_start(request):
    if (Path(request.policy_ref)/'quantization.json').is_file():
        return {'success': False, 'status_code': 400, 'message': 'Use the FP8 deployment page: /quantized'}
    return _native_start(request)
upstream_server.handle_start_inference = guarded_native_start
_native_resolver = rollout._resolve_policy_path
def guarded_native_resolver(reference):
    path = _native_resolver(reference)
    if (Path(path)/'quantization.json').is_file():
        raise ValueError('Use the FP8 deployment page: /quantized')
    return path
rollout._resolve_policy_path = guarded_native_resolver
_native_stop = upstream_server.handle_stop_inference
def guarded_native_stop():
    process = rollout._inference_proc
    if process and str(HERE/'robot_runner.py') in process.args:
        return {'success': True, **stop_robot()}
    return _native_stop()
upstream_server.handle_stop_inference = guarded_native_stop


@app.on_event('shutdown')
def shutdown_extension():
    if rollout.inference_active and rollout._inference_proc is None:
        rollout.handle_stop_inference()
    stop_robot()
    if _service and _service.poll() is None:
        _service.terminate()
        _service.wait(timeout=10)


from native_api import install_native_api
install_native_api(app, rollout, upstream_server)

if sys.platform.startswith('linux'):
    import wsl_hardware  # Linux/WSL V4L2 previews and configured-device preflight.
app.router.routes.sort(key=lambda route: getattr(route, 'name', None) == 'frontend')
