"""Serve WSL camera previews to a Windows browser without getUserMedia."""
import asyncio
import threading
import logging

from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from lelab import record, rollout, teleoperate
from lelab.server import app
from hardware_runtime import RUNTIME_REPORT, controller, install
from camera_preview import CameraPreview
from camera_inventory import cameras, resolve_camera

install(app)

_locks={}
_workers={}
# The WSL V4L2 bridge can return EBUSY when separate cameras negotiate
# formats concurrently. Serialize initialization only; streams stay parallel.
_initialization_lock=asyncio.Lock()
LOG=logging.getLogger(__name__)


@app.get('/api/deploy/hardware-status')
def hardware_status():
    import json
    from prepare_hardware import REPORT, discover
    startup = json.loads(REPORT.read_text()) if REPORT.exists() else None
    runtime = json.loads(RUNTIME_REPORT.read_text()) if RUNTIME_REPORT.exists() else None
    current = discover()
    return {'startup': startup, 'runtime': runtime,
            'watchdog': {'enabled': True, 'task_running': bool(controller.task and not controller.task.done()),
                         'recovering': controller.recovering,
                         'last_attempt': controller.last_attempt, 'last_error': controller.last_error},
            'current_devices': current,
            'current_devices_accessible': all(d['accessible'] for d in current.values()),
            'note': 'USB presence and access only; not a motor-power or camera-frame test.'}


@app.get('/api/deploy/cameras')
def list_cameras():
    return {'cameras':cameras(),'preview_backend':'wsl'}


@app.get('/api/deploy/camera/{identity}/preview')
async def preview(identity:str,request:Request):
    camera=resolve_camera(identity,cameras())
    if camera is None:
        raise HTTPException(404,'Camera is disconnected or its USB forwarding is unavailable')
    index=camera['index']
    key=camera['deviceId']
    if record.recording_active or rollout.inference_active or teleoperate.teleoperation_active:
        raise HTTPException(409,'Preview is paused during robot sessions')
    lock=_locks.setdefault(key,threading.Lock())
    if not lock.acquire(blocking=False):
        raise HTTPException(409,'This camera already has a preview; close the other preview')
    previous=_workers.get(key)
    worker=previous[1] if previous else None
    if previous and previous[0]==camera['generation'] and worker.process.returncode is None:
        lock.release()
        raise HTTPException(503,'Previous camera connection is still waiting for USB recovery')
    try:
        async with _initialization_lock:
            worker=await CameraPreview.start(camera['devicePath'])
            _workers[key]=(camera['generation'],worker)
            first_frame=await worker.read_frame(timeout=8)
    except (OSError, asyncio.TimeoutError, asyncio.IncompleteReadError, ValueError) as exc:
        try:
            if worker is not None:
                await asyncio.shield(worker.close())
        finally:
            lock.release()
        LOG.warning('Camera %s preview could not start: %s',index,type(exc).__name__)
        raise HTTPException(503,'Camera did not return a frame; reconnect its USB forwarding and retry') from exc
    except BaseException:
        try:
            if worker is not None:
                await asyncio.shield(worker.close())
        finally:
            lock.release()
        raise
    async def frames():
        try:
            jpeg=first_frame
            while not await request.is_disconnected():
                if record.recording_active or rollout.inference_active or teleoperate.teleoperation_active:
                    break
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+jpeg+b'\r\n'
                jpeg=await worker.read_frame()
        except (asyncio.TimeoutError, asyncio.IncompleteReadError, ValueError):
            LOG.warning('Camera %s preview stopped receiving frames',index)
        finally:
            try:
                await asyncio.shield(worker.close())
            finally:
                lock.release()
    return StreamingResponse(frames(),media_type='multipart/x-mixed-replace; boundary=frame',
                             headers={'Cache-Control':'no-store'})
