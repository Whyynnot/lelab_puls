"""Repair USB routing while idle and check routing before user-started robot work."""
import asyncio
import copy
import json
import logging
from pathlib import Path
import time

from fastapi.responses import JSONResponse
from prepare_hardware import DEVICES, ROOT, discover, prepare
from serial_preflight import probe_serial_port

LOG = logging.getLogger(__name__)
RUNTIME_REPORT = ROOT/'lelab/logs/hardware_runtime.json'
START_PATHS = {'/move-arm', '/start-recording', '/start-inference',
               '/start-calibration', '/api/deploy/robot/start'}


def required_roles(path, body):
    if path == '/start-recording' and body.get('test_mode') is True:
        return set()
    if path == '/start-calibration':
        return {'teleop': {'leader'}, 'robot': {'follower'}}.get(body.get('device_type'), set())
    roles = {'leader','follower'} if path in ('/move-arm','/start-recording') else {'follower'}
    cameras=body.get('cameras', {})
    if isinstance(cameras,dict):
        roles.update(name for name in cameras if name in ('head','wrist'))
    return roles


def remap_cameras(body, devices):
    body=copy.deepcopy(body)
    # ttyACM numbers change after reconnects. Preserve the selected calibration,
    # but route the known leader/follower identities through stable device names.
    for role in ('leader', 'follower'):
        device = devices.get(role, {})
        if device.get('accessible') and device.get('path'):
            if role + '_port' in body:
                body[role + '_port'] = device['path']
            if 'port' in body and body.get('device_type') == {'leader': 'teleop', 'follower': 'robot'}[role]:
                body['port'] = device['path']
    cameras=body.get('cameras', {})
    if not isinstance(cameras,dict):
        return body
    for role, camera in cameras.items():
        if isinstance(camera,dict) and role in ('head','wrist') and devices.get(role, {}).get('accessible') and 'index' in devices[role]:
            camera['camera_index']=devices[role]['index']
            # device_id is UI metadata, not a LeRobot CLI camera field.
            if 'device_id' in camera:
                camera['device_id']=f'wsl-camera:{role}'
    return body


def busy():
    from lelab import record, rollout, teleoperate
    from lelab.calibrate import calibration_manager
    return (record.recording_active or rollout.inference_active or teleoperate.teleoperation_active
            or calibration_manager.get_status().calibration_active)


class Controller:
    def __init__(self):
        self.gate=asyncio.Lock()
        self.last_attempt=0.0
        self.last_error=None
        self.task=None
        self.recovering=False

    async def repair(self, reason, roles=None, wait_seconds=12):
        self.recovering=True
        self.last_attempt=time.time()
        try:
            report=await asyncio.to_thread(prepare, wait_seconds, RUNTIME_REPORT, reason, roles)
            self.last_error='; '.join(report['issues']) or None
            return report
        finally:
            self.recovering=False

    async def watch(self):
        while True:
            await asyncio.sleep(8)
            try:
                async with self.gate:
                    if busy():
                        continue
                    if time.time()-self.last_attempt >= 20 and any(not d['accessible'] for d in discover().values()):
                        await self.repair('idle USB identity recovery',wait_seconds=3)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error=str(exc)
                LOG.exception('Idle USB recovery failed')


controller=Controller()


class HardwarePreflight:
    def __init__(self, app):
        self.app=app

    async def __call__(self, scope, receive, send):
        if scope['type']!='http' or scope['method']!='POST' or scope['path'] not in START_PATHS:
            await self.app(scope, receive, send)
            return
        chunks=[]
        while True:
            message=await receive()
            if message['type']!='http.request':
                return
            chunks.append(message.get('body',b''))
            if not message.get('more_body',False):
                break
        raw=b''.join(chunks)
        try:
            body=json.loads(raw)
            if not isinstance(body,dict): raise ValueError('Expected a JSON object')
            roles=required_roles(scope['path'],body) & {role for role, path in DEVICES.items() if path}
            # Reject a packed model via the native loader guard before any USB
            # recovery. This request cannot start a valid native rollout.
            if scope['path']=='/start-inference' and (Path(str(body.get('policy_ref','')))/'quantization.json').is_file():
                roles=set()
        except (ValueError,TypeError):
            body=None
            roles=set()
        async with controller.gate:
            if roles:
                if busy():
                    detail='Stop the active calibration/teleoperation/recording/inference session first.'
                    await JSONResponse({'success':False,'message':detail,'detail':detail},status_code=409)(scope,receive,send)
                    return
                devices=discover()
                if any(not devices[role]['accessible'] for role in roles):
                    await controller.repair('robot start preflight: '+scope['path'],roles=roles)
                    devices=discover()
                missing=[role for role in sorted(roles) if not devices[role]['accessible']]
                if missing:
                    detail=('WSL USB device unavailable: '+', '.join(missing)+'. '
                            'Automatic identity recovery was attempted. '+(controller.last_error or '')+
                            ' Check /api/deploy/hardware-status; robot motion was not started.')
                    await JSONResponse({'success':False,'message':detail,'detail':detail},status_code=503)(scope,receive,send)
                    return
                body=remap_cameras(body,devices)
                try:
                    for role in sorted(roles & {'leader', 'follower'}):
                        port=body.get(role+'_port')
                        if not port and 'port' in body:
                            port=body['port']
                        if port and devices[role].get('path'):
                            await asyncio.to_thread(probe_serial_port, port)
                except (TimeoutError, ConnectionError) as exc:
                    detail=str(exc)
                    await JSONResponse({'success':False,'message':detail,'detail':detail},status_code=503)(scope,receive,send)
                    return
                raw=json.dumps(body).encode()
            updated=dict(scope)
            updated['headers']=[(k,v) for k,v in scope['headers'] if k.lower()!=b'content-length']+[(b'content-length',str(len(raw)).encode())]
            delivered=False
            async def replay():
                nonlocal delivered
                if not delivered:
                    delivered=True
                    return {'type':'http.request','body':raw,'more_body':False}
                return await receive()
            await self.app(updated,replay,send)


def install(app):
    app.add_middleware(HardwarePreflight)
    @app.post('/api/deploy/hardware/recover')
    async def recover_without_motion():
        async with controller.gate:
            if busy():
                return JSONResponse({'success':False,'message':'Stop the active robot session first.'},status_code=409)
            return await controller.repair('manual USB recovery; no motor commands')
    @app.on_event('startup')
    async def start_watchdog():
        controller.task=asyncio.create_task(controller.watch(),name='idle-usb-identity-watchdog')
    @app.on_event('shutdown')
    async def stop_watchdog():
        if controller.task:
            controller.task.cancel()
            try:
                await controller.task
            except asyncio.CancelledError:
                pass
