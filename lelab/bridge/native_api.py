"""Native model session API. Deliberately isolated from /api/deploy FP8 state."""
from __future__ import annotations

import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Literal

from fastapi import HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from native_session import atomic_json, read_json

HERE = Path(__file__).resolve().parent
SESSIONS = HERE.parent / 'runtime' / 'native'
_command_lock = threading.Lock()


class SessionCommand(BaseModel):
    session_id: str = Field(pattern=r'^[a-f0-9]{32}$')
    action: Literal['reset', 'resume']
    task: str | None = Field(default=None, max_length=4096)
    parameters: dict | None = None


class NativeViewer(BaseModel):
    session_id: str = Field(pattern=r'^[a-f0-9]{32}$')
    enabled: bool = True
    images: bool = True


def install_native_api(app, rollout, upstream_server):
    os.environ['LELAB_NATIVE_RUNNER'] = str(HERE / 'native_runner.py')
    os.environ['LELAB_NATIVE_SESSIONS'] = str(SESSIONS)

    def current(session_id=None):
        with rollout._state_lock:
            process = rollout._inference_proc
            directory = rollout._inference_meta.get('native_session_dir')
            if (not directory or not process or process.poll() is not None
                    or str(HERE / 'native_runner.py') not in process.args):
                raise HTTPException(409, 'No active native model session')
            directory = Path(directory)
            if directory.parent != SESSIONS or (session_id and directory.name != session_id):
                raise HTTPException(409, 'Session changed; refresh before sending controls')
            return directory

    def read_status(directory):
        state = read_json(directory / 'status.json')
        state.setdefault('session_id', directory.name)
        state.setdefault('phase', 'setting_up')
        request = read_json(directory / 'command.json')
        state['pending'] = bool(request.get('id') and request['id'] != state.get('ack'))
        state['age_s'] = time.time() - state.get('published_at', time.time())
        return state

    @app.get('/api/native-inference/status')
    def status():
        return read_status(current())

    @app.post('/api/native-inference/command', status_code=202)
    def command(request: SessionCommand):
        with _command_lock:
            directory = current(request.session_id)
            state = read_status(directory)
            if state['pending']:
                raise HTTPException(409, 'Previous command is still being processed')
            allowed = {'reset': {'running', 'paused', 'error'}, 'resume': {'paused'}}
            if state['phase'] not in allowed[request.action]:
                raise HTTPException(409, f"Cannot {request.action} while {state['phase']}")
            if request.parameters is not None:
                if request.action != 'resume' or not state.get('parameters'):
                    raise HTTPException(400, 'Parameters can only be applied when resuming a supported session')
                from native_parameters import validate_parameters
                try:
                    validate_parameters(request.parameters, state['parameters'])
                except ValueError as exc:
                    raise HTTPException(400, str(exc)) from exc
            identifier = uuid.uuid4().hex
            payload = request.model_dump(exclude_none=True)
            atomic_json(directory / 'command.json', {**payload, 'id': identifier})
            return {'accepted': True, 'id': identifier}

    @app.post('/api/native-inference/viewer')
    def viewer(request: NativeViewer):
        directory = current(request.session_id)
        # Lease means hidden/closed browser tabs stop JPEG work automatically.
        with _command_lock:
            atomic_json(directory / 'viewer.json',
                        {**request.model_dump(), 'expires_at': time.time() + 4})
        return {'ok': True}

    @app.get('/api/native-inference/image/{session_id}/{filename}')
    def image(session_id: str, filename: str):
        directory = current(session_id)
        if not re.fullmatch(r'(input|execution)-\d+-\d+\.jpg', filename):
            raise HTTPException(404)
        path = directory / filename
        if not path.is_file():
            raise HTTPException(404)
        return FileResponse(path, media_type='image/jpeg', headers={'Cache-Control': 'no-store'})

    original_status = upstream_server.handle_inference_status

    def enriched_status():
        result = original_status()
        try:
            state = read_status(current())
        except HTTPException:
            return result
        result.update(native_session_id=state['session_id'], session_phase=state['phase'],
                      segment_elapsed_s=state.get('segment_elapsed_s', 0))
        return result

    upstream_server.handle_inference_status = enriched_status
