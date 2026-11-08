"""Cancelable checkpoint resolution isolated from the robot rollout process."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys


class PreparationCancelled(Exception):
    pass


def terminate_process(process, timeout=5):
    """Wait for confirmed exit; signal an owned session without touching the server group."""
    if process.poll() is not None:
        return
    owned_group = False
    if os.name == 'posix':
        try:
            owned_group = os.getpgid(process.pid) == process.pid
        except ProcessLookupError:
            pass
    try:
        if owned_group:
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if owned_group:
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait(timeout=5)
    except ProcessLookupError:
        process.wait(timeout=5)
    if process.poll() is None:
        raise RuntimeError('Inference process is still alive after stop')


def resolve_in_worker(reference, cancelled, register, log_handle):
    if cancelled.is_set():
        raise PreparationCancelled()
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), reference],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=log_handle,
        text=True, start_new_session=True,
    )
    try:
        register(process)
        while True:
            if cancelled.is_set():
                raise PreparationCancelled()
            try:
                output, _ = process.communicate(timeout=0.25)
                break
            except subprocess.TimeoutExpired:
                continue
        if cancelled.is_set():
            raise PreparationCancelled()
        if process.returncode:
            raise RuntimeError(f'Checkpoint preparation failed (exit {process.returncode}); see inference log')
        return json.loads(output)['path']
    finally:
        try:
            terminate_process(process)
        finally:
            if process.stdout is not None:
                process.stdout.close()


if __name__ == '__main__':
    # The parent app may have added this editable source directory to sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from lelab.rollout import _resolve_policy_path
    print(json.dumps({'path': _resolve_policy_path(sys.argv[1])}), flush=True)
