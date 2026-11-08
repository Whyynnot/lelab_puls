"""Single entry point: refresh groups (parent), restore USB, map devices, run app."""
import fcntl
import os
from pathlib import Path
import socket
import sys

from prepare_hardware import ROOT, prepare


def main():
    args = sys.argv[1:]
    if '--help' in args or '--version' in args:
        os.execv(sys.executable, [sys.executable, '-m', 'uvicorn', *args])
    log_dir = ROOT / 'lelab/logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    lock = (log_dir / 'startup.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('This LeLab is already running or starting; do not start a second instance.')
    os.set_inheritable(lock.fileno(), True)
    port = 8000
    for index, value in enumerate(args):
        if value == '--port' and index + 1 < len(args):
            port = int(args[index + 1])
        elif value.startswith('--port='):
            port = int(value.split('=', 1)[1])
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1', port)) == 0:
            raise SystemExit(f'Port {port} is occupied. Stop the existing LeLab first.')
    report = prepare()
    if '--prepare-only' in args:
        raise SystemExit(0 if report['device_mapping_ready'] else 2)
    os.execv(sys.executable, [sys.executable, '-m', 'uvicorn', 'app:app',
             '--host', '127.0.0.1', '--port', '8000', '--timeout-graceful-shutdown', '30', *args])


if __name__ == '__main__':
    main()
