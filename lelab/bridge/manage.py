"""Start/status/stop the owned LeLab process without touching other servers."""
import argparse
import json
import os
import signal
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
LOGS=ROOT/'lelab/logs'
LOGS.mkdir(exist_ok=True)
PID=LOGS/'lelab.pid'


def process_pid():
    if not PID.exists():
        return None
    pid=int(PID.read_text())
    try:
        args=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ')
        cwd=Path(f'/proc/{pid}/cwd').resolve()
        owned = (b'uvicorn app:app' in args or
                 str(ROOT/'lelab/bridge/launch.py').encode() in args or
                 str(ROOT/'start_lelab.sh').encode() in args)
        if owned and cwd==ROOT/'lelab/bridge' and Path(f'/proc/{pid}').stat().st_uid==os.getuid():
            return pid
    except (FileNotFoundError,ProcessLookupError):
        pass
    return None


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['start','stop','status'])
    args=parser.parse_args()
    pid=process_pid()
    if args.command=='start':
        if pid:
            print(f'LeLab already running, PID {pid}: http://localhost:8000')
            return
        import socket
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1',8000))==0:
                raise RuntimeError('Port 8000 belongs to another process; stop it or choose another port manually')
        with (LOGS/'lelab.log').open('a') as log:
            process=subprocess.Popen(['bash',str(ROOT/'start_lelab.sh')],cwd=ROOT/'lelab/bridge',
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        PID.write_text(str(process.pid))
        print(f'LeLab starting, PID {process.pid}: http://localhost:8000 (FP8: /quantized)')
    elif args.command=='stop':
        if pid:
            if os.getpgid(pid)==pid:
                os.killpg(pid,signal.SIGTERM)
            else:
                os.kill(pid,signal.SIGTERM)
            print('Stop requested; LeLab will disconnect its robot and unload its FP8 process.')
        else:
            print('No owned LeLab process running.')
    else:
        print(json.dumps({'running':bool(pid),'pid':pid,'url':'http://localhost:8000',
                          'log':str(LOGS/'lelab.log')},indent=2))


if __name__=='__main__':
    main()
