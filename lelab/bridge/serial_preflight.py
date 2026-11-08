"""Probe port opening in a killable process before invoking an in-process driver."""
import os
import signal
import subprocess
import sys
import threading


def _command(port):
    code = ('import serial,sys; '
            'p=serial.Serial(sys.argv[1],baudrate=1000000,timeout=.05,write_timeout=.05,exclusive=True); '
            'p.close()')
    return [sys.executable, '-c', code, port]


def probe_serial_port(port, timeout=4):
    process = subprocess.Popen(_command(port), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            def reap():
                process.wait()
                if process.stdout:
                    process.stdout.close()
            threading.Thread(target=reap, daemon=True).start()
        raise TimeoutError(f'USB 串口打开超时：{port}。请重连设备或恢复 USB 转发后重试。') from exc
    if process.returncode:
        detail = output.strip().splitlines()[-1] if output.strip() else 'Port open failed'
        raise ConnectionError(f'USB 串口无法打开：{port}。{detail[:300]}')
