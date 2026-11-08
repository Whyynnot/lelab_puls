"""Bounded process termination; a kernel-blocked worker never counts as stopped."""
import subprocess
import threading
import time
from pathlib import Path


class StopBlocked(RuntimeError):
    pass


class RobotStopper:
    def __init__(self, process, *, grace=8, kill_wait=2):
        self.process, self.grace, self.kill_wait = process, grace, kill_wait
        self.lock = threading.Lock()
        self.phase = 'running'
        self.fault = self.error = None

    def snapshot(self):
        exited = self.process.poll() is not None
        phase = ('fault_stopped' if self.fault else 'stopped') if exited else self.phase
        return {'pid':self.process.pid,'phase':phase,'stopped':exited,
                'fault':self.fault,'error':self.error}

    def stop(self):
        if not self.lock.acquire(blocking=False):
            raise StopBlocked('停止请求正在处理中，请等待状态更新。')
        try:
            if self.process.poll() is not None:
                return self.snapshot()
            self.phase, self.error = 'stopping', None
            try:
                self.process.terminate()
                self.process.wait(timeout=self.grace)
            except subprocess.TimeoutExpired:
                self.phase = 'force_stopping'
                self.process.kill()
                try:
                    self.process.wait(timeout=self.kill_wait)
                except subprocess.TimeoutExpired:
                    try:
                        channel = Path(f'/proc/{self.process.pid}/wchan').read_text().strip()
                    except OSError:
                        channel = 'unknown'
                    self.phase = 'kernel_blocked'
                    self.error = (f'任务仍未退出（内核等待：{channel}）。已发送强制终止，'
                                  '请保持从臂有支撑并恢复 USB 转接；旧进程退出前禁止启动新任务。')
                    raise StopBlocked(self.error) from None
            except ProcessLookupError:
                pass
            return self.snapshot()
        finally:
            self.lock.release()

    def watch_fault(self, log_path):
        """Faults stop producing commands already; bound their cleanup lifetime."""
        while self.process.poll() is None:
            try:
                with Path(log_path).open('rb') as handle:
                    handle.seek(0,2)
                    handle.seek(max(0,handle.tell()-16000))
                    lines = handle.read().decode('utf-8',errors='replace').splitlines()
                faults = [line.split('LELAB_INFERENCE_FAULT:',1)[1].strip()
                          for line in lines if 'LELAB_INFERENCE_FAULT:' in line]
                if faults:
                    self.fault = faults[0]
                    if self.phase == 'running':
                        self.phase = 'fault_cleanup'
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        try:
                            self.stop()
                        except StopBlocked:
                            pass
                    return
            except OSError:
                pass
            time.sleep(.5)
