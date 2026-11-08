"""Optional GPU sampling for offline benchmarks only."""
import subprocess
import threading
import time

class GPUMonitor:
    def __init__(self):
        self.rows=[]
        self.stop=threading.Event()

    def __enter__(self):
        self.start=time.perf_counter()
        def poll():
            while not self.stop.is_set():
                try:
                    out=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw',
                        '--format=csv,noheader,nounits'],text=True,timeout=3)
                    values=[float(v.strip()) for v in out.strip().split(',')]
                    self.rows.append([time.perf_counter()-self.start,*values])
                except (ValueError,subprocess.SubprocessError):
                    pass
                self.stop.wait(0.2)
        self.thread=threading.Thread(target=poll,daemon=True)
        self.thread.start()
        return self

    def __exit__(self,*_):
        self.stop.set()
        self.thread.join(timeout=4)
