"""Bounded latest-only monitoring. Encoding and disk writes stay off control ticks."""
import json
import threading
import time
import uuid
from pathlib import Path

RUNTIME = Path(__file__).resolve().parents[1] / 'runtime'
CONTROL = RUNTIME/'control.json'


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


class ControlStore:
    """Owned by the web process. Mode commands and short viewer leases only."""
    def __init__(self, path=CONTROL):
        self.path = Path(path)
        self.lock = threading.RLock()
        self.clients = {}
        self.value = {'mode': 'sync', 'revision': 0, 'monitor_until': 0, 'images': False}
        if self.path.exists():
            try:
                saved = json.loads(self.path.read_text())
                self.value.update(mode=saved['mode'], revision=int(saved['revision']))
            except (OSError, ValueError, KeyError):
                pass

    def get(self):
        with self.lock:
            return dict(self.value)

    def set_mode(self, mode):
        if mode not in {'sync', 'rtc'}:
            raise ValueError('mode must be sync or rtc')
        with self.lock:
            if self.value['mode'] != mode:
                self.value.update(mode=mode, revision=self.value['revision']+1)
            atomic_json(self.path, self.value)
            return dict(self.value)

    def lease(self, client, enabled, images):
        now = time.time()
        with self.lock:
            self.clients = {k: v for k, v in self.clients.items() if v[0] > now}
            if enabled:
                if len(self.clients) >= 16 and client not in self.clients:
                    raise ValueError('Too many monitoring viewers')
                self.clients[client] = (now+6, bool(images))
            else:
                self.clients.pop(client, None)
            self.value.update(monitor_until=max((v[0] for v in self.clients.values()), default=0),
                              images=any(v[1] for v in self.clients.values()))
            atomic_json(self.path, self.value)
            return dict(self.value)


class ControlReader:
    """Background file polling; the motor thread only reads an immutable reference."""
    def __init__(self, path=CONTROL):
        self.path = Path(path)
        self.value = {'mode': 'sync', 'revision': 0, 'monitor_until': 0, 'images': False}
        self.stop = threading.Event()
        self.refresh()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def refresh(self):
        try:
            value = json.loads(self.path.read_text())
            if value['mode'] in {'sync', 'rtc'}:
                self.value = value
        except (OSError, ValueError, KeyError):
            pass

    def _run(self):
        while not self.stop.wait(.1):
            self.refresh()

    def get(self):
        return self.value

    def close(self):
        self.stop.set()
        self.thread.join(timeout=1)


class LatestMonitor:
    """One slot; two 320x240 JPEGs at at most 2 Hz, only while viewers exist."""
    def __init__(self, control, directory=RUNTIME, session=None):
        self.control, self.directory = control, Path(directory)
        self.session = session or uuid.uuid4().hex
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.latest = None
        self.frames = None
        self.frame_version = 0
        self.encoded_version = -1
        self.sequence = 0
        self.dropped = 0
        self.encoded_frames = 0
        self.last_background_ms = 0.
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def offer(self, metadata, frames=None):
        # No encoding, serialization, image copies, disk or socket operations here.
        if not self.lock.acquire(blocking=False):
            self.dropped += 1
            return False
        try:
            if frames is not None:
                self.frames = frames
                self.frame_version += 1
            self.latest = (metadata, self.frames, self.frame_version)
        finally:
            self.lock.release()
        return True

    def publish(self):
        settings = self.control.get()
        if settings.get('monitor_until', 0) <= time.time():
            return
        if not self.lock.acquire(blocking=False):
            return
        try:
            item = self.latest
        finally:
            self.lock.release()
        if item is None:
            return
        started = time.perf_counter()
        metadata, frames, version = item
        image_names = []
        if settings.get('images') and frames:
            import cv2
            image_names = list(frames)
            if version != self.encoded_version:
                for name, frame in frames.items():
                    if name not in {'head', 'wrist'}:
                        continue
                    resized = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
                    ok, jpeg = cv2.imencode('.jpg', cv2.cvtColor(resized, cv2.COLOR_RGB2BGR),
                                            [cv2.IMWRITE_JPEG_QUALITY, 60])
                    if not ok:
                        raise RuntimeError('Monitor JPEG encoding failed')
                    target = self.directory/(name+'.jpg')
                    temporary = target.with_suffix('.tmp')
                    temporary.write_bytes(jpeg.tobytes())
                    temporary.replace(target)
                    self.encoded_frames += 1
                self.encoded_version = version
        self.sequence += 1
        snapshot = {**metadata, 'session': self.session, 'sequence': self.sequence,
                    'published_at': time.time(), 'image_version': self.encoded_version,
                    'images': image_names, 'monitor': {'max_fps': 2, 'image_size': [320, 240],
                    'dropped_offers': self.dropped, 'encoded_frames': self.encoded_frames,
                    'previous_background_ms': self.last_background_ms}}
        atomic_json(self.directory/'monitor.json', snapshot)
        self.last_background_ms = 1000*(time.perf_counter()-started)

    def _run(self):
        while not self.stop.wait(.5):
            try:
                self.publish()
            except Exception:
                # Monitoring failure must never stop or stall robot control.
                self.dropped += 1

    def close(self, flush=False):
        self.stop.set()
        self.thread.join(timeout=2)
        if flush and not self.thread.is_alive():
            try:
                self.publish()
            except Exception:
                pass
