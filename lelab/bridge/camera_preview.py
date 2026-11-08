"""Keep blocking V4L2/USB calls outside the web server process."""
import asyncio
from pathlib import Path
import struct
import sys
import time
import signal

MAX_JPEG_BYTES = 4 * 1024 * 1024


def worker_command(index: int | str) -> list[str]:
    return [sys.executable, str(Path(__file__).resolve()), str(index)]


class CameraPreview:
    def __init__(self, process: asyncio.subprocess.Process):
        self.process = process

    @classmethod
    async def start(cls, index: int | str):
        process = await asyncio.create_subprocess_exec(
            *worker_command(index), stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        return cls(process)

    async def read_frame(self, timeout: float = 5) -> bytes:
        async def read() -> bytes:
            header = await self.process.stdout.readexactly(4)
            length = struct.unpack('!I', header)[0]
            if not 0 < length <= MAX_JPEG_BYTES:
                raise ValueError('Invalid camera frame size')
            return await self.process.stdout.readexactly(length)

        return await asyncio.wait_for(read(), timeout)

    async def close(self) -> None:
        if self.process.returncode is None:
            try:
                self.process.terminate()
            except ProcessLookupError:
                pass
        try:
            await asyncio.wait_for(self.process.wait(), 1)
        except asyncio.TimeoutError:
            try:
                self.process.kill()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(self.process.wait(), 1)
            except asyncio.TimeoutError:
                # A kernel USB wait may survive SIGKILL. Retain the instance
                # until it exits or the physical device is re-enumerated.
                pass


def capture_frames(index: int | str) -> None:
    import cv2

    def stop(*_):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, stop)

    capture = cv2.VideoCapture(index, cv2.CAP_V4L2)
    try:
        if not capture.isOpened():
            return
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        capture.set(cv2.CAP_PROP_FPS, 30)
        while True:
            ok, frame = capture.read()
            if not ok:
                return
            ok, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ok:
                return
            data = jpeg.tobytes()
            sys.stdout.buffer.write(struct.pack('!I', len(data)) + data)
            sys.stdout.buffer.flush()
            time.sleep(.08)
    finally:
        capture.release()


if __name__ == '__main__':
    source=sys.argv[1]
    capture_frames(int(source) if source.isdecimal() else source)
