import asyncio
import sys
import time
import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile

from camera_preview import CameraPreview


class PreviewTests(unittest.IsolatedAsyncioTestCase):
    async def test_normal_close_allows_worker_to_release_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            marker=Path(directory)/'released'
            code=('import signal,sys,time,struct\n'
                  'def stop(*args): raise SystemExit(0)\n'
                  'signal.signal(signal.SIGTERM,stop)\n'
                  'try:\n'
                  " sys.stdout.buffer.write(struct.pack('!I',4)+b'jpeg'); sys.stdout.buffer.flush()\n"
                  ' time.sleep(30)\n'
                  'finally:\n'
                  " open(sys.argv[1],'w').write('released')\n")
            with patch('camera_preview.worker_command',return_value=[sys.executable,'-c',code,str(marker)]):
                worker=await CameraPreview.start(99)
            try:
                await worker.read_frame()
            finally:
                await worker.close()
            self.assertEqual(marker.read_text(),'released')

    async def test_hung_camera_times_out_while_event_loop_keeps_running(self):
        command=[sys.executable,'-c','import time; time.sleep(30)']
        ticks=[]
        async def heartbeat():
            await asyncio.sleep(.03)
            ticks.append(True)
        with patch('camera_preview.worker_command',return_value=command):
            worker=await CameraPreview.start(99)
        started=time.monotonic()
        try:
            pulse=asyncio.create_task(heartbeat())
            with self.assertRaises(asyncio.TimeoutError):
                await worker.read_frame(timeout=.15)
            await pulse
            self.assertEqual(ticks,[True])
        finally:
            await worker.close()
        self.assertIsNotNone(worker.process.returncode)
        self.assertLess(time.monotonic()-started,3)

    async def test_reads_complete_frame_from_worker(self):
        code="import sys,struct; sys.stdout.buffer.write(struct.pack('!I',4)+b'jpeg')"
        with patch('camera_preview.worker_command',return_value=[sys.executable,'-c',code]):
            worker=await CameraPreview.start(99)
        try:
            self.assertEqual(await worker.read_frame(),b'jpeg')
        finally:
            await worker.close()

    async def test_rejects_invalid_size_before_allocating_frame(self):
        code="import sys,struct; sys.stdout.buffer.write(struct.pack('!I',2**31))"
        with patch('camera_preview.worker_command',return_value=[sys.executable,'-c',code]):
            worker=await CameraPreview.start(99)
        try:
            with self.assertRaises(ValueError):
                await worker.read_frame()
        finally:
            await worker.close()
