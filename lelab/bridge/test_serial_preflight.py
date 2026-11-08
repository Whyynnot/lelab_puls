import sys
import time

import pytest

import serial_preflight


def test_hung_open_is_bounded_and_does_not_block_server_process(monkeypatch):
    monkeypatch.setattr(serial_preflight, '_command', lambda port: [sys.executable, '-c', 'import time; time.sleep(30)'])
    started=time.monotonic()
    with pytest.raises(TimeoutError, match='USB'):
        serial_preflight.probe_serial_port('/dev/fake', timeout=.1)
    assert time.monotonic()-started < 3
