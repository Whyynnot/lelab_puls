import json
import threading
import time
import numpy as np
from monitoring import ControlReader, ControlStore, LatestMonitor


def test_mode_commands_are_atomic_and_observed_without_control_thread_disk_io(tmp_path):
    store=ControlStore(tmp_path/'control.json')
    reader=ControlReader(store.path)
    try:
        first=store.set_mode('rtc')
        time.sleep(.15)
        assert reader.get()['mode']=='rtc'
        assert store.set_mode('rtc')['revision']==first['revision']
        assert store.set_mode('sync')['revision']==first['revision']+1
    finally:reader.close()


def test_no_viewer_means_no_jpeg_or_snapshot_writes(tmp_path):
    store=ControlStore(tmp_path/'control.json')
    monitor=LatestMonitor(store,tmp_path/'monitor')
    try:
        monitor.offer({'state':{'x':1}}, {'head':np.zeros((480,640,3),np.uint8)})
        time.sleep(.6)
        assert monitor.encoded_frames==0
        assert not (monitor.directory/'monitor.json').exists()
    finally:monitor.close()


def test_slow_jpeg_consumer_does_not_block_latest_slot_producer(tmp_path,monkeypatch):
    import cv2
    store=ControlStore(tmp_path/'control.json');store.lease('test',True,True)
    gate,release=threading.Event(),threading.Event()
    original=cv2.imencode
    def slow(*args,**kwargs):
        gate.set();release.wait(2)
        return original(*args,**kwargs)
    monkeypatch.setattr(cv2,'imencode',slow)
    monitor=LatestMonitor(store,tmp_path/'monitor')
    try:
        monitor.offer({'value':0},{'head':np.zeros((480,640,3),np.uint8)})
        assert gate.wait(1.5)
        start=time.perf_counter()
        for index in range(1000):monitor.offer({'value':index})
        assert time.perf_counter()-start<.15
        release.set();time.sleep(.7)
        snapshot=json.loads((monitor.directory/'monitor.json').read_text())
        assert snapshot['value']==999 and snapshot['monitor']['encoded_frames']==1
    finally:
        release.set();monitor.close()


def test_monitor_clients_can_disable_images_and_expire(tmp_path):
    store=ControlStore(tmp_path/'control.json')
    store.lease('a',True,True);store.lease('b',True,False)
    assert store.get()['images']
    store.lease('a',False,True)
    assert not store.get()['images'] and store.get()['monitor_until']>time.time()
    store.clients={'b':(time.time()-1,False)}
    store.lease('c',False,False)
    assert store.get()['monitor_until']==0
