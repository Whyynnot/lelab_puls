"""No hardware: test scheduling, live switches, stale results and coverage failures."""
import threading
import time
import numpy as np
import pytest
from adapter import JOINTS
from control_loop import ControlEngine, Prediction


class Settings:
    def __init__(self, mode='sync'):
        self.value = {'mode':mode,'revision':0}
    def get(self):
        return self.value
    def mode(self, mode):
        self.value = {'mode':mode,'revision':self.value['revision']+1}


class FakeRobot:
    def __init__(self):
        self.commands, self.io_threads = [], set()
    def get_observation(self):
        self.io_threads.add(threading.get_ident())
        return dict.fromkeys(JOINTS,0.)
    def send_action(self, action):
        self.io_threads.add(threading.get_ident())
        self.commands.append((time.monotonic(),action))
        return action


class Sink:
    def __init__(self):self.value=None
    def offer(self, metadata, frames=None):self.value=metadata


def make_predictor(delay=.03):
    calls=[]
    def predict(observation, options):
        calls.append(options)
        call=len(calls)
        time.sleep(delay)
        values=np.full((50,6),call,np.float32)
        return Prediction(values,[dict(zip(JOINTS,row.tolist())) for row in values],delay*1000,delay*1000)
    return predict,calls


def test_live_sync_rtc_sync_switch_and_single_robot_io_owner():
    settings,robot,sink=Settings(),FakeRobot(),Sink()
    predict,calls=make_predictor(.04)
    engine=ControlEngine(robot,predict,settings,sink,JOINTS,fps=50,horizon=4,initial_inference_s=.04)
    timers=[threading.Timer(.21,lambda:settings.mode('rtc')),threading.Timer(.72,lambda:settings.mode('sync'))]
    for timer in timers:timer.start()
    try:engine.run(threading.Event(),1.05)
    finally:
        for timer in timers:timer.join()
    assert robot.commands and len(robot.io_threads)==1
    assert any(c['enabled'] and c['previous'] is not None for c in calls)
    assert not calls[0]['enabled'] and not calls[-1]['enabled']
    assert engine.mode=='sync' and sink.value['phase']=='stopped'


def test_mode_change_discards_an_old_inflight_result():
    settings,robot,sink=Settings(),FakeRobot(),Sink()
    predict,_=make_predictor(.13)
    engine=ControlEngine(robot,predict,settings,sink,JOINTS,fps=50,horizon=3,initial_inference_s=.13)
    timer=threading.Timer(.04,lambda:settings.mode('rtc'));timer.start()
    engine.run(threading.Event(),.47);timer.join()
    assert engine.rejected==1
    assert robot.commands
    assert all(action[JOINTS[0]]!=1 for _,action in robot.commands)


def test_rtc_underrun_stops_instead_of_replaying_old_actions():
    settings,robot,sink=Settings('rtc'),FakeRobot(),Sink()
    calls=[]
    def predict(observation,options):
        calls.append(options)
        delay=.02 if len(calls)==1 else .65
        time.sleep(delay)
        values=np.ones((50,6),np.float32)
        return Prediction(values,[dict.fromkeys(JOINTS,1.) for _ in range(50)],delay*1000,delay*1000)
    engine=ControlEngine(robot,predict,settings,sink,JOINTS,fps=100,horizon=5,initial_inference_s=.02)
    with pytest.raises(RuntimeError,match='buffer exhausted'):
        engine.run(threading.Event(),2)
    assert len(robot.commands)==50 and sink.value['phase']=='error_stopped'
    assert all(action[JOINTS[0]]==1 for _,action in robot.commands)


def test_stop_does_not_wait_for_http_or_send_actions_after_stop():
    settings,robot,sink=Settings(),FakeRobot(),Sink()
    predict,_=make_predictor(.4)
    engine=ControlEngine(robot,predict,settings,sink,JOINTS,fps=50,initial_inference_s=.04)
    stopped=threading.Event();timer=threading.Timer(.06,stopped.set);timer.start()
    start=time.monotonic();engine.run(stopped,2);timer.join()
    assert time.monotonic()-start<.25 and not robot.commands


def test_nonfinite_physical_targets_are_rejected_before_motor_io():
    settings,robot,sink=Settings(),FakeRobot(),Sink()
    def invalid(observation,options):
        return Prediction(np.zeros((50,6),np.float32),[dict.fromkeys(JOINTS,float('nan')) for _ in range(50)],1,1)
    engine=ControlEngine(robot,invalid,settings,sink,JOINTS,fps=50,initial_inference_s=.01)
    with pytest.raises(ValueError,match='Invalid predicted'):
        engine.run(threading.Event(),1)
    assert not robot.commands
