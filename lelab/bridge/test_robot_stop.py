import subprocess
import sys
from unittest.mock import Mock
import pytest
from robot_stop import RobotStopper, StopBlocked


def test_normal_stop_waits_for_exit():
    process=Mock(pid=123)
    process.poll.side_effect=[None,0]
    stopper=RobotStopper(process)
    assert stopper.stop()['stopped']
    process.terminate.assert_called_once()
    process.kill.assert_not_called()


def test_unresponsive_worker_is_killed_and_reaped():
    process=Mock(pid=123)
    process.poll.side_effect=[None,-9]
    process.wait.side_effect=[subprocess.TimeoutExpired('robot',8),-9]
    stopper=RobotStopper(process)
    assert stopper.stop()['stopped']
    process.kill.assert_called_once()
    assert process.wait.call_count==2


def test_kernel_blocked_process_never_reports_stopped():
    process=Mock(pid=99999999)
    process.poll.return_value=None
    process.wait.side_effect=subprocess.TimeoutExpired('robot',1)
    stopper=RobotStopper(process)
    with pytest.raises(StopBlocked,match='旧进程退出前禁止启动新任务'):
        stopper.stop()
    assert stopper.snapshot()['phase']=='kernel_blocked'
    assert not stopper.snapshot()['stopped']


def test_failure_cleanup_is_bounded_and_preserves_original_error(tmp_path):
    path=tmp_path/'robot.log'
    path.write_text('ERROR: LELAB_INFERENCE_FAULT: motor read timeout\n')
    process=Mock(pid=123)
    process.poll.side_effect=[None,None,-9]
    process.wait.side_effect=[subprocess.TimeoutExpired('robot',5),subprocess.TimeoutExpired('robot',8),-9]
    stopper=RobotStopper(process)
    stopper.watch_fault(path)
    assert stopper.fault=='motor read timeout'
    process.kill.assert_called_once()


def test_concurrent_stop_does_not_send_duplicate_signals():
    process=Mock(pid=123)
    stopper=RobotStopper(process)
    stopper.lock.acquire()
    try:
        with pytest.raises(StopBlocked,match='正在处理中'):
            stopper.stop()
        process.terminate.assert_not_called()
    finally:
        stopper.lock.release()


def test_real_worker_ignoring_terminate_is_killed_without_hardware():
    process=subprocess.Popen([sys.executable,'-u','-c',
        "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print('ready'); time.sleep(60)"],
        stdout=subprocess.PIPE,text=True)
    try:
        assert process.stdout.readline().strip()=='ready'
        result=RobotStopper(process,grace=.05,kill_wait=2).stop()
        assert result['stopped'] and process.returncode < 0
    finally:
        if process.poll() is None:
            process.kill();process.wait(timeout=2)
        process.stdout.close()
