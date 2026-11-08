"""SO101 I/O owner; model requests, mode control and monitoring run independently."""
import argparse
import json
import logging
import signal
import threading
import time
from pathlib import Path

import torch
from adapter import ModelClient, RobotAdapter, read_profile
from control_loop import ControlEngine, Prediction
from monitoring import ControlReader, LatestMonitor
from calibration_guard import verify_follower_calibration


def build_camera_config(settings):
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lelab.camera_config import lerobot_camera_settings
    options = lerobot_camera_settings(settings)
    options['index_or_path'] = options.pop('camera_index')
    options.pop('type', None)
    # These UVC cameras expose 30fps at 640x480 through MJPG. In particular,
    # wrist YUYV caps at 25fps. Preserve an explicitly selected format.
    options.setdefault('fourcc', 'MJPG')
    return OpenCVCameraConfig(**{k:v for k,v in options.items() if v is not None})


def disconnect_robot(robot, logger, *, preserve_torque=False, timeout=3.):
    """Close the bus first; one stuck camera must not prevent closing the other."""
    def background(name, close):
        def run():
            try:
                close()
            except Exception as exc:
                logger.warning('Could not disconnect %s: %s', name, exc)
        thread = threading.Thread(target=run, name='cleanup-'+name, daemon=True)
        thread.start()
        return thread

    workers = []
    if robot.bus.is_connected:
        worker = background('motor-bus', lambda: robot.bus.disconnect(
            disable_torque=False if preserve_torque else robot.config.disable_torque_on_disconnect))
        workers.append(worker)
        worker.join(timeout=timeout)
    for name, camera in robot.cameras.items():
        if camera.is_connected:
            workers.append(background(name, camera.disconnect))
    deadline = time.monotonic()+timeout
    for worker in workers:
        worker.join(timeout=max(0,deadline-time.monotonic()))
    pending = [worker.name for worker in workers if worker.is_alive()]
    if pending:
        logger.error('LELAB_CLEANUP_TIMEOUT: %s; parent must confirm process exit', ', '.join(pending))
    return not pending


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text())
    profile = read_profile(request['profile'])
    profile = profile.model_copy(update={'fps': request.get('control_fps', profile.fps)})
    torch.set_num_threads(2)
    adapter = RobotAdapter(profile)
    client = ModelClient()
    client.require_artifact(profile.artifact)
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from lelab.utils.config import setup_follower_calibration_file, FOLLOWER_CONFIG_PATH

    calibration = request['follower_config']
    if not calibration or Path(calibration).name != calibration:
        raise ValueError('Select a LeLab calibration filename')
    robot_id = setup_follower_calibration_file(calibration)
    cameras = {}
    for name, settings in request['cameras'].items():
        cameras[name] = build_camera_config(settings)
        print(f'Camera {name}: device={cameras[name].index_or_path}, '
              f'fps={cameras[name].fps}, fourcc={cameras[name].fourcc}', flush=True)
    if set(cameras) != {'head', 'wrist'}:
        raise ValueError('Configure both head and wrist cameras')
    robot = SO101Follower(SO101FollowerConfig(
        port=request['follower_port'], id=robot_id, cameras=cameras,
        calibration_dir=Path(FOLLOWER_CONFIG_PATH),
        use_degrees=profile.use_degrees, max_relative_target=profile.max_relative_target))
    if not robot.calibration:
        raise ValueError('Missing follower calibration')
    stopped = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopped.set())
    controls = ControlReader()
    monitor = LatestMonitor(controls, session=request['session'])

    def predict(observation, options):
        started = time.perf_counter()
        batch = adapter.prepare(observation, request['task'])
        rtc = {k:v for k,v in options.items() if k != 'physical'}
        if options['previous'] is not None:
            rtc['previous'] = adapter.rtc_prefix(options['previous'], options['physical'], observation)
        actions, inference_ms = client.infer(batch, rtc)
        physical = adapter.decode(actions)
        return Prediction(actions[0], physical, inference_ms, 1000*(time.perf_counter()-started))

    failed = True
    try:
        verification = verify_follower_calibration(robot, calibration, FOLLOWER_CONFIG_PATH)
        print(f'Calibration verified: {verification["selected"]}, {verification["motors_checked"]} motors', flush=True)
        if stopped.is_set():
            return
        robot.connect(calibrate=False)
        if not robot.is_calibrated:
            raise ValueError('Calibration mismatch; calibrate in LeLab first')
        print('Rollout setup complete', flush=True)
        engine = ControlEngine(robot, predict, controls, monitor, profile.action_names,
                               fps=profile.fps, horizon=profile.execution_horizon)
        engine.run(stopped, request['duration_s'])
        failed = False
    except Exception as exc:
        logging.getLogger(__name__).exception('LELAB_INFERENCE_FAULT: %s: %s', type(exc).__name__, exc)
        raise
    finally:
        logger = logging.getLogger(__name__)
        for close in (lambda: monitor.close(flush=True), controls.close):
            try:
                close()
            except Exception:
                logger.exception('Monitor/control cleanup failed')
        disconnect_robot(robot, logger, preserve_torque=failed)


if __name__ == '__main__':
    main()
