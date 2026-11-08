"""Fault cleanup for the LeLab native rollout child; no automatic recovery motion."""
from functools import wraps
import logging
import sys

LOG = logging.getLogger(__name__)


def install_fault_cleanup():
    from lerobot.rollout.robot_wrapper import ThreadSafeRobot
    from lerobot.rollout.strategies.core import RolloutStrategy
    from lelab.utils.devices import safe_disconnect_device

    original_read = ThreadSafeRobot.get_observation
    if getattr(original_read, '_lelab_fault_cleanup', False):
        return
    original_teardown = RolloutStrategy._teardown_hardware

    @wraps(original_read)
    def get_observation(wrapper, *args, **kwargs):
        try:
            return original_read(wrapper, *args, **kwargs)
        except Exception as exc:
            # v0.7 records send/hold failures, but a failed observation must
            # also prohibit the segment-end hold and return-to-start motion.
            with wrapper._lock:
                wrapper._record_hardware_failure('get_observation', exc)
            raise

    @wraps(original_teardown)
    def teardown(strategy, hardware, return_to_initial_position=True):
        engine = strategy._engine
        failure = (hardware.robot_wrapper.hardware_failure
                   or getattr(hardware, '_lelab_failure', None)
                   or (engine.failure_traceback if engine is not None and engine.failed else None)
                   or sys.exc_info()[1])
        if not failure:
            try:
                return original_teardown(strategy, hardware, return_to_initial_position)
            except Exception:
                # A disconnect failure must not leave camera readers alive at
                # interpreter shutdown or leave the serial handle open.
                safe_disconnect_device(hardware.robot_wrapper.inner, LOG,
                                       'native teardown', preserve_torque=True)
                raise

        message = str(failure).strip().splitlines()[-1]
        LOG.error('LELAB_INFERENCE_FAULT: %s', message)
        LOG.warning('Fault shutdown: skipping return motion and torque-disable commands; '
                    'support the follower before handling its USB or power connections.')
        try:
            if strategy._interpolator is not None:
                strategy._interpolator.reset()
            strategy._cached_obs_processed = None
            if engine is not None:
                engine.stop()
        except Exception:
            LOG.exception('Inference engine cleanup failed after original fault')
        finally:
            safe_disconnect_device(hardware.robot_wrapper.inner, LOG,
                                   'native fault', preserve_torque=True)
            safe_disconnect_device(hardware.teleop, LOG,
                                   'native fault', preserve_torque=True)

    get_observation._lelab_fault_cleanup = True
    ThreadSafeRobot.get_observation = get_observation
    RolloutStrategy._teardown_hardware = teardown
