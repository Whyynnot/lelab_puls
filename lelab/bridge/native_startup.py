"""Use bounded native torque communication retries during SO follower setup only."""
from functools import wraps

TORQUE_EXTRA_RETRIES = 2


def install_startup_retries():
    from lerobot.robots.so_follower.so_follower import SOFollower

    original = SOFollower.configure
    if getattr(original, '_native_startup_retries', False):
        return

    @wraps(original)
    def configure(robot):
        bus = robot.bus
        saved = {name: getattr(bus, name) for name in ('enable_torque', 'disable_torque')}

        def with_retries(method):
            @wraps(method)
            def communicate(motors=None, num_retry=0):
                # The native transport retries only failed communication. No outer
                # reconnect, policy retry, target write or retry in control ticks.
                return method(motors, num_retry=max(num_retry, TORQUE_EXTRA_RETRIES))
            return communicate

        for name, method in saved.items():
            setattr(bus, name, with_retries(method))
        try:
            return original(robot)
        finally:
            for name, method in saved.items():
                setattr(bus, name, method)

    configure._native_startup_retries = True
    SOFollower.configure = configure
