"""Native rollout entry point with browser control instead of stdin interaction."""
from native_session import WebSession
from native_startup import install_startup_retries
from native_faults import install_fault_cleanup
from lerobot.scripts import lerobot_rollout

if __name__ == '__main__':
    # Scoped to this child process; no LeRobot source or model code is replaced.
    lerobot_rollout.InteractiveSession = WebSession
    install_startup_retries()
    install_fault_cleanup()
    lerobot_rollout.main()
