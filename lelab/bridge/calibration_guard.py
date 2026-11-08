"""Read-only calibration preflight before cameras, motor configuration or torque."""
import json
from pathlib import Path
from lerobot.motors import MotorCalibration


def matches(bus, actual, cached):
    original = bus.calibration
    try:
        bus.calibration = cached
        return set(actual) == set(cached) and all(bus._same_calibration(name, value) for name,value in actual.items())
    finally:
        bus.calibration = original


def verify_follower_calibration(robot, selected, directory):
    """Only ping/read the bus; always close without changing torque or EEPROM."""
    bus = robot.bus
    try:
        bus.connect()
        actual = bus.read_calibration()
        if matches(bus, actual, robot.calibration):
            return {'selected':selected, 'motors_checked':len(actual), 'matches':True}
        candidates = []
        for path in sorted(Path(directory).glob('*.json')):
            try:
                saved = {name:MotorCalibration(**row) for name,row in json.loads(path.read_text()).items()}
            except (OSError, ValueError, TypeError, AttributeError):
                continue
            if matches(bus, actual, saved):
                candidates.append(path.name)
        suggestion = ('当前电机匹配的标定文件：' + ', '.join(candidates) + '。请在部署页“标定文件”中选择它。') if candidates else '本地没有与当前电机完全匹配的文件，请核对设备与标定来源。'
        raise ValueError(f'所选从臂标定 {selected} 与电机不一致。{suggestion}未写入标定或配置电机。')
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)
