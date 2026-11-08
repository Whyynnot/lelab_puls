"""Optional stable USB routing; never writes motor registers.

Device identities belong in an ignored local JSON file. Windows forwarding is
opt-in through LELAB_USB_RECOVERY_SCRIPT. Other devices use LeLab form settings.
"""
import copy
import datetime
import grp
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'lelab/logs/hardware_startup.json'
ROLES = ('leader', 'follower', 'head', 'wrist')


def load_devices():
    filename = os.environ.get('LELAB_HARDWARE_CONFIG')
    if not filename:
        return {role: '' for role in ROLES}
    data = json.loads(Path(filename).expanduser().read_text())
    if not isinstance(data, dict) or set(data) - set(ROLES):
        raise ValueError('Hardware config must map leader/follower/head/wrist to device paths')
    for role, value in data.items():
        if not isinstance(value, str) or not value.startswith('/dev/'):
            raise ValueError(f'{role}: expected an absolute /dev/ path')
    return {role: data.get(role, '') for role in ROLES}


DEVICES = load_devices()


def discover():
    result = {}
    for role, name in DEVICES.items():
        if not name:
            continue
        path = Path(name)
        node = str(path.resolve()) if path.exists() else None
        item = {'path': name, 'node': node,
                'accessible': bool(node and os.access(path, os.R_OK | os.W_OK))}
        if node and role in ('head', 'wrist'):
            match = re.fullmatch(r'/dev/video(\d+)', node)
            if match:
                item['index'] = int(match[1])
            else:
                item['accessible'] = False
        result[role] = item
    return result


def mapped_record(record, devices):
    updated = copy.deepcopy(record)
    for role in ('leader', 'follower'):
        if DEVICES.get(role):
            updated[role + '_port'] = DEVICES[role]
    cameras = updated.setdefault('cameras', [])
    for role in ('head', 'wrist'):
        if role not in devices:
            continue
        matches = [camera for camera in cameras if camera.get('name') == role]
        if len(matches) > 1:
            raise ValueError(f'Duplicate {role} camera configuration; keep one camera per role')
        device = devices[role]
        if not device['accessible'] or 'index' not in device:
            continue
        if matches:
            camera = matches[0]
            if camera.get('type', 'opencv') != 'opencv':
                raise ValueError(f'{role} is not configured as an OpenCV camera')
        else:
            camera = {'id': f'{role}_usb_verified', 'name': role, 'type': 'opencv',
                      'width': 640, 'height': 480, 'fps': 30, 'rotation': 0}
            cameras.append(camera)
        camera.update(camera_index=device['index'], device_id=f'wsl-camera:{role}')
        camera.setdefault('fourcc', 'MJPG')
        camera.setdefault('backend', 'V4L2')
    return updated


def prepare(wait_seconds=20, report_path=REPORT, trigger='startup', wait_roles=None):
    report = {'started_at': datetime.datetime.now().astimezone().isoformat(),
              'effective_groups': [grp.getgrgid(g).gr_name for g in sorted(set(os.getgroups()) | {os.getgid()})],
              'issues': [], 'windows': {}, 'config_updated': False, 'trigger': trigger}
    helper = os.environ.get('LELAB_USB_RECOVERY_SCRIPT')
    if helper:
        try:
            response = subprocess.run([
                os.environ.get('LELAB_POWERSHELL', 'powershell.exe'),
                '-NoProfile', '-File', helper, '-Action', 'Start', '-AllowMissing', '-AsJson',
            ], capture_output=True, timeout=min(max(wait_seconds, 1) + 25, 45), check=True)
            report['windows'] = json.loads(response.stdout.decode('utf-8-sig'))
            report['issues'].extend(report['windows'].get('Issues', []))
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            report['issues'].append(f'Optional Windows USB recovery: {exc}')
    report['devices'] = discover()
    checked = set(report['devices']) if wait_roles is None else set(wait_roles) & set(report['devices'])
    for role in checked:
        if not report['devices'][role]['accessible']:
            report['issues'].append(f'{role}: device absent or permission denied ({DEVICES[role]})')
    if report['devices']:
        try:
            from lelab.utils.config import get_robot_record, save_robot_record, save_robot_port
            record = get_robot_record('so101')
            if record is not None:
                updated = mapped_record(record, report['devices'])
                if updated != record:
                    if not save_robot_record('so101', updated, allow_create=False):
                        raise RuntimeError('Failed to update so101 hardware mapping')
                    report['config_updated'] = True
                for role in ('leader', 'follower'):
                    if DEVICES.get(role):
                        save_robot_port(role, DEVICES[role])
                report['calibration_files'] = {role: updated.get(role + '_config') for role in ('leader', 'follower')}
        except Exception as exc:
            report['issues'].append(f'Configuration sync: {exc}')
    report['device_mapping_ready'] = not report['issues']
    report['note'] = 'Configured USB presence/access only. Motor power and live frames are not tested. Unconfigured roles use LeLab form settings.'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    temporary.replace(report_path)
    print('[hardware] ' + ('Configured devices ready.' if report['device_mapping_ready'] else '; '.join(report['issues'])), flush=True)
    return report


if __name__ == '__main__':
    raise SystemExit(0 if prepare()['device_mapping_ready'] else 2)
