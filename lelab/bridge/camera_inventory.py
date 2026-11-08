"""Stable camera identities independent of Linux video node numbering."""
from pathlib import Path

from prepare_hardware import DEVICES


def cameras(sys_root=Path('/sys/class/video4linux'), by_id=Path('/dev/v4l/by-id')):
    identities={str(path.resolve()):path for path in sorted(by_id.glob('*-video-index0')) if path.exists()}
    known={Path(DEVICES[role]).name:role for role in ('head','wrist') if DEVICES.get(role)}
    result=[]
    for device in sorted(sys_root.glob('video*')):
        try:
            if (device/'index').read_text().strip()!='0':
                continue
            index=int(device.name.removeprefix('video'))
            stable=next((path for node,path in identities.items() if Path(node).name==device.name),None)
            if identities and stable is None:
                continue
            role=known.get(stable.name) if stable else None
            identity=role or (stable.name if stable else str(index))
            name=(device/'name').read_text().strip()
            if role:
                name=f"{'主相机' if role=='head' else '手腕相机'} — {name}"
            usb=device.resolve()
            generation=None
            for parent in [usb,*usb.parents]:
                if (parent/'devnum').is_file():
                    generation=f'{parent}:{(parent / "devnum").read_text().strip()}'
                    break
            result.append({'index':index,'name':name,'available':True,
                           'deviceId':f'wsl-camera:{identity}','role':role,
                           'devicePath':str(stable) if stable else f'/dev/video{index}',
                           'generation':generation or str(device.stat().st_ino)})
        except (OSError,ValueError):
            # A camera can disappear between enumerating and reading sysfs.
            continue
    return result


def resolve_camera(identity, inventory):
    return next((camera for camera in inventory
                 if camera['deviceId']==f'wsl-camera:{identity}' or
                 (identity.isdecimal() and camera['index']==int(identity))),None)
