from pathlib import Path

from camera_inventory import cameras, resolve_camera
from prepare_hardware import DEVICES


def make_camera(sys_root, nodes, by_id, role, index):
    device=sys_root/f'video{index}'
    device.mkdir()
    (device/'index').write_text('0')
    (device/'name').write_text('camera')
    target=nodes/f'video{index}'
    target.touch()
    stable=by_id/Path(DEVICES[role]).name
    if stable.is_symlink(): stable.unlink()
    stable.symlink_to(target)


def test_replug_keeps_camera_identity_and_filters_stale_nodes(tmp_path):
    sys_root,nodes,by_id=[tmp_path/name for name in ['sys','dev','by-id']]
    for directory in [sys_root,nodes,by_id]: directory.mkdir()
    make_camera(sys_root,nodes,by_id,'head',0)
    make_camera(sys_root,nodes,by_id,'wrist',2)
    before=cameras(sys_root,by_id)
    make_camera(sys_root,nodes,by_id,'head',4)
    after=cameras(sys_root,by_id)
    assert {row['index'] for row in after}=={2,4}
    assert resolve_camera('head',before)['index']==0
    assert resolve_camera('head',after)['index']==4
    assert resolve_camera('wrist',after)['index']==2
    assert resolve_camera('4',after)['deviceId']=='wsl-camera:head'
    assert resolve_camera('0',after) is None
    assert resolve_camera('../../dev/video0',after) is None


def test_missing_camera_is_not_substituted_with_another_view(tmp_path):
    sys_root,nodes,by_id=[tmp_path/name for name in ['sys','dev','by-id']]
    for directory in [sys_root,nodes,by_id]: directory.mkdir()
    make_camera(sys_root,nodes,by_id,'head',0)
    inventory=cameras(sys_root,by_id)
    assert resolve_camera('wrist',inventory) is None
    assert '主相机' in inventory[0]['name']
