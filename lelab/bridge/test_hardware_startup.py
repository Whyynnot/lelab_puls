"""Device mapping regressions without robot I/O."""
import unittest
from prepare_hardware import DEVICES, mapped_record


class MappingTests(unittest.TestCase):
    def setUp(self):
        self.record = {'name': 'so101', 'leader_port': '/dev/ttyACM0',
                       'follower_port': '/dev/ttyACM1', 'leader_config': 'my-leader.json',
                       'follower_config': 'my-follower.json', 'cameras': [
            {'id': 'saved-head', 'name': 'head', 'type': 'opencv', 'camera_index': 0,
             'device_id': 'wsl-camera:0', 'rotation': 90, 'width': 640, 'fps': 15},
            {'id': 'saved-wrist', 'name': 'wrist', 'type': 'opencv', 'camera_index': 2,
             'device_id': 'wsl-camera:2', 'rotation': 180, 'fourcc': 'MJPG'},
        ]}
        self.devices = {role: {'accessible': True} for role in DEVICES}
        self.devices['head']['index'] = 4
        self.devices['wrist']['index'] = 0

    def test_reenumeration_preserves_roles_and_calibration(self):
        changed = mapped_record(self.record, self.devices)
        for role in ('leader', 'follower'):
            self.assertEqual(changed[role + '_port'], DEVICES[role])
            self.assertEqual(changed[role + '_config'], self.record[role + '_config'])
        for before, after, index in zip(self.record['cameras'], changed['cameras'], (4, 0)):
            self.assertEqual(after['camera_index'], index)
            self.assertEqual(after['device_id'], f"wsl-camera:{after['name']}")
            for key in ('id', 'rotation', 'fps', 'width'):
                self.assertEqual(before.get(key), after.get(key))
        self.assertEqual(self.record['cameras'][0]['camera_index'], 0)
        self.assertEqual(mapped_record(changed, self.devices), changed)

    def test_missing_camera_preserves_existing_config(self):
        self.devices['wrist'] = {'accessible': False}
        changed = mapped_record(self.record, self.devices)
        self.assertEqual(changed['cameras'][1], self.record['cameras'][1])

    def test_ambiguous_camera_role_fails_closed(self):
        self.record['cameras'].append(dict(self.record['cameras'][0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate head'):
            mapped_record(self.record, self.devices)

    def test_missing_role_can_be_created_from_verified_camera(self):
        self.record['cameras'] = []
        changed = mapped_record(self.record, self.devices)
        self.assertEqual([c['name'] for c in changed['cameras']], ['head', 'wrist'])
        self.assertTrue(all(c['fourcc'] == 'MJPG' for c in changed['cameras']))


if __name__ == '__main__':
    unittest.main()
