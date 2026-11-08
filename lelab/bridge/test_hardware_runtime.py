"""No-hardware tests of hotplug recovery, operation gating and request remapping."""
import asyncio
import copy
import json
import unittest
from unittest.mock import AsyncMock, patch

import hardware_runtime as runtime


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.controller=runtime.Controller()
        self.ready={role:{'accessible':True} for role in runtime.DEVICES}
        self.ready['head']['index']=4
        self.ready['wrist']['index']=0
        self.called=[]
        self.sent=[]

    async def request(self,path,body):
        raw=json.dumps(body).encode()
        async def receive():
            return {'type':'http.request','body':raw,'more_body':False}
        async def send(message):
            self.sent.append(message)
        async def downstream(scope,receive,send):
            data=await receive()
            self.called.append(json.loads(data['body']))
            await send({'type':'http.response.start','status':200,'headers':[]})
            await send({'type':'http.response.body','body':b'{}'})
        await runtime.HardwarePreflight(downstream)(
            {'type':'http','method':'POST','path':path,'headers':[(b'content-length',str(len(raw)).encode())]},receive,send)
        return next(m['status'] for m in self.sent if m['type']=='http.response.start')

    async def test_topology_loss_recovers_before_forward_and_remaps_camera(self):
        absent=copy.deepcopy(self.ready)
        absent['follower']['accessible']=False
        repair=AsyncMock(return_value={'issues':[]})
        self.controller.repair=repair
        body={'cameras':{'head':{'camera_index':0,'rotation':90,'fps':15},'wrist':{'camera_index':2}}}
        with patch.object(runtime,'controller',self.controller),patch.object(runtime,'busy',return_value=False),patch.object(runtime,'discover',side_effect=[absent,self.ready]):
            self.assertEqual(await self.request('/start-recording',body),200)
        repair.assert_awaited_once()
        self.assertEqual(self.called[0]['cameras']['head'],{'camera_index':4,'rotation':90,'fps':15})
        self.assertEqual(self.called[0]['cameras']['wrist']['camera_index'],0)

    async def test_failed_recovery_never_calls_robot_handler(self):
        absent=copy.deepcopy(self.ready)
        absent['follower']['accessible']=False
        self.controller.repair=AsyncMock(return_value={})
        with patch.object(runtime,'controller',self.controller),patch.object(runtime,'busy',return_value=False),patch.object(runtime,'discover',return_value=absent):
            self.assertEqual(await self.request('/move-arm',{}),503)
        self.assertEqual(self.called,[])
        self.assertIn(b'follower',self.sent[-1]['body'])

    async def test_active_robot_blocks_repair_and_second_start(self):
        self.controller.repair=AsyncMock()
        with patch.object(runtime,'controller',self.controller),patch.object(runtime,'busy',return_value=True):
            self.assertEqual(await self.request('/move-arm',{}),409)
        self.controller.repair.assert_not_awaited()
        self.assertEqual(self.called,[])

    async def test_teleoperation_does_not_require_cameras(self):
        ready=copy.deepcopy(self.ready)
        ready['head']['accessible']=False
        ready['wrist']['accessible']=False
        self.controller.repair=AsyncMock()
        with patch.object(runtime,'controller',self.controller),patch.object(runtime,'busy',return_value=False),patch.object(runtime,'discover',return_value=ready):
            self.assertEqual(await self.request('/move-arm',{}),200)
        self.controller.repair.assert_not_awaited()

    async def test_stale_serial_nodes_use_verified_identity_without_changing_calibration(self):
        for role in ('leader','follower'):
            self.ready[role]['path']=runtime.DEVICES[role]
        body={'leader_port':'/dev/ttyACM0','follower_port':'/dev/ttyACM1',
              'leader_config':'my-leader.json','follower_config':'my-follower.json'}
        with patch.object(runtime,'controller',self.controller),patch.object(runtime,'busy',return_value=False), \
             patch.object(runtime,'discover',return_value=self.ready),patch.object(runtime,'probe_serial_port') as probe:
            self.assertEqual(await self.request('/move-arm',body),200)
        self.assertEqual(probe.call_count,2)
        self.assertEqual(self.called[0]['follower_port'],runtime.DEVICES['follower'])
        self.assertEqual(self.called[0]['leader_port'],runtime.DEVICES['leader'])
        self.assertEqual(self.called[0]['follower_config'],'my-follower.json')
        self.assertEqual(body['follower_port'],'/dev/ttyACM1')

    async def test_unresponsive_port_never_enters_robot_driver(self):
        self.ready['follower']['path']=runtime.DEVICES['follower']
        with patch.object(runtime,'controller',self.controller),patch.object(runtime,'busy',return_value=False), \
             patch.object(runtime,'discover',return_value=self.ready), \
             patch.object(runtime,'probe_serial_port',side_effect=TimeoutError('USB open timed out')):
            self.assertEqual(await self.request('/start-inference',{'follower_port':'/dev/ttyACM1'}),503)
        self.assertEqual(self.called,[])

    async def test_idle_watchdog_repairs_disappeared_nodes(self):
        missing=copy.deepcopy(self.ready)
        missing['follower']['accessible']=False
        self.controller.repair=AsyncMock()
        with patch.object(runtime,'busy',return_value=False),patch.object(runtime,'discover',return_value=missing),patch.object(runtime.asyncio,'sleep',AsyncMock(side_effect=[None,asyncio.CancelledError()])):
            with self.assertRaises(asyncio.CancelledError):
                await self.controller.watch()
        self.controller.repair.assert_awaited_once()

    async def test_idle_watchdog_never_repairs_during_calibration_or_motion(self):
        self.controller.repair=AsyncMock()
        with patch.object(runtime,'busy',return_value=True),patch.object(runtime.asyncio,'sleep',AsyncMock(side_effect=[None,asyncio.CancelledError()])):
            with self.assertRaises(asyncio.CancelledError):
                await self.controller.watch()
        self.controller.repair.assert_not_awaited()


if __name__=='__main__':
    unittest.main()
