// @vitest-environment jsdom
import React, { act } from 'react';
import { createRoot, Root } from 'react-dom/client';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import { useCameraBindings } from './useCameraBindings';
import type { RobotRecord } from './useRobots';

const mocks=vi.hoisted(() => ({cameras:[] as {index:number;deviceId:string;role:string;available:boolean}[]}));
vi.mock('./useAvailableCameras',() => ({useAvailableCameras:() => ({cameras:mocks.cameras})}));
const robot={name:'robot',cameras:[{name:'head',device_id:'wsl-camera:head',camera_index:0}]} as RobotRecord;
const features={head:{width:640,height:480}};
let root:Root,container:HTMLDivElement;
function Probe() {
  const state=useCameraBindings(features,robot,true);
  return <div>{JSON.stringify({bindings:state.bindings,ready:state.allBound})}</div>;
}
beforeEach(() => {
  Object.assign(globalThis,{IS_REACT_ACT_ENVIRONMENT:true});
  container=document.createElement('div');root=createRoot(container);
});
afterEach(async () => {await act(async () => root.unmount());});
test('a bound view follows its identity and does not substitute another camera after disconnect',async () => {
  mocks.cameras=[{index:0,deviceId:'wsl-camera:head',role:'head',available:true}];
  await act(async () => root.render(<Probe/>));
  expect(JSON.parse(container.textContent!)).toEqual({bindings:{head:0},ready:true});
  mocks.cameras=[{index:0,deviceId:'wsl-camera:wrist',role:'wrist',available:true},
                 {index:4,deviceId:'wsl-camera:head',role:'head',available:true}];
  await act(async () => root.render(<Probe/>));
  expect(JSON.parse(container.textContent!)).toEqual({bindings:{head:4},ready:true});
  mocks.cameras=[mocks.cameras[0]];
  await act(async () => root.render(<Probe/>));
  expect(JSON.parse(container.textContent!)).toEqual({bindings:{head:null},ready:false});
});
