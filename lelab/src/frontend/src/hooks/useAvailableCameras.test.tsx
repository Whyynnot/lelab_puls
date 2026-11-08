// @vitest-environment jsdom
import React, { act } from 'react';
import { createRoot, Root } from 'react-dom/client';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import { useAvailableCameras } from './useAvailableCameras';

const mocks=vi.hoisted(() => ({fetcher:vi.fn(),browser:vi.fn()}));
vi.mock('@/contexts/ApiContext',() => ({useApi:() => ({baseUrl:'http://test',fetchWithHeaders:mocks.fetcher})}));
let root:Root, container:HTMLDivElement;
function Probe() {
  const {cameras}=useAvailableCameras();
  return <div>{cameras.map(camera => `${camera.deviceId}=${camera.index}`).join(',')}</div>;
}
beforeEach(() => {
  Object.assign(globalThis,{IS_REACT_ACT_ENVIRONMENT:true});
  vi.useFakeTimers();mocks.fetcher.mockReset();mocks.browser.mockReset();
  Object.defineProperty(navigator,'mediaDevices',{configurable:true,value:{getUserMedia:mocks.browser,addEventListener:vi.fn(),removeEventListener:vi.fn()}});
  container=document.createElement('div');root=createRoot(container);
});
afterEach(async () => {await act(async () => root.unmount());vi.useRealTimers();});

test('WSL hotplug refreshes without a Windows browser devicechange event',async () => {
  let index=0;
  mocks.fetcher.mockImplementation(async () => ({ok:true,json:async () => ({cameras:[{index,name:'head',deviceId:'wsl-camera:head',available:true}]})}));
  await act(async () => root.render(<Probe/>));
  expect(container.textContent).toBe('wsl-camera:head=0');
  index=4;
  await act(async () => {await vi.advanceTimersByTimeAsync(2000);});
  expect(container.textContent).toBe('wsl-camera:head=4');
  expect(mocks.browser).not.toHaveBeenCalled();
});

test('a failed WSL inventory does not grab a different Windows camera',async () => {
  mocks.fetcher.mockResolvedValue({ok:false,status:503});
  await act(async () => root.render(<Probe/>));
  expect(mocks.browser).not.toHaveBeenCalled();
  expect(container.textContent).toBe('');
});
