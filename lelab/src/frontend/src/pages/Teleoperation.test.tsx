// @vitest-environment jsdom
import React, { act } from 'react';
import { createRoot, Root } from 'react-dom/client';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import Teleoperation from './Teleoperation';

const mocks = vi.hoisted(() => ({ fetcher: vi.fn(), toast: vi.fn(), navigate: vi.fn() }));
vi.mock('@/contexts/ApiContext', () => ({ useApi: () => ({ baseUrl: 'http://test', fetchWithHeaders: mocks.fetcher }) }));
vi.mock('@/hooks/use-toast', () => ({ useToast: () => ({ toast: mocks.toast }) }));
vi.mock('react-router-dom', () => ({ useNavigate: () => mocks.navigate }));
vi.mock('@/components/control/VisualizerPanel', () => ({ default: () => <div>visualizer</div> }));
vi.mock('@/components/control/TeleopCameraPanel', () => ({ default: () => null }));

let root: Root, container: HTMLDivElement;
let status: { teleoperation_active: boolean; last_error: string | null; fault_message: string | null };
beforeEach(() => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  vi.useFakeTimers();
  status = { teleoperation_active: true, last_error: null, fault_message: null };
  mocks.fetcher.mockReset().mockImplementation(async (url: string) => ({
    ok: true, json: async () => url.endsWith('/teleoperation-status') ? status : { success: false },
  }));
  container = document.createElement('div');
  root = createRoot(container);
});
afterEach(async () => {
  await act(async () => root.unmount());
  vi.useRealTimers();
});

test('shows a persistent fault when the running session loses communication', async () => {
  await act(async () => root.render(<Teleoperation />));
  expect(container.querySelector('[role="alert"]')).toBeNull();
  status = { teleoperation_active: false, last_error: '读取主臂位置: USB timeout',
    fault_message: '请先支撑好从臂。异常退出未主动关闭从臂扭矩。' };
  await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
  expect(container.querySelector('[role="alert"]')?.textContent).toContain('请先支撑好从臂');
  expect(container.textContent).toContain('读取主臂位置: USB timeout');
  expect(mocks.fetcher.mock.calls.some(([url]) => url.endsWith('/move-arm') || url.endsWith('/stop-teleoperation'))).toBe(false);
  mocks.fetcher.mockRejectedValue(new Error('network lost'));
  await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
  expect(container.querySelector('[role="alert"]')).not.toBeNull();
});
