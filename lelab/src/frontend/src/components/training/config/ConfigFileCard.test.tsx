// @vitest-environment jsdom
import React, { act } from 'react';
import { createRoot, Root } from 'react-dom/client';
import { beforeEach, afterEach, expect, test, vi } from 'vitest';
import ConfigFileCard from './ConfigFileCard';
import type { TrainingConfig } from '../types';

const mocks = vi.hoisted(() => ({ fetcher: vi.fn() }));
vi.mock('@/contexts/ApiContext', () => ({ useApi: () => ({ baseUrl: 'http://test', fetchWithHeaders: mocks.fetcher }) }));
let root: Root;
let container: HTMLDivElement;
const apply = vi.fn();
const pending = vi.fn();
const config = { target: { runner: 'local' }, dataset_repo_id: 'user/data', policy_type: 'pi05', steps: 20 } as TrainingConfig;
beforeEach(() => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  container = document.createElement('div'); root = createRoot(container);
  mocks.fetcher.mockReset(); apply.mockReset(); pending.mockReset();
  act(() => root.render(<ConfigFileCard config={config} onApply={apply} onPendingChange={pending} />));
});
afterEach(() => act(() => root.unmount()));

function edit(value: string) {
  const textarea = container.querySelector('textarea')!;
  act(() => {
    Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')!.set!.call(textarea, value);
    textarea.dispatchEvent(new Event('input', { bubbles: true }));
  });
}

async function applyDraft() {
  const button = Array.from(container.querySelectorAll('button')).find((b) => b.textContent === '应用配置')!;
  await act(async () => { button.click(); });
}

test('edits block submission until native config is parsed and applied', async () => {
  const native = 'dataset:\n  repo_id: user/new\npolicy:\n  type: pi05\n  freeze_vision_encoder: true';
  edit(native);
  expect(pending).toHaveBeenLastCalledWith(true);
  const loaded = { ...config, dataset_repo_id: 'user/new', extra_config: { policy: { freeze_vision_encoder: true } } };
  mocks.fetcher.mockResolvedValue(new Response(JSON.stringify(loaded)));
  await applyDraft();
  expect(mocks.fetcher).toHaveBeenCalledWith('http://test/training/config/parse', expect.objectContaining({ body: JSON.stringify({ text: native }) }));
  expect(apply).toHaveBeenCalledWith(loaded);
  expect(pending).toHaveBeenLastCalledWith(false);
});

test('invalid edits stay pending and display the backend error', async () => {
  edit('{broken');
  mocks.fetcher.mockResolvedValue(new Response(JSON.stringify({ detail: 'Invalid config' }), { status: 400 }));
  await applyDraft();
  expect(apply).not.toHaveBeenCalled();
  expect(container.querySelector('[role="alert"]')?.textContent).toBe('Invalid config');
  expect(pending).toHaveBeenLastCalledWith(true);
});
