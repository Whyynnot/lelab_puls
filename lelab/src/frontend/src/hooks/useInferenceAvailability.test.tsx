// @vitest-environment jsdom
import React, { act } from "react";
import { createRoot, Root } from "react-dom/client";
import { beforeEach, afterEach, expect, test, vi } from "vitest";
import { useInferenceAvailability } from "./useInferenceAvailability";

const mocks = vi.hoisted(() => ({ fetcher: vi.fn() }));
vi.mock("@/contexts/ApiContext", () => ({ useApi: () => ({ baseUrl: "http://test", fetchWithHeaders: mocks.fetcher }) }));
let root: Root, container: HTMLDivElement;
let state: ReturnType<typeof useInferenceAvailability>;
let active: boolean;
function Probe({ open }: { open: boolean }) { state = useInferenceAvailability(open); return <span>{String(state.blocked)}</span>; }
beforeEach(() => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  vi.useFakeTimers();
  active = true;
  mocks.fetcher.mockReset().mockImplementation(async (url: string) => {
    if (url.endsWith("/stop-inference")) active = false;
    return { ok: true, json: async () => ({ inference_active: active, session_phase: "paused" }) };
  });
  container = document.createElement("div"); root = createRoot(container);
});
afterEach(() => { act(() => root.unmount()); vi.useRealTimers(); });
const render = async (open = true) => { await act(async () => root.render(<Probe open={open} />)); };

test("paused session blocks cameras and new Start until Stop confirms exit", async () => {
  await render();
  expect(state.status?.session_phase).toBe("paused");
  expect(state.blocked).toBe(true);
  await act(async () => state.stopPrevious());
  expect(state.blocked).toBe(false);
  expect(mocks.fetcher.mock.calls.filter(([url]) => url.endsWith("/stop-inference"))).toHaveLength(1);
});

test("failed status request does not expose cameras; reopening rechecks ownership", async () => {
  mocks.fetcher.mockRejectedValueOnce(new Error("offline"));
  await render();
  expect(state.blocked).toBe(true);
  expect(state.error).toContain("offline");
  active = false;
  await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
  expect(state.blocked).toBe(false);
  await render(false);
  expect(state.blocked).toBe(true);
  active = true;
  await render(true);
  expect(state.blocked).toBe(true);
});

test("late pre-stop status cannot restore stale active state after release", async () => {
  await render();
  let finishOld: (value: unknown) => void;
  mocks.fetcher.mockImplementationOnce(() => new Promise(resolve => { finishOld = resolve; }));
  await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
  await act(async () => state.stopPrevious());
  expect(state.blocked).toBe(false);
  await act(async () => finishOld!({ ok: true, json: async () => ({ inference_active: true }) }));
  expect(state.blocked).toBe(false);
});
