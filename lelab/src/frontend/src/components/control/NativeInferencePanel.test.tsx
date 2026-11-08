// @vitest-environment jsdom
import React, { act } from "react";
import { createRoot, Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import NativeInferencePanel, { NativeState } from "./NativeInferencePanel";

const mocked = vi.hoisted(() => ({ fetcher: vi.fn() }));
vi.mock("@/contexts/ApiContext", () => ({ useApi: () => ({ baseUrl: "http://localhost:8000", fetchWithHeaders: mocked.fetcher }) }));
let root: Root, container: HTMLDivElement, state: NativeState;
const id = "a".repeat(32);
const keys = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"].map(k => `${k}.pos`);
beforeEach(() => {
  vi.useFakeTimers();
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  Object.defineProperty(document, "hidden", { configurable: true, value: false });
  state = { session_id: id, phase: "running", task: "pick red block", joint_keys: keys,
    observed_at: Date.now() / 1000, actual: Object.fromEntries(keys.map(k => [k, 1])),
    target: Object.fromEntries(keys.map(k => [k, 2])) };
  mocked.fetcher.mockReset().mockImplementation(async (url: string, options?: RequestInit) => {
    if (url.endsWith("/command")) {
      const body = JSON.parse(options!.body as string);
      if (body.action === "reset") state = { ...state, phase: "resetting", ack: "cmd" };
      else state = { ...state, phase: "running", task: body.task, ack: "cmd" };
      return { ok: true, json: async () => ({ id: "cmd" }) };
    }
    return { ok: true, json: async () => url.endsWith("/status") ? { ...state } : { ok: true } };
  });
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});
afterEach(() => { act(() => root.unmount()); container.remove(); vi.useRealTimers(); });
async function render() { await act(async () => { root.render(<NativeInferencePanel sessionId={id} />); }); }
async function poll() { await act(async () => { await vi.advanceTimersByTimeAsync(550); }); }
const button = (text: string) => Array.from(container.querySelectorAll("button")).find(b => b.textContent === text)!;

test("reset waits for completion; keeps prompt; edited prompt is sent only with continue", async () => {
  await render();
  const reset = button("复位到推理初始位置"), resume = button("继续推理");
  expect(reset.disabled).toBe(false);
  expect(resume.disabled).toBe(true);
  expect(container.querySelector("textarea")!.value).toBe("pick red block");
  await act(async () => reset.click());
  await poll();
  expect(resume.disabled).toBe(true);
  state = { ...state, phase: "paused" };
  await poll();
  expect(resume.disabled).toBe(false);
  const input = container.querySelector("textarea")!;
  act(() => {
    Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")!.set!.call(input, "put it on plate");
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
  await poll();
  expect(input.value).toBe("put it on plate");
  const calls = mocked.fetcher.mock.calls.filter(([url]) => url.endsWith("/command"));
  expect(calls).toHaveLength(1);
  await act(async () => resume.click());
  const last = mocked.fetcher.mock.calls.filter(([url]) => url.endsWith("/command")).pop()!;
  expect(JSON.parse(last[1].body)).toEqual({ session_id: id, action: "resume", task: "put it on plate" });
  expect(mocked.fetcher.mock.calls.every(([url]) => !url.includes("/api/deploy/"))).toBe(true);
});

test("six joint charts distinguish feedback from policy output; monitor can be disabled", async () => {
  await render();
  expect(container.querySelectorAll('svg[role="img"]')).toHaveLength(6);
  expect(container.textContent).toContain("策略输出目标");
  expect(container.textContent).toContain("预测输入 state");
  expect(container.textContent).toContain("驱动下发目标");
  expect(container.textContent).toContain("实际反馈");
  const checkbox = container.querySelector('input[type="checkbox"]') as HTMLInputElement;
  await act(async () => checkbox.click());
  expect(container.querySelectorAll('svg[role="img"]')).toHaveLength(0);
  const viewer = mocked.fetcher.mock.calls.filter(([url]) => url.endsWith("/viewer")).pop()!;
  expect(JSON.parse(viewer[1].body).enabled).toBe(false);
});

test("failed reset and stale session status cannot enable continue", async () => {
  state = { ...state, phase: "error", error: "reset_failed" };
  await render();
  expect(button("继续推理").disabled).toBe(true);
  expect(container.querySelector('[role="alert"]')!.textContent).toContain("复位尚未通过初始位置确认");
  state = { ...state, phase: "paused", session_id: "b".repeat(32) };
  await poll();
  expect(button("继续推理").disabled).toBe(true);
});

function editPrompt(text: string) {
  const input = container.querySelector("textarea")!;
  act(() => {
    Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")!.set!.call(input, text);
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
  return input;
}

test("both effective prompt labels use the server task even without a prediction snapshot", async () => {
  state = { ...state, task: "pick yellow block", inference_count: 0 };
  await render();
  expect(container.textContent).toContain("当前生效 Prompt：pick yellow block");
  expect(container.textContent).toContain("当前生效输入 Prompt：pick yellow block");
  expect(container.textContent).toContain("预测输入快照尚未采集");
  editPrompt("unsaved new instruction");
  await poll();
  expect(container.textContent).toContain("当前生效输入 Prompt：pick yellow block");
});

test("a previous preview prompt is labeled separately from the current effective prompt", async () => {
  state = { ...state, task: "pick yellow block", image_observations: { input: { task: "pick red block" } } };
  await render();
  expect(container.textContent).toContain("当前生效输入 Prompt：pick yellow block");
  expect(container.textContent).toContain("当前预览来自先前预测，该帧 Prompt：pick red block");
});

test("a failed reset allows prompt editing, preserves the draft on polls, and still blocks motion", async () => {
  state = { ...state, phase: "error", error: "reset_failed", reset_residual: 3.340659 };
  await render();
  const input = container.querySelector("textarea")!;
  expect(input.disabled).toBe(false);
  editPrompt("pick blue block");
  await poll(); await poll();
  expect(input.value).toBe("pick blue block");
  expect(container.querySelector('[role="alert"]')!.textContent).toContain("3.34");
  expect(button("继续推理").disabled).toBe(true);
  expect(mocked.fetcher.mock.calls.filter(([url]) => url.endsWith("/command"))).toHaveLength(0);
});

test("reset acknowledgment cannot overwrite an existing prompt draft", async () => {
  await render();
  const input = editPrompt("new task after reset");
  await act(async () => button("复位到推理初始位置").click());
  await poll();
  expect(input.disabled).toBe(false);
  expect(input.value).toBe("new task after reset");
  state = { ...state, phase: "paused" };
  await poll(); await poll();
  expect(input.value).toBe("new task after reset");
  await act(async () => button("继续推理").click());
  const last = mocked.fetcher.mock.calls.filter(([url]) => url.endsWith("/command")).pop()!;
  expect(JSON.parse(last[1].body).task).toBe("new task after reset");
});

test("edits during a pending resume are kept as the next draft", async () => {
  state = { ...state, phase: "paused" };
  await render();
  editPrompt("task submitted now");
  await act(async () => button("继续推理").click());
  const input = editPrompt("task for next run");
  await poll(); await poll();
  expect(input.value).toBe("task for next run");
  expect(container.textContent).toContain("当前生效 Prompt：task submitted now");
  expect(mocked.fetcher.mock.calls.filter(([url]) => url.endsWith("/command"))).toHaveLength(1);
});

const parameterDescription = {
  policy_type: "smolvla", backend: "sync", guided: false,
  effective: { chunk_size: 50, n_action_steps: 50, fps: 30 },
  limits: { max_chunk_size: 50, min_chunk_size: 1, chunk_size_editable: true,
    n_action_steps_editable: true, min_fps: 1, max_fps: 60 }, note: "native sync",
};
function editParameter(label: string, value: string) {
  const input = container.querySelector(`input[aria-label="${label}"]`) as HTMLInputElement;
  act(() => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!.call(input, value);
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
}
test("runtime parameters are editable after reset, survive polling and apply with continue", async () => {
  state = { ...state, parameters: parameterDescription };
  await render();
  expect((container.querySelector('input[aria-label="动作 FPS"]') as HTMLInputElement).disabled).toBe(true);
  state = { ...state, phase: "paused" };
  await poll();
  editParameter("预测 chunk M", "20"); editParameter("每轮执行 N 步", "10"); editParameter("动作 FPS", "15");
  await poll();
  expect((container.querySelector('input[aria-label="每轮执行 N 步"]') as HTMLInputElement).value).toBe("10");
  expect(mocked.fetcher.mock.calls.filter(([url]) => url.endsWith("/command"))).toHaveLength(0);
  await act(async () => button("继续推理").click());
  const call=mocked.fetcher.mock.calls.filter(([url]) => url.endsWith("/command")).pop()!;
  expect(JSON.parse(call[1].body).parameters).toEqual({chunk_size:20,n_action_steps:10,fps:15});
});
test("invalid N>M blocks continue, and a fixed ACT chunk is read-only", async () => {
  state = {...state, phase:"paused", parameters:parameterDescription};
  await render();
  editParameter("预测 chunk M", "10"); editParameter("每轮执行 N 步", "11");
  expect(button("继续推理").disabled).toBe(true);
  expect(container.textContent).toContain("1 ≤ N ≤ M");
  state = {...state, parameters:{...parameterDescription,policy_type:"act",
    limits:{...parameterDescription.limits,chunk_size_editable:false}}};
  await poll();
  expect((container.querySelector('input[aria-label="预测 chunk M"]') as HTMLInputElement).disabled).toBe(true);
});
