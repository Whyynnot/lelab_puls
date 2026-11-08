// @vitest-environment jsdom
import React, { act } from "react";
import { createRoot, Root } from "react-dom/client";
import { beforeEach, afterEach, expect, test, vi } from "vitest";
import Inference from "./Inference";

const mocks = vi.hoisted(() => ({ toast: vi.fn(), navigate: vi.fn(), fetcher: vi.fn(), getStatus: vi.fn(), stop: vi.fn() }));
vi.mock("@/contexts/ApiContext", () => ({ useApi: () => ({ baseUrl: "http://test", fetchWithHeaders: mocks.fetcher }) }));
vi.mock("@/hooks/use-toast", () => ({ useToast: () => ({ toast: mocks.toast }) }));
vi.mock("react-router-dom", () => ({ useNavigate: () => mocks.navigate }));
vi.mock("@/lib/inferenceApi", () => ({ getInferenceStatus: mocks.getStatus, stopInference: mocks.stop, stopInferenceAndWait: mocks.stop }));
vi.mock("@/components/control/NativeInferencePanel", () => ({ default: () => <div>native controls</div> }));

let root: Root, container: HTMLDivElement;
beforeEach(() => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  mocks.toast.mockReset(); mocks.navigate.mockReset(); mocks.getStatus.mockReset(); mocks.stop.mockReset();
  mocks.getStatus.mockResolvedValue({ inference_active: true, native_session_id: "a".repeat(32), session_phase: "paused" });
  container = document.createElement("div"); document.body.appendChild(container); root = createRoot(container);
});
afterEach(() => { act(() => root.unmount()); container.remove(); });
const render = async () => { await act(async () => root.render(<Inference />)); };

test("failed native inference surfaces the motor error and hint", async () => {
  mocks.getStatus.mockResolvedValue({ inference_active: false, exited: true, exit_code: 1, log_path: "inference.log",
    hint: "机械臂串口通信包校验失败", error: "motor ID 2: Checksum error." });
  await render();
  expect(mocks.toast).toHaveBeenCalledWith(expect.objectContaining({ description: expect.stringContaining("motor ID 2: Checksum error."), variant: "destructive" }));
  expect(mocks.toast.mock.calls[0][0].description).toContain("串口通信包校验失败");
  expect(mocks.navigate).toHaveBeenCalledWith("/");
});

test("back from a paused session requires confirmed Stop and awaits device release", async () => {
  await render();
  await act(async () => (container.querySelector('[aria-label="结束推理并返回首页"]') as HTMLButtonElement).click());
  expect(mocks.navigate).not.toHaveBeenCalled();
  expect(mocks.stop).not.toHaveBeenCalled();
  let finish: () => void;
  mocks.stop.mockImplementationOnce(() => new Promise<void>(resolve => { finish = resolve; }));
  const confirm = Array.from(document.querySelectorAll('[role="alertdialog"] button')).find(b => b.textContent === "Stop") as HTMLButtonElement;
  await act(async () => confirm.click());
  expect(container.textContent).toContain("正在停止并释放设备");
  expect(mocks.navigate).not.toHaveBeenCalled();
  await act(async () => finish!());
  expect(mocks.navigate).toHaveBeenCalledWith("/");
});

test("failed Stop keeps the session visible and permits retry", async () => {
  mocks.stop.mockRejectedValueOnce(new Error("stop rejected"));
  await render();
  await act(async () => Array.from(container.querySelectorAll("button")).find(b => b.textContent === "Stop / 结束推理")!.click());
  await act(async () => (Array.from(document.querySelectorAll('[role="alertdialog"] button')).find(b => b.textContent === "Stop") as HTMLButtonElement).click());
  expect(mocks.navigate).not.toHaveBeenCalled();
  expect(mocks.toast).toHaveBeenCalledWith(expect.objectContaining({ title: "Stop failed", description: "stop rejected" }));
  expect((Array.from(container.querySelectorAll("button")).find(b => b.textContent === "Stop / 结束推理") as HTMLButtonElement).disabled).toBe(false);
});
