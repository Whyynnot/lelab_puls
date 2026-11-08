// @vitest-environment jsdom
import React, { act } from "react";
import { createRoot, Root } from "react-dom/client";
import { beforeEach, afterEach, expect, test, vi } from "vitest";
import InferenceModal from "./InferenceModal";
import { RobotRecord } from "@/hooks/useRobots";

const mocks = vi.hoisted(() => ({ fetcher: vi.fn(), start: vi.fn(), navigate: vi.fn(), toast: vi.fn() }));
vi.mock("@/contexts/ApiContext", () => ({ useApi: () => ({ baseUrl: "http://test", fetchWithHeaders: mocks.fetcher }) }));
vi.mock("@/hooks/use-toast", () => ({ useToast: () => ({ toast: mocks.toast }) }));
vi.mock("react-router-dom", () => ({ useNavigate: () => mocks.navigate }));
vi.mock("@/lib/checkpointsApi", async importOriginal => ({
  ...await importOriginal<typeof import("@/lib/checkpointsApi")>(),
  listJobCheckpoints: async () => [{ step: 5000, ref: "owner/act@checkpoints/005000" }],
  getCheckpointPolicyConfig: async () => ({ policy_type: "act", image_features: { head: { width: 640, height: 480 } } }) }));
vi.mock("@/lib/inferenceApi", async importOriginal => ({ ...await importOriginal<typeof import("@/lib/inferenceApi")>(), startInference: mocks.start }));
vi.mock("@/hooks/useCameraBindings", () => ({ useCameraBindings: () => ({ allBound: true, bindings: { head: 0 }, availableCameras: [],
  bind: vi.fn(), rotationFor: () => 0, setRotation: vi.fn(), toCameraDict: () => ({ head: { camera_index: 0, fps: 30 } }) }) }));
vi.mock("./CameraBindings", () => ({ default: ({ paused }: { paused: boolean }) => <span data-testid="camera">{paused ? "paused" : "preview"}</span> }));
vi.mock("@/components/jobs/CheckpointDropdown", () => ({ default: () => <span>checkpoint</span> }));
vi.mock("@/components/ui/dialog", () => ({
  Dialog: ({ open, children }: React.PropsWithChildren<{ open: boolean }>) => open ? <div>{children}</div> : null,
  DialogContent: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  DialogHeader: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  DialogTitle: ({ children }: React.PropsWithChildren) => <h1>{children}</h1>,
  DialogDescription: ({ children }: React.PropsWithChildren) => <p>{children}</p>,
}));

let root: Root, container: HTMLDivElement, active: boolean, phase: string;
const robot = { name: "so101", is_clean: true, follower_port: "/dev/ttyACM1", follower_config: "so101-v2", cameras: [] } as unknown as RobotRecord;
beforeEach(() => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true }); vi.useFakeTimers();
  active = true;
  phase = 'paused';
  mocks.start.mockReset().mockResolvedValue({}); mocks.navigate.mockReset();
  mocks.fetcher.mockReset().mockImplementation(async (url: string) => {
    if (url.endsWith("/stop-inference")) active = false;
    return { ok: true, json: async () => ({ inference_active: active, session_phase: phase, policy_ref: "old ACT" }) };
  });
  container = document.createElement("div"); root = createRoot(container);
});
afterEach(() => { act(() => root.unmount()); vi.useRealTimers(); });
const render = async (open = true) => { await act(async () => root.render(<InferenceModal open={open} onOpenChange={vi.fn()} robot={robot} jobId="act" initialStep={5000} />)); };
const button = (text: string) => Array.from(container.querySelectorAll("button")).find(b => b.textContent === text)!;

test("configuration offers Stop for an old paused run and recovers its previews", async () => {
  await render();
  expect(button("Start Inference").disabled).toBe(true);
  expect(container.querySelector('[data-testid="camera"]')!.textContent).toBe("paused");
  await act(async () => button("结束上一轮推理并释放设备").click());
  expect(button("Start Inference").disabled).toBe(false);
  expect(container.querySelector('[data-testid="camera"]')!.textContent).toBe("preview");
  expect(mocks.start).not.toHaveBeenCalled();
});

test("reopening after a successful launch does not leave submitting or cameras stuck", async () => {
  active = false;
  await render();
  act(() => button("Start Inference").click());
  await act(async () => { await vi.advanceTimersByTimeAsync(300); });
  expect(mocks.start).toHaveBeenCalledWith("http://test", mocks.fetcher, expect.objectContaining({
    policy_ref: "owner/act@checkpoints/005000", cameras: { head: { camera_index: 0, fps: 30 } }, duration_s: 60,
  }));
  expect(mocks.navigate).toHaveBeenCalledWith("/inference");
  await render(false);
  await render(true);
  expect(button("Start Inference").disabled).toBe(false);
  expect(container.querySelector('[data-testid="camera"]')!.textContent).toBe("preview");
});

test("model preparation offers cancellation without claiming the arm is already running", async () => {
  phase = 'preparing';
  await render();
  expect(container.textContent).toContain('尚未启动机械臂推理');
  expect(button('返回当前推理')).toBeUndefined();
  await act(async () => button('取消模型准备并释放任务').click());
  expect(button('Start Inference').disabled).toBe(false);
  expect(mocks.start).not.toHaveBeenCalled();
});

test("cancelled start replies keep the configuration open without navigating", async () => {
  active = false;
  mocks.start.mockResolvedValue({cancelled:true,message:'Inference preparation cancelled'});
  await render();
  act(() => button('Start Inference').click());
  await act(async () => { await vi.advanceTimersByTimeAsync(300); });
  expect(mocks.navigate).not.toHaveBeenCalled();
  expect(button('Start Inference').disabled).toBe(false);
  expect(container.querySelector('[data-testid="camera"]')!.textContent).toBe('preview');
});

test("a late successful start reply after Stop cannot navigate back into the old run", async () => {
  active = false;
  let finish: (value: object) => void;
  mocks.start.mockImplementation(() => new Promise(resolve => { finish = resolve; }));
  await render();
  act(() => button('Start Inference').click());
  await act(async () => { await vi.advanceTimersByTimeAsync(300); });
  expect(button('Cancel').disabled).toBe(true);
  active = true; phase = 'preparing';
  await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
  await act(async () => button('取消模型准备并释放任务').click());
  await act(async () => finish!({message:'Inference started'}));
  expect(mocks.navigate).not.toHaveBeenCalled();
  expect(button('Start Inference').disabled).toBe(false);
});
