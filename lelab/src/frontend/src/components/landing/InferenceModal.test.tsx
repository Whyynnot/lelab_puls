// @vitest-environment jsdom
import React, { act } from "react";
import { createRoot, Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import InferenceModal from "./InferenceModal";
import { RobotRecord } from "@/hooks/useRobots";
const mocks = vi.hoisted(() => ({ fetcher: vi.fn(), navigate: vi.fn(), toast: vi.fn() }));
const available = [
  { index: 1, name: "top", deviceId: "usb-top", available: true },
  { index: 2, name: "wrist", deviceId: "usb-wrist", available: true },
  { index: 3, name: "manual", deviceId: "usb-manual", available: true },
];
vi.mock("@/contexts/ApiContext", () => ({ useApi: () => ({ baseUrl: "http://test", fetchWithHeaders: mocks.fetcher }) }));
vi.mock("@/hooks/use-toast", () => ({ useToast: () => ({ toast: mocks.toast }) }));
vi.mock("react-router-dom", () => ({ useNavigate: () => mocks.navigate }));
vi.mock("@/hooks/useAvailableCameras", () => ({ useAvailableCameras: () => ({ cameras: available }) }));
vi.mock("@/hooks/useCameraStream", () => ({ useCameraStream: () => ({ videoRef: React.createRef<HTMLVideoElement>(), hasError: false }) }));
// Use native selectors to test modal state changes without Radix popup event details.
vi.mock("@/components/ui/select", () => ({
  Select: ({ value, onValueChange, disabled, children }: { value: string; onValueChange: (value: string) => void; disabled: boolean; children: React.ReactNode }) => (
    <select value={value ?? ""} disabled={disabled} onChange={(event) => onValueChange(event.target.value)}><option value="" />{children}</select>
  ),
  SelectContent: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  SelectItem: ({ value, children }: { value: string; children: React.ReactNode }) => <option value={value}>{children}</option>,
  SelectTrigger: () => null,
  SelectValue: () => null,
}));
vi.mock("@/components/ui/dialog", () => ({
  Dialog: ({ open, children }: { open: boolean; children: React.ReactNode }) => open ? <div>{children}</div> : null,
  DialogContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogHeader: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogTitle: ({ children }: { children: React.ReactNode }) => <h2>{children}</h2>,
  DialogDescription: ({ children }: { children: React.ReactNode }) => <p>{children}</p>,
}));
const robot: RobotRecord = {
  name: "so101", leader_port: "leader", follower_port: "follower", leader_config: "leader.json", follower_config: "follower.json", is_clean: true,
  cameras: [
    { id: "wrist", name: "wrist", type: "opencv", device_id: "usb-wrist", camera_index: 1, width: 640, height: 480, rotation: 90 },
    { id: "top", name: "top", type: "opencv", device_id: "usb-top", camera_index: 2, width: 640, height: 480, rotation: 180 },
  ],
};
let root: Root;
let container: HTMLDivElement;
beforeEach(() => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  container = document.createElement("div"); root = createRoot(container);
  mocks.fetcher.mockReset(); mocks.navigate.mockReset(); mocks.toast.mockReset();
  mocks.fetcher.mockImplementation(async (url: string) => new Response(JSON.stringify(
    url.endsWith("/inference-status") ? { inference_active: false }
      : url.endsWith("/checkpoints") ? { checkpoints: [{ step: 10, source: "local", ref: "checkpoint" }] }
      : url.endsWith("/policy-config") ? { policy_type: "act", requires_task: false, image_features: { wrist: { width: 480, height: 640 }, top: { width: 640, height: 480 } } }
      : { message: "Started", log_path: "rollout.log" },
  )));
});
afterEach(() => { act(() => root.unmount()); vi.useRealTimers(); });
const rotations = () => Array.from(container.querySelectorAll("select")).filter((select) => Array.from(select.options).some((option) => option.text.includes("clockwise")));
const bindings = () => Array.from(container.querySelectorAll("select")).filter((select) => Array.from(select.options).some((option) => option.text.includes("#")));
function change(select: HTMLSelectElement, value: string) {
  act(() => { select.value = value; select.dispatchEvent(new Event("change", { bubbles: true })); });
}
async function render(record = robot) {
  await act(async () => root.render(<InferenceModal open onOpenChange={vi.fn()} robot={record} jobId="job" initialStep={10} />));
}
async function start() {
  vi.useFakeTimers();
  const button = Array.from(container.querySelectorAll("button")).find((item) => item.textContent?.includes("Start Inference"))!;
  await act(async () => { button.click(); await vi.advanceTimersByTimeAsync(300); });
  const call = mocks.fetcher.mock.calls.find(([url]) => url.endsWith("/start-inference"));
  return call ? JSON.parse(call[1].body) : null;
}

test("camera reassignment carries physical rotation and sends native dimensions to inference", async () => {
  await render();
  expect(bindings().map((select) => select.value)).toEqual(["2", "1"]);
  expect(rotations().map((select) => select.value)).toEqual(["90", "180"]);
  change(bindings()[0], "1"); change(bindings()[1], "2");
  expect(rotations().map((select) => select.value)).toEqual(["180", "90"]);
  change(rotations()[0], "270");
  const request = await start();
  expect(request.cameras.wrist).toMatchObject({ camera_index: 1, rotation: 270, width: 640, height: 480 });
  expect(request.cameras.top).toMatchObject({ camera_index: 2, rotation: 90, width: 480, height: 640 });
  expect(mocks.navigate).toHaveBeenCalledWith("/inference");
});

test("an unsaved camera permits a manual rotation with a training-data reminder", async () => {
  await render(); change(bindings()[0], "3");
  expect(rotations()[0].value).toBe("0");
  change(rotations()[0], "90");
  expect(container.textContent).toContain("Use the same rotation as the training data");
  const request = await start();
  expect(request.cameras.wrist).toMatchObject({ camera_index: 3, rotation: 90, width: 640, height: 480 });
});

test("an invalid saved rotation blocks inference and gives a visible correction instruction", async () => {
  await render({ ...robot, cameras: [{ ...robot.cameras[0], rotation: 45 as never }] });
  expect(container.textContent).toContain("Correct the camera settings in Calibration");
  const button = Array.from(container.querySelectorAll("button")).find((item) => item.textContent?.includes("Start Inference"))! as HTMLButtonElement;
  expect(button.disabled).toBe(true);
  expect(await start()).toBeNull();
});

test("checkpoint defaults and edited native parameters are sent through the original start API", async () => {
  const original=mocks.fetcher.getMockImplementation()!;
  mocks.fetcher.mockImplementation(async (url: string, options?: RequestInit) => {
    if (url.endsWith("/policy-config")) return new Response(JSON.stringify({policy_type:"smolvla",requires_task:true,
      image_features:{wrist:{width:480,height:640},top:{width:640,height:480}},inference_parameters:{
        policy_type:"smolvla",backend:"sync",guided:false,effective:{chunk_size:50,n_action_steps:50,fps:30},
        limits:{max_chunk_size:50,min_chunk_size:1,chunk_size_editable:true,n_action_steps_editable:true,min_fps:1,max_fps:60},note:"sync"}}));
    return original(url,options);
  });
  await render();
  for (const [label,value] of [["预测 chunk M","20"],["每轮执行 N 步","10"],["动作 FPS","15"]]) {
    const input=container.querySelector(`input[aria-label="${label}"]`) as HTMLInputElement;
    act(() => {
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,"value")!.set!.call(input,value);
      input.dispatchEvent(new Event("input",{bubbles:true}));
    });
  }
  const request=await start();
  expect(request.inference_parameters).toEqual({chunk_size:20,n_action_steps:10,fps:15});
  expect(request.cameras.wrist.fps).toBe(30);
});

test("SmolVLA RTC selection locks N to M and reaches the native start API", async () => {
  const original=mocks.fetcher.getMockImplementation()!;
  const sync={policy_type:"smolvla",backend:"sync",guided:false,
    effective:{chunk_size:50,n_action_steps:10,fps:30},
    limits:{max_chunk_size:50,min_chunk_size:1,chunk_size_editable:true,n_action_steps_editable:true,min_fps:1,max_fps:60},note:"sync"};
  const rtc={...sync,backend:"rtc",guided:true,effective:{...sync.effective,n_action_steps:50},
    limits:{...sync.limits,min_chunk_size:2,n_action_steps_editable:false},note:"rtc"};
  mocks.fetcher.mockImplementation(async (url: string, options?: RequestInit) => {
    if (url.endsWith("/policy-config")) return new Response(JSON.stringify({policy_type:"smolvla",requires_task:true,
      image_features:{wrist:{width:480,height:640},top:{width:640,height:480}},inference_parameters:sync,
      inference_modes:{default:"auto",rtc_unavailable_reason:"",options:[
        {value:"auto",label:"自动",parameters:sync},{value:"sync",label:"普通",parameters:sync},
        {value:"rtc",label:"RTC",parameters:rtc}]}}));
    return original(url,options);
  });
  await render();
  const mode=container.querySelector('select[aria-label="推理模式"]') as HTMLSelectElement;
  change(mode,"rtc");
  const steps=container.querySelector('input[aria-label="每轮执行 N 步"]') as HTMLInputElement;
  expect(steps.disabled).toBe(true);expect(steps.value).toBe("50");
  const chunk=container.querySelector('input[aria-label="预测 chunk M"]') as HTMLInputElement;
  act(() => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,"value")!.set!.call(chunk,"20");
    chunk.dispatchEvent(new Event("input",{bubbles:true}));
  });
  expect(steps.value).toBe("20");
  change(mode,"sync");expect(steps.disabled).toBe(false);expect(steps.value).toBe("10");
  change(mode,"rtc");
  const request=await start();
  expect(request.inference_mode).toBe("rtc");
  expect(request.inference_parameters).toEqual({chunk_size:20,n_action_steps:20,fps:30});
});

function optionalSmolConfig() {
  const original = mocks.fetcher.getMockImplementation()!;
  mocks.fetcher.mockImplementation(async (url: string, options?: RequestInit) => {
    if (url.endsWith("/policy-config")) return new Response(JSON.stringify({
      policy_type: "smolvla", requires_task: true,
      camera_selection: { optional: true, min_cameras: 1 },
      image_features: Object.fromEntries(["camera1", "camera2", "camera3"].map(
        (name) => [name, { width: 640, height: 480 }],
      )),
    }));
    return original(url, options);
  });
}
function enableCamera(name: string) {
  const input = container.querySelector(`input[aria-label="启用 ${name}"]`) as HTMLInputElement;
  act(() => input.click());
}
function startButton() {
  return Array.from(container.querySelectorAll("button")).find(
    (item) => item.textContent?.includes("Start Inference"),
  )! as HTMLButtonElement;
}

test("SmolVLA can disable camera3 and send only the two physical camera inputs", async () => {
  optionalSmolConfig(); await render();
  change(bindings()[0], "1"); change(bindings()[1], "2");
  expect(startButton().disabled).toBe(true);
  enableCamera("camera3");
  expect(bindings()[2].disabled).toBe(true);
  expect(startButton().disabled).toBe(false);
  const request = await start();
  expect(Object.keys(request.cameras)).toEqual(["camera1", "camera2"]);
  // Rotation is applied after capture; do not request an unsupported 480x640 mode.
  expect(request.cameras.camera2).toMatchObject({ camera_index: 2, rotation: 90, width: 640, height: 480 });
});

test("optional camera selection rejects zero views and duplicate physical devices", async () => {
  optionalSmolConfig(); await render();
  change(bindings()[0], "1"); change(bindings()[1], "1");
  enableCamera("camera3");
  expect(startButton().disabled).toBe(true);
  expect(container.textContent).toContain("每路输入请选择不同的物理相机");
  enableCamera("camera2");
  expect(startButton().disabled).toBe(false);
  enableCamera("camera1");
  expect(startButton().disabled).toBe(true);
  expect(container.textContent).toContain("请至少启用一路相机");
  expect(await start()).toBeNull();
});

test("reenabling the unbound third camera requires a binding again", async () => {
  optionalSmolConfig(); await render();
  change(bindings()[0], "1"); change(bindings()[1], "2");
  enableCamera("camera3"); expect(startButton().disabled).toBe(false);
  enableCamera("camera3"); expect(startButton().disabled).toBe(true);
  change(bindings()[2], "3"); expect(startButton().disabled).toBe(false);
  expect(Object.keys((await start()).cameras)).toHaveLength(3);
});

test("ACT keeps its original mandatory camera bindings", async () => {
  await render();
  expect(container.querySelector('input[aria-label^="启用 camera"]')).toBeNull();
  expect(Object.keys((await start()).cameras)).toEqual(["wrist", "top"]);
});

function piConfig(policy_type: string, optional = true) {
  const original = mocks.fetcher.getMockImplementation()!;
  mocks.fetcher.mockImplementation(async (url: string, options?: RequestInit) => {
    if (url.endsWith("/policy-config")) return new Response(JSON.stringify({
      policy_type, requires_task: true, camera_selection: { optional, min_cameras: 1 },
      image_features: { top: { width: 640, height: 480 }, wrist: { width: 640, height: 480 },
        empty_camera_0: { width: 224, height: 224 } },
    }));
    return original(url, options);
  });
}

test.each(["pi05", "pi0"])("%s only needs its physical cameras and never transmits generated padding", async (policy) => {
  piConfig(policy); await render();
  expect(container.textContent).not.toContain("empty_camera_0");
  expect(bindings()).toHaveLength(2);
  expect(startButton().disabled).toBe(false);
  const request = await start();
  expect(Object.keys(request.cameras)).toEqual(["top", "wrist"]);
  expect(request.cameras.wrist).toMatchObject({ width: 640, height: 480, rotation: 90 });
});

test("legacy Pi05 API responses still cannot require a placeholder camera", async () => {
  piConfig("pi05", false); await render();
  expect(bindings()).toHaveLength(2);
  expect(startButton().disabled).toBe(false);
  expect(Object.keys((await start()).cameras)).toEqual(["top", "wrist"]);
});

test("Pi05 can explicitly omit a real view using its native missing-image handling", async () => {
  piConfig("pi05"); await render(); enableCamera("wrist");
  expect(startButton().disabled).toBe(false);
  expect(Object.keys((await start()).cameras)).toEqual(["top"]);
});
