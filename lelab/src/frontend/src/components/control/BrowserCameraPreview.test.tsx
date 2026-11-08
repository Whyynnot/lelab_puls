import { ApiProvider } from "@/contexts/ApiContext";
// @vitest-environment jsdom
import React, { act } from "react";
import { createRoot, Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import BrowserCameraPreview from "./BrowserCameraPreview";
import CameraRotationSelect from "./CameraRotationSelect";
vi.mock("@/hooks/useCameraStream", () => ({ useCameraStream: () => ({ videoRef: React.createRef<HTMLVideoElement>(), hasError: false }) }));
let root: Root;
let container: HTMLDivElement;
beforeEach(() => { Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true }); container = document.createElement("div"); root = createRoot(container); });
afterEach(() => act(() => root.unmount()));
test("the Rotation control exposes four clockwise values and emits a number", () => {
  const change = vi.fn();
  act(() => root.render(<CameraRotationSelect rotation={180} onChange={change} />));
  const select = container.querySelector("select")!;
  expect(container.querySelector("label")!.htmlFor).toBe(select.id);
  expect(select.value).toBe("180");
  expect(Array.from(select.options).map((option) => option.text)).toEqual(["0° clockwise", "90° clockwise", "180° clockwise", "270° clockwise"]);
  act(() => { select.value = "90"; select.dispatchEvent(new Event("change", { bubbles: true })); });
  expect(change).toHaveBeenCalledWith(90);
});
test("a quarter turn fits the whole non-square frame and uses actual stream dimensions", () => {
  act(() => root.render(<BrowserCameraPreview deviceId="wrist" rotation={90} width={640} height={480} />));
  const video = container.querySelector("video")!, wrapper = video.parentElement!;
  expect(wrapper.style.aspectRatio).toBe("480 / 640");
  expect(video.style.objectFit).toBe("contain");
  expect(video.style.maxWidth).toBe("none");
  expect(video.style.transform).toBe("translate(-50%, -50%) rotate(90deg)");
  expect(parseFloat(video.style.width)).toBeCloseTo(640 / 480 * 100);
  expect(parseFloat(video.style.height)).toBeCloseTo(480 / 640 * 100);
  Object.defineProperty(video, "videoWidth", { value: 1280 });
  Object.defineProperty(video, "videoHeight", { value: 720 });
  act(() => video.dispatchEvent(new Event("loadedmetadata")));
  expect(wrapper.style.aspectRatio).toBe("720 / 1280");
  expect(parseFloat(video.style.width)).toBeCloseTo(1280 / 720 * 100);
  expect(parseFloat(video.style.height)).toBeCloseTo(720 / 1280 * 100);
});
test("device reassignment does not reuse the old device's frame geometry", () => {
  act(() => root.render(<BrowserCameraPreview deviceId="A" rotation={270} />));
  const video = container.querySelector("video")!;
  Object.defineProperty(video, "videoWidth", { value: 1920 });
  Object.defineProperty(video, "videoHeight", { value: 1080 });
  act(() => video.dispatchEvent(new Event("loadedmetadata")));
  act(() => root.render(<BrowserCameraPreview deviceId="B" rotation={180} width={640} height={480} />));
  expect(video.parentElement!.style.aspectRatio).toBe("640 / 480");
  expect(video.style.width).toBe("100%");
  expect(video.style.transform).toContain("rotate(180deg)");
});

test("a WSL camera uses the backend MJPEG stream and releases it when paused", () => {
  act(() => root.render(<ApiProvider><BrowserCameraPreview deviceId="wsl-camera:2" /></ApiProvider>));
  expect(container.querySelector("video")).toBeNull();
  expect(container.querySelector("img")!.src).toBe("http://localhost:8000/api/deploy/camera/2/preview");
  act(() => root.render(<ApiProvider><BrowserCameraPreview deviceId="wsl-camera:2" paused /></ApiProvider>));
  expect(container.querySelector("img")).toBeNull();
  expect(container.textContent).toContain("Preview paused");
});

test('stable WSL identities survive index changes and failed previews offer retry', () => {
  act(() => root.render(<ApiProvider><BrowserCameraPreview deviceId="wsl-camera:head" /></ApiProvider>));
  expect(container.querySelector('img')!.src).toBe('http://localhost:8000/api/deploy/camera/head/preview');
  act(() => container.querySelector('img')!.dispatchEvent(new Event('error')));
  expect(container.textContent).toContain('预览失败');
  act(() => container.querySelector('button')!.click());
  expect(container.querySelector('img')!.src).toContain('/camera/head/preview?attempt=1');
});
