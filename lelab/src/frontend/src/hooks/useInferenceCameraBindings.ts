import { useMemo, useState } from "react";
import { useCameraBindings } from "@/hooks/useCameraBindings";
import { RobotRecord } from "@/hooks/useRobots";
import { ImageFeatures, isPlaceholderCamera, physicalImageFeatures } from "@/lib/checkpointsApi";
import { rotatedDimensions } from "@/lib/cameraConfig";

const EMPTY: Record<string, boolean> = {};

/** Optional inputs are scoped to this inference selection; recording is unchanged. */
export function useInferenceCameraBindings(
  features: ImageFeatures | null,
  robot: RobotRecord | null,
  open: boolean,
  optional: boolean,
  scope: string,
) {
  const [selection, setSelection] = useState({ scope, disabled: EMPTY });
  const disabled = selection.scope === scope ? selection.disabled : EMPTY;
  const physical = useMemo(() => features == null ? null : physicalImageFeatures(features), [features]);
  const isEnabled = (name: string) => !isPlaceholderCamera(name) && (!optional || !disabled[name]);
  const activeFeatures = useMemo(() => physical == null ? null : Object.fromEntries(
    Object.entries(physical).filter(([name]) => !optional || !disabled[name]),
  ), [physical, optional, disabled]);
  const base = useCameraBindings(activeFeatures, robot, open);
  const activeNames = Object.keys(activeFeatures ?? {});
  const indices = activeNames.map((name) => base.bindings[name]).filter((i) => i != null);
  const error = !optional ? null
    : activeNames.length === 0 ? "请至少启用一路相机。"
    : new Set(indices).size !== indices.length ? "每路输入请选择不同的物理相机。" : null;
  const setEnabled = (name: string, enabled: boolean) => setSelection((previous) => ({
    scope,
    disabled: { ...(previous.scope === scope ? previous.disabled : {}), [name]: !enabled },
  }));
  const savedFor = (name: string) => {
    const index = base.bindings[name];
    const live = base.availableCameras.find((camera) => camera.index === index);
    return robot?.cameras.find((camera) => live?.deviceId && camera.device_id === live.deviceId)
      ?? robot?.cameras.find((camera) => camera.camera_index === index);
  };
  const captureFor = (name: string) => {
    const dims = features?.[name] ?? { width: 640, height: 480 };
    if (!optional) return rotatedDimensions(dims.width, dims.height, base.rotationFor(name));
    const saved = savedFor(name);
    return { width: saved?.width ?? dims.width, height: saved?.height ?? dims.height };
  };
  const toCameraDict = (fps: number) => {
    if (error) throw new Error(error);
    const cameras = base.toCameraDict(fps);
    if (optional) for (const [name, camera] of Object.entries(cameras)) {
      const saved = savedFor(name);
      Object.assign(camera, captureFor(name), { fps: saved?.fps ?? fps });
      if (saved?.fourcc) camera.fourcc = saved.fourcc;
      if (saved?.backend) camera.backend = saved.backend;
    }
    return cameras;
  };
  return { ...base, allBound: base.allBound && !error, isEnabled, setEnabled,
    activeCount: activeNames.length, error, captureFor, toCameraDict };
}
