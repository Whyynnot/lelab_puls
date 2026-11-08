import React from "react";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { AvailableCamera } from "@/hooks/useAvailableCameras";
import { ImageFeatures } from "@/lib/checkpointsApi";
import BrowserCameraPreview from "@/components/control/BrowserCameraPreview";
import CameraRotationSelect from "@/components/control/CameraRotationSelect";
import { CameraRotation, rotatedDimensions } from "@/lib/cameraConfig";

interface CameraBindingsProps {
  imageFeatures: ImageFeatures;
  bindings: Record<string, number | null>;
  onBind: (name: string, index: number) => void;
  rotationFor: (name: string) => CameraRotation;
  onRotate: (name: string, rotation: CameraRotation) => void;
  availableCameras: AvailableCamera[];
  /** Drop the previews so the backend can open the same devices. */
  paused: boolean;
  optional?: boolean;
  isEnabled?: (name: string) => boolean;
  onEnabledChange?: (name: string, enabled: boolean) => void;
  captureFor?: (name: string) => { width: number; height: number };
}

/** One row per expected camera: its resolution, a physical camera picker and a preview. */
const CameraBindings: React.FC<CameraBindingsProps> = ({
  imageFeatures,
  bindings,
  onBind,
  rotationFor,
  onRotate,
  availableCameras,
  paused,
  optional = false,
  isEnabled,
  onEnabledChange,
  captureFor,
}) => (
  <>
    {Object.entries(imageFeatures).map(([name, dims]) => {
      const active = !optional || isEnabled?.(name) !== false;
      const value = bindings[name];
      const bound =
        active && value != null ? availableCameras.find((c) => c.index === value) : undefined;
      const rotation = rotationFor(name);
      const capture = captureFor?.(name) ?? rotatedDimensions(dims.width, dims.height, rotation);
      return (
        <div key={name} className="flex flex-wrap items-center gap-3">
          {optional && <input type="checkbox" aria-label={`启用 ${name}`} checked={active}
            onChange={(event) => onEnabledChange?.(name, event.target.checked)} />}
          <div className="flex-1">
            <Label className="text-sm font-medium text-gray-200">{name}</Label>
            <p className="text-xs text-gray-500">
              {active ? `${capture.width}×${capture.height}` : "不使用此视角"}
            </p>
          </div>
          <Select
            disabled={!active}
            value={value != null ? String(value) : undefined}
            onValueChange={(v) => onBind(name, Number(v))}
          >
            <SelectTrigger disabled={!active} className="bg-gray-800 border-gray-700 text-white w-56">
              <SelectValue placeholder="Select a camera" />
            </SelectTrigger>
            <SelectContent className="bg-gray-900 border-gray-700 text-white">
              {availableCameras.length === 0 ? (
                <div className="px-2 py-1.5 text-xs text-gray-500">
                  No cameras detected
                </div>
              ) : (
                availableCameras.map((cam) => (
                  <SelectItem key={cam.index} value={String(cam.index)}>
                    #{cam.index} — {cam.name}
                  </SelectItem>
                ))
              )}
            </SelectContent>
          </Select>
          <CameraRotationSelect
            rotation={rotation}
            disabled={!bound || paused}
            onChange={(angle) => onRotate(name, angle)}
          />
          {active && <BrowserCameraPreview
            deviceId={bound?.deviceId ?? ""}
            paused={paused}
            rotation={rotation}
            width={capture.width}
            height={capture.height}
            className="w-32 rounded border border-gray-700"
          />}
        </div>
      );
    })}
  </>
);

export default CameraBindings;
