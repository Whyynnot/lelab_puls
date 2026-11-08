import { useCallback, useEffect, useRef, useState } from "react";
import { useApi } from "@/contexts/ApiContext";

export interface AvailableCamera {
  index: number;
  name: string;
  deviceId: string;
  available: boolean;
  role?: string | null;
}

const norm = (s: string) => s.toLowerCase().replace(/\s+/g, " ").trim();

interface UseAvailableCamerasOptions {
  /** When false, do nothing. Use to gate on modal open. */
  enabled?: boolean;
}

/**
 * Enumerates cv2 camera indices from `/available-cameras` and merges each
 * with the matching browser deviceId (by AVFoundation localizedName) so
 * callers can render a preview alongside the bound dropdowns. Refreshes on
 * USB hotplug.
 */
export function useAvailableCameras({
  enabled = true,
}: UseAvailableCamerasOptions = {}) {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [cameras, setCameras] = useState<AvailableCamera[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const nativeBackend = useRef(false);

  const refresh = useCallback(async (quiet = false): Promise<AvailableCamera[]> => {
    if (!quiet) setIsLoading(true);
    try {
      // A Windows browser cannot enumerate USB devices forwarded to WSL.
      const native = await fetchWithHeaders(`${baseUrl}/api/deploy/cameras`);
      if (native.ok) {
        const data = await native.json();
        nativeBackend.current = true;
        setCameras((previous) => JSON.stringify(previous) === JSON.stringify(data.cameras) ? previous : data.cameras);
        return data.cameras;
      }
      if (native.status !== 404) throw new Error('Camera inventory is temporarily unavailable');
      nativeBackend.current = false;

      // Need a permission grant before enumerateDevices() returns labels.
      try {
        const probe = await navigator.mediaDevices.getUserMedia({ video: true });
        probe.getTracks().forEach((t) => t.stop());
      } catch {
        // ignore — we'll still try to enumerate, just without labels
      }

      const browserDevices = (await navigator.mediaDevices.enumerateDevices())
        .filter((d) => d.kind === "videoinput")
        .map((d) => ({ deviceId: d.deviceId, label: d.label }));

      const r = await fetchWithHeaders(`${baseUrl}/available-cameras`);
      if (!r.ok) {
        setCameras([]);
        return [];
      }
      const data = await r.json();
      const backendCams: {
        index: number;
        name?: string;
        available: boolean;
      }[] = data.cameras ?? [];

      // Browser's MediaDeviceInfo.label starts with AVFoundation's localizedName
      // but Chrome often appends "(vendorId:productId)". Match by exact, then
      // prefix, then either-contains.
      const used = new Set<string>();
      const merged: AvailableCamera[] = backendCams.map((cam) => {
        const label = cam.name || `Camera ${cam.index}`;
        const target = norm(label);
        const candidates = browserDevices.filter(
          (d) => !used.has(d.deviceId) && d.label
        );
        const match =
          candidates.find((d) => norm(d.label) === target) ||
          candidates.find((d) => norm(d.label).startsWith(target)) ||
          candidates.find(
            (d) => norm(d.label).includes(target) || target.includes(norm(d.label))
          );
        if (match) used.add(match.deviceId);
        return {
          index: cam.index,
          name: label,
          deviceId: match?.deviceId ?? "",
          available: cam.available,
        };
      });
      setCameras(merged);
      return merged;
    } catch {
      setCameras([]);
      return [];
    } finally {
      if (!quiet) setIsLoading(false);
    }
  }, [baseUrl, fetchWithHeaders]);

  useEffect(() => {
    if (!enabled) return;
    refresh();
    const handler = () => refresh();
    navigator.mediaDevices?.addEventListener("devicechange", handler);
    // Windows browser devicechange does not report devices forwarded to WSL.
    const timer = window.setInterval(() => {
      if (nativeBackend.current) void refresh(true);
    }, 2000);
    return () => {
      navigator.mediaDevices?.removeEventListener("devicechange", handler);
      window.clearInterval(timer);
    };
  }, [enabled, refresh]);

  return { cameras, isLoading, refresh };
}
