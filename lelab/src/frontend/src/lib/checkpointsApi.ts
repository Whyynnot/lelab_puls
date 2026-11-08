import { ParameterDescription, InferenceModes } from "./nativeParameters";
import { Fetcher, apiRequest } from "./apiClient";

export interface JobCheckpoint {
  step: number;
  source: "local" | "hub";
  ref: string;
}

/** Camera name → the resolution a checkpoint was trained or a dataset recorded at. */
export type ImageFeatures = Record<string, { height: number; width: number }>;

/** Reserved model-padding slots must never become physical capture requirements. */
export const isPlaceholderCamera = (name: string) => /^empty_camera_\d+$/.test(name);

export function physicalImageFeatures(features: ImageFeatures): ImageFeatures {
  return Object.fromEntries(Object.entries(features).filter(([name]) => !isPlaceholderCamera(name)));
}

export interface PolicyConfigSummary {
  policy_type: string | null;
  image_features: ImageFeatures;
  requires_task: boolean;
  camera_selection?: { optional: boolean; min_cameras: number };
  inference_parameters?: ParameterDescription;
  inference_modes?: InferenceModes;
}

export async function listJobCheckpoints(
  baseUrl: string,
  fetcher: Fetcher,
  jobId: string,
  signal?: AbortSignal,
): Promise<JobCheckpoint[]> {
  const body = await apiRequest<{ checkpoints: JobCheckpoint[] }>(
    baseUrl,
    fetcher,
    `/jobs/${jobId}/checkpoints`,
    { signal, action: "List checkpoints" },
  );
  return body.checkpoints;
}

export async function getCheckpointPolicyConfig(
  baseUrl: string,
  fetcher: Fetcher,
  jobId: string,
  step: number,
  signal?: AbortSignal,
): Promise<PolicyConfigSummary> {
  const body = await apiRequest<PolicyConfigSummary>(
    baseUrl,
    fetcher,
    `/jobs/${jobId}/checkpoints/${step}/policy-config`,
    { signal, action: "Load policy config" },
  );
  return { ...body, image_features: physicalImageFeatures(body.image_features) };
}
