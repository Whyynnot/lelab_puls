import { ParameterValues, InferenceMode } from "./nativeParameters";
import { Fetcher, apiRequest } from "./apiClient";
import { CameraRequest } from "./cameraConfig";

export interface StartInferenceRequest {
  follower_port: string;
  follower_config: string;
  policy_ref: string;
  task: string;
  cameras: Record<string, CameraRequest>;
  duration_s: number;
  inference_parameters?: ParameterValues;
  inference_mode?: InferenceMode;
}

export interface InferenceStatus {
  inference_active: boolean;
  native_session_id?: string;
  session_phase?: string;
  segment_elapsed_s?: number;
  started_at: number | null;
  rollout_started_at: number | null;
  elapsed_s: number;
  rollout_elapsed_s: number;
  duration_s: number | null;
  policy_ref: string | null;
  log_path: string | null;
  exited?: boolean;
  exit_code?: number | null;
  error?: string | null;
  hint?: string | null;
}

export async function startInference(
  baseUrl: string,
  fetcher: Fetcher,
  request: StartInferenceRequest,
): Promise<{ message: string; log_path: string; cancelled?: boolean }> {
  return apiRequest<{ message: string; log_path: string; cancelled?: boolean }>(
    baseUrl,
    fetcher,
    "/start-inference",
    { method: "POST", body: request, action: "Start inference" },
  );
}

export async function stopInference(
  baseUrl: string,
  fetcher: Fetcher,
): Promise<{ message: string }> {
  return apiRequest<{ message: string }>(baseUrl, fetcher, "/stop-inference", {
    method: "POST",
    action: "Stop inference",
  });
}

export async function getInferenceStatus(
  baseUrl: string,
  fetcher: Fetcher,
  signal?: AbortSignal,
): Promise<InferenceStatus> {
  return apiRequest<InferenceStatus>(baseUrl, fetcher, "/inference-status", {
    signal,
    action: "Get inference status",
  });
}

/** Do not release the UI's device gate until the backend confirms exit. */
export async function stopInferenceAndWait(
  baseUrl: string,
  fetcher: Fetcher,
): Promise<InferenceStatus> {
  const before = await getInferenceStatus(baseUrl, fetcher);
  if (!before.inference_active) return before;
  try {
    await stopInference(baseUrl, fetcher);
  } catch (error) {
    // A natural exit can race with Stop. Other failures must remain visible.
    const current = await getInferenceStatus(baseUrl, fetcher);
    if (!current.inference_active) return current;
    throw error;
  }
  const deadline = Date.now() + 10000;
  while (true) {
    const status = await getInferenceStatus(baseUrl, fetcher);
    if (!status.inference_active) return status;
    if (Date.now() >= deadline) throw new Error("旧推理尚未退出，设备仍被占用。请重试停止，或返回当前推理页面检查。");
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
}
