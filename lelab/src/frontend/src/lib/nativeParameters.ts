export interface ParameterValues { chunk_size: number | null; n_action_steps: number | null; fps: number }
export type InferenceMode = "auto" | "sync" | "rtc";
export interface InferenceModes {
  default: InferenceMode;
  options: { value: InferenceMode; label: string; parameters: ParameterDescription }[];
  rtc_unavailable_reason: string;
}
export interface ParameterDescription {
  policy_type?: string; backend: string; guided: boolean; revision?: number;
  effective: ParameterValues;
  limits: { max_chunk_size: number | null; min_chunk_size: number; chunk_size_editable: boolean;
    n_action_steps_editable: boolean; min_fps: number; max_fps: number };
  note: string;
}
export interface ParameterDraft { chunk_size: string; n_action_steps: string; fps: string }
export function parameterDraft(description: ParameterDescription): ParameterDraft {
  return { chunk_size: String(description.effective.chunk_size ?? ""),
    n_action_steps: String(description.effective.n_action_steps ?? ""), fps: String(description.effective.fps) };
}
export function parameterValues(draft: ParameterDraft, description: ParameterDescription): ParameterValues {
  const chunk = draft.chunk_size === "" ? null : Number(draft.chunk_size);
  const steps = draft.n_action_steps === "" ? null : Number(draft.n_action_steps);
  const fps = Number(draft.fps);
  if (chunk != null && (!Number.isInteger(chunk) || chunk < description.limits.min_chunk_size || chunk > (description.limits.max_chunk_size ?? 0)))
    throw new Error(`预测 M 必须为 ${description.limits.min_chunk_size}–${description.limits.max_chunk_size} 的整数`);
  if (steps != null && (!Number.isInteger(steps) || steps < 1 || (chunk != null && steps > chunk)))
    throw new Error("执行步数必须满足 1 ≤ N ≤ M");
  if ((description.effective.chunk_size != null && chunk == null) || (description.effective.n_action_steps != null && steps == null))
    throw new Error("请填写 M 和 N");
  if (!Number.isFinite(fps) || fps < 1 || fps > 60 || draft.fps.trim() === "") throw new Error("动作 FPS 必须为 1–60");
  return { chunk_size: chunk, n_action_steps: steps, fps };
}
