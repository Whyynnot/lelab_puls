import { useEffect, useRef, useState } from "react";
import { useApi } from "@/contexts/ApiContext";
import { apiRequest } from "@/lib/apiClient";
import { Button } from "@/components/ui/button";
import NativeParameterFields from "./NativeParameterFields";
import { ParameterDescription, ParameterDraft, parameterDraft, parameterValues } from "@/lib/nativeParameters";

type Values = Record<string, number>;
type ImageInfo = { name: string; file: string; version: string };
export interface NativeState {
  session_id: string; phase: string; task?: string; ack?: string; pending?: boolean;
  error?: string; command_error?: string; age_s?: number; published_at?: number;
  reset_residual?: number;
  actual?: Values; target?: Values; driver_target?: Values; initial_position?: Values;
  joint_keys?: string[]; observed_at?: number; input_at?: number; input_task?: string;
  inference_count?: number; segment?: number;
  input_state?: number[];
  images?: Record<string, ImageInfo[]>;
  image_observations?: Record<string, { at?: number; task?: string }>;
  parameters?: ParameterDescription;
}
type Sample = { time: number; actual?: Values; target?: Values; driver_target?: Values };
const labels: Record<string, string> = {
  setting_up: "准备中", running: "推理中", reset_requested: "等待当前推理结束后复位",
  resetting: "正在返回初始位置", paused: "已复位，等待继续", resuming: "正在继续",
  error: "操作失败，保持等待", completed: "本轮完成", stopped: "已停止",
};
const number = (value?: number) => value == null ? "—" : value.toFixed(2);
const age = (timestamp?: number) => timestamp == null ? "等待数据" : `${Math.max(0, Date.now() / 1000 - timestamp).toFixed(1)} 秒前`;

function Preview({ sessionId, item }: { sessionId: string; item: ImageInfo }) {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [url, setUrl] = useState<string>();
  const [loadedVersion, setLoadedVersion] = useState<string>();
  useEffect(() => {
    const controller = new AbortController();
    let objectUrl: string | undefined;
    fetchWithHeaders(`${baseUrl}/api/native-inference/image/${sessionId}/${item.file}?v=${item.version}`,
      { signal: controller.signal }).then(async response => {
        if (!response.ok) throw new Error("Image unavailable");
        const blob = await response.blob();
        if (controller.signal.aborted) return;
        objectUrl = URL.createObjectURL(blob);
        setUrl(objectUrl);
        setLoadedVersion(item.version);
      }).catch(() => { /* A missing immutable frame must not be relabeled as current. */ });
    return () => { controller.abort(); if (objectUrl) URL.revokeObjectURL(objectUrl); };
  }, [sessionId, item.file, item.version, baseUrl, fetchWithHeaders]);
  return <figure className="min-w-0"><figcaption className="text-xs text-slate-400 mb-1">{item.name}</figcaption>
    {url && loadedVersion === item.version ? <img src={url} alt={item.name} className="w-full rounded bg-black aspect-[4/3] object-contain" />
      : <div className="aspect-[4/3] bg-black rounded text-slate-500 p-4">等待图像</div>}</figure>;
}

export function JointChart({ name, samples }: { name: string; samples: Sample[] }) {
  const series = ["actual", "target", "driver_target"] as const;
  const colors = ["#38bdf8", "#fbbf24", "#c084fc"];
  const values = samples.flatMap(s => series.map(key => s[key]?.[name])).filter((v): v is number => v != null && Number.isFinite(v));
  const min = Math.min(...values, 0) - 1, max = Math.max(...values, 1) + 1;
  const end = samples[samples.length - 1]?.time ?? 0, begin = end - 60;
  return <div className="border border-slate-700 rounded p-3 min-w-0">
    <div className="text-sm mb-1">{name.replace(".pos", "")} <span className="float-right text-sky-300">{number(samples[samples.length - 1]?.actual?.[name])}</span></div>
    <svg viewBox="0 0 360 110" role="img" aria-label={`${name} 关节曲线`} className="w-full">
      <path d="M30 5V90H355" fill="none" stroke="#475569" />
      <text x="0" y="14" fill="#94a3b8" fontSize="9">{max.toFixed(0)}</text>
      <text x="0" y="90" fill="#94a3b8" fontSize="9">{min.toFixed(0)}</text>
      <text x="30" y="105" fill="#94a3b8" fontSize="9">−60s</text>
      <text x="310" y="105" fill="#94a3b8" fontSize="9">最新采样</text>
      {series.map((key, index) => {
        // A missing feedback sample breaks the line; do not fabricate a held measurement.
        const segments: string[][] = [[]];
        for (const s of samples) {
          const v = s[key]?.[name];
          if (v == null || !Number.isFinite(v)) { if (segments[segments.length - 1].length) segments.push([]); continue; }
          segments[segments.length - 1].push(`${30 + (s.time - begin) / 60 * 325},${90 - (v - min) / (max - min) * 80}`);
        }
        return segments.map((points, part) => <polyline key={`${key}-${part}`} points={points.join(" ")} fill="none" stroke={colors[index]} strokeWidth="1.6" />);
      })}
    </svg></div>;
}

export default function NativeInferencePanel({ sessionId }: { sessionId: string }) {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [state, setState] = useState<NativeState | null>(null);
  const [prompt, setPrompt] = useState("");
  const draft = useRef("");
  const dirty = useRef(false);
  const [parameters, setParameters] = useState<ParameterDraft | null>(null);
  const parameterRef = useRef<ParameterDraft | null>(null);
  const parametersDirty = useRef(false);
  const [enabled, setEnabled] = useState(true);
  const [images, setImages] = useState(true);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const pending = useRef<{ id: string; action: "reset" | "resume"; task: string } | null>(null);
  const lastObserved = useRef<number>();
  const request = <T,>(path: string, options = {}) => apiRequest<T>(baseUrl, fetchWithHeaders, `/api/native-inference/${path}`, options);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    let lastLease = 0;
    const tick = async () => {
      try {
        if (!document.hidden) {
          if (Date.now() - lastLease > 1500) {
            await apiRequest(baseUrl, fetchWithHeaders, "/api/native-inference/viewer", {
              method: "POST", body: { session_id: sessionId, enabled, images },
            });
            lastLease = Date.now();
          }
          const next = await apiRequest<NativeState>(baseUrl, fetchWithHeaders, "/api/native-inference/status");
          if (cancelled || next.session_id !== sessionId) return;
          setState(next);
          if (next.parameters && !parametersDirty.current) {
            parameterRef.current = parameterDraft(next.parameters);
            setParameters(parameterRef.current);
          }
          if (!dirty.current) {
            draft.current = next.task ?? "";
            setPrompt(draft.current);
          }
          if (pending.current && next.ack === pending.current.id) {
            const acknowledged = pending.current;
            pending.current = null;
            setBusy(false);
            setError(next.command_error ?? "");
            // Reset acknowledgments must never discard an edited draft. A user
            // can also prepare the next prompt while resume is in flight.
            if (!next.command_error && acknowledged.action === "resume"
                && draft.current === acknowledged.task) dirty.current = false;
            if (!next.command_error && acknowledged.action === "resume") parametersDirty.current = false;
          }
          if (enabled && next.observed_at && next.observed_at !== lastObserved.current) {
            lastObserved.current = next.observed_at;
            setSamples(old => [...old, { time: next.observed_at!, actual: next.actual,
              target: next.target, driver_target: next.driver_target }].filter(s => s.time > next.observed_at! - 60).slice(-120));
          }
        }
      } catch (e) { if (!cancelled) setError(e instanceof Error ? e.message : String(e)); }
      if (!cancelled) timer = setTimeout(tick, 500);
    };
    tick();
    return () => { cancelled = true; clearTimeout(timer); };
  }, [baseUrl, fetchWithHeaders, sessionId, enabled, images]);

  async function command(action: "reset" | "resume") {
    if (busy) return;
    setBusy(true); setError("");
    try {
      const submittedTask = draft.current;
      const result = await request<{ id: string }>("command", { method: "POST", body: {
        session_id: sessionId, action, ...(action === "resume" ? { task: submittedTask } : {}),
        ...(action === "resume" && parameterRef.current && state?.parameters
          ? { parameters: parameterValues(parameterRef.current, state.parameters) } : {}),
      } });
      pending.current = { id: result.id, action, task: submittedTask };
    } catch (e) { setBusy(false); setError(e instanceof Error ? e.message : String(e)); }
  }
  const phase = state?.phase ?? "setting_up";
  const stale = (state?.age_s ?? 0) > 3;
  const locked = busy || state?.pending || stale;
  let parameterError = "";
  if (parameters && state?.parameters) {
    try { parameterValues(parameters, state.parameters); }
    catch (e) { parameterError = e instanceof Error ? e.message : String(e); }
  }
  const resetFailure = state?.error === "reset_failed" || state?.error === "reset_skipped";
  const sessionError = resetFailure
    ? `复位尚未通过初始位置确认${state?.reset_residual != null ? `（最大偏差 ${number(state.reset_residual)} 个设备位置单位）` : ""}。可以编辑 Prompt 草稿；请再次复位成功后继续推理。`
    : state?.error;
  const keys = state?.joint_keys ?? Object.keys(state?.actual ?? {});
  return <section className="mx-auto mt-6 w-full max-w-6xl rounded-lg border border-slate-700 bg-gray-900 p-5 space-y-5">
    <div className="flex flex-wrap items-center justify-between gap-3">
      <h2 className="text-lg font-semibold">原生模型推理控制与监控</h2>
      <span className="text-sm text-sky-300" role="status">{labels[phase] ?? phase}{stale ? " · 状态更新延迟" : ""}</span>
    </div>
    <p className="text-sm text-slate-400">复位将用约 3 秒返回本次推理启动时的位置，并暂停等待。请先确保回程路径没有障碍。模型与设备保持加载，继续时清空旧动作队列；每次继续重新计算本轮时长。</p>
    <div className="flex flex-wrap gap-3">
      <Button onClick={() => command("reset")} disabled={!!locked || !["running", "paused", "error"].includes(phase)} variant="secondary">复位到推理初始位置</Button>
      <Button onClick={() => command("resume")} disabled={!!locked || phase !== "paused" || !!parameterError} className="bg-green-600 hover:bg-green-700">继续推理</Button>
    </div>
    <label className="block text-sm">Prompt 草稿（点击“继续推理”时应用；默认沿用上一次）
      <textarea aria-label="推理 Prompt" value={prompt} onChange={e => { dirty.current = true; draft.current = e.target.value; setPrompt(e.target.value); }}
        disabled={!state} maxLength={4096} rows={2}
        className="mt-2 w-full rounded border border-slate-600 bg-gray-950 p-3 disabled:opacity-60" />
    </label>
    <p className="text-xs text-slate-400 break-words">当前生效 Prompt：{state?.task || "（无）"}。编辑草稿不会立即改变正在执行的任务。</p>
    {parameters && state?.parameters && <div className="space-y-2 border-t border-slate-700 pt-4">
      <h3 className="font-semibold">推理参数（复位成功后修改，点击继续时应用）</h3>
      <p className="text-xs text-slate-400">当前生效：M={state.parameters.effective.chunk_size ?? "—"}，N={state.parameters.effective.n_action_steps ?? "—"}，动作 FPS={state.parameters.effective.fps} · {state.parameters.backend}</p>
      <NativeParameterFields description={state.parameters} value={parameters}
        disabled={phase !== "paused" || !!locked}
        onChange={next => { parametersDirty.current = true; parameterRef.current = next; setParameters(next); }} />
      {parameterError && <p role="alert" className="text-red-300">{parameterError}</p>}
    </div>}
    <p className="text-xs text-slate-500">ACT 等不使用语言条件的策略不会因修改 prompt 改变行为；语言模型继续使用原生任务预处理。</p>
    {(error || sessionError) && <p role="alert" className="text-red-300 break-words">{error || sessionError}</p>}
    <div className="flex flex-wrap gap-5 text-sm">
      <label><input type="checkbox" checked={enabled} onChange={e => setEnabled(e.target.checked)} /> 开启监控</label>
      <label><input type="checkbox" checked={images} disabled={!enabled} onChange={e => setImages(e.target.checked)} /> 图像预览（最多 2 FPS）</label>
      <span className="text-slate-400">新动作预测次数：{state?.inference_count ?? 0} · 实际反馈：{age(state?.observed_at)}</span>
    </div>
    {enabled && <>
      {images && <div className="grid md:grid-cols-2 gap-5">{["input", "execution"].map(group => <div key={group}>
        <h3 className="text-sm mb-2">{group === "input" ? "本次策略预测输入（预处理前）" : "实际执行观测"} · {age(state?.image_observations?.[group]?.at)}</h3>
        <div className="grid grid-cols-2 gap-2">{(state?.images?.[group] ?? []).map(item => <Preview key={item.name} sessionId={sessionId} item={item} />)}</div>
        {group === "input" && <div className="text-xs text-slate-400 mt-2 break-words">
          <p>当前生效输入 Prompt：{state?.task || "（无）"}</p>
          {state?.image_observations?.input?.task == null
            ? <p>预测输入快照尚未采集；执行缓存动作期间，预览保留最近一次预测帧。</p>
            : state.image_observations.input.task !== state.task
              ? <p>当前预览来自先前预测，该帧 Prompt：{state.image_observations.input.task || "（无）"}。新预测采集后更新画面。</p>
              : <p>预览对应最近一次预测输入；执行缓存动作期间保留该帧。</p>}
        </div>}
      </div>)}</div>}
      <div className="overflow-x-auto"><table className="w-full text-sm text-right"><thead><tr>
        <th className="text-left p-2">关节</th><th>预测输入 state</th><th>实际反馈</th><th>策略输出目标</th><th>驱动下发目标</th><th>推理初始位置</th>
      </tr></thead><tbody>{keys.map((key, index) => <tr key={key} className="border-t border-slate-800">
        <td className="text-left p-2">{key.replace(".pos", "")}</td><td>{number(state?.input_state?.[index])}</td><td>{number(state?.actual?.[key])}</td>
        <td>{number(state?.target?.[key])}</td><td>{number(state?.driver_target?.[key])}</td><td>{number(state?.initial_position?.[key])}</td>
      </tr>)}</tbody></table></div>
      <p className="text-xs text-slate-400"><span className="text-sky-400">蓝：实际反馈</span> · <span className="text-amber-300">黄：当前策略输出目标</span> · <span className="text-purple-300">紫：驱动下发目标</span>。显示最近 60 秒的降采样数据，单位沿用设备配置（SO101 通常为关节度数、夹爪百分比）。暂停时保留最后观测并显示数据年龄，不补造反馈。</p>
      <div className="grid sm:grid-cols-2 xl:grid-cols-3 gap-3">{keys.map(key => <JointChart key={key} name={key} samples={samples} />)}</div>
      <p className="text-xs text-slate-500">监控读取原有观测和 CPU 动作结果，后台编码最新帧；不会重新推理、打开相机或额外轮询电机。它是观测监控，不进行目标动作渲染或碰撞预测。</p>
    </>}
  </section>;
}
