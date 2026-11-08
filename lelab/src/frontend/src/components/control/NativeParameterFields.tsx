import { ParameterDescription, ParameterDraft } from "@/lib/nativeParameters";

export default function NativeParameterFields({ description, value, onChange, disabled = false }: {
  description: ParameterDescription; value: ParameterDraft;
  onChange: (next: ParameterDraft) => void; disabled?: boolean;
}) {
  const limits = description.limits;
  return <div className="space-y-2">
    <div className="grid grid-cols-3 gap-3">
      <label className="text-sm">预测 chunk M
        <input aria-label="预测 chunk M" type="number" min={limits.min_chunk_size} max={limits.max_chunk_size ?? undefined} step={1}
          value={value.chunk_size} disabled={disabled || !limits.chunk_size_editable || limits.max_chunk_size == null}
          onChange={e => onChange({ ...value, chunk_size: e.target.value,
            ...(description.guided ? { n_action_steps: e.target.value } : {}) })}
          className="mt-1 w-full rounded border border-slate-600 bg-gray-950 p-2 disabled:opacity-50" />
      </label>
      <label className="text-sm">每轮执行 N 步
        <input aria-label="每轮执行 N 步" type="number" min={1} max={Number(value.chunk_size) || undefined} step={1}
          value={value.n_action_steps} disabled={disabled || !limits.n_action_steps_editable || description.effective.n_action_steps == null}
          onChange={e => onChange({ ...value, n_action_steps: e.target.value })}
          className="mt-1 w-full rounded border border-slate-600 bg-gray-950 p-2 disabled:opacity-50" />
      </label>
      <label className="text-sm">动作 FPS
        <input aria-label="动作 FPS" type="number" min={1} max={60} step="any" value={value.fps} disabled={disabled}
          onChange={e => onChange({ ...value, fps: e.target.value })}
          className="mt-1 w-full rounded border border-slate-600 bg-gray-950 p-2 disabled:opacity-50" />
      </label>
    </div>
    {!limits.chunk_size_editable && description.effective.chunk_size != null &&
      <p className="text-xs text-slate-400">该模型的预测长度由训练结构固定，M 为只读；可在原生支持的范围修改 N 和 FPS。</p>}
    <p className="text-xs text-slate-400">{description.note} 动作 FPS 独立于相机采集 FPS。降低 FPS 会延长同一动作序列的执行时间。</p>
    <p className="text-xs text-slate-500">默认读取模型 config.json；修改仅用于本次会话，不覆盖权重目录中的配置。</p>
  </div>;
}
