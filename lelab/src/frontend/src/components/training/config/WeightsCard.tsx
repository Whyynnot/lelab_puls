import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { NumberInput } from '@/components/ui/number-input';
import { ConfigComponentProps } from '../types';

export default function WeightsCard({ config, updateConfig }: ConfigComponentProps) {
  const policy = (config.extra_config?.policy ?? {}) as Record<string, unknown>;
  const freezeSupported = ['pi0', 'pi05', 'smolvla'].includes(config.policy_type);
  const togglePolicy = (key: string, value: boolean) => updateConfig('extra_config', {
    ...config.extra_config, policy: { ...policy, [key]: value },
  });
  return (
    <Card className="bg-slate-800/50 border-slate-700 text-white">
      <CardHeader><CardTitle>模型初始化与训练参数范围</CardTitle></CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center gap-3">
          <Switch id="load-pretrained" checked={config.policy_load_pretrained ?? false}
            onCheckedChange={(v) => updateConfig('policy_load_pretrained', v)} />
          <Label htmlFor="load-pretrained">加载预训练权重</Label>
        </div>
        {config.policy_load_pretrained && <div className="space-y-1">
          <Label htmlFor="pretrained-path">Hub 模型 ID 或本地 pretrained_model 目录</Label>
          <Input id="pretrained-path" value={config.policy_pretrained_path ?? ''}
            onChange={(e) => updateConfig('policy_pretrained_path', e.target.value)}
            placeholder="lerobot/pi05_base 或 /mnt/e/lelab_cache/models/..."
            className="bg-slate-950 border-slate-600" />
        </div>}
        <p className="text-xs text-slate-400">关闭时不加载策略检查点；ACT 也不加载预训练视觉骨干。部分 VLA 架构仍可能需要基础语言模型或 tokenizer。</p>
        <div className="grid grid-cols-2 gap-4">
          <div><Label htmlFor="chunk-size">Chunk size</Label>
            <NumberInput id="chunk-size" value={config.policy_chunk_size} min={1}
              onChange={(v) => updateConfig('policy_chunk_size', v)} className="bg-slate-950 border-slate-600" /></div>
          <div><Label htmlFor="action-steps">Action steps</Label>
            <NumberInput id="action-steps" value={config.policy_n_action_steps} min={1}
              onChange={(v) => updateConfig('policy_n_action_steps', v)} className="bg-slate-950 border-slate-600" /></div>
        </div>
        {freezeSupported && <>
          <div className="flex items-center gap-3">
            <Switch id="freeze-vision" checked={Boolean(policy.freeze_vision_encoder ?? config.policy_type === 'smolvla')}
              onCheckedChange={(v) => togglePolicy('freeze_vision_encoder', v)} />
            <Label htmlFor="freeze-vision">冻结视觉编码器</Label>
          </div>
          <div className="flex items-center gap-3">
            <Switch id="expert-only" checked={Boolean(policy.train_expert_only ?? config.policy_type === 'smolvla')}
              onCheckedChange={(v) => togglePolicy('train_expert_only', v)} />
            <Label htmlFor="expert-only">仅训练动作专家与投影层</Label>
          </div>
        </>}
        <p className="text-xs text-slate-400">更多参数可在完整配置的 extra_config 中编辑，例如 policy.dtype、policy.train_state_proj、peft.target_modules、scheduler。</p>
        {config.resume && <div className="space-y-1">
          <Label htmlFor="resume-config">恢复训练：检查点 train_config.json 路径</Label>
          <Input id="resume-config" value={config.config_path ?? ''}
            onChange={(e) => updateConfig('config_path', e.target.value || null)}
            className="bg-slate-950 border-slate-600" />
        </div>}
      </CardContent>
    </Card>
  );
}
