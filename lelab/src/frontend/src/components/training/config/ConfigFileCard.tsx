import React, { useEffect, useRef, useState } from 'react';
import { useApi } from '@/contexts/ApiContext';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { TrainingConfig } from '../types';

interface Props {
  config: TrainingConfig;
  onApply: (config: TrainingConfig) => void;
  onPendingChange: (pending: boolean) => void;
}

export default function ConfigFileCard({ config, onApply, onPendingChange }: Props) {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [text, setText] = useState(() => JSON.stringify(config, null, 2));
  const [dirty, setDirty] = useState(false);
  const [error, setError] = useState('');
  const [applying, setApplying] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!dirty) setText(JSON.stringify(config, null, 2));
  }, [config, dirty]);
  useEffect(() => onPendingChange(dirty || applying), [dirty, applying, onPendingChange]);

  const edit = (value: string) => {
    setText(value);
    setDirty(true);
    setError('');
  };
  const apply = async () => {
    setApplying(true);
    setError('');
    try {
      const response = await fetchWithHeaders(`${baseUrl}/training/config/parse`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail));
      onApply(data);
      setDirty(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setApplying(false);
    }
  };
  const download = () => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = 'lelab-training-config.json';
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <Card className="max-w-3xl mx-auto mb-6 bg-slate-800/50 border-slate-700 text-white">
      <CardHeader><CardTitle>训练 Config · 导入与编辑</CardTitle></CardHeader>
      <CardContent className="space-y-3">
        <p className="text-sm text-slate-300">
          支持 LeRobot train_config.json、YAML 和 LeLab 配置。导入后点击“应用配置”，再在下方调整参数。
          extra_config 可配置 scheduler、PEFT/LoRA、冻结层、数据增强、精度等原生选项；其中的值优先于表单默认值。
        </p>
        <div className="flex flex-wrap gap-2">
          <input ref={fileRef} type="file" accept=".json,.yaml,.yml" className="hidden" aria-label="训练配置文件"
            onChange={async (event) => {
              const file = event.target.files?.[0];
              if (file) {
                try { edit(await file.text()); }
                catch (e) { setError(String(e)); }
              }
              event.target.value = '';
            }} />
          <Button variant="outline" onClick={() => fileRef.current?.click()}>加载配置文件</Button>
          <Button variant="outline" onClick={download} disabled={dirty}>导出当前配置</Button>
          <Button variant="outline" onClick={() => { setDirty(false); setError(''); setText(JSON.stringify(config, null, 2)); }}>恢复当前表单</Button>
        </div>
        <label htmlFor="training-config-editor" className="block text-sm">完整训练配置（JSON / YAML）</label>
        <textarea id="training-config-editor" value={text} onChange={(e) => edit(e.target.value)} spellCheck={false}
          className="w-full min-h-64 p-3 rounded-md bg-slate-950 border border-slate-600 font-mono text-xs" />
        {error && <p role="alert" className="text-red-300 text-sm whitespace-pre-wrap">{error}</p>}
        <div className="flex gap-3 items-center">
          <Button onClick={apply} disabled={applying || !dirty}>{applying ? '校验中…' : '应用配置'}</Button>
          <span className="text-xs text-slate-400">{dirty ? '有未应用的编辑；应用成功后可提交训练。' : '配置已同步，提交时会保存到任务记录。'}</span>
        </div>
      </CardContent>
    </Card>
  );
}
