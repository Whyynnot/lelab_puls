import NativeParameterFields from "@/components/control/NativeParameterFields";
import { InferenceMode, ParameterDraft, parameterDraft, parameterValues } from "@/lib/nativeParameters";
import React, { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { NumberInput } from "@/components/ui/number-input";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { AlertTriangle, CheckCircle, Loader2, Play } from "lucide-react";
import { RobotRecord } from "@/hooks/useRobots";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import { useNavigate } from "react-router-dom";
import {
  JobCheckpoint,
  PolicyConfigSummary,
  getCheckpointPolicyConfig,
  listJobCheckpoints,
} from "@/lib/checkpointsApi";
import { startInference } from "@/lib/inferenceApi";
import CheckpointDropdown from "@/components/jobs/CheckpointDropdown";
import { useInferenceCameraBindings } from "@/hooks/useInferenceCameraBindings";
import { cameraConfigurationError } from "@/lib/cameraConfig";
import CameraBindings from "./CameraBindings";
import { useInferenceAvailability } from "@/hooks/useInferenceAvailability";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  robot: RobotRecord | null;
  jobId: string;
  initialStep: number | null;
}

const DEFAULT_FPS = 30;

const InferenceModal: React.FC<Props> = ({
  open,
  onOpenChange,
  robot,
  jobId,
  initialStep,
}) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const navigate = useNavigate();

  const [checkpoints, setCheckpoints] = useState<JobCheckpoint[]>([]);
  const [selectedStep, setSelectedStep] = useState<number | null>(initialStep);
  const [parameters, setParameters] = useState<ParameterDraft | null>(null);
  const [inferenceMode, setInferenceMode] = useState<InferenceMode>("auto");
  const [task, setTask] = useState("");
  const [durationS, setDurationS] = useState(60);
  const [submitting, setSubmitting] = useState(false);
  const session = useInferenceAvailability(open);
  const preparing = session.status?.session_phase === 'preparing';
  const stopRequested = useRef(false);
  const stopPrevious = async () => {
    stopRequested.current = true;
    await session.stopPrevious();
  };

  // This modal may remain mounted after a successful launch/navigation.
  useEffect(() => { if (open) setSubmitting(false); }, [open]);

  const [policyConfig, setPolicyConfig] = useState<PolicyConfigSummary | null>(null);
  const [policyConfigLoading, setPolicyConfigLoading] = useState(false);
  const [policyConfigError, setPolicyConfigError] = useState<string | null>(null);

  const optionalCameras = policyConfig?.camera_selection?.optional ?? false;
  const cameraBindings = useInferenceCameraBindings(policyConfig?.image_features ?? null,
    robot, open, optionalCameras, `${jobId}:${selectedStep}`);

  // Load checkpoints when modal opens.
  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    listJobCheckpoints(baseUrl, fetchWithHeaders, jobId)
      .then((cks) => {
        if (cancelled) return;
        setCheckpoints(cks);
        if (cks.length > 0) {
          const latest = cks[cks.length - 1].step;
          setSelectedStep((prev) => (prev != null ? prev : latest));
        }
      })
      .catch(() => {
        if (cancelled) return;
        setCheckpoints([]);
      });
    return () => {
      cancelled = true;
    };
  }, [open, baseUrl, fetchWithHeaders, jobId]);


  // Load policy config when step changes.
  useEffect(() => {
    if (!open || selectedStep == null) {
      setPolicyConfig(null);
      setPolicyConfigError(null);
      return;
    }
    let cancelled = false;
    setPolicyConfigLoading(true);
    setPolicyConfigError(null);
    getCheckpointPolicyConfig(baseUrl, fetchWithHeaders, jobId, selectedStep)
      .then((cfg) => {
        if (cancelled) return;
        setPolicyConfig(cfg);
        setInferenceMode(cfg.inference_modes?.default ?? "auto");
        setParameters(cfg.inference_parameters ? parameterDraft(cfg.inference_parameters) : null);
      })
      .catch((e) => {
        if (cancelled) return;
        setPolicyConfig(null);
        setPolicyConfigError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (!cancelled) setPolicyConfigLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open, baseUrl, fetchWithHeaders, jobId, selectedStep]);

  const selectedRef =
    selectedStep != null
      ? checkpoints.find((c) => c.step === selectedStep)?.ref ?? null
      : null;

  const cameraError = robot ? cameraConfigurationError(robot.cameras ?? []) : null;
  const parameterDescription = policyConfig?.inference_modes?.options.find(option => option.value === inferenceMode)?.parameters
    ?? policyConfig?.inference_parameters;
  const changeInferenceMode = (mode: InferenceMode) => {
    const description = policyConfig?.inference_modes?.options.find(option => option.value === mode)?.parameters;
    if (!description) return;
    setInferenceMode(mode);
    setParameters(previous => previous ? { ...previous,
      n_action_steps: description.guided ? previous.chunk_size : String(description.effective.n_action_steps == null ? ""
        : Math.min(description.effective.n_action_steps, Number(previous.chunk_size) || description.effective.n_action_steps)),
    } : parameterDraft(description));
  };

  let parameterError = "";
  if (parameters && parameterDescription) {
    try { parameterValues(parameters, parameterDescription); }
    catch (e) { parameterError = e instanceof Error ? e.message : String(e); }
  }

  const canStart =
    !!robot &&
    robot.is_clean &&
    !cameraError &&
    selectedRef != null &&
    !!policyConfig &&
    cameraBindings.allBound &&
    !session.blocked &&
    !submitting &&
    !policyConfigLoading &&
    !parameterError;

  const handleStart = async () => {
    if (!canStart || !robot || selectedRef == null || !policyConfig || cameraError) return;
    // Setting submitting=true makes every CameraPreview drop its
    // browser stream — required so the rollout subprocess can open the
    // same camera index via OpenCV without colliding on the device.
    setSubmitting(true);
    stopRequested.current = false;
    await new Promise((r) => setTimeout(r, 300));
    try {
      const result = await startInference(baseUrl, fetchWithHeaders, {
        follower_port: robot.follower_port,
        follower_config: robot.follower_config,
        policy_ref: selectedRef,
        task,
        cameras: cameraBindings.toCameraDict(DEFAULT_FPS),
        duration_s: durationS,
        inference_mode: inferenceMode,
        ...(parameters && parameterDescription ? {
          inference_parameters: parameterValues(parameters, parameterDescription),
        } : {}),
      });
      if (result?.cancelled || stopRequested.current) {
        setSubmitting(false);
        return;
      }
      onOpenChange(false);
      navigate("/inference");
    } catch (e) {
      if (stopRequested.current) {
        setSubmitting(false);
        return;
      }
      toast({
        title: "Couldn't start inference",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
      // Failure: bring the previews back so the user can adjust.
      setSubmitting(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={(next) => { if (!submitting) onOpenChange(next); }}>
      <DialogContent className="bg-gray-900 border-gray-800 text-white sm:max-w-[600px] p-8 max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <div className="flex justify-center items-center mb-4">
            <div className="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center">
              <Play className="w-4 h-4 text-white" />
            </div>
          </div>
          <DialogTitle className="text-white text-center text-2xl font-bold">
            Configure Inference
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-6 py-4">
          <DialogDescription className="text-gray-400 text-base leading-relaxed text-center">
            Pick a checkpoint and confirm hardware. The selected policy will
            drive the follower autonomously for the configured duration.
          </DialogDescription>

          {session.status?.inference_active && (
            <Alert className="bg-amber-900/40 border-amber-700 text-amber-100">
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription className="space-y-3">
                {preparing
                  ? <p>模型正在下载或准备中，尚未启动机械臂推理。可以取消准备后重新选择模型；取消后的任务不会再启动机械臂。</p>
                  : <p>上一轮推理仍在占用机械臂和相机（{session.status.session_phase ?? "active"}）。复位后的暂停、返回首页都不会结束会话；结束后才能预览相机并启动新任务。</p>}
                <p className="text-xs break-all">{session.status.policy_ref}</p>
                <div className="flex flex-wrap gap-2">
                  <Button type="button" disabled={session.stopping} onClick={stopPrevious} className="bg-red-600 hover:bg-red-700 text-white">
                    {session.stopping ? "正在停止并释放设备…" : preparing ? "取消模型准备并释放任务" : "结束上一轮推理并释放设备"}
                  </Button>
                  {!preparing && <Button type="button" variant="outline" disabled={session.stopping} onClick={() => { onOpenChange(false); navigate("/inference"); }}>返回当前推理</Button>}
                </div>
                <p className="text-xs">{preparing ? "取消会停止模型准备，已下载的缓存保留；不会因此移动机械臂。" : "停止会执行原生退出流程，机械臂可能返回推理初始位置；请保持运动空间畅通。"}</p>
              </AlertDescription>
            </Alert>
          )}
          {session.checking && <p className="text-sm text-slate-400">正在检查后台推理任务…</p>}
          {session.error && <p role="alert" className="text-sm text-red-300">{session.error} 正在重新检查后台状态。</p>}

          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-white border-b border-gray-700 pb-2">
              Robot Configuration
            </h3>
            {!robot ? (
              <Alert className="bg-amber-900/40 border-amber-700 text-amber-100">
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>
                  Select and configure a robot on the Landing page first.
                </AlertDescription>
              </Alert>
            ) : !robot.is_clean ? (
              <Alert className="bg-amber-900/40 border-amber-700 text-amber-100">
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>
                  <strong>{robot.name}</strong> is missing a calibration.
                  Configure it before running inference.
                </AlertDescription>
              </Alert>
            ) : (
              <div className="flex items-center gap-2 text-sm">
                <CheckCircle className="w-4 h-4 text-green-400" />
                <span className="text-slate-200">
                  Running on <strong>{robot.name}</strong>
                </span>
              </div>
            )}
          </div>

          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-white border-b border-gray-700 pb-2">
              Checkpoint
            </h3>
            {checkpoints.length === 0 ? (
              <Alert className="bg-amber-900/40 border-amber-700 text-amber-100">
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>
                  No checkpoints available for this job yet.
                </AlertDescription>
              </Alert>
            ) : (
              <CheckpointDropdown
                checkpoints={checkpoints}
                selectedStep={selectedStep}
                onChange={setSelectedStep}
              />
            )}
          </div>

          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-white border-b border-gray-700 pb-2">
              Run parameters
            </h3>
            {policyConfig?.inference_modes && <div className="space-y-2">
              <label className="text-sm" htmlFor="inference-mode">推理模式</label>
              <select id="inference-mode" aria-label="推理模式" value={inferenceMode} disabled={submitting}
                onChange={event => changeInferenceMode(event.target.value as InferenceMode)}
                className="w-full rounded border border-slate-600 bg-gray-950 p-2">
                {policyConfig.inference_modes.options.map(option => <option key={option.value} value={option.value}>{option.label}</option>)}
                {!policyConfig.inference_modes.options.some(option => option.value === "rtc") &&
                  <option value="rtc" disabled>RTC（该配置不支持）</option>}
              </select>
              <p className="text-xs text-slate-400">{inferenceMode === "rtc"
                ? "边执行旧动作边预测下一块，并用剩余动作引导衔接；首次预测仍需等待。RTC 保留完整预测块，N 随 M。切换模式需停止后重新启动。"
                : policyConfig.inference_modes.rtc_unavailable_reason || "普通模式执行前 N 步后再次预测；RTC 可将执行与预测重叠。"}</p>
            </div>}
            {parameters && parameterDescription && <>
              <NativeParameterFields description={parameterDescription} value={parameters} onChange={setParameters} disabled={submitting} />
              {parameterError && <p role="alert" className="text-sm text-red-300">{parameterError}</p>}
            </>}
            {policyConfig?.requires_task ? (
              <div className="space-y-2">
                <Label htmlFor="task" className="text-sm font-medium text-gray-300">
                  Task description
                </Label>
                <Input
                  id="task"
                  value={task}
                  onChange={(e) => setTask(e.target.value)}
                  placeholder="e.g., pick up the red block"
                  className="bg-gray-800 border-gray-700 text-white"
                />
                <p className="text-xs text-gray-500">
                  This policy is language-conditioned ({policyConfig.policy_type}).
                </p>
              </div>
            ) : null}
            <div className="space-y-2">
              <Label htmlFor="durationS" className="text-sm font-medium text-gray-300">
                Max duration (seconds)
              </Label>
              <NumberInput
                id="durationS"
                min={1}
                value={durationS}
                onChange={(v) => {
                  if (v !== undefined) setDurationS(v);
                }}
                className="bg-gray-800 border-gray-700 text-white"
              />
            </div>
          </div>

          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-white border-b border-gray-700 pb-2">
              Cameras
            </h3>
            {cameraError ? (
              <Alert className="bg-red-900/40 border-red-700 text-red-100">
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>{cameraError} Correct the camera settings in Calibration.</AlertDescription>
              </Alert>
            ) : policyConfigLoading ? (
              <div className="flex items-center gap-2 text-sm text-slate-400">
                <Loader2 className="w-4 h-4 animate-spin" />
                Reading policy config…
              </div>
            ) : policyConfigError ? (
              <Alert className="bg-red-900/40 border-red-700 text-red-100">
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>
                  Couldn't load policy config: {policyConfigError}
                </AlertDescription>
              </Alert>
            ) : !policyConfig ? null : Object.keys(policyConfig.image_features).length === 0 ? (
              <p className="text-xs text-gray-500">
                This policy doesn't use cameras.
              </p>
            ) : (
              <div className="space-y-3">
                <p className="text-xs text-gray-500">
                  {optionalCameras ? <>勾选参与推理的真实相机，至少启用一路。未启用的视角不采集，模型空视角由策略自动处理。
                    已启用 {cameraBindings.activeCount} / {Object.keys(policyConfig.image_features).length} 路。</> : <>
                  Bind a physical camera to each name the policy was trained
                  with. Resolution comes from the checkpoint. Use the same
                  rotation as the training data.
                  </>}
                </p>
                {cameraBindings.error && <p role="alert" className="text-sm text-red-300">{cameraBindings.error}</p>}
                <CameraBindings
                  optional={optionalCameras}
                  isEnabled={cameraBindings.isEnabled}
                  onEnabledChange={cameraBindings.setEnabled}
                  captureFor={cameraBindings.captureFor}
                  imageFeatures={policyConfig.image_features}
                  bindings={cameraBindings.bindings}
                  onBind={cameraBindings.bind}
                  rotationFor={cameraBindings.rotationFor}
                  onRotate={cameraBindings.setRotation}
                  availableCameras={cameraBindings.availableCameras}
                  paused={!open || submitting || session.blocked}
                />
              </div>
            )}
          </div>

          <div className="flex flex-col sm:flex-row gap-4 justify-center pt-4">
            <Button
              onClick={handleStart}
              disabled={!canStart}
              className="w-full sm:w-auto bg-green-500 hover:bg-green-600 text-white px-10 py-6 text-lg disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <Play className="w-5 h-5 mr-2" />
              {submitting ? "Starting…" : "Start Inference"}
            </Button>
            <Button
              onClick={() => onOpenChange(false)}
              disabled={submitting}
              variant="outline"
              className="w-full sm:w-auto border-gray-500 hover:border-gray-200 px-10 py-6 text-lg text-zinc-500 bg-zinc-900 hover:bg-zinc-800"
            >
              Cancel
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default InferenceModal;
