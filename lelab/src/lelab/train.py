# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training-specific helpers: the request schema and the LeRobot CLI builder.

The actual job lifecycle (subprocess management, registry, log streaming)
lives in app/jobs.py.
"""

import json
import re
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, field_validator, model_validator

if TYPE_CHECKING:
    from lelab.jobs import JobTarget

_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")


class TrainingRequest(BaseModel):
    # Dataset configuration
    dataset_repo_id: str
    dataset_revision: str | None = None
    dataset_root: str | None = None
    dataset_episodes: list[int] | None = None
    dataset_image_transforms_enable: bool = False

    # Policy configuration
    policy_type: str = "act"
    policy_load_pretrained: bool = False
    policy_pretrained_path: str | None = None

    # Core training parameters
    steps: int = Field(default=10000, gt=0)
    batch_size: int = Field(default=8, gt=0)
    seed: int | None = 1000
    num_workers: int = Field(default=4, ge=0)

    # Logging and checkpointing
    log_freq: int = 250
    save_freq: int = 1000
    env_eval_freq: int = 0
    save_checkpoint: bool = True

    # Output configuration
    output_dir: str = "outputs/train"
    resume: bool = False
    job_name: str | None = None

    # Weights & Biases
    wandb_enable: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_notes: str | None = None
    wandb_run_id: str | None = None
    wandb_mode: str | None = "online"
    wandb_disable_artifact: bool = False

    # Environment / evaluation
    env_type: str | None = None
    env_task: str | None = None
    eval_n_episodes: int = 10
    eval_batch_size: int = 50
    eval_use_async_envs: bool = False

    # Policy-specific
    policy_device: str | None = "cuda"
    policy_use_amp: bool = False
    # Hub upload (set by HfCloudJobRunner; not exposed in the form)
    policy_push_to_hub: bool = False
    policy_repo_id: str | None = None

    # GR00T-specific policy options (only emitted when policy_type == "groot";
    # these flags don't exist on other policy configs, so draccus would reject
    # them). All optional — unset fields fall back to the groot config defaults.
    policy_base_model_path: str | None = None
    policy_embodiment_tag: str | None = None
    policy_chunk_size: int | None = None
    policy_n_action_steps: int | None = None
    policy_use_relative_actions: bool | None = None
    policy_relative_exclude_joints: list[str] | None = None
    policy_use_bf16: bool | None = None

    # Optimizer
    optimizer_type: str | None = None
    optimizer_lr: float | None = None
    optimizer_weight_decay: float | None = None
    optimizer_grad_clip_norm: float | None = None

    # Advanced
    use_policy_training_preset: bool = True
    config_path: str | None = None
    # Native LeRobot sections not represented by the form (PEFT, scheduler,
    # freeze flags, streaming, precision, parameter groups, etc.).
    extra_config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("extra_config")
    @classmethod
    def validate_extra_config(cls, value):
        def check(data, depth=0):
            if isinstance(data, dict):
                for key, item in data.items():
                    if not isinstance(key, str) or (depth == 0 and not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", key)):
                        raise ValueError(f"Invalid config key: {key}")
                    check(item, depth + 1)
            elif isinstance(data, list):
                for item in data:
                    check(item, depth + 1)
        check(value)
        if set(value) & {"output_dir", "job", "config_path"}:
            raise ValueError("output_dir/job/config_path must use the job or form controls")
        json.dumps(value, allow_nan=False)
        return value

    @model_validator(mode="after")
    def validate_pretrained(self):
        if self.policy_load_pretrained and not (self.policy_pretrained_path or "").strip():
            raise ValueError("Provide a pretrained Hub model ID or local checkpoint directory")
        return self


# Native config paths mapped to existing form fields. Unknown settings remain
# in extra_config, so importing a training config never drops tuning options.
_CONFIG_FIELDS = {
    "dataset.repo_id": "dataset_repo_id", "dataset.revision": "dataset_revision",
    "dataset.root": "dataset_root", "dataset.episodes": "dataset_episodes",
    "dataset.image_transforms.enable": "dataset_image_transforms_enable",
    "policy.type": "policy_type", "policy.device": "policy_device",
    "policy.use_amp": "policy_use_amp", "policy.push_to_hub": "policy_push_to_hub",
    "policy.repo_id": "policy_repo_id", "policy.base_model_path": "policy_base_model_path",
    "policy.embodiment_tag": "policy_embodiment_tag", "policy.chunk_size": "policy_chunk_size",
    "policy.n_action_steps": "policy_n_action_steps",
    "policy.use_relative_actions": "policy_use_relative_actions",
    "policy.relative_exclude_joints": "policy_relative_exclude_joints",
    "optimizer.type": "optimizer_type", "optimizer.lr": "optimizer_lr",
    "optimizer.weight_decay": "optimizer_weight_decay",
    "optimizer.grad_clip_norm": "optimizer_grad_clip_norm",
    "eval.n_episodes": "eval_n_episodes", "eval.batch_size": "eval_batch_size",
    "eval.use_async_envs": "eval_use_async_envs", "env.type": "env_type", "env.task": "env_task",
    "trackio.enable": "wandb_enable", "trackio.project": "wandb_project",
    "trackio.disable_artifact": "wandb_disable_artifact",
}


def parse_training_config(text: str) -> TrainingRequest:
    """Load native LeRobot JSON/YAML or an exported LeLab request."""
    import yaml

    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError("Training config must be a JSON/YAML object")
    if "config" in raw and isinstance(raw["config"], dict):
        raw = raw["config"]  # exported job.json / submission
    raw = deepcopy(raw)
    if "dataset_repo_id" in raw:
        unknown = set(raw) - set(TrainingRequest.model_fields) - {"target"}
        if unknown:
            raise ValueError(f"Unknown LeLab fields: {sorted(unknown)}; use extra_config for native options")
        raw.pop("target", None)
        return TrainingRequest.model_validate(raw)

    if 'eval_freq' in raw and 'env_eval_freq' not in raw:
        raw['env_eval_freq'] = raw.pop('eval_freq')
    # v0.6 represented disabled EMA as an object with enable=false; v0.7
    # represents it as null and enables a decay schedule via ema.type.
    ema = raw.get('ema')
    if isinstance(ema, dict) and 'enable' in ema:
        enabled = ema.pop('enable')
        ema.pop('use_for_eval', None)  # v0.7 always evaluates EMA when configured
        if not enabled:
            raw['ema'] = None
        else:
            ema.setdefault('type', 'warmup')

    def pop(path):
        parts = path.split('.')
        parent = raw
        for part in parts[:-1]:
            if not isinstance(parent, dict) or part not in parent:
                return False, None
            parent = parent[part]
        if not isinstance(parent, dict) or parts[-1] not in parent:
            return False, None
        return True, parent.pop(parts[-1])

    values = {}
    # v0.6 configs contain W&B; v0.7 uses Trackio.
    legacy = raw.pop("wandb", None)
    if isinstance(legacy, dict) and "trackio" not in raw:
        raw["trackio"] = {k: v for k, v in legacy.items()
                          if k in {"enable", "project", "disable_artifact"}}
    for path, field in _CONFIG_FIELDS.items():
        present, value = pop(path)
        if present:
            values[field] = value
    for field in TrainingRequest.model_fields:
        if field in raw and field not in {"extra_config", "output_dir"}:
            values[field] = raw.pop(field)
    for path in ("policy.pretrained_path", "policy.path"):
        present, value = pop(path)
        if present and value:
            values.update(policy_pretrained_path=str(value), policy_load_pretrained=True)
    # A new submission owns its output directory. Saved checkpoint paths are
    # metadata, not permission to overwrite the original run.
    raw.pop("output_dir", None)
    raw.pop("checkpoint_path", None)
    raw.pop("job", None)
    def prune(data):
        return {k: prune(v) if isinstance(v, dict) else v for k, v in data.items()
                if not isinstance(v, dict) or v}
    values["extra_config"] = prune(raw)
    return TrainingRequest.model_validate(values)


def _override_args(data: dict, prefix="") -> dict[str, str]:
    """Encode lists/maps/null as one JSON argument, preserving draccus types."""
    args = {}
    sections = {"dataset", "policy", "optimizer", "scheduler", "eval", "env",
                "accelerator", "parallelism", "peft", "trackio", "ema"}
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and (path in sections or path == "dataset.image_transforms"):
            args.update(_override_args(value, path))
        else:
            args[f"--{path}"] = (value if isinstance(value, str) else json.dumps(value, allow_nan=False))
    return args


def build_training_command(
    request: TrainingRequest,
    output_dir: str,
    python_executable: str = "python",
    job_target: "JobTarget | None" = None,
) -> list[str]:
    """Build the argv list to invoke `<python_executable> -m lerobot.scripts.lerobot_train`.

    `output_dir` is supplied separately from the request so the caller (the
    JobRegistry) can pin it to the per-job directory rather than relying on
    request.output_dir, which the frontend doesn't even send in the new world.

    `python_executable` defaults to "python" for the cloud runner (whose
    container has lerobot on PATH); the local runner must pass sys.executable
    so the subprocess uses the same interpreter as lelab itself — otherwise
    PATH lookup picks up a different env (uv tool venv, miniforge3 base, etc.)
    that lacks lerobot.
    """
    cmd: list[str] = [python_executable, "-m", "lerobot.scripts.lerobot_train"]
    if request.resume and not request.config_path:
        raise ValueError("Resume requires the checkpoint train_config.json path")

    # Dataset
    cmd.extend(["--dataset.repo_id", request.dataset_repo_id])
    if request.dataset_revision:
        cmd.extend(["--dataset.revision", request.dataset_revision])
    if request.dataset_root:
        cmd.extend(["--dataset.root", request.dataset_root])
    if request.dataset_episodes is not None:
        cmd.extend(["--dataset.episodes", json.dumps(request.dataset_episodes)])
    if request.dataset_image_transforms_enable:
        cmd.extend(["--dataset.image_transforms.enable", "true"])

    # Policy
    if request.policy_load_pretrained:
        cmd.extend(["--policy.path", request.policy_pretrained_path.strip()])
    else:
        cmd.extend(["--policy.type", request.policy_type])
        if request.policy_type == "act":
            cmd.extend(["--policy.pretrained_backbone_weights", "null"])
        elif request.policy_type == "smolvla":
            cmd.extend(["--policy.load_vlm_weights", "false"])

    # Core training params
    cmd.extend(["--steps", str(request.steps)])
    cmd.extend(["--batch_size", str(request.batch_size)])
    cmd.extend(["--num_workers", str(request.num_workers)])
    cmd.extend(["--seed", "null" if request.seed is None else str(request.seed)])

    # Policy device / AMP / hub
    if request.policy_device:
        cmd.extend(["--policy.device", request.policy_device])
    cmd.extend(["--policy.use_amp", "true" if request.policy_use_amp else "false"])
    # On HF Cloud, lerobot's submit_to_hf owns the model repo and sets push_to_hub on
    # the pod itself; _pod_forwarded_args drops any --policy.push_to_hub/--policy.repo_id
    # we'd pass, so we must not emit them. Local runs keep the existing behavior:
    # LeRobot defaults push_to_hub=True and demands --policy.repo_id when so.
    is_cloud = job_target is not None and job_target.runner == "hf_cloud"
    if not is_cloud:
        cmd.extend(["--policy.push_to_hub", "true" if request.policy_push_to_hub else "false"])
        if request.policy_push_to_hub and request.policy_repo_id:
            cmd.extend(["--policy.repo_id", request.policy_repo_id])

    # GR00T-specific policy flags. These options only exist on the groot config,
    # so emit them only for --policy.type=groot; sending them to another policy
    # would make draccus abort the run. Each is skipped when unset so the groot
    # config's own defaults apply.
    if request.policy_type == "groot":
        if request.policy_base_model_path:
            cmd.extend(["--policy.base_model_path", request.policy_base_model_path])
        if request.policy_embodiment_tag:
            cmd.extend(["--policy.embodiment_tag", request.policy_embodiment_tag])
        if request.policy_use_relative_actions is not None:
            cmd.extend(
                ["--policy.use_relative_actions", "true" if request.policy_use_relative_actions else "false"]
            )
        if request.policy_relative_exclude_joints is not None:
            # draccus parses list values from a single JSON token, e.g. '["gripper"]'.
            cmd.extend(
                ["--policy.relative_exclude_joints", json.dumps(request.policy_relative_exclude_joints)]
            )
        if request.policy_use_bf16 is not None:
            cmd.extend(["--policy.dtype", "bfloat16" if request.policy_use_bf16 else "float32"])

    if request.policy_chunk_size is not None:
        cmd.extend(["--policy.chunk_size", str(request.policy_chunk_size)])
    if request.policy_n_action_steps is not None:
        cmd.extend(["--policy.n_action_steps", str(request.policy_n_action_steps)])

    # Logging / checkpointing
    cmd.extend(["--log_freq", str(request.log_freq)])
    cmd.extend(["--save_freq", str(request.save_freq)])
    cmd.extend(["--env_eval_freq", str(request.env_eval_freq)])
    cmd.extend(["--save_checkpoint", "true" if request.save_checkpoint else "false"])

    # Output. On HF Cloud the pod, not this host, runs the trainer: an absolute host
    # output_dir (e.g. ~/.cache/.../outputs/train) is baked into the staged config and
    # the pod crashes trying to mkdir it under /Users. Checkpoints land on the Hub repo
    # anyway, so we omit it for cloud and let lerobot pick its in-pod default.
    if not is_cloud:
        cmd.extend(["--output_dir", output_dir])
    cmd.extend(["--resume", "true" if request.resume else "false"])
    if request.job_name:
        cmd.extend(["--job_name", request.job_name])

    # v0.7 tracks experiments with Trackio. Legacy request names are retained
    # for saved jobs; W&B-only settings are not sent to the new CLI.
    cmd.extend(["--trackio.enable", "true" if request.wandb_enable else "false"])
    if request.wandb_enable:
        if request.wandb_project:
            cmd.extend(["--trackio.project", request.wandb_project])
        if request.wandb_run_id:
            cmd.extend(["--trackio.run_name", request.wandb_run_id])
        cmd.extend(["--trackio.disable_artifact", "true" if request.wandb_disable_artifact else "false"])

    # Env
    if request.env_type:
        cmd.extend(["--env.type", request.env_type])
    if request.env_task:
        cmd.extend(["--env.task", request.env_task])

    # Eval
    cmd.extend(["--eval.n_episodes", str(request.eval_n_episodes)])
    cmd.extend(["--eval.batch_size", str(request.eval_batch_size)])
    cmd.extend(["--eval.use_async_envs", "true" if request.eval_use_async_envs else "false"])

    # Explicit optimizer edits must not be discarded by LeRobot's presets.
    custom_optimizer = any(value is not None for value in (
        request.optimizer_type, request.optimizer_lr, request.optimizer_weight_decay,
        request.optimizer_grad_clip_norm,
    )) or bool(request.extra_config.get("optimizer"))
    use_preset = request.use_policy_training_preset and not custom_optimizer and not request.extra_config.get("scheduler")
    if not use_preset:
        import draccus
        from lerobot.policies.factory import make_policy_config
        policy_config = make_policy_config(request.policy_type, device=request.policy_device)
        preset_optimizer = policy_config.get_optimizer_preset()
        preset_scheduler = policy_config.get_scheduler_preset()
        selected_optimizer = request.optimizer_type or (request.extra_config.get("optimizer") or {}).get("type")
        if preset_optimizer is not None and selected_optimizer in (None, preset_optimizer.type):
            optimizer_defaults = {"type": preset_optimizer.type, **draccus.encode(preset_optimizer)}
            for flag, value in _override_args({"optimizer": optimizer_defaults}).items():
                cmd.extend([flag, value])
        if preset_scheduler is not None and not request.extra_config.get("scheduler"):
            scheduler_defaults = {"type": preset_scheduler.type, **draccus.encode(preset_scheduler)}
            for flag, value in _override_args({"scheduler": scheduler_defaults}).items():
                cmd.extend([flag, value])
        else:
            cmd.extend(["--scheduler.type", "constant_with_warmup", "--scheduler.num_warmup_steps", "0"])
    if request.optimizer_type:
        cmd.extend(["--optimizer.type", request.optimizer_type])
    if request.optimizer_lr is not None:
        cmd.extend(["--optimizer.lr", str(request.optimizer_lr)])
    if request.optimizer_weight_decay is not None:
        cmd.extend(["--optimizer.weight_decay", str(request.optimizer_weight_decay)])
    if request.optimizer_grad_clip_norm is not None:
        cmd.extend(["--optimizer.grad_clip_norm", str(request.optimizer_grad_clip_norm)])

    # Advanced
    cmd.extend(["--use_policy_training_preset", "true" if use_preset else "false"])
    if request.config_path:
        cmd.extend(["--config_path", request.config_path])

    # HF Jobs: --job.target=<flavor> dispatches the run remotely (lerobot commit #3856).
    # Image/timeout use lerobot's JobConfig defaults. lelab tags its jobs; lerobot always
    # adds a "lerobot" tag too. A pod's local checkpoints die with it, so push each one to
    # the model repo's checkpoints/<step>/ tree (the native replacement for lelab's old
    # in-pod uploader) — that's what makes the trained checkpoints reachable afterwards.
    if is_cloud and job_target.flavor:
        cmd.extend(["--job.target", job_target.flavor])
        cmd.extend(["--job.tags", '["lelab"]'])
        # save_checkpoint_to_hub needs policy.repo_id, which submit_to_hf only sets on the
        # fresh-run path; on a resume it isn't set before validate(), so the flag would
        # abort the submit. A resume already pushes back to its source repo, so skip it.
        if request.save_checkpoint and not request.resume:
            cmd.extend(["--save_checkpoint_to_hub", "true"])

    overrides = _override_args(request.extra_config)
    # Model source and job ownership always come from the dedicated controls.
    for flag in ("--policy.path", "--policy.pretrained_path", "--policy.type", "--resume"):
        if flag in overrides:
            raise ValueError(f"{flag} must use the corresponding form control")
    if not request.policy_load_pretrained:
        if overrides.get("--policy.load_vlm_weights") == "true":
            raise ValueError("Enable pretrained weights before setting load_vlm_weights=true")
        if request.policy_type == "act" and overrides.get("--policy.pretrained_backbone_weights", "null") != "null":
            raise ValueError("Enable pretrained weights before using pretrained backbone weights")
    # One flag per value: custom settings replace generated defaults explicitly.
    result = cmd[:3]
    flags = dict(zip(cmd[3::2], cmd[4::2], strict=True))
    flags.update(overrides)
    for key, value in flags.items():
        result.extend([key, value])
    return result


def validate_training_config(request: TrainingRequest, job_target=None) -> None:
    """Check native config types/unknown fields before launching a costly job.

    No model weights, network access or robot connections are involved.
    Checkpoint sources are resolved later by LeRobot itself.
    """
    import contextlib
    import io
    import draccus
    import lerobot.policies  # registers supported policy config choices
    from lerobot.configs import parser  # registers torch.dtype serialization
    from lerobot.configs.train import TrainPipelineConfig

    argv = build_training_command(request, '/tmp/lelab-config-validation', job_target=job_target)[3:]
    flags = dict(zip(argv[::2], argv[1::2], strict=True))
    flags.pop('--policy.path', None)
    flags.pop('--config_path', None)
    flags['--policy.type'] = request.policy_type
    flags['--resume'] = 'false'
    args = [token for pair in flags.items() for token in pair]
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stderr(stderr):
            draccus.parse(TrainPipelineConfig, args=args)
    except SystemExit as exc:
        raise ValueError(stderr.getvalue().strip().splitlines()[-1]) from exc
    except Exception as exc:
        raise ValueError(str(exc)) from exc
