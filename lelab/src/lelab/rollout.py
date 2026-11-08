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

"""Inference mode: drives the SO-101 follower with a trained policy.

Mirrors `app/teleoperating.py` in shape — single global session, mutex
with teleoperation/recording (the follower's serial bus can only be
opened once), `lerobot.scripts.lerobot_rollout` running as a subprocess
for clean cancellation. Hub-checkpoint refs are resolved to a local dir
via huggingface_hub.snapshot_download before we spawn the subprocess.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, field_validator

from .camera_config import lerobot_camera_settings, validate_camera_rotations
from .policy_preparation import PreparationCancelled, resolve_in_worker, terminate_process
from .utils.config import setup_follower_calibration_file

logger = logging.getLogger(__name__)


class InferenceRequest(BaseModel):
    follower_port: str
    follower_config: str
    policy_ref: str  # opaque ref returned by /jobs/{id}/checkpoints
    task: str = ""
    cameras: dict[str, dict[str, Any]] = {}
    duration_s: int = 60
    inference_parameters: dict[str, Any] | None = None
    inference_mode: Literal['auto', 'sync', 'rtc'] = 'auto'

    @field_validator("cameras")
    @classmethod
    def validate_rotations(cls, cameras: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return validate_camera_rotations(cameras)


inference_active: bool = False
_inference_proc: subprocess.Popen | None = None
_inference_started_at: float | None = None
_inference_rollout_started_at: float | None = None
_inference_meta: dict[str, Any] = {}
_preparation_proc: subprocess.Popen | None = None
_inference_start_id: str | None = None
_inference_cancel: threading.Event | None = None
# Guards mutations to the globals above; held only for the short critical
# sections in start/stop/status.
_state_lock = threading.Lock()
_HUB_REF_RE = re.compile(r"^(?P<repo>[^@]+)@checkpoints/(?P<step_dir>\d+)$")
_HUB_ROOT_REF_RE = re.compile(r"^(?P<repo>[^@]+)@root$")
# lerobot prints this once per run, the moment its main control loop is
# about to take over from the setup phase. We watch stdout for it so the
# UI can present a "rollout time" separate from the multi-second policy
# load + bus connect + camera connect setup overhead.
_ROLLOUT_START_MARKER = "Rollout setup complete"


def _pump_stdout(proc: subprocess.Popen, log_handle) -> None:
    """Tee the subprocess's stdout to the log file and watch for the
    rollout-start marker."""
    global _inference_rollout_started_at
    try:
        for raw in iter(proc.stdout.readline, b""):
            try:
                line = raw.decode("utf-8", errors="replace")
            except Exception:
                continue
            try:
                log_handle.write(line)
                log_handle.flush()
            except Exception:
                pass
            if _ROLLOUT_START_MARKER in line:
                with _state_lock:
                    if _inference_proc is not proc or _inference_rollout_started_at is not None:
                        continue
                    _inference_rollout_started_at = time.time()
                logger.info(
                    "Inference rollout main loop started after %.1fs of setup",
                    _inference_rollout_started_at - (_inference_started_at or _inference_rollout_started_at),
                )
    except Exception as exc:
        logger.exception("Inference stdout pump failed: %s", exc)
    finally:
        with contextlib.suppress(Exception):
            log_handle.close()


def _detect_device() -> str:
    """cuda → mps → cpu, picked once at start time."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _resolve_policy_path(policy_ref: str) -> str:
    """Turn a checkpoints API ref into a local path that lerobot accepts.

    Local refs are already absolute paths to a pretrained_model dir.
    Hub refs look like 'user/repo@checkpoints/<step_dir>' where
    <step_dir> is lerobot's zero-padded directory name (e.g. 000050) — we
    forward it verbatim into snapshot_download's allow_patterns and the
    resolved local path.
    A 'user/repo@root' ref means the whole repo IS the pretrained_model
    (no checkpoints sub-tree); the full repo is downloaded via
    snapshot_download and its root is returned directly."""
    if Path(policy_ref).is_dir():
        return policy_ref
    from huggingface_hub import snapshot_download

    m = _HUB_REF_RE.match(policy_ref)
    if m:
        repo_id, step_dir = m.group("repo"), m.group("step_dir")
        from .cache import legacy_policy_snapshot
        cached = legacy_policy_snapshot(repo_id, f'checkpoints/{step_dir}/pretrained_model')
        if cached:
            return cached
        local_root = snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            allow_patterns=[f"checkpoints/{step_dir}/pretrained_model/*"],
        )
        return str(Path(local_root) / "checkpoints" / step_dir / "pretrained_model")
    m = _HUB_ROOT_REF_RE.match(policy_ref)
    if m:
        from .cache import legacy_policy_snapshot
        cached = legacy_policy_snapshot(m.group("repo"))
        if cached:
            return cached
        return snapshot_download(repo_id=m.group("repo"), repo_type="model")
    raise ValueError(f"Unrecognised policy ref: {policy_ref!r}")


def _read_policy_config(policy_path: str) -> dict[str, Any]:
    """Load pretrained_model/config.json if present."""
    config_path = Path(policy_path) / "config.json"
    if not config_path.is_file():
        return {}
    try:
        with open(config_path) as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning("Failed to read policy config at %s: %s", config_path, exc)
        return {}


def _rollout_inference_args(policy_path: str) -> list[str]:
    """Return extra lerobot-rollout flags for policies that reject sync inference.

    GR00T and other relative-action policies decode action chunks against the
    observation state at inference time. The default sync backend calls
    ``select_action`` per tick and can re-decode cached relative actions
    against newer states, so lerobot requires the RTC/chunked rollout path
    instead.

    GR00T also needs tuned RTC queue settings: the lerobot default
    ``queue_threshold=30`` makes the background thread replan while a 16-step
    chunk is still playing, which replaces the queue mid-motion and feels
    like one action then a full recalculation."""
    cfg = _read_policy_config(policy_path)
    policy_type = cfg.get("type")
    needs_rtc = bool(cfg.get("use_relative_actions")) or policy_type == "groot"
    if not needs_rtc:
        return []

    args = ["--inference.type=rtc"]

    n_action_steps = cfg.get("n_action_steps")
    if isinstance(n_action_steps, int) and n_action_steps > 0:
        args.append(f"--inference.rtc.execution_horizon={n_action_steps}")
    elif policy_type == "groot":
        args.append("--inference.rtc.execution_horizon=16")

    # Wait until the current chunk is consumed before replanning. GROOT docs
    # recommend keeping this at 0 (never > 5) for stable real-robot rollout.
    args.append("--inference.queue_threshold=0")
    return args


def _format_cameras_arg(cameras: dict[str, dict[str, Any]]) -> str:
    """Convert native camera settings into
    lerobot's CLI dict syntax. The frontend key `camera_index` is
    remapped to lerobot's `index_or_path`."""
    parts = []
    for name, cfg in cameras.items():
        settings = lerobot_camera_settings(cfg)
        remapped = {
            ("index_or_path" if k == "camera_index" else k): v for k, v in settings.items() if v is not None
        }
        body = ", ".join(f"{k}: {v}" for k, v in remapped.items())
        parts.append(f"{name}: {{{body}}}")
    return "{" + ", ".join(parts) + "}"


# Exception lines at the tail of a Python traceback look like
# "RuntimeError: ..." or "lerobot.errors.DeviceNotConnectedError: ...".
_EXC_LINE_RE = re.compile(r"^[A-Za-z_][\w.]*(?:Error|Exception|Interrupt|Timeout|Failure)\b")


def _extract_error_from_log(log_path: str | None) -> str | None:
    """Pull the meaningful error out of a failed rollout's log so the UI can
    show it directly instead of telling the user to open a file in the cache."""
    if not log_path:
        return None
    try:
        # Only the tail matters; avoid materializing a multi-MB verbose log.
        with open(log_path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - 64 * 1024))
            data = fh.read()
    except OSError:
        return None
    lines = data.decode("utf-8", errors="replace").splitlines()
    # Fault cleanup may emit secondary exceptions. Keep the original failure.
    marked = [line.split('LELAB_INFERENCE_FAULT:', 1)[1].strip()
              for line in lines if 'LELAB_INFERENCE_FAULT:' in line]
    if marked:
        return marked[0][:500]
    # Also recover the primary error from logs produced before the fault guard.
    start = next((i for i, line in enumerate(lines)
                  if 'Rollout strategy failed mid-segment' in line), None)
    if start is not None:
        primary = []
        for line in lines[start + 1:]:
            if line.startswith(('INFO ', 'WARNING ', 'ERROR ')):
                break
            if _EXC_LINE_RE.match(line):
                primary.append(line)
        if primary:
            return primary[-1][:500]
    tail = lines[-50:]
    # Prefer the last exception line + everything after it (the message body).
    exc_idx = next((i for i in range(len(tail) - 1, -1, -1) if _EXC_LINE_RE.match(tail[i])), None)
    if exc_idx is not None:
        snippet = "\n".join(tail[exc_idx:]).strip()
    else:
        non_empty = [ln for ln in tail if ln.strip()]
        snippet = "\n".join(non_empty[-6:]).strip()
    snippet = re.sub(r"\n\s*\n+", "\n", snippet)
    if len(snippet) > 500:
        snippet = snippet[:500].rstrip() + "…"
    return snippet or None


def _friendly_hint(error_text: str | None) -> str | None:
    """A plain-language, actionable headline for the common SO-101 failures."""
    if not error_text:
        return None
    low = error_text.lower()
    if 'failed to sync read' in low or 'failed to disable torque' in low:
        return (
            '从臂电机通信中断。请先支撑机械臂，检查 USB、Hub 和供电；'
            '若腕部相机也掉线，请将其与机械臂分开连接后再检测。'
            '故障退出不会自动回位；保留扭矩设置不能保证断电时保持姿态。'
        )
    if ("checksum" in low or "incorrect status packet" in low) and any(
        marker in low for marker in ("torque", "motor", "status packet", "serial")
    ):
        return (
            "机械臂串口通信包校验失败；模型加载与推理参数不是这条异常的报错位置。"
            "初始化已配置有限扭矩通信重试。若重试后仍失败，请检查机械臂供电、USB 和电机线连接后重新启动。"
        )
    if "overload" in low or "torque_enable" in low:
        return (
            "A motor overloaded — usually the gripper holding an object too hard. Release the object / "
            "open the gripper and power-cycle the arm before trying again."
        )
    if "missing motor ids" in low or "motor check failed" in low:
        return (
            "A follower motor isn't responding (often the gripper, id 6). If a skill was holding an object "
            "it likely overloaded — remove it, power-cycle the arm, then try teleoperation first."
        )
    if "could not connect" in low or "failed to connect" in low or "not connected" in low:
        return "Couldn't connect to the arm — make sure it's plugged in, powered on, and on the right port."
    if "frame is too old" in low or "no frame" in low or "frame timeout" in low:
        return (
            "A camera can't keep up — frames are arriving too slowly. Lower its resolution/FPS, "
            "set FOURCC=MJPG, and close other heavy apps, then try again."
        )
    if "failed to set capture_" in low or "actual_width" in low or "actual_height" in low:
        return "A camera doesn't support the configured resolution — open camera settings and click Auto."
    if "permission" in low and ("port" in low or "com" in low):
        return "Couldn't open the serial port — close anything else using it, or run `lelab --stop`."
    if "relative-action" in low or "relative chunk actions" in low:
        return (
            "This policy was trained with relative actions and needs RTC (chunked) inference. "
            "Update lelab and try again — recent versions enable that automatically."
        )
    return None


# Errors that mean the policy actually ran and only shutdown/cleanup tripped —
# e.g. disabling torque on a gripper still holding an object. Connection-loss
# errors are deliberately excluded: a mid-run disconnect is a real failure.
_CLEANUP_MARKERS = ("overload", "torque_enable")


def _classify_outcome(rc: int | None, rollout_started: bool, error_text: str | None) -> str:
    """ok | ran_with_warning | failed.

    A non-zero exit *after* the rollout main loop started, where the error is a
    torque-disable/overload on shutdown, means the skill ran but a motor (usually
    the loaded gripper) complained during cleanup — that's a warning, not a
    failure, so the UI shouldn't call a working run "failed"."""
    if not rc:
        return "ok"
    low = (error_text or "").lower()
    if rollout_started and any(marker in low for marker in _CLEANUP_MARKERS):
        return "ran_with_warning"
    return "failed"


def _clear_inference_state_locked():
    """Caller owns _state_lock. Invalidate late work before releasing the slot."""
    global inference_active, _inference_proc, _preparation_proc, _inference_start_id, _inference_cancel
    global _inference_started_at, _inference_rollout_started_at, _inference_meta
    if _inference_cancel is not None:
        _inference_cancel.set()
    inference_active = False
    _inference_proc = _preparation_proc = None
    _inference_start_id = _inference_cancel = None
    _inference_started_at = _inference_rollout_started_at = None
    _inference_meta = {}


def _check_start_locked(identifier, cancelled):
    if not inference_active or _inference_start_id != identifier or cancelled.is_set():
        raise PreparationCancelled()


def _prepare_policy_path(reference, identifier, cancelled, log_handle):
    global _preparation_proc
    if Path(reference).is_dir():
        with _state_lock:
            _check_start_locked(identifier, cancelled)
        return _resolve_policy_path(reference)

    def register(process):
        global _preparation_proc
        with _state_lock:
            _check_start_locked(identifier, cancelled)
            _preparation_proc = process

    path = resolve_in_worker(reference, cancelled, register, log_handle)
    with _state_lock:
        _check_start_locked(identifier, cancelled)
        _preparation_proc = None
    # Preserve the existing local bridge's quantized-artifact rejection guard.
    return _resolve_policy_path(path)


def handle_start_inference(request: InferenceRequest) -> dict[str, Any]:
    """Start a one-shot rollout subprocess. Returns a dict — the route
    layer turns it into a JSON response or HTTPException as appropriate."""
    global inference_active, _inference_proc, _inference_started_at
    global _inference_rollout_started_at, _inference_meta
    global _inference_start_id, _inference_cancel

    # Mutex with teleop and recording: all three drive the same serial bus.
    from . import record as _record, teleoperate as _teleoperate

    with _state_lock:
        if _teleoperate.teleoperation_active:
            return {
                "success": False,
                "status_code": 409,
                "message": "Teleoperation is currently active. Stop it first.",
            }
        if _record.recording_active:
            return {
                "success": False,
                "status_code": 409,
                "message": "Recording is currently active. Stop it first.",
            }
        if inference_active:
            return {
                "success": False,
                "status_code": 409,
                "message": "Inference is already active. Stop it first.",
            }
        # Claim the slot now so a concurrent caller losing the race sees us.
        inference_active = True
        identifier = uuid.uuid4().hex
        cancelled = threading.Event()
        _inference_start_id, _inference_cancel = identifier, cancelled
        _inference_started_at = time.time()
        _inference_rollout_started_at = None
        _inference_meta = {'policy_ref': request.policy_ref, 'duration_s': request.duration_s,
                           'session_phase': 'preparing', 'log_path': None}

    # Opened partway through setup; the stdout pump thread takes ownership once
    # it starts. Tracked here so a failure before that hand-off can close it
    # instead of leaking the file handle (which on Windows also keeps the log
    # locked).
    log_handle = None
    log_path = None
    proc = None
    try:
        log_dir = Path.home() / '.cache/huggingface/lerobot/inference_logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f'{int(time.time())}-{identifier[:8]}.log'
        log_handle = log_path.open('w', buffering=1)
        with _state_lock:
            _check_start_locked(identifier, cancelled)
            _inference_meta['log_path'] = str(log_path)
        # `setup_follower_calibration_file` returns the basename without the
        # .json extension. We need that stripped form for `--robot.id`,
        # because lerobot appends `.json` itself when constructing
        # `calibration_dir / f"{id}.json"`.
        follower_id = setup_follower_calibration_file(request.follower_config)
        policy_path = _prepare_policy_path(request.policy_ref, identifier, cancelled, log_handle)
        with _state_lock:
            _check_start_locked(identifier, cancelled)
        source_config = _read_policy_config(policy_path)
        from .camera_inputs import validate_physical_camera_selection
        validate_physical_camera_selection(source_config, request.cameras)
        parameter_flags = []
        inference_flags = _rollout_inference_args(policy_path)
        if request.inference_parameters is not None or request.inference_mode != 'auto':
            if not os.environ.get("LELAB_NATIVE_RUNNER"):
                raise ValueError("Runtime parameter overrides require the local native extension")
            from native_parameters import validate_parameters
            from native_modes import selected_description, explicit_mode_flags
            description = selected_description(source_config, request.inference_mode)
            values = validate_parameters(request.inference_parameters or {}, description)
            explicit_flags = explicit_mode_flags(source_config, request.inference_mode, values)
            if explicit_flags is not None:
                inference_flags = explicit_flags
            parameter_flags.append(f"--fps={values['fps']}")
            for name in ("chunk_size", "n_action_steps"):
                if values[name] is not None:
                    parameter_flags.append(f"--policy.{name}={values[name]}")

        cmd = [
            sys.executable,
            "-m",
            "lerobot.scripts.lerobot_rollout",
            "--strategy.type=base",
            f"--policy.path={policy_path}",
            f"--policy.device={_detect_device()}",
            "--robot.type=so101_follower",
            f"--robot.port={request.follower_port}",
            f"--robot.id={follower_id}",
            f"--task={request.task}",
            f"--duration={request.duration_s}",
            *inference_flags,
            *parameter_flags,
        ]
        native_session_dir = None
        native_runner = os.environ.get("LELAB_NATIVE_RUNNER")
        if native_runner:
            native_session_dir = Path(os.environ["LELAB_NATIVE_SESSIONS"]) / uuid.uuid4().hex
            native_session_dir.mkdir(parents=True)
            cmd[1:3] = [native_runner]
            cmd.append("--interactive=true")
        if request.cameras:
            cmd.append(f"--robot.cameras={_format_cameras_arg(request.cameras)}")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if native_session_dir:
            env["LELAB_NATIVE_SESSION_DIR"] = str(native_session_dir)
            env["LELAB_NATIVE_SOURCE_CONFIG"] = json.dumps(source_config)
        # Feed a single newline into stdin so SOFollower.calibrate()'s
        # `input("Press ENTER to use the calibration file ...")` returns "" and
        # writes the existing calibration to the motors instead of hanging
        # forever waiting for an interactive operator. Subsequent input()
        # calls in the recalibration path get EOF and raise — which is fine,
        # because we never want to enter that path from the UI.
        # Atomic hand-off: Stop either cancels preparation or sees this child.
        with _state_lock:
            _check_start_locked(identifier, cancelled)
            proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, env=env, start_new_session=True,
            )
            _inference_proc = proc
            _inference_meta.update(session_phase='setting_up',
                native_session_dir=str(native_session_dir) if native_session_dir else None)
        try:
            assert proc.stdin is not None
            proc.stdin.write(b"\n")
            proc.stdin.flush()
            proc.stdin.close()
        except Exception as exc:
            logger.warning("Failed to seed stdin for inference subprocess: %s", exc)
        threading.Thread(
            target=_pump_stdout,
            args=(proc, log_handle),
            name="inference-stdout-pump",
            daemon=True,
        ).start()
        log_handle = None  # pump thread owns and closes it from here on
        with _state_lock:
            _check_start_locked(identifier, cancelled)
    except PreparationCancelled:
        if log_handle is not None:
            log_handle.close()
        with _state_lock:
            if _inference_start_id == identifier:
                _clear_inference_state_locked()
        return {'success': True, 'cancelled': True, 'message': 'Inference preparation cancelled',
                'log_path': str(log_path) if log_path else ''}
    except Exception as exc:
        logger.exception("Failed to start inference")
        if proc is not None and proc.poll() is None:
            try:
                terminate_process(proc)
            except Exception as stop_error:
                return {'success': False, 'status_code': 500,
                        'message': f'Inference startup cleanup has not exited: {stop_error}'}
        # Startup failed before the pump thread took over (most often Popen
        # itself). Release the slot and close the log file if we opened it before
        # failing, so the handle isn't leaked.
        if log_handle is not None:
            log_handle.close()
        with _state_lock:
            if _inference_start_id == identifier:
                _clear_inference_state_locked()
        return {"success": False, "status_code": 500, "message": f"Failed to start inference: {exc}"}

    logger.info("Inference started: pid=%s policy=%s", proc.pid, policy_path)
    return {"success": True, "message": "Inference started", "log_path": str(log_path)}


def handle_stop_inference() -> dict[str, Any]:
    global inference_active, _inference_proc, _inference_started_at
    global _inference_rollout_started_at, _inference_meta

    with _state_lock:
        identifier = _inference_start_id
        if _inference_cancel is not None:
            _inference_cancel.set()
        proc = _inference_proc or _preparation_proc
        if proc is None:
            _clear_inference_state_locked()
            return {'success': True, 'message': 'Inference preparation cancelled; no robot process was started'}
        _inference_meta['session_phase'] = 'stopping'

    try:
        terminate_process(proc)
    except Exception as exc:
        logger.exception("Stop inference: %s", exc)
        return {'success': False, 'status_code': 500, 'message': f'Inference has not exited: {exc}'}

    with _state_lock:
        if _inference_start_id == identifier and proc in (_inference_proc, _preparation_proc):
            _clear_inference_state_locked()
    return {"success": True, "message": "Inference stopped"}


def handle_inference_status() -> dict[str, Any]:
    global inference_active, _inference_proc, _inference_started_at
    global _inference_rollout_started_at, _inference_meta

    # Finalise state lazily if the subprocess died on its own.
    with _state_lock:
        proc = _inference_proc
        if proc is not None and proc.poll() is not None:
            rc = proc.returncode
            logger.info("Inference subprocess exited rc=%s", rc)
            finished_meta = _inference_meta
            finished_started = _inference_started_at
            finished_rollout_started = _inference_rollout_started_at
            _clear_inference_state_locked()
            # On failure, surface the real error from the log so the UI doesn't
            # have to send the user digging through the cache.
            error = _extract_error_from_log(finished_meta.get("log_path")) if rc else None
            outcome = _classify_outcome(rc, finished_rollout_started is not None, error)
            return {
                "inference_active": False,
                "exited": True,
                "exit_code": rc,
                "outcome": outcome,
                "error": error,
                "hint": _friendly_hint(error),
                "policy_ref": finished_meta.get("policy_ref"),
                "duration_s": finished_meta.get("duration_s"),
                "log_path": finished_meta.get("log_path"),
                "started_at": finished_started,
                "rollout_started_at": finished_rollout_started,
                "rollout_elapsed_s": 0,
                "elapsed_s": 0,
            }
        elapsed = (time.time() - _inference_started_at) if _inference_started_at else 0
        rollout_elapsed = time.time() - _inference_rollout_started_at if _inference_rollout_started_at else 0
        return {
            "inference_active": inference_active,
            'session_phase': _inference_meta.get('session_phase'),
            "started_at": _inference_started_at,
            "rollout_started_at": _inference_rollout_started_at,
            "elapsed_s": elapsed,
            "rollout_elapsed_s": rollout_elapsed,
            "duration_s": _inference_meta.get("duration_s"),
            "policy_ref": _inference_meta.get("policy_ref"),
            "log_path": _inference_meta.get("log_path"),
        }
