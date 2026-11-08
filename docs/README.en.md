# LeLab Puls — community edition

A community distribution of improvements built on [Hugging Face LeLab](https://github.com/huggingface/leLab). Includes LeRobot v0.7.0 adaptation, configurable model caches, JSON/YAML training configuration editing, native guided RTC selection for compatible SmolVLA/PI0/PI05 configurations, browser inference controls, Linux/WSL camera handling, clearer fault/stop reporting, and optional PI05 FP8 deployment.

The full tutorial is in the [root README](../README.md). This repository contains source and templates; weights, datasets, tokens, calibration files and private device identities are excluded.

## Quick start (Ubuntu / WSL2, Python 3.12)

```bash
git clone https://github.com/Whyynnot/lelab_puls.git
cd lelab_puls
sudo apt-get update
sudo apt-get install -y python3.12-venv python3-dev build-essential git ffmpeg \
  libglib2.0-0 libegl1 libusb-1.0-0-dev portaudio19-dev
bash scripts/setup.sh
cp .env.example .env
bash start_lelab.sh
```

Open http://localhost:8000. Setup uses PyTorch 2.10.0 / CUDA 12.6, Torchvision 0.25.0, TorchCodec 0.10.0 and Transformers 5.22.2. The enhanced launcher targets Linux/WSL; complete native Windows/macOS support is not verified.

Edit `.env` to select `LELAB_CACHE_ROOT`, e.g. `/mnt/e/lelab_cache` in WSL. Default: `~/.cache/lelab`. This changes model caches; it does not move existing recordings or calibration files. Log in using `.venv/bin/hf auth login` if required by your model.

Configure/calibrate your own robot and verify camera roles before starting inference. Load a model, choose automatic/synchronous/RTC mode if supported, and enter the training task prompt. ACT does not use guided RTC. Native mode changes require a stopped task; the FP8 page uses a separate control engine.

Training → Configuration supports JSON/YAML import and a complete config editor. Apply changes before submitting; native settings go under `extra_config`. Field support depends on the policy. See `config/training-act.example.yaml`.

Optional stable USB identities: copy `config/hardware.example.json` to `config/hardware.local.json`, replace paths, set `LELAB_HARDWARE_CONFIG` to its absolute path. Windows USB recovery runs only if you configure your own helper. Keep the arm supported when diagnosing device loss; software cannot hold an arm through power/USB failure.

Run `.venv/bin/python scripts/doctor.py` for import/cache checks. See [inference architecture](inference.md), [hardware troubleshooting](hardware.md), [FP8 instructions](fp8.md), and [contribution/testing instructions](../CONTRIBUTING.md).

Apache-2.0; preserve upstream notices. This is an experimental community branch and is not an official Hugging Face release.
