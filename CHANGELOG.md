# Changelog

## 2026-11-08 — Initial community source distribution

- Export the working LeLab enhancements into a standalone repository, preserving upstream Apache-2.0 notices.
- Target official LeRobot v0.7.0 with an explicit reference PyTorch/Transformers stack.
- Add JSON/YAML training config import, validated manual editing, pretrained controls and native extra options.
- Expose compatible native guided RTC for SmolVLA, PI0 and PI05; validate mode/chunk settings and preserve automatic mode.
- Include native browser session controls, prompt reset/resume and observation monitoring.
- Include Linux/WSL stable camera identities, isolated previews, serial preflight, first-fault reporting and bounded stop feedback.
- Include the optional native PI05 language-only FP8 W8A8 exporter and deployment panel.
- Replace machine-specific cache paths and USB IDs with environment settings and local configuration examples; make Windows USB recovery opt-in.
- Exclude model/data/calibration/token/runtime files and original private deployment history.
- Add setup/build/doctor tooling, Chinese tutorials, English quick start, architecture diagrams and contribution instructions.
- Use the official PyPI LeRobot 0.7.0 distribution consistently for core and policy extras, avoiding mixed Git/index dependency candidates during clean installation.

Reference installation and selected no-hardware regression checks are recorded in docs/validation.md. No claim of universal robot or policy compatibility is made.
