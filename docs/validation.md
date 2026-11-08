# Initial distribution validation — 2026-11-08

These checks validate the exported source tree, not the original working directory. Model weights, real observations and calibration files were excluded.

| Check | Result |
| --- | --- |
| Portable configuration + bridge + training config/cache/rollout regression tests | **156 passed**, 3 local-artifact tests deselected |
| Core tests excluding the two media modules | **290 passed** |
| Frontend tests after dependency updates | **23 files / 90 tests passed** |
| TypeScript `tsc --noEmit -p tsconfig.app.json` | Passed |
| Clean `npm ci --ignore-scripts` + production build | Passed |
| npm audit after dependency updates | **0 reported vulnerabilities** at validation time |
| Python source compilation and Bash syntax | Passed |
| Doctor: imports, native RTC API, cache path, frontend | Passed in the reference environment |
| Isolated server on port 18081 | Health, built page, profiles, hardware status and camera inventory returned HTTP 200; server stopped afterwards |
| Export review | No credential-pattern matches, symlinks or files over 5 MiB in the reviewed publication files |

The backend rows overlap; their counts should not be added. The broader suite was interrupted after blocking inside native OpenCV video decoding in this WSL environment. No all-tests-passed claim is made for the complete inherited upstream suite. The two excluded modules are `test_datasets.py` and `test_episode_media.py`; their source remains available for further validation on other systems.

Reference runtime: Python 3.12, LeRobot 0.7.0, Torch 2.10.0+cu126, Torchvision 0.25.0+cu126, TorchCodec 0.10.0, Transformers 5.22.2, Rustypot 1.13.1. Tests used the existing reference Python environment with imports explicitly directed to the exported repository. A complete multi-gigabyte Python installation on a clean host was not repeated. The npm install/build was performed cleanly against the committed lockfile.

Dependency hygiene changes update compatible frontend packages, remove unused direct Zod, and override the typography parser to the fixed 6.1.5 line and the TypeScript ESLint minimatch dependency to 11.0.5+. Frontend tests/typecheck/build were repeated after these changes. Vite still reports a large bundle advisory and a legacy esbuild option warning; these are not build failures.

This publication pass did not run a real robot or execute the model/GPU artifact tests. Checkpoint and hardware behavior must be validated with the user's own model, camera topology and calibration. CI runs software checks only; it cannot certify physical behavior.
