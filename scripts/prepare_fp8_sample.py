"""Prepare one real saved RGB/state observation for offline-only FP8 validation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'lelab/src'), str(ROOT / 'lelab/bridge')]
from adapter import Profile, RobotAdapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--observation', type=Path, required=True,
                        help='NPZ containing raw_state[6], head[H,W,3], wrist[H,W,3]; RGB uint8')
    parser.add_argument('--task', required=True)
    args = parser.parse_args()
    profile = Profile.model_validate_json(args.profile.read_text())
    adapter = RobotAdapter(profile, offline=True)
    with np.load(args.observation, allow_pickle=False) as arrays:
        state = np.asarray(arrays['raw_state'], np.float32)
        if state.shape != (6,) or not np.isfinite(state).all():
            raise ValueError('raw_state must contain six finite joint positions in profile.state_names order')
        observation = dict(zip(profile.state_names, state.tolist(), strict=True))
        observation.update({name: arrays[name].copy() for name in profile.camera_map})
    batch = adapter.prepare(observation, args.task)
    manifest = (ROOT / profile.sample_manifest).resolve()
    sample = manifest.parent / 'sample.npz'
    if manifest.exists() or sample.exists():
        raise FileExistsError('Choose a new sample_manifest; existing samples are not overwritten')
    manifest.parent.mkdir(parents=True, exist_ok=True)
    np.savez(sample, **batch, raw_state=state)
    manifest.write_text(json.dumps({
        'artifact': profile.artifact, 'profile': profile.model_dump(),
        'dataset': str(args.observation.resolve()),
        'samples': [{'id': 'saved-observation', 'split': 'evaluation',
                     'file': str(sample), 'task': args.task,
                     'sha256': hashlib.sha256(sample.read_bytes()).hexdigest()}],
    }, ensure_ascii=False, indent=2))
    print(f'Prepared offline observation: {manifest}. No motor/camera devices were opened.')


if __name__ == '__main__':
    main()
