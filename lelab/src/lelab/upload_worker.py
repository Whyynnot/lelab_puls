"""Isolated native dataset upload with process-local transport configuration."""
import json
import sys
from pathlib import Path


def main():
    request = json.load(sys.stdin)
    from huggingface_hub import constants
    if not constants.HF_HUB_DISABLE_XET:
        raise RuntimeError('HTTP upload worker requires HF_HUB_DISABLE_XET=1 before imports')
    root = Path(request['root'])
    if not (root / 'meta/info.json').is_file():
        raise FileNotFoundError(f'Dataset metadata missing: {root / "meta/info.json"}')
    from lerobot.datasets import LeRobotDataset
    from lelab.hub_upload import upload_dataset_with_retry
    dataset = LeRobotDataset(request['repo_id'], root=root, video_backend='pyav')
    upload_dataset_with_retry(dataset, tags=request['tags'], private=request['private'],
                              attempts=request['attempts'])


if __name__ == '__main__':
    main()
