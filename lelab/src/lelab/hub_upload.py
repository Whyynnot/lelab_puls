"""Bounded retries for dataset uploads; leave TLS checks and auth failures intact."""
import logging
import errno
import json
import os
import subprocess
import sys
import ssl
import time

import httpx
from huggingface_hub.errors import HfHubHTTPError

logger = logging.getLogger(__name__)


def is_transient_upload_error(error):
    """Retry transport failures and temporary HTTP statuses, never auth/data errors."""
    seen = set()
    errors = []
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        errors.append(error)
        error = error.__cause__ or error.__context__
    for error in errors:
        if isinstance(error, HfHubHTTPError):
            response = error.response
            return response is not None and response.status_code in {408, 429, 500, 502, 503, 504}
    if any(isinstance(error, ssl.SSLCertVerificationError) for error in errors):
        return False
    return any(isinstance(error, (httpx.TransportError, ssl.SSLError, ConnectionError)) for error in errors)


def is_xet_upload_initialization_error(error):
    """Match the observed Xet ENOENT, not a missing local dataset file."""
    if not isinstance(error, OSError):
        return False
    if error.errno != errno.ENOENT and 'No such file or directory' not in str(error):
        return False
    traceback = error.__traceback__
    while traceback is not None:
        frame = traceback.tb_frame
        if (frame.f_globals.get('__name__') == 'huggingface_hub._upload_pipeline'
                and frame.f_code.co_name == '_coordinator_loop'):
            return True
        traceback = traceback.tb_next
    return False


def _upload_in_http_worker(dataset, *, tags, private, attempts):
    """Keep transport overrides local to a child; model downloads stay untouched."""
    request = {'repo_id': dataset.repo_id, 'root': str(dataset.root),
               'tags': tags, 'private': private, 'attempts': attempts}
    result = subprocess.run(
        [sys.executable, '-m', 'lelab.upload_worker'],
        input=json.dumps(request), text=True, capture_output=True,
        env={**os.environ, 'HF_HUB_DISABLE_XET': '1', 'PYTHONUNBUFFERED': '1'},
    )
    if result.returncode:
        lines = (result.stderr or result.stdout or '').strip().splitlines()
        detail = lines[-1][-2000:] if lines else f'worker exit code {result.returncode}'
        raise RuntimeError(f'HTTP/LFS dataset upload failed: {detail}')


def upload_dataset_with_retry(dataset, *, tags, private, attempts=5, sleep=time.sleep):
    if attempts < 1:
        raise ValueError('attempts must be positive')
    for attempt in range(attempts):
        try:
            # Use native LeRobot upload, card and v3 tag behavior. Repeating a
            # partial upload preserves the same repo and uses Hub file caching.
            return dataset.push_to_hub(tags=tags, private=private)
        except Exception as error:
            from huggingface_hub import constants
            if not constants.HF_HUB_DISABLE_XET and is_xet_upload_initialization_error(error):
                logger.warning('Xet upload initialization failed; retrying this dataset in an isolated HTTP/LFS worker')
                return _upload_in_http_worker(dataset, tags=tags, private=private, attempts=attempts)
            if attempt + 1 == attempts or not is_transient_upload_error(error):
                raise
            delay = min(8, 2**attempt)
            logger.warning('Dataset upload transport interrupted (%s); attempt %d/%d in %ss',
                           type(error).__name__, attempt + 2, attempts, delay)
            sleep(delay)
