"""Bounded retries for dataset uploads; leave TLS checks and auth failures intact."""
import logging
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


def upload_dataset_with_retry(dataset, *, tags, private, attempts=5, sleep=time.sleep):
    if attempts < 1:
        raise ValueError('attempts must be positive')
    for attempt in range(attempts):
        try:
            # Use native LeRobot upload, card and v3 tag behavior. Repeating a
            # partial upload preserves the same repo and uses Hub file caching.
            return dataset.push_to_hub(tags=tags, private=private)
        except Exception as error:
            if attempt + 1 == attempts or not is_transient_upload_error(error):
                raise
            delay = min(8, 2**attempt)
            logger.warning('Dataset upload transport interrupted (%s); attempt %d/%d in %ss',
                           type(error).__name__, attempt + 2, attempts, delay)
            sleep(delay)
