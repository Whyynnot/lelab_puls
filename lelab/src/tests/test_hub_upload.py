import errno
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from huggingface_hub import constants

from lelab import hub_upload


def xet_initialization_error():
    namespace = {'__name__': 'huggingface_hub._upload_pipeline'}
    exec('def _coordinator_loop():\n raise OSError("I/O error: No such file or directory (os error 2)")', namespace)
    try:
        namespace['_coordinator_loop']()
    except OSError as error:
        return error


def test_only_xet_initialization_enoent_matches():
    assert hub_upload.is_xet_upload_initialization_error(xet_initialization_error())
    assert not hub_upload.is_xet_upload_initialization_error(FileNotFoundError(errno.ENOENT, 'missing local video'))
    assert not hub_upload.is_xet_upload_initialization_error(ValueError('No such file or directory'))


def test_xet_failure_preserves_upload_target_and_isolates_environment(monkeypatch, tmp_path):
    dataset = Mock(repo_id='owner/dataset', root=tmp_path)
    dataset.push_to_hub.side_effect = xet_initialization_error()
    monkeypatch.setattr(constants, 'HF_HUB_DISABLE_XET', False)
    monkeypatch.delenv('HF_HUB_DISABLE_XET', raising=False)
    run = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(hub_upload.subprocess, 'run', run)
    hub_upload.upload_dataset_with_retry(dataset, tags=['LeLab'], private=True)
    dataset.push_to_hub.assert_called_once_with(tags=['LeLab'], private=True)
    args, kwargs = run.call_args
    assert args[0] == [hub_upload.sys.executable, '-m', 'lelab.upload_worker']
    assert json.loads(kwargs['input']) == {'repo_id': 'owner/dataset', 'root': str(tmp_path),
                                          'tags': ['LeLab'], 'private': True, 'attempts': 5}
    assert kwargs['env']['HF_HUB_DISABLE_XET'] == '1'
    assert 'HF_HUB_DISABLE_XET' not in os.environ
    assert constants.HF_HUB_DISABLE_XET is False


def test_missing_local_file_is_not_retried_or_sent_to_worker(monkeypatch):
    dataset = Mock()
    dataset.push_to_hub.side_effect = FileNotFoundError(errno.ENOENT, 'missing local video')
    run = Mock()
    monkeypatch.setattr(hub_upload.subprocess, 'run', run)
    with pytest.raises(FileNotFoundError):
        hub_upload.upload_dataset_with_retry(dataset, tags=[], private=False)
    run.assert_not_called()
    assert dataset.push_to_hub.call_count == 1


def test_worker_failure_remains_visible_without_restarting_xet(monkeypatch, tmp_path):
    dataset = Mock(repo_id='owner/dataset', root=tmp_path)
    dataset.push_to_hub.side_effect = xet_initialization_error()
    monkeypatch.setattr(constants, 'HF_HUB_DISABLE_XET', False)
    run = Mock(return_value=SimpleNamespace(returncode=1, stderr='PermissionError: upload denied', stdout=''))
    monkeypatch.setattr(hub_upload.subprocess, 'run', run)
    with pytest.raises(RuntimeError, match='upload denied'):
        hub_upload.upload_dataset_with_retry(dataset, tags=[], private=False)
    assert run.call_count == dataset.push_to_hub.call_count == 1


def test_http_worker_does_not_recursively_fall_back(monkeypatch):
    dataset = Mock()
    dataset.push_to_hub.side_effect = xet_initialization_error()
    monkeypatch.setattr(constants, 'HF_HUB_DISABLE_XET', True)
    run = Mock()
    monkeypatch.setattr(hub_upload.subprocess, 'run', run)
    with pytest.raises(OSError):
        hub_upload.upload_dataset_with_retry(dataset, tags=[], private=False)
    run.assert_not_called()


def test_real_worker_process_reads_request_and_uses_process_local_transport(tmp_path):
    fake_package = tmp_path / 'packages' / 'lerobot'
    fake_package.mkdir(parents=True)
    (fake_package / '__init__.py').write_text('')
    (fake_package / 'datasets.py').write_text('''import json, os
from pathlib import Path
class LeRobotDataset:
    def __init__(self, repo_id, root, video_backend):
        self.repo_id, self.root = repo_id, root
    def push_to_hub(self, **kwargs):
        from huggingface_hub import constants
        Path(os.environ['UPLOAD_TEST_RESULT']).write_text(json.dumps({
            'repo_id': self.repo_id, 'root': str(self.root),
            'xet_disabled': constants.HF_HUB_DISABLE_XET, **kwargs}))
''')
    root = tmp_path / 'dataset'
    (root / 'meta').mkdir(parents=True)
    (root / 'meta/info.json').write_text('{}')
    result = tmp_path / 'result.json'
    source = Path(hub_upload.__file__).resolve().parents[1]
    env = {**os.environ, 'HF_HUB_DISABLE_XET': '1', 'UPLOAD_TEST_RESULT': str(result),
           'PYTHONPATH': os.pathsep.join([str(fake_package.parent), str(source)])}
    request = {'repo_id': 'owner/dataset', 'root': str(root), 'tags': ['LeLab'],
               'private': True, 'attempts': 2}
    subprocess.run([sys.executable, '-m', 'lelab.upload_worker'], input=json.dumps(request),
                   text=True, capture_output=True, env=env, check=True, timeout=20)
    assert json.loads(result.read_text()) == {
        'repo_id': 'owner/dataset', 'root': str(root), 'tags': ['LeLab'],
        'private': True, 'xet_disabled': True}
