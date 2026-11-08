"""Existing packed weights must bind completely after Transformers upgrades."""
import os
import pytest
pytestmark = pytest.mark.local_artifact
import json
from pathlib import Path

import torch
from accelerate import init_empty_weights
from torch import nn

from checkpoint import tensor_name_candidates


def test_existing_fp8_artifact_matches_v070_native_blueprint():
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    artifact = Path(os.environ['LELAB_TEST_ARTIFACT'])
    config = PI05Config.from_pretrained(str(artifact))
    config.device = None
    config.dtype = torch.bfloat16
    with init_empty_weights():
        policy = PI05Policy(config)
    policy.model.paligemma_with_expert.paligemma.lm_head = nn.Identity()
    policy.model.paligemma_with_expert.gemma_expert.lm_head = nn.Identity()
    meta = json.loads((artifact / 'quantization.json').read_text())
    source = meta['source_tensors']
    bound = set()
    for name, parameter in policy.named_parameters():
        key = next((key for key in tensor_name_candidates(name) if key in source), None)
        assert key is not None, name
        assert list(parameter.shape) == source[key]['shape'], name
        bound.add(key)
    assert set(source) - bound == set(meta['ignored_source_tensors'])
