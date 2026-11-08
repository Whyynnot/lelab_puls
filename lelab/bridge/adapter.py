"""CPU observation/robot adapter for the separately running FP8 service."""
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import torch
import sentencepiece as spm
from pydantic import BaseModel, Field, model_validator

ROOT = Path(__file__).resolve().parents[2]
JOINTS = ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos',
          'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']


class Profile(BaseModel):
    name: str = Field(pattern=r'^[a-zA-Z0-9_-]{1,64}$')
    artifact: str
    allow_robot: bool = False
    state_names: list[str] = Field(default_factory=lambda: JOINTS.copy())
    action_names: list[str] = Field(default_factory=lambda: JOINTS.copy())
    camera_map: dict[str, str] = Field(default_factory=lambda: {
        'head': 'observation.images.head', 'wrist': 'observation.images.wrist'})
    use_degrees: bool = True
    execution_horizon: int = Field(default=10, ge=1, le=50)
    fps: int = Field(default=30, ge=1, le=60)
    max_relative_target: float = Field(default=5.0, gt=0, le=30)
    tokenizer_model: str | None = None
    default_task: str = ''
    note: str = ''
    sample_manifest: str = 'validation/sample_manifest.json'

    @model_validator(mode='after')
    def contract(self):
        if set(self.state_names) != set(JOINTS) or len(self.state_names) != 6:
            raise ValueError('state_names must list each of the six SO101 joints once')
        if set(self.action_names) != set(JOINTS) or len(self.action_names) != 6:
            raise ValueError('action_names must list each of the six SO101 joints once')
        if set(self.camera_map) != {'head', 'wrist'} or len(set(self.camera_map.values())) != 2:
            raise ValueError('camera_map must map head and wrist to distinct checkpoint image keys')
        return self


def read_profile(name):
    if not isinstance(name, str) or not __import__('re').fullmatch(r'[a-zA-Z0-9_-]{1,64}', name):
        raise ValueError('Invalid profile name')
    return Profile.model_validate_json((ROOT/'lelab/profiles'/f'{name}.json').read_text())


def read_sample_manifest(profile):
    path = ROOT / profile.sample_manifest
    manifest = json.loads(path.read_text())
    if Path(manifest['artifact']).resolve() != Path(profile.artifact).resolve():
        raise ValueError('Offline samples belong to another model; prepare samples with the selected checkpoint processors')
    saved = Profile(**manifest['profile'])
    for field in ['state_names', 'action_names', 'camera_map']:
        if getattr(saved, field) != getattr(profile, field):
            raise ValueError(f'Offline sample profile differs in {field}')
    return manifest


def validate_artifact(profile, robot=False, offline=False):
    path = Path(profile.artifact).resolve(strict=True)
    meta = json.loads((path/'quantization.json').read_text())
    cfg = json.loads((path/'config.json').read_text())
    if meta.get('format') != 'pi05-fp8-w8a8-v1' or meta.get('weights_kind') != 'pretrained':
        raise ValueError('Expected a real pretrained native PI05 FP8 W8A8 artifact')
    if (path/'EXPORT_INCOMPLETE').exists():
        raise ValueError('Incomplete quantized export')
    if meta['format'] == 'pi05-fp8-w8a8-v1':
        import hashlib
        if meta.get('backend') != 'lerobot' or hashlib.sha256((path/'config.json').read_bytes()).hexdigest() != meta.get('config_sha256'):
            raise ValueError('Invalid FP8 backend or configuration checksum')
        from pathlib import PurePosixPath
        index = json.loads((path/'model.safetensors.index.json').read_text())['weight_map']
        for name in set(index.values()):
            shard = PurePosixPath(name)
            if shard.is_absolute() or '..' in shard.parts or '\\' in name or not (path/name).is_file():
                raise ValueError('Missing or invalid FP8 shard')
    if cfg.get('use_visual_memory') or cfg.get('use_proprioceptive_memory'):
        raise ValueError('MEM is not supported by this FP8 runtime')
    if robot:
        if not profile.allow_robot and not offline:
            raise ValueError('This profile is offline only; configure a SO101-finetuned checkpoint first')
        if cfg['input_features']['observation.state']['shape'] != [6]:
            raise ValueError('SO101 deployment requires checkpoint state width 6')
        if cfg['output_features']['action']['shape'] != [6]:
            raise ValueError('SO101 deployment requires checkpoint action width 6; base-model slicing is forbidden')
        images = {k for k,v in cfg['input_features'].items()
                  if v['type'] == 'VISUAL' and '.empty_camera_' not in k}
        if set(profile.camera_map.values()) != images:
            raise ValueError(f'camera_map must exactly cover checkpoint cameras: {sorted(images)}')
        if profile.execution_horizon > cfg.get('chunk_size', 50):
            raise ValueError('execution_horizon exceeds checkpoint chunk_size')
        saved_names = cfg.get('action_feature_names')
        if saved_names and saved_names != profile.action_names:
            raise ValueError('Profile action order differs from checkpoint action_feature_names')
        for file in ['policy_preprocessor.json', 'policy_postprocessor.json']:
            if not (path/file).is_file():
                raise ValueError(f'Missing saved checkpoint processor: {file}')
    return path, cfg


class SentencePieceTokenizer:
    """Offline PaliGemma SP tokenizer; same BOS/padding as our fidelity tests."""
    def __init__(self, path):
        self.sp = spm.SentencePieceProcessor(model_file=str(path))

    def __call__(self, text, *, max_length, truncation, padding, padding_side, return_tensors):
        if padding_side != 'right' or padding != 'max_length' or return_tensors != 'pt':
            raise ValueError('Unsupported tokenizer contract')
        encoded = [self.sp.encode(t, add_bos=True) for t in text]
        if any(len(x) > max_length for x in encoded):
            raise ValueError('Task/state prompt exceeds token budget; shorten the task')
        return {'input_ids': torch.tensor([x+[0]*(max_length-len(x)) for x in encoded]),
                'attention_mask': torch.tensor([[1]*len(x)+[0]*(max_length-len(x)) for x in encoded])}


class ModelClient:
    def __init__(self, url='http://127.0.0.1:8765', timeout=20):
        self.url, self.timeout = url.rstrip('/'), timeout
        self.last_result = None

    def health(self):
        with urlopen(self.url+'/health', timeout=2) as response:
            return json.load(response)

    def require_artifact(self, artifact):
        health = self.health()
        if health.get('weights_kind') != 'pretrained' or health.get('format') != 'pi05-fp8-w8a8-v1':
            raise ValueError('FP8 server does not expose real pretrained weights')
        if health.get('artifact') != str(Path(artifact).resolve()):
            raise ValueError('Running FP8 server artifact differs from selected profile; restart it from this page')
        return health

    def infer(self, batch, rtc=None):
        batch = dict(batch)
        if rtc:
            batch['rtc_enabled'] = np.asarray(rtc.get('enabled', False), dtype=np.bool_)
            if rtc.get('previous') is not None:
                batch['prev_chunk_left_over'] = np.asarray(rtc['previous'], dtype=np.float32)
                batch['inference_delay'] = np.asarray(rtc['delay'], dtype=np.int64)
                batch['execution_horizon'] = np.asarray(rtc['horizon'], dtype=np.int64)
        data = io.BytesIO()
        np.savez(data, **batch)
        request = Request(self.url+'/infer', data=data.getvalue(),
                          headers={'Content-Type':'application/octet-stream'})
        with urlopen(request, timeout=self.timeout) as response:
            result = json.load(response)
        if result.get('space') != 'policy':
            raise ValueError('Unexpected server action space')
        actions = np.asarray(result['actions'], dtype=np.float32)
        if actions.ndim != 3 or actions.shape[0] != 1 or not np.isfinite(actions).all():
            raise ValueError('Invalid action chunk')
        self.last_result = {k:v for k,v in result.items() if k != 'actions'}
        return actions, result['inference_ms']


class RobotAdapter:
    """Restore training processors. Denormalize and anchor the whole chunk once."""
    def __init__(self, profile, *, offline=False):
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        # Import registers the policy-specific preparation step before restoration.
        from lerobot.policies.pi05 import processor_pi05  # noqa: F401
        from lerobot.policies.factory import make_pre_post_processors
        self.profile = profile
        artifact, raw = validate_artifact(profile, robot=True, offline=offline)
        config = PI05Config.from_pretrained(str(artifact))
        overrides = {'device_processor': {'device':'cpu'}}
        if (artifact/'tokenizer/tokenizer.json').is_file() and not profile.tokenizer_model:
            overrides['tokenizer_processor'] = {'tokenizer_name':str(artifact/'tokenizer')}
        if profile.tokenizer_model:
            if raw.get('text_tokenizer_name', 'google/paligemma-3b-pt-224') != 'google/paligemma-3b-pt-224':
                raise ValueError('Offline PaliGemma tokenizer override cannot be applied to another tokenizer')
            overrides['tokenizer_processor'] = {'tokenizer':SentencePieceTokenizer(profile.tokenizer_model)}
        self.pre, self.post = make_pre_post_processors(
            config, pretrained_path=artifact, preprocessor_overrides=overrides,
            postprocessor_overrides={'device_processor':{'device':'cpu'}})
        self.config = config
        from lerobot.processor import NormalizerProcessorStep, UnnormalizerProcessorStep
        for pipeline, cls, keys in [(self.pre, NormalizerProcessorStep, ['observation.state']),
                                     (self.post, UnnormalizerProcessorStep, ['action'])]:
            steps = [s for s in pipeline.steps if isinstance(s, cls)]
            if len(steps) != 1:
                raise ValueError('Expected exactly one saved normalization step')
            step = steps[0]
            for key in keys:
                stats = step.stats.get(key, {}) if step.stats else {}
                if not stats or key not in step.features:
                    raise ValueError(f'Missing training normalization/features for {key}')
                if not all(torch.isfinite(torch.as_tensor(v)).all() for v in stats.values()):
                    raise ValueError(f'Non-finite saved statistics for {key}')

    def prepare(self, observation, task):
        from lerobot.policies.common.vla_utils import resize_with_pad_torch
        state = np.asarray([observation[n] for n in self.profile.state_names], dtype=np.float32)
        if state.shape != (6,) or not np.isfinite(state).all():
            raise ValueError('Invalid SO101 observation state')
        raw = {'observation.state':torch.from_numpy(state), 'task':task}
        for camera, feature in self.profile.camera_map.items():
            image = np.asarray(observation[camera])
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError('Camera observations must be RGB uint8 HWC')
            height, width = self.config.input_features[feature].shape[-2:]
            tensor = resize_with_pad_torch(torch.from_numpy(image.copy()).float()[None]/255, height, width)
            raw[feature] = tensor[0].permute(2,0,1).contiguous()
        processed = self.pre(raw)
        names = set(self.profile.camera_map.values()) | {'observation.state',
            'observation.language.tokens','observation.language.attention_mask'}
        return {k:processed[k].detach().cpu().numpy() for k in names}

    def decode(self, actions):
        if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[-1] != 6:
            raise ValueError('Expected a [1, horizon, 6] SO101 action chunk')
        chunk = self.post(torch.from_numpy(actions.copy())).detach().cpu().numpy()[0]
        if not np.isfinite(chunk).all():
            raise ValueError('Non-finite decoded actions')
        return [{name:float(value) for name,value in zip(self.profile.action_names,row,strict=True)}
                for row in chunk]


    def rtc_prefix(self, previous, physical, observation):
        """Re-anchor relative-action prefixes before normalizing; absolute models reuse policy tensors."""
        if not self.config.use_relative_actions:
            return np.asarray(previous, np.float32)[None]
        from lerobot.policies.rtc.relative import reanchor_relative_rtc_prefix
        from lerobot.processor import RelativeActionsProcessorStep, NormalizerProcessorStep
        relative = next(s for s in self.pre.steps if isinstance(s, RelativeActionsProcessorStep))
        normalizer = next(s for s in self.pre.steps if isinstance(s, NormalizerProcessorStep))
        absolute = torch.tensor([[[row[n] for n in self.profile.action_names] for row in physical]], dtype=torch.float32)
        state = torch.tensor([observation[n] for n in self.profile.state_names], dtype=torch.float32)
        return reanchor_relative_rtc_prefix(absolute, state, relative, normalizer, 'cpu').numpy()
