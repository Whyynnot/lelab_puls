"""Strict native PI05 checkpoint reader, precision blueprint and sharded writer."""
import json
import re
import struct
from pathlib import Path, PurePosixPath

NATIVE = re.compile(r'^(?:model\.)?paligemma_with_expert\.paligemma\.model\.language_model\.layers\.(\d+)\.(self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)\.weight$')


def tensor_name_candidates(name):
    """Transformers 5.22 removed SigLIP's redundant vision_model wrapper.

    Keep artifacts immutable and map only this known module rename; tensor
    shapes, dtypes and complete coverage are still checked by each loader.
    """
    variants = [name]
    if '.vision_tower.vision_model.' in name:
        variants.append(name.replace('.vision_tower.vision_model.', '.vision_tower.'))
    elif '.vision_tower.' in name:
        variants.append(name.replace('.vision_tower.', '.vision_tower.vision_model.'))
    return list(dict.fromkeys(candidate for variant in variants
                              for candidate in (variant, variant.removeprefix('model.'))))

def safe_relative(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name:
        raise ValueError(f'Unsafe checkpoint path: {name}')
    return path

def file_header(path):
    with path.open('rb') as handle:
        length_bytes = handle.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f'Truncated safetensors: {path}')
        length = struct.unpack('<Q', length_bytes)[0]
        if not 0 < length <= 32 * 1024 * 1024:
            raise ValueError('Invalid safetensors header size')
        raw = handle.read(length)
        if len(raw) != length:
            raise ValueError('Truncated safetensors header')
    header = json.loads(raw)
    end = max((v['data_offsets'][1] for k, v in header.items() if k != '__metadata__'), default=0)
    if path.stat().st_size != 8 + length + end:
        raise ValueError(f'Truncated or inconsistent tensor payload: {path}')
    return header

class Source:
    def __init__(self, directory):
        self.path = Path(directory).resolve()
        self.config = json.loads((self.path / 'config.json').read_text())
        if self.config.get('type') != 'pi05':
            raise ValueError('Only PI05 checkpoints are supported')
        if (self.path / 'quantization.json').exists() or (self.path / 'EXPORT_INCOMPLETE').exists():
            raise ValueError('Input must be original weights, not an existing quantized artifact')
        self.info, self.files, self.aliases = {}, {}, {}
        index = self.path / 'model.safetensors.index.json'
        mapping = json.loads(index.read_text())['weight_map'] if index.exists() else None
        paths = sorted({str(safe_relative(n)) for n in mapping.values()}) if mapping else [p.name for p in sorted(self.path.glob('model*.safetensors'))]
        if not paths:
            raise FileNotFoundError('No model*.safetensors in source directory')
        for name in paths:
            header = file_header(self.path / name)
            for key, value in header.items():
                if key == '__metadata__':
                    self.aliases.update(value)
                    continue
                if key in self.info:
                    raise ValueError(f'Duplicate source tensor: {key}')
                if mapping is not None and mapping.get(key) != name:
                    raise ValueError(f'Shard index mismatch: {key}')
                self.info[key], self.files[key] = value, name
        if mapping is not None and set(mapping) != set(self.info):
            raise ValueError('Incomplete shard index')
        self.aliases = {k: v for k, v in self.aliases.items()
                        if isinstance(v, str) and k not in self.info and (v in self.info or v in self.aliases)}
        self.aliases = {k: self.canonical(k) for k in self.aliases}

    def canonical(self, key):
        seen = set()
        while key in self.aliases:
            if key in seen:
                raise ValueError('Cyclic tensor alias metadata')
            seen.add(key)
            key = self.aliases[key]
        if key not in self.info:
            raise ValueError(f'Unresolved alias: {key}')
        return key

    def tensor(self, key):
        from safetensors import safe_open
        with safe_open(self.path / self.files[key], framework='pt', device='cpu') as handle:
            return handle.get_tensor(key)

def native_float_storage(source):
    """Match the validated native PI05 mixed precision using a META blueprint.

    Public pi05_base is distributed in FP32. Retaining all its FP32 matrices
    would defeat the 8GB deployment goal. Do not guess which norms stay FP32.
    No complete model parameters are allocated on CPU or GPU here.
    """
    import torch
    from torch import nn
    from accelerate import init_empty_weights
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    config = PI05Config.from_pretrained(str(source.path))
    config.dtype = torch.bfloat16
    config.device = None
    config.compile_model = False
    config.gradient_checkpointing = False
    with init_empty_weights():
        policy = PI05Policy(config)
    policy.model.paligemma_with_expert.paligemma.lm_head = nn.Identity()
    policy.model.paligemma_with_expert.gemma_expert.lm_head = nn.Identity()
    dtypes = {}
    required_parameters = {name for name, _ in policy.named_parameters()}
    for name, value in [*policy.named_parameters(), *policy.named_buffers()]:
        matched = False
        for candidate in tensor_name_candidates(name):
            key = source.aliases.get(candidate, candidate)
            if key not in source.info:
                continue
            if list(value.shape) != source.info[key]['shape']:
                raise ValueError(f'Native blueprint shape mismatch: {name}')
            if key in dtypes and dtypes[key] != value.dtype:
                raise ValueError(f'Conflicting native precision for {key}')
            dtypes[key] = value.dtype
            matched = True
            break
        if not matched and name in required_parameters:
            raise ValueError(f'Missing native policy parameter: {name}')
    ignored = set(source.info) - set(dtypes)
    heads = ('paligemma_with_expert.paligemma.lm_head.', 'paligemma_with_expert.gemma_expert.lm_head.')
    if any(not key.removeprefix('model.').startswith(heads) for key in ignored):
        raise ValueError(f'Native blueprint cannot account for tensors: {sorted(ignored)[:8]}')
    return dtypes, sorted(ignored)

class ShardWriter:
    def __init__(self, output, limit):
        self.output, self.limit = output, limit
        self.buffer, self.size, self.mapping, self.total, self.count = {}, 0, {}, 0, 0

    def add(self, key, tensor):
        tensor = tensor.detach().cpu().contiguous()
        size = tensor.numel() * tensor.element_size()
        if self.buffer and self.size + size > self.limit:
            self.flush()
        self.buffer[key] = tensor
        self.size += size
        self.total += size

    def flush(self):
        if not self.buffer:
            return
        from safetensors.torch import save_file
        name = f'model-{self.count + 1:05d}.safetensors'
        save_file(self.buffer, self.output / name, metadata={'format': 'pt'})
        self.mapping.update(dict.fromkeys(self.buffer, name))
        self.buffer, self.size = {}, 0
        self.count += 1

    def finish(self):
        self.flush()
        (self.output / 'model.safetensors.index.json').write_text(json.dumps(
            {'metadata': {'total_size': self.total}, 'weight_map': self.mapping}, indent=2))

def make_plan(info, aliases, config, backend='lerobot'):
    if backend != 'lerobot' or config.get('use_visual_memory') or config.get('use_proprioceptive_memory'):
        raise ValueError('Only the validated native PI05 layout is supported')
    selected, layers = set(), {}
    for key in set(info) | set(aliases):
        match = NATIVE.match(key)
        if not match:
            continue
        canonical = aliases.get(key, key)
        descriptor = info[canonical]
        if len(descriptor['shape']) != 2 or descriptor['dtype'] not in {'BF16','F16','F32'}:
            raise ValueError('Expected floating 2D language weights')
        layers.setdefault(int(match.group(1)), set()).add(match.group(2))
        selected.add(canonical)
    expected = {'self_attn.'+x+'_proj' for x in 'qkvo'} | {'mlp.'+x+'_proj' for x in ['gate','up','down']}
    if not layers or sorted(layers) != list(range(max(layers)+1)) or any(v != expected for v in layers.values()):
        raise ValueError('Incomplete native language projections')
    if len(selected) != 7*len(layers):
        raise ValueError('Conflicting language projections')
    return {'backend':'lerobot','quantized_weights':sorted(selected),'language_layer_count':len(layers)}
