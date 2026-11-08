"""Native LeRobot PI05, language-only FP8 E4M3 W8A8 inference.

Weights: per-matrix absmax scaling. Activations: dynamic per-call absmax.
Real torch._scaled_mm on SM89+, no fake-quantized floating GEMM fallback.
"""
import argparse
import gc
import hashlib
import json
import shutil
import time
from pathlib import Path

import torch
from torch import nn
from safetensors import safe_open

from checkpoint import Source, make_plan, native_float_storage, ShardWriter, safe_relative
from model_core import skeleton, cpu_embedding_for_validation

FORMAT = 'pi05-fp8-w8a8-v1'
ROOT = Path(__file__).resolve().parent


class FP8Linear(nn.Module):
    def __init__(self, weight, scale, bias=None):
        super().__init__()
        if weight.dtype != torch.float8_e4m3fn or weight.ndim != 2:
            raise ValueError('Expected packed E4M3 weight matrix')
        if scale.numel() != 1 or not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError('Invalid FP8 weight scale')
        self.out_features, self.in_features = weight.shape
        self.register_buffer('weight_fp8', weight)
        self.register_buffer('weight_scale', scale.float().reshape(()))
        self.register_buffer('bias', bias)

    @property
    def weight(self):
        # PI05 checks projection weight.dtype. It calls the module for GEMM.
        return torch.empty(0, device=self.weight_fp8.device, dtype=torch.bfloat16)

    def forward(self, inputs):
        shape = inputs.shape
        value = inputs.to(torch.bfloat16).reshape(-1, self.in_features).contiguous()
        if value.shape[0] == 0:
            return value.new_empty((*shape[:-1], self.out_features))
        scale = (value.abs().amax().float() / 448).clamp_min(1e-12)
        quantized = (value / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        result = torch._scaled_mm(quantized, self.weight_fp8.t(), scale_a=scale,
                                  scale_b=self.weight_scale, out_dtype=torch.bfloat16,
                                  use_fast_accum=False)
        if self.bias is not None:
            result = result + self.bias.to(result.dtype)
        return result.reshape(*shape[:-1], self.out_features)


def require_cuda():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9):
        raise RuntimeError('FP8 runtime requires CUDA SM89 or newer')


def export(source_directory, output_directory, device='cuda'):
    if device not in {'cpu', 'cuda'}:
        raise ValueError('Export device must be cpu or cuda')
    if device == 'cuda':
        require_cuda()
    source = Source(source_directory)
    plan = make_plan(source.info, source.aliases, source.config, 'lerobot')
    output = Path(output_directory).resolve()
    if output == source.path or source.path in output.parents or output in source.path.parents:
        raise ValueError('Output must be separate from source')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Output is nonempty; existing artifacts will not be overwritten')
    dtypes, ignored = native_float_storage(source)
    selected = set(plan['quantized_weights'])
    output.mkdir(parents=True, exist_ok=True)
    marker = output/'EXPORT_INCOMPLETE'
    marker.write_text('Export in progress; do not load.\n')
    writer = ShardWriter(output, 128*2**20)
    errors = []
    started = time.perf_counter()
    for index, key in enumerate(sorted(source.info)):
        if key in ignored:
            continue
        value = source.tensor(key)
        if key in selected:
            original = value.to(device, dtype=torch.bfloat16).contiguous()
            if not torch.isfinite(original).all():
                raise ValueError(f'Non-finite weight: {key}')
            scale = (original.abs().amax().float()/448).clamp_min(1e-12)
            packed = (original/scale).clamp(-448,448).to(torch.float8_e4m3fn)
            prefix = key.removesuffix('.weight')
            writer.add(prefix+'.weight_fp8', packed)
            writer.add(prefix+'.weight_scale', scale)
            delta = packed.float()*scale - original.float()
            errors.append({'weight':key, 'mse':delta.square().mean().item(),
                           'max_abs':delta.abs().max().item()})
            del original, packed, delta, scale
        else:
            writer.add(key, value.to(dtypes[key]))
        del value
        if index % 100 == 0:
            print(f'Exported {index}/{len(source.info)} tensors', flush=True)
    writer.finish()
    for file in source.path.iterdir():
        if file.name.startswith('model') and (file.name.endswith('.safetensors') or file.name.endswith('.index.json')):
            continue
        if file.is_file() and file.suffix in {'.json','.safetensors','.model','.txt','.md','.jinja'}:
            shutil.copy2(file, output/file.name)
        elif file.is_dir() and file.name in {'tokenizer','assets'}:
            shutil.copytree(file, output/file.name)
    metadata = {'format':FORMAT, 'backend':'lerobot', 'weights_kind':'pretrained',
        'source':str(source.path), 'source_tensors':source.info, 'aliases':source.aliases,
        'quantized_weights':sorted(selected), 'ignored_source_tensors':ignored,
        'float_policy':'native_bf16_fp32_blueprint', 'weight_dtype':'float8_e4m3fn',
        'weight_granularity':'per_matrix_absmax', 'activation_dtype':'float8_e4m3fn',
        'activation_granularity':'dynamic_per_call_per_tensor_absmax',
        'output_dtype':'bfloat16', 'kernel':'torch._scaled_mm / cuBLAS FP8',
        'stored_tensor_bytes':writer.total, 'shards':writer.count,
        'config_sha256':hashlib.sha256((output/'config.json').read_bytes()).hexdigest(),
        'torch':torch.__version__, 'export_device':device,
        'export_seconds':time.perf_counter()-started,
        'weight_drift':errors}
    (output/'quantization.json').write_text(json.dumps(metadata, indent=2))
    marker.unlink()
    print(json.dumps({k:v for k,v in metadata.items() if k not in
        {'source_tensors','aliases','quantized_weights','weight_drift','huggingface'}}, indent=2), flush=True)
    return metadata


def load_artifact(directory, cpu_embedding=False):
    require_cuda()
    directory = Path(directory).resolve()
    if (directory/'EXPORT_INCOMPLETE').exists():
        raise ValueError('Incomplete FP8 export')
    meta = json.loads((directory/'quantization.json').read_text())
    if meta.get('format') != FORMAT or meta.get('backend') != 'lerobot':
        raise ValueError('Expected native PI05 FP8 W8A8 artifact')
    if hashlib.sha256((directory/'config.json').read_bytes()).hexdigest() != meta['config_sha256']:
        raise ValueError('FP8 configuration checksum mismatch')
    mapping = json.loads((directory/'model.safetensors.index.json').read_text())['weight_map']
    for name in mapping.values():
        safe_relative(name)
    def tensor(key):
        with safe_open(directory/mapping[key], framework='pt', device='cpu') as file:
            return file.get_tensor(key)
    policy = skeleton(directory)
    source = meta['source_tensors']
    def resolve(name):
        from checkpoint import tensor_name_candidates
        for candidate in tensor_name_candidates(name):
            canonical = meta['aliases'].get(candidate, candidate)
            if canonical in source:
                return canonical
        raise ValueError(f'Missing required source tensor: {name}')
    expected = {name:(tuple(p.shape),p.dtype) for name,p in policy.named_parameters()}
    bound = {name:resolve(name) for name in expected}
    ignored = set(meta['ignored_source_tensors'])
    heads = ('paligemma_with_expert.paligemma.lm_head.', 'paligemma_with_expert.gemma_expert.lm_head.')
    if any(not key.removeprefix('model.').startswith(heads) for key in ignored):
        raise ValueError('Unexpected ignored source tensor')
    if set(source) - set(bound.values()) - ignored:
        raise ValueError('Source tensors not accounted for by native policy')
    for name, (shape, _) in expected.items():
        if list(shape) != source[bound[name]]['shape']:
            raise ValueError(f'FP8 original shape mismatch: {name}')
    selected = set(meta['quantized_weights'])
    for name, (shape,dtype) in expected.items():
        key = bound[name]
        parent_name, leaf = name.rsplit('.',1)
        parent = policy.get_submodule(parent_name)
        if key in selected:
            module = policy.get_submodule(parent_name)
            if leaf != 'weight' or not isinstance(module,nn.Linear) or module.bias is not None:
                raise ValueError(f'Expected bias-free language linear: {name}')
            prefix = key.removesuffix('.weight')
            packed, scale = tensor(prefix+'.weight_fp8'), tensor(prefix+'.weight_scale')
            if tuple(packed.shape) != shape:
                raise ValueError(f'FP8 packed shape mismatch: {name}')
            replacement = FP8Linear(packed.to('cuda'), scale.to('cuda'))
            grandparent_name, child = parent_name.rsplit('.',1)
            setattr(policy.get_submodule(grandparent_name),child,replacement)
        else:
            device = 'cpu' if cpu_embedding and name.endswith('language_model.embed_tokens.weight') else 'cuda'
            value = tensor(key)
            if value.dtype != dtype:
                raise ValueError(f'Unexpected floating dtype: {key}')
            parent._parameters[leaf] = nn.Parameter(value.to(device),requires_grad=False)
    if any(p.is_meta for p in policy.parameters()):
        raise ValueError('Uninitialized FP8 policy parameters')
    if sum(isinstance(m,FP8Linear) for m in policy.modules()) != len(selected):
        raise ValueError('FP8 module count mismatch')
    policy.eval().requires_grad_(False)
    if cpu_embedding:
        cpu_embedding_for_validation(policy)
    gc.collect()
    torch.cuda.empty_cache()
    return policy, meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda',
                        help='Export device only; inference still requires CUDA SM89+.')
    parser.add_argument('--threads', type=int, default=2, help='CPU thread budget during export')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    export(args.source,args.output,args.device)
