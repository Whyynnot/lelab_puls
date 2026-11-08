"""Native PI05 construction and floating reference loader."""
import gc
import time
from pathlib import Path
import torch
from torch import nn
from accelerate import init_empty_weights
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from checkpoint import tensor_name_candidates

ROOT = Path(__file__).resolve().parent

def skeleton(source):
    config = PI05Config.from_pretrained(str(source))
    config.dtype = torch.bfloat16
    config.device = None
    config.compile_model = False
    config.gradient_checkpointing = False
    if config.use_visual_memory or config.use_proprioceptive_memory:
        raise ValueError('MEM checkpoints need a separate validated adapter')
    with init_empty_weights():
        policy = PI05Policy(config)
    # PI05 calls the decoder .model directly. These vocabulary-output heads are
    # never used for continuous actions and may be absent from base checkpoints.
    policy.model.paligemma_with_expert.paligemma.lm_head = nn.Identity()
    policy.model.paligemma_with_expert.gemma_expert.lm_head = nn.Identity()
    for name, buf in list(policy.named_buffers()):
        if buf.is_meta:
            raise RuntimeError(f'Unexpected meta buffer: {name}')
        parent, leaf = name.rsplit('.', 1)
        policy.get_submodule(parent)._buffers[leaf] = buf.to('cuda')
    policy.config.device = 'cuda'
    return policy

def assign_parameter(policy, name, data, dtype, device='cuda'):
    parent, leaf = name.rsplit('.', 1)
    policy.get_submodule(parent)._parameters[leaf] = nn.Parameter(
        data.to(device=device, dtype=dtype), requires_grad=False)

def cpu_embedding_for_validation(policy):
    """Keep identical floating embedding weights on CPU for paired fidelity tests.

    This is not the all-GPU deployment benchmark. Both reference and quantized
    fidelity runs use this same placement so the reference fits without INT4.
    """
    embedding=policy.model.paligemma_with_expert.paligemma.model.language_model.embed_tokens
    embedding.to('cpu')
    embedding.register_forward_pre_hook(lambda module,args:(args[0].to('cpu'),))
    embedding.register_forward_hook(lambda module,args,out:out.to('cuda'))

def load_source(source, mode='bf16', cpu_embedding=False):
    """Load the unquantized native reference, strictly and without random fallbacks."""
    if mode != 'bf16':
        raise ValueError('Floating reference loader only supports bf16')
    from checkpoint import Source
    source = Source(source)
    policy = skeleton(source.path)
    expected = {name:(tuple(p.shape),p.dtype) for name,p in policy.named_parameters()}
    consumed = set()
    started = time.perf_counter()
    for name,(shape,dtype) in expected.items():
        key = next((source.aliases.get(k,k) for k in tensor_name_candidates(name)
                    if source.aliases.get(k,k) in source.info),None)
        if key is None or tuple(source.info[key]['shape']) != shape:
            raise ValueError('Missing or mismatched native parameter: '+name)
        device = 'cpu' if cpu_embedding and name.endswith('language_model.embed_tokens.weight') else 'cuda'
        assign_parameter(policy,name,source.tensor(key),dtype,device)
        consumed.add(key)
    ignored = set(source.info)-consumed
    heads = ('paligemma_with_expert.paligemma.lm_head.','paligemma_with_expert.gemma_expert.lm_head.')
    if any(not k.removeprefix('model.').startswith(heads) for k in ignored):
        raise ValueError('Unexpected source parameters')
    if any(p.is_meta for p in policy.parameters()):
        raise RuntimeError('Uninitialized native parameters')
    policy.eval().requires_grad_(False)
    if cpu_embedding:
        cpu_embedding_for_validation(policy)
    gc.collect()
    torch.cuda.empty_cache()
    return policy,{'mode':'bf16','weights_kind':'pretrained','source':str(source.path),
                   'load_seconds':time.perf_counter()-started}
