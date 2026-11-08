"""FP8-only model service. Inputs/outputs are checkpoint-normalized tensors."""
import argparse
import io
import json
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
from fp8_runtime import ROOT, FORMAT, load_artifact
from rtc_policy import predict

MAX_BYTES = 16*1024**2


def decode(payload, policy, device='cuda'):
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if len(archive.infolist()) > 24 or sum(x.file_size for x in archive.infolist()) > MAX_BYTES:
            raise ValueError('NPZ expansion exceeds limit')
    with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
        values = {k: arrays[k] for k in arrays.files}
    options = {}
    for name, dtype, default in [('rtc_enabled', np.bool_, False), ('inference_delay', np.int64, 0),
                                 ('execution_horizon', np.int64, 0)]:
        value = values.pop(name, np.asarray(default, dtype=dtype))
        if value.shape != () or value.dtype != dtype:
            raise ValueError('Invalid RTC scalar: '+name)
        options[name] = value.item()
    previous = values.pop('prev_chunk_left_over', None)
    if previous is not None:
        if (previous.dtype != np.float32 or previous.ndim != 3 or previous.shape[0] != 1 or
            previous.shape[-1] != policy.config.output_features['action'].shape[0] or
            not np.isfinite(previous).all()):
            raise ValueError('Invalid normalized RTC previous chunk')
        if not options['rtc_enabled'] or not 0 <= options['inference_delay'] < options['execution_horizon'] <= previous.shape[1] <= policy.config.chunk_size:
            raise ValueError('RTC delay/prefix horizon is out of bounds')
        previous = torch.from_numpy(previous.copy()).to(device)
    elif options['inference_delay'] or options['execution_horizon']:
        raise ValueError('RTC prefix scalars require a previous chunk')
    images = policy.config.image_features
    allowed = set(images) | {'observation.state', 'observation.language.tokens',
                            'observation.language.attention_mask', 'noise'}
    if set(values)-allowed:
        raise ValueError('Unexpected tensor names')
    required = {'observation.state', 'observation.language.tokens', 'observation.language.attention_mask'}
    if required-set(values) or not set(images).intersection(values):
        raise ValueError('Missing policy input tensors')
    for name, feature in images.items():
        if name not in values:
            continue
        image = values[name]
        if (image.shape != (1, *feature.shape) or image.dtype != np.float32 or
            not np.isfinite(image).all() or image.min() < 0 or image.max() > 1):
            raise ValueError('Invalid RGB float32 image: '+name)
    state = values['observation.state']
    if (state.shape != (1, *policy.config.input_features['observation.state'].shape) or
        state.dtype != np.float32 or not np.isfinite(state).all()):
        raise ValueError('Invalid normalized state')
    tokens, mask = values['observation.language.tokens'], values['observation.language.attention_mask']
    if (tokens.ndim != 2 or tokens.shape[0] != 1 or not 1 <= tokens.shape[1] <= policy.config.tokenizer_max_length or
        tokens.dtype != np.int64 or tokens.min() < 0 or tokens.max() >= 257152):
        raise ValueError('Invalid token IDs')
    if mask.shape != tokens.shape or mask.dtype != np.bool_ or not mask.any():
        raise ValueError('Invalid attention mask')
    noise = values.pop('noise', None)
    if noise is not None:
        if noise.shape != (1, policy.config.chunk_size, policy.config.max_action_dim) or noise.dtype != np.float32 or not np.isfinite(noise).all():
            raise ValueError('Invalid fixed noise')
        noise = torch.from_numpy(noise.copy()).to(device)
    batch = {k: torch.from_numpy(v.copy()).to(device) for k, v in values.items()}
    return batch, noise, {'enabled': options['rtc_enabled'], 'previous': previous,
                          'delay': options['inference_delay'], 'horizon': options['execution_horizon']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact', required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--steps', type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.steps <= 50:
        raise ValueError('steps must be in [1,50]')
    torch.set_num_threads(6)
    policy, meta = load_artifact(args.artifact)
    policy.config.num_inference_steps = args.steps
    contract = {'status': 'ready', 'format': FORMAT, 'source': meta['source'],
                'backend': 'lerobot', 'precision': 'FP8 W8A8',
                'artifact': str(Path(args.artifact).resolve()), 'weights_kind': meta['weights_kind'],
                'steps': args.steps, 'chunk_size': policy.config.chunk_size,
                'max_action_dim': policy.config.max_action_dim,
                'input_images': {k: list(v.shape) for k, v in policy.config.image_features.items()},
                'state_shape': list(policy.config.input_features['observation.state'].shape),
                'action_shape': list(policy.config.output_features['action'].shape),
                'supported_modes': ['sync', 'rtc'], 'rtc': 'LeRobot guided inference-time RTC',
                'inference_worker': 'persistent_single_thread',
                'outputs': 'policy-space actions; not motor commands'}
    inference_lock = threading.Lock()
    # cuBLAS/CUDA state is thread-local. Reusing one model worker avoids
    # rebuilding that state in each short-lived HTTP request thread.
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='fp8-model')

    def infer_payload(payload):
        batch, noise, rtc = decode(payload, policy)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        actions = predict(policy, batch, noise, **rtc)
        torch.cuda.synchronize()
        elapsed = 1000*(time.perf_counter()-started)
        if not torch.isfinite(actions).all().item():
            raise RuntimeError('Non-finite model output')
        return {'actions': actions.float().cpu().tolist(), 'inference_ms': elapsed,
                'worker_thread_id': threading.get_ident(),
                'space': 'policy', 'mode': 'rtc' if rtc['enabled'] else 'sync',
                'guidance_applied': rtc['previous'] is not None,
                'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
                'peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, status, value):
            data = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.respond(200, contract) if self.path == '/health' else self.respond(404, {'error': 'not found'})

        def do_POST(self):
            if self.path != '/infer':
                self.respond(404, {'error': 'not found'}); return
            if not inference_lock.acquire(blocking=False):
                # Drain a bounded legitimate upload before closing the socket;
                # otherwise clients can see BrokenPipe instead of the 409 body.
                try:
                    rejected_length = int(self.headers.get('Content-Length', '0'))
                except ValueError:
                    rejected_length = 0
                if 0 < rejected_length <= MAX_BYTES:
                    self.rfile.read(rejected_length)
                self.respond(409, {'error': 'Inference is already active'}); return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= MAX_BYTES:
                    raise ValueError('Invalid payload size')
                result = worker.submit(infer_payload, self.rfile.read(length)).result()
                self.respond(200, result)
            except ValueError as exc:
                self.respond(400, {'error': str(exc)})
            except Exception as exc:
                torch.cuda.empty_cache()
                self.respond(500, {'error': type(exc).__name__, 'message': str(exc), 'actions': None})
            finally:
                inference_lock.release()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    print(json.dumps(contract), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        worker.shutdown(wait=True, cancel_futures=True)


if __name__ == '__main__':
    main()
