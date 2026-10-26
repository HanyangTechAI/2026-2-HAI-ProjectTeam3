"""Audited, single-GPU runner for the pinned official Diffusers 0.35.1 LoRA example.

Only --prepare-script uses the network, for Python source, never data or weights.
Training is offline and consumes only an explicitly selected local train split.
"""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time
import traceback
import types
from urllib.request import urlopen

from age.common import save_json, sha256, write_csv

SCRIPT_URL = ('https://raw.githubusercontent.com/huggingface/diffusers/v0.35.1/'
              'examples/text_to_image/train_text_to_image_lora.py')
SCRIPT_SHA256 = '3dcc88d582bb9f20d4e72278f0a76749a1190ed43a6521b2cdb297efa512136a'
# Upstream consumes this filter in AdamW, leaving gradient clipping an empty iterator.
# CPU serialization also avoids upcasting the entire frozen UNet on an 8GB GPU.
SOURCE_PATCHES = {
    'lora_layers = filter(lambda p: p.requires_grad, unet.parameters())':
        'lora_layers = [p for p in unet.parameters() if p.requires_grad]',
    'unet = unet.to(torch.float32)':
        'unet = unet.to(device="cpu", dtype=torch.float32)',
}
ALLOWED = set(('pretrained_model_name_or_path train_data_dir image_column caption_column resolution '
               'center_crop random_flip rank train_batch_size gradient_accumulation_steps '
               'gradient_checkpointing mixed_precision learning_rate lr_scheduler lr_warmup_steps '
               'adam_beta1 adam_beta2 adam_weight_decay adam_epsilon max_grad_norm max_train_steps '
               'checkpointing_steps checkpoints_total_limit dataloader_num_workers report_to seed '
               'enable_xformers_memory_efficient_attention use_8bit_adam').split())


def patched_source(source):
    if hashlib.sha256(source).hexdigest() != SCRIPT_SHA256:
        raise ValueError('Official script SHA-256 mismatch; refusing to execute changed source')
    text = source.decode('utf-8')
    for before, after in SOURCE_PATCHES.items():
        if text.count(before) != 1:
            raise ValueError(f'Expected exactly one upstream patch site: {before}')
        text = text.replace(before, after)
    return text


def training_arguments(config, output, resume=None):
    unknown = set(config) - ALLOWED
    if unknown:
        raise ValueError(f'Unsupported options (no Hub/dataset downloads): {sorted(unknown)}')
    base, train = Path(config['pretrained_model_name_or_path']), Path(config['train_data_dir'])
    if not (base / 'model_index.json').is_file():
        raise ValueError('A complete local Realistic Vision Diffusers model is required')
    if train.name != 'train' or not (train / 'images').is_dir() or not (train / 'metadata.jsonl').is_file():
        raise ValueError('train_data_dir must be a train split with images/ and metadata.jsonl, not its parent')
    if config.get('report_to') != 'none':
        raise ValueError('Use report_to=none; this runner writes local JSONL metrics')
    for key in ('resolution', 'rank', 'train_batch_size', 'gradient_accumulation_steps',
                'max_train_steps', 'checkpointing_steps', 'checkpoints_total_limit'):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('learning_rate', 'max_grad_norm'):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be finite and positive')
    args = []
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                args.append('--' + key)
        else:
            args.extend(['--' + key, str(value)])
    args.extend(['--output_dir', str(output), '--cache_dir', str(Path(output) / 'cache')])
    if resume:
        checkpoint = Path(output) / resume
        if Path(resume).name != resume or not resume.startswith('checkpoint-') or not checkpoint.is_dir():
            raise ValueError('Resume must name an existing checkpoint-N inside output')
        step = int(resume.removeprefix('checkpoint-'))
        if config['max_train_steps'] <= step:
            raise ValueError('max_train_steps must exceed the checkpoint step')
        args.extend(['--resume_from_checkpoint', resume])
    return args


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.log.flush()


def run_training(config, output, script, resume=None):
    argv = training_arguments(config, output, resume)
    source = patched_source(script.read_bytes())
    if resume:
        previous = json.loads((output / 'config.json').read_text())
        if ({k: v for k, v in config.items() if k != 'max_train_steps'} !=
                {k: v for k, v in previous.items() if k != 'max_train_steps'}):
            raise ValueError('Resume may only change max_train_steps; keep the same model/data/training settings')
    else:
        output.mkdir(parents=True, exist_ok=False)
        save_json(output / 'config.json', config)
    run = output / ('resume-' + str(config['max_train_steps']) if resume else 'initial')
    run.mkdir(exist_ok=False)
    save_json(run / 'command.json', dict(wrapper=[sys.executable, '-m', 'gen.train_lora', *sys.argv[1:]], official_argv=argv,
                                       script=str(script), url=SCRIPT_URL, sha256=SCRIPT_SHA256,
                                       in_memory_patches=SOURCE_PATCHES, effective_config=config,
                                       tracker_override='CLI report_to=none maps to Accelerator(log_with=None)'))
    packages = sorted(f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions())
    (run / 'packages.txt').write_text('\n'.join(packages) + '\n')
    for key in ('HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
        os.environ[key] = '1'
    report = dict(status='running', phase='preflight', is_mock=False, applied_steps=0, completed_global_step=None,
                  resume_requested=resume, checkpoint_reloaded=False, lora_reloaded=False,
                  training_succeeded=False, quality_validated=False,
                  started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    samples, stop, sampler = [], threading.Event(), None
    started = time.perf_counter()

    def sample_gpu():
        while not stop.is_set():
            try:
                result = subprocess.run(['nvidia-smi', '--id=0', '--query-gpu=memory.used,memory.total',
                                         '--format=csv,noheader,nounits'],
                                        capture_output=True, text=True, check=True, timeout=5)
                used, total = map(int, result.stdout.strip().split(','))
                samples.append(dict(seconds=time.perf_counter() - started, used_mib=used, total_mib=total))
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                report['gpu_sampling_error'] = str(exc)
                return
            stop.wait(.5)

    with (run / 'train.log').open('w') as log, redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
        torch = None
        old_argv = sys.argv
        try:
            import torch
            from accelerate import Accelerator
            import datasets
            from diffusers import StableDiffusionPipeline
            from diffusers.utils import convert_state_dict_to_diffusers
            from peft.utils import get_peft_model_state_dict
            from safetensors.torch import load_file

            if importlib.metadata.version('diffusers') != '0.35.1':
                raise RuntimeError('This audited runner requires diffusers==0.35.1')
            if not torch.cuda.is_available() or int(os.environ.get('WORLD_SIZE', '1')) != 1:
                raise RuntimeError('A single CUDA process is required; no CPU/mock fallback')
            torch.cuda.init()
            torch.cuda.reset_peak_memory_stats()
            report.update(gpu=torch.cuda.get_device_name(0), cuda_runtime=torch.version.cuda,
                          initial_free_mib=torch.cuda.mem_get_info()[0] / 2**20,
                          initial_total_mib=torch.cuda.mem_get_info()[1] / 2**20,
                          xformers_available=importlib.util.find_spec('xformers') is not None,
                          sdpa_available=hasattr(torch.nn.functional, 'scaled_dot_product_attention'))
            check = torch.ones((8, 8), device='cuda')
            if not torch.isfinite(check @ check).all():
                raise RuntimeError('CUDA matrix multiplication failed')
            del check
            sampler = threading.Thread(target=sample_gpu, daemon=True)
            sampler.start()
            # Fingerprint the same local components used for inference, without copying weights.
            report['base_files'] = {str(p.relative_to(config['pretrained_model_name_or_path'])): sha256(p)
                                    for p in sorted(Path(config['pretrained_model_name_or_path']).rglob('*'))
                                    if p.is_file() and '.cache' not in p.parts}
            report['metadata_sha256'] = sha256(Path(config['train_data_dir']) / 'metadata.jsonl')

            class ObservedAccelerator(Accelerator):
                def __init__(self, *args, **kwargs):
                    # The official CLI accepts strings; Accelerate requires Python None.
                    if kwargs.get('log_with') == 'none':
                        kwargs['log_with'] = None
                    super().__init__(*args, **kwargs)
                    if self.device.type != 'cuda' or self.num_processes != 1:
                        raise RuntimeError('Accelerator must select one CUDA device, not CPU/distributed execution')

                def prepare(self, *objects, **kwargs):
                    unet, optimizer, loader, scheduler = super().prepare(*objects, **kwargs)
                    self.parameters_to_audit = {n: p for n, p in unet.named_parameters() if p.requires_grad}
                    if not self.parameters_to_audit or any('lora_' not in n for n in self.parameters_to_audit):
                        raise RuntimeError('Only UNet LoRA parameters may be trained')
                    if any(p.device.type != 'cuda' for p in self.parameters_to_audit.values()):
                        raise RuntimeError('All trainable LoRA tensors must actually reside on CUDA')
                    self.initial_weights = {n: p.detach().cpu().clone() for n, p in self.parameters_to_audit.items()}
                    batch = next(iter(loader))
                    if batch['pixel_values'].shape[1:] != (3, config['resolution'], config['resolution']):
                        raise RuntimeError('Unexpected dataloader image shape')
                    if not torch.isfinite(batch['pixel_values']).all():
                        raise RuntimeError('Nonfinite dataloader pixels')
                    report.update(trainable_parameters=sum(p.numel() for p in self.parameters_to_audit.values()),
                                  training_device=str(self.device),
                                  trainable_dtypes=sorted({str(p.dtype) for p in self.parameters_to_audit.values()}),
                                  attention_processors=sorted({type(p).__name__ for p in unet.attn_processors.values()}),
                                  gradient_checkpointing=unet.is_gradient_checkpointing,
                                  dataloader=dict(rows=len(loader.dataset), pixel_shape=list(batch['pixel_values'].shape),
                                                  token_shape=list(batch['input_ids'].shape), finite=True))
                    return unet, optimizer, loader, scheduler

                def load_state(self, *args, **kwargs):
                    result = super().load_state(*args, **kwargs)
                    self.initial_weights = {n: p.detach().cpu().clone() for n, p in self.parameters_to_audit.items()}
                    report['checkpoint_reloaded'] = True
                    return result

                def backward(self, loss, **kwargs):
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite training loss')
                    return super().backward(loss, **kwargs)

                def clip_grad_norm_(self, parameters, max_norm, norm_type=2):
                    parameters = list(parameters)
                    if not parameters or not any(p.grad is not None for p in parameters):
                        raise RuntimeError('No LoRA gradients reached gradient clipping')
                    norm = super().clip_grad_norm_(parameters, max_norm, norm_type)
                    self.last_grad_norm = float(norm)
                    if not math.isfinite(self.last_grad_norm) or self.last_grad_norm <= 0:
                        raise RuntimeError(f'Nonfinite or zero gradient norm: {self.last_grad_norm}')
                    return norm

                def log(self, values, step=None, **kwargs):
                    if 'train_loss' in values:
                        skipped = self.optimizer_step_was_skipped
                        delta = max(float((p.detach().cpu() - self.initial_weights[n]).abs().max())
                                    for n, p in self.parameters_to_audit.items())
                        row = dict(step=step, loss=values['train_loss'], grad_norm=self.last_grad_norm,
                                   optimizer_step_skipped=skipped, lora_max_abs_delta=delta,
                                   peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                                   peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
                        with (run / 'metrics.jsonl').open('a') as stream:
                            stream.write(json.dumps(row, allow_nan=False) + '\n')
                        print('AUDIT ' + json.dumps(row), flush=True)
                        report['completed_global_step'] = step
                        report['applied_steps'] += int(not skipped)
                        report['lora_max_abs_delta'] = delta
                        if skipped or delta <= 0 or not math.isfinite(values['train_loss']):
                            raise RuntimeError('Optimizer did not apply a finite LoRA update')
                    return super().log(values, step=step, **kwargs)

            module = types.ModuleType('official_lora_training')
            module.__file__ = str(script)
            exec(compile(source, str(script), 'exec'), module.__dict__)
            module.Accelerator = ObservedAccelerator
            original_load = module.load_dataset

            def local_dataset(*args, **kwargs):
                dataset = original_load(*args, **kwargs)
                paths = dataset['train'].cast_column('image', datasets.Image(decode=False))['image']
                root = (Path(config['train_data_dir']) / 'images').resolve()
                if any(not Path(row['path']).resolve().is_relative_to(root) for row in paths):
                    raise RuntimeError('ImageFolder includes files outside train/images')
                with (root.parent / 'metadata.jsonl').open(encoding='utf-8') as stream:
                    expected = sum(1 for _ in stream)
                if len(dataset['train']) != expected:
                    raise RuntimeError('ImageFolder count differs from training metadata')
                report['imagefolder'] = dict(rows=len(dataset['train']), root=str(root), train_only=True)
                return dataset

            module.load_dataset = local_dataset
            sys.argv = [str(script), *argv]
            print('Official arguments: ' + shlex.join(sys.argv), flush=True)
            report['phase'] = 'training'
            module.main()
            if report['completed_global_step'] != config['max_train_steps']:
                raise RuntimeError('Training stopped before requested global step')
            report.update(training_succeeded=True, phase='lora_reload')
            gc.collect()
            torch.cuda.empty_cache()
            checkpoint = output / 'pytorch_lora_weights.safetensors'
            report.update(checkpoint=str(checkpoint.resolve()), checkpoint_sha256=sha256(checkpoint),
                          checkpoints=sorted(str(p.resolve()) for p in output.glob('checkpoint-*')))
            saved = load_file(str(checkpoint))
            if not saved or not all(torch.isfinite(value).all() for value in saved.values()):
                raise RuntimeError('Saved LoRA contains missing/nonfinite tensors')
            # Reload into a fresh base on CPU and compare every LoRA tensor exactly.
            pipe = StableDiffusionPipeline.from_pretrained(config['pretrained_model_name_or_path'],
                                                           torch_dtype=torch.float32, local_files_only=True)
            pipe.load_lora_weights(str(output), weight_name=checkpoint.name, adapter_name='age', local_files_only=True)
            loaded = convert_state_dict_to_diffusers(get_peft_model_state_dict(pipe.unet, adapter_name='age'))
            loaded = {'unet.' + key: value.cpu() for key, value in loaded.items()}
            if saved.keys() != loaded.keys() or any(not torch.equal(value, loaded[key]) for key, value in saved.items()):
                raise RuntimeError('Reloaded LoRA tensor mismatch')
            report.update(status='succeeded', phase='complete', lora_reloaded=True, lora_tensor_count=len(saved))
            del pipe
        except BaseException as exc:
            report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
            traceback.print_exc()
            raise
        finally:
            sys.argv = old_argv
            stop.set()
            if sampler is not None:
                sampler.join(timeout=6)
            if torch is not None and torch.cuda.is_available():
                report.update(peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                              peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
            report.update(elapsed_seconds=time.perf_counter() - started,
                          device_peak_sampled_mib=max((s['used_mib'] for s in samples), default=None),
                          memory_note='CUDA allocator peak includes training/save; nvidia-smi is sampled every 0.5s and includes other applications.')
            write_csv(run / 'gpu.csv', samples, fields=['seconds', 'used_mib', 'total_mib'])
            save_json(run / 'summary.json', report)
            print(json.dumps({k: v for k, v in report.items() if k not in ('base_files', 'traceback')}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-script', action='store_true', help='Fetch only the hash-pinned official Python example, then exit')
    parser.add_argument('--script', type=Path, default=Path('outputs/age_lora_tools/train_text_to_image_lora.py'))
    parser.add_argument('--config', type=Path, default=Path('gen/lora.smoke.json'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume-from-checkpoint')
    parser.add_argument('--max-train-steps', type=int)
    args = parser.parse_args()
    if args.prepare_script:
        if args.script.exists():
            patched_source(args.script.read_bytes())
        else:
            with urlopen(SCRIPT_URL, timeout=60) as response:
                source = response.read()
            patched_source(source)
            args.script.parent.mkdir(parents=True, exist_ok=True)
            args.script.write_bytes(source)
        print(f'Official source ready: {args.script} SHA-256={SCRIPT_SHA256}')
        return
    if args.output is None:
        parser.error('--output is required for training')
    config = json.loads(args.config.read_text())
    if args.max_train_steps is not None:
        config['max_train_steps'] = args.max_train_steps
    run_training(config, args.output, args.script, args.resume_from_checkpoint)


if __name__ == '__main__':
    main()
