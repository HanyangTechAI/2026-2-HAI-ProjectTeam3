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
import tempfile
import threading
import time
import traceback
import types
from urllib.request import urlopen

from age.common import save_json, sha256, write_csv

SCRIPT_URL = ('https://raw.githubusercontent.com/huggingface/diffusers/v0.35.1/'
              'examples/text_to_image/train_text_to_image_lora.py')
SCRIPT_SHA256 = '3dcc88d582bb9f20d4e72278f0a76749a1190ed43a6521b2cdb297efa512136a'
PROVENANCE_SCHEMA = 1
CHECKPOINT_FILES = {'model.safetensors', 'optimizer.bin', 'scheduler.bin',
                    'random_states_0.pkl', 'pytorch_lora_weights.safetensors'}
# Upstream consumes this filter in AdamW, leaving gradient clipping an empty iterator.
# CPU serialization also avoids upcasting the entire frozen UNet on an 8GB GPU.
SOURCE_PATCHES = {
    'lora_layers = filter(lambda p: p.requires_grad, unet.parameters())':
        'lora_layers = [p for p in unet.parameters() if p.requires_grad]',
    'unet = unet.to(torch.float32)':
        'unet = unet.to(device="cpu", dtype=torch.float32)',
    'for step, batch in enumerate(train_dataloader):':
        'for step, batch in enumerate(accelerator.epoch_loader(\n'
        '            train_dataloader, epoch, first_epoch, initial_global_step, num_update_steps_per_epoch)):',
    'logger.info(f"Saved state to {save_path}")':
        'save_path = accelerator.audit_checkpoint(save_path)\n'
        '                        logger.info(f"Saved state to {save_path}")',
    'num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes':
        'num_training_steps_for_scheduler = accelerator.scheduler_total_steps * accelerator.num_processes',
    'save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")':
        'save_path = accelerator.checkpoint_staging_path(global_step)',
}
ALLOWED = set(('pretrained_model_name_or_path train_data_dir image_column caption_column resolution '
               'center_crop random_flip rank train_batch_size gradient_accumulation_steps '
               'gradient_checkpointing mixed_precision learning_rate lr_scheduler lr_warmup_steps '
               'adam_beta1 adam_beta2 adam_weight_decay adam_epsilon max_grad_norm max_train_steps '
               'checkpointing_steps checkpoints_total_limit dataloader_num_workers report_to seed '
               'enable_xformers_memory_efficient_attention use_8bit_adam scheduler_total_steps').split())


def patched_source(source):
    if hashlib.sha256(source).hexdigest() != SCRIPT_SHA256:
        raise ValueError('Official script SHA-256 mismatch; refusing to execute changed source')
    text = source.decode('utf-8')
    for before, after in SOURCE_PATCHES.items():
        if text.count(before) != 1:
            raise ValueError(f'Expected exactly one upstream patch site: {before}')
        text = text.replace(before, after)
    return text


def epoch_loader(accelerator, loader, epoch, first_epoch, initial_step, updates_per_epoch):
    """Restore sample order without replaying already consumed microbatches."""
    skip = (initial_step % updates_per_epoch) * accelerator.gradient_accumulation_steps if epoch == first_epoch else 0
    active = accelerator.skip_first_batches(loader, skip) if skip else loader
    active.set_epoch(epoch)
    return active, skip


def verify_checkpoint(directory, *, allow_legacy=False, required_files=CHECKPOINT_FILES, expected_step=None,
                      require_provenance=False):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError('Checkpoint directory is missing')
    manifest = directory / 'integrity.json'
    if not manifest.exists() and (not allow_legacy or require_provenance):
        raise ValueError('Checkpoint integrity manifest is required; interrupted checkpoints cannot resume')
    for name in required_files:
        path = directory / name
        if not path.is_file() or path.is_symlink() or not path.stat().st_size:
            raise ValueError(f'Incomplete checkpoint state: {name}')
    if not manifest.exists():
        return None
    if manifest.is_symlink():
        raise ValueError('Checkpoint integrity manifest must be local')
    record = json.loads(manifest.read_text())
    if not isinstance(record, dict) or not isinstance(record.get('files'), dict) or not record['files']:
        raise ValueError('Checkpoint integrity files must be a nonempty object')
    actual = {p.name for p in directory.iterdir() if p.name != 'integrity.json'}
    if set(record['files']) != actual or not set(required_files).issubset(record['files']):
        raise ValueError('Checkpoint integrity file inventory mismatch')
    for name, digest in record['files'].items():
        path = directory / name
        if (Path(name).name != name or path.is_symlink() or not path.is_file() or
                not path.stat().st_size or sha256(path) != digest):
            raise ValueError(f'Checkpoint integrity mismatch: {name}')
    if expected_step is None and directory.name.startswith('checkpoint-'):
        expected_step = int(directory.name.removeprefix('checkpoint-'))
    if expected_step is not None:
        if type(record.get('step')) is not int or record['step'] != expected_step:
            raise ValueError('Checkpoint integrity step mismatch')
        digests = [record.get('metadata_sha256')]
        if not isinstance(record.get('base_files'), dict) or not record['base_files']:
            raise ValueError('Checkpoint model fingerprints are missing')
        digests.extend(record['base_files'].values())
        if any(not isinstance(value, str) or len(value) != 64 or
               any(c not in '0123456789abcdef' for c in value) for value in digests):
            raise ValueError('Checkpoint model/data fingerprints are invalid')
    if require_provenance or {'provenance_schema', 'finalized', 'train_images_sha256'} & record.keys():
        digest = record.get('train_images_sha256')
        if (type(record.get('provenance_schema')) is not int or record['provenance_schema'] != PROVENANCE_SCHEMA or
                record.get('finalized') is not True or
                not isinstance(digest, str) or len(digest) != 64 or
                any(c not in '0123456789abcdef' for c in digest)):
            raise ValueError('Checkpoint finalized provenance is missing or invalid')
    return record


def publish_checkpoint(directory, destination, record, required_files):
    """Keep interrupted saves hidden and publish only a verified, finalized checkpoint."""
    directory, destination = Path(directory), Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f'Refusing to overwrite checkpoint: {destination}')
    for name in required_files:
        path = directory / name
        if not path.is_file() or path.is_symlink() or not path.stat().st_size:
            raise ValueError(f'Incomplete checkpoint state: {name}')
    record = record | dict(provenance_schema=PROVENANCE_SCHEMA, finalized=True,
                           files={p.name: sha256(p) for p in sorted(directory.iterdir()) if p.is_file()})
    with (directory / 'integrity.json').open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    verify_checkpoint(directory, required_files=required_files, expected_step=record['step'], require_provenance=True)
    directory.rename(destination)


def training_data_fingerprint(directory):
    """Bind metadata order and image bytes, opening only local train/images references."""
    train = Path(directory).resolve()
    metadata = train / 'metadata.jsonl'
    if train.name != 'train' or metadata.is_symlink():
        raise ValueError('Training provenance requires a local train split and metadata')
    content = metadata.read_bytes()
    images = []
    for line in content.splitlines():
        row = json.loads(line)
        name = row.get('file_name') if isinstance(row, dict) else None
        if (not isinstance(name, str) or not name or Path(name).is_absolute() or
                '..' in Path(name).parts or '\\' in name):
            raise ValueError('Invalid training image reference')
        path = (train / name).resolve()
        if not path.is_relative_to(train / 'images') or not path.is_file():
            raise ValueError(f'Training image must exist inside train/images: {name}')
        images.append(dict(file_name=name, sha256=sha256(path)))
    if not images:
        raise ValueError('Training metadata must contain image references')
    digest = hashlib.sha256(json.dumps(images, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return dict(metadata_sha256=hashlib.sha256(content).hexdigest(), images=images, images_sha256=digest)


def verify_imagefolder(dataset, directory, provenance, image_column='image', caption_column='text'):
    from datasets import Image
    train = Path(directory).resolve()
    paths = dataset.cast_column(image_column, Image(decode=False))[image_column]
    expected = [(train / row['file_name']).resolve() for row in provenance['images']]
    if (len(paths) != len(expected) or
            any(row.get('bytes') is not None or not row.get('path') or
                Path(row['path']).resolve() != path for row, path in zip(paths, expected))):
        raise RuntimeError('ImageFolder image references/order differ from training metadata')
    metadata = (train / 'metadata.jsonl').read_bytes()
    if (hashlib.sha256(metadata).hexdigest() != provenance['metadata_sha256'] or
            list(dataset[caption_column]) != [json.loads(line)[caption_column] for line in metadata.splitlines()]):
        raise RuntimeError('ImageFolder captions/order differ from training metadata')
    return dict(rows=len(paths), root=str(train / 'images'), train_only=True, metadata_order_verified=True,
                caption_order_verified=True, ordered_images_sha256=provenance['images_sha256'])


def create_phase_directory(output, max_steps, resume):
    name = f'resume-{max_steps}' if resume else 'initial'
    attempt = 1
    while True:
        directory = Path(output) / (name if attempt == 1 else f'{name}-attempt-{attempt}')
        try:
            directory.mkdir(exist_ok=False)
            return directory
        except FileExistsError:
            if not resume:
                raise
            attempt += 1


def validate_optimizer_update(loss, grad_norm, learning_rates, step_delta, skipped, *, expected_zero_lr=False):
    """Validate an executed optimizer step, not the scheduler's next learning rate."""
    if skipped:
        raise RuntimeError('AMP skipped the optimizer step')
    if (not learning_rates or
            not all(math.isfinite(value) for value in (loss, grad_norm, step_delta, *learning_rates)) or
            grad_norm <= 0 or step_delta < 0 or any(lr < 0 for lr in learning_rates)):
        raise RuntimeError('Nonfinite or invalid optimizer update/gradient/learning rate')
    if all(lr == 0 for lr in learning_rates):
        if expected_zero_lr and step_delta == 0:
            return 'warmup_zero_lr'
        raise RuntimeError('Unexpected zero learning rate or weight movement at zero learning rate')
    if step_delta <= 0:
        raise RuntimeError('Optimizer did not move LoRA weights at positive learning rate')
    return 'weight_update'


def checkpoint_retention_plan(config, output, resume=None):
    """Fail before training if the official retention/save code would destroy checkpoints."""
    start = int(resume.removeprefix('checkpoint-')) if resume else 0
    cadence = config['checkpointing_steps']
    planned = [f'checkpoint-{step}' for step in
               range((start // cadence + 1) * cadence, config['max_train_steps'] + 1, cadence)]
    output = Path(output)
    existing = sorted(p.name for p in output.iterdir() if p.name.startswith('checkpoint')) if output.exists() else []
    collisions = sorted(set(existing) & set(planned))
    if collisions:
        raise ValueError(f'Refusing to overwrite existing checkpoints: {collisions}; use a new experiment output')
    required = len(existing) + len(planned)
    if planned and required > config['checkpoints_total_limit']:
        raise ValueError(
            f'Checkpoint retention would delete checkpoints: need checkpoints_total_limit >= {required}, '
            f"configured {config['checkpoints_total_limit']}. Start a new experiment with sufficient retention; "
            'resume settings cannot change. Existing outputs have not been modified.')
    return dict(existing_checkpoints=existing, planned_new_checkpoints=planned,
                required_limit=required, configured_limit=config['checkpoints_total_limit'])


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
    if 'scheduler_total_steps' in config:
        horizon = config['scheduler_total_steps']
        if type(horizon) is not int or horizon < 1:
            raise ValueError('scheduler_total_steps must be a positive integer')
        if config['max_train_steps'] > horizon:
            raise ValueError('max_train_steps must not exceed fixed scheduler_total_steps')
    args = []
    for key, value in config.items():
        if key == 'scheduler_total_steps':
            continue  # Wrapper-only field; the hash-checked source patch supplies it.
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
    retention = checkpoint_retention_plan(config, output, resume)
    training_data = training_data_fingerprint(config['train_data_dir'])
    required_files = CHECKPOINT_FILES | ({'scaler.pt'} if config.get('mixed_precision') == 'fp16' else set())
    integrity, legacy_images = None, False
    if resume:
        previous_command = json.loads((output / 'initial/command.json').read_text())
        schema = previous_command.get('provenance_schema')
        if {'provenance_schema', 'checkpoint_integrity_required', 'training_data'} & previous_command.keys():
            if (type(schema) is not int or schema != PROVENANCE_SCHEMA or
                    previous_command.get('checkpoint_integrity_required') is not True):
                raise ValueError('Unsupported or incomplete training provenance')
            if previous_command.get('training_data') != training_data:
                raise ValueError('Training metadata/image bytes/order differ from original run')
        else:
            legacy_images = True  # Historical runs did not bind image bytes/order before training.
        # Only the documented pre-audit smoke checkpoints may lack integrity, not later legacy runs.
        smoke_patches = {key: SOURCE_PATCHES[key] for key in (
            'lora_layers = filter(lambda p: p.requires_grad, unet.parameters())', 'unet = unet.to(torch.float32)')}
        step = int(resume.removeprefix('checkpoint-'))
        allow_legacy = (schema is None and previous_command.get('sha256') == SCRIPT_SHA256 and
                        previous_command.get('in_memory_patches') == smoke_patches and
                        step <= previous_command.get('effective_config', {}).get('max_train_steps', 0))
        integrity = verify_checkpoint(output / resume, allow_legacy=allow_legacy, required_files=required_files,
                                      expected_step=step, require_provenance=schema is not None)
        if integrity and (integrity['metadata_sha256'] != training_data['metadata_sha256'] or
                          ('train_images_sha256' in integrity and
                           integrity['train_images_sha256'] != training_data['images_sha256'])):
            raise ValueError('Checkpoint training metadata/image bytes/order differ from current run')
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
        save_json(output / 'config.json', config)
    run = create_phase_directory(output, config['max_train_steps'], resume)
    save_json(run / 'command.json', dict(wrapper=[sys.executable, '-m', 'gen.train_lora', *sys.argv[1:]], official_argv=argv,
                                        script=str(script), url=SCRIPT_URL, sha256=SCRIPT_SHA256,
                                        in_memory_patches=SOURCE_PATCHES, effective_config=config,
                                        provenance_schema=PROVENANCE_SCHEMA, checkpoint_integrity_required=True,
                                        training_data=training_data, resume_from_checkpoint=resume, phase_directory=run.name,
                                        scheduler_total_steps=config.get('scheduler_total_steps', config['max_train_steps']),
                                        checkpoint_retention=retention,
                                        learning_rate_note='learning_rate is used by the optimizer; next_learning_rate is after scheduler.step',
                                        tracker_override='CLI report_to=none maps to Accelerator(log_with=None)'))
    packages = sorted(f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions())
    (run / 'packages.txt').write_text('\n'.join(packages) + '\n')
    for key in ('HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
        os.environ[key] = '1'
    report = dict(status='running', phase='preflight', is_mock=False, applied_steps=0, completed_global_step=None,
                  optimizer_step_attempts=0, skipped_steps=0, weight_update_steps=0, warmup_zero_lr_steps=0,
                  training_microbatches=0,
                  applied_steps_note='Non-skipped validated optimizer steps, including zero-LR warmup; weight_update_steps excludes those',
                  resume_requested=resume, checkpoint_reloaded=False, lora_reloaded=False,
                  provenance_schema=PROVENANCE_SCHEMA, checkpoint_integrity_required=True,
                  metadata_sha256=training_data['metadata_sha256'], train_images_sha256=training_data['images_sha256'],
                  legacy_training_images_unverified=legacy_images,
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
            from accelerate import Accelerator, DataLoaderConfiguration
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

            class ObservedAccelerator(Accelerator):
                def __init__(self, *args, **kwargs):
                    # The official CLI accepts strings; Accelerate requires Python None.
                    if kwargs.get('log_with') == 'none':
                        kwargs['log_with'] = None
                    kwargs['dataloader_config'] = DataLoaderConfiguration(use_seedable_sampler=True, data_seed=config['seed'])
                    super().__init__(*args, **kwargs)
                    self.last_step_at = None
                    self.scheduler_total_steps = config.get('scheduler_total_steps', config['max_train_steps'])
                    if self.device.type != 'cuda' or self.num_processes != 1:
                        raise RuntimeError('Accelerator must select one CUDA device, not CPU/distributed execution')

                def prepare(self, *objects, **kwargs):
                    unet, optimizer, loader, scheduler = super().prepare(*objects, **kwargs)
                    self.audited_scheduler = scheduler
                    self.audited_optimizer = optimizer
                    self.parameters_to_audit = {n: p for n, p in unet.named_parameters() if p.requires_grad}
                    if not self.parameters_to_audit or any('lora_' not in n for n in self.parameters_to_audit):
                        raise RuntimeError('Only UNet LoRA parameters may be trained')
                    if any(p.device.type != 'cuda' for p in self.parameters_to_audit.values()):
                        raise RuntimeError('All trainable LoRA tensors must actually reside on CUDA')
                    self.initial_weights = {n: p.detach().cpu().clone() for n, p in self.parameters_to_audit.items()}
                    initial_digest = hashlib.sha256()
                    for name, tensor in sorted(self.initial_weights.items()):
                        initial_digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode() + b'\n')
                        initial_digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
                    report['initial_lora_sha256'] = initial_digest.hexdigest()
                    report['initial_lora_note'] = 'Fresh adapter initialization before checkpoint restore or data iteration'
                    self.previous_weights = self.initial_weights
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
                    if integrity and (integrity['metadata_sha256'] != report['metadata_sha256'] or
                                      integrity['base_files'] != report['base_files']):
                        raise ValueError('Checkpoint model/data fingerprints differ from current run')
                    result = super().load_state(*args, **kwargs)
                    self.initial_weights = {n: p.detach().cpu().clone() for n, p in self.parameters_to_audit.items()}
                    self.previous_weights = self.initial_weights
                    report['checkpoint_reloaded'] = True
                    report['resume_integrity_verified'] = integrity is not None
                    report['legacy_sampler_order_unverified'] = integrity is None
                    return result

                def epoch_loader(self, loader, epoch, first_epoch, initial_step, updates_per_epoch):
                    active, skip = epoch_loader(self, loader, epoch, first_epoch, initial_step, updates_per_epoch)
                    report['resume_skipped_microbatches'] = skip
                    report['sampler_policy'] = 'SeedableRandomSampler(seed) + set_epoch + skip_first_batches'
                    if self.last_step_at is None:
                        torch.cuda.synchronize()
                        self.last_step_at = time.perf_counter()
                        report['training_started_seconds'] = self.last_step_at - started
                    return active

                def checkpoint_staging_path(self, step):
                    return tempfile.mkdtemp(prefix=f'.pending-checkpoint-{step}-', dir=output)

                def audit_checkpoint(self, directory):
                    step = report['completed_global_step']
                    publish_checkpoint(directory, output / f'checkpoint-{step}',
                                       dict(metadata_sha256=report['metadata_sha256'], base_files=report['base_files'],
                                            train_images_sha256=report['train_images_sha256'], step=step,
                                            sampler_policy=report['sampler_policy']), required_files)
                    return str(output / f'checkpoint-{step}')

                def backward(self, loss, **kwargs):
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite training loss')
                    result = super().backward(loss, **kwargs)
                    report['training_microbatches'] += 1
                    return result

                def clip_grad_norm_(self, parameters, max_norm, norm_type=2):
                    parameters = list(parameters)
                    if not parameters or not any(p.grad is not None for p in parameters):
                        raise RuntimeError('No LoRA gradients reached gradient clipping')
                    norm = super().clip_grad_norm_(parameters, max_norm, norm_type)
                    self.last_grad_norm = float(norm)
                    if not math.isfinite(self.last_grad_norm) or self.last_grad_norm <= 0:
                        raise RuntimeError(f'Nonfinite or zero gradient norm: {self.last_grad_norm}')
                    # The pinned loop clips immediately before optimizer.step, then advances the scheduler.
                    self.used_learning_rates = [float(group['lr']) for group in self.audited_optimizer.param_groups]
                    return norm

                def log(self, values, step=None, **kwargs):
                    if 'train_loss' in values:
                        torch.cuda.synchronize()
                        now = time.perf_counter()
                        skipped = self.optimizer_step_was_skipped
                        report['optimizer_step_attempts'] += 1
                        report['skipped_steps'] += int(skipped)
                        report['last_attempted_global_step'] = step
                        weights = {n: p.detach().cpu() for n, p in self.parameters_to_audit.items()}
                        if not all(torch.isfinite(p).all() for p in weights.values()):
                            raise RuntimeError('Nonfinite LoRA weights after optimizer step')
                        delta = max(float((p - self.initial_weights[n]).abs().max()) for n, p in weights.items())
                        step_delta = max(float((p - self.previous_weights[n]).abs().max()) for n, p in weights.items())
                        expected_zero_lr = (step == 1 and config.get('lr_warmup_steps', 500) > 0 and
                                            config.get('lr_scheduler') in ('constant_with_warmup', 'linear', 'cosine',
                                                                           'cosine_with_restarts', 'polynomial'))
                        update_kind = validate_optimizer_update(values['train_loss'], self.last_grad_norm,
                                                               self.used_learning_rates, step_delta, skipped,
                                                               expected_zero_lr=expected_zero_lr)
                        if not math.isfinite(delta):
                            raise RuntimeError('Nonfinite cumulative LoRA weight movement')
                        row = dict(step=step, loss=values['train_loss'], grad_norm=self.last_grad_norm,
                                     learning_rate=self.used_learning_rates[0], learning_rates=self.used_learning_rates,
                                     next_learning_rate=float(self.audited_scheduler.get_last_lr()[0]),
                                     step_seconds=now - self.last_step_at, elapsed_seconds=now - started,
                                     optimizer_step_skipped=skipped, lora_max_abs_delta=delta,
                                     lora_step_max_abs_delta=step_delta, update_kind=update_kind,
                                    allocated_mib=torch.cuda.memory_allocated() / 2**20,
                                    reserved_mib=torch.cuda.memory_reserved() / 2**20,
                                   peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                                   peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
                        with (run / 'metrics.jsonl').open('a') as stream:
                            stream.write(json.dumps(row, allow_nan=False) + '\n')
                        print('AUDIT ' + json.dumps(row), flush=True)
                        report['completed_global_step'] = step
                        report['applied_steps'] += 1
                        report['weight_update_steps'] += int(update_kind == 'weight_update')
                        report['warmup_zero_lr_steps'] += int(update_kind == 'warmup_zero_lr')
                        report['lora_max_abs_delta'] = delta
                        report['lora_step_max_abs_delta'] = step_delta
                        report['training_step_seconds'] = report.get('training_step_seconds', 0) + row['step_seconds']
                        self.last_step_at = now
                        self.previous_weights = weights
                    return super().log(values, step=step, **kwargs)

            module = types.ModuleType('official_lora_training')
            module.__file__ = str(script)
            exec(compile(source, str(script), 'exec'), module.__dict__)
            module.Accelerator = ObservedAccelerator
            original_load = module.load_dataset

            def local_dataset(*args, **kwargs):
                dataset = original_load(*args, **kwargs)
                report['imagefolder'] = verify_imagefolder(dataset['train'], config['train_data_dir'], training_data,
                                                          config.get('image_column', 'image'),
                                                          config.get('caption_column', 'text'))
                return dataset

            module.load_dataset = local_dataset
            sys.argv = [str(script), *argv]
            print('Official arguments: ' + shlex.join(sys.argv), flush=True)
            report['phase'] = 'training'
            module.main()
            if report['completed_global_step'] != config['max_train_steps']:
                raise RuntimeError('Training stopped before requested global step')
            if not report['weight_update_steps'] or report['lora_max_abs_delta'] <= 0:
                raise RuntimeError('Training completed without verified LoRA weight movement')
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
            report['step_timing_note'] = ('CUDA-synchronized consecutive update intervals include data loading, audit, '
                                          'and any preceding checkpoint save; startup and final save/reload are separate.')
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
