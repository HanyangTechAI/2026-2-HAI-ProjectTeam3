#!/usr/bin/env python3
"""Explicit, revision-pinned model downloads to external storage.

No download runs during image build or boot. Existing different files are never
overwritten. Hugging Face authentication uses its standard runtime environment;
the tool never prints or writes tokens. Dataset and trained team checkpoints are
not downloaded by this helper.
"""

import argparse
import fnmatch
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile


DEFAULT_MANIFEST = Path('/opt/hai/config/models.json')


def relative_path(value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or '..' in path.parts or '\\' in value:
        raise ValueError(f'Unsafe relative asset path: {value!r}')
    return Path(*path.parts)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def install_file(source, destination):
    """Atomically link/copy a cache file without replacing a user's asset."""
    source = source.resolve(strict=True)
    digest = sha256(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(destination):
        if destination.is_file() and sha256(destination) == digest:
            return digest
        raise FileExistsError(f'Existing asset differs; preserved: {destination}')
    try:
        os.link(source, destination)
    except FileExistsError:
        if destination.is_file() and sha256(destination) == digest:
            return digest
        raise
    except OSError:
        # A cache on a different filesystem needs a copy. Publish the completed
        # temporary file with link(), which also refuses a concurrent overwrite.
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix='.hai-model-', delete=False) as handle:
            staged = Path(handle.name)
        try:
            shutil.copyfile(source, staged)
            staged.chmod(0o644)
            os.link(staged, destination)
        finally:
            staged.unlink(missing_ok=True)
    return digest


def install_json(document, destination):
    encoded = json.dumps(document, ensure_ascii=False, indent=2) + '\n'
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='hai-model-config-') as temporary:
        source = Path(temporary) / 'config.json'
        source.write_text(encoded)
        install_file(source, destination)


def download_group(group, root, snapshot_download):
    revision = group['revision']
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError(f'Expected immutable model revision for {group["repo_id"]}')
    target = root / relative_path(group['target'])
    prefix = group.get('strip_prefix')
    prefix = relative_path(prefix) if prefix else Path()
    patterns = group['patterns']
    snapshot = Path(snapshot_download(repo_id=group['repo_id'], revision=revision,
                                      allow_patterns=patterns))
    selected = []
    for source in sorted(snapshot.rglob('*')):
        if not source.is_file():
            continue
        relative = source.relative_to(snapshot)
        if not any(fnmatch.fnmatchcase(relative.as_posix(), pattern) for pattern in patterns):
            continue
        relative = relative.relative_to(prefix)
        destination = target / relative
        digest = install_file(source, destination)
        selected.append({'path': str(destination.relative_to(root)),
                         'sha256': digest, 'bytes': source.stat().st_size})
    if not selected:
        raise RuntimeError(f'No requested model files found: {group["repo_id"]}@{revision}')
    return {'repo_id': group['repo_id'], 'revision': revision, 'files': selected}


def faceid_config(root):
    generation = root / 'generation'
    return {
        'base_model': str(generation / 'realistic-vision-v5.1'),
        'image_encoder': str(generation / 'clip-vit-h-14'),
        'ip_adapter': str(generation / 'ip-adapter-faceid-plusv2_sd15.bin'),
        'faceid_lora': str(generation / 'ip-adapter-faceid-plusv2_sd15_lora.safetensors'),
        'age_lora': str(generation / 'age_lora.safetensors'),
        'model_root': str(root / 'insightface'),
        'device': 'cuda',
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'download'))
    parser.add_argument('profile', choices=('baseline', 'm1'))
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--models-root', type=Path)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    if manifest.get('schema_version') != 1:
        raise ValueError('Unsupported model manifest schema')
    groups = manifest['profiles'][args.profile]
    root = (args.models_root or Path(os.environ.get('HAI_WORKSPACE', '/workspace/hai')) / 'shared/models').resolve()
    if args.action == 'plan':
        print(json.dumps({'profile': args.profile, 'models_root': str(root), 'downloads': groups}, indent=2))
        return 0
    workspace = Path(os.environ.get('HAI_WORKSPACE', '/workspace/hai'))
    os.environ.setdefault('HF_HOME', str(workspace / 'cache/huggingface'))
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError('Use the installed hai-models command with the image Python environment') from error
    records = []
    for group in groups:
        print(f'Downloading {group["name"]}: {group["repo_id"]}@{group["revision"]}', flush=True)
        records.append(download_group(group, root, snapshot_download))
    install_json({'schema_version': 1, 'profile': args.profile, 'sources': records},
                 root / '_hai_assets' / f'{args.profile}.json')
    if args.profile == 'baseline':
        config = root / 'generation/hai-faceid.json'
        install_json(faceid_config(root), config)
        print(f'FaceID configuration: {config}')
        print('Prepare InsightFace separately: hai run main -- python -m age.setup_models --with-arcface')
        print('Age LoRA and the trained age/SAM checkpoints must be transferred from your existing work.')
    else:
        print('FLUX model access must be granted to your Hugging Face account before downloading gated files.')
        print('This prepares weights only; custom Age Adapter training and GPU memory use remain separate checks.')
    print(f'Prepared profile {args.profile} in {root}')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f'hai-models: {error}', file=sys.stderr)
        raise SystemExit(1)
