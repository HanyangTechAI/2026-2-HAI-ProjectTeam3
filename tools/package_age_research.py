"""Build the approved public research snapshot without modifying local artifacts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


REPLACEMENTS = (
    ('/workspace/hai/branches/main/', ''),
    ('/workspace/hai/shared/data', 'data'),
    ('/workspace/hai/shared/models', 'models'),
    ('/workspace/hai/outputs', 'outputs'),
)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def public_metadata(value):
    if isinstance(value, dict):
        return {key: public_metadata(item) for key, item in value.items() if key != 'split_rows'}
    if isinstance(value, list):
        return [public_metadata(item) for item in value]
    if isinstance(value, tuple):
        return tuple(public_metadata(item) for item in value)
    if isinstance(value, str):
        for old, new in REPLACEMENTS:
            value = value.replace(old, new)
    return value


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def public_predictions(source, destination):
    with source.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        allowed = [key for key in ('source', 'version', 'prediction') if key in reader.fieldnames]
        if 'prediction' not in allowed:
            raise ValueError(f'Unrecognized predictions schema: {source}')
        rows = [{key: row[key] for key in allowed} for row in reader]
    # Do not retain the original manifest order as an implicit sample identifier.
    rows.sort(key=lambda row: (row.get('source', ''), row.get('version', ''), float(row['prediction'])))
    with destination.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=allowed)
        writer.writeheader()
        writer.writerows(rows)


def copy_source(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source or destination.is_relative_to(source) or not (destination / '.git').exists():
        raise ValueError('Destination must be a separate Git worktree')
    names = subprocess.check_output(
        ['git', '-C', str(source), 'ls-files', '-z', '--cached', '--others', '--exclude-standard'])
    for name in names.decode().split('\0'):
        if not name:
            continue
        relative = Path(name)
        if relative.parts[0] in ('models', 'outputs', 'third_party'):
            continue
        if relative.parts[0] == 'data' and relative != Path('data/README.md'):
            continue
        path = source / relative
        if path.is_symlink():
            raise ValueError(f'Unexpected source symlink: {relative}')
        if path.is_file():
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def reference_sam(sam, destination):
    sam, destination = Path(sam).resolve(), Path(destination).resolve()
    target = destination / 'third_party' / 'SAM'
    metadata = destination / 'third_party' / 'SAM_SOURCE.json'
    # Clean only a previous generated source copy, never an initialized submodule.
    if target.exists() and any(target.iterdir()):
        if (target / '.git').exists() or not metadata.is_file():
            raise ValueError('Refusing to remove an unknown third-party directory')
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(['git', '-C', str(sam), 'rev-parse', 'HEAD']).decode().strip()
    write_json(metadata, dict(
        url='https://github.com/yuval-alaluf/SAM', revision=revision,
        note='Upstream Git submodule reference only. Mixed upstream licenses and sample images '
             'are not copied into this repository. Review upstream terms before use.'))


def build_snapshot(source, models, outputs, destination, sam):
    import torch

    source, models, outputs, destination, sam = map(
        lambda path: Path(path).resolve(), (source, models, outputs, destination, sam))
    if any((destination / name).exists() for name in ('models', 'outputs', 'third_party')):
        raise FileExistsError('Refusing to overwrite artifact directories')
    copy_source(source, destination)

    records = []

    def record(path, target, action, **extra):
        item = dict(path=str(target.relative_to(destination)), action=action,
                    original_sha256=digest(path), original_bytes=path.stat().st_size, **extra)
        if target.is_file():
            item.update(public_sha256=digest(target), public_bytes=target.stat().st_size)
        records.append(item)

    for path in sorted(models.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts:
            continue
        if path.is_symlink():
            raise ValueError(f'Unexpected models symlink: {path}')
        relative = path.relative_to(models)
        target = destination / 'models' / relative
        if path.suffix in ('.pt', '.pth', '.onnx', '.zip'):
            record(path, target, 'external_pretrained_omitted')
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == '.json':
            write_json(target, public_metadata(json.loads(path.read_text(encoding='utf-8'))))
        elif path.suffix == '.py' or path.name == 'LICENSE':
            shutil.copy2(path, target)
        else:
            raise ValueError(f'Unexpected model file: {path}')
        record(path, target, 'runtime_or_provenance')

    for path in sorted(outputs.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts:
            continue
        relative = path.relative_to(outputs)
        target = destination / 'outputs' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            if relative != Path('main/age_lora_tools/train_text_to_image_lora.py'):
                raise ValueError(f'Unexpected output symlink: {relative}')
            shutil.copy2(path.resolve(), target)
            action = 'licensed_external_script_materialized'
        elif path.suffix == '.pt':
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            checkpoint = public_metadata(checkpoint)
            checkpoint['public_export'] = dict(
                original_checkpoint_sha256=digest(path),
                removed=['split_rows'],
                note='Dataset rows removed. Not valid as proof of evaluation split independence.')
            torch.save(checkpoint, target)
            action = 'checkpoint_dataset_rows_removed_paths_rebased'
        elif path.suffix == '.csv' and 'predictions' in path.name:
            public_predictions(path, target)
            action = 'predictions_labels_ids_paths_removed_and_rows_sorted'
        elif path.suffix == '.json':
            write_json(target, public_metadata(json.loads(path.read_text(encoding='utf-8'))))
            action = 'metadata_paths_rebased'
        elif path.suffix in ('.csv', '.png'):
            shutil.copy2(path, target)
            action = 'aggregate_unmodified'
        else:
            raise ValueError(f'Unexpected output file: {relative}')
        record(path, target, action)

    reference_sam(sam, destination)
    apache = Path('/usr/share/common-licenses/Apache-2.0')
    if not apache.is_file():
        raise FileNotFoundError('Apache-2.0 license text required for the external LoRA script')
    shutil.copy2(apache, destination / 'outputs/main/age_lora_tools/LICENSE')
    write_json(destination / 'docs' / 'public-artifact-manifest-20261114.json', dict(
        policy='Approved public research snapshot: datasets excluded; external pretrained weights redownloaded.',
        note='Metrics refer to original checkpoint hashes. Public checkpoints have different hashes. '
             'Per-image output files no longer contain true labels or sample correspondence.',
        artifacts=records))
    print(json.dumps(dict(files=len(records), artifact_bytes=sum(
        record.get('public_bytes', 0) for record in records)), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--outputs', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--sam', type=Path, required=True)
    parser.add_argument('--refresh-source', action='store_true', help='Refresh source and upstream reference only')
    args = parser.parse_args()
    if args.refresh_source:
        copy_source(args.source, args.destination)
        reference_sam(args.sam, args.destination)
        return
    build_snapshot(args.source, args.models, args.outputs, args.destination, args.sam)


if __name__ == '__main__':
    main()
