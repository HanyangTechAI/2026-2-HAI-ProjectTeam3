"""Align public train/val with a padded retry, into a new output directory."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
from PIL import Image

from .common import FIELDS, ROOT, image_path, read_csv, sha256, write_csv
from .face import FaceAligner, read_rgb

SPLITS = ('train', 'val')
AUDIT_FIELDS = ['split', 'index', 'path', 'source', 'original_id', 'age', 'status',
                'stage', 'reason', 'error_type', 'black_pixel_ratio', 'aligned_sha256', 'disposition']


class AttemptError(Exception):
    def __init__(self, stage, cause):
        super().__init__(str(cause))
        self.stage = stage
        self.cause = cause


def align_with_retry(aligner, rgb):
    try:
        aligned, _ = aligner.align(rgb, training=True)
        return aligned, 'base_success'
    except Exception as exc:
        if not isinstance(exc, ValueError) or str(exc) != 'face_not_detected':
            raise AttemptError('base', exc) from exc

    try:
        height, width = rgb.shape[:2]
        dy, dx = height // 2, width // 2
        padded = np.pad(rgb, ((dy, dy), (dx, dx), (0, 0)), mode='constant')
        face = aligner.select(padded, training=True)
        face['bbox'] = face.bbox - np.array([dx, dy, dx, dy], dtype=np.float32)
        face['kps'] = face.kps - np.array([dx, dy], dtype=np.float32)
        return aligner.crop(rgb, face), 'retry_success'
    except Exception as exc:
        raise AttemptError('retry', exc) from exc


def prepare(data_dir, output, aligner, limit_per_source=0, sources=None, device='cpu',
            model_root=None, progress_every=1000, command=None):
    if limit_per_source < 0 or progress_every < 1:
        raise ValueError('limit must be nonnegative and progress interval positive')
    data_dir, output = Path(data_dir).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(data_dir) or data_dir.is_relative_to(output):
        raise ValueError('Output must be a new directory outside the input directory')
    roots = data_dir / 'sources'
    if roots.is_dir():
        for root in roots.iterdir():
            root = root.resolve()
            if output.is_relative_to(root) or root.is_relative_to(output):
                raise ValueError(f'Output overlaps source dataset: {root}')

    inputs = {}
    for split in SPLITS:
        rows = read_csv(data_dir / (split + '.csv'))
        if any(set(FIELDS) - row.keys() for row in rows):
            raise ValueError(f'Missing required fields in {split}.csv')
        counts = Counter()
        selected = []
        for index, row in enumerate(rows):
            path = image_path(row).resolve()
            if output.is_relative_to(path.parent) or path.is_relative_to(output):
                raise ValueError(f'Output overlaps source image directory: {path}')
            if 'test' in {part.lower() for part in path.parts}:
                raise ValueError(f'Train/val manifest references a forbidden test path: {path}')
            source = row['source']
            if sources and source not in sources:
                continue
            if limit_per_source and counts[source] >= limit_per_source:
                continue
            counts[source] += 1
            selected.append((index, row))
        inputs[split] = selected

    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    audit, accepted = [], []
    for split in SPLITS:
        total = len(inputs[split])
        for processed, (index, row) in enumerate(inputs[split], 1):
            entry = dict(split=split, index=index, path=row['path'], source=row['source'],
                         original_id=row['original_id'], age=row['age'], status='', stage='',
                         reason='', error_type='', black_pixel_ratio='', aligned_sha256='', disposition='')
            try:
                age = float(row['age'])
                if not age.is_integer() or not 0 <= age <= 100:
                    raise ValueError('invalid_age')
                rgb = read_rgb(image_path(row))
                aligned, status = align_with_retry(aligner, rgb)
                if aligned.shape != (224, 224, 3) or aligned.dtype != np.uint8:
                    raise ValueError('invalid_aligned_image')
            except Exception as exc:
                cause = exc.cause if isinstance(exc, AttemptError) else exc
                entry.update(status='error', stage=exc.stage if isinstance(exc, AttemptError) else 'input_or_output',
                             reason=str(cause), error_type=type(cause).__name__, disposition='failed')
            else:
                digest = hashlib.sha256(aligned.tobytes()).hexdigest()
                destination = output / 'images' / split / f'{index:08d}.png'
                destination.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(aligned).save(destination, format='PNG')
                result = {key: row[key] for key in FIELDS}
                result.update(path=str(destination), sha256=digest)
                entry.update(status=status, black_pixel_ratio=float(np.mean(np.all(aligned == 0, axis=2))),
                             aligned_sha256=digest, disposition='accepted')
                accepted.append((split, result, entry))
            audit.append(entry)
            if processed % progress_every == 0 or processed == total:
                print(f'{split}: processed {processed}/{total}', flush=True)

    groups = defaultdict(set)
    for split, result, _ in accepted:
        groups[result['sha256']].add(split)
    collisions = {digest for digest, splits in groups.items() if len(splits) > 1}
    for _, result, entry in accepted:
        if result['sha256'] in collisions:
            entry['disposition'] = 'cross_split_excluded'
    for split in SPLITS:
        write_csv(output / f'{split}.csv', [result for name, result, entry in accepted
                                           if name == split and entry['disposition'] == 'accepted'], FIELDS)
    write_csv(output / 'audit.csv', audit, AUDIT_FIELDS)
    write_csv(output / 'failures.csv', [row for row in audit if row['status'] == 'error'], AUDIT_FIELDS)
    write_csv(output / 'cross_split_duplicates.csv',
              [row for row in audit if row['disposition'] == 'cross_split_excluded'], AUDIT_FIELDS)

    counts = {}
    for split in SPLITS:
        counts[split] = {}
        for source in sorted({row['source'] for row in audit if row['split'] == split}):
            rows = [row for row in audit if row['split'] == split and row['source'] == source]
            ratios_by_status = {}
            for status in ('base_success', 'retry_success'):
                ratios = [row['black_pixel_ratio'] for row in rows if row['status'] == status]
                ratios_by_status[status] = dict(count=len(ratios), mean=float(np.mean(ratios)),
                    median=float(np.median(ratios)), max=float(np.max(ratios))) if ratios else None
            counts[split][source] = dict(input=len(rows), statuses=dict(Counter(r['status'] for r in rows)),
                dispositions=dict(Counter(r['disposition'] for r in rows)),
                errors=dict(Counter(f"{r['stage']}:{r['error_type']}:{r['reason']}" for r in rows if r['status'] == 'error')),
                black_pixel_ratio=ratios_by_status)
    versions = {}
    for package in ('numpy', 'Pillow', 'opencv-python', 'opencv-python-headless', 'onnxruntime',
                    'onnxruntime-gpu', 'scikit-image'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    report = dict(started_at_utc=started_at, elapsed_seconds=time.perf_counter() - started,
                  command=command, input_dir=str(data_dir), output_dir=str(output), device=device,
                  limit_per_source_per_split=limit_per_source, source_filter=sorted(sources) if sources else None,
                  selection='first N per source in original manifest order; index is original row index',
                  input_csv_sha256={s: sha256(data_dir / f'{s}.csv') for s in SPLITS},
                  code_sha256={p.name: sha256(p) for p in (Path(__file__), Path(__file__).with_name('face.py'),
                                                          Path(__file__).with_name('common.py'))},
                  python=platform.python_version(), python_executable=sys.executable, packages=versions, counts=counts,
                  environment={key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                               'CUDA_VISIBLE_DEVICES', 'PYTHONPATH')},
                  black_pixel_definition='fraction of output pixels with R=G=B=0, including original black pixels',
                  padding='floor(height/2), floor(width/2) on each side; RGB black',
                  cross_split_duplicate_hashes=len(collisions),
                  note='Cross-split exact aligned-pixel duplicates excluded from both manifests; images remain on disk.')
    if model_root is not None:
        import onnxruntime
        root = Path(model_root).resolve()
        report['onnxruntime_loaded_version'] = onnxruntime.__version__
        report['model_root'] = str(root)
        report['model_sha256'] = {name: sha256(root / 'models/buffalo_l' / name)
                                  for name in ('det_10g.onnx', 'genderage.onnx')}
        report['runtime_sha256'] = {name: sha256(root / 'python_runtime' / name) for name in
                                    ('model_zoo/scrfd.py', 'model_zoo/attribute.py', 'utils/face_align.py')}
        report['providers'] = {name: getattr(aligner, name).session.get_providers()
                               for name in ('detector', 'attribute')}
    (output / 'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/age_public')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-root', type=Path, default=ROOT / 'models/insightface')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--source', action='append', help='Restrict to a source; repeat for multiple sources')
    parser.add_argument('--limit-per-source', type=int, default=0, help='First N per source per split; 0 means all')
    parser.add_argument('--progress-every', type=int, default=1000)
    args = parser.parse_args()
    if args.limit_per_source < 0 or args.progress_every < 1:
        parser.error('limit must be nonnegative and progress interval positive')
    if args.device == 'cuda':
        import torch
        import onnxruntime as ort
        if not torch.cuda.is_available() or 'CUDAExecutionProvider' not in ort.get_available_providers():
            parser.error('CUDA runtime or CUDAExecutionProvider is unavailable')
    aligner = FaceAligner(args.model_root, args.device)
    if args.device == 'cuda':
        for component in (aligner.detector, aligner.attribute):
            if 'CUDAExecutionProvider' not in component.session.get_providers():
                parser.error('CUDA session initialization failed; refusing silent CPU fallback')
    report = prepare(args.data_dir, args.output, aligner, args.limit_per_source, args.source,
                     args.device, args.model_root, args.progress_every, [sys.executable, '-m', 'age.align_public_retry', *sys.argv[1:]])
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
