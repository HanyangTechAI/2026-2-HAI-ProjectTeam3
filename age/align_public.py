"""Align existing public train/val splits into a new, immutable output directory.

No test manifest or test image is read. Repeat a run with a new output path.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from .common import FIELDS, ROOT, image_path, read_csv, sha256, write_csv
from .face import FaceAligner, read_rgb

SPLITS = ('train', 'val')


def prepare(data_dir, output, aligner, limit=0, device='cpu', model_root=None):
    if limit < 0:
        raise ValueError('limit must be nonnegative')
    data_dir, output = Path(data_dir).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(data_dir) or data_dir.is_relative_to(output):
        raise ValueError('Output must be a new directory outside the input directory')
    sources = data_dir / 'sources'
    if sources.is_dir():
        for source_root in sources.iterdir():
            source_root = source_root.resolve()
            if output.is_relative_to(source_root) or source_root.is_relative_to(output):
                raise ValueError(f'Output overlaps source dataset: {source_root}')

    inputs = {}
    for split in SPLITS:
        path = data_dir / (split + '.csv')
        rows = read_csv(path)
        if any(set(FIELDS) - row.keys() for row in rows):
            raise ValueError(f'Missing required fields in {path}')
        inputs[split] = rows[:limit or None]
        for row in inputs[split]:
            source = image_path(row).resolve()
            if source.is_relative_to(output) or output.is_relative_to(source.parent):
                raise ValueError(f'Output overlaps source image directory: {source}')

    # mkdir(exist_ok=False) makes accidental reuse, including an interrupted run, fail closed.
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    accepted, failures = [], []
    for split in SPLITS:
        for index, row in enumerate(inputs[split]):
            source = image_path(row)
            try:
                age = float(row['age'])
                if not age.is_integer() or not 0 <= age <= 100:
                    raise ValueError('invalid_age')
                aligned, _ = aligner.align(read_rgb(source), training=True)
                if aligned.shape != (224, 224, 3) or aligned.dtype != np.uint8:
                    raise ValueError('invalid_aligned_image')
            except Exception as exc:
                failures.append(dict(split=split, index=index, path=row['path'], source=row['source'],
                                     original_id=row['original_id'], reason=str(exc), error_type=type(exc).__name__))
                continue
            digest = hashlib.sha256(aligned.tobytes()).hexdigest()
            destination = output / 'images' / split / f'{index:08d}.png'
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(aligned).save(destination, format='PNG')
            result = {key: row[key] for key in FIELDS}
            result.update(path=str(destination), sha256=digest)
            accepted.append((split, index, row, result))
            if (index + 1) % 1000 == 0:
                print(f'{split}: processed {index + 1}/{len(inputs[split])}', flush=True)

    groups = defaultdict(list)
    for entry in accepted:
        groups[entry[3]['sha256']].append(entry)
    collision_hashes = {digest for digest, entries in groups.items()
                        if len({entry[0] for entry in entries}) > 1}
    collisions = [dict(split=split, index=index, path=row['path'], source=row['source'],
                       original_id=row['original_id'], aligned_sha256=result['sha256'])
                  for split, index, row, result in accepted if result['sha256'] in collision_hashes]
    counts = {}
    for split in SPLITS:
        records = [result for name, _, _, result in accepted
                   if name == split and result['sha256'] not in collision_hashes]
        write_csv(output / f'{split}.csv', records, FIELDS)
        counts[split] = dict(input=len(inputs[split]), output=len(records),
                             failures=sum(f['split'] == split for f in failures),
                             cross_split_excluded=sum(c['split'] == split for c in collisions))
    write_csv(output / 'failures.csv', failures,
              ['split', 'index', 'path', 'source', 'original_id', 'reason', 'error_type'])
    write_csv(output / 'cross_split_duplicates.csv', collisions,
              ['split', 'index', 'path', 'source', 'original_id', 'aligned_sha256'])
    seconds = time.perf_counter() - started
    report = dict(input_dir=str(data_dir), output_dir=str(output), device=device, limit_per_split=limit,
                  input_csv_sha256={s: sha256(data_dir / f'{s}.csv') for s in SPLITS},
                  counts=counts, failure_reasons=dict(Counter(f['reason'] for f in failures)),
                  cross_split_duplicate_hashes=len(collision_hashes), elapsed_seconds=seconds,
                  processed_per_second=sum(len(v) for v in inputs.values()) / seconds if seconds else None,
                  alignment='age.face.FaceAligner.align(training=True), RGB 224x224',
                  sha256_definition='SHA-256 of 224x224 RGB uint8 pixels in row-major order',
                  note='Excluded duplicate PNGs remain in images/ but are absent from both CSVs. '
                       'Only exact aligned-pixel matches are checked; test and identity leakage are not checked.')
    if model_root is not None:
        model_root = Path(model_root).resolve()
        report['model_root'] = str(model_root)
        report['model_sha256'] = {name: sha256(model_root / 'models/buffalo_l' / name)
                                  for name in ('det_10g.onnx', 'genderage.onnx')}
    (output / 'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/age_public')
    parser.add_argument('--output', type=Path, required=True, help='New directory, outside the input and source directories')
    parser.add_argument('--model-root', type=Path, default=ROOT / 'models/insightface')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--limit-per-split', type=int, default=0, help='First N rows per split for a pilot; 0 means all')
    args = parser.parse_args()
    if args.limit_per_split < 0:
        parser.error('--limit-per-split must be nonnegative')
    if args.device == 'cuda':
        # Load PyTorch's CUDA libraries before ONNX Runtime creates its sessions.
        import torch
        import onnxruntime as ort
        if not torch.cuda.is_available():
            parser.error('PyTorch CUDA runtime is unavailable')
        if 'CUDAExecutionProvider' not in ort.get_available_providers():
            parser.error('CUDAExecutionProvider is unavailable; install a compatible GPU ONNX Runtime')
    aligner = FaceAligner(args.model_root, args.device)
    if args.device == 'cuda':
        for component in (aligner.detector, aligner.attribute):
            if 'CUDAExecutionProvider' not in component.session.get_providers():
                parser.error('CUDA session initialization failed; refusing a silent CPU fallback')
    print(json.dumps(prepare(args.data_dir, args.output, aligner, args.limit_per_split, args.device,
                             args.model_root), indent=2))


if __name__ == '__main__':
    main()
