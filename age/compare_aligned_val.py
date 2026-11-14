"""Compare original and successfully aligned validation images with one checkpoint."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .common import FIELDS, image_path, read_csv, sha256, write_csv
from .dataset import AgeDataset, transform
from .evaluate import metric_rows, predict
from .model import build_model


def paired_rows(original_csv, aligned_csv):
    original_csv, aligned_csv = Path(original_csv), Path(aligned_csv)
    if original_csv.name != 'val.csv' or aligned_csv.name != 'val.csv':
        raise ValueError('Only val.csv inputs are allowed')
    original, aligned = read_csv(original_csv), read_csv(aligned_csv)
    if not original or not aligned:
        raise ValueError('Validation CSVs must not be empty')
    index = {}
    for row in original:
        if set(FIELDS) - row.keys() or not row['source'] or not row['original_id']:
            raise ValueError('Missing original identity or fields')
        key = row['source'], row['original_id']
        if key in index:
            raise ValueError(f'Duplicate original identity: {key}')
        index[key] = row
    seen, pairs = set(), []
    aligned_dir = aligned_csv.resolve().parent / 'images' / 'val'
    for row in aligned:
        if set(FIELDS) - row.keys() or not row['source'] or not row['original_id']:
            raise ValueError('Missing aligned identity or fields')
        key = row['source'], row['original_id']
        if key in seen or key not in index:
            raise ValueError(f'Duplicate or unknown aligned identity: {key}')
        seen.add(key)
        source = index[key]
        path = image_path(row).resolve()
        if path.parent != aligned_dir or len(path.stem) != 8 or not path.stem.isascii() or not path.stem.isdigit() or path.suffix != '.png':
            raise ValueError(f'Invalid aligned val path: {path}')
        i = int(path.stem)
        if i >= len(original) or original[i] != source:
            raise ValueError(f'Aligned path does not match original row: {path}')
        for field in ('age', 'gender', 'source', 'person_id', 'original_id'):
            if row[field] != source[field]:
                raise ValueError(f'Original label mismatch for {key}: {field}')
        age = float(source['age'])
        if not age.is_integer() or not 0 <= age <= 100:
            raise ValueError(f'Invalid age: {source["age"]}')
        for record in (source, row):
            p = image_path(record).resolve()
            if any(part.lower() == 'test' for part in p.parts):
                raise ValueError(f'Test image path is forbidden: {p}')
            if not p.is_file():
                raise FileNotFoundError(p)
        pairs.append((source, row))
    return pairs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-csv', type=Path, required=True)
    parser.add_argument('--aligned-csv', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New output directory; existing directories are refused')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--flip', action='store_true')
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.workers < 0:
        parser.error('batch-size must be positive and workers nonnegative')
    output = args.output.resolve()
    for path in (args.original_csv.resolve().parent, args.aligned_csv.resolve().parent):
        if output.is_relative_to(path) or path.is_relative_to(output):
            raise ValueError('Output must not overlap either input dataset directory')
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    pairs = paired_rows(args.original_csv, args.aligned_csv)
    hashes = {'original_csv': sha256(args.original_csv), 'aligned_csv': sha256(args.aligned_csv),
              'checkpoint': sha256(args.checkpoint)}
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    run = checkpoint['run']
    if run.get('smoke_steps') or run.get('no_pretrained'):
        raise ValueError('Smoke/random-initialization checkpoint is not an evaluation candidate')
    if run.get('data_sha256', {}).get('val') != hashes['original_csv']:
        raise ValueError('Checkpoint validation CSV hash does not match original val CSV')
    model = build_model(False).to(args.device)
    model.load_state_dict(checkpoint['model'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    results, predictions = {}, []
    for version, position in (('original', 0), ('aligned', 1)):
        dataset = AgeDataset.__new__(AgeDataset)
        dataset.rows = [pair[position] for pair in pairs]
        dataset.transform = transform(False)
        rows = predict(model, DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers),
                       args.device, args.flip)
        if len(rows) != len(pairs):
            raise RuntimeError('Prediction count mismatch')
        for result, record in zip(rows, dataset.rows):
            result.update(version=version, original_id=record['original_id'], path=record['path'])
        results[version] = metric_rows(rows)
        predictions.extend(rows)
    report = dict(input_paths={'original_csv': str(args.original_csv.resolve()),
                               'aligned_csv': str(args.aligned_csv.resolve()),
                               'checkpoint': str(args.checkpoint.resolve())},
                  sha256=hashes, count=len(pairs), device=args.device, batch_size=args.batch_size,
                  workers=args.workers, flip=args.flip, torch_version=torch.__version__,
                  note='Validation subset used for model selection; not an independent test.', metrics=results)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / 'predictions.csv', predictions)
    write_csv(output / 'metrics.csv', [dict(version=version, **row)
                                       for version, rows in results.items() for row in rows])
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
