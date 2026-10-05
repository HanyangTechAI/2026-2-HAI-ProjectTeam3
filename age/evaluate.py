import argparse
from collections import defaultdict
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .common import save_json, write_csv, sha256, image_path
from .dataset import AgeDataset, validate_splits
from .model import build_model, expected_age


def metric_rows(predictions):
    groups = defaultdict(list)
    for row in predictions:
        source, age = row['source'], float(row['age'])
        group = f'{min(int(age) // 10, 9) * 10}-{min(min(int(age) // 10, 9) * 10 + 9, 100)}'
        if age >= 90:
            group = '90-100'
        error = abs(float(row['prediction']) - age)
        for key in [('all', 'all'), (source, 'all'), (source, group)]:
            groups[key].append(error)
    return [dict(source=s, age_group=g, count=len(e), mae=sum(e) / len(e),
                 cs_at_5=sum(x <= 5 for x in e) / len(e)) for (s, g), e in sorted(groups.items())]


@torch.inference_mode()
def predict(model, loader, device, flip=False):
    model.eval()
    rows = []
    for images, ages, sources in loader:
        images = images.to(device)
        predictions = expected_age(model(images))
        if flip:
            predictions = (predictions + expected_age(model(images.flip(-1)))) / 2
        rows.extend(dict(source=s, age=float(a), prediction=float(p))
                    for s, a, p in zip(sources, ages, predictions.cpu()))
    return rows


def write_metrics(output, predictions, prefix='model'):
    output = Path(output)
    metrics = metric_rows(predictions)
    write_csv(output / f'{prefix}_predictions.csv', predictions)
    write_csv(output / f'{prefix}_metrics.csv', metrics)
    if metrics:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        grouped = [r for r in metrics if r['age_group'] != 'all']
        fig, ax = plt.subplots(figsize=(max(6, len(grouped) * 0.6), 4))
        ax.bar([f'{r["source"]}\n{r["age_group"]}\nn={r["count"]}' for r in grouped], [r['mae'] for r in grouped])
        ax.set_ylabel('MAE (years)')
        ax.tick_params(axis='x', labelsize=8, rotation=45)
        fig.tight_layout()
        fig.savefig(output / f'{prefix}_age_mae.png', dpi=150)
        plt.close(fig)
    return metrics


def main():
    p = argparse.ArgumentParser(description='Evaluate frozen checkpoint once on held-out data.')
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--csv', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('outputs/eval'))
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--baseline', action='store_true')
    p.add_argument('--model-root', default='models/insightface')
    p.add_argument('--flip', action='store_true', help='Optional inference-time flip; keep fixed across comparisons')
    args = p.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if checkpoint['run'].get('smoke_steps') or checkpoint['run'].get('no_pretrained'):
        raise ValueError('Smoke/random-initialization checkpoint is not an evaluation candidate.')
    ds = AgeDataset(args.csv)
    for split in ('train', 'val'):
        # IDs captured inside checkpoint avoid dependence on subsequently rewritten CSVs.
        from types import SimpleNamespace
        validate_splits([ds, SimpleNamespace(rows=checkpoint['split_rows'][split])])
    args.output.mkdir(parents=True, exist_ok=True)
    token = sha256(args.checkpoint)[:16] + '_' + sha256(args.csv)[:16]
    marker = args.output / f'evaluated_{token}.json'
    if marker.exists():
        raise ValueError(f'This checkpoint/test pair was already evaluated: {marker}')
    model = build_model(False).to(args.device)
    model.load_state_dict(checkpoint['model'])
    rows = predict(model, DataLoader(ds, batch_size=args.batch_size), args.device, args.flip)
    for row, record in zip(rows, ds.rows):
        row['path'] = record['path']
    results = write_metrics(args.output, rows)
    if args.baseline:
        from .face import FaceAligner, read_rgb
        from .prep import failure
        aligner = FaceAligner(args.model_root, args.device)
        baseline, failures = [], []
        for record in ds.rows:
            try:
                face = aligner.select(read_rgb(image_path(record)), training=True)
                baseline.append(dict(source=record['source'], age=float(record['age']),
                                     prediction=float(face.age), path=record['path']))
            except (ValueError, OSError) as e:
                failures.append(failure(record['source'], record.get('original_id', ''), 'baseline', e))
        write_metrics(args.output, baseline, 'insightface')
        write_csv(args.output / 'baseline_failures.csv', failures, ['source', 'original_id', 'stage', 'reason'])
        # Also compare on exactly the successfully evaluated baseline subset.
        successful = {r['path'] for r in baseline}
        write_metrics(args.output, [r for r in rows if r['path'] in successful], 'model_baseline_subset')
    save_json(marker, dict(checkpoint_sha256=sha256(args.checkpoint), test_csv_sha256=sha256(args.csv),
                           flip=args.flip, metrics=results))
    print(results)


if __name__ == '__main__':
    main()
