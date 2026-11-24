"""Public age experiment: capped, softly balanced source/age sampling.

Only train.csv and val.csv are read. Labels can be estimated ages and the
available split metadata cannot rule out identity or near-duplicate leakage.
"""
import argparse
from collections import Counter
import importlib.metadata
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .common import ROOT, save_json, sha256, write_csv
from .dataset import AgeDataset, validate_splits
from .evaluate import metric_rows, predict
from .model import build_model, dldl_loss
from .train import seed_worker


def sampling_weights(rows, cap=5.0):
    """Inverse square root of source/decade frequency, capped at cap times mean."""
    if not rows or cap < 1:
        raise ValueError('Nonempty rows and cap >= 1 required')
    keys = [(r['source'], min(int(float(r['age'])) // 10, 9)) for r in rows]
    counts = Counter(keys)
    raw = [counts[key] ** -0.5 for key in keys]
    limit = cap * sum(raw) / len(raw)
    return [min(weight, limit) for weight in raw]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=ROOT / 'data/age_public')
    p.add_argument('--output', type=Path, required=True, help='New, nonexistent experiment directory')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args(argv)
    if args.batch_size < 2 or args.epochs < 1 or args.lr <= 0 or args.workers < 0:
        p.error('Invalid training hyperparameters')
    if args.output.resolve().is_relative_to(args.data_dir.resolve()):
        p.error('Output must not be inside the dataset directory')
    return args


def main(argv=None):
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    torch.backends.cudnn.benchmark = False

    train = AgeDataset(args.data_dir / 'train.csv', True)
    val = AgeDataset(args.data_dir / 'val.csv')
    validate_splits([train, val])
    for split, dataset in (('train', train), ('val', val)):
        sources = {row['source'] for row in dataset.rows}
        if sources != {'afad', 'megaage_asian'}:
            raise ValueError(f'{split} must contain exactly AFAD and MegaAge-Asian: {sorted(sources)}')

    weights = sampling_weights(train.rows)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True,
                                    generator=torch.Generator().manual_seed(args.seed))
    loader = DataLoader(train, batch_size=args.batch_size, sampler=sampler,
                        num_workers=args.workers, pin_memory=args.device.startswith('cuda'),
                        worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val, batch_size=args.batch_size, num_workers=args.workers)
    args.output.mkdir(parents=True, exist_ok=False)
    run = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    run.update(stage='public', smoke_steps=0, no_pretrained=False,
               initialization='imagenet', weights='IMAGENET1K_V2',
               loss_direction='KL(target || prediction)', sigma=2, kl_weight=1, l1_weight=1,
               weight_decay=0.01, amp=args.device.startswith('cuda'),
               sampler='source_decade_inverse_sqrt_capped_5x_mean',
               data_sources=sorted({r['source'] for r in train.rows}),
               data_sha256={s: sha256(args.data_dir / f'{s}.csv') for s in ('train', 'val')},
               split_counts={'train': len(train), 'val': len(val)},
               versions={name: importlib.metadata.version(name) for name in ('torch', 'torchvision', 'numpy', 'Pillow')},
               label_note='Ages may be estimated; identity and near-duplicate leakage cannot be excluded.')
    save_json(args.output / 'run.json', run)
    torch.hub.set_dir(str(ROOT / 'models/torch'))
    model = build_model(True).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    amp = args.device.startswith('cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    history, best, total_steps = [], float('inf'), 0
    for epoch in range(args.epochs):
        started = time.perf_counter()
        model.train()
        losses, steps, skipped = [], 0, 0
        for images, ages, _ in loader:
            if len(images) < 2:
                continue
            images, ages = images.to(args.device), ages.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda' if amp else 'cpu', enabled=amp):
                loss = dldl_loss(model(images), ages, 2, 1, 1)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
            scaler.scale(loss).backward()
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < previous_scale:
                skipped += 1
            else:
                steps += 1
            losses.append(loss.item())
        if not losses:
            raise ValueError('No training batches')
        metrics = metric_rows(predict(model, val_loader, args.device))
        mae = next(r['mae'] for r in metrics if r['source'] == 'all' and r['age_group'] == 'all')
        elapsed = time.perf_counter() - started
        total_steps += steps
        history.extend(dict(epoch=epoch + 1, train_loss=sum(losses) / len(losses),
                            optimizer_steps=steps, amp_skipped_steps=skipped,
                            amp_scale=scaler.get_scale(), epoch_seconds=elapsed,
                            lr=optimizer.param_groups[0]['lr'], **m) for m in metrics)
        if steps:
            scheduler.step()
        if steps and mae < best:
            best = mae
            torch.save(dict(model=model.state_dict(), epoch=epoch + 1, val_mae=best, run=run,
                            optimizer_steps=total_steps, split_rows={'train': train.rows, 'val': val.rows},
                            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict()),
                       args.output / 'best.pt')
        write_csv(args.output / 'history.csv', history)
        print(f'epoch={epoch + 1} loss={sum(losses) / len(losses):.4f} val_MAE={mae:.4f} '
              f'best={best:.4f} optimizer_steps={steps} amp_skipped_steps={skipped} '
              f'seconds={elapsed:.1f}', flush=True)
    if not total_steps:
        raise RuntimeError('No optimizer updates completed; no trained checkpoint was saved')


if __name__ == '__main__':
    main()
