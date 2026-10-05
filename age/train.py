import argparse
import importlib.metadata
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from .common import ROOT, save_json, sha256, write_csv
from .dataset import AgeDataset, age_sampler, validate_splits
from .evaluate import predict, metric_rows
from .model import build_model, dldl_loss


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='MobileNetV3-Large: KL(target || pred) + L1 expected age')
    p.add_argument('--data-dir', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--stage', choices=['aihub', 'stage1', 'stage2'], default='aihub',
                   help='Default: ImageNet backbone trained directly on AI Hub only')
    p.add_argument('--checkpoint', type=Path, help='Required Stage 1 checkpoint for Stage 2')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--epochs', type=int)
    p.add_argument('--lr', type=float)
    p.add_argument('--sigma', type=float, default=2)
    p.add_argument('--kl-weight', type=float, default=1)
    p.add_argument('--l1-weight', type=float, default=1)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--no-pretrained', action='store_true', help='Only for smoke verification')
    p.add_argument('--smoke-steps', type=int, default=0, help='Limit batches per epoch; outputs are NOT trained candidates')
    args = p.parse_args(argv)
    if args.data_dir is None:
        args.data_dir = ROOT / 'data' / {'aihub': 'age_aihub', 'stage1': 'age', 'stage2': 'age_stage2'}[args.stage]
    if args.output is None:
        args.output = ROOT / 'outputs/age' / args.stage
    if args.epochs is None:
        args.epochs = 10 if args.stage == 'stage2' else 30
    if args.lr is None:
        args.lr = 1e-4 if args.stage == 'stage2' else 5e-4
    if args.epochs < 1 or args.batch_size < 2 or args.lr <= 0 or args.sigma <= 0 or args.smoke_steps < 0:
        p.error('Invalid training hyperparameters (batch-size must be >= 2 for BatchNorm)')
    if args.no_pretrained and not args.smoke_steps:
        p.error('--no-pretrained is only permitted with --smoke-steps')
    if args.stage == 'stage2' and not args.checkpoint:
        p.error('Stage 2 requires --checkpoint from Stage 1')
    if args.stage != 'stage2' and args.checkpoint:
        p.error('--checkpoint is only supported for the legacy Stage 2 mode; AI Hub starts from ImageNet.')
    return args


def select_sources(datasets, stage, smoke=False):
    wanted = {'utkface', 'afad'} if stage == 'stage1' else {'aihub'}
    if smoke:
        wanted.add('synthetic')
    for split, ds in zip(('train', 'val', 'test'), datasets):
        if stage == 'aihub':
            extra = {r['source'] for r in ds.rows} - wanted
            if extra:
                raise ValueError(f'AI Hub-only mode rejects other sources in {split}: {sorted(extra)}')
        ds.rows = [r for r in ds.rows if r['source'] in wanted]
        if not ds.rows:
            raise ValueError(f'{stage} requires nonempty {split} from {sorted(wanted)}')
        if stage == 'stage1' and not smoke and {r['source'] for r in ds.rows} != wanted:
            raise ValueError('Stage 1 needs both UTKFace and AFAD in every split.')


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    torch.backends.cudnn.benchmark = False
    train = AgeDataset(args.data_dir / 'train.csv', True)
    val = AgeDataset(args.data_dir / 'val.csv')
    test = AgeDataset(args.data_dir / 'test.csv')
    validate_splits([train, val, test])
    select_sources((train, val, test), args.stage, smoke=bool(args.smoke_steps))
    if args.output.joinpath('best.pt').exists() or args.output.joinpath('run.json').exists():
        raise FileExistsError('Choose a new --output to preserve previous experiment.')
    args.output.mkdir(parents=True, exist_ok=True)
    run = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    run.update(loss_direction='KL(target || prediction)', amp=args.device.startswith('cuda'),
               weights='none' if args.no_pretrained else 'IMAGENET1K_V2',
               initialization='stage1_checkpoint' if args.stage == 'stage2' else ('random_smoke' if args.no_pretrained else 'imagenet'),
               data_sources=sorted({r['source'] for r in train.rows}),
               versions={name: importlib.metadata.version(name) for name in ('torch', 'torchvision', 'numpy', 'Pillow')},
               data_sha256={s: sha256(args.data_dir / f'{s}.csv') for s in ('train', 'val', 'test')},
               split_counts={s: len(ds) for s, ds in zip(('train', 'val', 'test'), (train, val, test))})
    torch.hub.set_dir(str(ROOT / 'models/torch'))
    model = build_model(not args.no_pretrained and args.stage != 'stage2').to(args.device)
    if args.stage == 'stage2':
        previous = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if previous['run']['stage'] != 'stage1' or previous['run'].get('smoke_steps'):
            raise ValueError('Stage 2 requires a real Stage 1 checkpoint')
        validate_splits([test, type('Rows', (), {'rows': previous['split_rows']['train']})(),
                         type('Rows', (), {'rows': previous['split_rows']['val']})()])
        model.load_state_dict(previous['model'])
        run['initial_checkpoint_sha256'] = sha256(args.checkpoint)
    print(model.classifier)
    save_json(args.output / 'run.json', run)
    # Train sampler only; validation/test retain their observed distributions.
    loader = DataLoader(train, batch_size=args.batch_size, sampler=age_sampler(train, args.seed),
                        num_workers=args.workers, pin_memory=args.device.startswith('cuda'),
                        worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val, batch_size=args.batch_size, num_workers=args.workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    amp = args.device.startswith('cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    history, best = [], float('inf')
    for epoch in range(args.epochs):
        epoch_started = time.perf_counter()
        model.train()
        losses = []
        for step, (images, ages, _) in enumerate(loader):
            if args.smoke_steps and step >= args.smoke_steps:
                break
            if len(images) < 2:
                continue
            images, ages = images.to(args.device), ages.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda' if amp else 'cpu', enabled=amp):
                loss = dldl_loss(model(images), ages, args.sigma, args.kl_weight, args.l1_weight)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        if not losses:
            raise ValueError('No training batches (at least two training images required)')
        metrics = metric_rows(predict(model, val_loader, args.device))
        mae = next(r['mae'] for r in metrics if r['source'] == 'all' and r['age_group'] == 'all')
        epoch_seconds = time.perf_counter() - epoch_started
        history.extend(dict(epoch=epoch + 1, train_loss=sum(losses) / len(losses),
                            epoch_seconds=epoch_seconds, lr=optimizer.param_groups[0]['lr'], **m)
                       for m in metrics)
        scheduler.step()
        if mae < best:
            best = mae
            torch.save(dict(model=model.state_dict(), epoch=epoch + 1, val_mae=best, run=run,
                            split_rows={'train': train.rows, 'val': val.rows},
                            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict()), args.output / 'best.pt')
        write_csv(args.output / 'history.csv', history)
        print(f'epoch={epoch + 1} loss={sum(losses) / len(losses):.4f} val_MAE={mae:.4f} best={best:.4f} '
              f'seconds={epoch_seconds:.1f} estimated_remaining_hours={epoch_seconds * (args.epochs - epoch - 1) / 3600:.2f}', flush=True)
    print(f'Checkpoint: {args.output / "best.pt"}. Test evaluation is a separate frozen-checkpoint command.')


if __name__ == '__main__':
    main()
