"""MobileNetV3-Large training/evaluation on native FFHQ-Aging groups, not exact ages."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .common import ROOT, save_json, sha256, write_csv
from .ffhq_data import GROUPS, GroupDataset, load_records, records_fingerprint
from .model import build_model
from .train import seed_worker

TASK = 'ffhq_age_group_classification'
LIMITATIONS = [
    'Group labels are not exact ages; no age MAE or CS@5 can be measured.',
    'Disjoint image IDs and file hashes do not establish disjoint identities.',
    'Uses existing FFHQ-aligned 512px images, not the SCRFD alignment of age.infer.',
    'This 10-class checkpoint is not compatible with the 101-bin scalar-age estimator.',
]


def classification_metrics(targets, predictions):
    targets, predictions = np.asarray(targets), np.asarray(predictions)
    if targets.ndim != 1 or targets.shape != predictions.shape or not len(targets):
        raise ValueError('Expected nonempty paired class-index arrays')
    for values in (targets, predictions):
        if not np.issubdtype(values.dtype, np.integer) or np.any((values < 0) | (values >= len(GROUPS))):
            raise ValueError('Invalid class index')
    confusion = np.zeros((len(GROUPS), len(GROUPS)), dtype=np.int64)
    np.add.at(confusion, (targets, predictions), 1)
    per_group = []
    for index, group in enumerate(GROUPS):
        tp, support, selected = int(confusion[index, index]), int(confusion[index].sum()), int(confusion[:, index].sum())
        precision = tp / selected if selected else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * tp / (support + selected) if support + selected else 0.0
        per_group.append(dict(age_group=group, count=support, precision=precision, recall=recall, f1=f1))
    return dict(count=len(targets), accuracy=float(np.mean(targets == predictions)),
                macro_f1=float(np.mean([row['f1'] for row in per_group])),
                balanced_accuracy=float(np.mean([row['recall'] for row in per_group if row['count']])),
                confusion_matrix=confusion.tolist(), per_group=per_group, age_mae=None)


@torch.inference_mode()
def predict_groups(model, loader, device):
    model.eval()
    targets, predictions, confidence = [], [], []
    total_loss = 0.0
    for images, labels in loader:
        logits = model(images.to(device))
        if not torch.isfinite(logits).all():
            raise ValueError('Nonfinite classification logits')
        total_loss += F.cross_entropy(logits, labels.to(device), reduction='sum').item()
        probabilities, indices = logits.softmax(-1).max(-1)
        targets.extend(labels.tolist())
        predictions.extend(indices.cpu().tolist())
        confidence.extend(probabilities.cpu().tolist())
    metrics = classification_metrics(targets, predictions)
    metrics['cross_entropy'] = total_loss / len(targets)
    return metrics, predictions, confidence


def train(args, records):
    counts = np.bincount([row['label'] for row in records['train']], minlength=len(GROUPS))
    if np.any(counts == 0) or len(records['train']) < 2:
        raise ValueError('Training requires every age group and at least two images')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    amp = torch.device(args.device).type == 'cuda'
    run = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    run.update(task=TASK, groups=list(GROUPS), weights='IMAGENET1K_V2',
               loss='inverse-training-frequency weighted cross entropy',
               selection='maximum validation macro-F1; tie: minimum validation cross entropy',
               dataset_fingerprint=records_fingerprint(records), provenance_sha256=sha256(args.provenance),
               split_counts={split: len(rows) for split, rows in records.items()},
               training_class_counts=counts.tolist(), identity_disjoint_verified=False,
               preprocessing='Existing FFHQ 512px RGB; train random crop/flip/color jitter; eval resize 224; ImageNet normalization',
               limitations=LIMITATIONS, command=[sys.executable, '-m', 'age.ffhq', *sys.argv[1:]],
               dataset_license='CC BY-NC-SA 4.0; preserve individual image attribution and licenses',
               dataset_citation='Or-El et al., Lifespan Age Transformation Synthesis, ECCV 2020',
               versions={name: importlib.metadata.version(name) for name in ('torch', 'torchvision', 'numpy', 'Pillow')},
               gpu=torch.cuda.get_device_name(args.device) if amp else None)
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / 'run.json', run)
    torch.hub.set_dir(str(ROOT / 'models/torch'))
    model = build_model(True, len(GROUPS)).to(args.device)
    initial_head = model.classifier[-1].weight.detach().cpu().clone()
    loader_options = dict(batch_size=args.batch_size, num_workers=args.workers,
                          pin_memory=amp, persistent_workers=args.workers > 0)
    loader = DataLoader(GroupDataset(records['train'], training=True), shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed), worker_init_fn=seed_worker,
                        **loader_options)
    val_loader = DataLoader(GroupDataset(records['val']), **loader_options)
    weights = torch.tensor(len(records['train']) / (len(GROUPS) * counts), dtype=torch.float32, device=args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    if amp:
        torch.cuda.reset_peak_memory_stats(args.device)
    best, best_epoch, updates, history = (-float('inf'), -float('inf')), None, 0, []
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        losses, skipped, steps, singletons = [], 0, 0, 0
        for images, labels in loader:
            if len(images) < 2:
                singletons += len(images)
                continue
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda' if amp else 'cpu', enabled=amp):
                loss = F.cross_entropy(model(images).float(), labels, weight=weights)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            scaler.scale(loss).backward()
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < previous_scale:
                skipped += 1
            else:
                steps += 1
            losses.append(loss.item())
        metrics, _, _ = predict_groups(model, val_loader, args.device)
        updates += steps
        lr = optimizer.param_groups[0]['lr']
        if steps:
            scheduler.step()
        candidate = (metrics['macro_f1'], -metrics['cross_entropy'])
        if steps and candidate > best:
            best, best_epoch = candidate, epoch
            torch.save(dict(task=TASK, groups=list(GROUPS), model=model.state_dict(), epoch=epoch,
                            optimizer_steps=updates, run=run, validation=metrics,
                            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                            scaler=scaler.state_dict()), args.output / 'best.pt')
        row = dict(epoch=epoch, train_mean_batch_loss=float(np.mean(losses)),
                   val_loss=metrics['cross_entropy'], val_accuracy=metrics['accuracy'], val_macro_f1=metrics['macro_f1'],
                   optimizer_steps=steps, amp_skipped_steps=skipped, singleton_images_skipped=singletons,
                   amp_scale=scaler.get_scale(), lr=lr, seconds=time.perf_counter() - started)
        history.append(row)
        write_csv(args.output / 'history.csv', history)
        print(json.dumps(row), flush=True)
    if not updates or best_epoch is None:
        raise RuntimeError('No optimizer updates completed; no trained checkpoint was saved')
    change = float((model.classifier[-1].weight.detach().cpu() - initial_head).abs().max())
    if not change > 0:
        raise RuntimeError('Classifier parameters did not change')
    save_json(args.output / 'training_summary.json', dict(
        task=TASK, epochs=args.epochs, optimizer_steps=updates,
        amp_skipped_steps=sum(row['amp_skipped_steps'] for row in history), best_epoch=best_epoch,
        best_val_macro_f1=best[0], final_classifier_max_change=change,
        peak_cuda_allocated_mib=torch.cuda.max_memory_allocated(args.device) / 2**20 if amp else None,
        checkpoint_sha256=sha256(args.output / 'best.pt'), test_used_for_selection=False,
        limitations=LIMITATIONS))


def evaluate(args, records):
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint.get('task') != TASK or checkpoint.get('groups') != list(GROUPS):
        raise ValueError('Expected a native FFHQ age-group checkpoint')
    if not checkpoint.get('optimizer_steps', 0):
        raise ValueError('Checkpoint has no optimizer updates')
    if checkpoint['run']['dataset_fingerprint'] != records_fingerprint(records):
        raise ValueError('Dataset changed since training')
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / 'evaluation_started.json', dict(checkpoint_sha256=sha256(args.checkpoint), split='test'))
    model = build_model(False, len(GROUPS)).to(args.device)
    model.load_state_dict(checkpoint['model'])
    loader = DataLoader(GroupDataset(records['test']), batch_size=args.batch_size, num_workers=args.workers)
    metrics, predictions, confidence = predict_groups(model, loader, args.device)
    targets = [row['label'] for row in records['test']]
    majority = int(np.argmax(checkpoint['run']['training_class_counts']))
    report = dict(task=TASK, split='test', groups=list(GROUPS), metrics=metrics,
                  majority_baseline=classification_metrics(targets, [majority] * len(targets)),
                  by_gender={gender: classification_metrics(
                      [targets[i] for i, row in enumerate(records['test']) if row['gender'] == gender],
                      [predictions[i] for i, row in enumerate(records['test']) if row['gender'] == gender])
                      for gender in sorted({row['gender'] for row in records['test']})},
                  checkpoint_sha256=sha256(args.checkpoint), selected_epoch=checkpoint['epoch'],
                  dataset_fingerprint=records_fingerprint(records), identity_disjoint_verified=False,
                  limitations=LIMITATIONS)
    save_json(args.output / 'metrics.json', report)
    write_csv(args.output / 'predictions.csv', [dict(image_number=row['image_number'], gender=row['gender'],
              actual_group=row['age_group'], predicted_group=GROUPS[prediction], confidence=score)
              for row, prediction, score in zip(records['test'], predictions, confidence)])
    write_csv(args.output / 'per_group.csv', metrics['per_group'])
    write_csv(args.output / 'confusion_matrix.csv', [dict(actual_group=group, **dict(zip(GROUPS, counts)))
              for group, counts in zip(GROUPS, metrics['confusion_matrix'])])
    print(json.dumps({key: value for key, value in metrics.items() if key not in ('per_group', 'confusion_matrix')}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['train', 'evaluate'])
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data/lora')
    parser.add_argument('--provenance', type=Path, default=ROOT / 'data/reports/ffhq_aging_samples.jsonl')
    parser.add_argument('--output', type=Path, required=True, help='A new directory; existing results are never overwritten')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 2 or args.workers < 0 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error('Invalid epochs, batch size, workers or learning rate')
    if (args.action == 'evaluate') != bool(args.checkpoint):
        parser.error('--checkpoint is required only for evaluate')
    if args.output.exists():
        raise FileExistsError('Choose a new --output directory')
    torch.set_num_threads(min(4, torch.get_num_threads()))
    print('Validating local image bytes, labels and existing splits...', flush=True)
    records = load_records(args.data_root, args.provenance)
    print(json.dumps({split: len(rows) for split, rows in records.items()}), flush=True)
    (train if args.action == 'train' else evaluate)(args, records)


if __name__ == '__main__':
    main()
