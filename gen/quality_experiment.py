"""Resumable same-model diagnostics on validated val inputs, optionally timed.

This is not a quality certification. Reference ages are not MAE, and the blank
blind-review form must be completed by people before claiming human validation.
"""
import argparse
from collections import defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from age.common import read_csv, save_json, sha256, write_csv
from .compare_lora import _asset, _face_data, load_manifest
from .eval import cosine_similarity
from .pipeline import AgeShiftPipeline, PipelineConfig, read_image, validate_targets


CONDITIONS = {
    'no_edit': ('copy', {}),
    'baseline': ('faceid', {}),
    'age_half': ('faceid_age', {'age_lora_scale': 0.5}),
    'age_full': ('faceid_age', {'age_lora_scale': 1.0}),
    'clip_half': ('faceid', {'shortcut_scale': 0.5}),
    'clip_zero': ('faceid', {'shortcut_scale': 0.0}),
    'ip_low': ('faceid', {'ip_scale': 0.6}),
    'strength_high': ('faceid', {'strength': 0.8}),
    'face_denoise': ('faceid', {'generation_mask_mode': 'face'}),
    'small_cores': ('faceid', {'feature_protection': 0.5}),
    'expanded_mask': ('faceid', {'mask_scale': 1.15}),
    'img2img_composite': ('faceid', {'masked_denoising': False}),
    'numeric': ('faceid', {'prompt_style': 'numeric'}),
}
DEFAULT_CONDITIONS = tuple(CONDITIONS)
CONDITIONS.update({
    'clip_quarter': ('faceid', {'shortcut_scale': 0.25}),
    'manual_mask': ('faceid', {}),
    'manual_clip_quarter': ('faceid', {'shortcut_scale': 0.25}),
    'accessory_negative': ('faceid', {'shortcut_scale': 0.25}),
    'manual_clip_quarter_negative': ('faceid', {'shortcut_scale': 0.25}),
})
MASK_CONDITIONS = {'manual_mask', 'manual_clip_quarter', 'manual_clip_quarter_negative'}
NEGATIVE_CONDITIONS = {'accessory_negative', 'manual_clip_quarter_negative'}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def parse_deadline(value):
    if value is None:
        return None
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('Deadline must include an explicit timezone')
    return result.timestamp()


def can_start(deadline, reserve_seconds=60, now=None):
    return deadline is None or (time.time() if now is None else now) + reserve_seconds < deadline


def append_event(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def row_key(row):
    return row['condition'], row['sample_id'], row['target'], row['seed']


def load_rows(output):
    path = output / 'rows.jsonl'
    if not path.exists():
        return []
    rows, keys = [], set()
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f'Invalid row journal line {number}; preserve and inspect before resuming') from exc
        if row_key(row) in keys:
            raise ValueError('Duplicate completed row in journal')
        for relative, digest in row.get('artifacts', {}).items():
            file = (output / relative).resolve()
            if not file.is_relative_to(output.resolve()) or not file.is_file() or sha256(file) != digest:
                raise ValueError(f'Completed artifact changed or missing: {relative}')
        rows.append(row)
        keys.add(row_key(row))
    return rows


def load_mask_inputs(path, samples):
    path = Path(path).resolve(strict=True)
    record = json.loads(path.read_text(encoding='utf-8'))
    if record.get('schema') != 1 or record.get('coordinate_space') != 'source':
        raise ValueError('Mask manifest must declare schema1/source coordinates')
    entries = {}
    for sample in samples:
        entry = record.get('samples', {}).get(sample['sample_id'])
        if not entry or entry.get('input_sha256') != sample['sha256']:
            raise ValueError('Missing mask entry or mismatched source hash')
        image = read_image(sample['input_path'])
        if entry.get('source_size') != list(image.size):
            raise ValueError('Mask manifest source dimensions do not match')
        checked = {}
        for key in ('source_edit_mask', 'source_generation_mask', 'source_protect_mask'):
            if key not in entry:
                if key == 'source_generation_mask':
                    continue
                raise ValueError(f'Mask manifest needs {key}')
            asset = entry[key]
            file = Path(asset['path']).resolve(strict=True)
            if not file.is_relative_to(path.parent) or sha256(file) != asset['sha256']:
                raise ValueError('Mask file escaped its manifest directory or changed hash')
            with Image.open(file) as mask:
                if mask.mode != 'L' or mask.size != image.size:
                    raise ValueError('Masks must be source-sized grayscale L without implicit conversion')
                if key != 'source_protect_mask' and not mask.getbbox():
                    raise ValueError('Editable mask is empty')
            checked[key] = dict(path=str(file), sha256=asset['sha256'])
        if not isinstance(entry.get('negative_prompt'), str):
            raise ValueError('Mask annotation must explicitly record its source-specific negative prompt')
        checked['negative_prompt'] = entry['negative_prompt']
        checked['input_sha256'] = sample['sha256']
        checked['notes'] = entry.get('notes', '')
        entries[sample['sample_id']] = checked
    return dict(path=str(path), sha256=sha256(path), author=record.get('author'),
                human_verified=record.get('human_verified', False), samples=entries)


def source_call_inputs(plan, settings, sample):
    """Recheck every external mask before use; never resize or silently replace it."""
    entry = plan.get('mask_inputs', {}).get('samples', {}).get(sample['sample_id'])
    if not entry:
        if settings.get('mask_policy') or settings.get('negative_policy'):
            raise ValueError('Condition requires recorded source annotations')
        return {}, {}
    masks = {}
    for name in ('source_edit_mask', 'source_generation_mask', 'source_protect_mask'):
        if name in entry:
            asset = entry[name]
            if sha256(asset['path']) != asset['sha256']:
                raise ValueError('Mask asset changed after preflight')
            with Image.open(asset['path']) as image:
                masks[name] = image.copy()
    arguments = dict(masks) if settings.get('mask_policy') == 'manual' else {}
    if settings.get('negative_policy') == 'source_annotation':
        arguments['negative_prompt'] = entry['negative_prompt']
    return arguments, masks


def build_plan(manifest, config, checkpoint, conditions, targets, seeds, sample_ids=None, mask_manifest=None):
    source = load_manifest(manifest)
    samples = source['samples']
    if sample_ids:
        indexed = {sample['sample_id']: sample for sample in samples}
        if len(set(sample_ids)) != len(sample_ids) or any(name not in indexed for name in sample_ids):
            raise ValueError('Sample IDs must be unique IDs from the validated val manifest')
        samples = [indexed[name] for name in sample_ids]
    targets = validate_targets(targets)
    if len(set(targets)) != len(targets):
        raise ValueError('Targets must be unique')
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or not 0 <= s < 2**63 for s in seeds):
        raise ValueError('Seeds must be unique integers in [0, 2**63)')
    if not conditions or len(set(conditions)) != len(conditions) or any(c not in CONDITIONS for c in conditions):
        raise ValueError('Choose unique known conditions')
    if set(conditions) & (MASK_CONDITIONS | NEGATIVE_CONDITIONS) and mask_manifest is None:
        raise ValueError('Manual-mask/accessory conditions require --mask-manifest')
    mask_inputs = load_mask_inputs(mask_manifest, samples) if mask_manifest is not None else None
    checkpoint = Path(checkpoint).resolve(strict=True)
    if not checkpoint.is_file():
        raise ValueError('An explicit Age LoRA FILE is required, never a stale config default')
    values = json.loads(Path(config).read_text(encoding='utf-8'))
    values['age_lora'] = str(checkpoint)
    cfg = PipelineConfig(**values)
    if not cfg.preserve_background:
        raise ValueError('These diagnostics require final source-image compositing')
    assets = {}
    for name in ('base_model', 'image_encoder', 'ip_adapter', 'faceid_lora', 'model_root', 'age_lora'):
        value = getattr(cfg, name)
        if not value and name == 'faceid_lora':
            continue
        assets[name] = _asset(value, name in ('base_model', 'image_encoder', 'model_root'))
    cfg = replace(cfg, **{name: asset['path'] for name, asset in assets.items()})
    root = Path(__file__).resolve().parents[1]
    code = {name: sha256(root / name) for name in (
        'gen/pipeline.py', 'gen/faces.py', 'gen/quality_experiment.py',
        'gen/compare_lora.py', 'gen/eval.py', 'age/face.py')}
    result = dict(schema=2, samples=samples, source_split='val', manifest_sha256=sha256(manifest),
                 base_config=asdict(cfg), assets=assets, code_sha256=code,
                conditions={name: {'mode': CONDITIONS[name][0], 'overrides': CONDITIONS[name][1],
                                   'mask_policy': 'manual' if name in MASK_CONDITIONS else None,
                                   'negative_policy': 'source_annotation' if name in NEGATIVE_CONDITIONS else None}
                            for name in conditions}, targets=targets, seeds=list(seeds),
                 note='Reference ages are not accuracy. Copy control is deliberately not age editing. '
                      'Geometric mask preservation is not semantic protection of all clothing/occlusions.')
    if mask_inputs is not None:
        result['mask_inputs'] = mask_inputs
    return result


def summarize(rows, expected, status):
    groups = defaultdict(list)
    for row in rows:
        groups[row['condition']].append(row)
    result = dict(status=status, completed_attempts=len(rows), expected_attempts=expected,
                  updated_utc=utc_now(), age_mae=None, human_review_complete=False, conditions={})
    for name, group in groups.items():
        stats = dict(attempts=len(group), saved=sum(bool(r.get('output_path')) for r in group),
                     errors=sum(bool(r.get('error')) for r in group),
                     output_face_detected=sum((r.get('output_face_count') or 0) > 0 for r in group),
                     face_detection_denominator=len(group))
        for field in ('identity_cosine', 'reference_age', 'raw_identity_cosine', 'raw_reference_age',
                      'elapsed_seconds', 'protected_changed_pixels', 'annotation_protected_changed_pixels',
                      'annotation_outside_edit_changed_pixels'):
            values = [r[field] for r in group if r.get(field) is not None]
            stats[field + '_mean'] = sum(values) / len(values) if values else None
            stats[field + '_count'] = len(values)
        result['conditions'][name] = stats
    return result


def write_review(output, plan, rows):
    """Condition and target keys are separate from blank human rating forms."""
    saved = [r for r in rows if r.get('output_path')]
    saved.sort(key=lambda r: hashlib.sha256('|'.join(map(str, row_key(r))).encode()).hexdigest())
    keyed = []
    for row in saved:
        review_id = 'R' + hashlib.sha256('|'.join(map(str, row_key(row))).encode()).hexdigest()[:12]
        keyed.append(dict(review_id=review_id, **{k: row[k] for k in ('condition', 'sample_id', 'target', 'seed', 'output_path')}))
    save_json(output / 'review_key.json', keyed)
    template = output / 'review_template.csv'
    previous = {row['review_id']: row for row in read_csv(template)} if template.exists() else {}
    write_csv(template, [previous.get(r['review_id'], dict(review_id=r['review_id'], reviewer='',
              perceived_age_band='', identity_1to5='', naturalness_1to5='', expression_ok='',
              occlusion_ok='', comment='')) for r in keyed],
              fields=['review_id', 'reviewer', 'perceived_age_band', 'identity_1to5',
                      'naturalness_1to5', 'expression_ok', 'occlusion_ok', 'comment'])
    samples = {s['sample_id']: s for s in plan['samples']}
    folder = output / 'review_sheets'
    folder.mkdir(exist_ok=True)
    for start in range(0, len(keyed), 8):
        batch = keyed[start:start + 8]
        sheet = Image.new('RGB', (4 * 256, 2 * 280), 'white')
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(batch):
            x, y = (index % 4) * 256, (index // 4) * 280
            for offset, path in ((0, Path(samples[row['sample_id']]['input_path'])),
                                 (128, output / row['output_path'])):
                tile = ImageOps.contain(read_image(path), (126, 248))
                sheet.paste(tile, (x + offset + (126 - tile.width) // 2, y + (248 - tile.height) // 2))
            draw.text((x + 5, y + 250), row['review_id'] + ' | source / result', fill='black')
        sheet.save(folder / f'{start // 8:03d}.png')


def run(plan, output, deadline=None, *, resume=False, reserve_seconds=60,
        pipeline_factory=AgeShiftPipeline, faces_factory=None):
    output = Path(output)
    if resume:
        if json.loads((output / 'plan.json').read_text(encoding='utf-8')) != plan:
            raise ValueError('Resume refused: inputs, models, code or settings changed; use a new output')
    else:
        output.mkdir(parents=True, exist_ok=False)
        save_json(output / 'plan.json', plan)
    rows = load_rows(output)
    completed = {row_key(row) for row in rows}
    expected = len(plan['conditions']) * len(plan['samples']) * len(plan['targets']) * len(plan['seeds'])
    append_event(output / 'events.jsonl', dict(event='start', utc=utc_now(), argv=sys.argv,
                 deadline_utc=None if deadline is None else datetime.fromtimestamp(deadline, timezone.utc).isoformat(),
                 resume=resume, already_completed=len(rows)))
    status = 'paused_deadline'
    try:
        for condition, settings in plan['conditions'].items():
            cfg = replace(PipelineConfig(**plan['base_config']), **settings['overrides'])
            pending = [(index, sample, target, seed) for index, sample in enumerate(plan['samples'])
                       for seed in plan['seeds'] for target in plan['targets']
                       if (condition, sample['sample_id'], target, seed) not in completed]
            if not pending:
                continue
            if not can_start(deadline, reserve_seconds):
                break
            pipeline = None
            try:
                started = time.perf_counter()
                if settings['mode'] == 'copy':
                    if faces_factory is None:
                        from .faces import FaceTools
                        faces = FaceTools(cfg.model_root, cfg.device, identity=True)
                    else:
                        faces = faces_factory(cfg)
                else:
                    pipeline = pipeline_factory(cfg, settings['mode'])
                    if pipeline.is_mock:
                        raise ValueError('Mock cannot produce quality experiment metrics')
                    pipeline.load()
                    faces = pipeline._faces
                append_event(output / 'events.jsonl', dict(event='loaded', condition=condition,
                             utc=utc_now(), seconds=time.perf_counter() - started, config=asdict(cfg)))
                for index, sample, target, seed in pending:
                    if not can_start(deadline, reserve_seconds):
                        break
                    row = dict(condition=condition, sample_id=sample['sample_id'], target=target, seed=seed,
                               source_age_group=sample['age_group'], status='ok', error='', artifacts={},
                               output_path='', identity_cosine=None, reference_age=None, raw_identity_cosine=None,
                               raw_reference_age=None, output_face_count=None, protected_changed_pixels=None,
                               elapsed_seconds=None, prompt=None, annotation_protected_changed_pixels=None,
                               annotation_outside_edit_changed_pixels=None, manual_mask_violation=False,
                               negative_prompt_override=None)
                    stem = Path('images') / condition / f'{index:02d}_{target:03d}_{seed}'
                    (output / stem).mkdir(parents=True, exist_ok=True)
                    def save_image(name, image):
                        relative = stem / (name + '.png')
                        image.save(output / relative)
                        row['artifacts'][relative.as_posix()] = sha256(output / relative)
                        return relative.as_posix()
                    diagnostics = {}
                    def capture(data):
                        diagnostics.update(data)
                    started = time.perf_counter()
                    try:
                        if sha256(sample['input_path']) != sample['sha256']:
                            raise ValueError('Input hash changed')
                        original = read_image(sample['input_path'])
                        call_inputs, annotation_masks = source_call_inputs(plan, settings, sample)
                        row['negative_prompt_override'] = call_inputs.get('negative_prompt')
                        source = _face_data(original, faces, True)
                        if source['errors']:
                            row['source_metric_errors'] = source['errors']
                        if pipeline is None:
                            result = original.copy()
                            row['protected_changed_pixels'] = 0
                        else:
                            generated = pipeline.age_shift(original.copy(), [target], gender=sample['gender'],
                                                            seed=seed, diagnostics=capture, **call_inputs)
                            if len(generated) != 1 or not diagnostics:
                                raise ValueError('Expected one generated image and diagnostic payload')
                            result = generated[0]
                            row['prompt'] = diagnostics['prompt']
                            edit = diagnostics['edit']
                            if edit is None or result.size != original.size:
                                raise ValueError('Source-space composition required')
                            save_image('crop', edit.crop)
                            save_image('blend_mask', edit.mask)
                            save_image('source_mask', edit.source_mask)
                            save_json(output / stem / 'geometry.json', dict(matrix=edit.matrix.tolist(),
                                      selection=edit.selection_info))
                            row['artifacts'][(stem / 'geometry.json').as_posix()] = sha256(output / stem / 'geometry.json')
                            for name in ('generation_mask', 'latent_mask', 'raw'):
                                if diagnostics.get(name) is not None:
                                    save_image(name, diagnostics[name])
                            before, after = np.asarray(original), np.asarray(result)
                            protected = np.asarray(edit.source_mask) == 0
                            row['protected_changed_pixels'] = int(np.count_nonzero(np.any(before != after, axis=2) & protected))
                            raw_data = _face_data(diagnostics['raw'], faces, True)
                            row['raw_reference_age'] = raw_data['age']
                            row['raw_metric_errors'] = raw_data['errors']
                            if source['embedding'] is not None and raw_data['embedding'] is not None:
                                row['raw_identity_cosine'] = cosine_similarity(source['embedding'], raw_data['embedding'])
                        if annotation_masks:
                            changed = np.any(np.asarray(original) != np.asarray(result), axis=2)
                            row['annotation_protected_changed_pixels'] = int(np.count_nonzero(
                                changed & (np.asarray(annotation_masks['source_protect_mask']) > 0)))
                            row['annotation_outside_edit_changed_pixels'] = int(np.count_nonzero(
                                changed & (np.asarray(annotation_masks['source_edit_mask']) == 0)))
                            if settings.get('mask_policy') == 'manual':
                                row['manual_mask_violation'] = bool(row['annotation_protected_changed_pixels'] or
                                                                   row['annotation_outside_edit_changed_pixels'])
                            for name, mask in annotation_masks.items():
                                save_image('annotation_' + name, mask)
                        row['output_path'] = save_image('output', result)
                        final = _face_data(result, faces, True)
                        row.update(reference_age=final['age'], output_face_count=final['count'],
                                   source_reference_age=source['age'], metric_errors=final['errors'])
                        if source['embedding'] is not None and final['embedding'] is not None:
                            row['identity_cosine'] = cosine_similarity(source['embedding'], final['embedding'])
                        if row['protected_changed_pixels'] or row['manual_mask_violation']:
                            raise ValueError('Protected source pixels changed')
                    except Exception as exc:
                        row.update(status='error', error=f'{type(exc).__name__}: {exc}')
                    row['elapsed_seconds'] = time.perf_counter() - started
                    append_event(output / 'rows.jsonl', row)
                    rows.append(row)
                    completed.add(row_key(row))
                    save_json(output / 'summary.json', summarize(rows, expected, 'running'))
                    print(json.dumps({k: row[k] for k in ('condition', 'sample_id', 'target', 'seed', 'status', 'error')}, ensure_ascii=False), flush=True)
                    if row.get('protected_changed_pixels') or row.get('manual_mask_violation'):
                        raise RuntimeError('Stopping after protection invariant violation')
            finally:
                if pipeline is not None:
                    pipeline.close()
                pipeline = None
                faces = None
                gc.collect()
        if len(rows) == expected:
            status = 'completed'
    except KeyboardInterrupt:
        status = 'paused_interrupt'
    except Exception as exc:
        status = 'failed'
        append_event(output / 'events.jsonl', dict(event='fatal', utc=utc_now(), error=f'{type(exc).__name__}: {exc}'))
        raise
    finally:
        summary = summarize(rows, expected, status)
        save_json(output / 'summary.json', summary)
        fields = sorted(set().union(*(r.keys() for r in rows))) if rows else ['condition', 'sample_id', 'target', 'seed', 'status']
        write_csv(output / 'results.csv', rows, fields=fields)
        write_review(output, plan, rows)
        append_event(output / 'events.jsonl', dict(event='stop', utc=utc_now(), status=status, completed=len(rows)))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--conditions', nargs='+', choices=list(CONDITIONS), default=list(DEFAULT_CONDITIONS))
    parser.add_argument('--targets', nargs='+', type=int, default=[25, 45, 75])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 31415])
    parser.add_argument('--sample-ids', nargs='+')
    parser.add_argument('--mask-manifest', type=Path, help='Hash-bound source-space diagnostic masks and annotations')
    parser.add_argument('--deadline', help='Explicit one-shot ISO UTC/offset time; omitted means no time limit')
    parser.add_argument('--reserve-seconds', type=float, default=60)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if not np.isfinite(args.reserve_seconds) or args.reserve_seconds < 10:
        parser.error('--reserve-seconds must be finite and at least 10')
    deadline = parse_deadline(args.deadline)
    for name in ('HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE'):
        os.environ[name] = '1'
    plan = build_plan(args.manifest, args.config, args.checkpoint, args.conditions,
                      args.targets, args.seeds, args.sample_ids, args.mask_manifest)
    print(json.dumps(run(plan, args.output, deadline, resume=args.resume,
                         reserve_seconds=args.reserve_seconds), indent=2), flush=True)


if __name__ == '__main__':
    main()
