"""Fixed val-only checkpoint comparison, without an official age estimator.

Run conditions separately with --baseline / --checkpoint LABEL=PATH, then use
--combine DIR... to assemble paired tables and visual-review sheets on CPU.
This is deliberately separate from gen.eval, whose real MAE requires age ONNX.
"""
import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import re
import time

from PIL import Image, ImageDraw, ImageOps

from age.common import read_csv, save_json, sha256, write_csv
from .eval import cosine_similarity
from .pipeline import AgeShiftPipeline, PipelineConfig, age_prompt, as_rgb, read_image

TARGETS = [8, 25, 45, 65, 80]
SEED = 42
SCHEMA_VERSION = 1
REFERENCE_FIELDS = ('mouth_eye_ratio', 'eye_nose_ratio', 'nose_mouth_ratio')
METRICS = ('identity_cosine', 'insightface_age_reference', 'landmark_procrustes_reference',
           *[f'{name}_reference' for name in REFERENCE_FIELDS])
NOTES = {
    'age': 'No official age ONNX is used. age_mae is null. InsightFace age is optional reference only, never MAE.',
    'geometry': 'Five-landmark geometry and rotation/translation/scale-normalized Procrustes error are references, '
                'not human evaluation. Geometry cannot establish the absence of artifacts.',
    'visual_review': 'Observer visual review is separate and has not been performed by this command.',
    'timing': 'One target per call; synchronized CUDA wall time includes face preprocessing, excludes PNG saving '
              'and scoring. Loading is timed separately. No warmup or tuning is added.',
    'memory': 'PyTorch maximum allocated/reserved bytes, not total GPU memory; external ONNX allocations are not measured. '
              'CPU memory fields are null. Per-condition peaks include model loading and generation.',
    'denominators': 'Output detection rate uses ALL planned sample-target rows, including loading, generation and '
                    'scoring errors. Evaluated-only rate is also reported; unknown is not a measured no-face result. '
                    'Metric means use available values with explicit counts. Paired differences use matched valid '
                    'values, retaining every planned pair in expected_pairs and error counts.',
    'selection': 'Fixed validation subset only; not a final test, an identity-disjoint guarantee, or a quality claim.',
    'missing_values': 'Null JSON fields and blank CSV cells mean unavailable, never substituted zero scores.',
}


def load_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(manifest, dict):
        raise ValueError('Manifest root must be a JSON object')
    if manifest.get('source_split') != 'val':
        raise ValueError('Only source_split=val is allowed; train/test are forbidden')
    if manifest.get('targets') != TARGETS or any(type(t) is not int for t in manifest['targets']):
        raise ValueError(f'Fixed targets must be {TARGETS}')
    if type(manifest.get('seed')) is not int or manifest['seed'] != SEED:
        raise ValueError('Fixed seed must be 42')
    samples = manifest.get('samples')
    if not isinstance(samples, list) or not samples:
        raise ValueError('Manifest needs a nonempty samples list')
    identifiers, originals, hashes = set(), set(), set()
    for sample in samples:
        if not isinstance(sample, dict) or not isinstance(sample.get('sample_id'), str) or not sample['sample_id']:
            raise ValueError('Every sample needs a nonempty string sample_id')
        if sample.get('gender') not in ('male', 'female') or not isinstance(sample.get('age_group'), str) or not sample['age_group']:
            raise ValueError('Every sample needs metadata gender=male/female and age_group')
        if any(sample.get(key, 'val') != 'val' for key in ('split', 'source_split')):
            raise ValueError('Sample split must be val, never train/test')
        digest = sample.get('sha256', '')
        if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise ValueError('Invalid sample sha256')
        for key in ('path', 'input_path'):
            value = sample.get(key)
            if not isinstance(value, str) or not Path(value).is_absolute():
                raise ValueError(f'{key} must be an absolute local path')
            resolved = Path(value).resolve(strict=True)
            if not resolved.is_file() or {'train', 'test'} & set(resolved.parts):
                raise ValueError(f'{key} must not refer to train/test: {resolved}')
            if key == 'path' and (resolved.parent.name != 'images' or resolved.parent.parent.name != 'val'):
                raise ValueError('Original path must resolve inside val/images, not train/test or a symlink to them')
            if sha256(resolved) != digest:
                raise ValueError(f'{key} SHA-256 mismatch for {sample["sample_id"]}')
        original = str(Path(sample['path']).resolve())
        if Path(sample['input_path']).resolve() == Path(original):
            raise ValueError('input_path must be a saved copy, not the original or a symlink to it')
        if sample['sample_id'] in identifiers or original in originals or digest in hashes:
            raise ValueError('Duplicate sample ID, original path or input hash')
        identifiers.add(sample['sample_id'])
        originals.add(original)
        hashes.add(digest)
    return manifest


def _asset(path, directory=False):
    path = Path(path).resolve()
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f'Missing local model asset: {path}')
    if not directory:
        return {'path': str(path), 'sha256': sha256(path)}
    files = {p.relative_to(path).as_posix(): sha256(p) for p in sorted(path.rglob('*'))
             if p.is_file() and not {'.git', '__pycache__'} & set(p.parts) and p.suffix != '.pyc'}
    if not files:
        raise ValueError(f'Empty local model directory: {path}')
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {'path': str(path), 'sha256': digest, 'files': files}


def _input_signature(manifest):
    return [{key: sample[key] for key in ('sample_id', 'path', 'age_group', 'gender', 'sha256')}
            for sample in sorted(manifest['samples'], key=lambda sample: sample['sample_id'])]


class RunClock:
    """Lazy CUDA access; injected clocks let unit tests avoid torch entirely."""
    def __init__(self, device):
        self.device = device
        self.cuda = None
        if device == 'cuda':
            import torch
            if not torch.cuda.is_available():
                raise ValueError('CUDA is unavailable; no automatic CPU/mock fallback')
            self.cuda = torch.cuda

    def now(self):
        return time.perf_counter()

    def synchronize(self):
        if self.cuda is not None:
            self.cuda.synchronize()

    def reset_peak(self):
        if self.cuda is not None:
            self.cuda.reset_peak_memory_stats()

    def memory(self):
        return {'max_allocated_bytes': self.cuda.max_memory_allocated() if self.cuda else None,
                'max_reserved_bytes': self.cuda.max_memory_reserved() if self.cuda else None}


def _timed(call, clock):
    result, seconds, error = None, None, ''
    try:
        clock.synchronize()
        started = clock.now()
        try:
            result = call()
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        clock.synchronize()
        seconds = clock.now() - started
    except Exception as exc:
        error = '; '.join(filter(None, (error, f'timing: {type(exc).__name__}: {exc}')))
    return result, seconds, error


def landmark_geometry(points):
    """Eye/nose/mouth distances in interocular units, not artifact detection."""
    points = [tuple(map(float, point)) for point in points]
    if len(points) != 5 or any(len(p) != 2 for p in points) or not all(math.isfinite(x) for p in points for x in p):
        raise ValueError('Expected five finite 2D landmarks')
    eye = math.dist(points[0], points[1])
    if eye <= 1e-12:
        raise ValueError('Degenerate eye landmarks')
    eye_mid = tuple((a + b) / 2 for a, b in zip(points[0], points[1]))
    mouth_mid = tuple((a + b) / 2 for a, b in zip(points[3], points[4]))
    ratios = (math.dist(points[3], points[4]) / eye, math.dist(eye_mid, points[2]) / eye,
              math.dist(points[2], mouth_mid) / eye)
    if not math.isfinite(eye) or not all(map(math.isfinite, ratios)):
        raise ValueError('Nonfinite landmark geometry')
    return dict(zip(REFERENCE_FIELDS, ratios))


def procrustes_error(source, output):
    """Unit-Frobenius landmark distance after optimal 2D rotation, no reflection."""
    normalized = []
    for points in (source, output):
        landmark_geometry(points)
        center = [sum(float(p[i]) for p in points) / 5 for i in (0, 1)]
        centered = [(float(p[0]) - center[0], float(p[1]) - center[1]) for p in points]
        norm = math.sqrt(sum(x*x + y*y for x, y in centered))
        if not math.isfinite(norm) or norm <= 1e-12:
            raise ValueError('Degenerate landmark normalization')
        normalized.append([(x / norm, y / norm) for x, y in centered])
    dot = sum(x*u + y*v for (x, y), (u, v) in zip(*normalized))
    cross = sum(x*v - y*u for (x, y), (u, v) in zip(*normalized))
    return math.sqrt(max(0.0, 2 - 2 * min(1.0, math.hypot(dot, cross))))


def _face_data(image, faces, reference_age):
    data = dict(count=None, points=None, embedding=None, age=None, geometry={}, errors=[])
    selected = None
    try:
        _, selected = faces._select(image)
        data['count'] = selected.detected_count
        data['points'] = selected.kps
        data['geometry'] = landmark_geometry(data['points'])
    except Exception as exc:
        info = getattr(exc, 'selection_info', None)
        if isinstance(info, dict) and isinstance(info.get('detected_count'), int):
            data['count'] = info['detected_count']
        elif str(exc) == 'face_not_detected':
            data['count'] = 0
        data['errors'].append(f'detection/landmarks: {type(exc).__name__}: {exc}')
    if data['count'] != 0:
        try:
            data['embedding'] = faces.embedding(image)
        except Exception as exc:
            data['errors'].append(f'ArcFace: {type(exc).__name__}: {exc}')
        if reference_age and selected is not None:
            try:
                value = selected.age
                if value is not None:
                    value = float(value)
                    if not math.isfinite(value) or not 0 <= value <= 100:
                        raise ValueError('Invalid InsightFace reference age')
                    data['age'] = value
            except Exception as exc:
                data['errors'].append(f'InsightFace reference: {type(exc).__name__}: {exc}')
    return data


def _stats(rows):
    count = len(rows)
    known = [r['output_face_count'] for r in rows if r['output_face_count'] is not None]
    detected = sum(n > 0 for n in known)
    times = [r['generation_seconds'] for r in rows if r['generation_seconds'] is not None]
    stats = {
        'count': count, 'generated_count': sum(r['generated'] for r in rows),
        'saved_count': sum(bool(r['output_path']) for r in rows),
        'error_count': sum(bool(r['error']) for r in rows),
        'status_counts': dict(Counter(r['status'] for r in rows)), 'age_mae': None,
        'output_face_detected_count': detected, 'output_no_face_count': sum(n == 0 for n in known),
        'output_detection_unknown_count': count - len(known),
        'output_detection_denominator': count, 'output_detection_evaluated_count': len(known),
        'output_face_detection_rate': detected / count if count else None,
        'output_face_detection_rate_evaluated': detected / len(known) if known else None,
        'output_multiple_face_count': sum(n > 1 for n in known),
        'output_multiple_face_denominator': count,
        'generation_timed_count': len(times), 'generation_seconds_total': sum(times) if times else None,
        'generation_seconds_mean': sum(times) / len(times) if times else None,
    }
    for field in METRICS:
        values = [r[field] for r in rows if r[field] is not None]
        stats[field + '_count'] = len(values)
        stats[field + '_mean'] = sum(values) / len(values) if values else None
    return stats


def _sheets(output, samples, targets, conditions, rows):
    folder = output / 'sheets'
    folder.mkdir()
    indexed = {(r['sample_id'], r['condition'], r['target_age']): r for r in rows}
    paths = []
    for index, sample in enumerate(samples):
        for start in range(0, len(conditions), 5):
            labels = conditions[start:start + 5]
            sheet = Image.new('RGB', (256 * (len(targets) + 1), 256 * len(labels) + 64), 'white')
            draw = ImageDraw.Draw(sheet)
            draw.text((8, 6), f'{sample["sample_id"]} | metadata: {sample["age_group"]}, {sample["gender"]}', fill='black')
            for y, condition in enumerate(labels):
                for x, target in enumerate([None, *targets]):
                    row = indexed[(sample['sample_id'], condition, target)] if target is not None else None
                    path = row['output_path'] if row else sample['input_path']
                    origin = (x * 256, y * 256 + 32)
                    draw.rectangle((*origin, origin[0] + 255, origin[1] + 219), fill='#dddddd')
                    if path:
                        try:
                            image = read_image(output / path)
                            tile = ImageOps.contain(image, (256, 220))
                            sheet.paste(tile, (origin[0] + (256 - tile.width) // 2,
                                               origin[1] + (220 - tile.height) // 2))
                        except Exception as exc:
                            draw.text((origin[0] + 6, origin[1] + 20), f'Image unavailable: {type(exc).__name__}', fill='black')
                    caption = f'{condition}: source' if row is None else f'target {target} | {row["status"]}'
                    draw.text((origin[0] + 5, origin[1] + 223), caption[:45], fill='black')
                    if row and row['identity_cosine'] is not None:
                        draw.text((origin[0] + 5, origin[1] + 238), f'ArcFace cos={row["identity_cosine"]:.4f}', fill='black')
            draw.text((8, sheet.height - 24), 'Observer review required. Geometry cannot establish absence of artifacts.', fill='black')
            path = Path('sheets') / f'{index:04d}_{start // 5 + 1:02d}.png'
            sheet.save(output / path)
            paths.append(path.as_posix())
    return paths


def run_comparison(manifest_path, config_path, output, *, checkpoints=(), baseline=False,
                   insightface_reference=False, pipeline_factory=AgeShiftPipeline, faces=None, clock=None):
    """Run a fixed experiment. Inject pipeline/config-mode factory, faces and clock for CPU tests.

    Production scoring reuses the pipeline's already-loaded FaceTools instance.
    A completed directory contains every planned row, even if all model calls fail.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError('--output must be a new directory')
    manifest = load_manifest(manifest_path)
    config_values = json.loads(Path(config_path).read_text(encoding='utf-8'))
    config = PipelineConfig(**config_values)
    if config.seed != manifest['seed']:
        raise ValueError('Config seed must match the fixed manifest seed')
    assets = {}
    for field in ('base_model', 'image_encoder', 'ip_adapter', 'faceid_lora', 'model_root'):
        value = getattr(config, field)
        if field == 'faceid_lora' and not value:
            continue
        if not value:
            raise ValueError(f'Missing local model path: {field}')
        assets[field] = _asset(value, field in ('base_model', 'image_encoder', 'model_root'))
    config = replace(config, **{key: asset['path'] for key, asset in assets.items()})
    conditions = {'baseline': {'mode': 'faceid', 'checkpoint': None}} if baseline else {}
    for value in checkpoints:
        label, separator, path = value.partition('=')
        if not separator or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', label) or label == 'baseline':
            raise ValueError('Checkpoint must be LABEL=local_path; baseline is reserved')
        if label in conditions:
            raise ValueError(f'Duplicate condition: {label}')
        path = Path(path)
        if path.is_dir():
            path = path / 'pytorch_lora_weights.safetensors'
        conditions[label] = {'mode': 'faceid_age', 'checkpoint': _asset(path)}
    if not conditions:
        raise ValueError('Choose --baseline and/or --checkpoint LABEL=PATH')
    settings = {key: value for key, value in asdict(config).items() if key != 'age_lora'}
    prompts = {gender: {str(age): age_prompt(age, gender) for age in TARGETS} for gender in ('male', 'female')}
    packages = {}
    for name in ('torch', 'diffusers', 'transformers', 'peft', 'onnxruntime', 'onnxruntime-gpu', 'Pillow'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    root = Path(__file__).resolve().parents[1]
    code = {name: sha256(root / name) for name in ('gen/pipeline.py', 'gen/faces.py', 'age/face.py')}
    comparison_key = dict(generation_settings=settings, input_samples=_input_signature(manifest), targets=TARGETS, seed=SEED,
                          prompts=prompts, asset_hashes={key: value['sha256'] for key, value in assets.items()},
                          inference_code_sha256=code, packages=packages, insightface_reference=insightface_reference)
    summary = dict(schema_version=SCHEMA_VERSION, kind='run', status='running', source_split='val',
                   seed=SEED, targets=TARGETS, sample_count=len(manifest['samples']), age_mae=None,
                   manifest_sha256=sha256(manifest_path), config_sha256=sha256(config_path),
                   original_config=config_values, assets=assets, comparison_key=comparison_key,
                   notes=NOTES, conditions={})
    for key in ('HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE'):
        os.environ[key] = '1'
    clock = clock if clock is not None else RunClock(config.device)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'images').mkdir()
    (output / 'manifest.json').write_bytes(Path(manifest_path).read_bytes())
    (output / 'config.json').write_bytes(Path(config_path).read_bytes())
    save_json(output / 'summary.json', summary)
    rows = []
    with (output / 'generation.jsonl').open('w', encoding='utf-8') as log:
        def record(event, **values):
            log.write(json.dumps(dict(event=event, **values), allow_nan=False) + '\n')
            log.flush()

        for label, condition in conditions.items():
            cfg = replace(config, age_lora=condition['checkpoint']['path'] if condition['checkpoint'] else '')
            pipeline, active_faces, load_error, load_seconds = None, None, '', None
            peak = {'max_allocated_bytes': None, 'max_reserved_bytes': None}
            condition_rows = []
            (output / 'images' / label).mkdir()
            try:
                clock.reset_peak()
                pipeline = pipeline_factory(cfg, condition['mode'])
                if getattr(pipeline, 'is_mock', False):
                    raise ValueError('Mock pipelines cannot provide checkpoint comparison metrics')
                _, load_seconds, load_error = _timed(pipeline.load, clock)
                peak = clock.memory()
                if not load_error:
                    active_faces = faces if faces is not None else pipeline._faces
                    if active_faces is None or getattr(active_faces, 'is_mock', False):
                        raise ValueError('Loaded FaceTools is required for reference scoring')
            except Exception as exc:
                load_error = f'{type(exc).__name__}: {exc}'
            record('model_load', condition=label, seconds=load_seconds, error=load_error,
                   config=asdict(cfg), checkpoint=condition['checkpoint'], **peak)
            try:
                for sample_index, sample in enumerate(manifest['samples']):
                    image, source, input_error = None, None, ''
                    if not load_error:
                        try:
                            if sha256(sample['input_path']) != sample['sha256']:
                                raise ValueError('Input SHA-256 changed since manifest validation')
                            image = read_image(sample['input_path'])
                            source = _face_data(image, active_faces, insightface_reference)
                        except Exception as exc:
                            input_error = f'{type(exc).__name__}: {exc}'
                    for target in TARGETS:
                        row = dict(condition=label, mode=condition['mode'], sample_id=sample['sample_id'],
                                   source_path=sample['path'], input_path=sample['input_path'],
                                   age_group=sample['age_group'], gender=sample['gender'], input_sha256=sample['sha256'],
                                   target_age=target, seed=SEED, prompt=age_prompt(target, sample['gender']),
                                   status='ok', error='', generated=False, output_path='', output_sha256='',
                                   generation_seconds=None, max_allocated_bytes=None, max_reserved_bytes=None,
                                   source_face_count=source['count'] if source else None, output_face_count=None,
                                   source_insightface_age_reference=source['age'] if source else None,
                                   estimated_age=None, absolute_error=None, **{field: None for field in METRICS})
                        if load_error or input_error:
                            row.update(status='load_error' if load_error else 'input_error', error=load_error or input_error)
                        else:
                            clock.reset_peak()
                            generated, seconds, error = _timed(
                                lambda: pipeline.age_shift(image.copy(), [target], gender=sample['gender'], seed=SEED), clock)
                            row['generation_seconds'] = seconds
                            row.update(clock.memory())
                            for key in peak:
                                values = [v for v in (peak[key], row[key]) if v is not None]
                                peak[key] = max(values) if values else None
                            try:
                                if error:
                                    raise ValueError(error)
                                if not isinstance(generated, (list, tuple)) or len(generated) != 1:
                                    raise ValueError('Expected exactly one generated image')
                                result = as_rgb(generated[0])
                                row['generated'] = True
                            except Exception as exc:
                                row.update(status='generation_error', error=f'{type(exc).__name__}: {exc}')
                            if row['generated']:
                                errors = [f'source: {e}' for e in source['errors']]
                                path = Path('images') / label / f'{sample_index:04d}_age_{target}.png'
                                try:
                                    result.save(output / path)
                                    row.update(output_path=path.as_posix(), output_sha256=sha256(output / path))
                                except Exception as exc:
                                    errors.append(f'save: {type(exc).__name__}: {exc}')
                                    row['status'] = 'save_error'
                                measured = _face_data(result, active_faces, insightface_reference)
                                row.update(output_face_count=measured['count'], insightface_age_reference=measured['age'])
                                row.update({f'{key}_reference': value for key, value in measured['geometry'].items()})
                                errors.extend(f'output: {e}' for e in measured['errors'])
                                try:
                                    if source['embedding'] is not None and measured['embedding'] is not None:
                                        row['identity_cosine'] = cosine_similarity(source['embedding'], measured['embedding'])
                                except Exception as exc:
                                    errors.append(f'identity: {type(exc).__name__}: {exc}')
                                try:
                                    if source['points'] is not None and measured['points'] is not None:
                                        row['landmark_procrustes_reference'] = procrustes_error(source['points'], measured['points'])
                                except Exception as exc:
                                    errors.append(f'geometry: {type(exc).__name__}: {exc}')
                                if errors and row['status'] == 'ok':
                                    row['status'] = 'metric_error'
                                row['error'] = '; '.join(errors)
                        condition_rows.append(row)
                        rows.append(row)
                        record('generation', **row)
            finally:
                if pipeline is not None:
                    pipeline.close()
            summary['conditions'][label] = dict(**condition, config=asdict(cfg), load_seconds=load_seconds,
                                               load_error=load_error, **peak, **_stats(condition_rows),
                                               per_target={str(age): _stats([r for r in condition_rows if r['target_age'] == age])
                                                           for age in TARGETS})
    write_csv(output / 'results.csv', rows)
    summary['results_sha256'] = sha256(output / 'results.csv')
    summary['sheets'] = _sheets(output, manifest['samples'], TARGETS, list(conditions), rows)
    summary.update(status='completed', count=len(rows))
    save_json(output / 'summary.json', summary)
    return summary


def _read_results(directory, summary):
    path = directory / 'results.csv'
    if sha256(path) != summary['results_sha256']:
        raise ValueError(f'Results checksum mismatch: {directory}')
    rows = read_csv(path)
    integers = ('target_age', 'seed', 'source_face_count', 'output_face_count', 'max_allocated_bytes', 'max_reserved_bytes')
    for row in rows:
        row['generated'] = {'True': True, 'False': False}[row['generated']]
        for key in integers:
            row[key] = int(row[key]) if row[key] else None
        for key in (*METRICS, 'source_insightface_age_reference', 'generation_seconds', 'estimated_age', 'absolute_error'):
            row[key] = float(row[key]) if row[key] else None
            if row[key] is not None and not math.isfinite(row[key]):
                raise ValueError(f'Nonfinite metric in {directory}')
        if row['estimated_age'] is not None or row['absolute_error'] is not None:
            raise ValueError('Reference comparison must not contain official age metrics')
        if row['output_path']:
            image = (directory / row['output_path']).resolve()
            if not image.is_relative_to(directory.resolve()) or sha256(image) != row['output_sha256']:
                raise ValueError(f'Output image checksum/path mismatch: {image}')
            row['output_path'] = str(image)
    samples = {s['sample_id']: s for s in summary['comparison_key']['input_samples']}
    expected = {(label, sample, age) for label in summary['conditions'] for sample in samples for age in summary['targets']}
    actual = {(r['condition'], r['sample_id'], r['target_age']) for r in rows}
    if expected != actual or len(rows) != len(expected):
        raise ValueError(f'Incomplete or duplicate result rows: {directory}')
    for row in rows:
        sample = samples[row['sample_id']]
        if (row['input_sha256'] != sample['sha256'] or row['gender'] != sample['gender'] or
                row['age_group'] != sample['age_group'] or row['source_path'] != sample['path'] or
                row['seed'] != summary['seed'] or
                row['prompt'] != summary['comparison_key']['prompts'][sample['gender']][str(row['target_age'])]):
            raise ValueError(f'Row does not match fixed experiment: {directory}')
    return rows


def combine_results(directories, output):
    """CPU-only aggregation; no inference or retroactive removal of failed rows."""
    output = Path(output)
    if output.exists():
        raise FileExistsError('--output must be a new directory')
    rows, conditions, sources, key, manifest = [], {}, [], None, None
    for directory in map(Path, directories):
        summary = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
        if summary.get('schema_version') != SCHEMA_VERSION or summary.get('kind') != 'run' or summary.get('status') != 'completed':
            raise ValueError(f'Not a completed comparison run: {directory}')
        if key is not None and summary['comparison_key'] != key:
            raise ValueError('Cannot combine differing generation settings, inputs/hashes, seed, targets, prompts or runtimes')
        if set(conditions) & set(summary['conditions']):
            raise ValueError('Duplicate conditions cannot be combined')
        for name in ('manifest', 'config'):
            if sha256(directory / f'{name}.json') != summary[f'{name}_sha256']:
                raise ValueError(f'Saved {name} checksum mismatch: {directory}')
        current_manifest = load_manifest(directory / 'manifest.json')
        if (_input_signature(current_manifest) != summary['comparison_key']['input_samples'] or
                summary['seed'] != SEED or summary['targets'] != TARGETS or
                summary['comparison_key']['seed'] != SEED or summary['comparison_key']['targets'] != TARGETS):
            raise ValueError('Saved manifest does not match fixed experiment')
        if key is None:
            key, manifest = summary['comparison_key'], current_manifest
        rows.extend(_read_results(directory, summary))
        conditions.update(summary['conditions'])
        sources.append({'directory': str(directory.resolve()), 'summary_sha256': sha256(directory / 'summary.json'),
                        'manifest_sha256': summary['manifest_sha256'], 'config_sha256': summary['config_sha256']})
    if key is None:
        raise ValueError('At least one completed directory is required')
    samples = manifest['samples']
    indexed = {(r['condition'], r['sample_id'], r['target_age']): r for r in rows}
    paired = []
    if 'baseline' in conditions:
        for label in conditions:
            if label == 'baseline':
                continue
            for target in [*key['targets'], 'all']:
                ages = key['targets'] if target == 'all' else [target]
                pairs = [(indexed['baseline', sample['sample_id'], age], indexed[label, sample['sample_id'], age])
                         for sample in samples for age in ages]
                entry = dict(condition=label, baseline='baseline', target_age=target, expected_pairs=len(pairs),
                             pairs_with_row_errors=sum(bool(a['error'] or b['error']) for a, b in pairs), age_mae=None)
                for field in ('identity_cosine', 'insightface_age_reference'):
                    valid = [(a[field], b[field]) for a, b in pairs if a[field] is not None and b[field] is not None]
                    entry.update({field + '_paired_count': len(valid), field + '_unpaired_count': len(pairs) - len(valid),
                                  field + '_baseline_mean': sum(a for a, b in valid) / len(valid) if valid else None,
                                  field + '_condition_mean': sum(b for a, b in valid) / len(valid) if valid else None,
                                  field + '_difference_mean': sum(b - a for a, b in valid) / len(valid) if valid else None})
                paired.append(entry)
    wide = []
    for sample in samples:
        for label in conditions:
            row = dict(sample_id=sample['sample_id'], condition=label, age_group=sample['age_group'],
                       gender=sample['gender'], input_path=sample['input_path'], input_sha256=sample['sha256'])
            for age in key['targets']:
                result = indexed[label, sample['sample_id'], age]
                row.update({f'age_{age}_{field}': result[field] for field in
                            ('status', 'error', 'output_path', 'identity_cosine', 'insightface_age_reference',
                             'landmark_procrustes_reference', 'generation_seconds')})
            wide.append(row)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / 'results.csv', rows)
    write_csv(output / 'per_input.csv', wide)
    write_csv(output / 'comparison.csv', paired, fields=list(paired[0]) if paired else
              ['condition', 'baseline', 'target_age', 'expected_pairs', 'pairs_with_row_errors', 'age_mae'])
    save_json(output / 'comparison.json', {'difference_definition': 'condition minus baseline, matched available pairs only',
                                         'age_mae': None, 'rows': paired, 'notes': NOTES})
    summary = dict(schema_version=SCHEMA_VERSION, kind='combined', status='completed', source_split='val',
                   seed=key['seed'], targets=key['targets'], sample_count=len(samples), count=len(rows), age_mae=None,
                   comparison_key=key, sources=sources, conditions=conditions, notes=NOTES,
                   results_sha256=sha256(output / 'results.csv'),
                   sheets=_sheets(output, samples, key['targets'], list(conditions), rows))
    save_json(output / 'summary.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, help='Fixed preflight validation manifest JSON')
    parser.add_argument('--config', type=Path, help='Original PipelineConfig JSON; settings are not tuned or overridden')
    parser.add_argument('--checkpoint', action='append', default=[], metavar='LABEL=PATH', help='Local LoRA file or checkpoint directory')
    parser.add_argument('--baseline', action='store_true', help='FaceID only, without Age LoRA')
    parser.add_argument('--insightface-reference', action='store_true', help='Record optional InsightFace age, never MAE')
    parser.add_argument('--combine', nargs='+', type=Path, metavar='DIR', help='Combine completed runs without loading models')
    parser.add_argument('--output', type=Path, required=True, help='New directory, never overwrite')
    args = parser.parse_args(argv)
    if args.combine:
        if args.manifest or args.config or args.checkpoint or args.baseline or args.insightface_reference:
            parser.error('--combine cannot be mixed with generation arguments')
    elif not args.manifest or not args.config or not (args.baseline or args.checkpoint):
        parser.error('Generation requires --manifest, --config and --baseline and/or --checkpoint')
    try:
        summary = (combine_results(args.combine, args.output) if args.combine else
                   run_comparison(args.manifest, args.config, args.output, checkpoints=args.checkpoint,
                                  baseline=args.baseline, insightface_reference=args.insightface_reference))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({'status': summary['status'], 'count': summary['count'],
                      'conditions': list(summary['conditions']), 'output': str(args.output), 'age_mae': None}))


if __name__ == '__main__':
    main()
