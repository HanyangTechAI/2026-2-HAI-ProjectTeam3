"""Sequential three-mode evaluation. Mock runs deliberately have no model metrics."""
import argparse
from collections import Counter
from dataclasses import asdict, is_dataclass
import json
import math
from pathlib import Path
from time import perf_counter

from age.common import read_csv, save_json, write_csv
from . import FaceNotFoundError, GenerationError
from .booth import make_age_estimator, validate_age
from .pipeline import MODES, add_pipeline_arguments, as_rgb, pipeline_from_args, read_image


def cosine_similarity(left, right):
    left, right = [float(x) for x in left], [float(x) for x in right]
    if not left or len(left) != len(right) or not all(math.isfinite(x) for x in left + right):
        raise ValueError('Embeddings must be finite, nonempty vectors of equal size')
    left_norm, right_norm = math.hypot(*left), math.hypot(*right)
    if not left_norm or not right_norm or not all(map(math.isfinite, (left_norm, right_norm))):
        raise ValueError('Embedding norms must be finite and nonzero')
    score = math.fsum((x / left_norm) * (y / right_norm) for x, y in zip(left, right))
    return max(-1.0, min(1.0, score))


def _evaluate_target(image, target, pipeline, age_estimator, identity_encoder, seed):
    row = {'status': 'ok', 'estimated_age': None, 'absolute_error': None,
           'identity_similarity': None, 'face_failure': None, 'elapsed_seconds': None,
           'is_mock': bool(pipeline.is_mock), 'error': '', 'output_path': ''}
    generated = None
    started = perf_counter()
    try:
        results = pipeline.age_shift(image.copy(), [target], seed=seed)
        if len(results) != 1:
            raise GenerationError('Expected exactly one generated frame')
        generated = as_rgb(results[0]).copy()
    except (ImportError, FileNotFoundError):
        raise
    except Exception as exc:
        face_failure = isinstance(exc, FaceNotFoundError) or str(exc) == 'face_not_detected'
        row.update(status='face_not_detected' if face_failure else 'inference_error',
                   face_failure=True if face_failure and not pipeline.is_mock else None,
                   error=f'inference: {type(exc).__name__}: {exc}')
    finally:
        row['elapsed_seconds'] = perf_counter() - started
    if generated is None:
        return row, None
    if pipeline.is_mock:
        row['status'] = 'mock'
        return row, generated

    metrics = (
        ('estimated_age', lambda: validate_age(age_estimator(generated.copy())['age'])),
        ('identity_similarity', lambda: cosine_similarity(identity_encoder(image.copy()),
                                                          identity_encoder(generated.copy()))),
    )
    errors = []
    for field, metric in metrics:
        try:
            row[field] = metric()
            if row['face_failure'] is None:
                row['face_failure'] = False
        except (ImportError, FileNotFoundError):
            raise
        except Exception as exc:
            face_failure = isinstance(exc, FaceNotFoundError) or str(exc) == 'face_not_detected'
            if face_failure:
                row.update(face_failure=True, status='face_not_detected')
            elif row['status'] != 'face_not_detected':
                row['status'] = 'metric_error'
            errors.append(f'{field}: {type(exc).__name__}: {exc}')
    if row['estimated_age'] is not None:
        row['absolute_error'] = abs(row['estimated_age'] - target)
    row['error'] = '; '.join(errors)
    return row, generated


def _evaluator_id(evaluator):
    if evaluator is None:
        return None
    return getattr(evaluator, 'evaluator_id',
                   f'{evaluator.__module__}.{getattr(evaluator, "__qualname__", type(evaluator).__qualname__)}')


def evaluate(samples, targets, pipeline_factory, age_estimator, identity_encoder, output, seed=42):
    """Write results.csv/summary.json into a new directory and return the summary.

    The factory receives each mode in MODES, sequentially. It must keep all other
    settings fixed. Fatal dependency/model errors propagate; individual image
    and metric failures are recorded without substituting zero scores.
    """
    samples = [(str(name), as_rgb(image).copy()) for name, image in samples]
    targets = list(targets)
    if not samples or not targets:
        raise ValueError('At least one sample and target are required')
    if any(type(target) is not int or not 0 <= target <= 100 for target in targets):
        raise ValueError('Targets must be integers in [0, 100]')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows, modes = [], {}
    for mode in MODES:
        pipeline = pipeline_factory(mode)
        mode_rows = []
        try:
            if not pipeline.is_mock:
                if age_estimator is None or identity_encoder is None:
                    raise ValueError('Real evaluation requires age and identity evaluators')
                if getattr(age_estimator, 'is_mock', False) or getattr(identity_encoder, 'is_mock', False):
                    raise ValueError('Mock evaluators cannot supply real model metrics')
            started = perf_counter()
            pipeline.load()
            load_seconds = perf_counter() - started
            (output / mode).mkdir()
            for sample_index, (sample, image) in enumerate(samples):
                for target_index, target in enumerate(targets):
                    result, generated = _evaluate_target(image, target, pipeline, age_estimator,
                                                         identity_encoder, seed)
                    row = dict(mode=mode, sample=sample, target=target, seed=seed, **result)
                    if generated is not None:
                        path = Path(mode) / f'{sample_index:04d}_{target_index:02d}.png'
                        generated.save(output / path)
                        row['output_path'] = path.as_posix()
                    mode_rows.append(row)
            age_errors = [r['absolute_error'] for r in mode_rows if r['absolute_error'] is not None]
            similarities = [r['identity_similarity'] for r in mode_rows if r['identity_similarity'] is not None]
            detections = [r['face_failure'] for r in mode_rows if r['face_failure'] is not None]
            elapsed = sum(r['elapsed_seconds'] for r in mode_rows)
            config = getattr(pipeline, 'config', None)
            modes[mode] = {
                'count': len(mode_rows), 'generated_count': sum(bool(r['output_path']) for r in mode_rows),
                'status_counts': dict(Counter(r['status'] for r in mode_rows)),
                'age_scored_count': len(age_errors), 'identity_scored_count': len(similarities),
                'age_mae': sum(age_errors) / len(age_errors) if age_errors else None,
                'identity_cosine_mean': sum(similarities) / len(similarities) if similarities else None,
                'face_failure_count': sum(detections) if detections else None,
                'face_failure_denominator': len(detections),
                'face_failure_rate': sum(detections) / len(detections) if detections else None,
                'load_seconds': load_seconds, 'inference_seconds': elapsed,
                'mean_inference_seconds': elapsed / len(mode_rows),
                'is_mock': bool(pipeline.is_mock),
                'timing_kind': 'mock' if pipeline.is_mock else 'model_run',
                'settings': asdict(config) if is_dataclass(config) else None,
            }
            rows.extend(mode_rows)
        finally:
            pipeline.close()
    is_mock = any(m['is_mock'] for m in modes.values())
    summary = {
        'seed': seed, 'targets': targets, 'sample_count': len(samples), 'count': len(rows),
        'is_mock': is_mock, 'modes': modes,
        'metric_evaluators': {'age': _evaluator_id(age_estimator), 'identity': _evaluator_id(identity_encoder)},
        'metrics_note': ('Mock modes skip all metric evaluators; null/blank means unavailable, not zero. '
                         'Model metrics require independent cross-checks and visual review.'),
        'face_failure_denominator_definition': (
            'Per-target attempts with an explicit face-not-detected error (input or output), '
            'or at least one successful face-based metric; excludes mocks and unknown error outcomes.'),
        'timing_note': 'Load and inference wall times are separate; inference excludes scoring and image saving. '
                       'Mock timings are not model latency measurements.',
    }
    write_csv(output / 'results.csv', rows)
    save_json(output / 'summary.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__ + ' Always runs all three modes, regardless of --mode.')
    add_pipeline_arguments(parser)
    parser.add_argument('--csv', type=Path, required=True, help='Manifest with path and optional sample_id columns')
    parser.add_argument('--targets', type=int, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True, help='New directory, must not exist')
    parser.add_argument('--age-onnx', type=Path, help='Trained age ONNX, required without --mock')
    args = parser.parse_args(argv)
    if not args.mock and args.age_onnx is None:
        parser.error('--age-onnx is required without --mock')
    if args.output.exists():
        parser.error('--output must be a new directory; choose a new path for each run')
    samples = []
    for row in read_csv(args.csv):
        path = Path(row['path'])
        path = path if path.is_absolute() else args.csv.parent / path
        samples.append((row.get('sample_id') or row['path'], read_image(path)))

    def factory(mode):
        return pipeline_from_args(argparse.Namespace(**(vars(args) | {'mode': mode})))

    age_estimator = identity_encoder = None
    if not args.mock:
        from .faces import FaceTools
        template = factory(MODES[0])
        config = template.config
        template.close()
        age_estimator = make_age_estimator(args.age_onnx, config.model_root, config.device)
        faces = FaceTools(config.model_root, config.device, identity=True)

        def identity_encoder(image):
            return faces.embedding(image)

        identity_encoder.evaluator_id = f'gen.faces.FaceTools.embedding:{Path(config.model_root).resolve()}'

    summary = evaluate(samples, args.targets, factory, age_estimator, identity_encoder, args.output, args.seed)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
