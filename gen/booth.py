"""CLI-only four-cut orchestration; mocks are not age or face predictions."""
import argparse
import math
from pathlib import Path
from time import perf_counter

from PIL import Image, ImageOps

from age.common import save_json, target_ages
from . import FaceNotFoundError, GenerationError
from .pipeline import add_pipeline_arguments, as_rgb, pipeline_from_args, read_image


def validate_age(value):
    if isinstance(value, (bool, str)):
        raise ValueError('Age must be a finite number in [0, 100]')
    try:
        age = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Age must be a finite number in [0, 100]') from exc
    if not math.isfinite(age) or not 0 <= age <= 100:
        raise ValueError('Age must be a finite number in [0, 100]')
    return age


class MockAgeEstimator:
    """Return a fixed, explicitly mock age without examining the image."""

    is_mock = True
    evaluator_id = 'gen.booth.MockAgeEstimator (fixed age, not a prediction)'

    def __init__(self, age=25):
        self.age = validate_age(age)

    def __call__(self, image):
        return {'age': self.age, 'gender': 'unknown', 'is_mock': True}


def make_age_estimator(onnx_path, model_root='models/insightface', device='cpu'):
    """Adapt the existing numpy age API only when real inference is requested."""
    if not Path(onnx_path).is_file():
        raise FileNotFoundError(f'Age ONNX not found: {onnx_path}')
    import numpy as np
    from age.infer import AgeEstimator

    estimator = AgeEstimator(onnx_path, model_root, device)

    def estimate(image):
        try:
            return estimator.estimate(np.asarray(as_rgb(image)).copy())
        except ValueError as exc:
            if str(exc) == 'face_not_detected':
                raise FaceNotFoundError('face_not_detected') from exc
            raise

    estimate.is_mock = False
    estimate.evaluator_id = f'age.infer.AgeEstimator:{Path(onnx_path).resolve()}'
    return estimate


def run_booth(image, age_estimator, generator, seed=42):
    """Return frames and metadata. The caller owns the generator's lifetime."""
    original = as_rgb(image).copy()
    started = perf_counter()
    estimate = age_estimator(original.copy())
    age_seconds = perf_counter() - started
    age = validate_age(estimate['age'])
    gender = estimate.get('gender', 'unknown')
    targets = target_ages(age)
    generated_slots = [0, 2, 3]
    started = perf_counter()
    generator.load()
    load_seconds = perf_counter() - started
    started = perf_counter()
    generated = generator.age_shift(original.copy(), [targets[i] for i in generated_slots],
                                    gender=gender, seed=seed)
    inference_seconds = perf_counter() - started
    if len(generated) != 3:
        raise GenerationError('Expected exactly three generated frames')
    frames = [as_rgb(generated[0]).copy(), original.copy(),
              as_rgb(generated[1]).copy(), as_rgb(generated[2]).copy()]
    age_mock = bool(getattr(age_estimator, 'is_mock', False) or estimate.get('is_mock', False))
    generation_mock = bool(generator.is_mock)
    metadata = {
        'age': age, 'gender': gender, 'targets': targets,
        'generated_slots': generated_slots, 'current_slot': 1,
        'seed': seed, 'mode': generator.mode,
        'is_mock': age_mock or generation_mock,
        'age_is_mock': age_mock, 'generation_is_mock': generation_mock,
        'timing_kind': 'mock' if age_mock or generation_mock else 'model_run',
        'age_estimation_seconds': age_seconds, 'load_seconds': load_seconds,
        'inference_seconds': inference_seconds,
        'note': ('Mock structure only; no age-editing or model-performance validation.'
                 if age_mock or generation_mock else
                 'Single run timings; not a performance benchmark or quality validation.'),
    }
    return {'frames': frames, 'metadata': metadata}


def compose_four_cut(frames, cell_size=(512, 512)):
    """Compose a 2x2 sheet, letterboxing without cropping or changing originals."""
    frames = list(frames)
    if len(frames) != 4:
        raise ValueError('Exactly four frames are required')
    if len(cell_size) != 2 or any(type(n) is not int or n <= 0 for n in cell_size):
        raise ValueError('cell_size must contain two positive integers')
    width, height = cell_size
    sheet = Image.new('RGB', (2 * width, 2 * height), 'white')
    for slot, frame in enumerate(frames):
        thumb = ImageOps.contain(as_rgb(frame), cell_size)
        x = (slot % 2) * width + (width - thumb.width) // 2
        y = (slot // 2) * height + (height - thumb.height) // 2
        sheet.paste(thumb, (x, y))
    return sheet


def save_booth(result, output):
    """Save a complete booth result in a new directory; never overwrite a run."""
    output = Path(output)
    sheet = compose_four_cut(result['frames'])
    output.mkdir(parents=True, exist_ok=False)
    names = ['01_past.png', '02_current.png', '03_future.png', '04_later.png']
    for name, frame in zip(names, result['frames']):
        frame.save(output / name)
    sheet.save(output / 'four_cut.png')
    metadata = dict(result['metadata'], frame_paths=names, composition_path='four_cut.png')
    save_json(output / 'metadata.json', metadata)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_pipeline_arguments(parser)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New directory, must not exist')
    parser.add_argument('--age-onnx', type=Path, help='Trained age ONNX, required without --mock')
    parser.add_argument('--mock-age', type=float, default=25, help='Fixed mock age, not a prediction')
    args = parser.parse_args(argv)
    if not args.mock and args.age_onnx is None:
        parser.error('--age-onnx is required without --mock')
    if args.output.exists():
        parser.error('--output must be a new directory; choose a new path for each run')
    generator = pipeline_from_args(args)
    try:
        estimator = (MockAgeEstimator(args.mock_age) if args.mock else
                     make_age_estimator(args.age_onnx, generator.config.model_root, generator.config.device))
        save_booth(run_booth(read_image(args.image), estimator, generator, args.seed), args.output)
    finally:
        generator.close()
    print(f'Saved {"MOCK structure-only" if args.mock else "model-run"} booth: {args.output}')


if __name__ == '__main__':
    main()
