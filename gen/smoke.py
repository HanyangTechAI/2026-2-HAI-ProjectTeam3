"""Pillow-only, synthetic mock integration check. No trained models are validated."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from age.common import save_json
from .booth import MockAgeEstimator, run_booth, save_booth
from .eval import evaluate
from .pipeline import MODES, MockPipeline


def run_smoke(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    image = Image.new('RGB', (512, 512), (220, 230, 240))
    draw = ImageDraw.Draw(image)
    draw.ellipse((150, 80, 360, 330), fill=(180, 160, 140))
    draw.rectangle((100, 340, 410, 511), fill=(80, 110, 140))
    draw.text((16, 20), 'SYNTHETIC MOCK INPUT - NOT A REAL PERSON', fill='black')
    image.save(output / 'synthetic.png')
    pipeline = MockPipeline(seed=42)
    try:
        booth = run_booth(image, MockAgeEstimator(25), pipeline, seed=42)
        save_booth(booth, output / 'booth')
    finally:
        pipeline.close()
    targets = [booth['metadata']['targets'][slot] for slot in (0, 2, 3)]
    summary = evaluate([('synthetic', image)], targets,
                       lambda mode: MockPipeline(mode=mode, seed=42),
                       None, None, output / 'evaluation', seed=42)
    if (booth['frames'][1].tobytes() != image.tobytes()
            or any(summary['modes'][mode]['status_counts'] != {'mock': 3} for mode in MODES)):
        raise RuntimeError('Mock structural checks failed')
    status = {
        'status': 'mock_structure_passed', 'is_mock': True,
        'validation': 'structure_only', 'trained_models': False, 'real_model_validation': False,
        'dataset_used': False, 'modes': list(MODES), 'evaluation_rows': summary['count'],
        'booth_metadata': 'booth/metadata.json', 'evaluation_summary': 'evaluation/summary.json',
        'note': 'Only orchestration, image/CSV/JSON output and mock ablations were exercised. '
                'No age accuracy, identity preservation, face detection or GPU latency was validated.',
    }
    save_json(output / 'status.json', status)
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/gen_smoke'),
                        help='New directory; choose a different path for every rerun')
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('--output must be a new directory; choose a new path for each run')
    print(json.dumps(run_smoke(args.output), indent=2))


if __name__ == '__main__':
    main()
