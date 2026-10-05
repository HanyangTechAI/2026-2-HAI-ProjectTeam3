"""Synthetic engineering verification ONLY; never a performance experiment."""
import argparse
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image
from .common import ROOT, FIELDS, write_csv, save_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/smoke')
    p.add_argument('--pretrained', action='store_true', help='Verify the cached ImageNet initialization instead of random weights')
    args = p.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'checkpoint').exists():
        raise FileExistsError('Use a fresh --output for a new smoke run')
    data = output / 'data'
    images = data / 'images'
    images.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for split, count in [('train', 8), ('val', 4), ('test', 4)]:
        rows = []
        for i in range(count):
            path = images / f'{split}_{i}.png'
            Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)).save(path)
            rows.append(dict(path=str(path), age=(i % 5) * 25, gender='unknown', source='synthetic',
                             person_id='', original_id=path.name, sha256=f'synthetic_{split}_{i}'))
        write_csv(data / f'{split}.csv', rows, FIELDS)
    subprocess.run([sys.executable, '-m', 'age.train', '--stage', 'aihub', '--data-dir', str(data), '--output', str(output / 'checkpoint'),
                    '--epochs', '1', '--batch-size', '4', '--device', 'cpu', '--smoke-steps', '2'] +
                   ([] if args.pretrained else ['--no-pretrained']),
                   check=True, cwd=ROOT)
    subprocess.run([sys.executable, '-m', 'age.export_onnx', '--checkpoint', str(output / 'checkpoint/best.pt'),
                    '--csv', str(data / 'val.csv'), '--output', str(output / 'smoke_only.onnx'), '--allow-smoke'],
                   check=True, cwd=ROOT)
    save_json(output / 'status.json', dict(purpose='synthetic engineering verification only',
              trained_age_model=False, data='random RGB arrays, not faces', verified='training loop and ONNX numerical parity'))


if __name__ == '__main__':
    main()
