import argparse
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader
from .common import MEAN, STD, save_json, sha256
from .dataset import AgeDataset
from .model import build_model, AgeOutput


def main():
    p = argparse.ArgumentParser(description='Export expected-age output, opset 17; verify on actual aligned validation faces.')
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--csv', type=Path, required=True, help='Aligned validation CSV, not test')
    p.add_argument('--output', type=Path, default=Path('outputs/age/age_estimator.onnx'))
    p.add_argument('--samples', type=int, default=20)
    p.add_argument('--allow-smoke', action='store_true', help='Only export a smoke artifact for pipeline verification')
    args = p.parse_args()
    if args.samples < 1:
        p.error('--samples must be positive')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    smoke = bool(checkpoint['run'].get('smoke_steps'))
    if smoke and not args.allow_smoke:
        raise ValueError('Smoke checkpoint is not a trained model; use --allow-smoke for validation only.')
    model = build_model(False)
    model.load_state_dict(checkpoint['model'])
    model = AgeOutput(model).eval()
    torch.set_num_threads(min(8, torch.get_num_threads()))
    ds = AgeDataset(args.csv)
    ds.rows = ds.rows[:args.samples]
    images = next(iter(DataLoader(ds, batch_size=min(4, len(ds)))))[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.pending.onnx')
    torch.onnx.export(model, images, str(temporary), opset_version=17, dynamo=False,
                      input_names=['images'], output_names=['age'],
                      dynamic_axes={'images': {0: 'batch'}, 'age': {0: 'batch'}})
    onnx.checker.check_model(onnx.load(str(temporary)))
    session = ort.InferenceSession(str(temporary), providers=['CPUExecutionProvider'])
    maximum = 0.0
    with torch.inference_mode():
        for images, _, _ in DataLoader(ds, batch_size=4):
            actual = session.run(['age'], {'images': images.numpy()})[0]
            expected = model(images).numpy()
            maximum = max(maximum, float(np.max(np.abs(actual - expected))))
            if not np.isfinite(actual).all() or maximum >= 0.01:
                raise AssertionError(f'ONNX mismatch: max age difference={maximum}')
    temporary.replace(args.output)
    save_json(args.output.with_suffix('.json'), dict(input='RGB float32 NCHW, [N,3,224,224]',
              range_before_normalization='0..1', mean=MEAN, std=STD, output='float32 age [N]', opset=17,
              face_alignment='buffalo_l SCRFD / ArcFace-112 scaled to 224; same as age.face.FaceAligner',
              checkpoint_sha256=sha256(args.checkpoint), validation_sha256=sha256(args.csv),
              onnx_sha256=sha256(args.output), verified_images=len(ds), max_absolute_error=maximum,
              smoke_artifact=smoke, training_run=checkpoint['run']))
    print(f'Export verified: {len(ds)} images, max difference {maximum:.8f} years')


if __name__ == '__main__':
    main()
