import argparse
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
from .common import MEAN, STD, target_ages
from .face import FaceAligner, read_rgb


class AgeEstimator:
    def __init__(self, onnx_path, model_root='models/insightface', device='cpu', allow_smoke=False):
        metadata_path = Path(onnx_path).with_suffix('.json')
        if metadata_path.exists() and json.loads(metadata_path.read_text(encoding='utf-8')).get('smoke_artifact') and not allow_smoke:
            raise ValueError('This ONNX is an untrained smoke artifact. It cannot estimate actual ages.')
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.aligner = FaceAligner(model_root, device)

    def estimate(self, rgb):
        aligned, face = self.aligner.align(rgb, training=False)
        images = np.stack([aligned, aligned[:, ::-1]]) / np.float32(255)
        images = ((images - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32))
        images = np.ascontiguousarray(images.transpose(0, 3, 1, 2), dtype=np.float32)
        age = float(self.session.run(None, {self.session.get_inputs()[0].name: images})[0].mean())
        if not np.isfinite(age) or not 0 <= age <= 100:
            raise ValueError('Invalid age prediction')
        return dict(age=age, gender='male' if face.gender == 1 else 'female', targets=target_ages(age))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--onnx', required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--model-root', default='models/insightface')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--allow-smoke', action='store_true', help='Engineering verification only, not actual age estimates')
    args = p.parse_args()
    print(json.dumps(AgeEstimator(args.onnx, args.model_root, args.device, args.allow_smoke).estimate(read_rgb(args.image))))


if __name__ == '__main__':
    main()
