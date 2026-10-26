"""Reuse the age alignment contract and the official buffalo_l ArcFace runtime."""
from pathlib import Path

from . import FaceNotFoundError, ModelUnavailableError


class FaceTools:
    def __init__(self, model_root='models/insightface', device='cpu', identity=True):
        root = Path(model_root)
        required = ['python_runtime/__init__.py', 'models/buffalo_l/det_10g.onnx', 'models/buffalo_l/genderage.onnx']
        if identity:
            required += ['python_runtime/model_zoo/arcface_onnx.py', 'models/buffalo_l/w600k_r50.onnx']
        for relative in required:
            if not (root / relative).is_file():
                raise ModelUnavailableError(f'Missing {root / relative}; run age.setup_models --with-arcface first')
        import onnxruntime as ort
        from age.face import FaceAligner, runtime_module
        self.aligner = FaceAligner(root, device)
        self.recognizer = None
        if identity:
            path = root / 'models/buffalo_l/w600k_r50.onnx'
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
            session = ort.InferenceSession(str(path), providers=providers)
            cls = runtime_module(root, 'model_zoo.arcface_onnx').ArcFaceONNX
            self.recognizer = cls(model_file=str(path), session=session)
            self.recognizer.prepare(ctx_id=0 if device == 'cuda' else -1)

    def _select(self, image):
        import numpy as np
        from .pipeline import as_rgb
        rgb = np.asarray(as_rgb(image))
        try:
            return rgb, self.aligner.select(rgb, training=False)
        except ValueError as e:
            if str(e) == 'face_not_detected':
                raise FaceNotFoundError('face_not_detected') from e
            raise

    def prepare(self, image):
        from PIL import Image
        rgb, face = self._select(image)
        crop = self.aligner.crop(rgb, face, size=512, context=True)
        return Image.fromarray(crop), 'male' if face.gender == 1 else 'female'

    def embedding(self, image):
        import numpy as np
        if self.recognizer is None:
            raise ModelUnavailableError('ArcFace was not loaded for this instance')
        rgb, face = self._select(image)
        embedding = np.asarray(self.recognizer.get(np.ascontiguousarray(rgb[:, :, ::-1]), face)).reshape(-1)
        norm = float(np.linalg.norm(embedding))
        if embedding.size != 512 or not np.isfinite(embedding).all() or not np.isfinite(norm) or norm <= 0:
            raise ValueError('Invalid ArcFace embedding')
        return (embedding / norm).tolist()
