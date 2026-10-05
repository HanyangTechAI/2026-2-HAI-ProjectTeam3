"""Shared contract: RGB uint8 in/out; training rejects multiple faces.

224px ArcFace similarity alignment is shared by preparation and inference.
512px uses a wider version of the same transform to retain hair/context.
No detected face, invalid landmarks, or ambiguous training face raises ValueError.
"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import importlib
import importlib.util
import sys


def runtime_module(root, name):
    runtime = Path(root).resolve() / 'python_runtime'
    package = '_age_insightface_v07'
    if not (runtime / '__init__.py').is_file():
        raise FileNotFoundError('Run python -m age.setup_models to prepare official buffalo_l runtime.')
    if package not in sys.modules:
        spec = importlib.util.spec_from_file_location(package, runtime / '__init__.py',
                                                       submodule_search_locations=[str(runtime)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[package] = module
        spec.loader.exec_module(module)
    return importlib.import_module(package + '.' + name)


class Face(dict):
    def __getattr__(self, key):
        return self.get(key)


def read_rgb(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert('RGB')
        if min(im.size) < 16:
            raise ValueError('image_too_small')
        return np.asarray(im).copy()


class FaceAligner:
    def __init__(self, model_root='models/insightface', device='cpu', threshold=0.5):
        import onnxruntime as ort
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        root = Path(model_root)
        detector_class = runtime_module(root, 'model_zoo.scrfd').SCRFD
        attribute_class = runtime_module(root, 'model_zoo.attribute').Attribute
        self.face_align = runtime_module(root, 'utils.face_align')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        def session(name):
            return ort.InferenceSession(str(root / 'models/buffalo_l' / name), sess_options=options, providers=providers)
        self.detector = detector_class(session=session('det_10g.onnx'))
        self.detector.prepare(ctx_id=0 if device == 'cuda' else -1, input_size=(640, 640), det_thresh=threshold)
        attribute_path = root / 'models/buffalo_l/genderage.onnx'
        self.attribute = attribute_class(model_file=str(attribute_path), session=session('genderage.onnx'))

    def select(self, rgb, training=False):
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        boxes, landmarks = self.detector.detect(bgr)
        faces = [Face(bbox=box[:4], kps=landmark, det_score=float(box[4])) for box, landmark in zip(boxes, landmarks)]
        if not faces:
            raise ValueError('face_not_detected')
        if training and len(faces) != 1:
            raise ValueError('multiple_faces')
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        if face.kps is None or face.kps.shape != (5, 2) or not np.isfinite(face.kps).all():
            raise ValueError('invalid_landmarks')
        self.attribute.get(bgr, face)
        return face

    def crop(self, rgb, face, size=224, context=False):
        import cv2
        # ArcFace template at 224 and 512 is explicitly scaled from 112.
        matrix = self.face_align.estimate_norm(face.kps, image_size=112) * (size / 112)
        if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix[:, :2])) < 1e-10:
            raise ValueError('invalid_similarity_transform')
        if context:
            matrix[:, :2] *= 0.65
            matrix[:, 2] = matrix[:, 2] * 0.65 + size * 0.175
        result = cv2.warpAffine(rgb, matrix, (size, size), borderMode=cv2.BORDER_CONSTANT)
        return result

    def align(self, rgb, training=False):
        face = self.select(rgb, training=training)
        return self.crop(rgb, face), face
