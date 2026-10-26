"""Generation-only primary-face selection and local, reversible edit geometry.

Masks are conservative geometric ellipses, not skin/hair segmentation. Occluding
hair, glasses and expression changes outside protected landmarks remain possible.
Only pixels with a nonzero source_mask can change during compositing.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from . import FaceNotFoundError, ModelUnavailableError


@dataclass
class FaceEdit:
    original: Image.Image
    crop: Image.Image
    mask: Image.Image
    matrix: np.ndarray  # Original -> 512 context crop, including rotation.
    face: dict
    source_mask: Image.Image
    selection_info: dict


def _same_face(left, right):
    lo = np.maximum(left.bbox[:2], right.bbox[:2])
    hi = np.minimum(left.bbox[2:], right.bbox[2:])
    intersection = float(np.prod(np.maximum(hi - lo, 0)))
    areas = [float(np.prod(face.bbox[2:] - face.bbox[:2])) for face in (left, right)]
    iou = intersection / (sum(areas) - intersection)
    # Box overlap alone can merge two nearby people. Check landmark agreement too.
    distance = float(np.sqrt(np.mean(np.sum((left.kps - right.kps) ** 2, axis=1))))
    return iou >= 0.5 and distance <= 0.2 * np.sqrt(min(areas))


def _smoothstep(value):
    value = np.clip(value, 0, 1)
    return value * value * (3 - 2 * value)


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
            options = ort.SessionOptions()
            options.intra_op_num_threads = 4
            session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
            cls = runtime_module(root, 'model_zoo.arcface_onnx').ArcFaceONNX
            self.recognizer = cls(model_file=str(path), session=session)
            self.recognizer.prepare(ctx_id=0 if device == 'cuda' else -1)

    def _select(self, image):
        from age.face import Face
        from .pipeline import as_rgb
        rgb = np.asarray(as_rgb(image))
        height, width = rgb.shape[:2]
        if min(height, width) < 32:
            raise FaceNotFoundError('image_too_small')
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        candidates, guards = [], []
        info = {'detector_sizes': [640, 320], 'det_thresh': 0.5,
                'raw_counts': {}, 'rejected': [], 'candidates': []}
        # Always inspect BOTH scales: a spurious small face at 640 is not success.
        self.aligner.detector.det_thresh = 0.5
        for size in info['detector_sizes']:
            boxes, landmarks = self.aligner.detector.detect(bgr, input_size=(size, size))
            info['raw_counts'][str(size)] = len(boxes)
            for index, box in enumerate(boxes):
                box = np.asarray(box, dtype=np.float64)
                reason = None
                if box.shape != (5,) or not np.isfinite(box).all() or not 0.5 <= box[4] <= 1:
                    reason = 'invalid_bbox_or_score'
                elif np.any(box[2:4] <= box[:2]):
                    reason = 'invalid_bbox'
                if reason:
                    info['rejected'].append({'detector_size': size, 'index': index, 'reason': reason})
                    continue
                clipped = np.clip(box[:4], [0, 0, 0, 0], [width, height, width, height])
                extent = clipped[2:] - clipped[:2]
                area = float(np.prod(extent))
                visible = area / float(np.prod(box[2:4] - box[:2]))
                kps = None if landmarks is None or index >= len(landmarks) else np.asarray(landmarks[index])
                if (np.all(extent > 0) and 0.4 <= extent[0] / extent[1] <= 2.5
                        and kps is not None and kps.shape == (5, 2) and np.isfinite(kps).all()):
                    face = Face(bbox=clipped, raw_bbox=box[:4].copy(), kps=kps.copy(),
                                det_score=float(box[4]), detector_size=size, visible_fraction=visible,
                                area_fraction=area / (width * height))
                    # Small/incomplete people still need protection, even though
                    # they cannot be selected as the editing subject.
                    guards.append(face)
                if np.any(extent <= 0):
                    reason = 'bbox_outside_image'
                elif min(extent) < 32 or area / (width * height) < 0.005:
                    reason = 'face_too_small'
                elif not 0.4 <= extent[0] / extent[1] <= 2.5:
                    reason = 'invalid_bbox_aspect'
                elif kps is None or kps.shape != (5, 2) or not np.isfinite(kps).all():
                    reason = 'invalid_landmarks'
                else:
                    eyes = kps[1] - kps[0]
                    eye_span = float(np.linalg.norm(eyes))
                    down = np.array([-eyes[1], eyes[0]]) / max(eye_span, 1e-8)
                    eye_mid, mouth_mid = kps[:2].mean(axis=0), kps[3:].mean(axis=0)
                    depth = float((mouth_mid - eye_mid) @ down)
                    nose_depth = float((kps[2] - eye_mid) @ down)
                    if visible < 0.6:
                        reason = 'mostly_clipped_face'
                    elif (kps < 0).any() or (kps >= [width, height]).any():
                        reason = 'landmarks_outside_image'
                    elif ((kps < clipped[:2] - extent * 0.05).any()
                          or (kps > clipped[2:] + extent * 0.05).any()
                          or eye_span < max(6, 0.12 * min(extent))
                          or depth < 0.15 * min(extent)
                          or not -0.15 * depth <= nose_depth <= 1.15 * depth
                          or float((kps[4] - kps[3]) @ eyes) <= 0
                          or np.linalg.norm(kps[4] - kps[3]) < 0.08 * min(extent)):
                        reason = 'invalid_landmark_geometry'
                    else:
                        offset = ((clipped[:2] + clipped[2:]) / 2 - [width / 2, height / 2]) / [width / 2, height / 2]
                        face['primary_score'] = float(face.area_fraction * visible * np.exp(-2 * (offset @ offset)))
                        candidates.append(face)
                if reason:
                    info['rejected'].append({'detector_size': size, 'index': index, 'reason': reason,
                                             'bbox': clipped.tolist(), 'raw_bbox': box[:4].tolist(),
                                             'det_score': float(box[4])})
        faces = []
        for face in sorted(candidates, key=lambda f: (-f.det_score, f.detector_size, *f.bbox)):
            if not any(_same_face(face, other) for other in faces):
                faces.append(face)
        faces.sort(key=lambda f: (-f.primary_score, -f.det_score, *f.bbox))
        info['duplicates_removed'] = len(candidates) - len(faces)
        info['detected_count'] = len(faces)
        info['candidates'] = [{key: value.tolist() if isinstance(value, np.ndarray) else value
                               for key, value in face.items()} for face in faces]
        error = None
        if not faces:
            error = 'no_usable_face' if sum(info['raw_counts'].values()) else 'face_not_detected'
        elif len(faces) > 1 and faces[1].primary_score >= 0.8 * faces[0].primary_score:
            error = 'ambiguous_primary_face'
        if error:
            info['status'] = error
            failure = FaceNotFoundError(error)
            failure.selection_info = info
            raise failure
        face = faces[0]
        info.update(status='selected', selected_index=0)
        face['detected_count'] = len(faces)
        face['selection_info'] = info
        face['other_bboxes'] = [other.bbox.copy() for other in guards if not _same_face(face, other)]
        self.aligner.attribute.get(bgr, face)
        return rgb, face

    def prepare(self, image):
        rgb, face = self._select(image)
        crop = self.aligner.crop(rgb, face, size=512, context=True)
        return Image.fromarray(crop), 'male' if face.gender == 1 else 'female'

    def prepare_edit(self, image):
        """Select once and retain exact original pixels outside compact face support.

        Eye/eyelid and mouth cores are also protected geometrically, not by a
        semantic expression model. Bangs or accessories inside the ellipse can
        still be edited; five landmarks cannot identify all skin or expression.
        """
        import cv2
        rgb, face = self._select(image)
        # EXACT FaceAligner.crop(size=512, context=True) math, without changing age.
        matrix = self.aligner.face_align.estimate_norm(face.kps, image_size=112) * (512 / 112)
        if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix[:, :2])) < 1e-10:
            raise ValueError('invalid_similarity_transform')
        matrix[:, :2] *= 0.65
        matrix[:, 2] = matrix[:, 2] * 0.65 + 512 * 0.175
        crop = cv2.warpAffine(rgb, matrix, (512, 512), borderMode=cv2.BORDER_CONSTANT)
        points = face.kps @ matrix[:, :2].T + matrix[:, 2]
        eye_mid, mouth_mid = points[:2].mean(axis=0), points[3:].mean(axis=0)
        eye_span = float(np.linalg.norm(points[1] - points[0]))
        depth = float(mouth_mid[1] - eye_mid[1])
        if depth <= 0 or eye_span <= 0:
            raise ValueError('invalid_aligned_landmarks')
        top, bottom = eye_mid[1] - 0.85 * depth, mouth_mid[1] + 0.8 * depth
        center = np.array([(eye_mid[0] + mouth_mid[0]) / 2, (top + bottom) / 2])
        radii = np.array([1.05 * eye_span, (bottom - top) / 2])
        inverse = cv2.invertAffineTransform(matrix)
        holes = [(point, np.array([0.24, 0.12]) * eye_span) for point in points[:2]]
        holes.append((mouth_mid + [0, 0.08 * eye_span],
                      np.array([np.linalg.norm(points[4] - points[3]) / 2 + 0.12 * eye_span,
                                0.24 * eye_span])))

        def mask_at(x, y):
            radius = np.sqrt(((x - center[0]) / radii[0]) ** 2 + ((y - center[1]) / radii[1]) ** 2)
            alpha = _smoothstep((1 - radius) / 0.18)
            for point, axes in holes:
                distance = np.sqrt(((x - point[0]) / axes[0]) ** 2 + ((y - point[1]) / axes[1]) ** 2)
                alpha *= _smoothstep((distance - 1) / 0.35)
            sx = inverse[0, 0] * x + inverse[0, 1] * y + inverse[0, 2]
            sy = inverse[1, 0] * x + inverse[1, 1] * y + inverse[1, 2]
            x1, y1, x2, y2 = face.bbox
            feather = 0.04 * min(x2 - x1, y2 - y1)
            edge = np.minimum(np.minimum(sx - x1, x2 - sx), np.minimum(sy - y1, y2 - sy))
            alpha *= _smoothstep(edge / feather)
            for x1, y1, x2, y2 in face.other_bboxes:
                outside = np.maximum(np.maximum(x1 - sx, sx - x2), np.maximum(y1 - sy, sy - y2))
                alpha *= _smoothstep(outside / feather)
            return np.rint(alpha * 255).astype(np.uint8)

        y, x = np.ogrid[:512, :512]
        mask = mask_at(x, y)
        height, width = rgb.shape[:2]
        source_mask = cv2.warpAffine(mask, matrix, (width, height),
                                     flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                     borderMode=cv2.BORDER_CONSTANT)
        y, x = np.ogrid[:height, :width]
        # Interpolation must not leak even one pixel into the outside or holes.
        source_mask = np.minimum(source_mask, mask_at(matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2],
                                                     matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]))
        if not source_mask.any():
            raise FaceNotFoundError('empty_face_edit_mask')
        info = dict(face.selection_info)
        info['mask_geometry'] = {'kind': 'compact_ellipse_with_protected_eyes_and_mouth',
                                 'center': center.tolist(), 'radii': radii.tolist(),
                                 'protected_features': [{'center': point.tolist(), 'radii': axes.tolist()}
                                                        for point, axes in holes],
                                 'semantic_segmentation': False}
        return FaceEdit(Image.fromarray(rgb.copy()), Image.fromarray(crop), Image.fromarray(mask),
                        matrix, face, Image.fromarray(source_mask), info)

    def finish_edit(self, edit, generated):
        """Inverse-warp generated pixels, never the original, then blend locally."""
        import cv2
        if not isinstance(generated, Image.Image) or generated.mode != 'RGB' or generated.size != (512, 512):
            raise ValueError('Expected generated RGB image of size 512x512')
        matrix = np.asarray(edit.matrix)
        if (matrix.shape != (2, 3) or not np.isfinite(matrix).all()
                or abs(np.linalg.det(matrix[:, :2])) < 1e-10):
            raise ValueError('invalid_similarity_transform')
        if (edit.original.mode != 'RGB' or edit.source_mask.mode != 'L'
                or edit.source_mask.size != edit.original.size):
            raise ValueError('invalid_source_mask')
        warped = cv2.warpAffine(np.asarray(generated), matrix, edit.original.size,
                                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_CONSTANT)
        result = np.asarray(edit.original).copy()
        mask = np.asarray(edit.source_mask)
        active = mask > 0
        alpha = mask[active, None].astype(np.uint32)
        result[active] = ((warped[active].astype(np.uint32) * alpha
                           + result[active].astype(np.uint32) * (255 - alpha) + 127) // 255).astype(np.uint8)
        return Image.fromarray(result)

    def embedding(self, image):
        if self.recognizer is None:
            raise ModelUnavailableError('ArcFace was not loaded for this instance')
        if isinstance(image, FaceEdit):
            rgb, face = np.asarray(image.original), image.face
        else:
            rgb, face = self._select(image)
        embedding = np.asarray(self.recognizer.get(np.ascontiguousarray(rgb[:, :, ::-1]), face)).reshape(-1)
        norm = float(np.linalg.norm(embedding))
        if embedding.size != 512 or not np.isfinite(embedding).all() or not np.isfinite(norm) or norm <= 0:
            raise ValueError('Invalid ArcFace embedding')
        return (embedding / norm).tolist()
