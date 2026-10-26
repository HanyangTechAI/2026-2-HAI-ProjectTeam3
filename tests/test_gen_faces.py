"""CPU geometry/selection contracts; fake detections are not model-quality evidence."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import cv2
import numpy as np
from PIL import Image

from age.face import FaceAligner
from gen import FaceNotFoundError, ModelUnavailableError
from gen.faces import FaceEdit, FaceTools


TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def detection(bbox=(150, 90, 370, 430), score=0.9, kps=None):
    box = np.asarray(bbox, dtype=float)
    if kps is None:
        kps = np.array([[0.28, 0.42], [0.72, 0.42], [0.5, 0.6], [0.32, 0.78], [0.68, 0.78]])
        kps = box[:2] + kps * (box[2:] - box[:2])
    return np.append(box, score), np.asarray(kps, dtype=float)


def estimate_norm(points, image_size):
    assert image_size == 112
    # Deterministic similarity fit; both old crop and new geometry use this fixture.
    x, y = points.T
    rows = np.zeros((10, 4))
    rows[::2] = np.stack([x, -y, np.ones(5), np.zeros(5)], axis=1)
    rows[1::2] = np.stack([y, x, np.zeros(5), np.ones(5)], axis=1)
    a, b, tx, ty = np.linalg.lstsq(rows, TEMPLATE.reshape(-1), rcond=None)[0]
    return np.array([[a, -b, tx], [b, a, ty]])


def tools(d640=(), d320=()):
    instance = FaceTools.__new__(FaceTools)
    aligner = FaceAligner.__new__(FaceAligner)
    aligner.face_align = SimpleNamespace(estimate_norm=estimate_norm)
    detections = {640: d640, 320: d320}

    def detect(bgr, input_size):
        found = detections[input_size[0]]
        return [item[0].copy() for item in found], [item[1].copy() for item in found]

    aligner.detector = SimpleNamespace(det_thresh=0.9, detect=Mock(side_effect=detect))
    def attribute(bgr, face):
        face.gender = 1
    aligner.attribute = SimpleNamespace(get=Mock(side_effect=attribute))
    instance.aligner = aligner
    instance.recognizer = SimpleNamespace(get=Mock(return_value=np.arange(1, 513, dtype=float)))
    return instance


def textured_image(width=512, height=512):
    y, x = np.mgrid[:height, :width]
    return Image.fromarray(np.stack([(x * 7 + y) % 256, (y * 5 + x) % 256,
                                     (x + y * 11) % 256], axis=-1).astype(np.uint8))


class FaceToolsTests(unittest.TestCase):
    def test_always_checks_both_scales_despite_small_face_at_640(self):
        tiny = detection((487, 34, 504, 51))
        primary = detection()
        instance = tools([tiny], [primary])
        image = Image.new('RGB', (512, 512), (17, 29, 43))
        rgb, face = instance._select(image)
        np.testing.assert_array_equal(face.bbox, primary[0][:4])
        self.assertEqual(face.detector_size, 320)
        self.assertEqual(instance.aligner.detector.det_thresh, 0.5)
        self.assertEqual([c.kwargs['input_size'] for c in instance.aligner.detector.detect.call_args_list],
                         [(640, 640), (320, 320)])
        for call in instance.aligner.detector.detect.call_args_list:
            self.assertEqual(call.args[0][0, 0].tolist(), [43, 29, 17])
        self.assertEqual(rgb[0, 0].tolist(), [17, 29, 43])
        self.assertEqual(face.selection_info['rejected'][0]['reason'], 'face_too_small')

    def test_nms_and_all_apis_use_same_high_confidence_face(self):
        low = detection(score=0.65)
        high = detection((151, 91, 371, 431), score=0.9, kps=low[1] + 1)
        instance = tools([low], [high])
        image = textured_image()
        edit = instance.prepare_edit(image)
        crop, gender = instance.prepare(image)
        vector = instance.embedding(image)
        calls = instance.aligner.detector.detect.call_count
        reused = instance.embedding(edit)
        self.assertEqual(instance.aligner.detector.detect.call_count, calls)
        self.assertIs(instance.recognizer.get.call_args.args[1], edit.face)
        self.assertEqual(gender, 'male')
        self.assertEqual(edit.face.detected_count, 1)
        self.assertEqual(edit.selection_info['duplicates_removed'], 1)
        self.assertEqual(edit.face.detector_size, 320)
        self.assertEqual(crop.tobytes(), edit.crop.tobytes())
        self.assertEqual(vector, reused)
        self.assertEqual(len(vector), 512)
        self.assertAlmostEqual(np.linalg.norm(vector), 1)
        for call in instance.aligner.attribute.get.call_args_list:
            np.testing.assert_array_equal(call.args[1].kps, high[1])
        for call in instance.recognizer.get.call_args_list:
            np.testing.assert_array_equal(call.args[1].kps, high[1])

    def test_center_and_clipped_area_not_largest_raw_box(self):
        center = detection((180, 160, 340, 360), score=0.7)
        larger_side = detection((0, 120, 180, 400), score=0.95)
        clipped_side = detection((372, 168, 627, 534), score=0.9)
        for side in (larger_side, clipped_side):
            with self.subTest(side=side[0]):
                instance = tools([side, center], [])
                _, face = instance._select(textured_image())
                np.testing.assert_array_equal(face.bbox, center[0][:4])
                self.assertTrue(face.other_bboxes)

    def test_similarly_prominent_distinct_faces_fail_without_guessing(self):
        left = detection((38, 116, 218, 396))
        right = detection((294, 116, 474, 396))
        instance = tools([left, right], [left, right])
        with self.assertRaisesRegex(FaceNotFoundError, 'ambiguous_primary_face') as caught:
            instance.prepare_edit(textured_image())
        self.assertEqual(caught.exception.selection_info['detected_count'], 2)
        self.assertEqual(caught.exception.selection_info['duplicates_removed'], 2)
        instance.aligner.attribute.get.assert_not_called()

    def test_rejects_bogus_tiny_outside_and_invalid_landmarks(self):
        valid = detection()
        bad = [detection((100, 100, 90, 300)), detection((550, 100, 700, 300)),
               detection((-1000, -1000, 2000, 2000)), detection((10, 10, 27, 28)),
               detection(score=0.49), detection(score=float('nan')),
               detection(kps=np.ones((4, 2))), detection(kps=np.ones((5, 2)) * np.nan),
               detection(kps=np.ones((5, 2)) * 200), detection(kps=valid[1] + [600, 0]),
               detection(kps=valid[1][[1, 0, 2, 3, 4]])]
        for found in bad:
            with self.subTest(box=found[0], kps=found[1]):
                instance = tools([found], [])
                with self.assertRaisesRegex(FaceNotFoundError, 'no_usable_face'):
                    instance._select(textured_image())
                instance.aligner.attribute.get.assert_not_called()
        instance = tools([detection(score=0.5)], [])
        self.assertEqual(instance._select(textured_image())[1].det_score, 0.5)

    def test_no_detections_and_small_inputs_fail_explicitly(self):
        instance = tools()
        with self.assertRaisesRegex(FaceNotFoundError, 'face_not_detected'):
            instance.prepare(textured_image())
        self.assertEqual(instance.aligner.detector.detect.call_count, 2)
        with self.assertRaisesRegex(FaceNotFoundError, 'image_too_small'):
            instance.prepare(Image.new('RGB', (31, 80)))
        with self.assertRaisesRegex(ValueError, 'RGB'):
            instance.prepare(Image.new('L', (100, 100)))

    def test_exact_context_affine_rotated_offcenter_nonsquare_and_background(self):
        angle = np.deg2rad(23)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        points = (TEMPLATE - [56, 70]) @ rotation.T * 2.4 + [260, 210]
        corners = (np.array([[18, 14], [94, 14], [94, 118], [18, 118]]) - [56, 70]) @ rotation.T * 2.4 + [260, 210]
        primary = detection(np.r_[corners.min(axis=0), corners.max(axis=0)], kps=points)
        other = detection((590, 100, 760, 310))
        instance = tools([other], [primary])
        image = textured_image(803, 477)
        before = image.tobytes()
        edit = instance.prepare_edit(image)
        self.assertIsInstance(edit, FaceEdit)
        self.assertIsNot(edit.original, image)
        self.assertEqual(edit.crop.size, (512, 512))
        self.assertEqual(edit.crop.mode, 'RGB')
        expected = estimate_norm(points, 112) * (512 / 112)
        expected[:, :2] *= 0.65
        expected[:, 2] = expected[:, 2] * 0.65 + 512 * 0.175
        np.testing.assert_array_equal(edit.matrix, expected)
        self.assertGreater(abs(edit.matrix[0, 1]), 0.1)
        np.testing.assert_array_equal(np.asarray(edit.crop), instance.aligner.crop(np.asarray(image), edit.face, 512, True))
        aligned = points @ edit.matrix[:, :2].T + edit.matrix[:, 2]
        inverse = cv2.invertAffineTransform(edit.matrix)
        np.testing.assert_allclose(aligned @ inverse[:, :2].T + inverse[:, 2], points, atol=1e-10)
        generated = Image.new('RGB', (512, 512), (255, 0, 255))
        result = instance.finish_edit(edit, generated)
        original, output, mask = np.asarray(image), np.asarray(result), np.asarray(edit.source_mask)
        self.assertEqual(result.size, image.size)
        self.assertEqual(result.mode, 'RGB')
        self.assertTrue((mask == 255).any())
        np.testing.assert_array_equal(output[mask == 0], original[mask == 0])
        np.testing.assert_array_equal(output[100:310, 590:760], original[100:310, 590:760])
        self.assertTrue((output[mask == 255] != original[mask == 255]).any())
        self.assertEqual(image.tobytes(), before)
        self.assertEqual(edit.original.tobytes(), before)

    def test_compact_smooth_support_and_exact_feature_holes(self):
        instance = tools([detection()], [])
        image = textured_image()
        edit = instance.prepare_edit(image)
        mask = np.asarray(edit.mask)
        self.assertEqual(edit.mask.mode, 'L')
        self.assertEqual(edit.source_mask.size, image.size)
        self.assertFalse(mask[[0, -1]].any())
        self.assertFalse(mask[:, [0, -1]].any())
        self.assertGreater(len(np.unique(mask)), 20)
        geometry = edit.selection_info['mask_geometry']
        y, x = np.mgrid[:512, :512]
        cx, cy = geometry['center']
        rx, ry = geometry['radii']
        self.assertFalse(mask[((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 >= 1].any())
        output = np.asarray(instance.finish_edit(edit, Image.new('RGB', (512, 512), 'white')))
        original, source = np.asarray(image), np.asarray(edit.source_mask)
        inverse = cv2.invertAffineTransform(edit.matrix)
        for feature in geometry['protected_features']:
            center = np.asarray(feature['center'])
            px, py = np.rint(inverse[:, :2] @ center + inverse[:, 2]).astype(int)
            self.assertFalse(source[py - 2:py + 3, px - 2:px + 3].any())
            np.testing.assert_array_equal(output[py - 2:py + 3, px - 2:px + 3],
                                          original[py - 2:py + 3, px - 2:px + 3])

    def test_adjacent_clipped_person_is_protected_even_if_unselectable(self):
        primary = detection()
        adjacent = detection((330, 240, 610, 510))
        image = textured_image()
        alone = tools([primary], []).prepare_edit(image)
        instance = tools([primary, adjacent], [primary])
        edit = instance.prepare_edit(image)
        self.assertTrue(np.asarray(alone.source_mask)[240:510, 330:512].any())
        self.assertFalse(np.asarray(edit.source_mask)[240:510, 330:512].any())
        output = instance.finish_edit(edit, Image.new('RGB', (512, 512), 'white'))
        np.testing.assert_array_equal(np.asarray(output)[240:510, 330:512], np.asarray(image)[240:510, 330:512])

    def test_small_secondary_face_is_protected_before_primary_size_filter(self):
        primary = detection()
        tiny = detection((255, 275, 272, 292), score=.95)
        image = textured_image()
        alone = tools([primary], []).prepare_edit(image)
        instance = tools([primary, tiny], [primary])
        edit = instance.prepare_edit(image)
        self.assertTrue(np.asarray(alone.source_mask)[275:292, 255:272].any())
        self.assertEqual(edit.face.detected_count, 1)
        self.assertFalse(np.asarray(edit.source_mask)[275:292, 255:272].any())
        result = instance.finish_edit(edit, Image.new('RGB', (512, 512), 'white'))
        np.testing.assert_array_equal(np.asarray(result)[275:292, 255:272], np.asarray(image)[275:292, 255:272])

    def test_finish_rejects_wrong_size_mode_or_invalid_geometry(self):
        instance = tools([detection()], [])
        edit = instance.prepare_edit(textured_image())
        for generated in (Image.new('L', (512, 512)), Image.new('RGB', (256, 512))):
            with self.assertRaisesRegex(ValueError, 'generated RGB'):
                instance.finish_edit(edit, generated)
        for matrix in (np.zeros((2, 3)), np.full((2, 3), np.nan), np.eye(3)):
            with self.assertRaisesRegex(ValueError, 'invalid_similarity_transform'):
                instance.finish_edit(replace(edit, matrix=matrix), edit.crop)
        with self.assertRaisesRegex(ValueError, 'invalid_source_mask'):
            instance.finish_edit(replace(edit, source_mask=Image.new('L', (16, 16))), edit.crop)

    def test_embedding_failures_are_explicit_not_quality_gates(self):
        instance = tools([detection()], [])
        image = textured_image()
        for value in (np.zeros(512), np.ones(511), np.full(512, np.nan)):
            instance.recognizer.get.return_value = value
            with self.assertRaisesRegex(ValueError, 'Invalid ArcFace embedding'):
                instance.embedding(image)
        instance.recognizer = None
        with self.assertRaises(ModelUnavailableError):
            instance.embedding(image)


if __name__ == '__main__':
    unittest.main()
