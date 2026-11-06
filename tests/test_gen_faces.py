from dataclasses import replace
import hashlib
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
    # Shared deterministic similarity fit for crop and edit geometry.
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

    def test_explicit_defaults_match_implicit_and_generation_mask_is_independent(self):
        instance = tools([detection()], [])
        image = textured_image()
        implicit = instance.prepare_edit(image)
        explicit = instance.prepare_edit(image, generation_mask_mode='protected', mask_scale=1.0,
                                         feature_protection=1.0)
        for field in ('crop', 'mask', 'source_mask', 'generation_mask'):
            self.assertEqual(getattr(implicit, field).tobytes(), getattr(explicit, field).tobytes())
        self.assertEqual(implicit.generation_mask.tobytes(), implicit.mask.tobytes())
        implicit.generation_mask.paste(255, (0, 0, 512, 512))
        self.assertEqual(implicit.mask.tobytes(), explicit.mask.tobytes())
        legacy = FaceEdit(explicit.original, explicit.crop, explicit.mask, explicit.matrix,
                          explicit.face, explicit.source_mask, explicit.selection_info)
        self.assertIsNone(legacy.generation_mask)
        generated = Image.new('RGB', (512, 512), 'white')
        self.assertEqual(instance.finish_edit(legacy, generated).tobytes(),
                         instance.finish_edit(explicit, generated).tobytes())

    def test_face_generation_fills_feature_holes_but_does_not_change_composition(self):
        instance = tools([detection()], [])
        image = textured_image()
        protected = instance.prepare_edit(image)
        face = instance.prepare_edit(image, generation_mask_mode='face')
        self.assertEqual(face.mask.tobytes(), protected.mask.tobytes())
        self.assertEqual(face.source_mask.tobytes(), protected.source_mask.tobytes())
        generation, composite = np.asarray(face.generation_mask), np.asarray(face.mask)
        self.assertTrue((generation >= composite).all())
        self.assertTrue((generation > composite).any())
        for feature in face.selection_info['mask_geometry']['protected_features']:
            x, y = np.rint(feature['center']).astype(int)
            self.assertEqual(composite[y, x], 0)
            self.assertGreater(generation[y, x], 0)
        generated = Image.new('RGB', (512, 512), 'white')
        self.assertEqual(instance.finish_edit(face, generated).tobytes(),
                         instance.finish_edit(protected, generated).tobytes())

    def test_mask_scale_and_feature_protection_are_monotonic_with_exact_zero_support(self):
        instance = tools([detection()], [])
        image = textured_image()
        default = instance.prepare_edit(image)
        expanded = instance.prepare_edit(image, mask_scale=1.25)
        reduced = instance.prepare_edit(image, feature_protection=.5)
        unprotected = instance.prepare_edit(image, feature_protection=0)
        self.assertTrue((np.asarray(expanded.mask) >= np.asarray(default.mask)).all())
        self.assertTrue((np.asarray(reduced.mask) >= np.asarray(default.mask)).all())
        self.assertTrue((np.asarray(unprotected.mask) >= np.asarray(reduced.mask)).all())
        self.assertGreater(np.asarray(expanded.mask).sum(), np.asarray(default.mask).sum())
        self.assertGreater(np.asarray(reduced.mask).sum(), np.asarray(default.mask).sum())
        self.assertGreater(np.asarray(unprotected.mask).sum(), np.asarray(reduced.mask).sum())
        for full, half in zip(default.selection_info['mask_geometry']['protected_features'],
                              reduced.selection_info['mask_geometry']['protected_features']):
            np.testing.assert_array_equal(np.asarray(half['radii']), np.asarray(full['radii']) * .5)
        self.assertEqual(unprotected.selection_info['mask_geometry']['protected_features'], [])
        for edit in (expanded, reduced, unprotected):
            output = np.asarray(instance.finish_edit(edit, Image.new('RGB', (512, 512), 'white')))
            mask = np.asarray(edit.source_mask)
            np.testing.assert_array_equal(output[mask == 0], np.asarray(image)[mask == 0])
            x1, y1, x2, y2 = edit.face.bbox.astype(int)
            self.assertFalse(mask[:y1 + 1].any())
            self.assertFalse(mask[y2:].any())
            self.assertFalse(mask[:, :x1 + 1].any())
            self.assertFalse(mask[:, x2:].any())

    def test_relaxed_generation_still_guards_other_faces_and_primary_bbox(self):
        primary, tiny = detection(), detection((255, 275, 272, 292), score=.95)
        image = textured_image()
        instance = tools([primary, tiny], [primary])
        edit = instance.prepare_edit(image, generation_mask_mode='face', mask_scale=1.25, feature_protection=0)
        inverse = cv2.invertAffineTransform(edit.matrix)
        y, x = np.ogrid[:512, :512]
        sx = inverse[0, 0] * x + inverse[0, 1] * y + inverse[0, 2]
        sy = inverse[1, 0] * x + inverse[1, 1] * y + inverse[1, 2]
        x1, y1, x2, y2 = edit.face.bbox
        outside = (sx <= x1) | (sx >= x2) | (sy <= y1) | (sy >= y2)
        other = (sx >= 255) & (sx <= 272) & (sy >= 275) & (sy <= 292)
        self.assertTrue(other.any())
        for mask in (edit.mask, edit.generation_mask):
            self.assertFalse(np.asarray(mask)[outside | other].any())
        self.assertFalse(np.asarray(edit.source_mask)[275:292, 255:272].any())
        output = np.asarray(instance.finish_edit(edit, Image.new('RGB', (512, 512), 'white')))
        source_mask = np.asarray(edit.source_mask)
        np.testing.assert_array_equal(output[source_mask == 0], np.asarray(image)[source_mask == 0])

    def test_mask_controls_reject_invalid_values_before_detection(self):
        instance = tools([detection()], [])
        for values in ({'generation_mask_mode': 'all'}, {'mask_scale': 0}, {'mask_scale': 1.251},
                       {'mask_scale': float('nan')}, {'mask_scale': True}, {'mask_scale': '1'},
                       {'feature_protection': -.1}, {'feature_protection': 1.01},
                       {'feature_protection': float('inf')}, {'feature_protection': None}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                instance.prepare_edit(textured_image(), **values)
        instance.aligner.detector.detect.assert_not_called()

    def test_default_fixture_bytes_remain_identical(self):
        instance = tools([detection()], [])
        edit = instance.prepare_edit(textured_image())
        expected = {
            'crop': '3ca5575ff998ee6879ec145e0504d67f950c5764ac558c0c52d19abb49746fbc',
            'mask': 'b815405df319226ea34597914f33941789584f57c36dae5471b842bdfe8696be',
            'source_mask': '3bbef2bd04e218dd2b1bbd5fcf954863cdb38b9868f202fdd7d996a3f46740f4',
            'output': 'd64b89a18b735262b401ba098b512d923efd3f959e882ba449a684fb83da3598',
        }
        actual = {name: getattr(edit, name) for name in ('crop', 'mask', 'source_mask')}
        actual['output'] = instance.finish_edit(edit, Image.new('RGB', (512, 512), 'white'))
        self.assertEqual({name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in actual.items()}, expected)

    def test_external_mask_validation_is_strict_and_precedes_detection(self):
        instance = tools([detection()], [])
        image = textured_image(600, 512)
        for name in ('source_edit_mask', 'source_generation_mask', 'source_protect_mask'):
            for invalid in (False, np.ones((512, 600), np.uint8), Image.new('RGB', image.size),
                            Image.new('1', image.size), Image.new('F', image.size),
                            Image.new('L', (512, 600)), Image.new('L', (512, 512))):
                with self.subTest(name=name, mode=getattr(invalid, 'mode', None)), self.assertRaisesRegex(ValueError, name):
                    instance.prepare_edit(image, **{name: invalid})
        for name in ('source_edit_mask', 'source_generation_mask'):
            with self.assertRaisesRegex(ValueError, 'nonempty'):
                instance.prepare_edit(image, **{name: Image.new('L', image.size)})
        instance.aligner.detector.detect.assert_not_called()

    def test_external_edit_replaces_holes_and_ellipse_and_drives_generation(self):
        instance = tools([detection()], [])
        image = textured_image()
        original = instance.prepare_edit(image)
        supplied = Image.new('L', image.size, 255)
        before = supplied.tobytes()
        for mode in ('protected', 'face'):
            edit = instance.prepare_edit(image, source_edit_mask=supplied, generation_mask_mode=mode)
            self.assertEqual(edit.mask.tobytes(), edit.generation_mask.tobytes())
            self.assertTrue(((np.asarray(original.mask) == 0) & (np.asarray(edit.mask) > 0)).any())
            for feature in original.selection_info['mask_geometry']['protected_features']:
                x, y = np.rint(feature['center']).astype(int)
                self.assertGreater(edit.mask.getpixel((x, y)), 0)
            metadata = edit.selection_info['mask_geometry']
            self.assertEqual(metadata['blend_origin'], 'source_edit_mask')
            self.assertEqual(metadata['generation_origin'], 'source_edit_mask')
            self.assertEqual(metadata['kind'], 'source_edit_mask')
        self.assertEqual(supplied.tobytes(), before)

    def test_external_generation_and_blend_support_are_independent(self):
        instance = tools([detection()], [])
        image = textured_image()
        blend, generation = Image.new('L', image.size), Image.new('L', image.size)
        blend.paste(192, (180, 180, 250, 340))
        generation.paste(255, (280, 180, 340, 340))
        edit = instance.prepare_edit(image, source_edit_mask=blend, source_generation_mask=generation)
        self.assertFalse((np.asarray(edit.mask) * np.asarray(edit.generation_mask).astype(int)).any())
        self.assertTrue(np.asarray(edit.mask).any())
        self.assertTrue(np.asarray(edit.generation_mask).any())
        source = np.asarray(edit.source_mask)
        self.assertTrue((source <= np.asarray(blend)).all())
        output = np.asarray(instance.finish_edit(edit, Image.new('RGB', (512, 512), 'white')))
        np.testing.assert_array_equal(output[np.asarray(blend) == 0], np.asarray(image)[np.asarray(blend) == 0])
        default = instance.prepare_edit(image)
        generation_only = instance.prepare_edit(image, source_generation_mask=generation)
        self.assertEqual(generation_only.mask.tobytes(), default.mask.tobytes())
        self.assertEqual(generation_only.source_mask.tobytes(), default.source_mask.tobytes())
        self.assertEqual(edit.selection_info['mask_geometry']['generation_origin'], 'source_generation_mask')

    def test_external_masks_cannot_cross_bbox_or_other_face_guards(self):
        instance = tools([detection(), detection((255, 275, 272, 292), score=.95)], [])
        image = textured_image()
        edit = instance.prepare_edit(image, source_edit_mask=Image.new('L', image.size, 255),
                                     source_generation_mask=Image.new('L', image.size, 255))
        inverse = cv2.invertAffineTransform(edit.matrix)
        y, x = np.ogrid[:512, :512]
        sx = inverse[0, 0] * x + inverse[0, 1] * y + inverse[0, 2]
        sy = inverse[1, 0] * x + inverse[1, 1] * y + inverse[1, 2]
        unsafe = (sx <= 150) | (sx >= 370) | (sy <= 90) | (sy >= 430)
        unsafe |= (sx >= 255) & (sx <= 272) & (sy >= 275) & (sy <= 292)
        for mask in (edit.mask, edit.generation_mask):
            self.assertFalse(np.asarray(mask)[unsafe].any())
        source = np.asarray(edit.source_mask)
        self.assertFalse(source[275:293, 255:273].any())
        self.assertFalse(source[:91].any())
        self.assertFalse(source[430:].any())
        self.assertFalse(source[:, :151].any())
        self.assertFalse(source[:, 370:].any())

    def test_protection_outside_blend_support_still_excludes_generation(self):
        instance = tools([detection()], [])
        image = textured_image()
        blend = Image.new('L', image.size)
        blend.paste(255, (180, 180, 235, 330))
        generation = Image.new('L', image.size, 255)
        protection = Image.new('L', image.size)
        protection.paste(1, (280, 270, 315, 300))
        baseline = instance.prepare_edit(image, source_edit_mask=blend, source_generation_mask=generation)
        edit = instance.prepare_edit(image, source_edit_mask=blend, source_generation_mask=generation,
                                     source_protect_mask=protection)
        self.assertEqual(edit.mask.tobytes(), baseline.mask.tobytes())
        self.assertEqual(edit.source_mask.tobytes(), baseline.source_mask.tobytes())
        x, y = np.rint(edit.matrix[:, :2] @ [295, 285] + edit.matrix[:, 2]).astype(int)
        self.assertEqual(edit.mask.getpixel((x, y)), 0)
        self.assertGreater(baseline.generation_mask.getpixel((x, y)), 0)
        self.assertEqual(edit.generation_mask.getpixel((x, y)), 0)
        self.assertEqual(edit.selection_info['mask_geometry']['protection_origin'], 'source_protect_mask')

    def test_protection_is_independent_hard_nonzero_and_conservative_when_rotated(self):
        angle = np.deg2rad(23)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        points = (TEMPLATE - [56, 70]) @ rotation.T * 5 + [600, 450]
        corners = (np.array([[18, 14], [94, 14], [94, 118], [18, 118]]) - [56, 70]) @ rotation.T * 5 + [600, 450]
        instance = tools([detection(np.r_[corners.min(axis=0), corners.max(axis=0)], kps=points)], [])
        image = textured_image(1606, 954)
        protect = Image.new('L', image.size)
        px, py = np.rint(points[2]).astype(int)
        protect.putpixel((px, py), 1)
        before = protect.tobytes()
        for custom in (False, True):
            kwargs = {'source_edit_mask': Image.new('L', image.size, 255),
                      'source_generation_mask': Image.new('L', image.size, 255)} if custom else {}
            base = instance.prepare_edit(image, **kwargs)
            edit = instance.prepare_edit(image, source_protect_mask=protect, **kwargs)
            cx, cy = edit.matrix[:, :2] @ [px, py] + edit.matrix[:, 2]
            # Every crop texel participating in the inverse bilinear sample is protected.
            for x in (int(np.floor(cx)), int(np.ceil(cx))):
                for y in (int(np.floor(cy)), int(np.ceil(cy))):
                    self.assertGreater(base.generation_mask.getpixel((x, y)), 0)
                    self.assertEqual(edit.mask.getpixel((x, y)), 0)
                    self.assertEqual(edit.generation_mask.getpixel((x, y)), 0)
            self.assertEqual(edit.source_mask.getpixel((px, py)), 0)
            output = instance.finish_edit(edit, Image.new('RGB', (512, 512), 'white'))
            self.assertEqual(output.getpixel((px, py)), image.getpixel((px, py)))
            self.assertGreater(edit.selection_info['mask_geometry']['protection_source_dilation'], 1)
            full = protect.point(lambda value: 255 if value else 0)
            self.assertEqual(instance.prepare_edit(image, source_protect_mask=full, **kwargs).mask.tobytes(),
                             edit.mask.tobytes())
        self.assertEqual(protect.tobytes(), before)

    def test_empty_effective_edit_or_generation_is_rejected_and_empty_protect_is_valid(self):
        instance = tools([detection()], [])
        image = textured_image()
        outside = Image.new('L', image.size)
        outside.paste(255, (0, 0, 20, 20))
        for name, reason in (('source_edit_mask', 'empty_face_edit_mask'),
                             ('source_generation_mask', 'empty_face_generation_mask')):
            with self.assertRaisesRegex(FaceNotFoundError, reason):
                instance.prepare_edit(image, **{name: outside})
        with self.assertRaisesRegex(FaceNotFoundError, 'empty_face_edit_mask'):
            instance.prepare_edit(image, source_protect_mask=Image.new('L', image.size, 1))
        default = instance.prepare_edit(image)
        no_protect = instance.prepare_edit(image, source_protect_mask=Image.new('L', image.size))
        for name in ('mask', 'generation_mask', 'source_mask'):
            self.assertEqual(getattr(default, name).tobytes(), getattr(no_protect, name).tobytes())

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
