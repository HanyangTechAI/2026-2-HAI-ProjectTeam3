import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock
import numpy as np
from age.common import ROOT
from age.face import FaceAligner


@unittest.skipUnless(importlib.util.find_spec('cv2') and importlib.util.find_spec('skimage') and
                    (ROOT / 'models/insightface/provenance.json').exists(), 'Requires prepared official buffalo_l runtime')
class FaceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.aligner = FaceAligner(ROOT / 'models/insightface')

    def test_no_face_is_explicit_failure(self):
        with self.assertRaisesRegex(ValueError, 'face_not_detected'):
            self.aligner.select(np.zeros((224, 224, 3), dtype=np.uint8))

    def test_training_rejects_two_faces_inference_selects_largest(self):
        old_detector, old_attribute = self.aligner.detector, self.aligner.attribute
        try:
            self.aligner.detector = Mock()
            self.aligner.attribute = Mock()
            self.aligner.detector.detect.return_value = (
                np.array([[0, 0, 20, 20, 0.9], [0, 0, 100, 100, 0.9]], dtype=np.float32),
                np.tile(self.aligner.face_align.arcface_dst, (2, 1, 1)))
            rgb = np.zeros((224, 224, 3), dtype=np.uint8)
            with self.assertRaisesRegex(ValueError, 'multiple_faces'):
                self.aligner.select(rgb, training=True)
            face = self.aligner.select(rgb, training=False)
            self.assertEqual(face.bbox[2], 100)
        finally:
            self.aligner.detector, self.aligner.attribute = old_detector, old_attribute

    def test_training_and_inference_alignment_matches(self):
        from age.common import read_csv, image_path
        rows = read_csv(ROOT / 'data/age_stage2/train.csv') if (ROOT / 'data/age_stage2/train.csv').exists() else []
        if not rows:
            self.skipTest('Requires actual prepared faces')
        from age.face import read_rgb
        record = rows[0]
        rgb = read_rgb(ROOT / 'data/raw/aihub' / record['original_id'])
        crop, _ = self.aligner.align(rgb, training=False)
        self.assertTrue(np.array_equal(crop, read_rgb(image_path(record))))

    def test_smoke_model_requires_explicit_override(self):
        from age.infer import AgeEstimator
        model = ROOT / 'outputs/smoke/smoke_only.onnx'
        if not model.exists():
            self.skipTest('Requires smoke ONNX')
        with self.assertRaisesRegex(ValueError, 'untrained smoke'):
            AgeEstimator(model)


if __name__ == '__main__':
    unittest.main()
