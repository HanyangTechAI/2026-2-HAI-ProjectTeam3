import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from .align_public_retry import AttemptError, align_with_retry, prepare
from .common import FIELDS, read_csv, write_csv
from .face import Face


class FakeAligner:
    def __init__(self, error=None, retry_error=None):
        self.error = error
        self.retry_error = retry_error
        self.calls = []

    def align(self, rgb, training=False):
        self.calls.append(('align', rgb.shape, training))
        if self.error:
            raise self.error
        return np.full((224, 224, 3), rgb[0, 0, 0], dtype=np.uint8), None

    def select(self, padded, training=False):
        self.calls.append(('select', padded.copy(), training))
        if self.retry_error:
            raise self.retry_error
        return Face(bbox=np.array([11, 10, 19, 20], dtype=np.float32),
                    kps=np.array([[11, 10], [12, 11], [13, 12], [14, 13], [15, 14]], dtype=np.float32))

    def crop(self, rgb, face):
        self.calls.append(('crop', rgb, face))
        return np.full((224, 224, 3), rgb[0, 0, 0], dtype=np.uint8)


class RetryTest(unittest.TestCase):
    def test_base_success_is_untouched(self):
        rgb = np.full((21, 23, 3), 7, dtype=np.uint8)
        a = FakeAligner()
        result, status = align_with_retry(a, rgb)
        self.assertEqual(status, 'base_success')
        self.assertEqual(a.calls, [('align', rgb.shape, True)])
        self.assertEqual(int(result[0, 0, 0]), 7)

    def test_retry_padding_and_coordinates_use_original_image(self):
        rgb = np.full((21, 23, 3), 9, dtype=np.uint8)
        a = FakeAligner(error=ValueError('face_not_detected'))
        _, status = align_with_retry(a, rgb)
        self.assertEqual(status, 'retry_success')
        _, padded, training = a.calls[1]
        self.assertTrue(training)
        self.assertEqual(padded.shape, (41, 45, 3))
        np.testing.assert_array_equal(padded[10:31, 11:34], rgb)
        self.assertEqual(int(padded[:10].sum()), 0)
        self.assertEqual(int(padded[:, :11].sum()), 0)
        _, crop_rgb, face = a.calls[2]
        self.assertIs(crop_rgb, rgb)
        np.testing.assert_array_equal(face.bbox, [0, 0, 8, 10])
        np.testing.assert_array_equal(face.kps, [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])

    def test_only_exact_value_error_retries_and_multiple_faces_fails(self):
        rgb = np.ones((20, 20, 3), dtype=np.uint8)
        for error in (ValueError('multiple_faces'), ValueError('face_not_detected extra'),
                      RuntimeError('face_not_detected')):
            a = FakeAligner(error=error)
            with self.subTest(error=error), self.assertRaises(AttemptError) as ctx:
                align_with_retry(a, rgb)
            self.assertEqual(ctx.exception.stage, 'base')
            self.assertIs(ctx.exception.cause, error)
            self.assertEqual(len(a.calls), 1)
        a = FakeAligner(ValueError('face_not_detected'), ValueError('multiple_faces'))
        with self.assertRaises(AttemptError) as ctx:
            align_with_retry(a, rgb)
        self.assertEqual(ctx.exception.stage, 'retry')
        self.assertEqual(str(ctx.exception.cause), 'multiple_faces')
        self.assertEqual(len(a.calls), 2)


class PrepareTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.data = self.root / 'input'
        self.images = self.root / 'originals'
        self.data.mkdir()
        self.images.mkdir()
        for name, value in [('a', 1), ('b', 2), ('c', 1), ('d', 3)]:
            Image.fromarray(np.full((21, 23, 3), value, dtype=np.uint8)).save(self.images / (name + '.png'))
        def row(name, source):
            return dict(path=str(self.images / (name + '.png')), age='25', gender='', source=source,
                        person_id='', original_id=name, sha256='original')
        self.train = [row('a', 'afad'), row('b', 'mega'), row('d', 'afad')]
        self.val = [row('c', 'afad')]
        write_csv(self.data / 'train.csv', self.train, FIELDS)
        write_csv(self.data / 'val.csv', self.val, FIELDS)

    def test_report_collision_limit_and_immutable_output(self):
        output = self.root / 'output'
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            report = prepare(self.data, output, FakeAligner(), limit_per_source=1, progress_every=1)
        self.assertIn('train: processed 2/2', stream.getvalue())
        self.assertEqual(report['counts']['train']['afad']['statuses'], {'base_success': 1})
        self.assertEqual(report['counts']['train']['afad']['dispositions'], {'cross_split_excluded': 1})
        self.assertEqual(len(read_csv(output / 'cross_split_duplicates.csv')), 2)
        self.assertEqual([r['original_id'] for r in read_csv(output / 'train.csv')], ['b'])
        self.assertEqual(read_csv(output / 'val.csv'), [])
        self.assertEqual(read_csv(output / 'audit.csv')[0]['black_pixel_ratio'], '0.0')
        with self.assertRaises(ValueError):
            prepare(self.data, output, FakeAligner())
        for unsafe in (self.data / 'out', self.images / 'out', self.root):
            with self.subTest(path=unsafe), self.assertRaises(ValueError):
                prepare(self.data, unsafe, FakeAligner())

    def test_retry_failure_and_source_filter(self):
        a = FakeAligner(ValueError('face_not_detected'), ValueError('multiple_faces'))
        output = self.root / 'retry'
        report = prepare(self.data, output, a, limit_per_source=1, sources=['afad'])
        self.assertEqual(report['counts']['train']['afad']['statuses'], {'error': 1})
        rows = read_csv(output / 'failures.csv')
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r['stage'] == 'retry' and r['reason'] == 'multiple_faces' for r in rows))

    def test_retry_success_is_recorded_and_crop_saved(self):
        a = FakeAligner(ValueError('face_not_detected'))
        output = self.root / 'retry_success'
        report = prepare(self.data, output, a, limit_per_source=1, sources=['mega'])
        self.assertEqual(report['counts']['train']['mega']['statuses'], {'retry_success': 1})
        self.assertEqual(report['counts']['train']['mega']['black_pixel_ratio']['retry_success']['mean'], 0.0)
        row = read_csv(output / 'audit.csv')[0]
        self.assertEqual(row['status'], 'retry_success')
        self.assertEqual(row['disposition'], 'accepted')
        self.assertTrue(Path(read_csv(output / 'train.csv')[0]['path']).is_file())

    def test_unselected_source_root_cannot_be_output(self):
        sources = self.data / 'sources'
        sources.mkdir()
        (sources / 'linked').symlink_to(self.images, target_is_directory=True)
        with self.assertRaises(ValueError):
            prepare(self.data, self.images / 'nested' / 'output', FakeAligner(), sources=['missing'])


if __name__ == '__main__':
    unittest.main()
