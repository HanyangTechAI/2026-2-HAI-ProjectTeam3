import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from .align_public import prepare
from .common import FIELDS, read_csv, write_csv


class FakeAligner:
    def align(self, rgb, training=False):
        assert training
        if rgb[0, 0, 0] == 3:
            raise ValueError('face_not_detected')
        # Two different inputs can produce identical aligned faces.
        return np.full((224, 224, 3), rgb[0, 0, 0] % 2, dtype=np.uint8), None


class AlignPublicTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / 'input'
        self.images = self.root / 'originals'
        self.data.mkdir()
        self.images.mkdir()
        for name, value in [('a', 1), ('b', 2), ('c', 5), ('d', 3)]:
            Image.fromarray(np.full((20, 20, 3), value, dtype=np.uint8)).save(self.images / f'{name}.png')
        def row(name):
            return dict(path=str(self.images / f'{name}.png'), age='25', gender='male', source='afad',
                        person_id='', original_id=name, sha256='original-' + name)
        write_csv(self.data / 'train.csv', [row('a'), row('b'), row('d')], FIELDS)
        write_csv(self.data / 'val.csv', [row('c')], FIELDS)
        # A deliberately invalid test manifest ensures the code never opens it.
        (self.data / 'test.csv').mkdir()

    def test_collisions_failures_and_immutable_rerun(self):
        output = self.root / 'output'
        report = prepare(self.data, output, FakeAligner())
        self.assertEqual(report['counts']['train'], dict(input=3, output=1, failures=1, cross_split_excluded=1))
        self.assertEqual(report['counts']['val'], dict(input=1, output=0, failures=0, cross_split_excluded=1))
        self.assertEqual(report['failure_reasons'], {'face_not_detected': 1})
        self.assertEqual(len(read_csv(output / 'cross_split_duplicates.csv')), 2)
        row = read_csv(output / 'train.csv')[0]
        self.assertEqual(list(row), FIELDS)
        self.assertEqual(row['original_id'], 'b')
        self.assertEqual(row['age'], '25')
        self.assertTrue(Path(row['path']).is_file())
        self.assertNotEqual(row['sha256'], 'original-b')
        with self.assertRaises(ValueError):
            prepare(self.data, output, FakeAligner())
        other = self.root / 'output2'
        prepare(self.data, other, FakeAligner())
        self.assertEqual(read_csv(other / 'train.csv')[0]['sha256'], row['sha256'])
        self.assertTrue((self.images / 'a.png').is_file())

    def test_limit_and_unsafe_output(self):
        report = prepare(self.data, self.root / 'pilot', FakeAligner(), limit=1)
        self.assertEqual(report['counts']['train']['input'], 1)
        self.assertEqual(report['counts']['val']['input'], 1)
        for path in (self.data / 'out', self.images / 'out', self.root):
            with self.subTest(path=path), self.assertRaises(ValueError):
                prepare(self.data, path, FakeAligner())

    def test_symlinked_source_root_is_protected_even_for_limited_pilot(self):
        sources = self.data / 'sources'
        sources.mkdir()
        (sources / 'dataset').symlink_to(self.images, target_is_directory=True)
        with self.assertRaises(ValueError):
            prepare(self.data, self.images / 'nested' / 'output', FakeAligner(), limit=1)


if __name__ == '__main__':
    unittest.main()
