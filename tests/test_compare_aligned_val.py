import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from age import compare_aligned_val as compare
from age.common import FIELDS, write_csv


class CompareAlignedValTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.orig = self.root / 'original' / 'val.csv'
        self.aligned = self.root / 'aligned' / 'val.csv'
        self.orig.parent.mkdir()
        (self.aligned.parent / 'images' / 'val').mkdir(parents=True)
        self.rows = []
        self.crops = []
        for i, age in enumerate((20, 35)):
            path = self.orig.parent / f'{i}.png'
            crop = self.aligned.parent / 'images' / 'val' / f'{i:08}.png'
            Image.new('RGB', (4, 4)).save(path)
            Image.new('RGB', (4, 4)).save(crop)
            row = dict(path=str(path), age=str(age), gender='', source='source', person_id='',
                       original_id=f'id{i}', sha256=f'hash{i}')
            self.rows.append(row)
            self.crops.append(dict(row, path=str(crop), sha256=f'crop{i}'))
        self.save()

    def save(self):
        write_csv(self.orig, self.rows, FIELDS)
        write_csv(self.aligned, self.crops, FIELDS)

    def test_join_preserves_aligned_order_and_subset(self):
        self.crops = [self.crops[1]]
        self.save()
        pairs = compare.paired_rows(self.orig, self.aligned)
        self.assertEqual([(a['original_id'], b['original_id']) for a, b in pairs], [('id1', 'id1')])

    def test_rejects_duplicate_missing_and_mismatched_labels(self):
        for change in ('duplicate', 'unknown', 'age', 'gender', 'path'):
            with self.subTest(change=change):
                original = [dict(r) for r in self.crops]
                if change == 'duplicate':
                    self.crops.append(dict(self.crops[0]))
                elif change == 'unknown':
                    self.crops[0]['original_id'] = 'unknown'
                elif change in ('age', 'gender'):
                    self.crops[0][change] = '99'
                else:
                    self.crops[0]['path'] = self.crops[1]['path']
                self.save()
                with self.assertRaises(ValueError):
                    compare.paired_rows(self.orig, self.aligned)
                self.crops = original
        self.save()

    def test_rejects_non_val_and_test_paths_before_opening_images(self):
        with self.assertRaisesRegex(ValueError, 'Only val.csv'):
            compare.paired_rows(self.orig.with_name('train.csv'), self.aligned)
        self.rows[0]['path'] = str(self.root / 'test' / 'image.png')
        self.save()
        with self.assertRaisesRegex(ValueError, 'Test image path'):
            compare.paired_rows(self.orig, self.aligned)

    def test_rejects_duplicate_original_and_checkpoint_val_mismatch(self):
        self.rows.append(dict(self.rows[0]))
        self.save()
        with self.assertRaisesRegex(ValueError, 'Duplicate original'):
            compare.paired_rows(self.orig, self.aligned)
        self.rows.pop()
        self.save()
        checkpoint = self.root / 'model.pt'
        checkpoint.write_bytes(b'checkpoint')
        output = self.root / 'result'
        args = ['--original-csv', str(self.orig), '--aligned-csv', str(self.aligned),
                '--checkpoint', str(checkpoint), '--output', str(output)]
        with patch.object(compare.torch, 'load', return_value={'run': {'data_sha256': {'val': 'wrong'}}}):
            with self.assertRaisesRegex(ValueError, 'validation CSV hash'):
                compare.main(args)
        self.assertFalse(output.exists())

    def test_output_protection_and_end_to_end_metrics(self):
        output = self.root / 'result'
        output.mkdir()
        (output / 'metrics.csv').write_text('untouched')
        args = ['--original-csv', str(self.orig), '--aligned-csv', str(self.aligned),
                '--checkpoint', str(self.root / 'model.pt'), '--output', str(output)]
        with self.assertRaises(FileExistsError):
            compare.main(args)
        self.assertEqual((output / 'metrics.csv').read_text(), 'untouched')
        args[-1] = str(self.root / 'new-result')

        class Model:
            def to(self, device):
                return self

            def load_state_dict(self, state):
                pass

        def fake_predict(model, loader, device, flip):
            return [dict(source=r['source'], age=float(r['age']), prediction=float(r['age']) + 5)
                    for r in loader.dataset.rows]

        checkpoint = {'run': {'data_sha256': {'val': compare.sha256(self.orig)}}, 'model': {}}
        with patch.object(compare.torch, 'load', return_value=checkpoint), \
             patch.object(compare, 'sha256', side_effect=lambda path: 'checkpoint' if path == self.root / 'model.pt' else __import__('age.common', fromlist=['sha256']).sha256(path)), \
             patch.object(compare, 'build_model', return_value=Model()), \
             patch.object(compare, 'predict', side_effect=fake_predict):
            compare.main(args)
        result = self.root / 'new-result'
        report = json.loads((result / 'summary.json').read_text())
        self.assertEqual(report['count'], 2)
        self.assertEqual(report['metrics']['original'], report['metrics']['aligned'])
        self.assertEqual(report['metrics']['original'][0]['mae'], 5)
        self.assertEqual(report['metrics']['original'][0]['cs_at_5'], 1)
        with (result / 'predictions.csv').open() as f:
            predictions = list(csv.DictReader(f))
        self.assertEqual([r['original_id'] for r in predictions], ['id0', 'id1', 'id0', 'id1'])


if __name__ == '__main__':
    unittest.main()
