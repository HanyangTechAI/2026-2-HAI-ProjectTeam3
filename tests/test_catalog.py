import csv
import hashlib
from pathlib import Path
import tempfile
import unittest

from age.catalog import FIELDS, age_band, build_catalog


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def add(self, dataset, sample='1', age=25, split='', content=b'image', path=None):
        root = self.root / dataset
        root.mkdir(exist_ok=True)
        name = path or f'{sample}.jpg'
        if path is None:
            (root / name).write_bytes(content)
        row = dict(dataset=dataset, sample_id=sample, path=name, age=age,
                   age_label_type='reported', gender='', person_id='', official_split=split,
                   sha256=hashlib.sha256(content).hexdigest(), width=300, height=300, mode='RGB')
        manifest = root / 'samples.csv'
        exists = manifest.exists()
        with manifest.open('a', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS[:-2])
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def test_classifies_permissions_official_test_and_deduplicates(self):
        self.add('aaf', content=b'aaf')
        self.add('afad', '1', content=b'adult')
        self.add('afad', '2', content=b'adult')
        self.add('megaage_asian', '1', split='train', content=b'train')
        self.add('megaage_asian', '2', split='test', content=b'test')
        self.add('fgnet', content=b'fgnet')
        summary = build_catalog(self.root, self.root / 'catalog', True)
        self.assertEqual(summary['selected_counts']['generation_candidates.csv'], 2)
        self.assertEqual(summary['selected_counts']['evaluation_only.csv'], 2)
        self.assertEqual(summary['usage_counts']['permission_review'], 1)
        self.assertEqual(summary['usage_counts']['exact_duplicate_excluded'], 1)
        self.assertEqual(summary['datasets']['afad']['official_splits'], {'unsplit': 2})
        with (self.root / 'catalog/all_samples.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[0]['path'], 'aaf/1.jpg')

    def test_test_overlap_excluded_and_conflicting_labels_reviewed(self):
        self.add('afad', '1', content=b'heldout')
        self.add('megaage_asian', '1', split='test', content=b'heldout')
        self.add('afad', '2', age=20, content=b'conflict')
        self.add('afad', '3', age=30, content=b'conflict')
        summary = build_catalog(self.root, self.root / 'catalog')
        self.assertEqual(summary['usage_counts']['heldout_duplicate_excluded'], 1)
        self.assertEqual(summary['usage_counts']['conflicting_labels_review'], 2)
        self.assertEqual(summary['selected_counts']['generation_candidates.csv'], 0)

    def test_rejects_escaping_path(self):
        self.add('afad', path='../outside.jpg')
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            build_catalog(self.root, self.root / 'catalog')

    def test_normalizes_heldout_splits_without_overriding_source_labels(self):
        self.add('afad', '1', split='validation', content=b'validation')
        self.add('afad', '2', split='test ', content=b'test')
        self.add('afad', '3', content=b'mega-test')
        self.add('megaage_asian', split=' TEST ', content=b'mega-test')
        self.add('afad', '4', split='unrecognized', content=b'unknown')
        self.add('aaf', split='val', content=b'aaf-heldout')
        self.add('afad', '5', content=b'aaf-heldout')
        summary = build_catalog(self.root, self.root / 'catalog', True)
        self.assertEqual(summary['selected_counts']['generation_candidates.csv'], 0)
        self.assertEqual(summary['usage_counts']['heldout_duplicate_excluded'], 2)
        self.assertEqual(summary['usage_counts']['split_review'], 1)
        self.assertEqual(summary['selected_counts']['evaluation_only.csv'], 3)
        with (self.root / 'catalog/all_samples.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(next(r for r in rows if r['dataset'] == 'megaage_asian')['official_split'], ' TEST ')

    def test_rejects_hash_mismatch(self):
        self.add('afad')
        (self.root / 'afad/1.jpg').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            build_catalog(self.root, self.root / 'catalog', True)

    def test_rejects_duplicate_ids_and_existing_output(self):
        self.add('afad')
        with self.assertRaises(FileExistsError):
            build_catalog(self.root, self.root / 'afad')
        self.add('afad', path='another.jpg')
        (self.root / 'afad/another.jpg').write_bytes(b'image')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            build_catalog(self.root, self.root / 'catalog')

    def test_age_band_boundaries_and_empty_input(self):
        self.assertEqual([age_band(a) for a in (0, 2, 3, 69, 70, 120)],
                         ['0-2', '0-2', '3-6', '50-69', '70-120', '70-120'])
        with self.assertRaisesRegex(ValueError, 'No acquired'):
            build_catalog(self.root, self.root / 'catalog')


if __name__ == '__main__':
    unittest.main()
