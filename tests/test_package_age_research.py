import csv
from pathlib import Path
import tempfile
import unittest

from tools.package_age_research import public_metadata, public_predictions


class PublicArtifactContracts(unittest.TestCase):
    def test_metadata_drops_dataset_rows_and_rebases_paths(self):
        value = {'model': {'weight': 1}, 'split_rows': {'train': [{'age': 20}]},
                 'run': {'data_dir': '/workspace/hai/shared/data/age_public',
                         'output': '/workspace/hai/outputs/age/public',
                         'count': 10}}
        exported = public_metadata(value)
        self.assertNotIn('split_rows', exported)
        self.assertEqual(exported['model'], value['model'])
        self.assertEqual(exported['run']['data_dir'], 'data/age_public')
        self.assertEqual(exported['run']['output'], 'outputs/age/public')
        self.assertIn('split_rows', value)

    def test_predictions_drop_labels_ids_and_original_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, target = [Path(temporary) / name for name in ('input.csv', 'public.csv')]
            source.write_text('source,age,prediction,version,original_id,path\n'
                              'afad,20,30,aligned,person1,/private/photo1.jpg\n'
                              'afad,40,10,aligned,person2,/private/photo2.jpg\n', encoding='utf-8')
            original = source.read_bytes()
            public_predictions(source, target)
            with target.open(newline='') as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)
                self.assertEqual(reader.fieldnames, ['source', 'version', 'prediction'])
            self.assertEqual([row['prediction'] for row in rows], ['10', '30'])
            self.assertEqual(source.read_bytes(), original)
            with self.assertRaises(FileExistsError):
                public_predictions(source, target)

    def test_unknown_prediction_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, target = [Path(temporary) / name for name in ('input.csv', 'public.csv')]
            source.write_text('source,age\nafad,20\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                public_predictions(source, target)
            self.assertFalse(target.exists())


if __name__ == '__main__':
    unittest.main()
