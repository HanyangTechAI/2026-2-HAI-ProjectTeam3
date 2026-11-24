import csv
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from age.prepare_public import main


class PreparePublicTest(unittest.TestCase):
    def test_conflicts_duplicates_and_official_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fields = ['dataset', 'sample_id', 'path', 'age', 'gender', 'person_id',
                      'official_split', 'sha256']
            for source in ('afad', 'megaage_asian'):
                directory = root / source
                directory.mkdir()
                records = []
                for name, age, digest, split in (
                    ('a', '20', 'same', 'train'),
                    ('b', '21' if source == 'afad' else '22', 'conflict', 'train'),
                ):
                    (directory / name).write_bytes(b'image')
                    records.append(dict(dataset=source, sample_id=name, path=name, age=age,
                                        gender='', person_id='', official_split=split,
                                        sha256=digest))
                if source == 'megaage_asian':
                    records[0]['official_split'] = 'test'
                with (directory / 'samples.csv').open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(records)
            output = root / 'prepared'
            with patch.object(sys, 'argv', ['prepare_public', '--data-root', str(root), '--output', str(output)]):
                main()
            with (output / 'test.csv').open(newline='') as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['source'], 'megaage_asian')
            self.assertTrue(Path(rows[0]['path']).is_file())
            for split in ('train', 'val'):
                with (output / f'{split}.csv').open(newline='') as f:
                    self.assertEqual(list(csv.DictReader(f)), [])
            with patch.object(sys, 'argv', ['prepare_public', '--data-root', str(root), '--output', str(output)]):
                with self.assertRaises(FileExistsError):
                    main()


if __name__ == '__main__':
    unittest.main()
