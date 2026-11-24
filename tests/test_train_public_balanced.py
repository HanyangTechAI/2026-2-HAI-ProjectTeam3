import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from age.train_public_balanced import main, parse_args, sampling_weights


class PublicBalancedTests(unittest.TestCase):
    def test_source_decade_weights_and_cap(self):
        rows = [{'source': 'afad', 'age': '20'} for _ in range(100)]
        rows += [{'source': 'megaage_asian', 'age': '20'}] * 4
        rows += [{'source': 'afad', 'age': '95'}]
        weights = sampling_weights(rows)
        self.assertAlmostEqual(weights[100] / weights[0], 5)
        self.assertLess(weights[-1], 10 * weights[0])
        raw_mean = (100 * 0.1 + 4 * 0.5 + 1) / 105
        self.assertLessEqual(max(weights), 5 * raw_mean)
        self.assertTrue(all(math.isfinite(w) and w > 0 for w in weights))

    def test_invalid_sampler(self):
        with self.assertRaises(ValueError):
            sampling_weights([])
        with self.assertRaises(ValueError):
            sampling_weights([{'source': 'afad', 'age': '20'}], 0.5)

    def test_requires_new_output_outside_data(self):
        with patch('sys.stderr'):
            with self.assertRaises(SystemExit):
                parse_args(['--data-dir', '/tmp/data', '--output', '/tmp/data/experiment'])
            with self.assertRaises(SystemExit):
                parse_args(['--output', '/tmp/new', '--batch-size', '1'])
        args = parse_args(['--data-dir', '/tmp/data', '--output', '/tmp/experiment'])
        self.assertEqual(args.epochs, 30)

    def test_entrypoint_reads_only_train_val_before_model(self):
        class FakeDataset:
            def __init__(self, path, training=False):
                self.path = path
                self.rows = [{'source': 'afad', 'age': '20'},
                             {'source': 'megaage_asian', 'age': '40'}]

            def __len__(self):
                return len(self.rows)

        with tempfile.TemporaryDirectory() as tmp:
            paths = []

            def dataset(path, training=False):
                paths.append((path.name, training))
                return FakeDataset(path, training)

            with patch('age.train_public_balanced.AgeDataset', side_effect=dataset), \
                 patch('age.train_public_balanced.validate_splits'), \
                 patch('age.train_public_balanced.sha256', return_value='digest'), \
                 patch('age.train_public_balanced.build_model', side_effect=RuntimeError('stop before training')):
                with self.assertRaisesRegex(RuntimeError, 'stop before training'):
                    main(['--data-dir', str(Path(tmp) / 'data'), '--output', str(Path(tmp) / 'out'),
                          '--workers', '0', '--device', 'cpu'])
            self.assertEqual(paths, [('train.csv', True), ('val.csv', False)])


if __name__ == '__main__':
    unittest.main()
