import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from PIL import Image
from torchvision import transforms as T

from age import dataset, train_public_balanced, train_public_weak_crop


class WeakCropTests(unittest.TestCase):
    def test_transform_preserves_full_image_and_existing_color_jitter(self):
        ops = train_public_weak_crop.weak_transform().transforms
        self.assertEqual([type(op) for op in ops],
                         [T.Resize, T.RandomHorizontalFlip, T.ColorJitter, T.ToTensor, T.Normalize])
        self.assertEqual(ops[0].size, (224, 224))
        original_jitter = dataset.transform(True).transforms[2]
        for attr in ('brightness', 'contrast', 'saturation', 'hue'):
            self.assertEqual(getattr(ops[2], attr), getattr(original_jitter, attr))
        self.assertEqual(ops[4].mean, dataset.MEAN)
        self.assertEqual(ops[4].std, dataset.STD)
        image = Image.new('RGB', (320, 160), 'white')
        self.assertEqual(tuple(train_public_weak_crop.weak_transform()(image).shape), (3, 224, 224))
        self.assertTrue(torch.isfinite(train_public_weak_crop.weak_transform()(image)).all())

    def test_entrypoint_only_train_val_and_records_transform(self):
        paths, datasets = [], []

        class FakeDataset:
            def __init__(self, path, training=False):
                paths.append((path.name, training))
                self.rows = [{'source': 'afad', 'age': '20'},
                             {'source': 'megaage_asian', 'age': '40'}]
                self.transform = dataset.transform(training)
                datasets.append(self)

            def __len__(self):
                return len(self.rows)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'experiment'
            with patch.object(train_public_weak_crop, 'AgeDataset', FakeDataset), \
                 patch.object(train_public_weak_crop, 'validate_splits'), \
                 patch.object(train_public_weak_crop, 'sha256', return_value='digest'), \
                 patch.object(train_public_weak_crop, 'build_model', side_effect=RuntimeError('stop before training')):
                with self.assertRaisesRegex(RuntimeError, 'stop before training'):
                    train_public_weak_crop.main(['--data-dir', str(Path(tmp) / 'data'),
                                                 '--output', str(out), '--workers', '0', '--device', 'cpu'])
            self.assertEqual(paths, [('train.csv', True), ('val.csv', False)])
            self.assertIsInstance(datasets[0].transform.transforms[0], T.Resize)
            self.assertEqual([type(op) for op in datasets[1].transform.transforms],
                             [type(op) for op in dataset.transform(False).transforms])
            run = json.loads((out / 'run.json').read_text())
            self.assertEqual(run['train_transform'], 'resize_224x224_horizontal_flip_color_jitter_no_crop')
            self.assertEqual(set(run['data_sha256']), {'train', 'val'})
            self.assertEqual(run['sampler'], 'source_decade_inverse_sqrt_capped_5x_mean')

    def test_training_loop_and_checkpoint_match_balanced_experiment(self):
        def epoch_loop(module):
            tree = ast.parse(Path(module.__file__).read_text())
            main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
            return next(node for node in main.body if isinstance(node, ast.For)
                        and isinstance(node.target, ast.Name) and node.target.id == 'epoch')

        self.assertEqual(ast.dump(epoch_loop(train_public_weak_crop)),
                         ast.dump(epoch_loop(train_public_balanced)))
        self.assertIs(train_public_weak_crop.sampling_weights, train_public_balanced.sampling_weights)
        self.assertIs(train_public_weak_crop.parse_args, train_public_balanced.parse_args)


if __name__ == '__main__':
    unittest.main()
