import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from age.ffhq import GROUPS, classification_metrics, main, predict_groups
from age.model import build_model


class FFHQTrainingContracts(unittest.TestCase):
    def test_classification_metrics_are_not_exact_age_mae(self):
        metrics = classification_metrics([0, 0, 1, 9], [0, 1, 1, 9])
        self.assertEqual(metrics['accuracy'], .75)
        self.assertAlmostEqual(metrics['macro_f1'], (2 / 3 + 2 / 3 + 1) / 10)
        self.assertAlmostEqual(metrics['balanced_accuracy'], 2.5 / 3)
        self.assertIsNone(metrics['age_mae'])
        self.assertEqual(metrics['confusion_matrix'][0][1], 1)
        self.assertEqual(metrics['per_group'][9]['age_group'], '70-120')

    def test_invalid_class_indices_and_empty_metrics_rejected(self):
        for targets, predictions in [([], []), ([0], []), ([10], [0]), ([0], [-1]), ([0.5], [0])]:
            with self.subTest(targets=targets, predictions=predictions), self.assertRaises(ValueError):
                classification_metrics(targets, predictions)

    def test_native_head_and_existing_exact_age_head(self):
        self.assertEqual(build_model(False).classifier[-1].out_features, 101)
        self.assertEqual(build_model(False, len(GROUPS)).classifier[-1].out_features, 10)

    def test_prediction_uses_eval_mode_and_has_no_age_conversion(self):
        logits = torch.eye(10)[:2] * 5
        model = torch.nn.Identity()
        model.train()
        metrics, predictions, confidence = predict_groups(model, [(logits, torch.tensor([0, 1]))], 'cpu')
        self.assertFalse(model.training)
        self.assertEqual(predictions, [0, 1])
        self.assertEqual(metrics['accuracy'], 1)
        self.assertTrue(all(0 < value < 1 for value in confidence))

    def test_output_reuse_and_checkpoint_contract(self):
        with TemporaryDirectory() as directory:
            with patch('age.ffhq.load_records') as load:
                with self.assertRaises(FileExistsError):
                    main(['train', '--output', directory])
                load.assert_not_called()
            for action, extra in [('train', ['--checkpoint', 'some.pt']), ('evaluate', [])]:
                with self.assertRaises(SystemExit):
                    main([action, '--output', str(Path(directory) / 'new'), *extra])


if __name__ == '__main__':
    unittest.main()
