import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from PIL import Image

from age.ffhq_data import GROUPS
from age.group_judge import GroupJudge, expected_age, group_of
from age.model import build_model


class GroupJudgeContracts(unittest.TestCase):
    def test_group_boundaries(self):
        cases = {0: '0-2', 2: '0-2', 2.5: '0-2', 3: '3-6', 9.9: '7-9', 10: '10-14', 19: '15-19',
                 20: '20-29', 49: '40-49', 50: '50-69', 69: '50-69', 70: '70-120', 95: '70-120'}
        for age, group in cases.items():
            with self.subTest(age=age):
                self.assertEqual(GROUPS[group_of(age)], group)
        with self.assertRaises(ValueError):
            group_of(-1)

    def test_expected_age_uses_group_midpoints(self):
        probabilities = np.zeros(len(GROUPS))
        probabilities[5], probabilities[6] = .7, .3
        self.assertAlmostEqual(expected_age(probabilities), 24.5 * .7 + 34.5 * .3)
        for invalid in ([.5, .5], np.full(len(GROUPS), .5)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                expected_age(invalid)

    def test_loads_ffhq_checkpoint_and_returns_probabilities(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'best.pt'
            torch.save(dict(task='ffhq_age_group_classification', groups=list(GROUPS),
                            model=build_model(False, len(GROUPS)).state_dict()), path)
            judge = GroupJudge(path)
        images = [Image.new('RGB', (512, 512), (i * 60, 90, 120)) for i in range(2)]
        probabilities = judge.probabilities(images)
        self.assertEqual(probabilities.shape, (2, len(GROUPS)))
        np.testing.assert_allclose(probabilities.sum(1), 1, rtol=1e-5)

    def test_rejects_other_checkpoints(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'best.pt'
            torch.save(dict(task='exact_age', model=build_model(False).state_dict()), path)
            with self.assertRaises(ValueError):
                GroupJudge(path)


if __name__ == '__main__':
    unittest.main()
