import json
import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from age.ffhq_data import GROUPS
from age.group_judge import expected_age, group_of
from gen import FaceNotFoundError
from gen.pairs import TARGET_AGES, build, judge, report, select_sources


def one_hot(index, value=.9):
    probabilities = np.full(len(GROUPS), (1 - value) / (len(GROUPS) - 1))
    probabilities[index] = value
    return probabilities


class FakeTeacher:
    def age(self, images, target_ages, size=512):
        return [Image.new('RGB', (size, size), (age, age, age)) for age in target_ages]


class ExplodingTeacher:
    def age(self, *args, **kwargs):
        raise AssertionError('finished pairs must not be regenerated')


class FakeJudge:
    """Image shade encodes age; target 85 is judged into the wrong group."""
    def probabilities(self, images):
        shades = [image.getpixel((0, 0))[0] for image in images]
        return np.array([one_hot(0 if shade == 85 else group_of(shade)) for shade in shades])


class FakeFaces:
    """Target 5 has no detectable face; target 75 becomes a different person."""
    def embedding(self, image):
        shade = image.getpixel((0, 0))[0]
        if shade == 5:
            raise FaceNotFoundError('no face')
        vector = np.zeros(512)
        vector[1 if shade == 75 else 0] = 1
        return vector.tolist()


def make_records(root, count=2):
    rows = []
    for index in range(count):
        path = Path(root) / f'{index:05d}.png'
        Image.new('RGB', (512, 512), (30, 30, 30)).save(path)
        rows.append(dict(image_number=f'{index:05d}', path=str(path), age_group='30-39', gender='male'))
    return rows


def read_manifest(output):
    return [json.loads(line) for line in (Path(output) / 'manifest.jsonl').read_text(encoding='utf-8').splitlines()]


class PairJudgeContracts(unittest.TestCase):
    def test_judge_rules(self):
        same = [1.0] + [0.0] * 511
        other = [0.0, 1.0] + [0.0] * 510
        hit = one_hot(group_of(45))
        self.assertTrue(judge(same, same, hit, 45)['keep'])
        verdict = judge(same, other, hit, 45)
        self.assertEqual((verdict['keep'], verdict['reasons'], verdict['arcface']), (False, ['identity'], 0.0))
        self.assertEqual(judge(same, None, hit, 45)['reasons'], ['no_face'])
        self.assertEqual(judge(same, same, one_hot(0), 45)['reasons'], ['age_group'])
        self.assertEqual(judge(same, same, hit, 45)['target_group'], '40-49')
        with self.assertRaises(ValueError):
            judge(same, same, [1.0], 45)

    def test_select_sources_is_stratified_and_deterministic(self):
        records = [dict(image_number=f'{i:05d}', age_group=GROUPS[i % 3]) for i in range(30)]
        picked = select_sources(records, 9, seed=1)
        self.assertEqual(picked, select_sources(records, 9, seed=1))
        self.assertEqual(sorted(Counter(row['age_group'] for row in picked).values()), [3, 3, 3])
        self.assertEqual(len(select_sources(records, 100, seed=1)), 30)


class PairBuildContracts(unittest.TestCase):
    settings = dict(generator='sam', target_ages=list(TARGET_AGES))

    def test_build_writes_manifest_and_resumes(self):
        with TemporaryDirectory() as tmp:
            records, output = make_records(tmp), Path(tmp) / 'pairs'
            build(records, FakeTeacher(), FakeJudge(), FakeFaces(), output, self.settings)
            rows = read_manifest(output)
            self.assertEqual(len(rows), 2 * len(TARGET_AGES))
            reasons = {row['target_age']: row['reasons'] for row in rows if row['source_image_number'] == '00000'}
            self.assertEqual((reasons[5], reasons[75], reasons[85], reasons[45]),
                             (['no_face'], ['identity'], ['age_group'], []))
            self.assertEqual(sum(row['keep'] for row in rows), 12)
            self.assertTrue(all(Path(row['output_path']).is_file() for row in rows))
            self.assertAlmostEqual(rows[0]['source_expected_age'], expected_age(one_hot(group_of(30))))
            self.assertEqual(rows[0]['pair_id'], '00000_05')

            build(records, ExplodingTeacher(), FakeJudge(), FakeFaces(), output, self.settings)
            self.assertEqual(len(read_manifest(output)), 2 * len(TARGET_AGES))
            with self.assertRaises(ValueError):
                build(records, FakeTeacher(), FakeJudge(), FakeFaces(), output, dict(self.settings, generator='edit'))

    def test_excluded_and_faceless_sources(self):
        with TemporaryDirectory() as tmp:
            records, output = make_records(tmp), Path(tmp) / 'pairs'
            Image.new('RGB', (512, 512), (5, 5, 5)).save(records[1]['path'])
            build(records, FakeTeacher(), FakeJudge(), FakeFaces(), output, self.settings, exclude={'00000'})
            rows = read_manifest(output)
            self.assertEqual({row['source_image_number'] for row in rows}, {'00001'})
            self.assertEqual(len(rows), len(TARGET_AGES))
            self.assertTrue(all(row['reasons'] == ['source_face'] and not row['keep'] and row['output_path'] is None
                                for row in rows))

    def test_report_summarises_by_target_group(self):
        with TemporaryDirectory() as tmp:
            records, output = make_records(tmp), Path(tmp) / 'pairs'
            build(records, FakeTeacher(), FakeJudge(), FakeFaces(), output, self.settings)
            summary = report(output)
            self.assertEqual((summary['total'], summary['kept']), (18, 12))
            self.assertEqual(summary['reasons'], {'no_face': 2, 'identity': 2, 'age_group': 2})
            self.assertEqual(summary['by_target_group']['70-120'], dict(total=4, kept=0, mean_arcface=0.5))
            self.assertEqual(summary['by_target_group']['40-49']['kept'], 2)
            self.assertTrue((output / 'summary.json').is_file())
            with Image.open(output / 'contact_sheet.png') as sheet:
                self.assertEqual(sheet.size, (128 * (len(TARGET_AGES) + 1), 128 * 2))


if __name__ == '__main__':
    unittest.main()
