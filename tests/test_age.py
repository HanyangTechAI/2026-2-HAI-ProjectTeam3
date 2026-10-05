import unittest
from contextlib import contextmanager
import uuid
import shutil
from pathlib import Path
from types import SimpleNamespace
import zipfile
import torch
from age.common import target_ages
from age.common import ROOT
from age.dataset import validate_splits
from age.evaluate import metric_rows
from age.model import build_model, gaussian_targets, expected_age, dldl_loss
from age.prep import split_records, source_records, extract_archives
from age.train import parse_args, select_sources


@contextmanager
def workspace_temp(root):
    # Avoid Windows sandbox ACL issues with tempfile's restrictive directory mode.
    path = (root / uuid.uuid4().hex).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Invalid test path')
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


class AgeContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_root = ROOT / 'outputs/test_temp'
        cls.temp_root.mkdir(parents=True, exist_ok=True)
    def test_boundary_targets_and_gradients(self):
        ages = torch.tensor([0., 50., 100.])
        targets = gaussian_targets(ages)
        self.assertTrue(torch.allclose(targets.sum(-1), torch.ones(3)))
        self.assertTrue(torch.equal(targets.argmax(-1), ages.long()))
        logits = torch.zeros(3, 101, requires_grad=True)
        loss = dldl_loss(logits, ages)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.allclose(expected_age(logits), torch.full((3,), 50.)))

    def test_aihub_default_starts_from_imagenet_without_checkpoint(self):
        args = parse_args([])
        self.assertEqual(args.stage, 'aihub')
        self.assertEqual(args.data_dir, ROOT / 'data/age_aihub')
        self.assertEqual(args.output, ROOT / 'outputs/age/aihub')
        self.assertEqual(args.epochs, 30)
        self.assertEqual(args.lr, 5e-4)
        self.assertFalse(args.no_pretrained)
        self.assertIsNone(args.checkpoint)
        splits = [SimpleNamespace(rows=[dict(source='aihub')]) for _ in range(3)]
        select_sources(splits, args.stage)
        self.assertTrue(all(len(ds.rows) == 1 for ds in splits))

    def test_aihub_only_rejects_mixed_sources_and_empty_splits(self):
        mixed = [SimpleNamespace(rows=[dict(source='aihub'), dict(source='utkface')])]
        with self.assertRaisesRegex(ValueError, 'rejects other sources'):
            select_sources(mixed, 'aihub')
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            select_sources([SimpleNamespace(rows=[])], 'aihub')
        with self.assertRaisesRegex(ValueError, 'rejects other sources'):
            select_sources([SimpleNamespace(rows=[dict(source='synthetic')])], 'aihub')

    def test_no_person_leakage_and_reproducibility(self):
        rows = [dict(path=f'{p}_{i}.png', age=i, source='aihub', person_id=str(p),
                     original_id=f'{p}_{i}', sha256=f'hash_{p}_{i}') for p in range(20) for i in range(3)]
        a, b = split_records(rows), split_records(rows)
        self.assertEqual(a, b)
        self.assertEqual([len(a[s]) for s in ('train', 'val', 'test')], [48, 6, 6])
        validate_splits([SimpleNamespace(rows=r) for r in a.values()])
        # A single person's timeline must never become three independent splits.
        single = split_records(rows[:3])
        self.assertEqual([len(single[s]) for s in ('train', 'val', 'test')], [3, 0, 0])

    def test_pixel_duplicate_connects_groups(self):
        rows = [dict(path=f'{p}.png', source='aihub', person_id=str(p),
                     original_id=str(p), sha256='same' if p < 2 else str(p)) for p in range(10)]
        splits = split_records(rows)
        locations = {r['person_id']: s for s, rs in splits.items() for r in rs}
        self.assertEqual(locations['0'], locations['1'])
        validate_splits([SimpleNamespace(rows=r) for r in splits.values()])

    def test_age_past_is_capture_label(self):
        import json
        with workspace_temp(self.temp_root) as d:
            root = Path(d)
            (root / 'face.png').touch()
            (root / 'face.json').write_text(json.dumps(dict(filename='face', id=1, age_now=31,
                                                           age_past=3, gender='male')))
            rows, failures = source_records(root, 'aihub')
            self.assertEqual(rows[0]['age'], 3)
            self.assertFalse(failures)

    def test_zip_leading_slash_and_traversal(self):
        with workspace_temp(self.temp_root) as d:
            root = Path(d)
            with zipfile.ZipFile(root / 'good.zip', 'w') as z:
                z.writestr('/face.txt', 'valid')
            extract_archives(root, root / 'out')
            self.assertEqual((root / 'out/good/face.txt').read_text(), 'valid')
            with zipfile.ZipFile(root / 'bad.zip', 'w') as z:
                z.writestr('../../outside.txt', 'bad')
            with self.assertRaises(ValueError):
                extract_archives(root, root / 'out')

    def test_metrics_and_targets(self):
        metrics = metric_rows([dict(age=20, prediction=25, source='utkface'),
                               dict(age=30, prediction=40, source='utkface')])
        overall = next(r for r in metrics if r['source'] == 'all')
        self.assertEqual(overall['mae'], 7.5)
        self.assertEqual(overall['cs_at_5'], 0.5)
        self.assertEqual(target_ages(1), [3, 3, 11, 21])
        self.assertEqual(target_ages(95), [85, 90, 90, 90])

    def test_model_forward_backward(self):
        torch.set_num_threads(4)
        model = build_model(False)
        logits = model(torch.randn(2, 3, 224, 224))
        self.assertEqual(tuple(logits.shape), (2, 101))
        dldl_loss(logits, torch.tensor([10., 90.])).backward()
        self.assertTrue(torch.isfinite(model.classifier[-1].weight.grad).all())


if __name__ == '__main__':
    unittest.main()
