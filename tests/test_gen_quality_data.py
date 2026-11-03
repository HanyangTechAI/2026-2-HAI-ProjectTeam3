"""Synthetic CPU preparation contracts, not real-data/model-quality evidence."""
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from gen.prepare_quality_data import (AGE_GROUPS, GENDERS, ORIGINAL_CAPTIONS,
                                     DeadlineReached, caption_digest, digest,
                                     load_train, main, prepare, range_caption,
                                     sampling_plan)
from test_gen_faces import detection, tools


class QualityDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='quality-data-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'train'
        (self.source / 'images').mkdir(parents=True)
        self.provenance = self.root / 'provenance.jsonl'
        self.metadata, self.origins = [], []
        groups = [('0-2', 'female'), ('70-120', 'male'), ('0-2', 'female'),
                  ('10-14', 'female'), ('50-69', 'male')]
        for index, (age, gender) in enumerate(groups):
            number = f'{index:05d}'
            name = f'images/{number}.png'
            Image.new('RGB', (512, 512), (17 + index, 29, 43)).save(self.source / name)
            original = ORIGINAL_CAPTIONS[age, gender]
            caption = 'a photo of a elderly man aged 70 or older' if age == '70-120' else original
            self.metadata.append(dict(file_name=name, text=caption, caption=caption,
                                      caption_sha256=caption_digest(caption),
                                      unrelated={'keep': [index, {'caption': 'historical nested provenance'}]}))
            self.origins.append(dict(image_number=number, path=f'data/lora/train/{name}',
                                     split='train', age_group=age, gender=gender, caption=original,
                                     sha256=digest(self.source / name), age_group_confidence='1', head_yaw='0',
                                     attribution=dict(author='synthetic', photo_url='fixture://photo',
                                                      license='fixture', license_url='fixture://license'),
                                     archive='synthetic-only', extra={'preserve': True}))
        self.save()

    def save(self):
        (self.source / 'metadata.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in self.metadata))
        self.provenance.write_text(''.join(json.dumps(r) + '\n' for r in self.origins))

    def run_prepare(self, command, name='output', **kwargs):
        return prepare(command, self.source, self.provenance, self.root / name,
                       deadline=time.time() + 120, reserve_seconds=0, **kwargs)

    def rows(self, name='output', file='rows.jsonl'):
        return [json.loads(line) for line in (self.root / name / file).read_text().splitlines()]

    def test_range_captions_use_all_actual_intervals_without_exact_ages(self):
        for age in AGE_GROUPS:
            for gender in GENDERS:
                text = range_caption(age, gender)
                self.assertIn(f'{gender} person', text)
                if age == '70-120':
                    self.assertIn('70 or older', text)
                    self.assertNotIn('70s', text)
                else:
                    self.assertIn(age.replace('-', ' to '), text)
                self.assertNotIn('preteen', text)
        for labels in [('55', 'male'), ('50-59', 'female'), ('70-120', 'unknown')]:
            with self.assertRaises(ValueError):
                range_caption(*labels)

    def test_caption_full_independent_copy_preserves_fields_source_and_hashes(self):
        before = {p: p.read_bytes() for p in self.source.rglob('*') if p.is_file()}
        original_provenance = self.provenance.read_bytes()
        report = self.run_prepare('captions')
        self.assertEqual(report['status'], 'prepared', report)
        self.assertEqual(report['changed_captions'], 5)
        self.assertTrue(report['train_published'])
        self.assertFalse((self.root / 'output/.pending_train').exists())
        result = self.rows(file='train/metadata.jsonl')
        snapshot = self.rows(file='source_provenance.jsonl')
        self.assertEqual(snapshot, self.origins)
        self.assertEqual(self.provenance.read_bytes(), original_provenance)
        for index, (old, new) in enumerate(zip(self.metadata, result)):
            expected = deepcopy(old)
            expected['text'] = range_caption(self.origins[index]['age_group'], self.origins[index]['gender'])
            expected['caption'] = expected['text']
            expected['caption_sha256'] = caption_digest(expected['text'])
            self.assertEqual(expected, new)
            src, dst = self.source / old['file_name'], self.root / 'output/train' / old['file_name']
            self.assertEqual(src.read_bytes(), dst.read_bytes())
            self.assertNotEqual(src.stat().st_ino, dst.stat().st_ino)
            self.assertFalse(dst.is_symlink())
        for path, contents in before.items():
            self.assertEqual(path.read_bytes(), contents)
        self.assertEqual(report['metadata_sha256'], digest(self.source / 'metadata.jsonl'))
        self.assertEqual(report['derived_metadata_sha256'], digest(self.root / 'output/train/metadata.jsonl'))
        self.assertTrue(all(r['source_sha256'] == r['copied_sha256'] == r['source_sha256_after'] for r in self.rows()))
        loaded, _ = load_train(self.root / 'output/train', self.provenance)
        self.assertEqual(len(loaded), 5)
        with self.assertRaises(FileExistsError):
            self.run_prepare('captions')

    def test_sampling_deterministic_order_normalization_counts_and_ess(self):
        rows, _ = load_train(self.source, self.provenance)
        plan = sampling_plan(rows)
        self.assertEqual(plan, sampling_plan(rows))
        self.assertFalse(plan['trainer_consumes_plan'])
        self.assertEqual([r['file_name'] for r in plan['rows']], [m['file_name'] for m in self.metadata])
        self.assertEqual([r['index'] for r in plan['rows']], list(range(5)))
        weights = [r['weight'] for r in plan['rows']]
        self.assertAlmostEqual(math.fsum(weights), 1)
        self.assertAlmostEqual(weights[1] / weights[0], math.sqrt(2))
        self.assertAlmostEqual(plan['effective_sample_size'], 1 / math.fsum(w*w for w in weights))
        self.assertGreater(plan['effective_sample_size'], 1)
        self.assertLessEqual(plan['effective_sample_size'], len(rows))
        self.assertEqual(len(plan['groups']), 20)
        self.assertEqual(sum(g['count'] for g in plan['groups']), 5)
        report = self.run_prepare('sampling')
        self.assertEqual(report['status'], 'prepared', report)
        written = json.loads((self.root / 'output/sampling_plan.json').read_text())
        self.assertEqual(written['rows'], plan['rows'])
        self.assertEqual(written['source_metadata_sha256'], digest(self.source / 'metadata.jsonl'))
        self.assertFalse((self.root / 'output/train').exists())
        for power in [float('nan'), float('inf'), -1, 1.1, True]:
            with self.assertRaises(ValueError):
                sampling_plan(rows, power)

    def test_reject_nontrain_source_or_mislabeled_train_rows(self):
        for split in ('val', 'test', 'Test', 'validation'):
            wrong = self.root / split
            (wrong / 'images').mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'train'):
                load_train(wrong, self.provenance)
        self.metadata[0]['split'] = 'test'
        self.save()
        report = self.run_prepare('captions')
        self.assertEqual(report['status'], 'failed')
        self.assertFalse((self.root / 'output/train').exists())

    def test_nontrain_provenance_is_never_a_train_input(self):
        unrelated = deepcopy(self.origins[0])
        unrelated.update(split='test', path='data/lora/test/images/99999.png', image_number='99999')
        self.origins.append(unrelated)
        self.save()
        rows, info = load_train(self.source, self.provenance)
        self.assertEqual(len(rows), 5)
        self.assertEqual(info['ignored_nontrain_provenance_rows'], {'test': 1})
        self.origins[0]['split'] = 'test'
        self.save()
        with self.assertRaisesRegex(ValueError, 'train provenance'):
            load_train(self.source, self.provenance)

    def test_reject_duplicate_unreferenced_traversal_and_fabricated_labels(self):
        original = deepcopy(self.metadata)
        mutations = [lambda: self.metadata.append(deepcopy(self.metadata[0])),
                     lambda: self.metadata[0].update(file_name='../test/images/00000.png'),
                     lambda: self.metadata[0].update(text='a photo of a 55 year old person'),
                     lambda: self.metadata[0].update(age_group='50-69'),
                     lambda: self.metadata[0].update(caption_sha256='0' * 64)]
        for mutate in mutations:
            self.metadata = deepcopy(original)
            mutate()
            self.save()
            with self.assertRaises(ValueError):
                load_train(self.source, self.provenance)
        self.metadata = original
        self.save()
        (self.source / 'images/extra.png').write_bytes(b'not referenced')
        with self.assertRaisesRegex(ValueError, 'inventory'):
            load_train(self.source, self.provenance)

    def test_corrupted_source_or_copy_never_publishes_training_data(self):
        (self.source / self.metadata[-1]['file_name']).write_bytes(b'corrupted')
        report = self.run_prepare('captions')
        self.assertEqual(report['status'], 'failed')
        self.assertFalse(report['train_published'])
        self.assertFalse((self.root / 'output/train').exists())
        self.assertEqual(self.rows()[-1]['status'], 'error')
        self.assertIn('hash differs', self.rows()[-1]['error'])

    def test_copy_checksum_failure_never_publishes_train(self):
        real_digest = digest

        def mismatch(path, check=lambda: None):
            value = real_digest(path, check)
            return 'f' * 64 if '.pending_train' in Path(path).parts and Path(path).suffix == '.png' else value

        with patch('gen.prepare_quality_data.digest', side_effect=mismatch):
            report = self.run_prepare('captions')
        self.assertEqual(report['status'], 'failed')
        self.assertFalse((self.root / 'output/train').exists())
        self.assertIn('copy hash', self.rows()[0]['error'])

    def test_symlinks_missing_provenance_and_duplicate_json_rejected(self):
        image = self.source / self.metadata[0]['file_name']
        outside = self.root / 'outside.png'
        image.rename(outside)
        image.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'nonlocal'):
            load_train(self.source, self.provenance)
        image.unlink()
        outside.rename(image)
        self.origins.pop()
        self.save()
        with self.assertRaisesRegex(ValueError, 'train provenance'):
            load_train(self.source, self.provenance)
        (self.source / 'metadata.jsonl').write_text('{"file_name":"images/00000.png","file_name":"images/00001.png"}\n')
        with self.assertRaisesRegex(ValueError, 'Duplicate JSON key'):
            load_train(self.source, self.provenance)

    def test_caption_interruption_preserves_journal_but_has_no_train(self):
        real_digest = digest

        def interrupt_second_copy(path, check=lambda: None):
            if '.pending_train' in Path(path).parts and Path(path).name == '00001.png':
                raise DeadlineReached('synthetic interrupted copy verification')
            return real_digest(path, check)

        with patch('gen.prepare_quality_data.digest', side_effect=interrupt_second_copy):
            report = self.run_prepare('captions')
        self.assertEqual(report['status'], 'paused_deadline')
        self.assertEqual(report['completed_rows'], 1)
        self.assertEqual(len(self.rows()), 1)
        self.assertFalse((self.root / 'output/train').exists())
        self.assertTrue((self.root / 'output/.pending_train').exists())

    def test_metadata_mutation_during_copy_is_rejected(self):
        real_digest = digest
        changed = []

        def mutate(path, check=lambda: None):
            value = real_digest(path, check)
            if '.pending_train' in Path(path).parts and not changed:
                changed.append(True)
                with (self.source / 'metadata.jsonl').open('a') as stream:
                    stream.write('\n')
            return value

        with patch('gen.prepare_quality_data.digest', side_effect=mutate):
            report = self.run_prepare('captions')
        self.assertEqual(report['status'], 'failed')
        self.assertIn('changed during preparation', report['error'])
        self.assertFalse((self.root / 'output/train').exists())

    def test_preview_default_crop_stratification_and_ambiguous_failure_evidence(self):
        normal = tools([detection()], [])
        ambiguous = tools([detection((38, 116, 218, 396)), detection((294, 116, 474, 396))], [])
        calls = []

        def crop(image, **kwargs):
            self.assertEqual(kwargs, {})
            calls.append(1)
            return (ambiguous if len(calls) == 2 else normal).prepare_edit(image)

        face_tools = Mock(prepare_edit=Mock(side_effect=crop))
        factory = Mock(return_value=face_tools)
        report = self.run_prepare('crop-preview', faces_factory=factory)
        self.assertEqual(report['status'], 'preview_complete_with_failures', report)
        self.assertEqual(report['selected_rows'], 4)
        self.assertEqual(report['completed_rows'], 4)
        self.assertEqual(report['failed_rows'], 1)
        self.assertFalse(report['production_crop_dataset_complete'])
        self.assertFalse((self.root / 'output/train').exists())
        rows = self.rows()
        self.assertEqual([r['index'] for r in rows], [0, 1, 3, 4])
        self.assertEqual(rows[0]['gender'], 'female')
        self.assertEqual(rows[0]['attribute_gender_reference'], 1)
        self.assertEqual(rows[1]['selection_info']['status'], 'ambiguous_primary_face')
        self.assertEqual(rows[1]['selection_info']['detected_count'], 2)
        self.assertIn('ambiguous_primary_face', rows[1]['error'])
        self.assertNotIn('preview_file', rows[1])
        self.assertEqual(len(rows[0]['source_to_crop']), 2)
        self.assertFalse(rows[0]['selected_person_verified'])
        self.assertEqual(rows[0]['preview_sha256'], digest(self.root / 'output' / rows[0]['preview_file']))
        with self.assertRaisesRegex(ValueError, 'named train'):
            prepare('crop-preview', self.source, self.provenance, self.root / 'other/train', deadline=time.time() + 120)

    def test_deadline_retains_completed_records_and_preview_is_incomplete(self):
        normal = tools([detection()], [])
        calls = []

        def crop(image):
            calls.append(1)
            if len(calls) == 2:
                raise DeadlineReached('synthetic deadline')
            return normal.prepare_edit(image)

        report = self.run_prepare('crop-preview', faces_factory=lambda *_: Mock(prepare_edit=crop))
        self.assertEqual(report['status'], 'paused_deadline')
        self.assertEqual(report['completed_rows'], 1)
        self.assertEqual(len(self.rows()), 1)
        self.assertFalse(report['production_crop_dataset_complete'])
        self.assertEqual(json.loads((self.root / 'output/summary.json').read_text()), report)

    def test_expired_deadline_and_cli_invalid_deadline_do_not_start(self):
        report = prepare('captions', self.source, self.provenance, self.root / 'expired', deadline=1)
        self.assertEqual(report['status'], 'paused_deadline')
        self.assertFalse((self.root / 'expired/train').exists())
        with self.assertRaises(SystemExit) as error:
            main(['sampling', '--output', str(self.root / 'bad-cli'), '--deadline', '2030-01-01T00:00:00'])
        self.assertEqual(error.exception.code, 2)
        self.assertFalse((self.root / 'bad-cli').exists())


if __name__ == '__main__':
    unittest.main()
