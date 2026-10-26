"""CPU-only contracts: similarity candidates are not identities or test-based selection."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from age.common import sha256
from gen import FaceNotFoundError
from gen.audit_lora_split import (
    AGE_GROUPS, GENDERS, ORIGINAL_CAPTIONS, TARGETS, cross_split_candidates,
    embed_record, group_counts, load_records, save_manifest, scan, select_validation, stable_key, write_report,
)


def record(name, split='val', age='20-29', gender='female'):
    return dict(sample_id=name, split=split, age_group=age, gender=gender,
                path=f'/fixture/{split}/images/{name}.png',
                sha256=hashlib.sha256(name.encode()).hexdigest(), caption=ORIGINAL_CAPTIONS[age, gender])


def embedding(values=(1., 0.), status='ok', default_status='ok'):
    return dict(embedding=list(values) + [0.] * (512 - len(values)) if status == 'ok' else None,
                status=status, default_status=default_status, detector_size=640 if default_status == 'ok' else 320)


class CandidateContracts(unittest.TestCase):
    def test_cross_split_scores_normalized_and_candidates_never_identities(self):
        rows = [record('train_a', 'train'), record('train_b', 'train'),
                record('val'), record('test', 'test')]
        embeddings = dict(train_a=embedding((2., 0.)), train_b=embedding((0., 0., 1.)),
                          val=embedding((1., math.sqrt(3))), test=embedding((-1., 0.)))
        candidates, nearest, stats = cross_split_candidates(rows, embeddings, .5)
        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual((candidate['sample_id_a'], candidate['sample_id_b']), ('train_a', 'val'))
        self.assertAlmostEqual(candidate['cosine'], .5, places=6)
        self.assertEqual(candidate['relation'], 'candidate')
        self.assertFalse(candidate['verified_identity'])
        self.assertTrue(candidate['affects_fixed_val_selection'])
        self.assertEqual(stats['train-val']['comparisons'], 2)
        self.assertTrue(all(r['split'] != r['neighbor_split'] for r in nearest))
        self.assertEqual(len(nearest), 8)

    def test_test_hits_are_screening_only_and_metadata_does_not_filter_similarity(self):
        rows = [record('a', 'train', '0-2', 'male'), record('b', 'val', '70-120', 'female'),
                record('c', 'test', '20-29', 'female')]
        candidates, _, _ = cross_split_candidates(rows, {r['sample_id']: embedding() for r in rows})
        self.assertEqual(len(candidates), 3)
        for row in candidates:
            self.assertEqual(row['affects_fixed_val_selection'], {row['split_a'], row['split_b']} == {'train', 'val'})
            self.assertFalse(row['verified_identity'])

    def test_failed_embeddings_are_not_zero_vectors_or_false_negatives(self):
        rows = [record('a', 'train'), record('b'), record('c', 'test')]
        values = dict(a=embedding(), b=embedding(status='face_not_detected', default_status='face_not_detected'),
                      c=embedding())
        candidates, nearest, stats = cross_split_candidates(rows, values)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(stats['train-val']['comparisons'], 0)
        self.assertEqual(len(nearest), 2)

    def test_invalid_threshold_and_invalid_cache_vectors_fail(self):
        rows = [record('a', 'train'), record('b')]
        values = {r['sample_id']: embedding() for r in rows}
        for threshold in (float('nan'), float('inf'), 1.1, -1.1):
            with self.assertRaises(ValueError):
                cross_split_candidates(rows, values, threshold)
        for bad in ((0., 0.), (float('nan'), 0.)):
            with self.assertRaises(ValueError):
                cross_split_candidates(rows, values | {'a': embedding(bad)})


class FixedValidationContracts(unittest.TestCase):
    def setUp(self):
        self.rows = [record(f'val_{i}_{j}_{k}', 'val', age, gender)
                     for i, age in enumerate(AGE_GROUPS) for j, gender in enumerate(GENDERS) for k in range(3)]
        self.values = {r['sample_id']: embedding() for r in self.rows}

    def test_deterministic_seed42_hash_stratification_independent_of_input_order(self):
        chosen, cells = select_validation(self.rows, self.values, [])
        reversed_chosen, _ = select_validation(list(reversed(self.rows)), self.values, [])
        self.assertEqual(chosen, reversed_chosen)
        self.assertEqual(len(chosen), 20)
        self.assertEqual({(r['age_group'], r['gender']) for r in chosen},
                         {(age, gender) for age in AGE_GROUPS for gender in GENDERS})
        self.assertTrue(all(c['eligible'] == 3 for c in cells))
        for row in chosen:
            pool = [r for r in self.rows if (r['age_group'], r['gender']) == (row['age_group'], row['gender'])]
            self.assertEqual(row, min(pool, key=lambda r: stable_key(r, 42)))

    def test_train_gate_but_never_test_gate_both_pair_orientations(self):
        initial, _ = select_validation(self.rows, self.values, [])
        name = initial[0]['sample_id']
        test_pair = dict(split_a='val', sample_id_a=name, split_b='test', sample_id_b='test_1', cosine=1.)
        unchanged, _ = select_validation(self.rows, self.values, [test_pair])
        self.assertEqual(initial, unchanged)
        train_pair = test_pair | dict(split_b='train', sample_id_b='train_1', cosine=.5)
        changed, _ = select_validation(self.rows, self.values, [test_pair, train_pair])
        self.assertNotIn(name, [r['sample_id'] for r in changed])
        reversed_pair = dict(split_a='train', sample_id_a='train_1', split_b='val', sample_id_b=name, cosine=.5)
        self.assertEqual(changed, select_validation(self.rows, self.values, [reversed_pair])[0])
        self.assertEqual(initial, select_validation(self.rows, self.values, [train_pair | {'cosine': .4999}])[0])

    def test_default_face_failure_ineligible_even_when_fallback_embedding_available(self):
        initial, _ = select_validation(self.rows, self.values, [])
        failed = initial[0]['sample_id']
        self.values[failed] = embedding(default_status='face_not_detected')
        chosen, cells = select_validation(self.rows, self.values, [])
        self.assertNotIn(failed, [r['sample_id'] for r in chosen])
        self.assertEqual(cells[0]['default_detectable'], 2)

    def test_missing_stratum_fails_instead_of_substituting_train_or_test(self):
        missing = self.rows[0]['age_group'], self.rows[0]['gender']
        rows = [r for r in self.rows if (r['age_group'], r['gender']) != missing]
        for split in ('train', 'test'):
            extra = record(split, split, *missing)
            rows.append(extra)
            self.values[split] = embedding()
        with self.assertRaisesRegex(ValueError, 'No default-face-detectable'):
            select_validation(rows, self.values, [])


class OriginalProvenanceContracts(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.data = self.root / 'data'
        self.sources = []
        for i, split in enumerate(('train', 'val', 'test')):
            directory = self.data / split
            (directory / 'images').mkdir(parents=True)
            path = directory / 'images' / f'{i:05d}.png'
            Image.new('RGB', (512, 512), (i * 70, 50, 20)).save(path)
            age = ('70-120', '50-69', '20-29')[i]
            caption = ORIGINAL_CAPTIONS[age, 'female']
            (directory / 'metadata.jsonl').write_text(json.dumps(dict(file_name=f'images/{path.name}', text=caption)) + '\n')
            self.sources.append(dict(image_number=path.stem, path=str(path), split=split, age_group=age,
                                     gender='female', caption=caption, sha256=sha256(path),
                                     age_group_confidence='1', head_yaw='0',
                                     attribution=dict(author='Fixture only', photo_url='https://example.invalid/image',
                                                      license='Fixture only', license_url='https://example.invalid/license')))
        self.provenance = self.root / 'provenance.jsonl'
        self.write_sources()

    def write_sources(self):
        self.provenance.write_text(''.join(json.dumps(r) + '\n' for r in self.sources))

    def test_original_age_labels_and_captions_preserved_not_new_prompt_mapping(self):
        rows = load_records(self.data, self.provenance, self.root)
        self.assertEqual(rows[0]['age_group'], '70-120')
        self.assertIn('70s', rows[0]['caption'])
        self.assertEqual(rows[1]['age_group'], '50-69')
        self.assertIn('50s or 60s', rows[1]['caption'])
        counts = group_counts(rows)
        self.assertEqual(len(counts), 60)
        self.assertEqual(sum(r['count'] for r in counts), 3)

    def test_provenance_gender_split_hash_and_age_mismatch_rejected(self):
        original = copy.deepcopy(self.sources)
        for key, value in (('gender', 'male'), ('split', 'val'), ('sha256', '0' * 64), ('age_group', '50-69')):
            self.sources = copy.deepcopy(original)
            self.sources[0][key] = value
            self.write_sources()
            with self.assertRaisesRegex(ValueError, 'metadata/provenance mismatch'):
                load_records(self.data, self.provenance, self.root)

    def test_path_escape_and_duplicate_provenance_rejected(self):
        self.sources.append(self.sources[0])
        self.write_sources()
        with self.assertRaisesRegex(ValueError, 'Duplicate provenance'):
            load_records(self.data, self.provenance, self.root)
        self.sources.pop()
        self.write_sources()
        path = self.data / 'train/metadata.jsonl'
        path.write_text(json.dumps(dict(file_name='../val/images/00001.png', text='invalid')) + '\n')
        with self.assertRaisesRegex(ValueError, 'outside original split'):
            load_records(self.data, self.provenance, self.root)

    def test_manifest_original_copies_and_hashes_with_fixed_selection_metadata(self):
        val = [r for r in load_records(self.data, self.provenance, self.root) if r['split'] == 'val']
        output = self.root / 'preflight'
        output.mkdir()
        manifest = save_manifest(val, output)
        self.assertEqual(manifest['source_split'], 'val')
        self.assertEqual(manifest['targets'], TARGETS)
        self.assertEqual(manifest['seed'], 42)
        self.assertFalse(manifest['selection']['test_used_for_selection'])
        sample = manifest['samples'][0]
        self.assertEqual(set(sample), {'sample_id', 'path', 'input_path', 'age_group', 'gender', 'sha256'})
        self.assertTrue(Path(sample['path']).is_absolute())
        self.assertTrue(Path(sample['input_path']).is_absolute())
        self.assertEqual(Path(sample['path']).read_bytes(), Path(sample['input_path']).read_bytes())
        self.assertEqual(save_manifest(val, output), manifest)
        Path(sample['input_path']).write_bytes(b'changed copy')
        with self.assertRaisesRegex(ValueError, 'overwrite changed'):
            save_manifest(val, output)

    def test_audit_fallback_records_default_failure_and_restores_detector(self):
        row = load_records(self.data, self.provenance, self.root)[0]
        detector = SimpleNamespace(input_size=(640, 640))
        calls = []

        def fake_embedding(image):
            calls.append(detector.input_size)
            if detector.input_size == (640, 640):
                raise FaceNotFoundError('face_not_detected')
            return embedding()['embedding']

        faces = SimpleNamespace(aligner=SimpleNamespace(detector=detector), embedding=fake_embedding)
        with patch('gen.audit_lora_split._faces', faces), patch('gen.audit_lora_split._fallback_size', 320):
            result = embed_record(row)
        self.assertEqual(calls, [(640, 640), (320, 320)])
        self.assertEqual(result['default_status'], 'face_not_detected')
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['detector_size'], 320)
        self.assertEqual(detector.input_size, (640, 640))
        self.assertEqual(result['sha256'], row['sha256'])

    def test_double_detection_failure_is_cached_explicitly_not_success(self):
        row = load_records(self.data, self.provenance, self.root)[0]
        detector = SimpleNamespace(input_size=(640, 640))

        def no_face(image):
            raise FaceNotFoundError('face_not_detected')

        faces = SimpleNamespace(aligner=SimpleNamespace(detector=detector), embedding=no_face)
        with patch('gen.audit_lora_split._faces', faces), patch('gen.audit_lora_split._fallback_size', 320):
            result = embed_record(row)
        self.assertEqual(result['status'], 'face_not_detected')
        self.assertEqual(result['default_status'], 'face_not_detected')
        self.assertIsNone(result['embedding'])
        self.assertEqual(detector.input_size, (640, 640))

    def test_cache_reuse_keeps_failures_and_rejects_hash_or_runtime_mismatch(self):
        rows = load_records(self.data, self.provenance, self.root)
        output = self.root / 'cache'
        output.mkdir()
        context = dict(embedding_contract_version=1, fallback_size=320)
        (output / 'cache_context.json').write_text(json.dumps(context))
        cached = [dict(sample_id=r['sample_id'], path=r['path'], sha256=r['sha256'],
                       status='face_not_detected', default_status='face_not_detected',
                       detector_size=None, embedding=None, seconds=.1, error='face_not_detected') for r in rows]
        cache_file = output / 'embedding_cache.jsonl'
        cache_file.write_text(''.join(json.dumps(r) + '\n' for r in cached))
        with patch('gen.audit_lora_split.cache_context', return_value=context), \
                patch('gen.audit_lora_split.ProcessPoolExecutor') as pool:
            result = scan(rows, self.root, output, 1, 1, 320)
            pool.assert_not_called()
            self.assertEqual(len(result), 3)
            self.assertTrue(all(r['status'] == 'face_not_detected' for r in result.values()))
            cached[0]['sha256'] = '0' * 64
            cache_file.write_text(''.join(json.dumps(r) + '\n' for r in cached))
            with self.assertRaisesRegex(ValueError, 'provenance changed'):
                scan(rows, self.root, output, 1, 1, 320)
        with patch('gen.audit_lora_split.cache_context', return_value=context | {'fallback_size': 0}):
            with self.assertRaisesRegex(ValueError, 'context changed'):
                scan(rows, self.root, output, 1, 1, 0)


class PreflightReportContracts(unittest.TestCase):
    def test_actual_report_and_twenty_copies_keep_all_test_matching_val_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'preflight'
            output.mkdir()
            (output / 'cache_context.json').write_text('{}')
            rows, embeddings = [], {}
            for split in ('train', 'val', 'test'):
                images = root / split / 'images'
                images.mkdir(parents=True)
                for i, age in enumerate(AGE_GROUPS):
                    for j, gender in enumerate(GENDERS):
                        name = f'{split}_{i}_{j}'
                        path = images / f'{name}.png'
                        Image.new('RGB', (512, 512), (i * 20, j * 100, len(rows))).save(path)
                        row = record(name, split, age, gender) | dict(path=str(path), sha256=sha256(path))
                        rows.append(row)
                        embeddings[name] = embedding((1., 0.) if split == 'train' else (0., 1.))
                        embeddings[name].update(error=None, seconds=.01)
            report = write_report(rows, embeddings, output, .5, {})
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(report['selected_samples'], 20)
            self.assertEqual(report['candidate_pairs']['val-test']['candidate_pairs'], 400)
            self.assertEqual(report['candidate_pairs']['train-val']['candidate_pairs'], 0)
            self.assertEqual(report['fixed_val_max_train_cosine'], 0.)
            self.assertFalse(report['identity_disjoint_verified'])
            self.assertFalse(report['test_used_for_selection'])
            self.assertEqual(len(list((output / 'inputs').glob('*.png'))), 20)
            for name in ('report.json', 'report.md', 'candidates.csv', 'nearest_cross_split.csv',
                         'face_failures.csv', 'default_face_failures.csv', 'fixed_val_train_screening.csv',
                         'manifest.json', 'candidates_val_test.jpg'):
                self.assertTrue((output / name).is_file(), name)


if __name__ == '__main__':
    unittest.main()
