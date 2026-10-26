"""CPU-only contracts with injected doubles, not evidence of model quality."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from age.common import read_csv, save_json, sha256, write_csv
from gen.compare_lora import (RunClock, TARGETS, combine_results, landmark_geometry,
                              load_manifest, main, procrustes_error, run_comparison, _face_data)
from gen import FaceNotFoundError
from gen.pipeline import PipelineConfig, age_prompt


class FakeClock:
    def __init__(self):
        self.value, self.events = 0.0, []

    def now(self):
        self.events.append('clock')
        return self.value

    def synchronize(self):
        self.events.append('sync')

    def reset_peak(self):
        self.events.append('reset_peak')

    def memory(self):
        return {'max_allocated_bytes': None, 'max_reserved_bytes': None}


class FakeFaces:
    def __init__(self, clock, *, no_faces=(), multiple=(), embedding_errors=(), detection_errors=(), age=None):
        self.clock = clock
        self.no_faces, self.multiple = no_faces, multiple
        self.embedding_errors, self.detection_errors = embedding_errors, detection_errors
        self.age = age
        self.points = [(0, 0), (4, 0), (2, 2), (1, 4), (3, 4)]
        self.aligner = SimpleNamespace(detector=SimpleNamespace(detect=self.detect), select=self.select)
        self.reference_calls = []

    def detect(self, bgr):
        import numpy as np
        self.clock.value += 11
        target = int(bgr[0, 0, 2])
        if target in self.detection_errors:
            raise RuntimeError('detector failure')
        count = 0 if target in self.no_faces else 2 if target in self.multiple else 1
        return np.array([[0, 0, 20, 20, .9]] * count), np.array([self.points] * count)

    def embedding(self, image):
        self.clock.value += 13
        target, condition, _ = image.getpixel((0, 0))
        if target in self.embedding_errors:
            return [0, 0]
        return [1, 0] if target >= 200 else [4, 3] if condition else [3, 4]

    def _select(self, image):
        import numpy as np
        rgb = np.asarray(image)
        boxes, points = self.detect(np.ascontiguousarray(rgb[:, :, ::-1]))
        if not len(boxes):
            raise ValueError('face_not_detected')
        faces = self
        class Selected:
            detected_count = len(boxes)
            kps = points[0]

            @property
            def age(self):
                return faces.select(rgb, training=False).age
        return rgb, Selected()

    def select(self, rgb, training=False):
        self.clock.value += 17
        self.reference_calls.append(training)
        target, condition, _ = map(int, rgb[0, 0])
        value = self.age if self.age is not None else 38 if target >= 200 else target + condition + 2
        return SimpleNamespace(age=value)


class FakePipeline:
    is_mock = False

    def __init__(self, config, mode, clock, *, failures=(), load_error=False):
        self.config, self.mode, self.clock = config, mode, clock
        self.failures, self.load_error = failures, load_error
        self.calls, self.closed = [], False

    def load(self):
        self.clock.events.append('load')
        self.clock.value += 5
        if self.load_error:
            raise RuntimeError('model load failed')

    def age_shift(self, image, targets, gender='unknown', seed=None):
        self.calls.append((image.tobytes(), list(targets), gender, seed))
        self.clock.events.append('generate')
        self.clock.value += 2
        if targets[0] in self.failures:
            raise RuntimeError('inference failed')
        return [Image.new('RGB', (32, 40), (targets[0], int(self.mode == 'faceid_age'), 0))]

    def close(self):
        self.closed = True
        self.clock.events.append('close')


class CompareLoraTests(unittest.TestCase):
    def test_selection_rejection_preserves_known_detection_counts(self):
        for reason, count in [('ambiguous_primary_face', 2), ('no_usable_face', 0)]:
            error = FaceNotFoundError(reason)
            error.selection_info = {'status': reason, 'detected_count': count}
            faces = SimpleNamespace(_select=Mock(side_effect=error), embedding=Mock(side_effect=error))
            data = _face_data(Image.new('RGB', (64, 64)), faces, True)
            self.assertEqual(data['count'], count)
            self.assertIsNone(data['embedding'])
            self.assertIsNone(data['points'])
            self.assertIsNone(data['age'])
            self.assertIn(reason, data['errors'][0])

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='compare-lora-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest_path, self.config_path = self.root / 'manifest.json', self.root / 'config.json'
        originals, copies = self.root / 'data/lora/val/images', self.root / 'preflight/inputs'
        originals.mkdir(parents=True)
        copies.mkdir(parents=True)
        samples = []
        for index, gender in enumerate(('male', 'female')):
            original, copied = originals / f'{index}.png', copies / f'{index}.png'
            Image.new('RGB', (40, 48), (200 + index, 0, 0)).save(original)
            copied.write_bytes(original.read_bytes())
            samples.append(dict(sample_id=f'portrait-{index}', path=str(original), input_path=str(copied),
                                age_group='20~29', gender=gender, sha256=sha256(original)))
        self.manifest = dict(source_split='val', targets=TARGETS.copy(), seed=42, samples=samples)
        save_json(self.manifest_path, self.manifest)
        paths = {}
        for name in ('base_model', 'image_encoder', 'model_root'):
            directory = self.root / name
            directory.mkdir()
            (directory / 'weights.bin').write_bytes(b'TEST DOUBLE, NOT A MODEL')
            paths[name] = str(directory)
        for name in ('ip_adapter', 'faceid_lora'):
            path = self.root / f'{name}.bin'
            path.write_bytes(b'TEST DOUBLE, NOT A MODEL')
            paths[name] = str(path)
        self.checkpoint = self.root / 'pytorch_lora_weights.safetensors'
        self.checkpoint.write_bytes(b'TEST DOUBLE, NOT A CHECKPOINT')
        self.config = PipelineConfig(**paths, age_lora='unused-original-path', device='cpu')
        save_json(self.config_path, asdict(self.config))
        self.clock, self.pipelines = FakeClock(), []

    def run_case(self, name='run', *, baseline=True, labels=(), failures=(), load_error=False,
                 faces=None, reference=True):
        def factory(config, mode):
            if self.pipelines:
                self.assertTrue(self.pipelines[-1].closed)
            pipeline = FakePipeline(config, mode, self.clock, failures=failures, load_error=load_error)
            self.pipelines.append(pipeline)
            return pipeline

        return run_comparison(self.manifest_path, self.config_path, self.root / name, baseline=baseline,
                              checkpoints=[f'{label}={self.checkpoint}' for label in labels],
                              insightface_reference=reference, pipeline_factory=factory,
                              faces=faces or FakeFaces(self.clock), clock=self.clock)

    def rows(self, name='run'):
        return read_csv(self.root / name / 'results.csv')

    def test_fixed_inputs_gender_targets_seed_and_sequential_conditions(self):
        summary = self.run_case(labels=('original12',))
        self.assertEqual(summary['count'], 20)
        self.assertEqual(summary['status'], 'completed')
        self.assertEqual(list(summary['conditions']), ['baseline', 'original12'])
        self.assertEqual(summary['manifest_sha256'], sha256(self.manifest_path))
        self.assertEqual(summary['config_sha256'], sha256(self.config_path))
        self.assertEqual(summary['conditions']['original12']['checkpoint']['sha256'], sha256(self.checkpoint))
        for pipeline in self.pipelines:
            self.assertTrue(pipeline.closed)
            self.assertEqual(len(pipeline.calls), 10)
            for sample_index, sample in enumerate(self.manifest['samples']):
                image = Image.open(sample['input_path']).convert('RGB')
                for index, target in enumerate(TARGETS):
                    self.assertEqual(pipeline.calls[sample_index * 5 + index],
                                     (image.tobytes(), [target], sample['gender'], 42))
        self.assertEqual(self.pipelines[0].mode, 'faceid')
        self.assertEqual(self.pipelines[0].config.age_lora, '')
        self.assertEqual(self.pipelines[1].config.age_lora, str(self.checkpoint))
        for row in self.rows():
            self.assertEqual(row['prompt'], age_prompt(int(row['target_age']), row['gender']))
            self.assertEqual(row['input_sha256'], self.manifest['samples'][int(row['sample_id'][-1])]['sha256'])
            self.assertEqual(row['output_sha256'], sha256(self.root / 'run' / row['output_path']))
        for source in self.manifest['samples']:
            self.assertEqual(sha256(source['path']), source['sha256'])
        self.assertTrue(all(key in summary['comparison_key'] for key in ('prompts', 'inference_code_sha256', 'asset_hashes')))

    def test_timer_synchronizes_and_excludes_scoring_and_png_saving(self):
        save = Image.Image.save

        def slow_save(image, path, *args, **kwargs):
            self.clock.value += 100
            self.clock.events.append('save')
            return save(image, path, *args, **kwargs)

        with patch.object(Image.Image, 'save', slow_save):
            summary = self.run_case()
        baseline = summary['conditions']['baseline']
        self.assertEqual(baseline['load_seconds'], 5)
        self.assertEqual(baseline['generation_seconds_total'], 20)
        self.assertTrue(all(float(row['generation_seconds']) == 2 for row in self.rows()))
        for index, event in enumerate(self.clock.events):
            if event in ('load', 'generate'):
                self.assertEqual(self.clock.events[index-2:index], ['sync', 'clock'])
                self.assertEqual(self.clock.events[index+1:index+3], ['sync', 'clock'])
        self.assertIsNone(baseline['max_allocated_bytes'])
        self.assertIsNone(baseline['max_reserved_bytes'])

    def test_mae_always_null_and_reference_age_is_opt_in(self):
        faces = FakeFaces(self.clock)
        summary = self.run_case(faces=faces, reference=False)
        self.assertEqual(faces.reference_calls, [])
        self.assertIsNone(summary['age_mae'])
        stats = summary['conditions']['baseline']
        self.assertIsNone(stats['age_mae'])
        self.assertEqual(stats['insightface_age_reference_count'], 0)
        self.assertAlmostEqual(stats['identity_cosine_mean'], .6)
        self.assertEqual(stats['identity_cosine_count'], 10)
        for row in self.rows():
            for field in ('estimated_age', 'absolute_error', 'insightface_age_reference'):
                self.assertEqual(row[field], '')
        self.assertIn('not human evaluation', summary['notes']['geometry'])
        self.assertIn('not been performed', summary['notes']['visual_review'])
        summary = self.run_case('reference', faces=faces)
        self.assertTrue(faces.reference_calls)
        self.assertTrue(all(value is False for value in faces.reference_calls))
        self.assertEqual(summary['conditions']['baseline']['insightface_age_reference_count'], 10)
        self.assertIsNone(summary['conditions']['baseline']['age_mae'])

    def test_failures_keep_all_denominators_and_partial_metrics(self):
        faces = FakeFaces(self.clock, no_faces=(25,), multiple=(45,), embedding_errors=(65,))
        summary = self.run_case(failures=(8,), faces=faces)
        stats = summary['conditions']['baseline']
        self.assertEqual(stats['count'], 10)
        self.assertEqual(stats['generated_count'], 8)
        self.assertEqual(stats['error_count'], 6)
        self.assertEqual(stats['output_detection_denominator'], 10)
        self.assertEqual(stats['output_detection_evaluated_count'], 8)
        self.assertEqual(stats['output_detection_unknown_count'], 2)
        self.assertEqual(stats['output_no_face_count'], 2)
        self.assertEqual(stats['output_face_detected_count'], 6)
        self.assertEqual(stats['output_face_detection_rate'], .6)
        self.assertEqual(stats['output_face_detection_rate_evaluated'], .75)
        self.assertEqual(stats['output_multiple_face_count'], 2)
        self.assertEqual(stats['output_multiple_face_denominator'], 10)
        self.assertEqual(stats['identity_cosine_count'], 4)
        self.assertEqual(stats['landmark_procrustes_reference_count'], 6)
        for row in self.rows():
            if row['target_age'] == '65':
                self.assertEqual(row['identity_cosine'], '')
                self.assertLess(float(row['landmark_procrustes_reference']), 1e-6)
                self.assertIn('identity:', row['error'])
            if row['target_age'] == '8':
                self.assertEqual(row['status'], 'generation_error')
                self.assertEqual(row['output_face_count'], '')
                self.assertEqual(row['output_path'], '')

    def test_unknown_detection_and_invalid_reference_are_not_zero(self):
        faces = FakeFaces(self.clock, detection_errors=(45,), age=float('nan'))
        summary = self.run_case(faces=faces)
        stats = summary['conditions']['baseline']
        self.assertEqual(stats['output_detection_unknown_count'], 2)
        self.assertEqual(stats['output_no_face_count'], 0)
        self.assertIsNone(stats['insightface_age_reference_mean'])
        self.assertEqual(stats['identity_cosine_count'], 10)
        self.assertTrue(all('Invalid InsightFace' in row['error'] for row in self.rows() if row['target_age'] != '45'))
        self.assertTrue(all('detector failure' in row['error'] for row in self.rows() if row['target_age'] == '45'))

    def test_model_load_failures_preserve_every_row_and_close(self):
        summary = self.run_case(load_error=True)
        stats = summary['conditions']['baseline']
        self.assertEqual(stats['count'], 10)
        self.assertEqual(stats['status_counts'], {'load_error': 10})
        self.assertEqual(stats['output_detection_denominator'], 10)
        self.assertEqual(stats['output_detection_unknown_count'], 10)
        self.assertEqual(stats['generated_count'], 0)
        self.assertEqual(stats['generation_timed_count'], 0)
        self.assertIsNone(stats['identity_cosine_mean'])
        self.assertIsNone(stats['generation_seconds_total'])
        self.assertIsNone(stats['age_mae'])
        self.assertTrue(self.pipelines[0].closed)

    def test_save_failures_keep_generation_and_metrics(self):
        save = Image.Image.save

        def fail_one_age(image, path, *args, **kwargs):
            if str(path).endswith('_age_45.png'):
                raise OSError('disk failure')
            return save(image, path, *args, **kwargs)

        with patch.object(Image.Image, 'save', fail_one_age):
            summary = self.run_case()
        stats = summary['conditions']['baseline']
        self.assertEqual(stats['generated_count'], 10)
        self.assertEqual(stats['saved_count'], 8)
        self.assertEqual(stats['status_counts']['save_error'], 2)
        self.assertEqual(stats['identity_cosine_count'], 10)
        self.assertTrue(all(row['output_path'] == '' for row in self.rows() if row['target_age'] == '45'))

    def test_manifest_fixed_protocol_and_metadata_validation(self):
        cases = [dict(source_split='test'), dict(source_split='train'), dict(seed=43), dict(seed=True),
                 dict(targets=[8, 25, 45, 65]), dict(targets=[8, 25, 45, 65, 80.0]), dict(samples=[])]
        for field, value in [('sample_id', ''), ('gender', 'unknown'), ('age_group', ''),
                             ('path', 'relative.png'), ('input_path', 'relative.png'),
                             ('sha256', 'bad'), ('split', 'test')]:
            samples = deepcopy(self.manifest['samples'])
            samples[0][field] = value
            cases.append(dict(samples=samples))
        for index, changed in enumerate(cases):
            with self.subTest(changed=changed):
                path = self.root / f'invalid-{index}.json'
                save_json(path, self.manifest | changed)
                with self.assertRaises(ValueError):
                    load_manifest(path)
        self.assertEqual(load_manifest(self.manifest_path), self.manifest)
        save_json(self.manifest_path, [])
        with self.assertRaisesRegex(ValueError, 'JSON object'):
            load_manifest(self.manifest_path)
        changed = deepcopy(self.manifest)
        changed['samples'][0]['input_path'] = changed['samples'][0]['path']
        save_json(self.manifest_path, changed)
        with self.assertRaisesRegex(ValueError, 'saved copy'):
            load_manifest(self.manifest_path)

    def test_refuses_original_test_paths_and_val_symlinks_to_test(self):
        sample = self.manifest['samples'][0]
        test_path = self.root / 'data/lora/test/images/test.png'
        test_path.parent.mkdir(parents=True)
        test_path.write_bytes(Path(sample['path']).read_bytes())
        link = self.root / 'data/lora/val/images/symlink.png'
        link.symlink_to(test_path)
        for original in (test_path, link):
            changed = deepcopy(self.manifest)
            changed['samples'][0]['path'] = str(original)
            save_json(self.manifest_path, changed)
            with self.assertRaisesRegex(ValueError, 'train/test'):
                load_manifest(self.manifest_path)

    def test_hash_tampering_and_duplicate_inputs_are_refused(self):
        sample = self.manifest['samples'][0]
        for field in ('path', 'input_path'):
            path = Path(sample[field])
            original = path.read_bytes()
            path.write_bytes(original + b'changed')
            with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                load_manifest(self.manifest_path)
            path.write_bytes(original)
        duplicate = deepcopy(self.manifest)
        duplicate['samples'][1] = deepcopy(duplicate['samples'][0])
        save_json(self.manifest_path, duplicate)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            load_manifest(self.manifest_path)

    def test_local_assets_seed_and_output_policy(self):
        config = asdict(self.config)
        save_json(self.config_path, config | {'seed': 43})
        with self.assertRaisesRegex(ValueError, 'seed'):
            self.run_case()
        save_json(self.config_path, config | {'base_model': 'remote/model-not-local'})
        with self.assertRaisesRegex(ValueError, 'local model'):
            self.run_case()
        save_json(self.config_path, config)
        self.run_case()
        with self.assertRaisesRegex(FileExistsError, 'new directory'):
            self.run_case()
        self.assertEqual(len(self.pipelines), 1)

    def test_production_scoring_reuses_loaded_pipeline_face_tools(self):
        faces = FakeFaces(self.clock)

        def factory(config, mode):
            pipeline = FakePipeline(config, mode, self.clock)
            pipeline._faces = faces
            return pipeline

        summary = run_comparison(self.manifest_path, self.config_path, self.root / 'run', baseline=True,
                                 pipeline_factory=factory, clock=self.clock)
        self.assertEqual(summary['conditions']['baseline']['identity_cosine_count'], 10)
        self.assertEqual(faces.reference_calls, [])

    def test_checkpoint_directory_and_duplicate_label_contract(self):
        def factory(config, mode):
            self.assertEqual(config.age_lora, str(self.checkpoint))
            return FakePipeline(config, mode, self.clock)

        summary = run_comparison(self.manifest_path, self.config_path, self.root / 'run',
                                 checkpoints=[f'step100={self.root}'], pipeline_factory=factory,
                                 faces=FakeFaces(self.clock), clock=self.clock)
        self.assertEqual(summary['conditions']['step100']['checkpoint']['sha256'], sha256(self.checkpoint))
        for index, labels in enumerate((['baseline'], ['bad/path'], ['same', 'same'])):
            with self.subTest(labels=labels), self.assertRaisesRegex(ValueError, 'Checkpoint|Duplicate'):
                self.run_case(f'invalid-label-{index}', baseline=False, labels=labels)

    def test_mock_and_wrong_output_count_never_create_quality_scores(self):
        for name, attribute, value in [('mock', 'is_mock', True), ('count', 'age_shift', lambda *a, **k: [])]:
            with patch.object(FakePipeline, attribute, value):
                summary = self.run_case(name)
            stats = summary['conditions']['baseline']
            self.assertEqual(stats['count'], 10)
            self.assertEqual(stats['error_count'], 10)
            self.assertEqual(stats['generated_count'], 0)
            self.assertIsNone(stats['identity_cosine_mean'])
            self.assertIsNone(stats['age_mae'])

    def test_landmark_geometry_invariance_and_degeneracy(self):
        points = [(0, 0), (4, 0), (2, 2), (1, 4), (3, 4)]
        transformed = [(10 - 3*y, 20 + 3*x) for x, y in points]
        self.assertEqual(landmark_geometry(points), landmark_geometry(transformed))
        self.assertLess(procrustes_error(points, transformed), 1e-6)
        changed = points[:2] + [(2, 3)] + points[3:]
        self.assertGreater(procrustes_error(points, changed), .01)
        self.assertGreater(procrustes_error(points, [(-x, y) for x, y in points]), .1)
        for invalid in (points[:4], [(0, 0)] * 5, [(float('nan'), 0)] + points[1:]):
            with self.assertRaises(ValueError):
                landmark_geometry(invalid)

    def test_combine_pairs_separate_runs_and_wide_rows(self):
        baseline = self.run_case('baseline')
        candidate = self.run_case('candidate', baseline=False, labels=('step100',))
        self.assertEqual(baseline['comparison_key'], candidate['comparison_key'])
        summary = combine_results([self.root / 'baseline', self.root / 'candidate'], self.root / 'combined')
        self.assertEqual(summary['count'], 20)
        self.assertEqual(list(summary['conditions']), ['baseline', 'step100'])
        self.assertIsNone(summary['age_mae'])
        paired = read_csv(self.root / 'combined/comparison.csv')
        self.assertEqual(len(paired), 6)
        self.assertEqual([row['target_age'] for row in paired], ['8', '25', '45', '65', '80', 'all'])
        for row in paired:
            self.assertAlmostEqual(float(row['identity_cosine_difference_mean']), .2)
            self.assertEqual(float(row['insightface_age_reference_difference_mean']), 1)
            self.assertEqual(row['age_mae'], '')
            self.assertEqual(int(row['identity_cosine_unpaired_count']), 0)
        wide = read_csv(self.root / 'combined/per_input.csv')
        self.assertEqual(len(wide), 4)
        self.assertIn('age_80_identity_cosine', wide[0])
        self.assertTrue(all(Path(row['age_25_output_path']).is_absolute() for row in wide))
        comparison = json.loads((self.root / 'combined/comparison.json').read_text())
        self.assertIn('condition minus baseline', comparison['difference_definition'])
        for sheet in summary['sheets']:
            with Image.open(self.root / 'combined' / sheet) as image:
                self.assertEqual(image.size, (1536, 576))

    def test_combined_failures_are_retained_in_expected_pairs(self):
        self.run_case('baseline')
        self.run_case('candidate', baseline=False, labels=('step100',), failures=(8,),
                      faces=FakeFaces(self.clock, no_faces=(25,)))
        combine_results([self.root / 'baseline', self.root / 'candidate'], self.root / 'combined')
        paired = read_csv(self.root / 'combined/comparison.csv')
        for row in paired[:2]:
            self.assertEqual(row['expected_pairs'], '2')
            self.assertEqual(row['pairs_with_row_errors'], '2')
            self.assertEqual(row['identity_cosine_paired_count'], '0')
            self.assertEqual(row['identity_cosine_unpaired_count'], '2')
            self.assertEqual(row['identity_cosine_difference_mean'], '')
        self.assertEqual(paired[-1]['expected_pairs'], '10')
        self.assertEqual(paired[-1]['identity_cosine_paired_count'], '6')
        self.assertEqual(paired[-1]['pairs_with_row_errors'], '4')
        errors = [row for row in self.rows('combined') if row['error']]
        self.assertEqual(len(errors), 4)

    def test_combine_rejects_different_settings_hashes_seed_targets_and_prompts(self):
        self.run_case('baseline')
        original = self.run_case('candidate', baseline=False, labels=('step100',))
        changes = [('generation_settings', {'strength': .7}), ('seed', 43), ('targets', [8]),
                   ('input_samples', []), ('prompts', {}), ('asset_hashes', {}), ('inference_code_sha256', {})]
        for index, (field, value) in enumerate(changes):
            summary = deepcopy(original)
            summary['comparison_key'][field] = value
            save_json(self.root / 'candidate/summary.json', summary)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'differing'):
                combine_results([self.root / 'baseline', self.root / 'candidate'], self.root / f'combined-{index}')
            self.assertFalse((self.root / f'combined-{index}').exists())

    def test_combine_rejects_duplicates_partial_runs_rows_and_corrupt_images(self):
        original = self.run_case()
        with self.assertRaisesRegex(ValueError, 'Duplicate conditions'):
            combine_results([self.root / 'run'] * 2, self.root / 'duplicate')
        save_json(self.root / 'run/summary.json', original | {'status': 'running'})
        with self.assertRaisesRegex(ValueError, 'completed'):
            combine_results([self.root / 'run'], self.root / 'partial')
        save_json(self.root / 'run/summary.json', original)
        rows = self.rows()
        write_csv(self.root / 'run/results.csv', rows[:-1])
        with self.assertRaisesRegex(ValueError, 'checksum'):
            combine_results([self.root / 'run'], self.root / 'missing-checksum')
        changed = original | {'results_sha256': sha256(self.root / 'run/results.csv')}
        save_json(self.root / 'run/summary.json', changed)
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            combine_results([self.root / 'run'], self.root / 'missing-row')
        write_csv(self.root / 'run/results.csv', rows)
        save_json(self.root / 'run/summary.json', original)
        image = self.root / 'run' / rows[0]['output_path']
        image.write_bytes(image.read_bytes() + b'tampered')
        with self.assertRaisesRegex(ValueError, 'checksum/path'):
            combine_results([self.root / 'run'], self.root / 'corrupt')

    def test_combine_checks_saved_config_manifest_and_single_run_seed(self):
        summary = self.run_case()
        for name in ('manifest', 'config'):
            path = self.root / 'run' / f'{name}.json'
            original = path.read_bytes()
            self.assertEqual(sha256(path), summary[f'{name}_sha256'])
            path.write_bytes(original + b' ')
            with self.assertRaisesRegex(ValueError, f'Saved {name} checksum'):
                combine_results([self.root / 'run'], self.root / f'bad-{name}')
            path.write_bytes(original)
        summary['comparison_key']['seed'] = 43
        save_json(self.root / 'run/summary.json', summary)
        with self.assertRaisesRegex(ValueError, 'fixed experiment'):
            combine_results([self.root / 'run'], self.root / 'bad-seed')

    def test_sheets_paginate_at_five_conditions_and_no_baseline_is_allowed(self):
        summary = self.run_case(baseline=False, labels=tuple(f'step{i}' for i in range(6)))
        self.assertEqual(len(summary['sheets']), 4)
        for index, relative in enumerate(summary['sheets']):
            with Image.open(self.root / 'run' / relative) as image:
                self.assertEqual(image.size, (1536, 1344 if index % 2 == 0 else 320))
        combined = combine_results([self.root / 'run'], self.root / 'combined')
        self.assertEqual(combined['count'], 60)
        self.assertEqual(read_csv(self.root / 'combined/comparison.csv'), [])

    def test_cuda_meter_uses_injected_runtime_only(self):
        cuda = Mock()
        cuda.is_available.return_value = True
        cuda.max_memory_allocated.return_value = 123
        cuda.max_memory_reserved.return_value = 456
        with patch.dict(sys.modules, {'torch': SimpleNamespace(cuda=cuda)}):
            clock = RunClock('cuda')
            clock.synchronize()
            clock.reset_peak()
            self.assertEqual(clock.memory(), {'max_allocated_bytes': 123, 'max_reserved_bytes': 456})
            cuda.synchronize.assert_called_once_with()
            cuda.reset_peak_memory_stats.assert_called_once_with()
        self.assertEqual(RunClock('cpu').memory(), {'max_allocated_bytes': None, 'max_reserved_bytes': None})

    def test_help_and_combine_do_not_import_torch_or_model_runtimes(self):
        self.run_case()
        script = '''
import importlib.abc, runpy, sys
class NoML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'onnxruntime', 'diffusers', 'transformers', 'cv2'}:
            raise AssertionError('Unexpected model import: ' + fullname)
sys.meta_path.insert(0, NoML())
sys.argv = ['gen.compare_lora', *sys.argv[1:]]
runpy.run_module('gen.compare_lora', run_name='__main__')
'''
        for args in (['--help'], ['--combine', str(self.root / 'run'), '--output', str(self.root / 'cli-combined')]):
            result = subprocess.run([sys.executable, '-B', '-c', script, *args], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / 'cli-combined/summary.json').is_file())

    def test_cli_rejects_mixed_mode_and_missing_required_arguments(self):
        for args in (['--output', str(self.root / 'x')],
                     ['--combine', 'x', '--baseline', '--output', str(self.root / 'x')]):
            with patch('sys.stderr'), self.assertRaises(SystemExit) as error:
                main(args)
            self.assertEqual(error.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
