"""Pillow-only contract tests. Numeric scores below are test doubles, not model evidence."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image

from gen import FaceNotFoundError, GenerationError, ModelUnavailableError
from gen.booth import MockAgeEstimator, compose_four_cut, run_booth, save_booth
from gen.eval import cosine_similarity, evaluate
from gen.pipeline import MODES, MockPipeline
from gen.smoke import run_smoke


class TestPipeline:
    """A deterministic nonmock *test double*, never used outside these tests."""

    is_mock = False

    def __init__(self, mode='faceid_age', events=None, failure=None):
        self.mode, self.failure = mode, failure
        self.events = events if events is not None else []
        self.calls, self.loads, self.closed = [], 0, False

    def load(self):
        self.loads += 1
        self.events.append((self.mode, 'load'))

    def close(self):
        self.closed = True
        self.events.append((self.mode, 'close'))

    def age_shift(self, image, targets, gender='unknown', seed=None):
        self.calls.append((list(targets), gender, seed, image.tobytes()))
        self.events.append((self.mode, 'infer'))
        if self.failure is not None:
            raise self.failure
        return [Image.new('RGB', image.size, (target, 0, 0)) for target in targets]


class GenIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='gen-integration-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.image = Image.new('RGB', (48, 32), (200, 100, 50))

    def evaluate(self, factory=TestPipeline, targets=(20, 40), age_estimator=None,
                 identity_encoder=None, name='evaluation'):
        return evaluate([('portrait', self.image)], targets, factory,
                        age_estimator if age_estimator is not None else
                        lambda image: {'age': image.getpixel((0, 0))[0] + 2},
                        identity_encoder if identity_encoder is not None else lambda image: [3, 4],
                        self.root / name, seed=7)

    def rows(self, name='evaluation'):
        with (self.root / name / 'results.csv').open(newline='', encoding='utf-8') as stream:
            return list(csv.DictReader(stream))

    def test_booth_generates_three_and_preserves_current(self):
        pipeline = TestPipeline()
        before = self.image.tobytes()
        result = run_booth(self.image, lambda image: {'age': 25, 'gender': 'female'}, pipeline, seed=9)
        self.assertEqual(result['metadata']['targets'], [15, 25, 35, 45])
        self.assertEqual(result['metadata']['generated_slots'], [0, 2, 3])
        self.assertEqual(pipeline.calls[0][:3], ([15, 35, 45], 'female', 9))
        self.assertEqual(pipeline.loads, 1)
        self.assertFalse(pipeline.closed)  # The injected pipeline is owned by the caller.
        self.assertEqual(len(result['frames']), 4)
        self.assertEqual([result['frames'][i].getpixel((0, 0))[0] for i in (0, 2, 3)], [15, 35, 45])
        self.assertEqual(result['frames'][1].tobytes(), before)
        self.assertIsNot(result['frames'][1], self.image)
        result['frames'][1].putpixel((0, 0), (0, 0, 0))
        self.assertEqual(self.image.tobytes(), before)
        self.assertFalse(result['metadata']['is_mock'])

    def test_booth_clamped_duplicates_are_not_deduplicated(self):
        for age, targets, generated in [(100, [90, 90, 90, 90], [90, 90, 90]),
                                         (0, [3, 3, 10, 20], [3, 10, 20])]:
            with self.subTest(age=age):
                pipeline = TestPipeline()
                result = run_booth(self.image, MockAgeEstimator(age), pipeline)
                self.assertEqual(result['metadata']['targets'], targets)
                self.assertEqual(pipeline.calls[0][0], generated)
                self.assertEqual(len(result['frames']), 4)
                self.assertTrue(result['metadata']['is_mock'])

    def test_invalid_age_fails_before_generation(self):
        for age in [float('nan'), float('inf'), -1, 101, None, '25', True]:
            with self.subTest(age=age):
                pipeline = TestPipeline()
                with self.assertRaises(ValueError):
                    run_booth(self.image, lambda image: {'age': age}, pipeline)
                self.assertEqual(pipeline.loads, 0)
                self.assertEqual(pipeline.calls, [])
                with self.assertRaises(ValueError):
                    MockAgeEstimator(age)

    def test_booth_errors_propagate_and_output_count_is_checked(self):
        for failure in [GenerationError('generation failed'), FaceNotFoundError('no face')]:
            with self.subTest(failure=failure):
                with self.assertRaises(type(failure)):
                    run_booth(self.image, MockAgeEstimator(), TestPipeline(failure=failure))
        pipeline = TestPipeline()
        pipeline.age_shift = lambda *args, **kwargs: []
        with self.assertRaisesRegex(GenerationError, 'three'):
            run_booth(self.image, MockAgeEstimator(), pipeline)

    def test_compose_preserves_order_and_aspect_ratio(self):
        colors = ['red', 'green', 'blue', 'black']
        frames = [Image.new('RGB', (20, 10), color) for color in colors]
        sheet = compose_four_cut(frames, cell_size=(20, 20))
        self.assertEqual(sheet.size, (40, 40))
        self.assertEqual(sheet.mode, 'RGB')
        for frame, position in zip(frames, [(10, 10), (30, 10), (10, 30), (30, 30)]):
            self.assertEqual(sheet.getpixel(position), frame.getpixel((0, 0)))
            self.assertEqual(frame.size, (20, 10))
        self.assertEqual(sheet.getpixel((0, 0)), (255, 255, 255))
        with self.assertRaises(ValueError):
            compose_four_cut(frames[:3])
        with self.assertRaises(ValueError):
            compose_four_cut(frames, (0, 10))

    def test_booth_output_is_fresh_and_mock_is_explicit(self):
        result = run_booth(self.image, MockAgeEstimator(), MockPipeline())
        output = self.root / 'booth'
        metadata = save_booth(result, output)
        self.assertTrue(metadata['is_mock'])
        self.assertEqual(metadata['timing_kind'], 'mock')
        self.assertEqual(len(list(output.glob('*.png'))), 5)
        with Image.open(output / '02_current.png') as current:
            self.assertEqual(current.tobytes(), self.image.tobytes())
        with self.assertRaises(FileExistsError):
            save_booth(result, output)

    def test_known_metrics_and_sequential_ablation_lifetime(self):
        pipelines, events = [], []

        def factory(mode):
            if pipelines:
                self.assertTrue(pipelines[-1].closed)
            pipeline = TestPipeline(mode, events)
            pipelines.append(pipeline)
            return pipeline

        def identity(image):
            return [3, 0] if image.getpixel((0, 0))[0] == 200 else [3, 4]

        summary = self.evaluate(factory=factory, identity_encoder=identity)
        self.assertEqual([p.mode for p in pipelines], list(MODES))
        self.assertFalse(summary['is_mock'])
        self.assertEqual(summary['count'], 6)
        for pipeline in pipelines:
            self.assertEqual(pipeline.loads, 1)
            self.assertTrue(pipeline.closed)
            self.assertEqual([call[0] for call in pipeline.calls], [[20], [40]])
            self.assertTrue(all(call[2] == 7 for call in pipeline.calls))
            self.assertTrue(all(call[3] == self.image.tobytes() for call in pipeline.calls))
            mode = summary['modes'][pipeline.mode]
            self.assertEqual(mode['status_counts'], {'ok': 2})
            self.assertEqual(mode['age_mae'], 2)
            self.assertAlmostEqual(mode['identity_cosine_mean'], 0.6)
            self.assertEqual(mode['age_scored_count'], 2)
            self.assertEqual(mode['identity_scored_count'], 2)
            self.assertEqual(mode['face_failure_denominator'], 2)
            self.assertEqual(mode['face_failure_rate'], 0)
            self.assertGreaterEqual(mode['load_seconds'], 0)
            self.assertGreaterEqual(mode['inference_seconds'], 0)
        self.assertEqual(events, [(mode, action) for mode in MODES for action in ('load', 'infer', 'infer', 'close')])
        self.assertIn('identity', summary['metric_evaluators']['identity'])
        self.assertTrue(all((self.root / 'evaluation' / row['output_path']).is_file() for row in self.rows()))

    def test_cosine_normalizes_and_rejects_invalid_embeddings(self):
        self.assertAlmostEqual(cosine_similarity([3, 0], [3, 4]), 0.6)
        self.assertEqual(cosine_similarity([2, 0], [-8, 0]), -1)
        self.assertEqual(cosine_similarity([1e308, 0], [1e308, 0]), 1)
        for left, right in [([], []), ([0, 0], [1, 2]), ([1], [1, 2]),
                            ([float('nan')], [1]), ([1], [float('inf')])]:
            with self.subTest(left=left, right=right), self.assertRaises(ValueError):
                cosine_similarity(left, right)

    def test_face_failures_are_separate_and_keep_partial_scores(self):
        def estimator(image):
            if image.getpixel((0, 0))[0] == 20:
                raise FaceNotFoundError('face_not_detected')
            return {'age': 43}

        summary = self.evaluate(age_estimator=estimator)
        for mode in summary['modes'].values():
            self.assertEqual(mode['age_scored_count'], 1)
            self.assertEqual(mode['identity_scored_count'], 2)
            self.assertEqual(mode['age_mae'], 3)
            self.assertEqual(mode['face_failure_count'], 1)
            self.assertEqual(mode['face_failure_denominator'], 2)
            self.assertEqual(mode['face_failure_rate'], 0.5)
            self.assertEqual(mode['status_counts'], {'face_not_detected': 1, 'ok': 1})
        self.assertEqual(self.rows()[0]['absolute_error'], '')
        self.assertEqual(self.rows()[0]['identity_similarity'], '1.0')

    def test_generation_face_failure_and_other_failure_denominators(self):
        for index, failure in enumerate([FaceNotFoundError('face_not_detected'), GenerationError('bad image')]):
            summary = self.evaluate(factory=lambda mode: TestPipeline(mode, failure=failure), name=f'failure-{index}')
            for mode in summary['modes'].values():
                self.assertIsNone(mode['age_mae'])
                self.assertIsNone(mode['identity_cosine_mean'])
                self.assertEqual(mode['generated_count'], 0)
                self.assertEqual(mode['face_failure_denominator'], 2 if index == 0 else 0)
                self.assertEqual(mode['face_failure_rate'], 1 if index == 0 else None)
                self.assertEqual(mode['status_counts'], {'face_not_detected' if index == 0 else 'inference_error': 2})

    def test_metric_errors_do_not_become_zero_scores(self):
        summary = self.evaluate(age_estimator=lambda image: {'age': float('nan')},
                                identity_encoder=lambda image: [0, 0])
        for mode in summary['modes'].values():
            self.assertEqual(mode['status_counts'], {'metric_error': 2})
            self.assertEqual(mode['age_scored_count'], 0)
            self.assertEqual(mode['identity_scored_count'], 0)
            self.assertIsNone(mode['age_mae'])
            self.assertIsNone(mode['identity_cosine_mean'])
            self.assertIsNone(mode['face_failure_rate'])
        self.assertIn('nonzero', self.rows()[0]['error'])

    def test_fatal_dependency_or_model_errors_propagate_and_close(self):
        for index, error in enumerate([ModuleNotFoundError('missing dependency'), ModelUnavailableError('missing weights')]):
            pipeline = TestPipeline(failure=error)
            with self.assertRaises(type(error)):
                self.evaluate(factory=lambda mode: pipeline, name=f'fatal-{index}')
            self.assertTrue(pipeline.closed)
        pipeline = TestPipeline()

        def broken_load():
            raise GenerationError('cannot load model')

        pipeline.load = broken_load
        with self.assertRaises(GenerationError):
            self.evaluate(factory=lambda mode: pipeline, name='load-failure')
        self.assertTrue(pipeline.closed)

    def test_mock_skips_evaluators_and_has_no_invented_metrics(self):
        def forbidden(image):
            self.fail('Mock evaluation must not call metric evaluators')

        summary = self.evaluate(factory=MockPipeline, age_estimator=forbidden, identity_encoder=forbidden)
        self.assertTrue(summary['is_mock'])
        for mode in summary['modes'].values():
            self.assertEqual(mode['status_counts'], {'mock': 2})
            self.assertEqual(mode['timing_kind'], 'mock')
            for key in ('age_mae', 'identity_cosine_mean', 'face_failure_rate', 'face_failure_count'):
                self.assertIsNone(mode[key])
            self.assertEqual(mode['age_scored_count'], 0)
            self.assertEqual(mode['identity_scored_count'], 0)
            self.assertEqual(mode['face_failure_denominator'], 0)
        for row in self.rows():
            for field in ('estimated_age', 'absolute_error', 'identity_similarity', 'face_failure'):
                self.assertEqual(row[field], '')
        with self.assertRaisesRegex(ValueError, 'Mock evaluators'):
            self.evaluate(age_estimator=MockAgeEstimator(), name='mixed')

    def test_eval_requires_new_output_and_preserves_duplicate_targets(self):
        summary = self.evaluate(targets=(90, 90))
        self.assertEqual(summary['targets'], [90, 90])
        self.assertEqual(summary['count'], 6)
        self.assertEqual(len({row['output_path'] for row in self.rows()}), 6)
        with self.assertRaises(FileExistsError):
            self.evaluate()
        for index, targets in enumerate([[], [True], [1.5], [-1], [101]]):
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                self.evaluate(targets=targets, name=f'invalid-{index}')

    def test_smoke_integration_and_rerun_policy(self):
        output = self.root / 'smoke'
        status = run_smoke(output)
        self.assertEqual(status['status'], 'mock_structure_passed')
        self.assertTrue(status['is_mock'])
        self.assertFalse(status['trained_models'])
        self.assertFalse(status['real_model_validation'])
        self.assertEqual(status['evaluation_rows'], 9)
        self.assertEqual(json.loads((output / 'status.json').read_text()), status)
        summary = json.loads((output / 'evaluation/summary.json').read_text())
        self.assertTrue(all(mode['age_mae'] is None for mode in summary['modes'].values()))
        with self.assertRaises(FileExistsError):
            run_smoke(output)

    def test_cli_help_and_smoke_do_not_import_ml_dependencies(self):
        script = '''
import importlib.abc, runpy, sys
class NoML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'torch', 'onnx', 'onnxruntime', 'diffusers', 'transformers', 'cv2'}:
            raise AssertionError('Unexpected ML import: ' + fullname)
sys.meta_path.insert(0, NoML())
module, *arguments = sys.argv[1:]
sys.argv = [module, *arguments]
runpy.run_module(module, run_name='__main__')
'''
        repo = Path(__file__).resolve().parents[1]
        for module in ('gen.booth', 'gen.eval', 'gen.smoke'):
            result = subprocess.run([sys.executable, '-B', '-c', script, module, '--help'],
                                    cwd=repo, text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('--output', result.stdout)
        result = subprocess.run([sys.executable, '-B', '-c', script, 'gen.smoke',
                                 '--output', str(self.root / 'cli-smoke')],
                                cwd=repo, text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(result.stdout)['is_mock'])
        image_path = self.root / 'portrait.png'
        self.image.save(image_path)
        manifest = self.root / 'manifest.csv'
        with manifest.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['sample_id', 'path'])
            writer.writeheader()
            writer.writerow({'sample_id': 'relative-portrait', 'path': image_path.name})
        for module, arguments in [
            ('gen.booth', ['--image', str(image_path), '--mock-age', '100']),
            ('gen.eval', ['--csv', str(manifest), '--targets', '90', '90']),
        ]:
            output = self.root / module
            command = [sys.executable, '-B', '-c', script, module, '--mock',
                       '--output', str(output), *arguments]
            result = subprocess.run(command, cwd=repo, text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            if module == 'gen.eval':
                summary = json.loads(result.stdout)
                self.assertEqual(summary['count'], 6)
                self.assertTrue(summary['is_mock'])
                with (output / 'results.csv').open(newline='', encoding='utf-8') as stream:
                    self.assertTrue(all(row['sample'] == 'relative-portrait' for row in csv.DictReader(stream)))
            else:
                metadata = json.loads((output / 'metadata.json').read_text())
                self.assertEqual(metadata['targets'], [90, 90, 90, 90])
                self.assertTrue(metadata['is_mock'])
            result = subprocess.run(command, cwd=repo, text=True, capture_output=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('new directory', result.stderr)


if __name__ == '__main__':
    unittest.main()
