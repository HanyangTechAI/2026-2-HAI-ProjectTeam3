from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from age.common import read_csv, sha256, write_csv
from gen.pipeline import PipelineConfig
from gen.analyze_quality import analyze
from gen.quality_experiment import (can_start, load_mask_inputs, load_rows, parse_deadline,
                                    main, run, source_call_inputs, summarize, DEFAULT_CONDITIONS)


class Faces:
    def _select(self, image):
        return np.asarray(image), SimpleNamespace(detected_count=1, age=40,
               kps=np.array([[8, 8], [24, 8], [16, 16], [10, 24], [22, 24]]))

    def embedding(self, image):
        return [1, 0]


class Pipeline:
    is_mock = False
    calls = 0

    def __init__(self, config, mode):
        self._faces = Faces()

    def load(self):
        return self

    def close(self):
        pass

    def age_shift(self, image, targets, gender, seed, diagnostics):
        type(self).calls += 1
        edit = SimpleNamespace(crop=image.copy(), mask=Image.new('L', image.size, 255),
                               source_mask=Image.new('L', image.size, 255),
                               matrix=np.array([[1, 0, 0], [0, 1, 0]]), selection_info={})
        diagnostics(dict(prompt='test only', edit=edit, generation_mask=edit.mask,
                         latent_mask=Image.new('L', (4, 4), 255), raw=image.copy(), output=image.copy()))
        return [image.copy()]


class QualityExperimentTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='quality-experiment-')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        image = self.root / 'input.png'
        Image.new('RGB', (32, 32), 'blue').save(image)
        self.plan = dict(samples=[dict(sample_id='sample', input_path=str(image), sha256=sha256(image),
                                      gender='male', age_group='30-39')],
                         base_config=asdict(PipelineConfig(device='cpu')),
                         conditions={'baseline': {'mode': 'faceid', 'overrides': {}}},
                         targets=[25, 75], seeds=[42, 31415])
        self.output = self.root / 'run'
        Pipeline.calls = 0

    def test_explicit_timezone_and_deadline_budget(self):
        with self.assertRaises(ValueError):
            parse_deadline('2026-10-30T14:50:06')
        self.assertEqual(parse_deadline('2026-10-30T14:50:06Z'),
                         parse_deadline('2026-10-30T23:50:06+09:00'))
        self.assertFalse(can_start(100, 60, now=40))
        self.assertTrue(can_start(100, 60, now=39))

    def test_no_deadline_completes_and_can_resume_without_duplicate_work(self):
        self.assertIsNone(parse_deadline(None))
        self.assertTrue(can_start(None, now=10**12))
        result = run(self.plan, self.output, pipeline_factory=Pipeline)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(Pipeline.calls, 4)
        events = [json.loads(line) for line in (self.output / 'events.jsonl').read_text().splitlines()]
        self.assertIsNone(events[0]['deadline_utc'])
        run(self.plan, self.output, resume=True, pipeline_factory=Pipeline)
        self.assertEqual(Pipeline.calls, 4)

    def test_cli_omitted_deadline_is_not_replaced_by_a_timer(self):
        with patch('gen.quality_experiment.build_plan', return_value=self.plan), \
                patch('gen.quality_experiment.run', return_value={}) as execute, \
                patch.dict('os.environ'):
            main(['--manifest', 'manifest.json', '--config', 'config.json',
                  '--checkpoint', 'weights.safetensors', '--output', str(self.output)])
        self.assertIsNone(execute.call_args.args[2])

    def test_pause_then_resume_no_duplicate_and_preserve_ratings(self):
        paused = run(self.plan, self.output, time.time() - 1, pipeline_factory=Pipeline)
        self.assertEqual(paused['status'], 'paused_deadline')
        self.assertEqual(Pipeline.calls, 0)
        result = run(self.plan, self.output, time.time() + 1000, resume=True, pipeline_factory=Pipeline)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['completed_attempts'], 4)
        self.assertEqual(Pipeline.calls, 4)
        ratings = read_csv(self.output / 'review_template.csv')
        ratings[0]['comment'] = 'Keep this human-entered response'
        write_csv(self.output / 'review_template.csv', ratings)
        run(self.plan, self.output, time.time() + 1000, resume=True, pipeline_factory=Pipeline)
        self.assertEqual(Pipeline.calls, 4)
        self.assertEqual(read_csv(self.output / 'review_template.csv')[0]['comment'], ratings[0]['comment'])
        self.assertEqual(len(load_rows(self.output)), 4)
        self.assertFalse(result['human_review_complete'])

    def test_resume_rejects_settings_changes_and_changed_artifact(self):
        run(self.plan, self.output, time.time() + 1000, pipeline_factory=Pipeline)
        changed = dict(self.plan, seeds=[999])
        with self.assertRaisesRegex(ValueError, 'Resume refused'):
            run(changed, self.output, time.time() + 1000, resume=True, pipeline_factory=Pipeline)
        row = load_rows(self.output)[0]
        (self.output / row['output_path']).write_bytes(b'damaged')
        with self.assertRaisesRegex(ValueError, 'changed or missing'):
            load_rows(self.output)

    def test_failed_rows_are_journaled_without_zero_metrics(self):
        class Broken(Pipeline):
            def age_shift(self, *args, **kwargs):
                raise RuntimeError('deliberate failure')
        result = run(self.plan, self.output, time.time() + 1000, pipeline_factory=Broken)
        self.assertEqual(result['conditions']['baseline']['errors'], 4)
        self.assertIsNone(result['conditions']['baseline']['identity_cosine_mean'])
        self.assertEqual(len(load_rows(self.output)), 4)
        self.assertEqual(result['expected_attempts'], 4)

    def test_partial_journal_is_not_silently_discarded(self):
        self.output.mkdir()
        (self.output / 'rows.jsonl').write_text('{"condition":', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'preserve and inspect'):
            load_rows(self.output)

    def test_copy_control_is_not_generated_and_never_has_age_mae(self):
        self.plan['conditions'] = {'no_edit': {'mode': 'copy', 'overrides': {}}}
        result = run(self.plan, self.output, time.time() + 1000,
                     pipeline_factory=Pipeline, faces_factory=lambda config: Faces())
        self.assertEqual(Pipeline.calls, 0)
        self.assertIsNone(result['age_mae'])
        self.assertEqual(result['conditions']['no_edit']['identity_cosine_mean'], 1)
        self.assertEqual(result['conditions']['no_edit']['protected_changed_pixels_mean'], 0)

    def test_analysis_matches_endpoints_and_renders_both_seeds(self):
        run(self.plan, self.output, time.time() + 1000, pipeline_factory=Pipeline)
        result = analyze(self.output, self.root / 'analysis')
        baseline = result['conditions']['baseline']
        self.assertEqual(baseline['reference_age_paired_span_count'], 2)
        self.assertEqual(baseline['reference_age_paired_span_mean'], 0)
        self.assertEqual(baseline['identical_endpoint_pairs'], 2)
        self.assertEqual(baseline['endpoint_pixel_pairs'], 2)
        self.assertEqual(baseline['endpoint_changed_pixels_mean'], 0)
        self.assertEqual(set(result['sheets']), {'42', '31415'})
        self.assertFalse(result['human_review_complete'])
        self.assertIsNone(result['age_mae'])

    def test_partial_analysis_does_not_invent_endpoint_pairs(self):
        run(self.plan, self.output, time.time() - 1, pipeline_factory=Pipeline)
        result = analyze(self.output, self.root / 'analysis')
        baseline = result['conditions']['baseline']
        self.assertEqual(baseline['reference_age_paired_span_count'], 0)
        self.assertIsNone(baseline['reference_age_paired_span_mean'])
        self.assertEqual(baseline['planned_attempts'], 4)
        self.assertEqual(baseline['endpoint_pixel_pairs'], 0)
        self.assertIsNone(baseline['endpoint_changed_pixels_mean'])

    def mask_fixture(self):
        folder = self.root / 'masks'
        folder.mkdir()
        entry = dict(input_sha256=self.plan['samples'][0]['sha256'], source_size=[32, 32],
                     negative_prompt='test negative, glasses')
        for key, color in (('source_edit_mask', 255), ('source_protect_mask', 0)):
            path = folder / (key + '.png')
            Image.new('L', (32, 32), color).save(path)
            entry[key] = dict(path=str(path), sha256=sha256(path))
        path = folder / 'manifest.json'
        path.write_text(json.dumps(dict(schema=1, coordinate_space='source', human_verified=False,
                                       samples={'sample': entry})), encoding='utf-8')
        return path

    def test_mask_inputs_are_hash_bound_and_only_applied_to_selected_conditions(self):
        manifest = self.mask_fixture()
        masks = load_mask_inputs(manifest, self.plan['samples'])
        self.plan['mask_inputs'] = masks
        sample = self.plan['samples'][0]
        args, fixed = source_call_inputs(self.plan, {}, sample)
        self.assertEqual(args, {})
        self.assertEqual(set(fixed), {'source_edit_mask', 'source_protect_mask'})
        args, _ = source_call_inputs(self.plan, dict(mask_policy='manual', negative_policy='source_annotation'), sample)
        self.assertEqual(args['negative_prompt'], 'test negative, glasses')
        self.assertEqual(args['source_edit_mask'].mode, 'L')
        Image.new('L', (32, 32), 128).save(masks['samples']['sample']['source_edit_mask']['path'])
        with self.assertRaisesRegex(ValueError, 'changed after preflight'):
            source_call_inputs(self.plan, {}, sample)

    def test_mask_manifest_rejects_wrong_source_and_color_mode(self):
        manifest = self.mask_fixture()
        record = json.loads(manifest.read_text())
        record['samples']['sample']['input_sha256'] = '0' * 64
        manifest.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, 'source hash'):
            load_mask_inputs(manifest, self.plan['samples'])
        record['samples']['sample']['input_sha256'] = self.plan['samples'][0]['sha256']
        asset = record['samples']['sample']['source_edit_mask']
        Image.new('RGB', (32, 32), 'white').save(asset['path'])
        asset['sha256'] = sha256(asset['path'])
        manifest.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, 'grayscale L'):
            load_mask_inputs(manifest, self.plan['samples'])

    def test_default_conditions_remain_usable_without_manual_annotations(self):
        self.assertEqual(len(DEFAULT_CONDITIONS), 13)
        self.assertNotIn('manual_mask', DEFAULT_CONDITIONS)
        with self.assertRaisesRegex(ValueError, 'recorded source annotations'):
            source_call_inputs(self.plan, {'mask_policy': 'manual'}, self.plan['samples'][0])


if __name__ == '__main__':
    unittest.main()
