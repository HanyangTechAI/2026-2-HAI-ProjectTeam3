"""Offline validation/runner contracts. GPU training is a separate real smoke run."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from gen.train_lora import SOURCE_PATCHES, patched_source, run_training, training_arguments
from gen.validate_lora_data import validate_dataset, validate_provenance


class LoRADataContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for i, split in enumerate(('train', 'val', 'test')):
            directory = self.root / split
            (directory / 'images').mkdir(parents=True)
            Image.new('RGB', (512, 512), (i * 80, 40, 10)).save(directory / f'images/{i}.png')
            (directory / 'metadata.jsonl').write_text(json.dumps({
                'file_name': f'images/{i}.png', 'text': 'a photo of a woman in her 20s'}) + '\n')

    def test_valid_images_counts_captions_and_no_identity_claim(self):
        report = validate_dataset(self.root, dict(train=1, val=1, test=1))
        self.assertTrue(report['valid'])
        self.assertEqual(report['splits']['train']['valid_images'], 1)
        self.assertEqual(report['splits']['val']['valid_references'], 1)
        self.assertFalse(report['identity_disjoint_verified'])
        self.assertEqual(report['corrections'], [])
        self.assertEqual(len(report['groups']), 60)

    def test_missing_caption_image_and_malformed_row_are_reported(self):
        metadata = self.root / 'train/metadata.jsonl'
        metadata.write_text(json.dumps({'file_name': 'images/missing.png', 'text': ' '}) + '\n{\n')
        report = validate_dataset(self.root)
        errors = report['splits']['train']['error_counts']
        for kind in ('empty_caption', 'missing_image', 'invalid_json', 'unreferenced_image', 'count_mismatch'):
            self.assertEqual(errors[kind], 1)
        self.assertFalse(report['valid'])

    def test_paths_cannot_escape_split_even_through_symlink(self):
        (self.root / 'train/images/link.png').symlink_to(self.root / 'val/images/1.png')
        metadata = self.root / 'train/metadata.jsonl'
        with metadata.open('a') as stream:
            for name in ('../val/images/1.png', 'images/link.png', str(self.root / 'test/images/2.png')):
                stream.write(json.dumps({'file_name': name, 'text': 'a photo of a woman in her 20s'}) + '\n')
        errors = validate_dataset(self.root)['splits']['train']['error_counts']
        self.assertEqual(errors['path_outside_split_images'], 2)
        self.assertEqual(errors['invalid_file_name'], 1)

    def test_duplicates_are_errors_not_identity_verification(self):
        (self.root / 'val/images/1.png').write_bytes((self.root / 'train/images/0.png').read_bytes())
        metadata = self.root / 'train/metadata.jsonl'
        metadata.write_text(metadata.read_text() * 2)
        report = validate_dataset(self.root)
        self.assertEqual(report['splits']['val']['error_counts']['duplicate_image_hash'], 1)
        self.assertEqual(report['splits']['train']['error_counts']['duplicate_reference'], 1)

    def test_wrong_size_mode_and_corruption_are_not_auto_repaired(self):
        Image.new('L', (64, 64)).save(self.root / 'train/images/0.png')
        broken = self.root / 'val/images/1.png'
        broken.write_bytes(b'not an image')
        report = validate_dataset(self.root)
        self.assertEqual(report['splits']['train']['error_counts']['wrong_size_or_mode'], 1)
        self.assertEqual(report['splits']['val']['error_counts']['unreadable_image'], 1)
        self.assertEqual(broken.read_bytes(), b'not an image')

    def test_missing_split_and_expected_counts_fail(self):
        (self.root / 'test/metadata.jsonl').unlink()
        report = validate_dataset(self.root, dict(train=2, val=1, test=1))
        self.assertFalse(report['valid'])
        self.assertEqual(report['splits']['train']['error_counts']['unexpected_count'], 1)
        self.assertEqual(report['splits']['test']['error_counts']['missing_metadata'], 1)

    def test_provenance_thresholds_hashes_and_attribution(self):
        report = validate_dataset(self.root)
        rows = []
        for path, record in report['records'].items():
            rows.append(dict(path=path, split=record['split'], caption=record['text'], sha256=record['sha256'],
                             age_group_confidence='0.9', head_yaw='-29.9',
                             attribution=dict(author='Fixture', photo_url='https://example.org/image',
                                              license='Fixture only', license_url='https://example.org/license')))
        manifest = self.root / 'provenance.jsonl'
        manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        self.assertTrue(validate_provenance(report, manifest, self.root)['valid'])
        rows[0]['age_group_confidence'] = '0.8'
        manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        self.assertFalse(validate_provenance(report, manifest, self.root)['valid'])


class LoRARunnerContracts(unittest.TestCase):
    def test_pinned_source_and_two_explicit_patches(self):
        source = '\n'.join(SOURCE_PATCHES).encode()
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            patched_source(source)
        with patch('gen.train_lora.SCRIPT_SHA256', hashlib.sha256(source).hexdigest()):
            result = patched_source(source)
        for before, after in SOURCE_PATCHES.items():
            self.assertNotIn(before, result)
            self.assertIn(after, result)

    def test_official_arguments_train_only_and_resume_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, train, output = root / 'model', root / 'data/train', root / 'output'
            base.mkdir()
            (base / 'model_index.json').write_text('{}')
            (train / 'images').mkdir(parents=True)
            (train / 'metadata.jsonl').touch()
            config = json.loads((Path(__file__).resolve().parents[1] / 'gen/lora.smoke.json').read_text())
            config.update(pretrained_model_name_or_path=str(base), train_data_dir=str(train))
            args = training_arguments(config, output)
            self.assertIn('--gradient_checkpointing', args)
            self.assertNotIn('--dataset_name', args)
            self.assertEqual(args[args.index('--train_data_dir') + 1], str(train))
            with self.assertRaisesRegex(ValueError, 'train split'):
                training_arguments(config | {'train_data_dir': str(train.parent)}, output)
            with self.assertRaisesRegex(ValueError, 'Unsupported'):
                training_arguments(config | {'dataset_name': 'online/dataset'}, output)
            with self.assertRaisesRegex(ValueError, 'positive integer'):
                training_arguments(config | {'max_train_steps': 0}, output)
            with self.assertRaisesRegex(ValueError, 'existing checkpoint'):
                training_arguments(config, output, 'checkpoint-5')
            (output / 'checkpoint-5').mkdir(parents=True)
            self.assertIn('--resume_from_checkpoint', training_arguments(config, output, 'checkpoint-5'))
            with self.assertRaisesRegex(ValueError, 'exceed'):
                training_arguments(config | {'max_train_steps': 5}, output, 'checkpoint-5')

    def test_resume_cannot_delete_or_change_existing_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, script = root / 'output', root / 'script.py'
            output.mkdir()
            script.write_bytes(b'fixture')
            original = {'max_train_steps': 10, 'mixed_precision': 'fp16', 'lr_scheduler': 'cosine'}
            (output / 'config.json').write_text(json.dumps(original))
            configs = [original | {'max_train_steps': 12, 'mixed_precision': 'no'},
                       {'max_train_steps': 12, 'mixed_precision': 'fp16'}]
            for config in configs:
                with self.subTest(config=config), patch('gen.train_lora.training_arguments', return_value=[]), \
                        patch('gen.train_lora.patched_source', return_value=''):
                    with self.assertRaisesRegex(ValueError, 'only change max_train_steps'):
                        run_training(config, output, script, 'checkpoint-10')
            self.assertFalse((output / 'resume-12').exists())


if __name__ == '__main__':
    unittest.main()
