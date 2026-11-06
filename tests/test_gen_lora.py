import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from gen.train_lora import (CHECKPOINT_FILES, PROVENANCE_SCHEMA, SCRIPT_SHA256, SOURCE_PATCHES,
                            checkpoint_retention_plan, create_phase_directory, epoch_loader, patched_source,
                            publish_checkpoint, run_training, training_arguments, training_data_fingerprint,
                            validate_optimizer_update, verify_checkpoint, verify_imagefolder)
from gen.prepare_lora_stages import prepare
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

    def test_original_70s_caption_remains_valid_with_visible_warning(self):
        metadata = self.root / 'train/metadata.jsonl'
        metadata.write_text(json.dumps({'file_name': 'images/0.png',
                                       'text': 'a photo of a elderly man in his 70s'}) + '\n')
        report = validate_dataset(self.root)
        self.assertTrue(report['valid'])
        self.assertEqual(report['splits']['train']['legacy_caption_rows'], 1)

    def test_preparation_copies_images_and_preserves_original_labels(self):
        source, previous, output = self.root / 'train', self.root / 'previous', self.root / 'stages'
        caption = 'a photo of a elderly man in his 70s'
        metadata = source / 'metadata.jsonl'
        metadata.write_text(json.dumps({'file_name': 'images/0.png', 'text': caption}) + '\n')
        original = metadata.read_bytes()
        provenance = self.root / 'provenance.jsonl'
        provenance.write_text(json.dumps(dict(path='train/images/0.png', split='train', caption=caption,
                              sha256=hashlib.sha256((source / 'images/0.png').read_bytes()).hexdigest(),
                              age_group='70-120', gender='male')) + '\n')
        for name, steps in [('initial', 10), ('resume-12', 2)]:
            (previous / name).mkdir(parents=True)
            (previous / name / 'summary.json').write_text(json.dumps({'applied_steps': steps, 'elapsed_seconds': 20}))
        (previous / 'initial/train.log').write_text('10/10 [00:17<00:00]')
        config = Path(__file__).resolve().parents[1] / 'gen/lora.smoke.json'
        result = prepare(source, provenance, previous, output, config)
        self.assertEqual(result['changed_captions'], 1)
        self.assertEqual(metadata.read_bytes(), original)
        self.assertIn('70 or older', (output / 'data/train/metadata.jsonl').read_text())
        self.assertNotEqual((source / 'images/0.png').stat().st_ino,
                            (output / 'data/train/images/0.png').stat().st_ino)
        self.assertFalse((output / 'data/val').exists())
        with self.assertRaises(FileExistsError):
            prepare(source, provenance, previous, output, config)

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
    def test_pinned_source_and_explicit_patches(self):
        source = '\n'.join(SOURCE_PATCHES).encode()
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            patched_source(source)
        with patch('gen.train_lora.SCRIPT_SHA256', hashlib.sha256(source).hexdigest()):
            result = patched_source(source)
        for before, after in SOURCE_PATCHES.items():
            if before not in after:
                self.assertNotIn(before, result)
            self.assertIn(after, result)

    def test_cached_official_source_compiles_without_modifying_it(self):
        script = Path(__file__).resolve().parents[1] / 'outputs/age_lora_tools/train_text_to_image_lora.py'
        if not script.is_file():
            self.skipTest('Pinned official source is not cached; this test never downloads it')
        original = script.read_bytes()
        result = patched_source(original)
        compile(result, str(script), 'exec')
        self.assertEqual(script.read_bytes(), original)
        self.assertIn('accelerator.scheduler_total_steps * accelerator.num_processes', result)

    def test_resume_skips_consumed_batches_with_reproducible_order(self):
        import torch
        from accelerate import Accelerator, DataLoaderConfiguration
        accelerator = Accelerator(cpu=True, gradient_accumulation_steps=4,
                                  dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True, data_seed=42))
        for count in (84, 83):
            with self.subTest(count=count):
                loader = accelerator.prepare(torch.utils.data.DataLoader(list(range(count)), batch_size=1, shuffle=True))
                uninterrupted, _ = epoch_loader(accelerator, loader, 0, 0, 0, 21)
                expected = [int(batch[0]) for batch in uninterrupted]
                for step in (10, 20):
                    resumed, skipped = epoch_loader(accelerator, loader, 0, 0, step, 21)
                    self.assertEqual(skipped, step * 4)
                    self.assertEqual([int(batch[0]) for batch in resumed], expected[step * 4:])
                next_epoch, skipped = epoch_loader(accelerator, loader, 1, 0, 10, 21)
                self.assertEqual(skipped, 0)
                next_order = [int(batch[0]) for batch in next_epoch]
                self.assertNotEqual(next_order, expected)
                for step in (21, 22):
                    resumed, skipped = epoch_loader(accelerator, loader, 1, 1, step, 21)
                    self.assertEqual(skipped, (step % 21) * 4)
                    self.assertEqual([int(batch[0]) for batch in resumed], next_order[skipped:])

    def test_checkpoint_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, 'integrity manifest is required'):
                verify_checkpoint(root)
            self.assertIsNone(verify_checkpoint(root, allow_legacy=True, required_files=()))
            (root / 'weights').write_bytes(b'valid')
            (root / 'integrity.json').write_text(json.dumps({'files': {'weights': hashlib.sha256(b'valid').hexdigest()}}))
            self.assertIsNotNone(verify_checkpoint(root, required_files=()))
            (root / 'weights').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'integrity mismatch'):
                verify_checkpoint(root, required_files=())

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
            fixed = config | {'scheduler_total_steps': 100}
            self.assertNotIn('--scheduler_total_steps', training_arguments(fixed, output))
            self.assertIn('--resume_from_checkpoint', training_arguments(fixed | {'max_train_steps': 30},
                                                                         output, 'checkpoint-5'))
            for value in (None, True, 0, -1, 1.5, '100'):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'positive integer'):
                    training_arguments(config | {'scheduler_total_steps': value}, output)
            with self.assertRaisesRegex(ValueError, 'fixed scheduler_total_steps'):
                training_arguments(fixed | {'max_train_steps': 101}, output)

    def test_resume_cannot_delete_or_change_existing_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, script = root / 'output', root / 'script.py'
            output.mkdir()
            script.write_bytes(b'fixture')
            original = {'max_train_steps': 10, 'mixed_precision': 'fp16', 'lr_scheduler': 'cosine'}
            (output / 'config.json').write_text(json.dumps(original))
            configs = [original | {'max_train_steps': 12, 'mixed_precision': 'no'},
                       {'max_train_steps': 12, 'mixed_precision': 'fp16'},
                       original | {'max_train_steps': 12, 'scheduler_total_steps': 100}]
            for config in configs:
                with self.subTest(config=config), patch('gen.train_lora.training_arguments', return_value=[]), \
                        patch('gen.train_lora.patched_source', return_value=''):
                    with self.assertRaisesRegex(ValueError, 'only change max_train_steps'):
                        run_training(config, output, script, 'checkpoint-10')
            self.assertFalse((output / 'resume-12').exists())

    def test_retention_new_config_can_preserve_full_planned_run(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'new-experiment'
            config = dict(max_train_steps=2100, checkpointing_steps=100, checkpoints_total_limit=24)
            plan = checkpoint_retention_plan(config, output)
            self.assertEqual(plan['required_limit'], 21)
            self.assertEqual(len(plan['planned_new_checkpoints']), 21)
            with self.assertRaisesRegex(ValueError, 'need checkpoints_total_limit >= 21'):
                checkpoint_retention_plan(config | {'checkpoints_total_limit': 12}, output)
            self.assertFalse(output.exists())

    def test_retention_guard_rejects_deletion_before_touching_old_run(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            original = dict(max_train_steps=1200, checkpointing_steps=100, checkpoints_total_limit=12)
            (output / 'config.json').write_text(json.dumps(original))
            script = output / 'fixture.py'
            script.write_bytes(b'fixture')
            for step in range(100, 1201, 100):
                (output / f'checkpoint-{step}').mkdir()
                (output / f'checkpoint-{step}/weights').write_bytes(str(step).encode())
            before = {str(p.relative_to(output)): p.read_bytes() for p in output.rglob('*') if p.is_file()}
            with patch('gen.train_lora.training_arguments', return_value=[]), \
                    patch('gen.train_lora.patched_source', return_value=''):
                with self.assertRaisesRegex(ValueError, 'retention would delete'):
                    run_training(original | {'max_train_steps': 1300}, output, script, 'checkpoint-1200')
            self.assertFalse((output / 'resume-1300').exists())
            self.assertEqual({str(p.relative_to(output)): p.read_bytes() for p in output.rglob('*') if p.is_file()}, before)
            self.assertEqual(len(list(output.glob('checkpoint-*'))), 12)

    def test_retention_preserves_legacy_resume_and_refuses_checkpoint_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for step in (5, 10):
                (output / f'checkpoint-{step}').mkdir()
            config = dict(max_train_steps=12, checkpointing_steps=5, checkpoints_total_limit=2)
            self.assertEqual(checkpoint_retention_plan(config, output, 'checkpoint-10')['planned_new_checkpoints'], [])
            with self.assertRaisesRegex(ValueError, 'overwrite existing checkpoints'):
                checkpoint_retention_plan(config | {'checkpoints_total_limit': 24}, output, 'checkpoint-5')
            (output / 'checkpoint-15').mkdir()
            with self.assertRaisesRegex(ValueError, 'retention would delete'):
                checkpoint_retention_plan(config | {'max_train_steps': 20}, output, 'checkpoint-15')


class LoRARecoveryContracts(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.train, self.output = self.root / 'train', self.root / 'output'
        (self.train / 'images').mkdir(parents=True)
        self.rows = [dict(file_name='images/b.png', text='older person'),
                     dict(file_name='images/a.png', text='younger person')]
        for i, row in enumerate(self.rows):
            Image.new('RGB', (8, 8), (i * 80, 40, 10)).save(self.train / row['file_name'])
        self.metadata = self.train / 'metadata.jsonl'
        self.metadata.write_text(''.join(json.dumps(row) + '\n' for row in self.rows))
        base = self.root / 'model'
        base.mkdir()
        (base / 'model_index.json').write_text('{}')
        self.config = json.loads((Path(__file__).resolve().parents[1] / 'gen/lora.smoke.json').read_text())
        self.config.update(pretrained_model_name_or_path=str(base), train_data_dir=str(self.train),
                           max_train_steps=2100, scheduler_total_steps=2100, checkpointing_steps=300,
                           checkpoints_total_limit=24)
        self.script = self.root / 'fixture.py'
        self.script.write_bytes(b'fixture')
        self.required_files = CHECKPOINT_FILES | {'scaler.pt'}

    def record_phase(self, resume=None):
        # Stop after command provenance is saved, before imports, CUDA checks, or any training.
        with patch('gen.train_lora.patched_source', return_value=''), \
                patch('gen.train_lora.importlib.metadata.distributions', side_effect=RuntimeError('fixture stop')):
            with self.assertRaisesRegex(RuntimeError, 'fixture stop'):
                run_training(self.config, self.output, self.script, resume)

    def make_checkpoint(self, step=300):
        pending = Path(tempfile.mkdtemp(prefix=f'.pending-checkpoint-{step}-', dir=self.output))
        for name in self.required_files:
            (pending / name).write_bytes(name.encode())
        data = training_data_fingerprint(self.train)
        record = dict(step=step, metadata_sha256=data['metadata_sha256'], train_images_sha256=data['images_sha256'],
                      base_files={'model_index.json': hashlib.sha256(b'{}').hexdigest()},
                      sampler_policy='SeedableRandomSampler(seed) + set_epoch + skip_first_batches')
        destination = self.output / f'checkpoint-{step}'
        publish_checkpoint(pending, destination, record, self.required_files)
        return destination

    def test_repeated_resume_horizon_appends_attempts_without_overwriting(self):
        self.record_phase()
        self.make_checkpoint()
        initial = json.loads((self.output / 'initial/command.json').read_text())
        self.assertEqual(initial['provenance_schema'], PROVENANCE_SCHEMA)
        self.assertTrue(initial['checkpoint_integrity_required'])
        self.assertEqual(initial['training_data']['images'][0]['file_name'], 'images/b.png')
        self.record_phase('checkpoint-300')
        first = self.output / 'resume-2100/command.json'
        before = first.read_bytes()
        self.record_phase('checkpoint-300')
        self.record_phase('checkpoint-300')
        self.assertEqual(first.read_bytes(), before)
        for number in (2, 3):
            command = json.loads((self.output / f'resume-2100-attempt-{number}/command.json').read_text())
            self.assertEqual(command['phase_directory'], f'resume-2100-attempt-{number}')
            self.assertEqual(command['resume_from_checkpoint'], 'checkpoint-300')
            self.assertEqual(command['effective_config']['max_train_steps'], 2100)
            self.assertEqual(command['scheduler_total_steps'], 2100)
        with self.assertRaises(FileExistsError):
            create_phase_directory(self.output, 2100, None)

    def test_new_resume_rejects_missing_malformed_or_unfinalized_manifest(self):
        self.record_phase()
        checkpoint = self.make_checkpoint()
        manifest = checkpoint / 'integrity.json'
        original = json.loads(manifest.read_text())
        missing_file = dict(original['files'])
        del missing_file['optimizer.bin']
        cases = [None, '{', '[]', json.dumps(original | {'files': {}}),
                 json.dumps(original | {'files': []}), json.dumps(original | {'files': missing_file}),
                 json.dumps(original | {'step': 600}), json.dumps(original | {'finalized': False}),
                 json.dumps(original | {'provenance_schema': True}),
                 json.dumps({k: v for k, v in original.items() if k != 'provenance_schema'}),
                 json.dumps(original | {'base_files': {}}),
                 json.dumps(original | {'train_images_sha256': ''})]
        for content in cases:
            with self.subTest(content=content):
                if content is None:
                    manifest.unlink()
                else:
                    manifest.write_text(content)
                with patch('gen.train_lora.patched_source', return_value=''):
                    with self.assertRaises(ValueError):
                        run_training(self.config, self.output, self.script, 'checkpoint-300')
                self.assertFalse((self.output / 'resume-2100').exists())
        manifest.write_text(json.dumps(original))
        for name in sorted(self.required_files):
            path = checkpoint / name
            content = path.read_bytes()
            for incomplete in (None, b''):
                with self.subTest(name=name, incomplete=incomplete):
                    if incomplete is None:
                        path.unlink()
                    else:
                        path.write_bytes(incomplete)
                    with patch('gen.train_lora.patched_source', return_value=''):
                        with self.assertRaisesRegex(ValueError, 'Incomplete checkpoint state'):
                            run_training(self.config, self.output, self.script, 'checkpoint-300')
                    path.write_bytes(content)
        self.assertFalse((self.output / 'resume-2100').exists())

    def test_incomplete_initial_provenance_cannot_downgrade_to_legacy(self):
        self.record_phase()
        self.make_checkpoint()
        command_path = self.output / 'initial/command.json'
        original = json.loads(command_path.read_text())
        for key in ('provenance_schema', 'checkpoint_integrity_required', 'training_data'):
            with self.subTest(key=key):
                command_path.write_text(json.dumps({k: v for k, v in original.items() if k != key}))
                with patch('gen.train_lora.patched_source', return_value=''):
                    with self.assertRaises(ValueError):
                        run_training(self.config, self.output, self.script, 'checkpoint-300')
                self.assertFalse((self.output / 'resume-2100').exists())

    def test_resume_rejects_changed_training_bytes_and_metadata_order(self):
        self.record_phase()
        self.make_checkpoint()
        original_data = training_data_fingerprint(self.train)
        image = self.train / self.rows[0]['file_name']
        content = image.read_bytes()
        image.write_bytes(content + b'changed bytes')
        changed = training_data_fingerprint(self.train)
        self.assertEqual(changed['metadata_sha256'], original_data['metadata_sha256'])
        self.assertNotEqual(changed['images_sha256'], original_data['images_sha256'])
        with patch('gen.train_lora.patched_source', return_value=''):
            with self.assertRaisesRegex(ValueError, 'bytes/order differ'):
                run_training(self.config, self.output, self.script, 'checkpoint-300')
        image.write_bytes(content)
        self.metadata.write_text(''.join(json.dumps(row) + '\n' for row in reversed(self.rows)))
        self.assertNotEqual(training_data_fingerprint(self.train)['images_sha256'], original_data['images_sha256'])
        with patch('gen.train_lora.patched_source', return_value=''):
            with self.assertRaisesRegex(ValueError, 'bytes/order differ'):
                run_training(self.config, self.output, self.script, 'checkpoint-300')
        self.assertFalse((self.output / 'resume-2100').exists())

    def test_historical_no_manifest_exception_requires_known_pre_audit_provenance(self):
        self.record_phase()
        checkpoint = self.make_checkpoint()
        manifest = checkpoint / 'integrity.json'
        manifest.unlink()
        command_path = self.output / 'initial/command.json'
        command = json.loads(command_path.read_text())
        historical = {k: v for k, v in command.items()
                      if k not in ('provenance_schema', 'checkpoint_integrity_required', 'training_data')}
        historical['effective_config']['max_train_steps'] = 300
        historical['in_memory_patches'] = {key: SOURCE_PATCHES[key] for key in (
            'lora_layers = filter(lambda p: p.requires_grad, unet.parameters())', 'unet = unet.to(torch.float32)')}
        self.assertEqual(historical['sha256'], SCRIPT_SHA256)
        for change in ({'in_memory_patches': SOURCE_PATCHES}, {'in_memory_patches': {}},
                       {'sha256': 'unknown'}, {'effective_config': {'max_train_steps': 100}}):
            with self.subTest(change=change):
                command_path.write_text(json.dumps(historical | change))
                with patch('gen.train_lora.patched_source', return_value=''):
                    with self.assertRaisesRegex(ValueError, 'integrity manifest is required'):
                        run_training(self.config, self.output, self.script, 'checkpoint-300')
        command_path.write_text(json.dumps(historical))
        before = command_path.read_bytes()
        self.record_phase('checkpoint-300')
        self.assertEqual(command_path.read_bytes(), before)
        self.assertFalse(manifest.exists())

    def test_historical_manifest_without_schema_remains_verifiable(self):
        self.record_phase()
        checkpoint = self.make_checkpoint()
        command_path = self.output / 'initial/command.json'
        command = json.loads(command_path.read_text())
        for key in ('provenance_schema', 'checkpoint_integrity_required', 'training_data'):
            del command[key]
        command_path.write_text(json.dumps(command))
        manifest = checkpoint / 'integrity.json'
        record = json.loads(manifest.read_text())
        for key in ('provenance_schema', 'finalized', 'train_images_sha256'):
            del record[key]
        manifest.write_text(json.dumps(record))
        before = manifest.read_bytes()
        self.record_phase('checkpoint-300')
        self.assertEqual(manifest.read_bytes(), before)

    def test_partial_saves_stay_pending_and_do_not_block_recovery(self):
        self.record_phase()
        pending = self.output / '.pending-checkpoint-300-interrupted'
        pending.mkdir()
        (pending / 'model.safetensors').write_bytes(b'partial state')
        with self.assertRaisesRegex(ValueError, 'Incomplete checkpoint state'):
            publish_checkpoint(pending, self.output / 'checkpoint-300', {'step': 300}, self.required_files)
        self.assertFalse((self.output / 'checkpoint-300').exists())
        self.assertFalse((pending / 'integrity.json').exists())
        self.assertIn('checkpoint-300', checkpoint_retention_plan(self.config, self.output)['planned_new_checkpoints'])
        checkpoint = self.make_checkpoint()
        self.assertIsNotNone(verify_checkpoint(checkpoint, required_files=self.required_files,
                                              expected_step=300, require_provenance=True))
        before = (checkpoint / 'integrity.json').read_bytes()
        with self.assertRaisesRegex(FileExistsError, 'overwrite checkpoint'):
            publish_checkpoint(pending, checkpoint, {'step': 300}, self.required_files)
        self.assertEqual((checkpoint / 'integrity.json').read_bytes(), before)
        self.assertEqual((pending / 'model.safetensors').read_bytes(), b'partial state')

    def test_manifest_sync_failure_does_not_publish_a_checkpoint(self):
        self.record_phase()
        with patch('gen.train_lora.os.fsync', side_effect=OSError('fixture interrupted save')):
            with self.assertRaisesRegex(OSError, 'fixture interrupted save'):
                self.make_checkpoint()
        self.assertFalse((self.output / 'checkpoint-300').exists())
        pending = list(self.output.glob('.pending-checkpoint-300-*'))
        self.assertEqual(len(pending), 1)
        self.assertTrue((pending[0] / 'integrity.json').exists())
        self.make_checkpoint()
        self.assertTrue((self.output / 'checkpoint-300').is_dir())
        self.assertTrue(pending[0].exists())

    def test_training_fingerprint_rejects_external_paths_before_reading_images(self):
        outside = self.root / 'outside.png'
        outside.write_bytes(b'not training data')
        (self.train / 'images/link.png').symlink_to(outside)
        for name in ('../outside.png', str(outside), 'images/link.png', 'metadata.jsonl', 'images/missing.png'):
            with self.subTest(name=name):
                self.metadata.write_text(json.dumps(dict(file_name=name, text='caption')) + '\n')
                with patch('gen.train_lora.sha256') as image_hash:
                    with self.assertRaises(ValueError):
                        training_data_fingerprint(self.train)
                    image_hash.assert_not_called()

    def test_imagefolder_matches_actual_metadata_order_and_captions_offline(self):
        from datasets import Dataset, load_dataset
        data = training_data_fingerprint(self.train)
        dataset = load_dataset('imagefolder', data_files={'train': str(self.train / '**')},
                               cache_dir=str(self.root / 'cache'))['train']
        result = verify_imagefolder(dataset, self.train, data)
        self.assertTrue(result['metadata_order_verified'])
        self.assertTrue(result['caption_order_verified'])
        self.assertEqual(result['ordered_images_sha256'], data['images_sha256'])
        with self.assertRaisesRegex(RuntimeError, 'references/order'):
            verify_imagefolder(dataset.select([1, 0]), self.train, data)
        with self.assertRaisesRegex(RuntimeError, 'references/order'):
            verify_imagefolder(dataset.select([0]), self.train, data)
        wrong_captions = Dataset.from_dict({'image': [str(self.train / row['file_name']) for row in self.rows],
                                            'text': [row['text'] for row in reversed(self.rows)]})
        with self.assertRaisesRegex(RuntimeError, 'captions/order'):
            verify_imagefolder(wrong_captions, self.train, data)


class LoRAOptimizerContracts(unittest.TestCase):
    def test_cosine_warmup_uses_zero_lr_before_scheduler_advance(self):
        import torch
        from diffusers.optimization import get_scheduler
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=1e-4)
        scheduler = get_scheduler('cosine', optimizer, num_warmup_steps=10, num_training_steps=100)
        kinds, used_lrs, next_lrs = [], [], []
        for step in range(1, 4):
            before = parameter.detach().clone()
            loss = parameter.square().sum()
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_([parameter], 1.0))
            used = [float(group['lr']) for group in optimizer.param_groups]
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            delta = float((parameter.detach() - before).abs().max())
            kinds.append(validate_optimizer_update(float(loss.detach()), norm, used, delta, False,
                                                   expected_zero_lr=step == 1))
            used_lrs.append(used[0])
            next_lrs.append(scheduler.get_last_lr()[0])
        self.assertEqual(kinds, ['warmup_zero_lr', 'weight_update', 'weight_update'])
        self.assertEqual(used_lrs[0], 0)
        self.assertEqual(next_lrs[0], 1e-5)
        self.assertEqual(used_lrs[1:], next_lrs[:-1])
        self.assertNotEqual(float(parameter.detach()), 1.0)

    def test_warmup_exception_never_hides_invalid_gradients_skips_or_frozen_updates(self):
        valid = dict(loss=.1, grad_norm=.2, learning_rates=[0.0], step_delta=0.0,
                     skipped=False, expected_zero_lr=True)
        self.assertEqual(validate_optimizer_update(**valid), 'warmup_zero_lr')
        invalid = [dict(skipped=True), dict(grad_norm=0), dict(grad_norm=float('nan')),
                   dict(grad_norm=float('inf')), dict(loss=float('nan')), dict(step_delta=float('inf')),
                   dict(learning_rates=[float('nan')]), dict(learning_rates=[-1e-4]), dict(learning_rates=[]),
                   dict(learning_rates=[1e-4]), dict(step_delta=.1), dict(expected_zero_lr=False)]
        for change in invalid:
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                validate_optimizer_update(**(valid | change))
        self.assertEqual(validate_optimizer_update(.1, .2, [0.0, 1e-4], .001, False), 'weight_update')

    def test_constant_lower_lr_without_warmup_still_requires_each_step_to_move(self):
        import torch
        from diffusers.optimization import get_scheduler
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=3e-5)
        scheduler = get_scheduler('constant', optimizer, num_warmup_steps=0, num_training_steps=10)
        initial = parameter.detach().clone()
        parameter.square().sum().backward()
        used = [optimizer.param_groups[0]['lr']]
        norm = float(torch.nn.utils.clip_grad_norm_([parameter], 1.0))
        optimizer.step()
        scheduler.step()
        delta = float((parameter.detach() - initial).abs().max())
        self.assertEqual(validate_optimizer_update(.1, norm, used, delta, False), 'weight_update')
        self.assertEqual(used, [3e-5])
        self.assertEqual(scheduler.get_last_lr(), used)
        # Cumulative movement from a prior update must not excuse a later frozen step.
        self.assertGreater(delta, 0)
        with self.assertRaisesRegex(RuntimeError, 'positive learning rate'):
            validate_optimizer_update(.1, norm, used, 0, False)

    def test_fixed_horizon_cosine_resume_matches_uninterrupted_cpu_trajectory(self):
        import torch
        from diffusers.optimization import get_scheduler

        def setup(stage_stop):
            parameter = torch.nn.Parameter(torch.tensor([1.0]))
            optimizer = torch.optim.AdamW([parameter], lr=1e-4)
            site = 'num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes'
            namespace = dict(args=SimpleNamespace(max_train_steps=stage_stop),
                             accelerator=SimpleNamespace(scheduler_total_steps=40, num_processes=1))
            exec(SOURCE_PATCHES[site], namespace)
            self.assertEqual(namespace['num_training_steps_for_scheduler'], 40)
            scheduler = get_scheduler('cosine', optimizer, num_warmup_steps=5,
                                      num_training_steps=namespace['num_training_steps_for_scheduler'])
            return parameter, optimizer, scheduler

        def advance(state, count):
            parameter, optimizer, scheduler = state
            rates = []
            for _ in range(count):
                parameter.square().sum().backward()
                rates.append(optimizer.param_groups[0]['lr'])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            return rates

        full = setup(20)
        expected = advance(full, 20)
        first = setup(7)
        actual = advance(first, 7)
        optimizer_state = copy.deepcopy(first[1].state_dict())
        scheduler_state = copy.deepcopy(first[2].state_dict())
        resumed = setup(20)
        with torch.no_grad():
            resumed[0].copy_(first[0])
        resumed[1].load_state_dict(optimizer_state)
        resumed[2].load_state_dict(scheduler_state)
        actual.extend(advance(resumed, 13))
        self.assertEqual(actual, expected)
        self.assertTrue(torch.equal(resumed[0], full[0]))
        self.assertEqual(resumed[2].state_dict(), full[2].state_dict())


if __name__ == '__main__':
    unittest.main()
