import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


MODULE = Path(__file__).resolve().parents[1] / 'docker/assets/prepare_models.py'
SPEC = importlib.util.spec_from_file_location('hai_prepare_models', MODULE)
assets = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assets)


class ModelAssetTests(unittest.TestCase):
    def test_preserves_existing_different_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / 'cached', root / 'model'
            source.write_bytes(b'new')
            target.write_bytes(b'user checkpoint')
            with self.assertRaises(FileExistsError):
                assets.install_file(source, target)
            self.assertEqual(target.read_bytes(), b'user checkpoint')

    def test_repeated_identical_install_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / 'cached', root / 'models/model'
            source.write_bytes(b'weights')
            first = assets.install_file(source, target)
            self.assertEqual(assets.install_file(source, target), first)
            self.assertEqual(target.read_bytes(), b'weights')

    def test_profile_download_uses_pin_and_does_not_copy_other_snapshot_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / 'cache'
            (snapshot / 'models/image_encoder').mkdir(parents=True)
            (snapshot / 'models/image_encoder/config.json').write_text('{}')
            (snapshot / 'unrequested.bin').write_bytes(b'large unrelated model')
            group = {'repo_id': 'owner/repo', 'revision': 'a' * 40,
                     'target': 'generation/encoder', 'strip_prefix': 'models/image_encoder',
                     'patterns': ['models/image_encoder/config.json']}
            calls = []
            def download(**kwargs):
                calls.append(kwargs)
                return str(snapshot)
            report = assets.download_group(group, root / 'models', download)
            self.assertEqual(calls[0]['revision'], 'a' * 40)
            self.assertTrue((root / 'models/generation/encoder/config.json').is_file())
            self.assertFalse((root / 'models/unrequested.bin').exists())
            self.assertEqual(len(report['files']), 1)

    def test_rejects_mutable_model_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                assets.download_group({'repo_id': 'a/b', 'revision': 'main'}, Path(directory), None)

    def test_paths_cannot_escape_asset_root(self):
        for path in ('../escape', '/absolute', 'safe/../../escape', 'bad\\name'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                assets.relative_path(path)

    def test_all_committed_model_sources_are_immutable(self):
        manifest = json.loads((MODULE.parent / 'models.json').read_text())
        for groups in manifest['profiles'].values():
            for group in groups:
                self.assertRegex(group['revision'], r'^[0-9a-f]{40}$')
                assets.relative_path(group['target'])


if __name__ == '__main__':
    unittest.main()
