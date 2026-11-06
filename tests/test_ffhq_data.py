from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zlib

from PIL import Image
import torch

from age.ffhq_data import GROUPS, GroupDataset, load_records, records_fingerprint


class FFHQDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / 'lora'
        self.provenance = self.base / 'samples.jsonl'
        self.sources = []
        self.metadata = {split: [] for split in ('train', 'val', 'test')}
        for index, split in enumerate(('train', 'train', 'val', 'test')):
            sample_id = f'{index + 1:05d}'
            path = self.root / split / 'images' / f'{sample_id}.png'
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (512, 512), (index * 50, 30, 100)).save(path)
            caption = f'caption for {sample_id}, not an age label'
            self.sources.append(dict(
                image_number=sample_id, age_group=GROUPS[index],
                gender='male' if index % 2 else 'female', split=split,
                path=f'/obsolete/root/{split}/images/{sample_id}.png',
                caption=caption, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                attribution={'author': f'Author {index}', 'license': 'CC BY-NC-SA 4.0'},
                original_extra={'confidence': 1},
            ))
            self.metadata[split].append(dict(file_name=f'images/{sample_id}.png', text=caption))
        self.flush()

    def flush(self):
        self.provenance.write_text(''.join(json.dumps(row) + '\n' for row in self.sources), encoding='utf-8')
        for split, rows in self.metadata.items():
            (self.root / split / 'metadata.jsonl').write_text(
                ''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')

    def load(self):
        return load_records(self.root, self.provenance)

    def image_path(self, index=0):
        row = self.sources[index]
        return self.root / row['split'] / 'images' / f'{row["image_number"]}.png'

    def refresh_hash(self, index=0):
        self.sources[index]['sha256'] = hashlib.sha256(self.image_path(index).read_bytes()).hexdigest()
        self.flush()

    def test_labels_order_attribution_and_source_files_preserved(self):
        self.assertEqual(GROUPS, ('0-2', '3-6', '7-9', '10-14', '15-19',
                                  '20-29', '30-39', '40-49', '50-69', '70-120'))
        self.metadata['train'].reverse()
        self.flush()
        before = {p: p.read_bytes() for p in self.base.rglob('*') if p.is_file()}
        records = self.load()
        self.assertEqual(list(records), ['train', 'val', 'test'])
        self.assertEqual([r['image_number'] for r in records['train']], ['00002', '00001'])
        for split, rows in records.items():
            for row in rows:
                source = next(r for r in self.sources if r['image_number'] == row['image_number'])
                self.assertEqual(row, dict(source, path=str(self.root / split / 'images' /
                                                          f'{row["image_number"]}.png'),
                                           label=GROUPS.index(source['age_group'])))
                self.assertNotIn('age', row)
        self.assertEqual(before, {p: p.read_bytes() for p in self.base.rglob('*') if p.is_file()})
        records['train'][0]['attribution']['author'] = 'Changed copy'
        self.assertNotEqual(self.load()['train'][0]['attribution']['author'], 'Changed copy')

    def test_all_groups_are_explicit_labels_not_caption_inferences(self):
        for label, group in enumerate(GROUPS):
            with self.subTest(group=group):
                self.sources[0]['age_group'] = group
                self.flush()
                self.assertEqual(self.load()['train'][0]['label'], label)

    def test_relocation_preserves_fingerprint(self):
        original = self.load()
        moved = self.base / 'relocated'
        shutil.move(str(self.root), moved)
        relocated = load_records(moved, self.provenance)
        self.assertNotEqual(original['train'][0]['path'], relocated['train'][0]['path'])
        self.assertEqual(records_fingerprint(original), records_fingerprint(relocated))
        self.assertTrue(all(Path(r['path']).is_absolute() for rows in relocated.values() for r in rows))

    def test_fingerprint_is_stable_and_binds_required_fields(self):
        records = self.load()
        expected = records_fingerprint(records)
        self.assertRegex(expected, r'^[0-9a-f]{64}$')
        shuffled = {split: list(reversed(rows)) for split, rows in reversed(list(records.items()))}
        self.assertEqual(expected, records_fingerprint(shuffled))
        for field, value in [('image_number', '09999'), ('age_group', '70-120'),
                             ('gender', 'male'), ('sha256', '0' * 64)]:
            with self.subTest(field=field):
                changed = deepcopy(records)
                changed['train'][0][field] = value
                self.assertNotEqual(expected, records_fingerprint(changed))
        changed = deepcopy(records)
        changed['train'][0], changed['val'][0] = changed['val'][0], changed['train'][0]
        self.assertNotEqual(expected, records_fingerprint(changed))
        Image.new('RGB', (512, 512), (230, 12, 34)).save(self.image_path())
        self.refresh_hash()
        self.assertNotEqual(expected, records_fingerprint(self.load()))

    def test_dataset_returns_only_tensor_and_integer(self):
        records = self.load()['train']
        dataset = GroupDataset(records)
        self.assertEqual(len(dataset), 2)
        tensor, label = dataset[0]
        self.assertEqual(tuple(tensor.shape), (3, 224, 224))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertTrue(torch.isfinite(tensor).all())
        self.assertIs(type(label), int)
        self.assertEqual(label, 0)
        self.assertTrue(torch.equal(tensor, dataset[0][0]))

    def test_dataset_passes_training_flag_and_converts_rgb(self):
        path = self.base / 'gray.png'
        Image.new('L', (512, 512), 50).save(path)
        for training in (False, True):
            with self.subTest(training=training), patch('age.ffhq_data.transform') as factory:
                factory.return_value.side_effect = lambda image: image.mode
                dataset = GroupDataset([{'path': str(path), 'label': 9}], training=training)
                self.assertEqual(dataset[0], ('RGB', 9))
                factory.assert_called_once_with(training)

    def test_invalid_provenance_fields(self):
        invalid = [('age_group', '20s'), ('age_group', 20), ('gender', 'unknown'),
                   ('gender', 0), ('split', 'validation'), ('image_number', '../01'),
                   ('image_number', 1), ('sha256', 'bad'), ('caption', None),
                   ('path', ''), ('attribution', None)]
        for field, value in invalid:
            original = self.sources[0][field]
            with self.subTest(field=field, value=value):
                self.sources[0][field] = value
                self.flush()
                with self.assertRaises(ValueError):
                    self.load()
            self.sources[0][field] = original

    def test_missing_provenance_fields(self):
        for field in list(self.sources[0]):
            if field == 'original_extra':
                continue
            value = self.sources[0].pop(field)
            with self.subTest(field=field):
                self.flush()
                with self.assertRaisesRegex(ValueError, 'Missing provenance fields'):
                    self.load()
            self.sources[0][field] = value

    def test_missing_provenance_file(self):
        self.provenance.unlink()
        with self.assertRaises(FileNotFoundError):
            self.load()

    def test_missing_provenance_row(self):
        self.sources.pop(0)
        self.flush()
        with self.assertRaisesRegex(ValueError, 'Missing provenance for'):
            self.load()

    def test_unused_provenance_row(self):
        self.sources.append(dict(self.sources[0], image_number='09999'))
        self.flush()
        with self.assertRaisesRegex(ValueError, 'Unused provenance'):
            self.load()

    def test_duplicate_provenance_ids_within_or_across_splits(self):
        for split in ('train', 'test'):
            with self.subTest(split=split):
                self.sources.append(dict(self.sources[0], split=split))
                self.flush()
                with self.assertRaisesRegex(ValueError, 'Duplicate provenance'):
                    self.load()
                self.sources.pop()

    def test_duplicate_metadata_reference(self):
        self.metadata['train'].append(self.metadata['train'][0].copy())
        self.flush()
        with self.assertRaisesRegex(ValueError, 'Duplicate metadata'):
            self.load()

    def test_cross_split_duplicate_metadata_id(self):
        shutil.copyfile(self.image_path(), self.root / 'val/images/00001.png')
        self.metadata['val'].append(self.metadata['train'][0].copy())
        self.flush()
        with self.assertRaisesRegex(ValueError, 'Duplicate metadata'):
            self.load()

    def test_wrong_split(self):
        self.sources[0]['split'] = 'val'
        self.flush()
        with self.assertRaisesRegex(ValueError, 'split mismatch'):
            self.load()

    def test_caption_mismatch(self):
        self.metadata['train'][0]['text'] = 'different caption'
        self.flush()
        with self.assertRaisesRegex(ValueError, 'caption mismatch'):
            self.load()

    def test_hash_mismatch(self):
        self.sources[0]['sha256'] = '0' * 64
        self.flush()
        with self.assertRaisesRegex(ValueError, 'sha256 mismatch'):
            self.load()

    def test_duplicate_hashes_within_and_across_splits(self):
        for index in (1, 2):
            path = self.image_path(index)
            before = path.read_bytes()
            with self.subTest(split=self.sources[index]['split']):
                shutil.copyfile(self.image_path(), path)
                self.refresh_hash(index)
                with self.assertRaisesRegex(ValueError, 'Duplicate image sha256'):
                    self.load()
            path.write_bytes(before)
            self.refresh_hash(index)

    def test_empty_splits(self):
        for split in ('train', 'val', 'test'):
            rows = self.metadata[split]
            with self.subTest(split=split):
                self.metadata[split] = []
                self.flush()
                with self.assertRaisesRegex(ValueError, 'nonempty split'):
                    self.load()
            self.metadata[split] = rows

    def test_missing_metadata_and_image(self):
        for path in (self.root / 'train/metadata.jsonl', self.image_path()):
            content = path.read_bytes()
            with self.subTest(path=path):
                path.unlink()
                with self.assertRaises(FileNotFoundError):
                    self.load()
            path.write_bytes(content)

    def test_unreferenced_files(self):
        for name in ('extra.png', 'hidden/note.txt'):
            path = self.root / 'train' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with self.subTest(name=name):
                path.write_bytes(b'unreferenced')
                with self.assertRaisesRegex(ValueError, 'Unreferenced file'):
                    self.load()
            path.unlink()

    def test_absolute_traversal_and_cross_split_paths(self):
        for name in (str(self.image_path()), '../val/images/00003.png',
                     'images/../../val/images/00003.png', 'images\\00001.png'):
            with self.subTest(name=name):
                self.metadata['train'][0]['file_name'] = name
                self.flush()
                with self.assertRaisesRegex(ValueError, 'path escapes split|Invalid file_name'):
                    self.load()

    def test_symlink_image_escape(self):
        path = self.image_path()
        outside = self.base / path.name
        path.rename(outside)
        path.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'path escapes split'):
            self.load()

    def test_symlink_metadata_escape(self):
        path = self.root / 'train/metadata.jsonl'
        outside = self.base / 'metadata.jsonl'
        path.rename(outside)
        path.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'Metadata path escapes split'):
            self.load()

    def test_symlink_directory_escape(self):
        path = self.root / 'train/images'
        outside = self.base / 'images'
        path.rename(outside)
        path.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'path escapes split'):
            self.load()

    def test_unreferenced_symlink_directory(self):
        (self.root / 'train/extra').symlink_to(self.root / 'val', target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'Symlink directory'):
            self.load()

    def test_symlink_split(self):
        path = self.root / 'train'
        outside = self.base / 'train'
        path.rename(outside)
        path.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'Split directory must not be a symlink'):
            self.load()

    def test_wrong_mode_size_and_format(self):
        for mode, size, fmt in [('L', (512, 512), 'PNG'), ('RGB', (224, 224), 'PNG'),
                                ('RGB', (512, 512), 'JPEG')]:
            with self.subTest(mode=mode, size=size, fmt=fmt):
                Image.new(mode, size).save(self.image_path(), format=fmt)
                self.refresh_hash()
                with self.assertRaisesRegex(ValueError, 'RGB 512x512 PNG'):
                    self.load()

    def test_corrupt_image_with_matching_hash(self):
        content = self.image_path().read_bytes()
        for broken in (b'not an image', content[:100], content[:-20]):
            with self.subTest(length=len(broken)):
                self.image_path().write_bytes(broken)
                self.refresh_hash()
                with self.assertRaisesRegex(ValueError, 'Invalid image'):
                    self.load()

    def test_invalid_pixels_with_valid_png_chunk_checksums(self):
        content = bytearray(self.image_path().read_bytes())
        start = content.index(b'IDAT')
        length = int.from_bytes(content[start - 4:start], 'big')
        content[start + 4:start + 4 + length] = b'\x00' * length
        checksum = zlib.crc32(content[start:start + 4 + length])
        content[start + 4 + length:start + 8 + length] = checksum.to_bytes(4, 'big')
        self.image_path().write_bytes(content)
        self.refresh_hash()
        with Image.open(self.image_path()) as image:
            image.verify()  # Valid container alone is insufficient: load must decode pixels.
        with self.assertRaisesRegex(ValueError, 'Invalid image'):
            self.load()

    def test_invalid_jsonl(self):
        for path in (self.provenance, self.root / 'train/metadata.jsonl'):
            original = path.read_bytes()
            for content in ('{broken\n', '[]\n'):
                with self.subTest(path=path, content=content):
                    path.write_text(content, encoding='utf-8')
                    with self.assertRaisesRegex(ValueError, 'Invalid JSON|Expected JSON object'):
                        self.load()
            path.write_bytes(original)


if __name__ == '__main__':
    unittest.main()
