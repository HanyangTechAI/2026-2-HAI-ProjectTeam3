"""Strict provenance-backed loading of the prepared FFHQ-Aging group dataset."""

from copy import deepcopy
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re

from PIL import Image
from torch.utils.data import Dataset

from .dataset import transform


GROUPS = ('0-2', '3-6', '7-9', '10-14', '15-19', '20-29', '30-39',
          '40-49', '50-69', '70-120')
SPLITS = ('train', 'val', 'test')


def _read_jsonl(path):
    with path.open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f'Invalid JSON: {path}:{number}') from exc
            if not isinstance(row, dict):
                raise ValueError(f'Expected JSON object: {path}:{number}')
            yield row


def load_records(root='data/lora', provenance='data/reports/ffhq_aging_samples.jsonl'):
    """Validate all three splits without changing data; preserve metadata order.

    Provenance paths are historical, not input locations. Labels are joined by
    image_number and split only, never inferred from captions or exact ages.
    """
    root = Path(root).resolve(strict=True)
    sources = {}
    required = {'image_number', 'age_group', 'gender', 'split', 'path',
                'caption', 'sha256', 'attribution'}
    for row in _read_jsonl(Path(provenance)):
        if not required <= row.keys():
            raise ValueError(f'Missing provenance fields: {sorted(required - row.keys())}')
        sample_id = row['image_number']
        if not isinstance(sample_id, str) or not re.fullmatch(r'[0-9]{5}', sample_id):
            raise ValueError(f'Invalid image_number: {sample_id!r}')
        if sample_id in sources:
            raise ValueError(f'Duplicate provenance image_number: {sample_id}')
        if row['split'] not in SPLITS:
            raise ValueError(f'Invalid provenance split: {row["split"]!r}')
        if row['age_group'] not in GROUPS:
            raise ValueError(f'Invalid age_group: {row["age_group"]!r}')
        if row['gender'] not in ('male', 'female'):
            raise ValueError(f'Invalid gender: {row["gender"]!r}')
        if not isinstance(row['caption'], str) or not row['caption']:
            raise ValueError(f'Invalid provenance caption: {sample_id}')
        if not isinstance(row['path'], str) or not row['path']:
            raise ValueError(f'Invalid provenance path: {sample_id}')
        if not isinstance(row['attribution'], dict):
            raise ValueError(f'Invalid attribution: {sample_id}')
        if not isinstance(row['sha256'], str) or not re.fullmatch(r'[0-9a-f]{64}', row['sha256']):
            raise ValueError(f'Invalid sha256: {sample_id}')
        sources[sample_id] = row

    records = {split: [] for split in SPLITS}
    used_ids, used_hashes = set(), set()
    for split in SPLITS:
        split_root = root / split
        if split_root.is_symlink():
            raise ValueError(f'Split directory must not be a symlink: {split_root}')
        split_root = split_root.resolve(strict=True)
        metadata = split_root / 'metadata.jsonl'
        if not metadata.resolve(strict=True).is_relative_to(split_root):
            raise ValueError(f'Metadata path escapes split: {metadata}')
        referenced = {metadata}
        for entry in _read_jsonl(metadata):
            name = entry.get('file_name')
            if not isinstance(name, str) or not name or '\\' in name:
                raise ValueError(f'Invalid file_name in {metadata}: {name!r}')
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError(f'Image path escapes split: {name}')
            path = split_root / relative
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(split_root):
                raise ValueError(f'Image path escapes split: {name}')
            if not resolved.is_file() or relative.suffix != '.png':
                raise ValueError(f'Expected PNG file: {name}')
            sample_id = relative.stem
            if sample_id in used_ids:
                raise ValueError(f'Duplicate metadata image_number across splits: {sample_id}')
            source = sources.get(sample_id)
            if source is None:
                raise ValueError(f'Missing provenance for image_number: {sample_id}')
            if source['split'] != split:
                raise ValueError(f'Provenance split mismatch: {sample_id}')
            if entry.get('text') != source['caption']:
                raise ValueError(f'Provenance caption mismatch: {sample_id}')

            # Hash and decode the same bytes, rather than trusting manifest hashes.
            content = resolved.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            if digest != source['sha256']:
                raise ValueError(f'Provenance sha256 mismatch: {sample_id}')
            if digest in used_hashes:
                raise ValueError(f'Duplicate image sha256 across dataset: {sample_id}')
            try:
                with Image.open(BytesIO(content)) as image:
                    if image.format != 'PNG' or image.mode != 'RGB' or image.size != (512, 512):
                        raise ValueError('Expected RGB 512x512 PNG')
                    image.verify()
                with Image.open(BytesIO(content)) as image:
                    image.load()
            except (OSError, SyntaxError, ValueError) as exc:
                raise ValueError(f'Invalid image {path}: {exc}') from exc

            row = deepcopy(source)
            row.update(path=str(resolved), label=GROUPS.index(source['age_group']), sha256=digest)
            records[split].append(row)
            used_ids.add(sample_id)
            used_hashes.add(digest)
            referenced.add(path)

        if not records[split]:
            raise ValueError(f'Expected nonempty split: {split}')
        # Do not silently ignore extra files or symlinked directory subtrees.
        def walk_error(error):
            raise error

        for directory, dirs, files in os.walk(split_root, onerror=walk_error):
            for name in dirs:
                if (Path(directory) / name).is_symlink():
                    raise ValueError(f'Symlink directory in split: {Path(directory) / name}')
            for name in files:
                path = Path(directory) / name
                if path not in referenced:
                    raise ValueError(f'Unreferenced file in split: {path}')

    unused = sources.keys() - used_ids
    if unused:
        raise ValueError(f'Unused provenance image_number: {sorted(unused)}')
    return records


def records_fingerprint(records):
    """Hash validated load_records output, independent of paths and row order.

    The sha256 values must come from load_records' actual-byte verification.
    Call load_records again when evaluating to detect changed dataset files.
    """
    rows = [(split, row['image_number'], row['age_group'], row['gender'], row['sha256'])
            for split in SPLITS for row in records[split]]
    payload = json.dumps(sorted(rows), ensure_ascii=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


class GroupDataset(Dataset):
    def __init__(self, records, training=False):
        self.rows = records
        self.transform = transform(training)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row['path']) as image:
            tensor = self.transform(image.convert('RGB'))
        return tensor, int(row['label'])
