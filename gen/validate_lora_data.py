"""Read-only validation of prepared local FFHQ-Aging ImageFolder splits."""
import argparse
from collections import Counter
import json
from pathlib import Path

from PIL import Image

from age.common import save_json, sha256, write_csv
from .pipeline import AGE_PHRASES, age_prompt

EXPECTED_COUNTS = {'train': 8163, 'val': 400, 'test': 100}
CAPTIONS = {age_prompt(upper, gender): (upper, gender)
            for upper, _ in AGE_PHRASES for gender in ('male', 'female')}
# The original prepared dataset is immutable; recognize its documented old labels.
LEGACY_CAPTIONS = {'a photo of a elderly man in his 70s': age_prompt(100, 'male'),
                   'a photo of a elderly woman in her 70s': age_prompt(100, 'female')}


def validate_dataset(root, expected_counts=None):
    """Decode every image, audit references and hashes; never mutate the dataset."""
    root = Path(root).resolve()
    errors, splits, hashes, ids, records, groups = [], {}, {}, {}, {}, []

    def issue(split, kind, item):
        errors.append(dict(split=split, kind=kind, item=str(item)))

    for split in ('train', 'val', 'test'):
        directory = root / split
        images, metadata = directory / 'images', directory / 'metadata.jsonl'
        stats = dict(images_exists=images.is_dir(), metadata_exists=metadata.is_file(),
                     image_count=0, metadata_rows=0, valid_images=0, valid_references=0,
                      nonempty_captions=0, legacy_caption_rows=0)
        splits[split] = stats
        if not images.is_dir():
            issue(split, 'missing_images_directory', images)
        if not metadata.is_file():
            issue(split, 'missing_metadata', metadata)
        files = sorted(p for p in images.rglob('*') if p.is_file())
        stats['image_count'] = len(files)
        referenced, caption_counts = set(), Counter()
        if metadata.is_file():
            with metadata.open(encoding='utf-8') as stream:
                for line_number, line in enumerate(stream, 1):
                    stats['metadata_rows'] += 1
                    try:
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            raise ValueError('Expected an object')
                    except (ValueError, TypeError):
                        issue(split, 'invalid_json', line_number)
                        continue
                    caption = row.get('text')
                    if not isinstance(caption, str) or not caption.strip():
                        issue(split, 'empty_caption', line_number)
                    else:
                        stats['nonempty_captions'] += 1
                        caption_counts[LEGACY_CAPTIONS.get(caption, caption)] += 1
                        stats['legacy_caption_rows'] += int(caption in LEGACY_CAPTIONS)
                        if caption not in CAPTIONS and caption not in LEGACY_CAPTIONS:
                            issue(split, 'noncanonical_caption', line_number)
                    name = row.get('file_name')
                    if not isinstance(name, str) or not name or Path(name).is_absolute():
                        issue(split, 'invalid_file_name', line_number)
                        continue
                    path = (directory / name).resolve()
                    if not path.is_relative_to(images.resolve()) or not path.is_relative_to(directory):
                        issue(split, 'path_outside_split_images', name)
                        continue
                    if path in referenced:
                        issue(split, 'duplicate_reference', name)
                    referenced.add(path)
                    if not path.is_file():
                        issue(split, 'missing_image', name)
                    else:
                        stats['valid_references'] += 1
                        records[str(path)] = dict(split=split, text=caption)
        for path in files:
            resolved = path.resolve()
            if not resolved.is_relative_to(directory):
                issue(split, 'image_outside_split', path)
                continue
            if resolved not in referenced:
                issue(split, 'unreferenced_image', path.name)
            try:
                with Image.open(path) as image:
                    image.load()
                    if image.size != (512, 512) or image.mode != 'RGB':
                        issue(split, 'wrong_size_or_mode', f'{path.name}: {image.size} {image.mode}')
                    else:
                        stats['valid_images'] += 1
                digest = sha256(path)
                if digest in hashes:
                    issue(split, 'duplicate_image_hash', f'{path}: {hashes[digest]}')
                hashes[digest] = str(path)
                if path.stem in ids:
                    issue(split, 'duplicate_image_id', f'{path}: {ids[path.stem]}')
                ids[path.stem] = str(path)
                if str(resolved) in records:
                    records[str(resolved)]['sha256'] = digest
            except (OSError, ValueError) as exc:
                issue(split, 'unreadable_image', f'{path.name}: {exc}')
        if stats['image_count'] != stats['metadata_rows']:
            issue(split, 'count_mismatch', stats)
        if expected_counts is not None and stats['image_count'] != expected_counts[split]:
            issue(split, 'unexpected_count', stats['image_count'])
        for caption, (upper, gender) in CAPTIONS.items():
            count, cap = caption_counts[caption], {'train': 500, 'val': 20, 'test': 5}[split]
            groups.append(dict(split=split, age_upper=upper, gender=gender, caption=caption,
                               count=count, cap=cap, shortfall=max(0, cap - count)))
            if count > cap:
                issue(split, 'cell_over_cap', caption)
        stats['error_counts'] = dict(Counter(e['kind'] for e in errors if e['split'] == split))
        stats['valid'] = not stats['error_counts']
        stats['metadata_sha256'] = sha256(metadata) if metadata.is_file() else None
    return dict(root=str(root), valid=not errors, splits=splits, errors=errors,
                groups=groups, records=records, corrections=[],
                identity_disjoint_verified=False,
                limitations=['File hashes/IDs do not establish person-disjoint splits.',
                             'Youth and oldest training groups are undersampled.',
                             'Age captions describe groups, not exact ages; original 70s captions are legacy labels.'])


def validate_provenance(report, manifest, project_root):
    """Check prepared source records without downloading original data."""
    seen, errors, licenses = set(), [], Counter()
    with Path(manifest).open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
                path = str((Path(project_root) / row['path']).resolve())
                record = report['records'][path]
                if path in seen:
                    raise ValueError('duplicate provenance')
                seen.add(path)
                if not (float(row['age_group_confidence']) > .8 and abs(float(row['head_yaw'])) < 30):
                    raise ValueError('selection thresholds failed')
                if any(row[field] != record[key] for field, key in
                       [('split', 'split'), ('caption', 'text'), ('sha256', 'sha256')]):
                    raise ValueError('split/caption/hash mismatch')
                attribution = row['attribution']
                if not all(attribution.get(key) for key in ('author', 'photo_url', 'license', 'license_url')):
                    raise ValueError('incomplete attribution')
                licenses[attribution['license']] += 1
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(dict(line=number, error=str(exc)))
    missing = sorted(set(report['records']) - seen)
    result = dict(path=str(Path(manifest).resolve()), checked=len(seen), errors=errors,
                  missing=missing, licenses=dict(licenses), sha256=sha256(manifest),
                  valid=not errors and not missing)
    report['provenance'] = result
    report['valid'] = report['valid'] and result['valid']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/lora'))
    parser.add_argument('--output', type=Path, required=True, help='New report directory')
    parser.add_argument('--provenance', type=Path)
    parser.add_argument('--project-root', type=Path, default=Path('.'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = validate_dataset(args.root, EXPECTED_COUNTS)
    if args.provenance:
        validate_provenance(report, args.provenance, args.project_root)
    write_csv(args.output / 'groups.csv', report['groups'])
    save_json(args.output / 'images.json', report.pop('records'))
    save_json(args.output / 'validation.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('groups', 'errors')}, indent=2))
    raise SystemExit(0 if report['valid'] else 1)


if __name__ == '__main__':
    main()
