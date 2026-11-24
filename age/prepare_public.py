"""Prepare an explicitly experimental public age dataset without altering sources."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

from .common import FIELDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('/workspace/hai/shared/data'))
    parser.add_argument('--output', type=Path, default=Path('/workspace/hai/shared/data/age_public'))
    args = parser.parse_args()
    root, output = args.data_root.resolve(), args.output.absolute()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    rows = defaultdict(list)
    for source in ('afad', 'megaage_asian'):
        with (root / source / 'samples.csv').open(encoding='utf-8', newline='') as f:
            for row in csv.DictReader(f):
                path = (root / source / row['path']).resolve()
                if not path.is_relative_to(root / source) or not path.is_file():
                    raise ValueError(f'Missing or invalid source image: {path}')
                row['source_path'] = path
                rows[row['sha256']].append(row)

    splits = {name: [] for name in ('train', 'val', 'test')}
    counts = Counter()
    for digest, group in sorted(rows.items()):
        if len({row['age'] for row in group}) != 1:
            counts['conflicting_images_excluded'] += len(group)
            continue
        # A published test image and all its duplicates stay out of train and val.
        group.sort(key=lambda r: (r['official_split'] != 'test', r['dataset'], r['sample_id']))
        row = group[0]
        counts['duplicate_images_excluded'] += len(group) - 1
        if row['official_split'] == 'test':
            split = 'test'
        else:
            bucket = int(hashlib.sha256(('age-public-v1:' + digest).encode()).hexdigest()[:8], 16) % 100
            split = 'test' if row['dataset'] == 'afad' and bucket < 10 else 'val' if bucket < 20 else 'train'
        relative = row['source_path'].relative_to(root / row['dataset'])
        splits[split].append({
            'path': str(output / 'sources' / row['dataset'] / relative),
            'age': row['age'], 'gender': row['gender'], 'source': row['dataset'],
            'person_id': row['person_id'], 'original_id': row['sample_id'], 'sha256': digest,
        })
        counts[f'{split}_{row["dataset"]}'] += 1

    output.mkdir(parents=True)
    (output / 'sources').mkdir()
    for source in ('afad', 'megaage_asian'):
        (output / 'sources' / source).symlink_to(root / source, target_is_directory=True)
    for split, records in splits.items():
        with (output / f'{split}.csv').open('x', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(records)
    summary = {
        'counts': dict(counts),
        'split_counts': {split: len(records) for split, records in splits.items()},
        'seed': 'age-public-v1',
        'label_types': {'afad': 'reported chronological age', 'megaage_asian': 'estimated age, not verified chronological age'},
        'limitations': [
            'Research use only; verify source-specific conditions before use.',
            'Identity IDs are not available. Different images of the same person can cross splits.',
            'Byte-identical duplicates and age conflicts are excluded using existing source SHA-256 catalogues; near-duplicates are not checked.',
            'Original MegaAge test is held out. No independent identity or cross-dataset leakage guarantee.',
            'Images are not newly face-aligned or quality-filtered; input differs from the existing AI Hub pipeline.',
        ],
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
