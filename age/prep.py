"""Inspect/import source data, align faces, then split before augmentation."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re
import zipfile
from PIL import Image
from .common import ROOT, FIELDS, sha256, save_json, write_csv

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def failure(source, identifier, stage, reason):
    return dict(source=source, original_id=str(identifier), stage=stage, reason=str(reason))


def extract_archives(root, dest):
    """AI Hub ZIPs can contain leading '/'; never extract outside dest."""
    dest = Path(dest).resolve()
    stats = []
    source = Path(root)
    archives = [source] if source.is_file() else sorted(source.rglob('*.zip'))
    for archive in archives:
        target = dest / archive.stem
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                name = member.filename.replace('\\', '/').lstrip('/')
                out = (target / name).resolve()
                if not out.is_relative_to(target.resolve()) or ':' in name or (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError(f'Unsafe ZIP entry in {archive.name}')
                if member.is_dir():
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                # Reading through ZipFile verifies each entry's CRC.
                contents = z.read(member)
                if not out.exists():
                    out.write_bytes(contents)
                elif out.read_bytes() != contents:
                    raise ValueError(f'Existing extracted file differs: {out}')
            stats.append(dict(archive=archive.name, bytes=archive.stat().st_size,
                              sha256=sha256(archive), members=len(z.infolist()), crc='passed'))
    return stats


def valid_age(value):
    number = float(value)
    if not number.is_integer() or not 0 <= number <= 100:
        raise ValueError('age_out_of_range_or_noninteger')
    return int(number)


def source_records(root, source):
    root = Path(root)
    records, failures = [], []
    images = sorted(p for p in root.rglob('*') if p.suffix.lower() in EXTENSIONS)
    if source == 'aihub':
        by_stem = defaultdict(list)
        for image in images:
            by_stem[image.stem].append(image)
        matched = set()
        for label_path in sorted(root.rglob('*.json')):
            try:
                label = json.loads(label_path.read_text(encoding='utf-8-sig'))
                name = Path(str(label['filename'])).stem
                candidates = by_stem.get(name, [])
                if len(candidates) != 1:
                    raise ValueError('missing_or_ambiguous_image')
                # age_past is the age AT CAPTURE; age_now is current age.
                age = valid_age(label['age_past'])
                person = str(label['id']).strip()
                if not person:
                    raise ValueError('missing_person_id')
                gender = str(label['gender']).lower()
                if gender not in {'male', 'female'}:
                    raise ValueError('unknown_gender')
                path = candidates[0]
                matched.add(path)
                records.append(dict(raw_path=path, age=age, gender=gender, source=source,
                                    person_id=person, original_id=path.relative_to(root).as_posix()))
            except (ValueError, KeyError, TypeError, OSError) as e:
                failures.append(failure(source, label_path.relative_to(root), 'label', e))
        for image in set(images) - matched:
            failures.append(failure(source, image.relative_to(root), 'label', 'unmatched_image'))
    else:
        for path in images:
            try:
                if source == 'utkface':
                    match = re.match(r'^(\d+)_(\d+)_(\d+)_', path.name)
                    if not match:
                        raise ValueError('invalid_utk_filename')
                    age, gender = valid_age(match[1]), {'0': 'male', '1': 'female'}[match[2]]
                elif source == 'afad':
                    age, gender = valid_age(path.parent.parent.name), {'111': 'male', '112': 'female'}[path.parent.name]
                else:
                    raise ValueError(f'Unsupported source: {source}')
                records.append(dict(raw_path=path, age=age, gender=gender, source=source,
                                    person_id='', original_id=path.relative_to(root).as_posix()))
            except (ValueError, KeyError, TypeError) as e:
                failures.append(failure(source, path.relative_to(root), 'label', e))
    return records, failures


def split_records(rows, seed=42):
    """Keep people and pixel duplicates together, including duplicates across sources."""
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, row in enumerate(rows):
        keys = [('sha256', row['sha256'])]
        if row.get('person_id'):
            keys.append(('person', row['source'], row['person_id']))
        for key in keys:
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    per_source = defaultdict(list)
    for group in groups.values():
        per_source[tuple(sorted({r['source'] for r in group}))].append(group)
    result = {k: [] for k in ('train', 'val', 'test')}
    rng = random.Random(seed)
    for source, groups in sorted(per_source.items()):
        groups.sort(key=lambda g: min(r['original_id'] for r in g))
        rng.shuffle(groups)
        n = len(groups)
        # Tiny datasets are not a reliable evaluation; never split a person's photos.
        n_val = max(1, round(n * 0.1)) if n >= 3 else 0
        n_test = max(1, round(n * 0.1)) if n >= 3 else 0
        n_train = n - n_val - n_test
        for i, group in enumerate(groups):
            split = 'train' if i < n_train else ('val' if i < n_train + n_val else 'test')
            result[split].extend(group)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', choices=['utkface', 'afad', 'aihub'], required=True)
    p.add_argument('--raw-root', type=Path, required=True)
    p.add_argument('--extract-to', type=Path)
    p.add_argument('--output', type=Path, help='Default: data/age_aihub for AI Hub; data/age for public sources')
    p.add_argument('--reports', type=Path, default=ROOT / 'data/reports')
    p.add_argument('--inspect-only', action='store_true')
    p.add_argument('--limit', type=int, default=0, help='Positive value creates a separately named pilot dataset')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--model-root', default='models/insightface')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = p.parse_args()
    if args.output is None:
        args.output = ROOT / 'data' / ('age_aihub' if args.source == 'aihub' else 'age')
    if args.limit < 0:
        p.error('--limit must be >= 0')
    args.reports.mkdir(parents=True, exist_ok=True)
    raw = args.raw_root
    archives = []
    if args.extract_to:
        archives = extract_archives(raw, args.extract_to)
        raw = args.extract_to
    records, failures = source_records(raw, args.source)
    summary = dict(source=args.source, raw_root=str(raw.resolve()), archives=archives,
                   raw_images=sum(p.suffix.lower() in EXTENSIONS for p in raw.rglob('*')),
                   label_records=len(records), label_failures=len(failures),
                   people=len({r['person_id'] for r in records if r['person_id']}),
                   age_counts=dict(sorted(Counter(r['age'] for r in records).items())),
                   age_field='age_past' if args.source == 'aihub' else 'filename/directory',
                   identity_limit='No person IDs; same-person leakage cannot be ruled out.' if args.source != 'aihub' else '')
    inventory_path = args.reports / f'{args.source}_inventory.json'
    if not archives and inventory_path.exists():
        previous = json.loads(inventory_path.read_text(encoding='utf-8'))
        if previous.get('raw_root') == summary['raw_root']:
            summary['archives'] = previous.get('archives', [])
    save_json(inventory_path, summary)
    if args.inspect_only:
        write_csv(args.reports / f'{args.source}_label_failures.csv', failures,
                  ['source', 'original_id', 'stage', 'reason'])
        print(json.dumps({k: v for k, v in summary.items() if k not in {'raw_root', 'archives'}}, indent=2))
        return
    if not records:
        raise ValueError('No usable labels/images. Inspect source layout before processing.')
    from .face import FaceAligner, read_rgb
    import hashlib
    aligner = FaceAligner(args.model_root, args.device)
    # A pilot output cannot silently replace the full dataset.
    output = args.output / f'pilot_{args.source}_{args.limit}' if args.limit else args.output
    image_dir = output / 'images'
    image_dir.mkdir(parents=True, exist_ok=True)
    accepted, pixels_seen = [], set()
    for index, record in enumerate(records[:args.limit or None]):
        try:
            rgb = read_rgb(record['raw_path'])
            digest = hashlib.sha256(str(rgb.shape).encode() + rgb.tobytes()).hexdigest()
            if digest in pixels_seen:
                raise ValueError('duplicate_pixels')
            aligned, _ = aligner.align(rgb, training=True)
            pixels_seen.add(digest)
            name = hashlib.sha256((record['source'] + '/' + record['original_id']).encode()).hexdigest()[:24]
            dest = image_dir / f'{args.source}_{name}.png'
            Image.fromarray(aligned).save(dest)
            resolved = dest.resolve()
            stored_path = resolved.relative_to(ROOT).as_posix() if resolved.is_relative_to(ROOT) else str(resolved)
            accepted.append(dict(path=stored_path, age=record['age'], gender=record['gender'],
                                 source=record['source'], person_id=record['person_id'],
                                 original_id=record['original_id'], sha256=digest))
        except (ValueError, OSError) as e:
            failures.append(failure(args.source, record['original_id'], 'preprocess', e))
        if (index + 1) % 25 == 0:
            print(f'processed={index + 1} accepted={len(accepted)}', flush=True)
    # Preserve other sources' prepared manifests; rebuild the joint split.
    manifest_path = output / f'{args.source}_manifest.csv'
    write_csv(manifest_path, accepted, FIELDS)
    from .common import read_csv
    all_rows = [r for manifest in sorted(output.glob('*_manifest.csv')) for r in read_csv(manifest)]
    splits = split_records(all_rows, args.seed)
    from .dataset import validate_splits
    from types import SimpleNamespace
    validate_splits([SimpleNamespace(rows=rows) for rows in splits.values()])
    for split, rows in splits.items():
        write_csv(output / f'{split}.csv', rows, FIELDS)
    report_dir = args.reports / f'pilot_{args.source}_{args.limit}' if args.limit else args.reports
    write_csv(report_dir / f'{args.source}_preprocess_failures.csv', failures,
              ['source', 'original_id', 'stage', 'reason'])
    write_csv(report_dir / 'split_summary.csv',
              [dict(split=s, source=source, age_group=age_group, count=count, leakage='passed')
               for s, rows in splits.items()
               for (source, age_group), count in sorted(Counter((r['source'], min(int(r['age']) // 10, 9)) for r in rows).items())],
              ['split', 'source', 'age_group', 'count', 'leakage'])
    dataset_summary = []
    for source, count in sorted(Counter(r['source'] for r in all_rows).items()):
        inventory = json.loads((args.reports / f'{source}_inventory.json').read_text(encoding='utf-8'))
        dataset_summary.append(dict(source=source, raw_images=inventory.get('raw_images', ''),
                                    valid_labels=inventory['label_records'], label_failures=inventory['label_failures'],
                                    used=count, pilot=bool(args.limit)))
    write_csv(report_dir / 'dataset_summary.csv', dataset_summary,
              ['source', 'raw_images', 'valid_labels', 'label_failures', 'used', 'pilot'])
    save_json(report_dir / f'{args.source}_preprocess.json',
              dict(**summary, processed=min(args.limit or len(records), len(records)), accepted=len(accepted),
                   failures=len(failures), failure_counts=dict(Counter(f['reason'] for f in failures)),
                   seed=args.seed, split_counts={k: len(v) for k, v in splits.items()},
                   manifest_sha256=sha256(manifest_path), alignment='buffalo_l SCRFD / ArcFace-112 scaled to 224'))
    print('Split sizes:', {k: len(v) for k, v in splits.items()})
    if not splits['val'] or not splits['test']:
        print('Insufficient independent groups for validation/test. Training is intentionally blocked.')


if __name__ == '__main__':
    main()
