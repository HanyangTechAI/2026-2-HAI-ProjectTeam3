"""Catalog acquired public age datasets without changing training splits or labels."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re


DATASETS = ('aaf', 'megaage_asian', 'afad', 'fgnet')
FIELDS = ('dataset,sample_id,path,age,age_label_type,gender,person_id,'
          'official_split,sha256,width,height,mode,age_band,usage').split(',')


def age_band(age):
    for maximum, label in ((2, '0-2'), (6, '3-6'), (9, '7-9'), (14, '10-14'),
                           (19, '15-19'), (29, '20-29'), (39, '30-39'),
                           (49, '40-49'), (69, '50-69'), (120, '70-120')):
        if age <= maximum:
            return label
    raise ValueError('Unsupported age')


def build_catalog(data_root, output, verify_hashes=False):
    """Paths in output CSVs are relative to data_root, not the output directory.

    These are acquisition/selection catalogs, not aligned age.train CSVs or
    gen.train_lora metadata. Unknown identities never imply disjoint people.
    """
    data_root, output = Path(data_root).resolve(), Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to replace existing catalog: {output}')
    rows, ids, paths = [], set(), set()
    summaries = {}
    for dataset in DATASETS:
        root = data_root / dataset
        manifest = root / 'samples.csv'
        if not manifest.is_file():
            summaries[dataset] = dict(status='unavailable')
            continue
        ages, bands, splits, genders, modes = Counter(), Counter(), Counter(), Counter(), Counter()
        identities = set()
        resolution = Counter()
        with manifest.open(newline='', encoding='utf-8-sig') as stream:
            reader = csv.DictReader(stream)
            required = set(FIELDS) - {'age_band', 'usage'}
            if not required <= set(reader.fieldnames or []):
                raise ValueError(f'Incomplete catalog schema: {manifest}')
            for source_row in reader:
                row = {key: source_row[key] for key in required}
                relative = Path(row['path'])
                if relative.is_absolute() or '..' in relative.parts:
                    raise ValueError(f'Unsafe source path: {relative}')
                image = root / relative
                if not image.resolve().is_relative_to(root.resolve()) or not image.is_file():
                    raise ValueError(f'Missing or escaping source image: {image}')
                key = (dataset, row['sample_id'])
                canonical = image.resolve()
                if row['dataset'] != dataset or not row['sample_id'] or key in ids or canonical in paths:
                    raise ValueError(f'Duplicate or inconsistent sample: {key}')
                ids.add(key)
                paths.add(canonical)
                age = int(row['age'])
                width, height = int(row['width']), int(row['height'])
                if not 0 <= age <= 120 or min(width, height) <= 0:
                    raise ValueError(f'Invalid image labels or dimensions: {image}')
                if not row['age_label_type'] or not re.fullmatch('[0-9a-f]{64}', row['sha256']):
                    raise ValueError(f'Missing label provenance or file digest: {image}')
                if verify_hashes:
                    with image.open('rb') as binary:
                        actual = hashlib.file_digest(binary, 'sha256').hexdigest()
                    if actual != row['sha256']:
                        raise ValueError(f'Image hash mismatch: {image}')
                row.update(path=(Path(dataset) / relative).as_posix(), age_band=age_band(age))
                split = row['official_split'].strip().lower()
                # AAF has no confirmed dataset license; FG-NET stays out of training.
                if dataset == 'aaf':
                    row['usage'] = 'permission_review'
                elif dataset == 'fgnet' or split in ('val', 'validation', 'test'):
                    row['usage'] = 'evaluation_only'
                elif (dataset == 'afad' and not split) or (dataset == 'megaage_asian' and split == 'train'):
                    row['usage'] = 'research_training_candidate'
                else:
                    row['usage'] = 'split_review'
                rows.append(row)
                ages[age] += 1
                bands[row['age_band']] += 1
                splits[row['official_split'] or 'unsplit'] += 1
                genders[row['gender'] or 'unknown'] += 1
                modes[row['mode']] += 1
                if row['person_id']:
                    identities.add(row['person_id'])
                for threshold in (64, 224, 256, 512):
                    if min(width, height) >= threshold:
                        resolution[str(threshold)] += 1
        summaries[dataset] = dict(status='cataloged', images=sum(ages.values()),
                                  age_range=[min(ages), max(ages)] if ages else None,
                                  ages=dict(sorted(ages.items())), age_bands=dict(bands),
                                  official_splits=dict(splits), genders=dict(genders), modes=dict(modes),
                                  annotated_identities=len(identities), min_side_at_least=dict(resolution),
                                  acquisition_report=f'{dataset}/acquisition.json')
    if not rows:
        raise ValueError('No acquired image catalogs found')
    hashes = defaultdict(list)
    for row in rows:
        hashes[row['sha256']].append(row)
    duplicates, conflicts = [], []
    for digest, group in hashes.items():
        if len(group) < 2:
            continue
        record = dict(sha256=digest, paths=[r['path'] for r in group])
        duplicates.append(record)
        if len({int(r['age']) for r in group}) > 1:
            conflicts.append(dict(record, ages=[int(r['age']) for r in group]))
            for row in group:
                if row['usage'] == 'research_training_candidate':
                    row['usage'] = 'conflicting_labels_review'
        elif any(r['usage'] == 'evaluation_only' or r['official_split'].strip().lower() in ('val', 'validation', 'test') for r in group):
            for row in group:
                if row['usage'] == 'research_training_candidate':
                    row['usage'] = 'heldout_duplicate_excluded'
        else:
            candidates = [r for r in group if r['usage'] == 'research_training_candidate']
            for row in candidates[1:]:
                row['usage'] = 'exact_duplicate_excluded'
    output.mkdir(parents=True)
    selections = {
        'all_samples.csv': rows,
        'generation_candidates.csv': [r for r in rows if r['usage'] == 'research_training_candidate' and min(int(r['width']), int(r['height'])) >= 256],
        'age_supervision_candidates.csv': [r for r in rows if r['usage'] == 'research_training_candidate' and min(int(r['width']), int(r['height'])) >= 64],
        'evaluation_only.csv': [r for r in rows if r['usage'] == 'evaluation_only'],
        'review_or_excluded.csv': [r for r in rows if r['usage'] not in ('research_training_candidate', 'evaluation_only')],
    }
    for name, selected in selections.items():
        with (output / name).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(selected)
    summary = dict(data_root=str(data_root), path_base='data_root', datasets=summaries,
                   selected_counts={name: len(selected) for name, selected in selections.items()},
                   usage_counts=dict(Counter(r['usage'] for r in rows)),
                   hashes_verified=verify_hashes, duplicate_hash_groups=len(duplicates),
                   conflicting_age_hash_groups=len(conflicts),
                   limitations=[
                       'Selections are not training-ready: face/quality filtering and model-specific preparation remain required.',
                       'No changes to AI Hub-only regression training or existing FFHQ splits.',
                       'AAF excluded from training candidates pending dataset permission confirmation.',
                       'AFAD and MegaAge-Asian are restricted to academic/non-commercial research; no commercial permission established.',
                       'FG-NET is evaluation-only here; publication availability does not establish unrestricted use.',
                       'No inferred identity labels; source identity annotations and image hashes cannot prove cross-dataset person disjointness.',
                       'AFAD has no official split in this catalog; supplied validation/test labels and their exact duplicates remain held out.',
                       'Age bands are derived for inventory only, not replacements for original age labels.',
                       'Generation candidates require min(width,height)>=256; age-supervision candidates require >=64; no upscaling was performed.',
                   ])
    for name, value in (('summary.json', summary), ('duplicates.json', duplicates), ('conflicting_labels.json', conflicts)):
        (output / name).write_text(json.dumps(value, indent=2, ensure_ascii=True) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-hashes', action='store_true')
    args = parser.parse_args()
    print(json.dumps(build_catalog(args.data_root, args.output, args.verify_hashes), indent=2))


if __name__ == '__main__':
    main()
