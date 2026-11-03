"""Render recorded diagnostics without running models or claiming human review."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from PIL import Image

from age.common import save_json
from .compare_lora import _sheets
from .quality_experiment import load_rows


def analyze(source, output):
    source, output = Path(source).resolve(), Path(output)
    plan = json.loads((source / 'plan.json').read_text(encoding='utf-8'))
    rows = load_rows(source)
    output.mkdir(parents=True, exist_ok=False)
    results = {}
    low, high = min(plan['targets']), max(plan['targets'])
    for condition in plan['conditions']:
        selected = [r for r in rows if r['condition'] == condition]
        paired = defaultdict(dict)
        for row in selected:
            paired[(row['sample_id'], row['seed'])][row['target']] = row
        stats = dict(completed_attempts=len(selected),
                     planned_attempts=len(plan['samples']) * len(plan['seeds']) * len(plan['targets']),
                     errors=sum(bool(r['error']) for r in selected))
        for field in ('reference_age', 'raw_reference_age'):
            deltas = [group[high][field] - group[low][field] for group in paired.values()
                      if high in group and low in group and high != low
                      and group[high].get(field) is not None and group[low].get(field) is not None
                      and group[high]['status'] == group[low]['status'] == 'ok']
            stats[field + '_paired_span_mean'] = sum(deltas) / len(deltas) if deltas else None
            stats[field + '_paired_span_count'] = len(deltas)
        for field in ('identity_cosine', 'raw_identity_cosine', 'protected_changed_pixels', 'elapsed_seconds',
                      'annotation_protected_changed_pixels', 'annotation_outside_edit_changed_pixels'):
            values = [r[field] for r in selected if r.get(field) is not None]
            stats[field + '_mean'] = sum(values) / len(values) if values else None
            stats[field + '_count'] = len(values)
        differences = []
        for group in paired.values():
            if (low == high or low not in group or high not in group or
                    any(group[target]['status'] != 'ok' or not group[target].get('output_path') for target in (low, high))):
                continue
            with Image.open(source / group[low]['output_path']) as before, Image.open(source / group[high]['output_path']) as after:
                differences.append(int(np.count_nonzero(np.any(np.asarray(before) != np.asarray(after), axis=2))))
        stats['endpoint_pixel_pairs'] = len(differences)
        stats['identical_endpoint_pairs'] = sum(value == 0 for value in differences)
        stats['endpoint_changed_pixels_mean'] = sum(differences) / len(differences) if differences else None
        results[condition] = stats
    sheets = {}
    for seed in plan['seeds']:
        folder = output / f'seed_{seed}'
        folder.mkdir()
        indexed = {(r['condition'], r['sample_id'], r['target']): r for r in rows if r['seed'] == seed}
        display = []
        for condition in plan['conditions']:
            for sample in plan['samples']:
                for target in plan['targets']:
                    row = indexed.get((condition, sample['sample_id'], target))
                    display.append(dict(condition=condition, sample_id=sample['sample_id'], target_age=target,
                                        status=row['status'] if row else 'NOT RUN',
                                        output_path=str(source / row['output_path']) if row and row['output_path'] else '',
                                        identity_cosine=row.get('identity_cosine') if row else None))
        sheets[str(seed)] = _sheets(folder, plan['samples'], plan['targets'], list(plan['conditions']), display)
    result = dict(conditions=results, source=str(source), low_target=low, high_target=high,
                  age_mae=None, human_review_complete=False, sheets=sheets,
                  note='Spans use matched endpoint pairs, not chronological accuracy. '
                       'Raw crop and composited full image differ in geometry; compare only diagnostically. '
                       'Different endpoint pixels do not establish correct age control. '
                       'AI visual inspection is not blinded human validation.')
    save_json(output / 'analysis.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.source, args.output), indent=2))


if __name__ == '__main__':
    main()
