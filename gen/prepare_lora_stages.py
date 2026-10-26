"""Prepare a separate corrected train split and preserve stage-three fingerprints."""
import argparse
import json
from pathlib import Path
import re
import shutil

from age.common import save_json, sha256, write_csv
from .pipeline import age_prompt


def prepare(source, provenance, previous, output, smoke_config):
    source, previous, output = Path(source).resolve(), Path(previous).resolve(), Path(output).resolve()
    if source.name != 'train' or not (source / 'images').is_dir():
        raise ValueError('Only an explicit train split is permitted')
    destination = output / 'data/train'
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'images').mkdir()
    records = {Path(row['path']).name: row for row in
               (json.loads(line) for line in Path(provenance).read_text().splitlines()) if row['split'] == 'train'}
    original_metadata = source / 'metadata.jsonl'
    rows = [json.loads(line) for line in original_metadata.read_text().splitlines()]
    if len(rows) != len(records) or {Path(row['file_name']).name for row in rows} != set(records):
        raise ValueError('Training metadata and source provenance counts differ')
    changes = []
    for row in rows:
        if Path(row['file_name']).is_absolute() or '..' in Path(row['file_name']).parts:
            raise ValueError('Training image filename must stay relative to train')
        path = (source / row['file_name']).resolve()
        if not path.is_relative_to(source / 'images'):
            raise ValueError('Image outside train/images')
        record = records[path.name]
        if record['caption'] != row['text'] or sha256(path) != record['sha256']:
            raise ValueError('Original training caption or hash differs from provenance')
        target = destination / row['file_name']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        if sha256(target) != record['sha256']:
            raise ValueError('Training image copy checksum mismatch')
        if record['age_group'] == '70-120':
            corrected = age_prompt(100, record['gender'])
            changes.append(dict(file_name=row['file_name'], age_group=record['age_group'],
                                original=row['text'], corrected=corrected,
                                reason='70-120 is not an exact 70s label; preserve broad 70+ interval'))
            row['text'] = corrected
    (destination / 'metadata.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    write_csv(output / 'caption_changes.csv', changes)
    initial = json.loads((previous / 'initial/summary.json').read_text())
    resumed = json.loads((previous / 'resume-12/summary.json').read_text())
    progress = re.findall(r'10/10 \[(\d+):(\d+)<', (previous / 'initial/train.log').read_text())
    loop_seconds = max(60 * int(minutes) + int(seconds) for minutes, seconds in progress)
    # Old logs lack per-update timers; use the displayed loop and total run as a bracket.
    lower = loop_seconds / initial['applied_steps']
    upper = initial['elapsed_seconds'] / initial['applied_steps']
    estimate = dict(source_initial=str(previous / 'initial/summary.json'),
                    source_resume=str(previous / 'resume-12/summary.json'),
                    old_applied_steps=initial['applied_steps'] + resumed['applied_steps'],
                    old_total_seconds=initial['elapsed_seconds'] + resumed['elapsed_seconds'],
                    old_loop_display_seconds=loop_seconds,
                    planning_seconds_per_step=[lower, upper],
                    stages={str(steps): dict(low_seconds=lower * steps, high_seconds=upper * steps)
                            for steps in (100, 300, 1000)},
                    note='Planning bracket, not confidence interval; old loop includes two large saves. '
                         'Recalibrate using actual 100-step synchronized measurements.')
    save_json(output / 'time_estimate_before.json', estimate)
    protected = {str(p): sha256(p) for p in sorted(previous.rglob('*')) if p.is_file() and 'cache' not in p.parts}
    protected.update({str(original_metadata): sha256(original_metadata)})
    save_json(output / 'preserved_sha256.json', protected)
    config = json.loads(Path(smoke_config).read_text())
    config.update(train_data_dir=str(destination), max_train_steps=100,
                  checkpointing_steps=100, checkpoints_total_limit=12)
    save_json(output / 'training_config.json', config)
    report = dict(original_train=str(source), derived_train=str(destination), train_images=len(rows),
                  changed_captions=len(changes), original_metadata_sha256=sha256(original_metadata),
                  derived_metadata_sha256=sha256(destination / 'metadata.jsonl'),
                  provenance_sha256=sha256(provenance), image_copy_method='independent byte-verified copy2',
                  validation_or_test_in_training=False, original_data_modified=False,
                  start_step=0, original12_is_separate_run=True,
                  start_reason='No full step12 state; corrected captions and deterministic resume sampling. '
                               'Keep old12 intact; new100/new300 belong to a single new trajectory.',
                  retention='No deletion: preserve old experiment; every 100-step full checkpoint through 1000 '
                            '(10 < limit12), exports, logs, validation inputs and outputs.',
                  free_disk_bytes=shutil.disk_usage(output).free)
    save_json(output / 'preparation.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('data/lora/train'))
    parser.add_argument('--provenance', type=Path, default=Path('data/reports/ffhq_aging_samples.jsonl'))
    parser.add_argument('--previous', type=Path, default=Path('outputs/age_lora_validation_20261026/train_retry'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke-config', type=Path, default=Path('gen/lora.smoke.json'))
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.provenance, args.previous, args.output, args.smoke_config), indent=2))


if __name__ == '__main__':
    main()
