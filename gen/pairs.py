"""Shared pair dataset P: (source face, target age, generated face), filtered by group judge + ArcFace."""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from age.common import save_json, sha256
from age.ffhq_data import GROUPS, load_records
from age.group_judge import expected_age, group_of
from gen import FaceNotFoundError

TARGET_AGES = (5, 15, 25, 35, 45, 55, 65, 75, 85)
MIN_ARCFACE = 0.6


def judge(source_embedding, output_embedding, probabilities, target_age, min_arcface=MIN_ARCFACE):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (len(GROUPS),):
        raise ValueError('Expected one probability per age group')
    predicted, target = int(probabilities.argmax()), group_of(target_age)
    reasons, arcface = [], None
    if output_embedding is None:
        reasons.append('no_face')
    else:
        arcface = float(np.dot(source_embedding, output_embedding))
        if arcface < min_arcface:
            reasons.append('identity')
    if predicted != target:
        reasons.append('age_group')
    return dict(keep=not reasons, reasons=reasons, arcface=arcface, predicted_group=GROUPS[predicted],
                predicted_probability=float(probabilities[predicted]), target_group=GROUPS[target])


def select_sources(records, limit, seed):
    """Round-robin across age groups so rare groups (children, 70+) are not crowded out."""
    rng, buckets = random.Random(seed), {}
    for row in records:
        buckets.setdefault(row['age_group'], []).append(row)
    queues = [buckets[group] for group in GROUPS if group in buckets]
    for rows in queues:
        rng.shuffle(rows)
    picked = []
    while len(picked) < limit and any(queues):
        for rows in queues:
            if rows and len(picked) < limit:
                picked.append(rows.pop())
    return picked


def _embedding(faces, image):
    try:
        return faces.embedding(image)
    except (FaceNotFoundError, ValueError):
        return None


def _write(handle, record):
    handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    handle.flush()


def build(records, teacher, judge_model, faces, output, settings, target_ages=TARGET_AGES, exclude=()):
    """Append one manifest row per (source, target age); rerunning with the same settings resumes."""
    output = Path(output)
    (output / 'images').mkdir(parents=True, exist_ok=True)
    settings_path, manifest = output / 'settings.json', output / 'manifest.jsonl'
    if settings_path.exists() and json.loads(settings_path.read_text(encoding='utf-8')) != settings:
        raise ValueError(f'{output} was built with different settings; use a new --output')
    save_json(settings_path, settings)
    done = set()
    if manifest.exists():
        done = {json.loads(line)['pair_id'] for line in manifest.read_text(encoding='utf-8').splitlines() if line}
    with manifest.open('a', encoding='utf-8') as out:
        for row in records:
            if row['image_number'] in exclude:
                continue
            pending = [age for age in target_ages if f"{row['image_number']}_{age:02d}" not in done]
            if not pending:
                continue
            with Image.open(row['path']) as image:
                source = image.convert('RGB')
            base = dict(generator=settings['generator'], source_image_number=row['image_number'],
                        source_path=str(row['path']), source_group=row['age_group'], gender=row['gender'])
            source_embedding = _embedding(faces, source)
            if source_embedding is None:
                for age in pending:
                    _write(out, dict(base, pair_id=f"{row['image_number']}_{age:02d}", target_age=age,
                                     output_path=None, keep=False, reasons=['source_face']))
                continue
            generated = teacher.age([source] * len(pending), pending)
            probabilities = judge_model.probabilities([source] + generated)
            source_age = expected_age(probabilities[0])
            for age, image, probs in zip(pending, generated, probabilities[1:]):
                pair_id = f"{row['image_number']}_{age:02d}"
                path = output / 'images' / f'{pair_id}.png'
                image.save(path)
                verdict = judge(source_embedding, _embedding(faces, image), probs, age)
                _write(out, dict(base, pair_id=pair_id, target_age=age, source_expected_age=source_age,
                                 output_path=str(path), **verdict))


def _read_manifest(output):
    return [json.loads(line) for line in (Path(output) / 'manifest.jsonl').read_text(encoding='utf-8').splitlines()
            if line]


def _contact_sheet(rows, path, sources, thumb):
    """Source in column 0, then each target age; rejected outputs get a red frame."""
    by_source = {}
    for row in rows:
        by_source.setdefault(row['source_image_number'], []).append(row)
    chosen = list(by_source.values())[:sources]
    ages = sorted({row['target_age'] for row in rows})
    sheet = Image.new('RGB', (thumb * (len(ages) + 1), thumb * max(1, len(chosen))), 'white')
    draw = ImageDraw.Draw(sheet)
    for y, items in enumerate(chosen):
        with Image.open(items[0]['source_path']) as image:
            sheet.paste(image.convert('RGB').resize((thumb, thumb)), (0, y * thumb))
        for item in items:
            if item['output_path'] is None:
                continue
            x = (ages.index(item['target_age']) + 1) * thumb
            with Image.open(item['output_path']) as image:
                sheet.paste(image.convert('RGB').resize((thumb, thumb)), (x, y * thumb))
            if not item['keep']:
                draw.rectangle([x, y * thumb, x + thumb - 1, (y + 1) * thumb - 1], outline='red', width=4)
    sheet.save(path)


def report(output, sheet_sources=8, thumb=128):
    output = Path(output)
    rows = _read_manifest(output)
    by_group = {}
    for row in rows:
        group = by_group.setdefault(GROUPS[group_of(row['target_age'])], dict(total=0, kept=0, scores=[]))
        group['total'] += 1
        group['kept'] += int(row['keep'])
        if row.get('arcface') is not None:
            group['scores'].append(row['arcface'])
    for group in by_group.values():
        scores = group.pop('scores')
        group['mean_arcface'] = float(np.mean(scores)) if scores else None
    summary = dict(total=len(rows), kept=sum(int(row['keep']) for row in rows),
                   reasons=dict(Counter(reason for row in rows for reason in row['reasons'])),
                   by_target_group=by_group)
    save_json(output / 'summary.json', summary)
    _contact_sheet(rows, output / 'contact_sheet.png', sheet_sources, thumb)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description='Pair dataset P: SAM teacher + age-group judge + ArcFace filter.')
    commands = parser.add_subparsers(dest='command', required=True)
    build_args = commands.add_parser('build')
    build_args.add_argument('--data-root', default='data/lora')
    build_args.add_argument('--provenance', default='data/reports/ffhq_aging_samples.jsonl')
    build_args.add_argument('--limit', type=int, default=2000)
    build_args.add_argument('--seed', type=int, default=42)
    build_args.add_argument('--exclude', type=Path, help='Text file of image_numbers to skip (identity-leak candidates)')
    build_args.add_argument('--sam-repo', type=Path, default=Path('third_party/SAM'))
    build_args.add_argument('--sam-checkpoint', type=Path, default=Path('models/sam/sam_ffhq_aging.pt'))
    build_args.add_argument('--judge', type=Path, required=True, help='best.pt from age.ffhq train')
    build_args.add_argument('--model-root', default='models/insightface')
    build_args.add_argument('--device', default='cuda')
    build_args.add_argument('--output', type=Path, required=True)
    report_args = commands.add_parser('report')
    report_args.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'build':
        from age.group_judge import GroupJudge
        from gen.faces import FaceTools
        from gen.sam_teacher import SamTeacher
        records = select_sources(load_records(args.data_root, args.provenance)['train'], args.limit, args.seed)
        exclude = set(args.exclude.read_text(encoding='utf-8').split()) if args.exclude else set()
        settings = dict(generator='sam', limit=args.limit, seed=args.seed, target_ages=list(TARGET_AGES),
                        min_arcface=MIN_ARCFACE, excluded=len(exclude),
                        sam_checkpoint_sha256=sha256(args.sam_checkpoint), judge_sha256=sha256(args.judge))
        build(records, SamTeacher(args.sam_repo, args.sam_checkpoint, args.device),
              GroupJudge(args.judge, args.device), FaceTools(args.model_root, 'cpu'),
              args.output, settings, exclude=exclude)
    print(json.dumps(report(args.output), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
