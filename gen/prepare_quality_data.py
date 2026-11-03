"""Train-only data preparation, never training or production crop approval.

Use a NEW output directory and an explicit UTC deadline. Wrap real commands in
gen.record_command for a hard process deadline, including native ONNX calls.
Interrupted runs retain fsynced journals and are not resumable; use a new output.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time
from unittest.mock import patch

from .audit_lora_split import AGE_GROUPS, GENDERS, ORIGINAL_CAPTIONS


class DeadlineReached(TimeoutError):
    pass


def check_deadline(deadline, reserve_seconds=0):
    if time.time() >= deadline - reserve_seconds:
        raise DeadlineReached('Preparation deadline/reserve reached')


def digest(path, check=lambda: None):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            check()
            block = stream.read(1024 * 1024)
            if not block:
                break
            value.update(block)
    return value.hexdigest()


def caption_digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def range_caption(age_group, gender):
    """Use label intervals, never invented exact ages or cross-age pairs."""
    if age_group not in AGE_GROUPS or gender not in GENDERS:
        raise ValueError('Unsupported age_group or gender')
    interval = '70 or older' if age_group == '70-120' else age_group.replace('-', ' to ') + ' years'
    return f'a photo of a {gender} person aged {interval}'


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False)


def _write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        stream.write(_json(value) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _append(stream, value):
    stream.write(_json(value) + '\n')
    stream.flush()
    os.fsync(stream.fileno())


def _read_rows(path, check):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result

    rows = []
    with path.open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            check()
            row = json.loads(line, object_pairs_hook=unique_object)
            if not isinstance(row, dict):
                raise ValueError(f'Expected JSON object at {path}:{number}')
            rows.append(row)
    return rows


def load_train(source, provenance, check=lambda: None):
    """Validate the entire train inventory; never open val/Test image inputs.

    The existing shared provenance index includes other splits. Only its train
    rows are joined; all train rows must match, without label inference from text.
    """
    source, provenance = Path(source).resolve(), Path(provenance).resolve()
    if source.name != 'train' or not (source / 'images').is_dir():
        raise ValueError('Only an explicit train/images source is allowed')
    if (source / 'images').is_symlink() or (source / 'metadata.jsonl').is_symlink():
        raise ValueError('Symlinked source inventory is not allowed')
    metadata_path = source / 'metadata.jsonl'
    before = dict(metadata_sha256=digest(metadata_path, check), provenance_sha256=digest(provenance, check))
    metadata = _read_rows(metadata_path, check)
    provenance_rows = _read_rows(provenance, check)
    sources, ignored = {}, Counter()
    for row in provenance_rows:
        check()
        if row.get('split') != 'train':
            if row.get('split') not in ('val', 'validation', 'test', 'Test'):
                raise ValueError('Missing/unknown provenance split')
            ignored[row['split']] += 1
            continue
        name = row.get('image_number')
        if not isinstance(name, str) or not name or name in sources:
            raise ValueError('Missing or duplicate train provenance image_number')
        origin = Path(row.get('path', ''))
        if (origin.parent.name != 'images' or origin.parent.parent.name != 'train'
                or origin.stem != name or '..' in origin.parts):
            raise ValueError('Provenance image path is not an explicit train image')
        if not all(row.get('attribution', {}).get(key) for key in
                   ('author', 'photo_url', 'license', 'license_url')):
            raise ValueError('Missing original image attribution')
        confidence, yaw = float(row['age_group_confidence']), float(row['head_yaw'])
        if not math.isfinite(confidence) or not .8 < confidence <= 1 or not math.isfinite(yaw) or abs(yaw) >= 30:
            raise ValueError('Provenance selection thresholds failed')
        sources[name] = row
    records, referenced, hashes = [], set(), set()
    for index, meta in enumerate(metadata):
        check()
        name = meta.get('file_name')
        if not isinstance(name, str):
            raise ValueError('Missing metadata file_name')
        relative = Path(name)
        # This dataset has one flat images directory. Reject aliases/traversal.
        if (len(relative.parts) != 2 or relative.parts[0] != 'images'
                or relative.as_posix() != name or relative.name in ('.', '..')):
            raise ValueError('Metadata must reference a direct train/images file')
        path = source / relative
        if path.is_symlink() or not path.is_file() or path.resolve().parent != source / 'images':
            raise ValueError('Missing or nonlocal train image')
        if name in referenced:
            raise ValueError('Duplicate metadata reference')
        referenced.add(name)
        original = sources.pop(path.stem, None)
        if original is None or Path(original['path']).name != path.name:
            raise ValueError('Metadata has no matching train provenance')
        age, gender = original['age_group'], original['gender']
        expected = ORIGINAL_CAPTIONS.get((age, gender))
        if not expected or original.get('caption') != expected:
            raise ValueError('Original provenance label/caption mismatch')
        corrected = f'a photo of a elderly {"man" if gender == "male" else "woman"} aged 70 or older'
        allowed = {expected, range_caption(age, gender)}
        if age == '70-120':
            allowed.add(corrected)
        if meta.get('text') not in allowed:
            raise ValueError('Training caption is not supported by the source label')
        for key, expected_value in (('split', 'train'), ('age_group', age), ('gender', gender),
                                    ('image_number', path.stem), ('sha256', original['sha256'])):
            if key in meta and meta[key] != expected_value:
                raise ValueError(f'Metadata/provenance mismatch: {key}')
        if 'caption' in meta and meta['caption'] != meta['text']:
            raise ValueError('Metadata caption/text mismatch')
        if 'caption_sha256' in meta and meta['caption_sha256'] != caption_digest(meta['text']):
            raise ValueError('Stale metadata caption_sha256')
        expected_hash = original['sha256']
        if (not isinstance(expected_hash, str) or len(expected_hash) != 64
                or any(c not in '0123456789abcdef' for c in expected_hash) or expected_hash in hashes):
            raise ValueError('Invalid or duplicate provenance image SHA256')
        hashes.add(expected_hash)
        records.append(dict(index=index, file_name=name, sample_id=f'ffhq_{path.stem}',
                            age_group=age, gender=gender, source_sha256=expected_hash,
                            metadata=meta, provenance=original))
    actual = set()
    for path in (source / 'images').iterdir():
        check()
        if not path.is_file() or path.is_symlink():
            raise ValueError('Unexpected image-directory entry')
        actual.add('images/' + path.name)
    if not records or sources or actual != referenced:
        raise ValueError('Empty/incomplete train inventory or unreferenced images')
    if digest(metadata_path, check) != before['metadata_sha256'] or digest(provenance, check) != before['provenance_sha256']:
        raise ValueError('Metadata/provenance changed during validation')
    return records, dict(before, ignored_nontrain_provenance_rows=dict(ignored))


def sampling_plan(records, power=.5):
    if isinstance(power, bool) or not math.isfinite(power) or not 0 <= power <= 1:
        raise ValueError('power must be finite in [0, 1]')
    if not records:
        raise ValueError('Cannot sample an empty dataset')
    counts = Counter((r['age_group'], r['gender']) for r in records)
    raw = [counts[r['age_group'], r['gender']] ** (-power) for r in records]
    total = math.fsum(raw)
    probabilities = [weight / total for weight in raw]
    if not all(math.isfinite(p) and p > 0 for p in probabilities) or not math.isclose(math.fsum(probabilities), 1):
        raise ValueError('Invalid sampling probabilities')
    rows = [dict(index=r['index'], file_name=r['file_name'], sample_id=r['sample_id'],
                 age_group=r['age_group'], gender=r['gender'], source_sha256=r['source_sha256'],
                 count=counts[r['age_group'], r['gender']], raw_weight=weight,
                 weight=probability, probability=probability)
            for r, weight, probability in zip(records, raw, probabilities)]
    groups = [dict(age_group=age, gender=gender, count=counts[age, gender],
                   probability_mass=counts[age, gender] ** (1 - power) / total if counts[age, gender] else 0)
              for age in AGE_GROUPS for gender in GENDERS]
    return dict(kind='tempered_inverse_frequency_sampling_plan', preparation_only=True,
                trainer_consumes_plan=False, sampling_order_generated=False,
                weighting='count ** (-power), normalized to sum one in metadata order',
                intended_draws='with replacement; not integrated into trainer or resume state',
                power=power, count=len(rows), groups=groups, rows=rows,
                weight_sum=math.fsum(probabilities), min_weight=min(probabilities), max_weight=max(probabilities),
                effective_sample_size=1 / math.fsum(p * p for p in probabilities))


def _cpu_faces(model_root, threads):
    if threads < 1:
        raise ValueError('threads must be positive')
    import cv2
    import onnxruntime as ort
    from .faces import FaceTools
    cv2.setNumThreads(1)
    factory = ort.InferenceSession

    def cpu_session(path, sess_options=None, **kwargs):
        options = sess_options or ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        kwargs['providers'] = ['CPUExecutionProvider']
        return factory(path, sess_options=options, **kwargs)

    with patch.object(ort, 'InferenceSession', cpu_session):
        faces = FaceTools(model_root, device='cpu', identity=False)
    for model in (faces.aligner.detector, faces.aligner.attribute):
        if model.session.get_providers() != ['CPUExecutionProvider']:
            raise RuntimeError('Crop preview requires CPU-only providers')
    return faces


def _plain(value):
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, 'tolist'):
        return value.tolist()
    return value


def prepare(command, source, provenance, output, *, deadline, reserve_seconds=15,
            power=.5, limit_per_group=1, model_root='models/insightface', threads=2,
            faces_factory=None):
    """Return a durable status report. Non-success never publishes train/."""
    if command not in ('captions', 'sampling', 'crop-preview'):
        raise ValueError('Unknown preparation command')
    if not math.isfinite(deadline) or not math.isfinite(reserve_seconds) or reserve_seconds < 0:
        raise ValueError('Explicit finite deadline and nonnegative reserve required')
    if not isinstance(limit_per_group, int) or isinstance(limit_per_group, bool) or limit_per_group < 1:
        raise ValueError('limit_per_group must be positive')
    if command == 'sampling' and (isinstance(power, bool) or not math.isfinite(power) or not 0 <= power <= 1):
        raise ValueError('power must be finite in [0, 1]')
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
        raise ValueError('threads must be a positive integer')
    source, provenance, output = Path(source).resolve(), Path(provenance).resolve(), Path(output).resolve()
    if output.is_relative_to(source) or source.is_relative_to(output) or provenance.is_relative_to(output):
        raise ValueError('Output must be separate from source/provenance')
    if command == 'crop-preview' and output.name.lower() == 'train':
        raise ValueError('Preview output must not be named train')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(command=command, status='running', preparation_only=True,
                  trainer_consumes_plan=False, production_crop_dataset_complete=False,
                  source=str(source), provenance=str(provenance), output=str(output),
                  deadline_utc=datetime.fromtimestamp(deadline, timezone.utc).isoformat(),
                  reserve_seconds=reserve_seconds, completed_rows=0, failed_rows=0,
                  changed_captions=0, train_published=False, validation_or_test_images_used=False,
                  interrupted_outputs_must_not_be_used=True,
                  tool_sha256=digest(Path(__file__)))
    _write(output / 'summary.json', report)
    check = lambda: check_deadline(deadline, reserve_seconds)
    try:
        check()
        records, fingerprint = load_train(source, provenance, check)
        report.update(fingerprint, input_rows=len(records))
        _write(output / 'summary.json', report)
        with (output / 'source_provenance.jsonl').open('x', encoding='utf-8') as stream:
            for row in records:
                check()
                stream.write(_json(row['provenance']) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        report['train_provenance_snapshot_sha256'] = digest(output / 'source_provenance.jsonl', check)
        # Preserve the exact source metadata bytes, including unrelated fields.
        shutil.copyfile(source / 'metadata.jsonl', output / 'source_metadata.jsonl')
        with (output / 'source_metadata.jsonl').open('rb') as stream:
            os.fsync(stream.fileno())
        if digest(output / 'source_metadata.jsonl', check) != fingerprint['metadata_sha256']:
            raise ValueError('Source metadata snapshot mismatch')
        selected = records
        if command == 'crop-preview':
            selected, seen = [], Counter()
            for row in records:
                cell = row['age_group'], row['gender']
                if seen[cell] < limit_per_group:
                    selected.append(row)
                    seen[cell] += 1
            report.update(selected_rows=len(selected), limit_per_group=limit_per_group,
                          selection='first N in original metadata order per age_group x gender; no quality filtering',
                          source_group_counts=sampling_plan(records)['groups'],
                          providers=['CPUExecutionProvider'], identity_model_loaded=False,
                          cpu_threads=threads,
                          crop_contract='FaceTools.prepare_edit(image), all default arguments; no label correction',
                          crop_code_sha256={str(p): digest(p, check) for p in
                                           (Path(__file__).with_name('faces.py'),
                                            Path(__file__).parents[1] / 'age/face.py')})
            (output / 'preview').mkdir(exist_ok=False)
            _write(output / 'preview_plan.json', [dict(index=r['index'], file_name=r['file_name'],
                   sample_id=r['sample_id'], age_group=r['age_group'], gender=r['gender']) for r in selected])
            check()
            faces = (faces_factory or _cpu_faces)(model_root, threads)
        elif command == 'captions':
            (output / '.pending_train').mkdir(exist_ok=False)
            (output / '.pending_train/images').mkdir(exist_ok=False)
        derived = []
        with (output / 'rows.jsonl').open('x', encoding='utf-8') as stream:
            for row in selected:
                check()
                path = source / row['file_name']
                result = {key: row[key] for key in ('index', 'sample_id', 'file_name', 'age_group', 'gender', 'source_sha256')}
                result.update(status='ok', source_path=str(path))
                try:
                    before_hash = digest(path, check)
                    result['source_sha256_observed'] = before_hash
                    if before_hash != row['source_sha256']:
                        raise ValueError('Source image hash differs from provenance')
                    if command == 'captions':
                        target = output / '.pending_train' / row['file_name']
                        # Exclusive byte copy, never a symlink or hardlink.
                        with path.open('rb') as incoming, target.open('xb') as outgoing:
                            while True:
                                check()
                                block = incoming.read(1024 * 1024)
                                if not block:
                                    break
                                outgoing.write(block)
                            outgoing.flush()
                            os.fsync(outgoing.fileno())
                        result['copied_sha256'] = digest(target, check)
                        if result['copied_sha256'] != before_hash or os.path.samefile(path, target):
                            raise ValueError('Independent image copy hash check failed')
                        meta = deepcopy(row['metadata'])
                        old, new = meta['text'], range_caption(row['age_group'], row['gender'])
                        meta['text'] = new
                        if 'caption' in meta:
                            meta['caption'] = new
                        if 'caption_sha256' in meta:
                            meta['caption_sha256'] = caption_digest(new)
                        derived.append(meta)
                        result.update(caption_before=old, caption_after=new, changed=old != new,
                                      caption_before_sha256=caption_digest(old), caption_after_sha256=caption_digest(new))
                        report['changed_captions'] += old != new
                    elif command == 'crop-preview':
                        from PIL import Image
                        with Image.open(path) as image:
                            image.load()
                            # No overrides: exactly deployment's default crop/selection.
                            edit = faces.prepare_edit(image)
                        result.update(source_to_crop=_plain(edit.matrix),
                                      selection_info=_plain(edit.selection_info),
                                      selected_person=_plain(edit.face),
                                      attribute_gender_reference=_plain(getattr(edit.face, 'gender', None)),
                                      attribute_age_reference=_plain(getattr(edit.face, 'age', None)),
                                      selected_person_verified=False,
                                      metadata_labels_are_source_labels=True,
                                      source_size=list(edit.original.size), crop_size=list(edit.crop.size))
                        relative = f'preview/{row["index"]:06d}_{row["sample_id"]}.png'
                        target = output / relative
                        with target.open('xb') as image_stream:
                            edit.crop.save(image_stream, format='PNG')
                            image_stream.flush()
                            os.fsync(image_stream.fileno())
                        result.update(preview_file=relative, preview_sha256=digest(target, check))
                    result['source_sha256_after'] = digest(path, check)
                    if result['source_sha256_after'] != before_hash:
                        raise ValueError('Source image changed during preparation')
                except (DeadlineReached, KeyboardInterrupt):
                    raise
                except Exception as exc:
                    result.update(status='error', error=f'{type(exc).__name__}: {exc}',
                                  selection_info=_plain(getattr(exc, 'selection_info', result.get('selection_info'))))
                    if command == 'crop-preview' and path.is_file():
                        result['source_sha256_after'] = digest(path, check)
                    report['failed_rows'] += 1
                    _append(stream, result)
                    report['completed_rows'] += 1
                    if command != 'crop-preview':
                        raise
                    continue
                _append(stream, result)
                report['completed_rows'] += 1
        check()
        if digest(source / 'metadata.jsonl', check) != fingerprint['metadata_sha256'] or digest(provenance, check) != fingerprint['provenance_sha256']:
            raise ValueError('Source metadata/provenance changed during preparation')
        if command == 'captions':
            staging = output / '.pending_train'
            with (staging / 'metadata.jsonl').open('x', encoding='utf-8') as stream:
                for meta in derived:
                    check()
                    stream.write(_json(meta) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            # Independently re-read all serialized rows and all source/copy hashes
            # before the only publication rename. No partial train/ is exposed.
            if _read_rows(staging / 'metadata.jsonl', check) != derived or len(derived) != len(records):
                raise ValueError('Derived metadata failed round-trip validation')
            for row, meta in zip(records, derived):
                check()
                expected = deepcopy(row['metadata'])
                expected['text'] = range_caption(row['age_group'], row['gender'])
                if 'caption' in expected:
                    expected['caption'] = expected['text']
                if 'caption_sha256' in expected:
                    expected['caption_sha256'] = caption_digest(expected['text'])
                if meta != expected or digest(staging / row['file_name'], check) != row['source_sha256'] or digest(source / row['file_name'], check) != row['source_sha256']:
                    raise ValueError('Final metadata/image integrity check failed')
            if {p.name for p in (staging / 'images').iterdir()} != {Path(r['file_name']).name for r in records}:
                raise ValueError('Derived inventory differs from source')
            if digest(source / 'metadata.jsonl', check) != fingerprint['metadata_sha256'] or digest(provenance, check) != fingerprint['provenance_sha256']:
                raise ValueError('Source metadata/provenance changed during final verification')
            report.update(derived_metadata_sha256=digest(staging / 'metadata.jsonl', check),
                          image_copy_method='independent byte copy with before/copy/after SHA256 verification')
            check()
            staging.rename(output / 'train')
            report['train_published'] = True
        elif command == 'sampling':
            plan = sampling_plan(records, power)
            plan.update(source_metadata_sha256=fingerprint['metadata_sha256'],
                        source_provenance_sha256=fingerprint['provenance_sha256'])
            _write(output / 'sampling_plan.json', plan)
            report.update(effective_sample_size=plan['effective_sample_size'], power=power,
                          sampling_plan_sha256=digest(output / 'sampling_plan.json', check))
        report['status'] = ('preview_complete_with_failures' if report['failed_rows'] else 'preview_complete') if command == 'crop-preview' else 'prepared'
    except (DeadlineReached, KeyboardInterrupt) as exc:
        report.update(status='paused_deadline' if isinstance(exc, DeadlineReached) else 'paused_interrupt',
                      error=f'{type(exc).__name__}: {exc}')
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    finally:
        _write(output / 'summary.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    for name in ('captions', 'sampling', 'crop-preview'):
        sub = subparsers.add_parser(name)
        sub.add_argument('--source', type=Path, default=Path('outputs/age_lora_stages_20261026/data/train'))
        sub.add_argument('--provenance', type=Path, default=Path('data/reports/ffhq_aging_samples.jsonl'))
        sub.add_argument('--output', type=Path, required=True)
        sub.add_argument('--deadline', required=True)
        sub.add_argument('--reserve-seconds', type=float, default=15)
        if name == 'sampling':
            sub.add_argument('--power', type=float, default=.5)
        if name == 'crop-preview':
            sub.add_argument('--limit-per-group', type=int, default=1)
            sub.add_argument('--model-root', type=Path, default=Path('models/insightface'))
            sub.add_argument('--threads', type=int, default=2)
    args = vars(parser.parse_args(argv))
    try:
        stamp = datetime.fromisoformat(args['deadline'].replace('Z', '+00:00'))
        if stamp.tzinfo is None or stamp.utcoffset().total_seconds() != 0:
            raise ValueError('deadline must include UTC timezone')
        args['deadline'] = stamp.timestamp()
        if not math.isfinite(args['reserve_seconds']) or args['reserve_seconds'] < 0:
            raise ValueError('reserve-seconds must be finite and nonnegative')
        check_deadline(args['deadline'], args['reserve_seconds'])
    except (ValueError, DeadlineReached) as exc:
        parser.error(str(exc))

    def interrupt(signum, frame):
        raise DeadlineReached('UTC deadline timer fired')

    old_handler = signal.signal(signal.SIGALRM, interrupt)
    signal.setitimer(signal.ITIMER_REAL, max(.001, args['deadline'] - args['reserve_seconds'] - time.time()))
    try:
        report = prepare(**args)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
    print(_json(report), flush=True)
    return 0 if report['status'] in ('prepared', 'preview_complete', 'preview_complete_with_failures') else 1


if __name__ == '__main__':
    sys.exit(main())
