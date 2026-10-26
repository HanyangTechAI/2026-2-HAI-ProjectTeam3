"""Read-only CPU leakage-candidate screening, not person identification.

Test is read only for leakage screening. Fixed validation selection depends on
TRAIN similarities, face availability and original metadata, never test matches.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
from itertools import combinations
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import statistics
import time
from unittest.mock import patch

from PIL import Image, ImageDraw, ImageOps

from age.common import save_json, sha256, write_csv
from . import FaceNotFoundError

SPLITS = ('train', 'val', 'test')
AGE_GROUPS = ('0-2', '3-6', '7-9', '10-14', '15-19', '20-29', '30-39',
              '40-49', '50-69', '70-120')
GENDERS = ('male', 'female')
TARGETS = [8, 25, 45, 65, 80]
# These are the persisted preparation captions, independent of future prompts.
ORIGINAL_PHRASES = (
    ('baby boy', 'baby girl'), ('young boy', 'young girl'), ('boy', 'girl'),
    ('preteen boy', 'preteen girl'), ('teenage boy', 'teenage girl'),
    ('man in his 20s', 'woman in her 20s'), ('man in his 30s', 'woman in her 30s'),
    ('man in his 40s', 'woman in her 40s'),
    ('middle-aged man in his 50s or 60s', 'middle-aged woman in her 50s or 60s'),
    ('elderly man in his 70s', 'elderly woman in her 70s'),
)
ORIGINAL_CAPTIONS = {(age, gender): 'a photo of a ' + ORIGINAL_PHRASES[i][j]
                     for i, age in enumerate(AGE_GROUPS) for j, gender in enumerate(GENDERS)}
_faces = None
_fallback_size = 0
CANDIDATE_FIELDS = [f'{field}_{side}' for side in ('a', 'b')
                    for field in ('sample_id', 'split', 'path', 'age_group', 'gender', 'sha256', 'detector_size')]
CANDIDATE_FIELDS += ['cosine', 'threshold', 'relation', 'verified_identity', 'affects_fixed_val_selection']
NEIGHBOR_FIELDS = ['sample_id', 'split', 'neighbor_id', 'neighbor_split', 'cosine', 'above_threshold']


def load_records(root, provenance, project_root):
    """Join original metadata to provenance and verify every image SHA-256."""
    root, project_root = Path(root).resolve(), Path(project_root).resolve()
    sources = {}
    with Path(provenance).open(encoding='utf-8') as stream:
        for line in stream:
            source = json.loads(line)
            path = str((project_root / source['path']).resolve())
            if path in sources:
                raise ValueError(f'Duplicate provenance: {path}')
            sources[path] = source
    records, ids, hashes = [], set(), set()
    for split in SPLITS:
        directory = root / split
        referenced = set()
        with (directory / 'metadata.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                meta = json.loads(line)
                name = Path(meta['file_name'])
                path = (directory / name).resolve()
                if name.is_absolute() or not path.is_relative_to(directory / 'images'):
                    raise ValueError(f'Path outside original split: {path}')
                if path in referenced:
                    raise ValueError(f'Duplicate metadata reference: {path}')
                referenced.add(path)
                source = sources.pop(str(path))
                age, gender = source['age_group'], source['gender']
                digest = sha256(path)
                if (source['split'] != split or source['image_number'] != path.stem
                        or source['caption'] != meta['text'] or source['sha256'] != digest
                        or ORIGINAL_CAPTIONS.get((age, gender)) != meta['text']):
                    raise ValueError(f'Original metadata/provenance mismatch: {path}')
                if not (float(source['age_group_confidence']) > .8
                        and abs(float(source['head_yaw'])) < 30):
                    raise ValueError(f'Provenance selection thresholds failed: {path}')
                if not all(source['attribution'].get(key) for key in
                           ('author', 'photo_url', 'license', 'license_url')):
                    raise ValueError(f'Incomplete attribution: {path}')
                if path.stem in ids or digest in hashes:
                    raise ValueError(f'Duplicate image ID or SHA-256: {path}')
                ids.add(path.stem)
                hashes.add(digest)
                records.append(dict(sample_id=f'ffhq_{path.stem}', path=str(path), split=split,
                                    age_group=age, gender=gender, sha256=digest, caption=meta['text']))
        if referenced != {p.resolve() for p in (directory / 'images').rglob('*') if p.is_file()}:
            raise ValueError(f'Unreferenced or missing images in {split}')
    if sources:
        raise ValueError('Provenance contains rows not in original splits')
    return sorted(records, key=lambda row: (SPLITS.index(row['split']), row['sample_id']))


def group_counts(records):
    counts = Counter((r['split'], r['age_group'], r['gender']) for r in records)
    return [dict(split=split, age_group=age, gender=gender, count=counts[split, age, gender])
            for split in SPLITS for age in AGE_GROUPS for gender in GENDERS]


def stable_key(row, seed=42):
    payload = f"{seed}:{row['sample_id']}:{row['sha256']}".encode('utf-8')
    return hashlib.sha256(payload).hexdigest(), row['sample_id']


def init_worker(model_root, threads, fallback_size=0):
    """Bound all repository FaceTools sessions without changing that API/code."""
    global _faces, _fallback_size
    _fallback_size = fallback_size
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = str(threads)
    import cv2
    import onnxruntime as ort
    from .faces import FaceTools
    cv2.setNumThreads(1)
    session_factory = ort.InferenceSession

    def cpu_session(path, sess_options=None, **kwargs):
        options = sess_options or ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.log_severity_level = 3  # Dynamic 320px outputs have stale static ONNX shape hints.
        kwargs['providers'] = ['CPUExecutionProvider']
        return session_factory(path, sess_options=options, **kwargs)

    with patch.object(ort, 'InferenceSession', cpu_session):
        _faces = FaceTools(model_root, device='cpu', identity=True)
    for model in (_faces.aligner.detector, _faces.aligner.attribute, _faces.recognizer):
        if model.session.get_providers() != ['CPUExecutionProvider']:
            raise RuntimeError('Audit must run on CPU only')
        if model.session.get_session_options().intra_op_num_threads != threads:
            raise RuntimeError('CPU thread limit was not applied')


def embed_record(record):
    started = time.perf_counter()
    result = {key: record[key] for key in ('sample_id', 'path', 'sha256')}
    result.update(default_status='error', detector_size=None)
    try:
        if sha256(record['path']) != record['sha256']:
            raise ValueError('Image changed after provenance verification')
        with Image.open(record['path']) as image:
            image.load()
            if image.mode != 'RGB' or image.size != (512, 512):
                raise ValueError('Expected original 512x512 RGB image')
            try:
                embedding = _faces.embedding(image)
                result.update(default_status='ok', detector_size=640)
            except FaceNotFoundError:
                result['default_status'] = 'face_not_detected'
                if not _fallback_size:
                    raise
                # Audit-only second scale. The fixed val set still requires default success.
                _faces.aligner.detector.input_size = (_fallback_size, _fallback_size)
                try:
                    embedding = _faces.embedding(image)
                    result['detector_size'] = _fallback_size
                finally:
                    _faces.aligner.detector.input_size = (640, 640)
        result.update(status='ok', embedding=embedding, error=None)
    except FaceNotFoundError as exc:
        result.update(status='face_not_detected', embedding=None, error=str(exc))
    except Exception as exc:
        result.update(status='error', embedding=None, error=f'{type(exc).__name__}: {exc}')
    result['seconds'] = time.perf_counter() - started
    return result


def benchmark(records, model_root, threads, output, fallback_size=0):
    selected = []
    for age in AGE_GROUPS:
        for gender in GENDERS:
            pool = [r for r in records if (r['split'], r['age_group'], r['gender'])
                    == ('train', age, gender)]
            selected.append(min(pool, key=stable_key))
    started = time.perf_counter()
    init_worker(model_root, threads, fallback_size)
    loaded = time.perf_counter()
    results = [embed_record(row) for row in selected]
    elapsed = time.perf_counter() - loaded
    report = dict(images=len(results), model_load_seconds=loaded - started,
                  image_seconds=elapsed, mean_seconds_per_image=elapsed / len(results),
                  median_seconds_per_image=statistics.median(r['seconds'] for r in results),
                  estimated_serial_scan_minutes=elapsed / len(results) * len(records) / 60,
                  total_images=len(records), cpu_threads=threads, workers=1,
                  providers=['CPUExecutionProvider'],
                  fallback_size=fallback_size,
                  statuses=dict(Counter(r['status'] for r in results)),
                  samples=[{k: v for k, v in r.items() if k != 'embedding'} for r in results])
    save_json(output / ('benchmark_fallback.json' if fallback_size else 'benchmark.json'), report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def cache_context(model_root, fallback_size):
    root = Path(model_root).resolve()
    files = [root / 'models/buffalo_l' / name
             for name in ('det_10g.onnx', 'genderage.onnx', 'w600k_r50.onnx')]
    files += sorted((root / 'python_runtime').rglob('*.py'))
    files += [Path(__file__).with_name('faces.py'), Path(__file__).parents[1] / 'age/face.py']
    return dict(embedding_contract_version=1, providers=['CPUExecutionProvider'],
                detector_size=640, detector_threshold=.5, fallback_size=fallback_size,
                face_selection='largest detected face, FaceTools.embedding, training=False',
                files={str(p.resolve()): sha256(p) for p in files},
                packages={name: version(name) for name in
                          ('numpy', 'onnxruntime', 'Pillow', 'opencv-python-headless', 'scikit-image')})


def scan(records, model_root, output, workers, threads, fallback_size):
    context = cache_context(model_root, fallback_size)
    context_path = output / 'cache_context.json'
    if context_path.exists():
        if json.loads(context_path.read_text()) != context:
            raise ValueError('Cache model/runtime context changed; use a new output directory')
    else:
        save_json(context_path, context)
    cache_file = output / 'embedding_cache.jsonl'
    cached, inventory = {}, {r['sample_id']: r for r in records}
    if cache_file.exists():
        with cache_file.open(encoding='utf-8') as stream:
            for line in stream:
                result = json.loads(line)
                source = inventory[result['sample_id']]
                if any(result[key] != source[key] for key in ('path', 'sha256')):
                    raise ValueError('Cached image provenance changed')
                if result['sample_id'] in cached:
                    raise ValueError('Duplicate cached image')
                cached[result['sample_id']] = result
    pending = [row for row in records if row['sample_id'] not in cached]
    started, reused = time.perf_counter(), len(cached)
    print(f'CPU scan: {len(pending)} pending; {reused} cached; {workers} workers x {threads} threads', flush=True)
    if pending:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=init_worker,
                                 initargs=(str(model_root), threads, fallback_size)) as pool, \
                cache_file.open('a', encoding='utf-8') as stream:
            for index, result in enumerate(pool.map(embed_record, pending, chunksize=8), 1):
                cached[result['sample_id']] = result
                stream.write(json.dumps(result, separators=(',', ':')) + '\n')
                if index % 100 == 0 or index == len(pending):
                    stream.flush()
                    elapsed = time.perf_counter() - started
                    progress = dict(completed=len(cached), total=len(records), reused=reused,
                                    elapsed_seconds=elapsed,
                                    eta_minutes=(len(pending) - index) * elapsed / index / 60,
                                    statuses=dict(Counter(r['status'] for r in cached.values())),
                                    default_statuses=dict(Counter(r['default_status'] for r in cached.values())))
                    save_json(output / 'scan_progress.json', progress)
                    print(json.dumps(progress), flush=True)
    save_json(output / ('scan_run.json' if pending else 'cache_reuse_run.json'),
              dict(seconds=time.perf_counter() - started, reused=reused, newly_processed=len(pending),
                   workers=workers, threads=threads, fallback_size=fallback_size))
    return cached


def cross_split_candidates(records, embeddings, threshold=.5):
    """All cross-split cosine hits are candidates, not verified identities."""
    import numpy as np
    if not np.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError('Cosine threshold must be finite and in [-1, 1]')
    groups, matrices = {}, {}
    for split in SPLITS:
        group = sorted((r for r in records if r['split'] == split
                        and embeddings[r['sample_id']]['status'] == 'ok'), key=lambda r: r['sample_id'])
        groups[split] = group
        matrix = np.asarray([embeddings[r['sample_id']]['embedding'] for r in group], dtype=np.float32)
        if group:
            if matrix.shape != (len(group), 512) or not np.isfinite(matrix).all():
                raise ValueError('Invalid cached ArcFace embedding')
            norms = np.linalg.norm(matrix, axis=1)
            if not (norms > 0).all():
                raise ValueError('Invalid cached ArcFace embedding')
            matrices[split] = matrix / norms[:, None]
    candidates, neighbors, summaries = [], [], {}
    for split_a, split_b in combinations(SPLITS, 2):
        left, right = groups[split_a], groups[split_b]
        stats = dict(images_a=len(left), images_b=len(right), comparisons=len(left) * len(right),
                     candidate_pairs=0, candidate_images_a=0, candidate_images_b=0,
                     min_candidate_cosine=None, max_candidate_cosine=None)
        summaries[f'{split_a}-{split_b}'] = stats
        if not left or not right:
            continue
        scores = np.clip(matrices[split_a] @ matrices[split_b].T, -1., 1.)
        for a, b, similarities in ((left, right, scores), (right, left, scores.T)):
            for i, j in enumerate(np.argmax(similarities, axis=1)):
                score = float(similarities[i, j])
                neighbors.append(dict(sample_id=a[i]['sample_id'], split=a[i]['split'],
                                      neighbor_id=b[j]['sample_id'], neighbor_split=b[j]['split'],
                                      cosine=score, above_threshold=score >= threshold))
        hits = np.argwhere(scores >= threshold)
        stats.update(candidate_pairs=len(hits), candidate_images_a=len(set(hits[:, 0])),
                     candidate_images_b=len(set(hits[:, 1])))
        if len(hits):
            values = scores[hits[:, 0], hits[:, 1]]
            stats.update(min_candidate_cosine=float(values.min()), max_candidate_cosine=float(values.max()))
        for i, j in hits:
            candidate = {}
            for side, row in (('a', left[i]), ('b', right[j])):
                candidate.update({f'{key}_{side}': row[key] for key in
                                  ('sample_id', 'split', 'path', 'age_group', 'gender', 'sha256')})
                candidate[f'detector_size_{side}'] = embeddings[row['sample_id']]['detector_size']
            candidate.update(cosine=float(scores[i, j]), threshold=threshold, relation='candidate',
                             verified_identity=False,
                             affects_fixed_val_selection={split_a, split_b} == {'train', 'val'})
            candidates.append(candidate)
    candidates.sort(key=lambda row: (-row['cosine'], row['sample_id_a'], row['sample_id_b']))
    return candidates, neighbors, summaries


def select_validation(records, embeddings, candidates, threshold=.5, seed=42):
    """One original val sample per cell; TEST pairs cannot change eligibility."""
    blocked = set()
    for pair in candidates:
        if {pair['split_a'], pair['split_b']} == {'train', 'val'} and pair['cosine'] >= threshold:
            blocked.add(pair['sample_id_a'] if pair['split_a'] == 'val' else pair['sample_id_b'])
    selected, cells = [], []
    for age in AGE_GROUPS:
        for gender in GENDERS:
            pool = [r for r in records if (r['split'], r['age_group'], r['gender']) == ('val', age, gender)]
            detectable = [r for r in pool if embeddings[r['sample_id']]['status'] == 'ok'
                          and embeddings[r['sample_id']]['default_status'] == 'ok']
            eligible = [r for r in detectable if r['sample_id'] not in blocked]
            cells.append(dict(age_group=age, gender=gender, original_val=len(pool),
                              default_detectable=len(detectable), train_candidates=sum(
                                  r['sample_id'] in blocked for r in pool), eligible=len(eligible)))
            if not eligible:
                raise ValueError(f'No default-face-detectable, train-candidate-free validation sample: {age}/{gender}')
            selected.append(min(eligible, key=lambda row: stable_key(row, seed)))
    return selected, cells


def save_manifest(selected, output, threshold=.5, seed=42):
    output = Path(output).resolve()
    inputs = output / 'inputs'
    inputs.mkdir(exist_ok=True)
    samples = []
    for row in selected:
        original = Path(row['path'])
        if row['split'] != 'val' or not original.is_absolute() or original.parent.name != 'images' \
                or original.parent.parent.name != 'val' or sha256(original) != row['sha256']:
            raise ValueError('Fixed input provenance must match an original val image')
        copied = inputs / f"{row['sample_id']}{original.suffix}"
        if copied.exists():
            if sha256(copied) != row['sha256']:
                raise ValueError('Refusing to overwrite changed fixed input copy')
        else:
            shutil.copy2(original, copied)
        if sha256(copied) != row['sha256']:
            raise ValueError('Copied input hash mismatch')
        sample = {key: row[key] for key in ('sample_id', 'path', 'age_group', 'gender', 'sha256')}
        sample['input_path'] = str(copied)
        samples.append(sample)
    manifest = dict(source_split='val', seed=seed, targets=TARGETS, samples=samples,
                    selection=dict(method='minimum SHA-256(seed:sample_id:image_sha256) per original age/gender cell',
                                   required_detector_size=640, train_candidate_threshold=threshold,
                                   threshold_calibrated=False, test_used_for_selection=False,
                                   candidates_are_verified_identities=False))
    manifest_path = output / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Refusing to change an existing fixed manifest')
    save_json(manifest_path, manifest)
    return manifest


def candidate_sheets(candidates, output):
    """Small local review sheets, with image IDs only and explicit candidate labels."""
    paths = []
    for split_a, split_b in combinations(SPLITS, 2):
        top = [r for r in candidates if (r['split_a'], r['split_b']) == (split_a, split_b)][:8]
        if not top:
            continue
        sheet = Image.new('RGB', (768, 48 + 216 * ((len(top) + 1) // 2)), 'white')
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 8), f'{split_a}/{split_b}: SIMILARITY CANDIDATES, NOT VERIFIED IDENTITIES', fill='black')
        draw.text((8, 24), 'Exploratory threshold; no identification or age/gender inference.', fill='black')
        for index, pair in enumerate(top):
            x, y = (index % 2) * 384, 48 + (index // 2) * 216
            for side_index, side in enumerate(('a', 'b')):
                with Image.open(pair[f'path_{side}']) as image:
                    tile = ImageOps.contain(image.convert('RGB'), (184, 168))
                    sheet.paste(tile, (x + side_index * 192 + (184 - tile.width) // 2, y))
                draw.text((x + side_index * 192 + 4, y + 170),
                          f"{pair[f'split_{side}']} {pair[f'sample_id_{side}']}", fill='black')
            draw.text((x + 4, y + 188), f"candidate cosine={pair['cosine']:.4f}", fill='black')
        path = output / f'candidates_{split_a}_{split_b}.jpg'
        sheet.save(path, quality=88)
        paths.append(str(path.resolve()))
    return paths


def write_report(records, embeddings, output, threshold, source_hashes):
    started = time.perf_counter()
    candidates, neighbors, pairs = cross_split_candidates(records, embeddings, threshold)
    write_csv(output / 'candidates.csv', candidates, CANDIDATE_FIELDS)
    write_csv(output / 'nearest_cross_split.csv', neighbors, NEIGHBOR_FIELDS)
    inventory, split_stats, face_groups = [], {}, []
    for row in records:
        result = embeddings[row['sample_id']]
        inventory.append(row | {key: result[key] for key in
                                ('status', 'default_status', 'detector_size', 'error', 'seconds')})
    for split in SPLITS:
        rows = [r for r in inventory if r['split'] == split]
        split_stats[split] = dict(images=len(rows), final_statuses=dict(Counter(r['status'] for r in rows)),
                                 default_statuses=dict(Counter(r['default_status'] for r in rows)),
                                 fallback_recovered=sum(r['detector_size'] == 320 for r in rows))
        for age in AGE_GROUPS:
            for gender in GENDERS:
                cell = [r for r in rows if (r['age_group'], r['gender']) == (age, gender)]
                face_groups.append(dict(split=split, age_group=age, gender=gender, images=len(cell),
                                        default_ok=sum(r['default_status'] == 'ok' for r in cell),
                                        fallback_ok=sum(r['detector_size'] == 320 for r in cell),
                                        failed=sum(r['status'] != 'ok' for r in cell)))
    write_csv(output / 'image_audit.csv', inventory)
    write_csv(output / 'face_groups.csv', face_groups)
    write_csv(output / 'face_failures.csv', [r for r in inventory if r['status'] != 'ok'], list(inventory[0]))
    write_csv(output / 'default_face_failures.csv',
              [r for r in inventory if r['default_status'] != 'ok'], list(inventory[0]))
    selection_error, manifest, cells, fixed_screening = None, None, [], []
    try:
        selected, cells = select_validation(records, embeddings, candidates, threshold)
        selected_ids = {r['sample_id'] for r in selected}
        fixed_screening = [r for r in neighbors if r['sample_id'] in selected_ids and r['neighbor_split'] == 'train']
        if len(fixed_screening) != len(selected) or any(r['above_threshold'] for r in fixed_screening):
            raise ValueError('Every fixed sample must have a measured train maximum below the threshold')
        manifest = save_manifest(selected, output, threshold)
        write_csv(output / 'selection_cells.csv', cells)
        write_csv(output / 'fixed_val_train_screening.csv', fixed_screening, NEIGHBOR_FIELDS)
    except ValueError as exc:
        selection_error = str(exc)
    labels = []
    for age, issue in (('70-120', 'Original 70-120 range is narrowed incorrectly to 70s in persisted captions.'),
                       ('50-69', 'Pooled 50-69 label: no independent 50s/60s labels or exact ages.')):
        rows = [r for r in records if r['age_group'] == age]
        labels.append(dict(age_group=age, interpretation=issue,
                           split_counts=dict(Counter(r['split'] for r in rows)),
                           caption_counts=dict(Counter(r['caption'] for r in rows))))
    caveats = [
        'All cosine hits are unverified candidates, not identified persons or confirmed leakage.',
        f'The exploratory cosine threshold {threshold:g} is uncalibrated; false positives/negatives are unknown.',
        'No person_id ground truth exists; even complete embedding coverage cannot prove split independence.',
        'Default detector640 failures are retained; audit-only detector320 fallback changes alignment and scores.',
        'This audit uses the legacy largest-face aligner, not the generation primary-face policy; it is not an all-person inventory.',
        'Fixed val uses default640 successes and TRAIN candidate exclusions only; this introduces selection bias.',
        'Test was read only for leakage screening, never training, checkpoint selection, or quality evaluation.',
        'Metadata gender is an original label, not inferred gender or self-described gender identity.',
        'Original group ages are not exact chronological ages; 20 fixed samples cannot establish fairness or generalization.',
        'Original images/splits/captions are unchanged. No automatic train exclusions or corrected labels are produced here.',
        'Data and pretrained InsightFace weights remain subject to their noncommercial research and attribution terms.',
    ]
    changed = [path for path, digest in source_hashes.items() if sha256(path) != digest]
    if changed:
        raise ValueError(f'Original metadata changed during audit: {changed}')
    report = dict(status='complete' if manifest and not selection_error
                  and not any(r['status'] == 'error' for r in inventory) else 'incomplete',
                  completed_utc=datetime.now(timezone.utc).isoformat(), source_hashes=source_hashes,
                  provenance_validated=True, image_hashes_verified=len(records), source_split_counts={
                      split: len([r for r in records if r['split'] == split]) for split in SPLITS},
                  group_counts=group_counts(records), label_audit=labels, face_detection=split_stats,
                  candidate_threshold=threshold, threshold_calibrated=False,
                  identity_disjoint_verified=False, candidate_pairs=pairs, total_candidate_pairs=len(candidates),
                  test_usage='leakage_screening_only', test_used_for_selection=False,
                  selection_error=selection_error, selected_samples=len(manifest['samples']) if manifest else 0,
                  manifest=str((output / 'manifest.json').resolve()) if manifest else None,
                  selection_cells=cells, sheets=candidate_sheets(candidates, output),
                  fixed_val_max_train_cosine=max((r['cosine'] for r in fixed_screening), default=None),
                  cache_context=str((output / 'cache_context.json').resolve()),
                  cache_context_sha256=sha256(output / 'cache_context.json'),
                  original_mutations=[], report_seconds=time.perf_counter() - started, caveats=caveats)
    save_json(output / 'report.json', report)
    text = ['# Age LoRA CPU Preflight', '', f"Status: {report['status']}", '',
            'Similarity candidates only. No verified identity, confirmed leakage, or quality metric is claimed.', '',
            '## Original Counts', '', '| Age Group | Gender | Train | Val | Test |', '|---|---|---:|---:|---:|']
    counts = {(r['split'], r['age_group'], r['gender']): r['count'] for r in report['group_counts']}
    for age in AGE_GROUPS:
        for gender in GENDERS:
            text.append(f'| {age} | {gender} | ' + ' | '.join(str(counts[s, age, gender]) for s in SPLITS) + ' |')
    text += ['', '## Label Audit', ''] + [f"- {r['interpretation']} Counts: {r['split_counts']}" for r in labels]
    text += ['', '## Detection And Candidates', '', '```json',
             json.dumps(dict(face_detection=split_stats, pairs=pairs), indent=2), '```', '',
             '## Fixed Validation', '', f"Manifest: {report['manifest']}; selected: {report['selected_samples']}",
             'One sample per original age group and metadata gender; seed42 hash ordering.',
             'Only default640 success and no train candidate >= threshold are required. Test matches never gate selection.',
             f'Selection error: {selection_error}', '', '## Caveats', ''] + ['- ' + c for c in caveats]
    (output / 'report.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('status', 'face_detection', 'candidate_pairs',
                                                 'selected_samples', 'selection_error', 'manifest')}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/lora'))
    parser.add_argument('--provenance', type=Path, default=Path('data/reports/ffhq_aging_samples.jsonl'))
    parser.add_argument('--project-root', type=Path, default=Path('.'))
    parser.add_argument('--model-root', type=Path, default=Path('models/insightface'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--benchmark-only', action='store_true')
    parser.add_argument('--scan-only', action='store_true')
    parser.add_argument('--fallback-size', type=int, choices=(0, 320), default=320)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--threshold', type=float, default=.5)
    args = parser.parse_args()
    if not 1 <= args.threads <= 4 or not 1 <= args.workers <= 4 or args.threads * args.workers > 8:
        parser.error('Use 1..4 threads/workers and at most 8 total inference threads')
    args.output.mkdir(parents=True, exist_ok=True)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = str(args.threads)
    source_hashes = {str(path.resolve()): sha256(path) for path in
                     [args.root / split / 'metadata.jsonl' for split in SPLITS] + [args.provenance]}
    records = load_records(args.root, args.provenance, args.project_root)
    write_csv(args.output / 'groups.csv', group_counts(records))
    print(f'Verified original metadata/provenance/hashes for {len(records)} images', flush=True)
    if args.benchmark_only:
        benchmark(records, args.model_root, args.threads, args.output, args.fallback_size)
    else:
        embeddings = scan(records, args.model_root, args.output, args.workers, args.threads, args.fallback_size)
        if not args.scan_only:
            report = write_report(records, embeddings, args.output, args.threshold, source_hashes)
            raise SystemExit(0 if report['status'] == 'complete' else 1)


if __name__ == '__main__':
    main()
