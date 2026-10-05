import csv
import hashlib
import json
from pathlib import Path

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['path', 'age', 'gender', 'source', 'person_id', 'original_id', 'sha256']


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields or (list(rows[0]) if rows else FIELDS))
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def image_path(row):
    p = Path(row['path'])
    return p if p.is_absolute() else ROOT / p


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def target_ages(age):
    return [max(3, min(90, round(age + d))) for d in (-10, 0, 10, 20)]
