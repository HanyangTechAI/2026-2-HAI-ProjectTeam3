import csv
import hashlib
import json
import math
from dataclasses import dataclass
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


@dataclass(frozen=True)
class BoothSlot:
    key: str
    age: int
    label: str
    filename: str


EDIT_KEYS = ('past', 'future10', 'future20')


def frame_name(key, age):
    return f'{key}_{age}.png'


def booth_slots(prediction):
    """Round once (half up). Bound only the edits to 3..90, not the photo."""
    if not math.isfinite(prediction) or not 0 <= prediction <= 100:
        raise ValueError('Invalid age estimate')
    initial = max(1, math.floor(prediction + 0.5))  # FADING accepts 1..100
    past, _, future10, future20 = target_ages(initial)
    slots = []
    for key, age, delta, name in zip(EDIT_KEYS, (past, future10, future20),
                                     (-10, 10, 20), ('Past (-10y)', 'Future (+10y)', 'Future (+20y)')):
        limit = ' (limited)' if age != initial + delta else ''
        slots.append(BoothSlot(key, age, f'{name}: age {age}{limit}', frame_name(key, age)))
    slots.insert(1, BoothSlot('current', initial, f'Current photo: est. {initial}', 'current.png'))
    return tuple(slots)
