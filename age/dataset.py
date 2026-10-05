from collections import Counter
from PIL import Image
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms as T
from .common import MEAN, STD, read_csv, image_path


def transform(training=False):
    ops = [T.RandomResizedCrop(224, scale=(0.8, 1.0)), T.RandomHorizontalFlip(),
           T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)] if training else [T.Resize((224, 224))]
    return T.Compose(ops + [T.ToTensor(), T.Normalize(MEAN, STD)])


class AgeDataset(Dataset):
    def __init__(self, csv_path, training=False):
        self.rows = read_csv(csv_path)
        if not self.rows:
            raise ValueError(f'Empty split: {csv_path}. More independent people/data are required.')
        self.transform = transform(training)
        for row in self.rows:
            age = float(row['age'])
            if not age.is_integer() or not 0 <= age <= 100:
                raise ValueError(f'Invalid age: {row["age"]}')
            if not image_path(row).is_file():
                raise FileNotFoundError(image_path(row))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        with Image.open(image_path(row)) as im:
            x = self.transform(im.convert('RGB'))
        return x, torch.tensor(float(row['age'])), row['source']


def age_sampler(dataset, seed):
    bins = [min(int(r['age']) // 10, 9) for r in dataset.rows]
    counts = Counter(bins)
    return WeightedRandomSampler([1 / counts[b] for b in bins], len(bins), replacement=True,
                                 generator=torch.Generator().manual_seed(seed))


def validate_splits(datasets):
    keys = ['sha256']
    for i, a in enumerate(datasets):
        for b in datasets[i + 1:]:
            if {str(image_path(r).resolve()) for r in a.rows} & {str(image_path(r).resolve()) for r in b.rows}:
                raise ValueError('Split leakage: path')
            for key in keys:
                left = {r.get(key) for r in a.rows if r.get(key)}
                right = {r.get(key) for r in b.rows if r.get(key)}
                if left & right:
                    raise ValueError(f'Split leakage: {key}')
            left = {(r['source'], r['person_id']) for r in a.rows if r.get('person_id')}
            right = {(r['source'], r['person_id']) for r in b.rows if r.get('person_id')}
            if left & right:
                raise ValueError('Split leakage: person_id')
    for ds in datasets:
        if any(r['source'] == 'aihub' and not r.get('person_id') for r in ds.rows):
            raise ValueError('AI Hub requires person_id')
