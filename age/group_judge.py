"""FFHQ age-group classifier as a reusable judge: group probabilities, not exact ages."""
import numpy as np
import torch

from .dataset import transform
from .ffhq_data import GROUPS
from .model import build_model

TASK = 'ffhq_age_group_classification'
BOUNDS = tuple(tuple(int(v) for v in group.split('-')) for group in GROUPS)
# ponytail: open-ended 70-120 bin uses 75, not 95; recalibrate on real elderly faces if bias shows.
MIDPOINTS = np.array([(lo + hi) / 2 for lo, hi in BOUNDS[:-1]] + [75.0])


def group_of(age):
    if age < 0:
        raise ValueError('age must be nonnegative')
    return max(index for index, (lo, _) in enumerate(BOUNDS) if age >= lo)


def expected_age(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (len(GROUPS),) or not np.isclose(probabilities.sum(), 1, atol=1e-4):
        raise ValueError('Expected one probability per age group summing to 1')
    return float(probabilities @ MIDPOINTS)


class GroupJudge:
    def __init__(self, checkpoint, device='cpu'):
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if state.get('task') != TASK or list(state.get('groups', [])) != list(GROUPS):
            raise ValueError(f'{checkpoint} is not an FFHQ age-group checkpoint')
        self.model = build_model(False, len(GROUPS))
        self.model.load_state_dict(state['model'])
        self.model.to(device).eval()
        self.device = device
        self.transform = transform(False)

    @torch.inference_mode()
    def probabilities(self, images):
        batch = torch.stack([self.transform(image.convert('RGB')) for image in images]).to(self.device)
        return self.model(batch).float().softmax(-1).cpu().numpy()
