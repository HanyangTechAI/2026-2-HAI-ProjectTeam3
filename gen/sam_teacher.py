"""Public SAM (Alaluf et al. 2021) as the age teacher for pair dataset P. Inputs must be FFHQ-aligned."""
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SAM_SIZE = 256


def to_input(image, target_age):
    """FFHQ-aligned RGB -> (4, 256, 256): image in [-1, 1] plus a target_age/100 channel (SAM AgeTransformer)."""
    if not 0 <= target_age <= 100:
        raise ValueError('target_age must be within 0..100')
    rgb = np.asarray(image.convert('RGB').resize((SAM_SIZE, SAM_SIZE), Image.BICUBIC), dtype=np.float32)
    tensor = torch.from_numpy(rgb / 127.5 - 1).permute(2, 0, 1)
    return torch.cat([tensor, torch.full((1, SAM_SIZE, SAM_SIZE), target_age / 100)])


def to_image(tensor, size=512):
    array = ((tensor.detach().float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array).resize((size, size), Image.LANCZOS)


class SamTeacher:
    def __init__(self, repo, checkpoint, device='cuda', net=None):
        self.device = device
        if net is None:
            sys.path.insert(0, str(Path(repo).resolve(strict=True)))
            from models.psp import pSp  # SAM repository module, not this project's models/ directory
            # Official SAM checkpoint stores argparse options as a plain dict; trusted local file.
            opts = torch.load(checkpoint, map_location='cpu', weights_only=False)['opts']
            opts.update(checkpoint_path=str(checkpoint))
            net = pSp(Namespace(**opts))
        self.net = net.to(device).eval()

    @torch.inference_mode()
    def age(self, images, target_ages, size=512):
        if not images or len(images) != len(target_ages):
            raise ValueError('Expected matching nonempty images and target ages')
        batch = torch.stack([to_input(image, age) for image, age in zip(images, target_ages)]).to(self.device)
        outputs = self.net(batch.float(), randomize_noise=False, resize=False)
        return [to_image(output, size) for output in outputs]
