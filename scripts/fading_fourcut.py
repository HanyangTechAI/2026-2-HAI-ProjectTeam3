"""Relative booth runner. Execute as `python -m scripts.fading_fourcut`."""
import argparse
import json
import os
import random
import sys
from pathlib import Path

from age.common import EDIT_KEYS, frame_name


SETTINGS = dict(num_ddim_steps=35, num_inner_steps=5, early_stop_epsilon=1e-5,
                self_replace_steps=0.70, cross_replace_steps=0.8, seed=0)


def generate(image, age, gender, targets, output):
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    sys.path.insert(0, str(Path.home() / '.cache/fading-src'))
    import numpy as np
    import torch
    import null_inversion
    import p2p
    from diffusers import DDIMScheduler, StableDiffusionPipeline
    from FADING_util import util
    from PIL import Image

    output.mkdir(mode=0o700, exist_ok=False)
    steps, seed = SETTINGS['num_ddim_steps'], SETTINGS['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Separate upstream globals drive the inversion trajectory and attention schedule.
    null_inversion.NUM_DDIM_STEPS = p2p.NUM_DDIM_STEPS = steps
    generator = torch.Generator(device='cuda:0')
    scheduler = DDIMScheduler(beta_start=.00085, beta_end=.012, beta_schedule='scaled_linear',
                              clip_sample=False, set_alpha_to_one=False, steps_offset=1)
    model = Path.home() / '.cache/fading-models/finetune_double_prompt_150_random'
    pipe = StableDiffusionPipeline.from_pretrained(str(model), scheduler=scheduler,
                                                  safety_checker=None).to('cuda:0')
    female = int(gender == 'female')
    person = util.get_person_placeholder(age, female)
    source_prompt = f'photo of {age} year old {person}'
    (original, _), latent, embeddings = null_inversion.NullInversion(pipe).invert(
        str(image), source_prompt, offsets=(0, 0, 0, 0), num_inner_steps=SETTINGS['num_inner_steps'],
        early_stop_epsilon=SETTINGS['early_stop_epsilon'], verbose=True)
    if len(embeddings) != steps:
        raise RuntimeError('Inversion did not match the edit schedule')
    Image.fromarray(original).save(output / 'current.png')
    for key, target in zip(EDIT_KEYS, targets, strict=True):
        new_person = util.get_person_placeholder(target, female)
        prompts = [source_prompt, f'photo of {target} year old {new_person}']
        controller = p2p.make_controller(
            prompts, True, {'default_': SETTINGS['cross_replace_steps']}, SETTINGS['self_replace_steps'],
            pipe.tokenizer, ((str(age), person), (str(target), new_person)),
            {'words': str(target), 'values': (1,)})
        images, _ = p2p.p2p_text2image(pipe, prompts, controller, generator=generator.manual_seed(seed),
                                       latent=latent, uncond_embeddings=embeddings,
                                       num_inference_steps=steps, start_time=steps)
        Image.fromarray(images[-1]).save(output / frame_name(key, target))
    (output / 'settings.json').write_text(json.dumps(dict(**SETTINGS, initial_age=age,
                                                         targets=targets, gender=gender), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--age', type=int, choices=range(1, 101), required=True)
    parser.add_argument('--gender', choices=['female', 'male'], required=True)
    parser.add_argument('--targets', type=int, choices=range(3, 91), nargs=3, required=True)
    parser.add_argument('--output', type=Path, required=True)
    generate(**vars(parser.parse_args()))
