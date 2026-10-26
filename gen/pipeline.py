"""Local SD1.5 face-only inpainting + FaceID Plus v2 + optional Age LoRA.

Real loading/inference needs external weights and is not validated by mock tests.
Instances are reused sequentially, not shared across concurrent requests.
"""
import argparse
from dataclasses import dataclass
import json
import math
from numbers import Integral
from pathlib import Path
import time

from PIL import Image, ImageDraw, ImageOps

from age.common import save_json
from . import FaceNotFoundError, GenerationError, ModelUnavailableError

MODES = ('prompt_only', 'faceid', 'faceid_age')
AGE_PHRASES = (
    (2, ('baby boy', 'baby girl', 'baby')),
    (6, ('young boy', 'young girl', 'young child')),
    (9, ('boy', 'girl', 'child')),
    (14, ('preteen boy', 'preteen girl', 'preteen child')),
    (19, ('teenage boy', 'teenage girl', 'teenager')),
    (29, ('man in his 20s', 'woman in her 20s', 'person in their 20s')),
    (39, ('man in his 30s', 'woman in her 30s', 'person in their 30s')),
    (49, ('man in his 40s', 'woman in her 40s', 'person in their 40s')),
    (69, ('middle-aged man in his 50s or 60s', 'middle-aged woman in her 50s or 60s', 'middle-aged person')),
    (100, ('elderly man aged 70 or older', 'elderly woman aged 70 or older', 'elderly person aged 70 or older')),
)


def validate_targets(targets):
    targets = list(targets)
    if not targets or any(isinstance(a, bool) or not isinstance(a, Integral) or not 0 <= a <= 100 for a in targets):
        raise ValueError('targets must be a nonempty sequence of integer ages in 0..100')
    return [int(a) for a in targets]


def age_prompt(age, gender='unknown'):
    """Single caption mapping for inference and future LoRA data preparation."""
    age = validate_targets([age])[0]
    if gender not in ('male', 'female', 'unknown'):
        raise ValueError('gender must be male, female or unknown')
    column = {'male': 0, 'female': 1, 'unknown': 2}[gender]
    return 'a photo of a ' + next(phrases[column] for upper, phrases in AGE_PHRASES if age <= upper)


def as_rgb(image):
    if isinstance(image, Image.Image):
        if image.mode != 'RGB':
            raise ValueError('Expected an RGB image; use read_image for image files')
        return image
    try:
        import numpy as np
    except ImportError as e:
        raise ValueError('Use a PIL RGB image, or install numpy for uint8 HWC arrays') from e
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Expected an RGB PIL image or uint8 HWC array with 3 channels')
    return Image.fromarray(image)


def read_image(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert('RGB')


@dataclass(frozen=True)
class PipelineConfig:
    base_model: str = ''
    image_encoder: str = ''
    ip_adapter: str = ''
    faceid_lora: str = ''
    age_lora: str = ''
    model_root: str = 'models/insightface'
    device: str = 'cuda'
    seed: int = 42
    strength: float = 0.6
    steps: int = 25
    guidance_scale: float = 7.5
    ip_scale: float = 0.8
    faceid_lora_scale: float = 1.0
    age_lora_scale: float = 1.0
    preserve_background: bool = True

    def __post_init__(self):
        if self.device not in ('cpu', 'cuda'):
            raise ValueError('device must be cpu or cuda')
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**63:
            raise ValueError('seed must be an integer in 0..2**63-1')
        if not isinstance(self.steps, int) or isinstance(self.steps, bool) or self.steps < 1:
            raise ValueError('steps must be a positive integer')
        if not math.isfinite(self.strength) or not 0 < self.strength <= 1 or int(self.steps * self.strength) < 1:
            raise ValueError('strength must be in (0,1] and steps * strength must be >= 1')
        if not math.isfinite(self.guidance_scale) or self.guidance_scale <= 1:
            raise ValueError('This FaceID integration requires guidance_scale > 1 (CFG)')
        if any(not math.isfinite(v) or v < 0 for v in (self.ip_scale, self.faceid_lora_scale, self.age_lora_scale)):
            raise ValueError('adapter scales must be finite and nonnegative')
        if not isinstance(self.preserve_background, bool):
            raise ValueError('preserve_background must be a boolean')


class AgeShiftPipeline:
    is_mock = False

    def __init__(self, config, mode='faceid_age'):
        if mode not in MODES:
            raise ValueError(f'Unknown mode: {mode}')
        self.config, self.mode = config, mode
        self._pipe = self._faces = None

    def load(self):
        if self._pipe is not None:
            return self
        cfg = self.config
        required = {'base_model (local Diffusers directory)': (cfg.base_model, True)}
        if self.mode != 'prompt_only':
            required.update(image_encoder=(cfg.image_encoder, True), ip_adapter=(cfg.ip_adapter, False))
            if cfg.faceid_lora:
                required['faceid_lora'] = (cfg.faceid_lora, False)
        if self.mode == 'faceid_age':
            required['age_lora'] = (cfg.age_lora, False)
        for name, (value, directory) in required.items():
            if not value or not (Path(value).is_dir() if directory else Path(value).is_file()):
                raise ModelUnavailableError(f'Missing {name}: {value or "not configured"}. No mock fallback.')
        try:
            import torch
            from diffusers import DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
            from .faces import FaceTools

            if cfg.device == 'cuda' and not torch.cuda.is_available():
                raise ModelUnavailableError('CUDA is unavailable; configure device=cpu or explicitly use --mock')
            dtype = torch.float16 if cfg.device == 'cuda' else torch.float32
            faces = FaceTools(cfg.model_root, cfg.device, identity=self.mode != 'prompt_only')
            kwargs = dict(torch_dtype=dtype, local_files_only=True)
            if self.mode != 'prompt_only':
                from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
                kwargs['image_encoder'] = CLIPVisionModelWithProjection.from_pretrained(
                    cfg.image_encoder, torch_dtype=dtype, local_files_only=True)
                kwargs['feature_extractor'] = CLIPImageProcessor.from_pretrained(
                    cfg.image_encoder, local_files_only=True)
            # The pinned Diffusers inpaint pipeline supports the existing 4-channel
            # UNet and restores unmasked latents at each denoising step.
            pipeline_class = StableDiffusionInpaintPipeline if cfg.preserve_background else StableDiffusionImg2ImgPipeline
            pipe = pipeline_class.from_pretrained(cfg.base_model, **kwargs)
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config, algorithm_type='dpmsolver++', solver_order=2, use_karras_sigmas=True)
            if self.mode != 'prompt_only':
                adapter = Path(cfg.ip_adapter)
                pipe.load_ip_adapter(str(adapter.parent), subfolder='', weight_name=adapter.name,
                                     image_encoder_folder=None, local_files_only=True)
                pipe.set_ip_adapter_scale(cfg.ip_scale)
                # Diffusers can auto-load FaceID LoRA from bundled checkpoint keys.
                automatic = 'faceid_0' in pipe.get_list_adapters().get('unet', [])
                if cfg.faceid_lora:
                    if automatic:
                        pipe.delete_adapters('faceid_0')
                    lora = Path(cfg.faceid_lora)
                    pipe.load_lora_weights(str(lora.parent), weight_name=lora.name,
                                           adapter_name='faceid', local_files_only=True)
                    faceid_name = 'faceid'
                elif automatic:
                    faceid_name = 'faceid_0'
                else:
                    raise ModelUnavailableError('FaceID checkpoint has no bundled LoRA; configure faceid_lora')
                names, scales = [faceid_name], [cfg.faceid_lora_scale]
                if self.mode == 'faceid_age':
                    lora = Path(cfg.age_lora)
                    pipe.load_lora_weights(str(lora.parent), weight_name=lora.name,
                                           adapter_name='age', local_files_only=True)
                    names.append('age')
                    scales.append(cfg.age_lora_scale)
                pipe.set_adapters(names, adapter_weights=scales)
                projection = pipe.unet.encoder_hid_proj.image_projection_layers[0]
                projection.shortcut = True
                projection.shortcut_scale = 1.0
            self._pipe, self._faces = pipe.to(cfg.device), faces
        except ImportError as e:
            raise ModelUnavailableError('Real generation needs requirements-generation.txt; no automatic mock fallback') from e
        except (ModelUnavailableError, GenerationError):
            raise
        except FileNotFoundError as e:
            raise ModelUnavailableError(str(e)) from e
        except Exception as e:
            raise GenerationError(f'Model loading failed: {e}') from e
        return self

    def age_shift(self, image, targets, gender='unknown', seed=None):
        image, targets = as_rgb(image), validate_targets(targets)
        seed = self.config.seed if seed is None else seed
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**63:
            raise ValueError('seed must be an integer in 0..2**63-1')
        age_prompt(targets[0], gender)
        self.load()
        try:
            import torch
            cfg, pipe = self.config, self._pipe
            edit = self._faces.prepare_edit(image) if cfg.preserve_background else None
            if edit is not None:
                crop = edit.crop
                detected_gender = 'male' if edit.face.gender == 1 else 'female'
            else:
                crop, detected_gender = self._faces.prepare(image)
            gender = detected_gender if gender == 'unknown' else gender
            conditioning = {}
            if edit is not None:
                conditioning.update(mask_image=edit.mask,
                                    negative_prompt='deformed face, distorted anatomy, blurry, waxy skin')
            with torch.inference_mode():
                if self.mode != 'prompt_only':
                    embedding = torch.tensor(self._faces.embedding(edit if edit is not None else image),
                                             device=cfg.device, dtype=pipe.dtype)
                    embedding = embedding.reshape(1, 1, 1, -1)
                    conditioning['ip_adapter_image_embeds'] = [torch.cat([torch.zeros_like(embedding), embedding])]
                    clip = pipe.prepare_ip_adapter_image_embeds(
                        [crop], None, torch.device(cfg.device), 1, True)[0]
                    pipe.unet.encoder_hid_proj.image_projection_layers[0].clip_embeds = clip.to(dtype=pipe.dtype)
                images = []
                for target in targets:
                    # Recreate the same seed for each age, not a shared advancing generator.
                    result = pipe(prompt=age_prompt(target, gender), image=crop,
                                  strength=cfg.strength, num_inference_steps=cfg.steps,
                                  guidance_scale=cfg.guidance_scale,
                                  generator=torch.Generator(device=cfg.device).manual_seed(seed),
                                  **conditioning)
                    if any(getattr(result, 'nsfw_content_detected', None) or []):
                        raise GenerationError('Generation was blocked by the model safety checker')
                    if len(result.images) != 1 or not isinstance(result.images[0], Image.Image):
                        raise GenerationError('Expected one generated PIL image per target')
                    generated = result.images[0].convert('RGB')
                    images.append(self._faces.finish_edit(edit, generated) if edit is not None else generated)
                return images
        except GenerationError:
            raise
        except Exception as e:
            raise GenerationError(f'Inference failed: {e}') from e

    def close(self):
        self._pipe = self._faces = None
        # No heavyweight imports on the mock path or when loading never succeeded.
        import sys
        torch = sys.modules.get('torch')
        if torch is not None and self.config.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()


class MockPipeline:
    """Visible placeholders ONLY: no face detection, diffusion or age editing."""
    is_mock = True

    def __init__(self, mode='faceid_age', seed=42):
        if mode not in MODES:
            raise ValueError(f'Unknown mode: {mode}')
        self.mode, self.seed = mode, seed

    def load(self):
        return self

    def close(self):
        pass

    def age_shift(self, image, targets, gender='unknown', seed=None):
        image, targets = as_rgb(image), validate_targets(targets)
        outputs = []
        for age in targets:
            age_prompt(age, gender)
            result = ImageOps.pad(image, (512, 512), color='white')
            draw = ImageDraw.Draw(result)
            draw.rectangle((0, 0, 512, 64), fill='black')
            draw.text((10, 8), 'MOCK - NOT AN AGE-EDITED IMAGE', fill='white')
            draw.text((10, 28), f'{self.mode} | target={age} | seed={self.seed if seed is None else seed}', fill='white')
            outputs.append(result)
        return outputs


def add_pipeline_arguments(parser):
    parser.add_argument('--mock', action='store_true', help='Explicit placeholders; no model inference or quality metrics')
    parser.add_argument('--config', type=Path, help='JSON PipelineConfig with local model paths; required without --mock')
    parser.add_argument('--mode', choices=MODES, default='faceid_age')
    parser.add_argument('--seed', type=int, default=42)


def pipeline_from_args(args):
    if args.mock:
        return MockPipeline(args.mode, args.seed)
    if args.config is None:
        raise ValueError('Real generation requires --config. Use --mock explicitly for offline verification.')
    values = json.loads(args.config.read_text(encoding='utf-8'))
    values['seed'] = args.seed
    return AgeShiftPipeline(PipelineConfig(**values), args.mode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_pipeline_arguments(parser)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--targets', type=int, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gender', choices=['male', 'female', 'unknown'], default='unknown')
    args = parser.parse_args()
    pipeline = pipeline_from_args(args)
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        image = read_image(args.image)
        started = time.perf_counter()
        pipeline.load()
        load_seconds = time.perf_counter() - started
        started = time.perf_counter()
        images = pipeline.age_shift(image, args.targets, args.gender)
        seconds = time.perf_counter() - started
        for i, (target, result) in enumerate(zip(args.targets, images)):
            result.save(args.output / f'{i:02d}_age_{target}.png')
        save_json(args.output / 'run.json', dict(mode=args.mode, is_mock=pipeline.is_mock,
                  targets=args.targets, seed=args.seed, load_seconds=load_seconds,
                  generation_seconds=seconds, config=str(args.config) if args.config else None,
                  note='Mock placeholders are not generated age transformations.' if args.mock else 'Quality not assessed by this command.'))
        print(f'Saved {len(images)} images to {args.output}; is_mock={pipeline.is_mock}')
    finally:
        pipeline.close()


if __name__ == '__main__':
    main()
