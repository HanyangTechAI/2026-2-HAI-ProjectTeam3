import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

from PIL import Image

from gen import FaceNotFoundError, GenerationError, ModelUnavailableError
from gen.pipeline import (AgeShiftPipeline, MockPipeline, PipelineConfig, age_prompt,
                          as_rgb, read_image, add_pipeline_arguments, pipeline_from_args)


@contextmanager
def fake_runtime(automatic=True):
    """Provide test doubles for model loading and inference calls."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for name in ('base', 'clip'):
            (root / name).mkdir()
        for name in ('ip.bin', 'faceid.safetensors', 'age.safetensors'):
            (root / name).touch()
        config = PipelineConfig(base_model=str(root / 'base'), image_encoder=str(root / 'clip'),
                                ip_adapter=str(root / 'ip.bin'), age_lora=str(root / 'age.safetensors'), device='cpu')
        pipe = Mock()
        pipe.dtype = 'float32'
        pipe.scheduler = SimpleNamespace(config={})
        projection = SimpleNamespace(shortcut=False, shortcut_scale=0)
        pipe.unet.encoder_hid_proj.image_projection_layers = [projection]
        pipe.get_list_adapters.return_value = {'unet': ['faceid_0'] if automatic else []}
        def load_ip_adapter(path, subfolder, weight_name, **kwargs):
            pass
        pipe.load_ip_adapter.side_effect = load_ip_adapter
        pipe.to.return_value = pipe
        pipe.return_value = SimpleNamespace(images=[Image.new('RGB', (512, 512), 'blue')], nsfw_content_detected=None)
        faces = Mock()
        faces.prepare.return_value = (Image.new('RGB', (512, 512)), 'female')
        faces.prepare_edit.side_effect = lambda image, **kwargs: SimpleNamespace(
            original=image, crop=Image.new('RGB', (512, 512)),
            mask=Image.new('L', (512, 512), 255), generation_mask=None, face=SimpleNamespace(gender=0))
        faces.finish_edit.side_effect = lambda edit, generated: generated.resize(edit.original.size)
        faces.embedding.return_value = [1.0] + [0.0] * 511
        torch = ModuleType('torch')
        torch.float16, torch.float32 = 'float16', 'float32'
        torch.cuda = Mock()
        torch.cuda.is_available.return_value = False
        tensor = Mock()
        tensor.reshape.return_value = tensor
        pipe.prepare_ip_adapter_image_embeds.return_value = [tensor]
        torch.tensor = Mock(return_value=tensor)
        torch.cat = Mock(return_value=tensor)
        torch.zeros_like = Mock(return_value=tensor)
        torch.device = lambda value: value
        torch.inference_mode = nullcontext
        torch.Generator = Mock()
        diffusers = ModuleType('diffusers')
        diffusers.StableDiffusionImg2ImgPipeline = Mock()
        diffusers.StableDiffusionImg2ImgPipeline.from_pretrained.return_value = pipe
        diffusers.StableDiffusionInpaintPipeline = Mock()
        diffusers.StableDiffusionInpaintPipeline.from_pretrained.return_value = pipe
        diffusers.DPMSolverMultistepScheduler = Mock()
        transformers = ModuleType('transformers')
        transformers.CLIPVisionModelWithProjection = Mock()
        transformers.CLIPImageProcessor = Mock()
        face_module = ModuleType('gen.faces')
        face_module.FaceTools = Mock(return_value=faces)
        with patch.dict('sys.modules', {'torch': torch, 'diffusers': diffusers,
                                      'transformers': transformers, 'gen.faces': face_module}):
            yield SimpleNamespace(config=config, pipe=pipe, faces=faces, torch=torch,
                                  diffusers=diffusers, transformers=transformers, face_module=face_module,
                                  projection=projection, tensor=tensor, root=root)


class PipelineContracts(unittest.TestCase):
    def test_caption_boundaries_and_gender(self):
        for age in (0, 2, 3, 6, 7, 9, 10, 14, 15, 19, 20, 29, 30, 39, 40, 49, 50, 69, 70, 100):
            for gender in ('male', 'female', 'unknown'):
                self.assertTrue(age_prompt(age, gender).startswith('a photo of a '))
        self.assertEqual(age_prompt(29, 'female'), 'a photo of a woman in her 20s')
        self.assertEqual(age_prompt(30, 'male'), 'a photo of a man in his 30s')
        self.assertEqual(age_prompt(80, 'female'), 'a photo of a elderly woman aged 70 or older')
        self.assertEqual(age_prompt(50, 'male'), age_prompt(69, 'male'))
        for age in (-1, 101, 25.5, True):
            with self.assertRaises(ValueError):
                age_prompt(age)
        with self.assertRaises(ValueError):
            age_prompt(25, 'invalid')

    def test_mock_is_visible_deterministic_and_keeps_input(self):
        image = Image.new('RGB', (72, 96), 'red')
        before = image.tobytes()
        pipeline = MockPipeline()
        results = pipeline.age_shift(image, [90, 90, 90])
        self.assertEqual(len(results), 3)
        self.assertTrue(pipeline.is_mock)
        self.assertTrue(all(im.mode == 'RGB' and im.size == (512, 512) for im in results))
        self.assertEqual(results[0].getpixel((0, 0)), (0, 0, 0))
        self.assertEqual(results[0].tobytes(), results[1].tobytes())
        self.assertEqual(image.tobytes(), before)
        with self.assertRaises(ValueError):
            pipeline.age_shift(image, [])
        with self.assertRaises(ValueError):
            as_rgb(Image.new('L', (16, 16)))

    def test_config_validation(self):
        for values in ({'steps': 0}, {'strength': float('nan')}, {'steps': 1, 'strength': .1},
                       {'guidance_scale': 1}, {'age_lora_scale': -1}, {'device': 'magic'}, {'seed': -1},
                       {'preserve_background': 'true'}, {'masked_denoising': 1},
                       {'prompt_style': 'exact'}, {'generation_mask_mode': 'all'}):
            with self.assertRaises(ValueError):
                PipelineConfig(**values)
        for name, invalid in (('shortcut_scale', [-1, float('inf'), float('nan'), True, '1', None]),
                              ('mask_scale', [0, .99, 1.26, float('nan'), True, '1', None]),
                              ('feature_protection', [-.01, 1.01, float('inf'), True, '1', None])):
            for value in invalid:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    PipelineConfig(**{name: value})
        config = PipelineConfig()
        self.assertEqual((config.shortcut_scale, config.prompt_style, config.masked_denoising,
                          config.generation_mask_mode, config.mask_scale, config.feature_protection),
                         (1.0, 'group', True, 'protected', 1.0, 1.0))
        PipelineConfig(shortcut_scale=0, mask_scale=1.25, feature_protection=0)

    def test_missing_real_assets_never_fall_back_to_mock(self):
        pipeline = AgeShiftPipeline(PipelineConfig())
        self.assertFalse(pipeline.is_mock)
        with self.assertRaises(ModelUnavailableError):
            pipeline.load()
        self.assertIsNone(pipeline._pipe)

    def test_cli_requires_explicit_mock_or_config(self):
        parser = argparse.ArgumentParser()
        add_pipeline_arguments(parser)
        self.assertTrue(pipeline_from_args(parser.parse_args(['--mock'])).is_mock)
        with self.assertRaises(ValueError):
            pipeline_from_args(parser.parse_args([]))

    def test_read_image_applies_exif_and_rgb(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.png'
            image = Image.new('RGB', (20, 30), 'red')
            exif = Image.Exif()
            exif[274] = 6
            image.save(path, exif=exif)
            self.assertEqual(read_image(path).size, (30, 20))

    def test_prompt_only_has_no_identity_dependencies_or_adapter_residue(self):
        with fake_runtime() as rt:
            pipeline = AgeShiftPipeline(replace(rt.config, ip_adapter='', image_encoder='', age_lora=''), 'prompt_only')
            pipeline.load().load()
            rt.diffusers.StableDiffusionInpaintPipeline.from_pretrained.assert_called_once()
            rt.diffusers.StableDiffusionImg2ImgPipeline.from_pretrained.assert_not_called()
            rt.pipe.load_ip_adapter.assert_not_called()
            rt.pipe.load_lora_weights.assert_not_called()
            rt.transformers.CLIPVisionModelWithProjection.from_pretrained.assert_not_called()
            rt.face_module.FaceTools.assert_called_once_with('models/insightface', 'cpu', identity=False)
            pipeline.age_shift(Image.new('RGB', (64, 64)), [15, 35, 45], seed=7)
            self.assertEqual(rt.pipe.call_count, 3)
            self.assertNotIn('ip_adapter_image_embeds', rt.pipe.call_args.kwargs)
            self.assertEqual([c.args[0] for c in rt.torch.Generator.return_value.manual_seed.call_args_list], [7, 7, 7])
            rt.faces.embedding.assert_not_called()

    def test_faceid_uses_bundled_lora_without_loading_age_lora(self):
        with fake_runtime() as rt:
            pipeline = AgeShiftPipeline(replace(rt.config, age_lora=''), 'faceid').load()
            rt.pipe.set_adapters.assert_called_once_with(['faceid_0'], adapter_weights=[1.0])
            rt.pipe.load_lora_weights.assert_not_called()
            self.assertTrue(rt.projection.shortcut)
            self.assertEqual(rt.projection.shortcut_scale, 1.0)
            pipeline.age_shift(Image.new('RGB', (64, 64)), [35])
            rt.tensor.reshape.assert_called_once_with(1, 1, 1, -1)
            self.assertEqual(len(rt.pipe.call_args.kwargs['ip_adapter_image_embeds']), 1)
            self.assertIsNotNone(rt.projection.clip_embeds)
            self.assertTrue(rt.pipe.load_ip_adapter.call_args.kwargs['local_files_only'])
            self.assertEqual(rt.pipe.load_ip_adapter.call_args.kwargs['subfolder'], '')

    def test_full_mode_selects_both_loras_with_separate_scales(self):
        with fake_runtime() as rt:
            AgeShiftPipeline(replace(rt.config, faceid_lora_scale=.7, age_lora_scale=.9)).load()
            rt.pipe.load_lora_weights.assert_called_once()
            self.assertEqual(rt.pipe.load_lora_weights.call_args.kwargs['adapter_name'], 'age')
            rt.pipe.set_adapters.assert_called_once_with(['faceid_0', 'age'], adapter_weights=[.7, .9])

    def test_shortcut_scale_is_separate_from_image_attention_scale(self):
        for scale in (0, .25, 1.5):
            with self.subTest(scale=scale), fake_runtime() as rt:
                AgeShiftPipeline(replace(rt.config, shortcut_scale=scale, ip_scale=.6), 'faceid').load()
                self.assertTrue(rt.projection.shortcut)
                self.assertEqual(rt.projection.shortcut_scale, scale)
                rt.pipe.set_ip_adapter_scale.assert_called_once_with(.6)

    def test_numeric_prompt_distinguishes_55_and_65_without_changing_caption_mapping(self):
        self.assertEqual(age_prompt(55, 'female'), age_prompt(65, 'female'))
        for style in ('group', 'numeric'):
            with self.subTest(style=style), fake_runtime() as rt:
                pipeline = AgeShiftPipeline(replace(rt.config, prompt_style=style), 'prompt_only')
                pipeline.age_shift(Image.new('RGB', (64, 64)), [55, 65])
                prompts = [call.kwargs['prompt'] for call in rt.pipe.call_args_list]
                expected = [age_prompt(age, 'female') + (f', {age} years old' if style == 'numeric' else '')
                            for age in (55, 65)]
                self.assertEqual(prompts, expected)
                self.assertEqual(prompts[0] == prompts[1], style == 'group')
                rt.pipe.mask_processor.preprocess.assert_not_called()
                self.assertFalse(any('diagnostic' in key for key in vars(pipeline)))

    def test_explicit_faceid_lora_replaces_not_duplicates_automatic_one(self):
        with fake_runtime() as rt:
            AgeShiftPipeline(replace(rt.config, faceid_lora=str(rt.root / 'faceid.safetensors')), 'faceid').load()
            rt.pipe.delete_adapters.assert_called_once_with('faceid_0')
            rt.pipe.set_adapters.assert_called_once_with(['faceid'], adapter_weights=[1.0])
        with fake_runtime(automatic=False) as rt:
            with self.assertRaises(ModelUnavailableError):
                AgeShiftPipeline(rt.config, 'faceid').load()

    def test_face_failure_and_inference_failure_are_explicit(self):
        with fake_runtime() as rt:
            pipeline = AgeShiftPipeline(rt.config, 'prompt_only').load()
            prepare_edit = rt.faces.prepare_edit.side_effect
            rt.faces.prepare_edit.side_effect = FaceNotFoundError('face_not_detected')
            with self.assertRaises(FaceNotFoundError):
                pipeline.age_shift(Image.new('RGB', (64, 64)), [35])
            rt.faces.prepare_edit.side_effect = prepare_edit
            rt.pipe.side_effect = RuntimeError('device failure')
            with self.assertRaisesRegex(GenerationError, 'Inference failed'):
                pipeline.age_shift(Image.new('RGB', (64, 64)), [35])

    def test_face_only_inpainting_composites_into_original_dimensions(self):
        with fake_runtime() as rt:
            source = Image.new('RGB', (320, 240), 'red')
            before = source.tobytes()
            result = AgeShiftPipeline(rt.config, 'faceid').age_shift(source, [25, 65])
            self.assertEqual([image.size for image in result], [source.size, source.size])
            self.assertEqual(source.tobytes(), before)
            rt.faces.prepare_edit.assert_called_once_with(
                source, generation_mask_mode='protected', mask_scale=1.0, feature_protection=1.0)
            rt.faces.prepare.assert_not_called()
            self.assertEqual(rt.faces.finish_edit.call_count, 2)
            self.assertEqual(rt.pipe.call_args.kwargs['mask_image'].mode, 'L')
            self.assertIn('negative_prompt', rt.pipe.call_args.kwargs)

    def test_generation_mask_and_face_controls_are_forwarded_separately(self):
        with fake_runtime() as rt:
            source = Image.new('RGB', (320, 240))
            edit = rt.faces.prepare_edit.side_effect(source)
            edit.generation_mask = Image.new('L', (512, 512), 192)
            rt.faces.prepare_edit.side_effect = None
            rt.faces.prepare_edit.return_value = edit
            pipeline = AgeShiftPipeline(replace(rt.config, generation_mask_mode='face', mask_scale=1.2,
                                               feature_protection=.5), 'prompt_only')
            pipeline.age_shift(source, [25])
            rt.faces.prepare_edit.assert_called_once_with(
                source, generation_mask_mode='face', mask_scale=1.2, feature_protection=.5)
            self.assertIs(rt.pipe.call_args.kwargs['mask_image'], edit.generation_mask)
            self.assertIs(rt.faces.finish_edit.call_args.args[0], edit)
            self.assertEqual(edit.mask.getpixel((0, 0)), 255)

    def test_img2img_can_retain_source_composition_and_identical_negative_prompt(self):
        with fake_runtime() as rt:
            source = Image.new('RGB', (320, 240), 'red')
            events = []
            pipeline = AgeShiftPipeline(replace(rt.config, masked_denoising=False), 'prompt_only')
            results = pipeline.age_shift(source, [55, 65], seed=9, diagnostics=events.append)
            rt.diffusers.StableDiffusionImg2ImgPipeline.from_pretrained.assert_called_once()
            rt.diffusers.StableDiffusionInpaintPipeline.from_pretrained.assert_not_called()
            self.assertNotIn('mask_image', rt.pipe.call_args.kwargs)
            self.assertEqual(rt.pipe.call_args.kwargs['negative_prompt'],
                             'deformed face, distorted anatomy, blurry, waxy skin')
            self.assertEqual(rt.faces.finish_edit.call_count, 2)
            rt.pipe.mask_processor.preprocess.assert_not_called()
            self.assertEqual([event['target'] for event in events], [55, 65])
            for event, output in zip(events, results):
                self.assertEqual(set(event), {'target', 'seed', 'prompt', 'edit', 'generation_mask',
                                             'latent_mask', 'raw', 'output'})
                self.assertEqual(event['seed'], 9)
                self.assertIsNotNone(event['edit'])
                self.assertIsNone(event['generation_mask'])
                self.assertIsNone(event['latent_mask'])
                self.assertEqual(event['raw'].size, (512, 512))
                self.assertEqual(event['output'].size, source.size)
                event['raw'].putpixel((0, 0), (255, 0, 0))
                self.assertEqual(event['output'].getpixel((0, 0)), (0, 0, 255))
                event['output'].putpixel((0, 0), (0, 255, 0))
                self.assertEqual(output.getpixel((0, 0)), (0, 0, 255))

    def test_external_masks_are_forwarded_only_when_supplied(self):
        with fake_runtime() as rt:
            source = Image.new('RGB', (320, 240))
            masks = {name: Image.new('L', source.size, value) for name, value in
                     (('source_edit_mask', 255), ('source_generation_mask', 192), ('source_protect_mask', 1))}
            pipeline = AgeShiftPipeline(rt.config, 'prompt_only')
            pipeline.age_shift(source, [25], **masks)
            rt.faces.prepare_edit.assert_called_once_with(
                source, generation_mask_mode='protected', mask_scale=1.0, feature_protection=1.0, **masks)
            rt.faces.prepare_edit.reset_mock()
            pipeline.age_shift(source, [25], source_edit_mask=None, source_generation_mask=None, source_protect_mask=None)
            rt.faces.prepare_edit.assert_called_once_with(
                source, generation_mask_mode='protected', mask_scale=1.0, feature_protection=1.0)

    def test_negative_override_preserves_default_empty_string_and_positive_prompts(self):
        for masked in (False, True):
            for negative in (None, '', 'eyeglasses, sunglasses'):
                with self.subTest(masked=masked, negative=negative), fake_runtime() as rt:
                    pipeline = AgeShiftPipeline(replace(rt.config, masked_denoising=masked), 'prompt_only')
                    pipeline.age_shift(Image.new('RGB', (320, 240)), [55, 65], negative_prompt=negative)
                    for call in rt.pipe.call_args_list:
                        self.assertEqual(call.kwargs['negative_prompt'],
                                         'deformed face, distorted anatomy, blurry, waxy skin' if negative is None else negative)
                        self.assertEqual(call.kwargs['prompt'], age_prompt(55, 'female'))
                    pipeline.age_shift(Image.new('RGB', (320, 240)), [55])
                    self.assertEqual(rt.pipe.call_args.kwargs['negative_prompt'],
                                     'deformed face, distorted anatomy, blurry, waxy skin')

    def test_override_validation_rejects_ignored_crop_mode_and_bad_negative_before_load(self):
        source = Image.new('RGB', (320, 240))
        pipeline = AgeShiftPipeline(PipelineConfig())
        for value in (False, 0, [], {}):
            with self.assertRaisesRegex(ValueError, 'negative_prompt'):
                pipeline.age_shift(source, [25], negative_prompt=value)
        self.assertIsNone(pipeline._pipe)
        crop_pipeline = AgeShiftPipeline(PipelineConfig(preserve_background=False))
        for kwargs in ({'negative_prompt': ''}, {'negative_prompt': 'glasses'},
                       *({name: Image.new('L', source.size, 255)} for name in
                         ('source_edit_mask', 'source_generation_mask', 'source_protect_mask'))):
            with self.assertRaisesRegex(ValueError, 'preserve_background=True'):
                crop_pipeline.age_shift(source, [25], **kwargs)
        self.assertIsNone(crop_pipeline._pipe)

    def test_crop_diagnostics_have_no_edit_or_masks_and_cannot_mutate_results(self):
        with fake_runtime() as rt:
            events = []
            def record(event):
                event['raw'].putpixel((0, 0), (255, 0, 0))
                self.assertEqual(event['output'].getpixel((0, 0)), (0, 0, 255))
                event['output'].putpixel((0, 0), (0, 255, 0))
                events.append(event)
            pipeline = AgeShiftPipeline(replace(rt.config, preserve_background=False), 'prompt_only')
            results = pipeline.age_shift(Image.new('RGB', (320, 240)), [25, 65], diagnostics=record)
            for event, output in zip(events, results):
                self.assertIsNone(event['edit'])
                self.assertIsNone(event['generation_mask'])
                self.assertIsNone(event['latent_mask'])
                self.assertEqual(output.getpixel((0, 0)), (0, 0, 255))
            self.assertNotIn('negative_prompt', rt.pipe.call_args.kwargs)

    def test_diagnostics_validation_and_callback_failures_are_explicit(self):
        pipeline = AgeShiftPipeline(PipelineConfig())
        with self.assertRaisesRegex(ValueError, 'diagnostics'):
            pipeline.age_shift(Image.new('RGB', (64, 64)), [25], diagnostics=True)
        self.assertIsNone(pipeline._pipe)
        with fake_runtime() as rt:
            callback = Mock(side_effect=OSError('diagnostic disk failure'))
            pipeline = AgeShiftPipeline(replace(rt.config, masked_denoising=False), 'prompt_only')
            with self.assertRaisesRegex(GenerationError, 'diagnostic disk failure'):
                pipeline.age_shift(Image.new('RGB', (64, 64)), [25, 65], diagnostics=callback)
            self.assertEqual(rt.pipe.call_count, 1)

    def test_explicit_crop_mode_remains_available_for_existing_callers(self):
        with fake_runtime() as rt:
            result = AgeShiftPipeline(replace(rt.config, preserve_background=False), 'prompt_only').age_shift(
                Image.new('RGB', (320, 240)), [25])
            rt.diffusers.StableDiffusionImg2ImgPipeline.from_pretrained.assert_called_once()
            rt.faces.prepare_edit.assert_not_called()
            rt.faces.finish_edit.assert_not_called()
            self.assertNotIn('mask_image', rt.pipe.call_args.kwargs)
            self.assertEqual(result[0].size, (512, 512))

    def test_safety_checker_rejection_is_not_a_successful_image(self):
        with fake_runtime() as rt:
            rt.pipe.return_value.nsfw_content_detected = [True]
            pipeline = AgeShiftPipeline(rt.config, 'prompt_only').load()
            with self.assertRaisesRegex(GenerationError, 'blocked'):
                pipeline.age_shift(Image.new('RGB', (64, 64)), [35])


class DiffusersMaskContracts(unittest.TestCase):
    def test_diagnostics_match_real_mask_preprocessing_and_prepare_mask_latents(self):
        try:
            import numpy as np
            import torch
            from diffusers import StableDiffusionInpaintPipeline
            from diffusers.image_processor import VaeImageProcessor
        except ImportError:
            self.skipTest('Optional CPU torch/diffusers stack is not installed')
        from gen.faces import FaceEdit

        # Include values on both sides of the actual processor's 0.5 threshold.
        generation = Image.fromarray(np.tile(np.array([0, 126, 127, 128, 129, 255], dtype=np.uint8), (35, 9)))
        source = Image.new('RGB', (320, 240), 'red')
        for latent_size in (32, 64):
            with self.subTest(latent_size=latent_size):
                edit = FaceEdit(source, Image.new('RGB', (512, 512)), Image.new('L', (512, 512)),
                                np.array([[1., 0, 0], [0, 1., 0]]), SimpleNamespace(gender=1),
                                Image.new('L', source.size), {}, generation.copy())
                pipe = Mock()
                pipe.unet = SimpleNamespace(config=SimpleNamespace(sample_size=latent_size))
                pipe.vae_scale_factor = 8
                processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False,
                                              do_binarize=True, do_convert_grayscale=True)
                pipe.mask_processor = processor
                size = latent_size * 8
                processed = processor.preprocess(generation, height=size, width=size)
                expected, _ = StableDiffusionInpaintPipeline.prepare_mask_latents(
                    pipe, processed, torch.zeros(1, 4, latent_size, latent_size), 1, size, size,
                    torch.float32, torch.device('cpu'), None, True)
                processor.preprocess = Mock(wraps=processor.preprocess)
                pipe.return_value = SimpleNamespace(images=[Image.new('RGB', (512, 512), 'blue')])
                pipeline = AgeShiftPipeline(PipelineConfig(device='cpu', prompt_style='numeric'), 'prompt_only')
                pipeline._pipe = pipe
                pipeline._faces = SimpleNamespace(prepare_edit=Mock(return_value=edit),
                                                  finish_edit=Mock(return_value=source.copy()))
                events = []
                def record(event):
                    events.append(event)
                    self.assertIs(event['edit'], edit)
                    self.assertEqual(event['raw'].mode, 'RGB')
                    self.assertEqual(event['output'].mode, 'RGB')
                    self.assertEqual(event['generation_mask'].mode, 'L')
                    self.assertEqual(event['latent_mask'].mode, 'L')
                    self.assertEqual(event['latent_mask'].size, (latent_size, latent_size))
                    np.testing.assert_array_equal(np.asarray(event['latent_mask']),
                                                  (expected[0, 0] * 255).to(torch.uint8).numpy())
                    event['generation_mask'].paste(0, (0, 0, *generation.size))
                    event['latent_mask'].paste(0, (0, 0, latent_size, latent_size))
                result = pipeline.age_shift(source, [55, 65], diagnostics=record)
                processor.preprocess.assert_called_once_with(
                    edit.generation_mask, height=size, width=size, resize_mode='default', crops_coords=None)
                self.assertEqual([event['prompt'] for event in events],
                                 [age_prompt(age, 'male') + f', {age} years old' for age in (55, 65)])
                self.assertEqual(edit.generation_mask.tobytes(), generation.tobytes())
                self.assertEqual(result[0].tobytes(), source.tobytes())
                self.assertEqual(pipe.call_count, 2)


class ModelPreparationContracts(unittest.TestCase):
    def test_arcface_is_an_optional_extension_of_existing_setup(self):
        from age import setup_models
        for arcface in (False, True):
            with self.subTest(arcface=arcface), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                archive = root / 'buffalo_l.zip'
                with zipfile.ZipFile(archive, 'w') as z:
                    for name in ('det_10g.onnx', 'genderage.onnx', 'w600k_r50.onnx'):
                        z.writestr(name, b'fixture-only-not-a-real-model')
                destination = root / 'models'
                args = ['setup_models', '--model-root', str(destination), '--archive', str(archive)]
                if arcface:
                    args.append('--with-arcface')
                def fake_download(url, path):
                    Path(path).write_bytes(b'fixture-only-not-runtime-code')
                with patch('sys.argv', args), patch.object(setup_models, 'download', side_effect=fake_download):
                    setup_models.main()
                self.assertEqual((destination / 'models/buffalo_l/w600k_r50.onnx').exists(), arcface)
                self.assertEqual((destination / 'python_runtime/model_zoo/arcface_onnx.py').exists(), arcface)
                provenance = json.loads((destination / 'provenance.json').read_text())
                self.assertEqual(sum('model' in row for row in provenance['source_files']), 3 if arcface else 2)


if __name__ == '__main__':
    unittest.main()
