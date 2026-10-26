"""Offline contracts only. Fake runtimes do not validate diffusion/tensor numerics."""
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
    """Only tests loader/call wiring, never emulates numerical inference."""
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
        faces.prepare_edit.side_effect = lambda image: SimpleNamespace(
            original=image, crop=Image.new('RGB', (512, 512)),
            mask=Image.new('L', (512, 512), 255), face=SimpleNamespace(gender=0))
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
                       {'preserve_background': 'true'}):
            with self.assertRaises(ValueError):
                PipelineConfig(**values)

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
            rt.faces.prepare_edit.assert_called_once_with(source)
            rt.faces.prepare.assert_not_called()
            self.assertEqual(rt.faces.finish_edit.call_count, 2)
            self.assertEqual(rt.pipe.call_args.kwargs['mask_image'].mode, 'L')
            self.assertIn('negative_prompt', rt.pipe.call_args.kwargs)

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
