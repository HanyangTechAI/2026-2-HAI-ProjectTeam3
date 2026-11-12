#!/usr/bin/env python3
"""CPU-only image acceptance checks. Never downloads or loads model weights.

This is deliberately separate from first-rental GPU acceptance. A passing report
does not establish SAM CUDA compilation, model quality, VRAM use or training speed.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile


IMPORT_PROBES = {
    "main": """
import json, importlib.metadata as md
import torch, torchvision, cv2, onnxruntime, onnx, numpy, skimage
import datasets, accelerate, peft, safetensors
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionImg2ImgPipeline
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
assert torch.__version__ == '2.8.0+cu128', torch.__version__
assert torchvision.__version__ == '0.23.0+cu128', torchvision.__version__
assert md.version('diffusers') == '0.35.1'
assert torch.version.cuda == '12.8', torch.version.cuda
print(json.dumps({p: md.version(p) for p in (
    'torch','torchvision','diffusers','transformers','peft','accelerate',
    'huggingface-hub','datasets','numpy','onnxruntime','Pillow')}))
""",
    "web": """
import json, importlib.metadata as md
import torch, torchvision, cv2, onnxruntime, numpy
import fastapi, uvicorn, flask, serial, qrcode, requests
from fastapi.testclient import TestClient
from diffusers import StableDiffusionInpaintPipeline
from transformers import CLIPVisionModelWithProjection
assert torch.__version__ == '2.8.0+cu128', torch.__version__
assert md.version('diffusers') == '0.35.1'
assert md.version('numpy') == '2.2.6'
assert md.version('onnxruntime') == '1.23.2'
assert md.version('Pillow') == '12.1.1'
print(json.dumps({p: md.version(p) for p in (
    'torch','diffusers','numpy','onnxruntime','Pillow','fastapi','uvicorn',
    'Flask','pyserial','qrcode')}))
""",
    "m1": """
import json, importlib.metadata as md
import torch, torchvision, accelerate, peft, datasets, safetensors
from diffusers import ModularPipeline, Flux3Transformer2DModel, AutoencoderKLFlux2
from diffusers.modular_pipelines.flux3 import Flux3AutoBlocks
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
assert md.version('diffusers') == '0.42.0'
print(json.dumps({p: md.version(p) for p in (
    'torch','torchvision','diffusers','transformers','peft','accelerate',
    'huggingface-hub','datasets','safetensors')}))
""",
}

API_SMOKE = """
import io
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app
with TestClient(app) as client:
    health = client.get('/')
    assert health.status_code == 200, health.text
    assert health.json()['mock_generation_mode'] is True
    qr = client.get('/api/v1/generate-qr', params={'url':'https://example.com/photo'})
    assert qr.status_code == 200, qr.text
    assert Image.open(io.BytesIO(qr.content)).format == 'PNG'
    source = io.BytesIO()
    Image.new('RGB', (96, 96), 'white').save(source, format='PNG')
    generated = client.post('/api/v1/generate-timemachine-photo',
        files={'file':('synthetic.png', source.getvalue(), 'image/png')},
        data={'target_age':25, 'seed':42})
    assert generated.status_code == 200, generated.text
    record = generated.json()
    assert record['is_mock'] is True, record
    photo = client.get('/photos/' + record['filename'])
    assert photo.status_code == 200, photo.text
    assert Image.open(io.BytesIO(photo.content)).size == (1024, 1024)
print('FastAPI health, QR, explicit Mock generation and photo download: PASS')
"""

RASPI_SMOKE = """
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image
camera = MagicMock()
printer = MagicMock()
with patch('cv2.VideoCapture', return_value=camera), \
     patch('serial.Serial', return_value=printer), patch('time.sleep'):
    spec = importlib.util.spec_from_file_location('hai_raspi_client', Path('raspi/app.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    response = module.app.test_client().get('/')
    assert response.status_code == 200
    module.process_and_print_image(Image.new('RGB', (32, 32), 'white'), 'test')
    assert printer.write.call_count > 0
    assert printer.flush.called and printer.close.called
print('Flask UI and image-to-printer-byte path: PASS (camera/serial mocked)')
"""


def run(command, *, env, timeout=600):
    print('+ ' + shlex.join([str(p) for p in command]), flush=True)
    result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        print(result.stdout[-16000:], flush=True)
        print(result.stderr[-16000:], flush=True)
        raise RuntimeError(f'Acceptance command failed with exit {result.returncode}')
    # Keep build logs useful without repeating hundreds of successful tests.
    if result.stdout.strip():
        print(result.stdout[-4000:], flush=True)
    if result.stderr.strip():
        print(result.stderr[-4000:], flush=True)
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('/opt/hai/config/branches.json'))
    parser.add_argument('--report', type=Path, default=Path('/opt/hai/build-report.json'))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    report = {
        'scope': 'CPU/import/Mock only; no real GPU, weights, age accuracy or training validation',
        'verified_at_utc': datetime.now(timezone.utc).isoformat(),
        'real_gpu_validation': False,
        'real_model_validation': False,
        'environments': {},
        'branches': config['branches'],
        'checks': [],
    }
    with tempfile.TemporaryDirectory(prefix='hai-image-verify-') as temporary:
        env = dict(os.environ, HAI_CONFIG=str(args.config),
                   HAI_WORKSPACE=str(Path(temporary) / 'workspace'),
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   USE_MOCK_GEN='true', GEN_DEVICE='cpu',
                   OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        for name, prefix in config['environments'].items():
            output = run([str(Path(prefix) / 'bin/python'), '-c', IMPORT_PROBES[name]], env=env)
            report['environments'][name] = json.loads(output.strip().splitlines()[-1])
        run(['hai', 'setup', '--all'], env=env)
        for branch, details in config['branches'].items():
            head = run(['hai', 'run', branch, '--', 'git', 'rev-parse', 'HEAD'], env=env)
            if head.strip().splitlines()[-1] != details['commit']:
                raise RuntimeError(f'Unexpected source snapshot for {branch}')
            run(['hai', 'run', branch, '--', 'python', '-m', 'age.train', '--help'], env=env)
        report['checks'].append('all four pinned source snapshots and age CLI imports')

        run(['hai', 'run', 'main', '--', 'python', '-m', 'unittest',
             'discover', '-s', 'tests', '-v'], env=env, timeout=900)
        report['checks'].append('main existing unit test suite (model-dependent skips are reported)')
        run(['hai', 'run', 'feat/generation-midterm', '--', 'python', '-m',
             'gen.smoke', '--output', 'outputs/image-build-smoke'], env=env)
        report['checks'].append('generation-midterm Mock CLI and four-cut composition')
        run(['hai', 'run', 'feat/raspi-fastapi-pipeline', '--',
             'python', '-c', API_SMOKE], env=env)
        report['checks'].append('FastAPI health/QR/Mock generation/photo retrieval')
        run(['hai', 'run', 'feat/raspi-client', '--',
             'python', '-c', RASPI_SMOKE], env=env)
        report['checks'].append('Raspberry Pi UI/image processing with mocked hardware')
        m1_script = Path('/opt/hai/tools/train_dreambooth_lora_flux3_klein_img2img.py')
        if not m1_script.is_file():
            raise RuntimeError(f'Missing pinned upstream training example: {m1_script}')
        run([str(Path(config['environments']['m1']) / 'bin/python'),
             str(m1_script), '--help'], env=env)
        report['checks'].append('FLUX 3 official image-editing LoRA example imports and CLI')

    report['status'] = 'passed'
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(f'CPU-only image acceptance passed; report: {args.report}', flush=True)


if __name__ == '__main__':
    main()
