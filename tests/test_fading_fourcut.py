"""Deterministic booth contract tests; no weights, downloads, or GPU calls."""
import importlib.util
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from age import common
from scripts import fading_fourcut as runner


@pytest.mark.parametrize('prediction,ages', [
    (27, [17, 27, 37, 47]), (26.49, [16, 26, 36, 46]),
    (26.5, [17, 27, 37, 47]), (8, [3, 8, 18, 28]),
    (85, [75, 85, 90, 90]), (0, [3, 1, 11, 21]), (100, [90, 100, 90, 90]),
])
def test_timeline(prediction, ages):
    slots = common.booth_slots(prediction)
    assert [s.age for s in slots] == ages
    assert [s.key for s in slots] == ['past', 'current', 'future10', 'future20']
    assert len({s.filename for s in slots}) == 4
    for slot, expected, offset in zip(slots, ages, [-10, 0, 10, 20]):
        assert str(expected) in slot.label
        assert ('limited' in slot.label) == (expected != ages[1] + offset)


@pytest.mark.parametrize('prediction', [float('nan'), float('inf'), -1, 101])
def test_bad_prediction(prediction):
    with pytest.raises(ValueError):
        common.booth_slots(prediction)


@pytest.mark.parametrize('age,targets', [(27, [17, 37, 47]), (8, [3, 18, 28]), (85, [75, 90, 90])])
def test_api_dispatch_and_sheet(monkeypatch, tmp_path, age, targets):
    estimate = Mock(return_value={'age': age})
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setitem(sys.modules, 'app.age_service', SimpleNamespace(RealAgeEstimator=lambda: estimate))
    spec = importlib.util.spec_from_file_location('booth_test_api', common.ROOT / 'app/fading_booth_api.py')
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    calc = Mock(wraps=common.target_ages)
    monkeypatch.setattr(common, 'target_ages', calc)
    colors = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11)]

    def generate(args, **kwargs):
        assert args[1:3] == ['-m', 'scripts.fading_fourcut']
        assert args[args.index('--age')+1] == str(age)
        i = args.index('--targets') + 1
        assert args[i:i+3] == list(map(str, targets))
        output = Path(args[args.index('--output') + 1])
        output.mkdir()
        names = [f'past_{targets[0]}.png', 'current.png', f'future10_{targets[1]}.png', f'future20_{targets[2]}.png']
        for name, rgb in zip(names, colors):
            Image.new('RGB', (512, 512), rgb).save(output / name)
        (output / 'settings.json').write_text(json.dumps(dict(**runner.SETTINGS, initial_age=age,
                                                              gender='male', targets=targets)))

    run = Mock(side_effect=generate)
    monkeypatch.setattr(api.subprocess, 'run', run)
    upload = io.BytesIO()
    Image.new('RGB', (64, 64)).save(upload, format='PNG')
    upload.seek(0)
    response = api.make_fourcut(SimpleNamespace(file=upload), 'male')
    estimate.assert_called_once()
    calc.assert_called_once_with(age)
    run.assert_called_once()
    assert response.headers['x-target-ages'] == ','.join(map(str, targets))
    assert response.headers['x-initial-age'] == str(age)
    metadata = json.loads((Path(response.path).parent / 'metadata.json').read_text())
    assert metadata['job_id'] == response.headers['x-job-id']
    assert metadata['targets'] == targets
    assert [s['age'] for s in metadata['slots']] == [targets[0], age, *targets[1:]]
    with Image.open(response.path) as im:
        assert (im.size, im.format, im.mode) == ((1024, 1112), 'PNG', 'RGB')
        assert [im.getpixel(xy) for xy in [(0, 44), (512, 44), (0, 600), (512, 600)]] == colors


def test_shared_inversion_and_settings(monkeypatch, tmp_path):
    original = np.zeros((512, 512, 3), dtype=np.uint8)
    latent, embeddings = object(), [object() for _ in range(35)]
    inversion, p2p, torch = Mock(), Mock(), Mock()
    inversion.NullInversion.return_value.invert.return_value = ((original, None), latent, embeddings)
    p2p.p2p_text2image.return_value = ([original], None)
    for name, module in [('null_inversion', inversion), ('p2p', p2p), ('torch', torch),
                         ('diffusers', Mock()), ('FADING_util', SimpleNamespace(util=Mock(
                             get_person_placeholder=Mock(return_value='man'))))]:
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys, 'path', sys.path.copy())
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    monkeypatch.setenv('TRANSFORMERS_OFFLINE', '1')
    out = tmp_path / 'frames'
    runner.generate(tmp_path / 'input.jpg', 85, 'male', [75, 90, 90], out)
    assert {p.name for p in out.glob('*.png')} == {'current.png', 'past_75.png', 'future10_90.png', 'future20_90.png'}
    assert p2p.NUM_DDIM_STEPS == inversion.NUM_DDIM_STEPS == 35
    inversion.NullInversion.return_value.invert.assert_called_once_with(str(tmp_path / 'input.jpg'),
        'photo of 85 year old man', offsets=(0, 0, 0, 0), num_inner_steps=5, early_stop_epsilon=1e-5, verbose=True)
    assert p2p.p2p_text2image.call_count == 3
    for call, controller, age in zip(p2p.p2p_text2image.call_args_list, p2p.make_controller.call_args_list, [75, 90, 90]):
        assert call.kwargs['latent'] is latent and call.kwargs['uncond_embeddings'] is embeddings
        assert call.kwargs['num_inference_steps'] == call.kwargs['start_time'] == 35
        assert controller.args[:4] == (['photo of 85 year old man', f'photo of {age} year old man'], True, {'default_': 0.8}, 0.7)
