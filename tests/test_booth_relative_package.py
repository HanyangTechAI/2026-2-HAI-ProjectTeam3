"""Source-only installer and mocked model contract tests; no GPU or private photos."""
import ast
import importlib.util
import io
import json
from pathlib import Path
import sys
import types
from unittest.mock import Mock

import pytest
from PIL import Image

spec = importlib.util.spec_from_file_location('booth_packager', Path(__file__).resolve().parents[1] / 'tools/package_booth_relative.py')
packager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packager)


@pytest.fixture
def installer():
    ns = {'__name__': 'runpod_install_test'}
    exec(packager.runpod_installer(), ns)
    return ns


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, 'home', classmethod(lambda cls: tmp_path))
    root = tmp_path / 'project'
    for name in ('app', 'scripts'):
        (root / name).mkdir(parents=True)
    (root / 'app/fading_booth_api.py').write_text(packager.git_source(packager.PR2, 'app/fading_booth_api.py'))
    (root / 'app/age_service.py').write_text('# untouched actual estimator placeholder\n')
    for name in ('fading-src/null_inversion.py', 'fading-models/finetune_double_prompt_150_random/model_index.json'):
        p = tmp_path / '.cache' / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('synthetic model placeholder')
    return root


def test_install_backup_and_repeat(installer, project):
    original = (project / 'app/fading_booth_api.py').read_text()
    backup = installer['install'](project)
    assert (backup / 'app/fading_booth_api.py').read_text() == original
    assert (project / 'app/age_service.py').read_text() == '# untouched actual estimator placeholder\n'
    assert (project / 'scripts/fading_fourcut.py').exists()
    assert json.loads((backup / 'manifest.json').read_text())['scripts/booth_timeline.py'] is False
    with pytest.raises(ValueError, match='이미'):
        installer['install'](project)


def test_unknown_code_is_never_overwritten(installer, project):
    api = project / 'app/fading_booth_api.py'
    current = api.read_text() + '\ncustom_setting = "keep me"\n'
    api.write_text(current)
    with pytest.raises(ValueError, match='검토한 버전과 달라'):
        installer['install'](project)
    assert api.read_text() == current
    assert not list(project.glob('.booth_relative_backup_*'))


def test_failure_rolls_back_only_files_it_changed(installer, project):
    current = (project / 'app/fading_booth_api.py').read_text()
    write = installer['_write']
    def fail_last(path, content, mode):
        if path.name == 'fading_fourcut.py':
            raise OSError('synthetic disk failure')
        return write(path, content, mode)
    installer['_write'] = fail_last
    with pytest.raises(OSError, match='synthetic'):
        installer['install'](project)
    assert (project / 'app/fading_booth_api.py').read_text() == current
    assert not (project / 'scripts/booth_timeline.py').exists()
    assert (project / 'app/age_service.py').exists()


def module(name, source, filename):
    module = types.ModuleType(name)
    module.__file__ = str(filename)
    exec(compile(source, str(filename), 'exec'), module.__dict__)
    return module


@pytest.fixture
def modules(tmp_path, monkeypatch):
    files, _ = packager.payloads()
    timeline = module('scripts.booth_timeline', files['scripts/booth_timeline.py'], tmp_path/'scripts/booth_timeline.py')
    monkeypatch.setitem(sys.modules, 'scripts.booth_timeline', timeline)
    runner = module('scripts.fading_fourcut', files['scripts/fading_fourcut.py'], tmp_path/'scripts/fading_fourcut.py')
    monkeypatch.setitem(sys.modules, 'scripts.fading_fourcut', runner)
    return files, timeline, runner


@pytest.mark.parametrize('predicted,ages', [
    (24.2,[14,24,34,44]), (26.5,[17,27,37,47]), (8,[3,8,18,28]),
    (85,[75,85,90,90]), (0,[3,1,11,21]), (100,[90,100,90,90]),
])
def test_relative_ages_and_unique_files(modules, predicted, ages):
    _, timeline, _ = modules
    slots = timeline.booth_slots(predicted)
    assert [s.age for s in slots] == ages
    assert [s.key for s in slots] == ['past','current','future10','future20']
    assert len({s.filename for s in slots}) == 4


@pytest.mark.parametrize('predicted', [float('nan'),float('inf'),-1,101])
def test_invalid_age_fails_closed(modules, predicted):
    with pytest.raises(ValueError):
        modules[1].booth_slots(predicted)


@pytest.mark.parametrize('predicted,targets', [(24.2,[14,34,44]),(8,[3,18,28]),(85,[75,90,90])])
def test_api_dispatch_metadata_and_sheet(modules, tmp_path, monkeypatch, predicted, targets):
    files, timeline, runner = modules
    monkeypatch.setattr(Path, 'home', classmethod(lambda cls: tmp_path))
    monkeypatch.setitem(sys.modules, 'app.age_service', types.SimpleNamespace(RealAgeEstimator=lambda: lambda image: {'age':predicted}))
    api = module('synthetic_api', files['app/fading_booth_api.py'], tmp_path/'app/fading_booth_api.py')
    colors = [(20,30,40),(50,60,70),(80,90,100),(110,120,130)]
    def run(args, **kwargs):
        assert args[1:3] == ['-m','scripts.fading_fourcut']
        assert list(map(int,args[args.index('--targets')+1:args.index('--targets')+4])) == targets
        output = Path(args[args.index('--output')+1]);output.mkdir()
        slots = timeline.booth_slots(predicted)
        for slot, color in zip(slots,colors):
            Image.new('RGB',(512,512),color).save(output/slot.filename)
        (output/'settings.json').write_text(json.dumps(dict(**runner.SETTINGS, initial_age=slots[1].age, targets=targets, gender='female')))
    monkeypatch.setattr(api.subprocess,'run',run)
    image = io.BytesIO();Image.new('RGB',(64,64),'white').save(image,'PNG');image.seek(0)
    response = api.make_fourcut(types.SimpleNamespace(file=image),'female')
    assert response.headers['x-target-ages'] == ','.join(map(str,targets))
    metadata = json.loads((Path(response.path).parent/'metadata.json').read_text())
    assert metadata['slots'][1]['key'] == 'current'
    with Image.open(response.path) as sheet:
        assert sheet.size == (1024,1112)
        assert [sheet.getpixel(xy) for xy in ((0,44),(512,44),(0,600),(512,600))] == colors
    assert api.health()['target_offsets'] == [-10,10,20]


def test_runner_reuses_one_inversion_and_explicit_steps(modules, tmp_path, monkeypatch):
    import numpy as np
    _, _, runner = modules
    original = np.zeros((512,512,3),dtype=np.uint8)
    latent, embeddings = object(), [object() for _ in range(35)]
    inversion, p2p = Mock(), Mock()
    inversion.NullInversion.return_value.invert.return_value = ((original,None),latent,embeddings)
    p2p.p2p_text2image.return_value = ([original],None)
    for name, value in [('null_inversion',inversion),('p2p',p2p),('torch',Mock()),('diffusers',Mock()),('FADING_util',types.SimpleNamespace(util=Mock(get_person_placeholder=Mock(return_value='person'))))]:
        monkeypatch.setitem(sys.modules,name,value)
    monkeypatch.setattr(sys,'path',sys.path.copy())
    monkeypatch.setenv('HF_HUB_OFFLINE','1');monkeypatch.setenv('TRANSFORMERS_OFFLINE','1')
    runner.generate(tmp_path/'input.jpg',85,'female',[75,90,90],tmp_path/'frames')
    assert inversion.NullInversion.return_value.invert.call_count == 1
    assert p2p.p2p_text2image.call_count == 3
    assert inversion.NUM_DDIM_STEPS == p2p.NUM_DDIM_STEPS == 35
    for call in p2p.p2p_text2image.call_args_list:
        assert call.kwargs['latent'] is latent
        assert call.kwargs['uncond_embeddings'] is embeddings
        assert call.kwargs['num_inference_steps'] == 35
    assert len(list((tmp_path/'frames').glob('*.png'))) == 4
