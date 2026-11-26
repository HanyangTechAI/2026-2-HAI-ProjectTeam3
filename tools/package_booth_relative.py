#!/usr/bin/env python3
"""Package reviewed team PR #3 into offline, source-only field installers.

Pinned PR code is read as text, not imported; no models or photographs are bundled.
"""
import ast
import hashlib
from pathlib import Path
import subprocess
import shutil

ROOT = Path(__file__).resolve().parents[1]
PR3 = '568b3c8'
PR2 = 'b526edc'


def git_source(commit, path):
    return subprocess.check_output(['git', 'show', f'{commit}:{path}'], cwd=ROOT, text=True)


def fingerprint(source):
    return hashlib.sha256(ast.dump(ast.parse(source), include_attributes=False).encode()).hexdigest()


def payloads():
    common = git_source(PR3, 'age/common.py')
    target = next(n for n in ast.parse(common).body if isinstance(n, ast.FunctionDef) and n.name == 'target_ages')
    timeline = ('"""Relative age contract adapted from team PR #3 (568b3c8)."""\n'
                'import math\nfrom dataclasses import dataclass\n\n' +
                ast.get_source_segment(common, target) + '\n\n' + common[common.index('@dataclass(frozen=True)'):])
    runner = git_source(PR3, 'scripts/fading_fourcut.py')
    runner = runner.replace('from age.common import EDIT_KEYS, frame_name',
                            'from scripts.booth_timeline import EDIT_KEYS, frame_name')
    runner = runner.replace('safety_checker=None).to', 'safety_checker=None, local_files_only=True).to')
    api = git_source(PR3, 'app/fading_booth_api.py')
    api = '# booth-relative-api-v1\n' + api.replace('from age.common import booth_slots',
                                                   'from scripts.booth_timeline import booth_slots')
    api = api.replace('print("実モデル".replace("実モデル", "실제 나이 추정 모델 로딩 중..."), flush=True)',
                      'print("실제 나이 추정 모델 로딩 중...", flush=True)')
    api = api.replace('HAI ', 'Runpod ')
    files = {'app/fading_booth_api.py': api, 'scripts/booth_timeline.py': timeline,
             'scripts/fading_fourcut.py': runner}
    for path, source in files.items():
        compile(source, path, 'exec')
    accepted = {fingerprint(git_source(PR2, 'app/fading_booth_api.py')),
                fingerprint(git_source(PR3, 'app/fading_booth_api.py'))}
    return files, accepted


INSTALLER = r'''
import argparse
import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil


def fingerprint(source):
    return hashlib.sha256(ast.dump(ast.parse(source), include_attributes=False).encode()).hexdigest()


def _write(path, content, mode):
    temporary = path.with_name(path.name + '.relative-tmp')
    if temporary.exists():
        raise ValueError('이전 임시 파일이 있어 중단했어요: ' + str(temporary))
    try:
        temporary.write_text(content, encoding='utf-8')
        temporary.chmod(mode)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def install(project):
    project = project.resolve()
    path = project / 'app/fading_booth_api.py'
    current = path.read_text(encoding='utf-8')
    if '# booth-relative-api-v1' in current:
        raise ValueError('상대 연령 변경본이 이미 있어요. 중복 설치하지 않았어요.')
    if fingerprint(current) not in ACCEPTED_API:
        raise ValueError('현재 사진 처리 코드가 검토한 버전과 달라요. 덮어쓰지 않았으니 출력 내용을 알려주세요.')
    if not (project / 'app/age_service.py').is_file():
        raise ValueError('실제 나이 추정 코드를 찾지 못했어요.')
    cache = Path.home() / '.cache'
    if not (cache / 'fading-src/null_inversion.py').is_file():
        raise ValueError('기존 FADING 코드 연결을 찾지 못했어요.')
    if not (cache / 'fading-models/finetune_double_prompt_150_random/model_index.json').is_file():
        raise ValueError('기존 FADING 가중치 연결을 찾지 못했어요.')
    originals = {}
    for name, source in FILES.items():
        compile(source, name, 'exec')
        item = project / name
        if item.is_symlink() or not item.parent.is_dir():
            raise ValueError('예상하지 못한 코드 경로여서 중단했어요: ' + name)
        if name != 'app/fading_booth_api.py' and item.exists():
            # Preserve unknown team changes instead of silently replacing them.
            allowed = KNOWN_HELPERS.get(name, set())
            if fingerprint(item.read_text(encoding='utf-8')) not in allowed:
                raise ValueError('다른 생성 코드가 이미 있어 덮어쓰지 않았어요: ' + name)
        originals[name] = (item.read_text(encoding='utf-8'), item.stat().st_mode & 0o777) if item.exists() else None
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    backup = project / ('.booth_relative_backup_' + stamp)
    backup.mkdir(mode=0o700)
    for name, original in originals.items():
        if original is not None:
            saved = backup / name
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(project / name, saved)
    (backup / 'manifest.json').write_text(json.dumps({name: value is not None for name, value in originals.items()}, indent=2))
    changed = []
    try:
        for name, content in FILES.items():
            original = originals[name]
            _write(project / name, content, original[1] if original else 0o644)
            changed.append(name)
    except Exception:
        for name in reversed(changed):
            item = project / name
            original = originals[name]
            if original is None:
                item.unlink()
            else:
                shutil.copy2(backup / name, item)
        raise
    return backup


def main():
    parser = argparse.ArgumentParser(description='Runpod의 실제 모델·가중치를 그대로 사용하며 상대 연령 생성 코드만 백업 후 변경해요.')
    parser.add_argument('--project', type=Path, default=Path('/workspace/age-photobooth'))
    args = parser.parse_args()
    try:
        backup = install(args.project)
    except (ValueError, OSError, SyntaxError) as error:
        raise SystemExit('설치 중단: ' + str(error))
    print('상대 연령 생성 코드 준비 완료. 아직 실행 중인 프로그램은 바뀌지 않았어요.')
    print('백업:', backup)
    print('실행 중인 촬영·생성이 없는지 확인한 뒤 기존 처리 프로그램을 종료하고 같은 명령으로 다시 실행해야 해요.')
    print('라즈베리파이 변경 후 실제 촬영 시험 전까지 상대 연령 생성 성공으로 보지 마세요.')


if __name__ == '__main__':
    main()
'''


def runpod_installer():
    files, accepted = payloads()
    known = {
        'scripts/fading_fourcut.py': {
            fingerprint(files['scripts/fading_fourcut.py']),
            fingerprint(git_source(PR3, 'scripts/fading_fourcut.py')),
        },
        'scripts/booth_timeline.py': {fingerprint(files['scripts/booth_timeline.py'])},
    }
    return ('#!/usr/bin/env python3\n'
            '"""Runpod installer based on reviewed team PR #3; no models or photos."""\n'
            f'FILES = {files!r}\nACCEPTED_API = {accepted!r}\nKNOWN_HELPERS = {known!r}\n' + INSTALLER)


def main():
    out = ROOT / '.baze-files/outputs'
    out.mkdir(parents=True, exist_ok=True)
    source = runpod_installer()
    compile(source, 'upgrade_booth_relative_runpod.py', 'exec')
    (out / 'upgrade_booth_relative_runpod.py').write_text(source, encoding='utf-8')
    shutil.copy2(ROOT / 'tools/upgrade_booth_ui_qr.py', out / 'upgrade_booth_relative_pi.py')
    shutil.copy2(ROOT / 'docs/booth_ui_qr_upgrade.md', out / 'booth_relative_setup.md')
    print('상대 연령 설치 파일 2개와 적용 안내 준비 완료')


if __name__ == '__main__':
    main()
