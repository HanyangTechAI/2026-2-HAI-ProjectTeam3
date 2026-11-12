# M1-A 쌍 데이터셋 P 생성 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** M1 Age Adapter 학습에 쓸 "원본 얼굴 → 목표 나이 → 변환 얼굴" 쌍 데이터셋 P-SAM을 RTX 4060(8GB) 로컬 PC에서 생성하고, 팀 나이 추정 모델(FFHQ 10구간 분류기)과 ArcFace로 걸러 manifest로 남긴다.

**Architecture:**
- 공개 SAM 체크포인트를 teacher로 쓴다. FFHQ-Aging train 얼굴을 목표 나이 9개(5~85세)로 변환한다.
- 결과마다 두 가지를 판정한다.
  - `GroupJudge`(11/05 학습한 MobileNetV3 10구간 분류기): 목표 구간이 맞는지
  - `FaceTools.embedding`(ArcFace): 같은 사람인지
- 판정 결과는 `manifest.jsonl` 한 줄씩 기록한다. 중간에 끊겨도 같은 설정이면 이어서 실행할 수 있다.
- 모든 단계는 가짜 teacher/judge/faces로 단위 테스트한다. 실제 모델은 마지막 파일럿에서만 쓴다.

**Tech Stack:** Python 3.11, PyTorch 2.8 (cu128), torchvision, Pillow, numpy, onnxruntime(InsightFace ArcFace), SAM(yuval-alaluf/SAM, StyleGAN2 custom CUDA op), unittest

**Spec:** `docs/superpowers/specs/2026-11-06-age-transformation-training-options.md` (M1, 2.5 쌍 데이터셋 P)

## Global Constraints

- 작업 저장소: `HanyangTechAI/2026-2-HAI-ProjectTeam3`. `feat/generation-midterm`에서 `feat/m1-pair-dataset` 브랜치를 새로 만들어 작업한다. 이 브랜치에는 `gen/`, `age/ffhq.py`, `data/lora/`가 있다.
- 실행 장비: `data/lora/`(FFHQ-Aging 8,163/400/100)와 11/05 분류기 `best.pt`가 있는 팀원 로컬 RTX 4060 Linux PC
- 입력은 **train split만** 쓴다. val/test는 쓰지 않는다. 동일 인물 누수 후보는 `--exclude` 파일로 제외한다.
- 목표 나이: `5, 15, 25, 35, 45, 55, 65, 75, 85` (스펙 2.5: 5~85세, 10살 간격)
- ArcFace 기준: 코사인 유사도 **≥ 0.6**
- 나이 판정 기준 (스펙에서 바꾼 부분): 스펙은 "추정 나이 ±5세(65세 이상 ±8세)"였다. 하지만 학습된 나이 모델은 숫자가 아니라 **10구간 분류기**다. 그래서 **분류기의 최고 확률 구간 == 목표 나이가 속한 구간**일 때 통과로 판정한다.
- 출력 이미지 형식: 512×512 RGB PNG, FFHQ 정렬 (입력 `data/lora`와 같음)
- 추정기 분리 원칙: 필터는 MobileNet 구간 분류기로만 판정한다. 이후 평가(M1-C 계획)는 InsightFace genderage로 한다.
- 데이터, 가중치, 출력물, `third_party/`는 Git에 올리지 않는다 (`.gitignore`).
- 테스트는 저장소 관례대로 `python -m unittest`로 실행한다. 모델이나 GPU가 없어도 통과해야 한다.
- 커밋 메시지는 저장소 관례대로 `feat:`/`chore:`/`test:` 접두사를 쓴다.

## 범위

- **이 계획(M1-A):** P-SAM 생성, 필터, manifest, 요약 리포트
- **다음 계획(M1-B):** GPU VM에서 P-Edit(편집 모델 쌍) 생성, Age Adapter 학습. FLUX 3 Klein / FLUX.1 Kontext dev의 diffusers 학습 API를 먼저 확인하는 spike부터 시작한다.
- **다음 계획(M1-C):** 고정 val 20장 평가, 손실 ablation 4종, FastAPI 연동
- P-Real(AgeDB)과 아시아인 보충 입력(AAF, MegaAge-Asian)은 이번 파일럿 결과를 보고 필요하면 M1-B에서 다룬다.

## File Structure

| 파일 | 책임 |
|---|---|
| `age/group_judge.py` (생성) | 10구간 분류기 체크포인트 로드, 확률 계산, 나이→구간, 구간 확률→기댓값 나이 |
| `gen/sam_teacher.py` (생성) | SAM 입력 변환(이미지 + 나이 채널), 배치 추론, 출력 → PIL |
| `gen/pairs.py` (생성) | 원본 선택(구간 균형), 쌍 판정, 생성 루프(이어서 실행 가능), 요약·contact sheet, CLI |
| `tests/test_group_judge.py` (생성) | group_judge 계약 테스트 |
| `tests/test_gen_sam_teacher.py` (생성) | sam_teacher 계약 테스트 (가짜 net) |
| `tests/test_gen_pairs.py` (생성) | pairs 계약 테스트 (가짜 teacher/judge/faces) |
| `.gitignore` (수정) | `third_party/` 추가 |

---

### Task 1: 환경 준비 (SAM 코드, 체크포인트, ArcFace)

**Files:**
- Modify: `.gitignore`

**Interfaces:**
- Produces:
  - `third_party/SAM/` (SAM 저장소 클론)
  - `models/sam/sam_ffhq_aging.pt`
  - `models/insightface/models/buffalo_l/w600k_r50.onnx`
  - 이 세 경로를 Task 3과 Task 5가 기본값으로 쓴다.

- [ ] **Step 1: 브랜치 생성**

```bash
git fetch origin
git switch -c feat/m1-pair-dataset origin/feat/generation-midterm
```

- [ ] **Step 2: `.gitignore`에 third_party 추가**

`.gitignore` 맨 아래에 다음 두 줄을 추가한다.

```
# External research code cloned for local runs (SAM etc.).
third_party/
```

- [ ] **Step 3: SAM 클론, 체크포인트·ArcFace 준비**

```bash
git clone https://github.com/yuval-alaluf/SAM third_party/SAM
mkdir -p models/sam
# SAM README의 "Pretrained Models" 표에서 "SAM" (FFHQ aging) 체크포인트를 받아 아래 이름으로 저장
ls -la models/sam/sam_ffhq_aging.pt
python -m age.setup_models --with-arcface
ls models/insightface/models/buffalo_l/w600k_r50.onnx
pip install ninja
```

Expected: 세 경로가 모두 존재한다. `ninja`는 SAM의 StyleGAN2 CUDA op를 처음 import할 때 컴파일하는 데 필요하다.

- [ ] **Step 4: 11/05 분류기 체크포인트 위치 확인**

```bash
JUDGE=$(find outputs/ffhq_mobilenet_20261105 -name best.pt | head -1); echo "$JUDGE"
python -c "import torch,sys; s=torch.load(sys.argv[1],map_location='cpu',weights_only=True); print(s['task'], s['groups'])" "$JUDGE"
```

Expected: `ffhq_age_group_classification ['0-2', '3-6', ..., '70-120']`. 이 경로를 이후 `--judge` 값으로 쓴다.

- [ ] **Step 5: Commit**

```bash
git add .gitignore
git commit -m "chore: ignore third_party research checkouts"
```

---

### Task 2: GroupJudge (10구간 분류기를 판정기로 감싸기)

**Files:**
- Create: `age/group_judge.py`
- Test: `tests/test_group_judge.py`

**Interfaces:**
- Consumes:
  - `age.ffhq_data.GROUPS`
  - `age.model.build_model(pretrained, num_classes)`
  - `age.dataset.transform(training)`
  - `age/ffhq.py`가 저장한 체크포인트 dict (`task`, `groups`, `model` 키)
- Produces:
  - `group_of(age: float) -> int`: GROUPS 인덱스. 음수면 `ValueError`
  - `expected_age(probabilities) -> float`: 길이 10, 합이 1인 확률. 아니면 `ValueError`
  - `MIDPOINTS: np.ndarray` (10,)
  - `GroupJudge(checkpoint, device='cpu')`
  - `.probabilities(images: list[PIL.Image]) -> np.ndarray` (N, 10)

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_group_judge.py`:

```python
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from PIL import Image

from age.ffhq_data import GROUPS
from age.group_judge import GroupJudge, expected_age, group_of
from age.model import build_model


class GroupJudgeContracts(unittest.TestCase):
    def test_group_boundaries(self):
        cases = {0: '0-2', 2: '0-2', 2.5: '0-2', 3: '3-6', 9.9: '7-9', 10: '10-14', 19: '15-19',
                 20: '20-29', 49: '40-49', 50: '50-69', 69: '50-69', 70: '70-120', 95: '70-120'}
        for age, group in cases.items():
            with self.subTest(age=age):
                self.assertEqual(GROUPS[group_of(age)], group)
        with self.assertRaises(ValueError):
            group_of(-1)

    def test_expected_age_uses_group_midpoints(self):
        probabilities = np.zeros(len(GROUPS))
        probabilities[5], probabilities[6] = .7, .3
        self.assertAlmostEqual(expected_age(probabilities), 24.5 * .7 + 34.5 * .3)
        for invalid in ([.5, .5], np.full(len(GROUPS), .5)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                expected_age(invalid)

    def test_loads_ffhq_checkpoint_and_returns_probabilities(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'best.pt'
            torch.save(dict(task='ffhq_age_group_classification', groups=list(GROUPS),
                            model=build_model(False, len(GROUPS)).state_dict()), path)
            judge = GroupJudge(path)
        images = [Image.new('RGB', (512, 512), (i * 60, 90, 120)) for i in range(2)]
        probabilities = judge.probabilities(images)
        self.assertEqual(probabilities.shape, (2, len(GROUPS)))
        np.testing.assert_allclose(probabilities.sum(1), 1, rtol=1e-5)

    def test_rejects_other_checkpoints(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'best.pt'
            torch.save(dict(task='exact_age', model=build_model(False).state_dict()), path)
            with self.assertRaises(ValueError):
                GroupJudge(path)


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m unittest tests/test_group_judge.py -v`
Expected: FAIL. `ModuleNotFoundError: No module named 'age.group_judge'`

- [ ] **Step 3: 구현**

`age/group_judge.py`:

```python
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
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m unittest tests/test_group_judge.py -v`
Expected: 4 tests OK

- [ ] **Step 5: Commit**

```bash
git add age/group_judge.py tests/test_group_judge.py
git commit -m "feat: wrap ffhq age-group classifier as a reusable judge"
```

---

### Task 3: SamTeacher (공개 SAM 추론 래퍼)

**Files:**
- Create: `gen/sam_teacher.py`
- Test: `tests/test_gen_sam_teacher.py`

**Interfaces:**
- Consumes:
  - Task 1의 `third_party/SAM/`(`models.psp.pSp`)와 `models/sam/sam_ffhq_aging.pt`
- Produces:
  - `to_input(image: PIL.Image, target_age: int) -> torch.Tensor` (4, 256, 256). 이미지는 [-1, 1], 4번째 채널은 target_age/100. 0~100 범위 밖이면 `ValueError`
  - `to_image(tensor, size=512) -> PIL.Image`
  - `SamTeacher(repo, checkpoint, device='cuda', net=None)`
  - `.age(images: list[PIL.Image], target_ages: list[int], size=512) -> list[PIL.Image]`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_gen_sam_teacher.py`:

```python
import unittest

import torch
from PIL import Image

from gen.sam_teacher import SamTeacher, to_input


class FakeNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, x, randomize_noise=True, resize=True):
        self.calls.append(dict(shape=tuple(x.shape), randomize_noise=randomize_noise, resize=resize))
        return x[:, :3]


class SamTeacherContracts(unittest.TestCase):
    def test_input_has_image_and_age_channel(self):
        tensor = to_input(Image.new('RGB', (512, 512), (255, 0, 128)), 65)
        self.assertEqual(tuple(tensor.shape), (4, 256, 256))
        self.assertAlmostEqual(float(tensor[0].mean()), 1.0, places=5)
        self.assertAlmostEqual(float(tensor[1].mean()), -1.0, places=5)
        self.assertTrue(torch.allclose(tensor[3], torch.full((256, 256), .65)))
        for age in (-1, 101):
            with self.subTest(age=age), self.assertRaises(ValueError):
                to_input(Image.new('RGB', (8, 8)), age)

    def test_age_batches_and_disables_noise(self):
        net = FakeNet()
        teacher = SamTeacher(None, None, device='cpu', net=net)
        outputs = teacher.age([Image.new('RGB', (512, 512), (10, 200, 30))] * 2, [25, 75])
        self.assertEqual(net.calls, [dict(shape=(2, 4, 256, 256), randomize_noise=False, resize=False)])
        self.assertEqual([image.size for image in outputs], [(512, 512)] * 2)
        self.assertEqual(outputs[0].getpixel((5, 5)), (10, 200, 30))
        with self.assertRaises(ValueError):
            teacher.age([Image.new('RGB', (8, 8))], [20, 30])


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m unittest tests/test_gen_sam_teacher.py -v`
Expected: FAIL. `ModuleNotFoundError: No module named 'gen.sam_teacher'`

- [ ] **Step 3: 구현**

`gen/sam_teacher.py`:

```python
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
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m unittest tests/test_gen_sam_teacher.py -v`
Expected: 2 tests OK

- [ ] **Step 5: 실제 SAM으로 한 장 확인 (RTX 4060)**

```bash
python - <<'EOF'
from pathlib import Path
from PIL import Image
from gen.sam_teacher import SamTeacher
source = sorted(Path('data/lora/train/images').glob('*.png'))[0]
teacher = SamTeacher('third_party/SAM', 'models/sam/sam_ffhq_aging.pt', 'cuda')
ages = [5, 45, 85]
outputs = teacher.age([Image.open(source)] * 3, ages)
sheet = Image.new('RGB', (512 * 4, 512))
sheet.paste(Image.open(source).convert('RGB'), (0, 0))
for i, image in enumerate(outputs, 1):
    sheet.paste(image, (512 * i, 0))
Path('outputs/pairs_smoke').mkdir(parents=True, exist_ok=True)
sheet.save('outputs/pairs_smoke/sam_one.png')
print('saved', source.name, ages)
EOF
```

Expected: `outputs/pairs_smoke/sam_one.png`에 원본과 5/45/85세 얼굴이 한 줄로 저장된다. **눈으로 확인한다.** 같은 사람이고 나이가 단계적으로 변해야 한다.

알려진 오류와 대처:
- `RuntimeError ... ninja` 또는 CUDA op 컴파일 실패: `pip install ninja`를 실행하고, `nvcc --version`이 PyTorch CUDA(12.8)와 맞는지 확인한다.
- `_pickle.UnpicklingError: Weights only load failed` (SAM 내부 `torch.load`): 실행 앞에 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`을 붙인다. 공식 체크포인트라 안전하다.
- `TypeError: forward() got an unexpected keyword argument 'resize'`: `third_party/SAM/models/psp.py`의 `forward` 인자 이름을 확인하고, `gen/sam_teacher.py`와 FakeNet 테스트의 키워드를 같은 이름으로 바꾼다.

- [ ] **Step 6: Commit**

```bash
git add gen/sam_teacher.py tests/test_gen_sam_teacher.py
git commit -m "feat: add public SAM teacher wrapper for pair generation"
```

---

### Task 4: 쌍 판정·원본 선택·생성 루프 (`gen/pairs.py` 본체)

**Files:**
- Create: `gen/pairs.py`
- Test: `tests/test_gen_pairs.py`

**Interfaces:**
- Consumes:
  - `group_of`, `expected_age` (Task 2)
  - teacher 객체: `.age(images, target_ages)` (Task 3 또는 가짜)
  - judge 객체: `.probabilities(images)` (Task 2 또는 가짜)
  - faces 객체: `.embedding(image) -> list[float]` (`gen.faces.FaceTools`. 얼굴이 없으면 `gen.FaceNotFoundError` 또는 `ValueError`)
- Produces:
  - `TARGET_AGES`, `MIN_ARCFACE`
  - `judge(source_embedding, output_embedding, probabilities, target_age, min_arcface=MIN_ARCFACE) -> dict`
    - 키: `keep`, `reasons` (`'no_face'`/`'identity'`/`'age_group'`), `arcface`, `predicted_group`, `predicted_probability`, `target_group`
  - `select_sources(records, limit, seed) -> list[dict]`
  - `build(records, teacher, judge_model, faces, output, settings, target_ages=TARGET_AGES, exclude=()) -> None`
  - manifest 한 줄의 키: `pair_id`(`{image_number}_{age:02d}`), `generator`, `source_image_number`, `source_path`, `source_group`, `gender`, `target_age`, `source_expected_age`(원본 얼굴이 있을 때만), `output_path`(원본 얼굴이 없으면 `None`), 그리고 judge의 키들. 원본 얼굴이 없으면 `reasons == ['source_face']`.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_gen_pairs.py`:

```python
import json
import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from age.ffhq_data import GROUPS
from age.group_judge import expected_age, group_of
from gen import FaceNotFoundError
from gen.pairs import TARGET_AGES, build, judge, select_sources


def one_hot(index, value=.9):
    probabilities = np.full(len(GROUPS), (1 - value) / (len(GROUPS) - 1))
    probabilities[index] = value
    return probabilities


class FakeTeacher:
    def age(self, images, target_ages, size=512):
        return [Image.new('RGB', (size, size), (age, age, age)) for age in target_ages]


class ExplodingTeacher:
    def age(self, *args, **kwargs):
        raise AssertionError('finished pairs must not be regenerated')


class FakeJudge:
    """Image shade encodes age; target 85 is judged into the wrong group."""
    def probabilities(self, images):
        shades = [image.getpixel((0, 0))[0] for image in images]
        return np.array([one_hot(0 if shade == 85 else group_of(shade)) for shade in shades])


class FakeFaces:
    """Target 5 has no detectable face; target 75 becomes a different person."""
    def embedding(self, image):
        shade = image.getpixel((0, 0))[0]
        if shade == 5:
            raise FaceNotFoundError('no face')
        vector = np.zeros(512)
        vector[1 if shade == 75 else 0] = 1
        return vector.tolist()


def make_records(root, count=2):
    rows = []
    for index in range(count):
        path = Path(root) / f'{index:05d}.png'
        Image.new('RGB', (512, 512), (30, 30, 30)).save(path)
        rows.append(dict(image_number=f'{index:05d}', path=str(path), age_group='30-39', gender='male'))
    return rows


def read_manifest(output):
    return [json.loads(line) for line in (Path(output) / 'manifest.jsonl').read_text(encoding='utf-8').splitlines()]


class PairJudgeContracts(unittest.TestCase):
    def test_judge_rules(self):
        same = [1.0] + [0.0] * 511
        other = [0.0, 1.0] + [0.0] * 510
        hit = one_hot(group_of(45))
        self.assertTrue(judge(same, same, hit, 45)['keep'])
        verdict = judge(same, other, hit, 45)
        self.assertEqual((verdict['keep'], verdict['reasons'], verdict['arcface']), (False, ['identity'], 0.0))
        self.assertEqual(judge(same, None, hit, 45)['reasons'], ['no_face'])
        self.assertEqual(judge(same, same, one_hot(0), 45)['reasons'], ['age_group'])
        self.assertEqual(judge(same, same, hit, 45)['target_group'], '40-49')
        with self.assertRaises(ValueError):
            judge(same, same, [1.0], 45)

    def test_select_sources_is_stratified_and_deterministic(self):
        records = [dict(image_number=f'{i:05d}', age_group=GROUPS[i % 3]) for i in range(30)]
        picked = select_sources(records, 9, seed=1)
        self.assertEqual(picked, select_sources(records, 9, seed=1))
        self.assertEqual(sorted(Counter(row['age_group'] for row in picked).values()), [3, 3, 3])
        self.assertEqual(len(select_sources(records, 100, seed=1)), 30)


class PairBuildContracts(unittest.TestCase):
    settings = dict(generator='sam', target_ages=list(TARGET_AGES))

    def test_build_writes_manifest_and_resumes(self):
        with TemporaryDirectory() as tmp:
            records, output = make_records(tmp), Path(tmp) / 'pairs'
            build(records, FakeTeacher(), FakeJudge(), FakeFaces(), output, self.settings)
            rows = read_manifest(output)
            self.assertEqual(len(rows), 2 * len(TARGET_AGES))
            reasons = {row['target_age']: row['reasons'] for row in rows if row['source_image_number'] == '00000'}
            self.assertEqual((reasons[5], reasons[75], reasons[85], reasons[45]),
                             (['no_face'], ['identity'], ['age_group'], []))
            self.assertEqual(sum(row['keep'] for row in rows), 12)
            self.assertTrue(all(Path(row['output_path']).is_file() for row in rows))
            self.assertAlmostEqual(rows[0]['source_expected_age'], expected_age(one_hot(group_of(30))))
            self.assertEqual(rows[0]['pair_id'], '00000_05')

            build(records, ExplodingTeacher(), FakeJudge(), FakeFaces(), output, self.settings)
            self.assertEqual(len(read_manifest(output)), 2 * len(TARGET_AGES))
            with self.assertRaises(ValueError):
                build(records, FakeTeacher(), FakeJudge(), FakeFaces(), output, dict(self.settings, generator='edit'))

    def test_excluded_and_faceless_sources(self):
        with TemporaryDirectory() as tmp:
            records, output = make_records(tmp), Path(tmp) / 'pairs'
            Image.new('RGB', (512, 512), (5, 5, 5)).save(records[1]['path'])
            build(records, FakeTeacher(), FakeJudge(), FakeFaces(), output, self.settings, exclude={'00000'})
            rows = read_manifest(output)
            self.assertEqual({row['source_image_number'] for row in rows}, {'00001'})
            self.assertEqual(len(rows), len(TARGET_AGES))
            self.assertTrue(all(row['reasons'] == ['source_face'] and not row['keep'] and row['output_path'] is None
                                for row in rows))


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m unittest tests/test_gen_pairs.py -v`
Expected: FAIL. `ModuleNotFoundError: No module named 'gen.pairs'`

- [ ] **Step 3: 구현**

`gen/pairs.py`:

```python
"""Shared pair dataset P: (source face, target age, generated face), filtered by group judge + ArcFace."""
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from age.common import save_json
from age.ffhq_data import GROUPS
from age.group_judge import expected_age, group_of
from gen import FaceNotFoundError

TARGET_AGES = (5, 15, 25, 35, 45, 55, 65, 75, 85)
MIN_ARCFACE = 0.6


def judge(source_embedding, output_embedding, probabilities, target_age, min_arcface=MIN_ARCFACE):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (len(GROUPS),):
        raise ValueError('Expected one probability per age group')
    predicted, target = int(probabilities.argmax()), group_of(target_age)
    reasons, arcface = [], None
    if output_embedding is None:
        reasons.append('no_face')
    else:
        arcface = float(np.dot(source_embedding, output_embedding))
        if arcface < min_arcface:
            reasons.append('identity')
    if predicted != target:
        reasons.append('age_group')
    return dict(keep=not reasons, reasons=reasons, arcface=arcface, predicted_group=GROUPS[predicted],
                predicted_probability=float(probabilities[predicted]), target_group=GROUPS[target])


def select_sources(records, limit, seed):
    """Round-robin across age groups so rare groups (children, 70+) are not crowded out."""
    rng, buckets = random.Random(seed), {}
    for row in records:
        buckets.setdefault(row['age_group'], []).append(row)
    queues = [buckets[group] for group in GROUPS if group in buckets]
    for rows in queues:
        rng.shuffle(rows)
    picked = []
    while len(picked) < limit and any(queues):
        for rows in queues:
            if rows and len(picked) < limit:
                picked.append(rows.pop())
    return picked


def _embedding(faces, image):
    try:
        return faces.embedding(image)
    except (FaceNotFoundError, ValueError):
        return None


def _write(handle, record):
    handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    handle.flush()


def build(records, teacher, judge_model, faces, output, settings, target_ages=TARGET_AGES, exclude=()):
    """Append one manifest row per (source, target age); rerunning with the same settings resumes."""
    output = Path(output)
    (output / 'images').mkdir(parents=True, exist_ok=True)
    settings_path, manifest = output / 'settings.json', output / 'manifest.jsonl'
    if settings_path.exists() and json.loads(settings_path.read_text(encoding='utf-8')) != settings:
        raise ValueError(f'{output} was built with different settings; use a new --output')
    save_json(settings_path, settings)
    done = set()
    if manifest.exists():
        done = {json.loads(line)['pair_id'] for line in manifest.read_text(encoding='utf-8').splitlines() if line}
    with manifest.open('a', encoding='utf-8') as out:
        for row in records:
            if row['image_number'] in exclude:
                continue
            pending = [age for age in target_ages if f"{row['image_number']}_{age:02d}" not in done]
            if not pending:
                continue
            with Image.open(row['path']) as image:
                source = image.convert('RGB')
            base = dict(generator=settings['generator'], source_image_number=row['image_number'],
                        source_path=str(row['path']), source_group=row['age_group'], gender=row['gender'])
            source_embedding = _embedding(faces, source)
            if source_embedding is None:
                for age in pending:
                    _write(out, dict(base, pair_id=f"{row['image_number']}_{age:02d}", target_age=age,
                                     output_path=None, keep=False, reasons=['source_face']))
                continue
            generated = teacher.age([source] * len(pending), pending)
            probabilities = judge_model.probabilities([source] + generated)
            source_age = expected_age(probabilities[0])
            for age, image, probs in zip(pending, generated, probabilities[1:]):
                pair_id = f"{row['image_number']}_{age:02d}"
                path = output / 'images' / f'{pair_id}.png'
                image.save(path)
                verdict = judge(source_embedding, _embedding(faces, image), probs, age)
                _write(out, dict(base, pair_id=pair_id, target_age=age, source_expected_age=source_age,
                                 output_path=str(path), **verdict))
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m unittest tests/test_gen_pairs.py -v`
Expected: 4 tests OK

- [ ] **Step 5: Commit**

```bash
git add gen/pairs.py tests/test_gen_pairs.py
git commit -m "feat: build filtered pair dataset manifest with resume"
```

---

### Task 5: 요약 리포트, contact sheet, CLI

**Files:**
- Modify: `gen/pairs.py` (`report`, `_contact_sheet`, `main` 추가)
- Modify: `tests/test_gen_pairs.py` (리포트 테스트 추가)

**Interfaces:**
- Consumes:
  - Task 4의 `build`와 manifest 형식
  - `age.ffhq_data.load_records(root, provenance) -> dict[split, list[row]]`
  - `age.common.sha256(path)`
  - `GroupJudge`, `SamTeacher`, `gen.faces.FaceTools(model_root, device)`
- Produces:
  - `report(output, sheet_sources=8, thumb=128) -> dict`: `total`, `kept`, `reasons`, `by_target_group[group] = {total, kept, mean_arcface}`. `summary.json`과 `contact_sheet.png`를 쓴다.
  - CLI 하위 명령: `python -m gen.pairs build ...` / `python -m gen.pairs report --output ...`

- [ ] **Step 1: 실패하는 테스트 추가**

`tests/test_gen_pairs.py`의 import 줄을 다음으로 바꾼다.

```python
from gen.pairs import TARGET_AGES, build, judge, report, select_sources
```

`PairBuildContracts` 클래스 안에 테스트를 추가한다.

```python
    def test_report_summarises_by_target_group(self):
        with TemporaryDirectory() as tmp:
            records, output = make_records(tmp), Path(tmp) / 'pairs'
            build(records, FakeTeacher(), FakeJudge(), FakeFaces(), output, self.settings)
            summary = report(output)
            self.assertEqual((summary['total'], summary['kept']), (18, 12))
            self.assertEqual(summary['reasons'], {'no_face': 2, 'identity': 2, 'age_group': 2})
            self.assertEqual(summary['by_target_group']['70-120'], dict(total=4, kept=0, mean_arcface=0.5))
            self.assertEqual(summary['by_target_group']['40-49']['kept'], 2)
            self.assertTrue((output / 'summary.json').is_file())
            with Image.open(output / 'contact_sheet.png') as sheet:
                self.assertEqual(sheet.size, (128 * (len(TARGET_AGES) + 1), 128 * 2))
```

`mean_arcface` 0.5의 근거: 70-120 구간에는 목표 75(ArcFace 0)와 85(ArcFace 1)가 원본마다 하나씩 있다.

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m unittest tests/test_gen_pairs.py -v`
Expected: FAIL. `ImportError: cannot import name 'report'`

- [ ] **Step 3: 구현**

`gen/pairs.py` 상단 import를 다음으로 바꾼다.

```python
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from age.common import save_json, sha256
from age.ffhq_data import GROUPS, load_records
from age.group_judge import expected_age, group_of
from gen import FaceNotFoundError
```

파일 끝에 다음을 추가한다.

```python
def _read_manifest(output):
    return [json.loads(line) for line in (Path(output) / 'manifest.jsonl').read_text(encoding='utf-8').splitlines()
            if line]


def _contact_sheet(rows, path, sources, thumb):
    """Source in column 0, then each target age; rejected outputs get a red frame."""
    by_source = {}
    for row in rows:
        by_source.setdefault(row['source_image_number'], []).append(row)
    chosen = list(by_source.values())[:sources]
    ages = sorted({row['target_age'] for row in rows})
    sheet = Image.new('RGB', (thumb * (len(ages) + 1), thumb * max(1, len(chosen))), 'white')
    draw = ImageDraw.Draw(sheet)
    for y, items in enumerate(chosen):
        with Image.open(items[0]['source_path']) as image:
            sheet.paste(image.convert('RGB').resize((thumb, thumb)), (0, y * thumb))
        for item in items:
            if item['output_path'] is None:
                continue
            x = (ages.index(item['target_age']) + 1) * thumb
            with Image.open(item['output_path']) as image:
                sheet.paste(image.convert('RGB').resize((thumb, thumb)), (x, y * thumb))
            if not item['keep']:
                draw.rectangle([x, y * thumb, x + thumb - 1, (y + 1) * thumb - 1], outline='red', width=4)
    sheet.save(path)


def report(output, sheet_sources=8, thumb=128):
    output = Path(output)
    rows = _read_manifest(output)
    by_group = {}
    for row in rows:
        group = by_group.setdefault(GROUPS[group_of(row['target_age'])], dict(total=0, kept=0, scores=[]))
        group['total'] += 1
        group['kept'] += int(row['keep'])
        if row.get('arcface') is not None:
            group['scores'].append(row['arcface'])
    for group in by_group.values():
        scores = group.pop('scores')
        group['mean_arcface'] = float(np.mean(scores)) if scores else None
    summary = dict(total=len(rows), kept=sum(int(row['keep']) for row in rows),
                   reasons=dict(Counter(reason for row in rows for reason in row['reasons'])),
                   by_target_group=by_group)
    save_json(output / 'summary.json', summary)
    _contact_sheet(rows, output / 'contact_sheet.png', sheet_sources, thumb)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description='Pair dataset P: SAM teacher + age-group judge + ArcFace filter.')
    commands = parser.add_subparsers(dest='command', required=True)
    build_args = commands.add_parser('build')
    build_args.add_argument('--data-root', default='data/lora')
    build_args.add_argument('--provenance', default='data/reports/ffhq_aging_samples.jsonl')
    build_args.add_argument('--limit', type=int, default=2000)
    build_args.add_argument('--seed', type=int, default=42)
    build_args.add_argument('--exclude', type=Path, help='Text file of image_numbers to skip (identity-leak candidates)')
    build_args.add_argument('--sam-repo', type=Path, default=Path('third_party/SAM'))
    build_args.add_argument('--sam-checkpoint', type=Path, default=Path('models/sam/sam_ffhq_aging.pt'))
    build_args.add_argument('--judge', type=Path, required=True, help='best.pt from age.ffhq train')
    build_args.add_argument('--model-root', default='models/insightface')
    build_args.add_argument('--device', default='cuda')
    build_args.add_argument('--output', type=Path, required=True)
    report_args = commands.add_parser('report')
    report_args.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'build':
        from age.group_judge import GroupJudge
        from gen.faces import FaceTools
        from gen.sam_teacher import SamTeacher
        records = select_sources(load_records(args.data_root, args.provenance)['train'], args.limit, args.seed)
        exclude = set(args.exclude.read_text(encoding='utf-8').split()) if args.exclude else set()
        settings = dict(generator='sam', limit=args.limit, seed=args.seed, target_ages=list(TARGET_AGES),
                        min_arcface=MIN_ARCFACE, excluded=len(exclude),
                        sam_checkpoint_sha256=sha256(args.sam_checkpoint), judge_sha256=sha256(args.judge))
        build(records, SamTeacher(args.sam_repo, args.sam_checkpoint, args.device),
              GroupJudge(args.judge, args.device), FaceTools(args.model_root, 'cpu'),
              args.output, settings, exclude=exclude)
    print(json.dumps(report(args.output), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: 전체 테스트 통과 확인**

Run: `python -m unittest tests/test_gen_pairs.py tests/test_gen_sam_teacher.py tests/test_group_judge.py -v`
Expected: 11 tests OK

Run: `python -m unittest discover -s tests -v`
Expected: 기존 테스트를 포함해 실패 0개. 모델이 필요한 기존 테스트는 skip으로 처리된다.

- [ ] **Step 5: Commit**

```bash
git add gen/pairs.py tests/test_gen_pairs.py
git commit -m "feat: add pair dataset report, contact sheet and CLI"
```

---

### Task 6: 파일럿 (원본 100장) → 검토 → 본 실행 (원본 2,000장)

**Files:**
- 코드 변경 없음
- 결과는 `outputs/pairs/` 아래에 저장 (Git 제외)

**Interfaces:**
- Consumes:
  - Task 5의 CLI
  - Task 1 Step 4에서 확인한 `best.pt` 경로 (아래에서는 `$JUDGE`)
- Produces:
  - `outputs/pairs/sam_v1/{manifest.jsonl, images/, summary.json, contact_sheet.png}`
  - M1-B 계획이 이 manifest에서 `keep == true`인 행만 학습에 쓴다.

- [ ] **Step 1: 파일럿 실행**

```bash
JUDGE=$(find outputs/ffhq_mobilenet_20261105 -name best.pt | head -1)
python -m gen.pairs build --limit 100 --judge "$JUDGE" --device cuda --output outputs/pairs/sam_pilot
```

Expected: 900행 manifest와 요약 JSON이 출력된다. 완료 시간을 기록해 두면 본 실행 시간을 추정할 수 있다.

- [ ] **Step 2: 검토 게이트 (팀과 함께, 통과해야 다음 단계 진행)**

`outputs/pairs/sam_pilot/summary.json`과 `contact_sheet.png`를 열어서 다음을 확인한다.

| 확인 항목 | 통과 기준 | 미달일 때 |
|---|---|---|
| 전체 통과율 `kept / total` | ≥ 0.56 (원본 2,000 × 9 × 0.56 ≈ 1만 쌍) | 본 실행 `--limit`을 `ceil(10000 / (9 × 통과율))`로 늘린다 |
| 구간별 `kept` | 모든 목표 구간이 0보다 큼 | 0인 구간(예: 0-2, 70-120)을 기록해 두고, M1-B에서 보충 데이터(AAF, MegaAge-Asian)로 채운다 |
| `reasons` 중 `identity` 비율 | 목표 구간별 `mean_arcface`가 0.6 근처 | 큰 나이 변화(5↔85세)에서만 낮으면 정상이다. 전부 낮으면 Task 3 Step 5의 입력 정렬을 다시 확인한다 |
| contact sheet 육안 확인 | 통과(테두리 없음) 칸이 같은 사람이고 나이가 단계적으로 변함 | 통과 칸이 이상하면 판정기 문제다. 분류기 출력과 입력 전처리를 확인한다 |

- [ ] **Step 3: 본 실행**

```bash
python -m gen.pairs build --limit 2000 --judge "$JUDGE" --device cuda --output outputs/pairs/sam_v1
```

Expected: 18,000행 manifest가 생기고, `kept`가 1만 이상이다. 중간에 끊기면 **같은 명령을 다시 실행해서 이어 간다.** 디스크는 PNG 512px 기준 약 7GB가 필요하다.

- [ ] **Step 4: 결과 기록 commit**

`docs/m1-pair-dataset-results.md`를 만들어 다음 내용을 적는다.
- 실행 명령
- 소요 시간
- `summary.json` 전문
- contact sheet 관찰 3~5줄
- 미달 구간

```bash
git add docs/m1-pair-dataset-results.md
git commit -m "docs: record m1 sam pair dataset pilot and full run"
```
