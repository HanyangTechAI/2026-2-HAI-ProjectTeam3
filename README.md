# 2026-2-HAI-ProjectTeam3

얼굴 사진에서 나이를 추정하고, 같은 인물의 다양한 연령대 모습을 생성하는 **연령 변환 포토부스** 프로젝트입니다.

## 주요 기능

| 기능 | 모델·학습 방식 | 현재 상태 |
|---|---|---|
| 나이 추정 | MobileNetV3-Large를 AI Hub 데이터로 전체 파인튜닝 | 전처리·학습·평가·ONNX 코드 구현, 실제 학습 전 |
| 연령 변환 | Realistic Vision + IP-Adapter FaceID + Age LoRA | 로딩·생성·평가 코드와 Mock 검증 구현, 실제 가중치 미검증 |
| 최소 연동 | 나이 추정 → 목표 나이 → 생성 3장 + 원본 | orchestration·2×2 합성 구현, Mock으로 확인 |

전체 흐름: **얼굴 검출·정렬 → 나이 추정 → 목표 연령 설정 → 연령 변환 이미지 생성**

## 파일 구조

```text
age/                         # 나이 추정 파이프라인
  acquire.py                 # 공개 데이터 다운로드 유틸리티 (AI Hub 학습에는 불필요)
  setup_models.py            # 얼굴 검출·정렬용 모델 다운로드
  prep.py, face.py            # 데이터 준비·얼굴 검출·정렬
  model.py, dataset.py        # MobileNet 모델·학습 데이터 로더
  train.py, evaluate.py       # 학습·성능 평가
  export_onnx.py, infer.py    # ONNX 내보내기·나이 추론
  smoke.py                   # 합성 입력으로 학습·ONNX 흐름 검증
  common.py, __init__.py      # 공통 유틸리티·패키지 초기화
gen/                         # 중간 발표용 생성·연동 파이프라인
  pipeline.py                # 실제 모델 로딩 경로·age_shift·명시적 Mock
  faces.py                   # 기존 얼굴 정렬 재사용·ArcFace 임베딩
  eval.py                    # 3개 ablation·평가 CSV·실패/시간 기록
  booth.py, smoke.py          # 원본+생성 3장 연동·모델 없는 종단 검증
  config.example.json        # 실제 실행용 로컬 모델 경로 예시
tests/                       # 나이 추정·얼굴 전처리·생성/연동 테스트
docs/                        # 구현 상태·서버 안내·전체 설계
data/README.md               # 데이터 준비 안내 (데이터는 별도)
requirements.txt             # 기본 의존성
requirements-insightface.txt # 전체 InsightFace SDK 사용 시 선택 의존성
requirements-generation.txt # 실제 생성에만 필요한 선택 의존성
.gitignore                   # 데이터·가중치·실행 결과 제외
README.md                    # 프로젝트 소개·실행 안내
```

실제 학습 데이터·정식 나이 모델·Age LoRA·생성 가중치는 포함하지 않습니다. 생성 코드의 실제 모델 결합, 이미지 품질, 처리 시간은 아직 검증하지 않았습니다. HTTP 서버·UI·하드웨어 제어는 구현하지 않았습니다.

## 중간 발표용 Mock 실행

Pillow만 있으면 GPU·데이터셋·모델 다운로드 없이 실행할 수 있습니다. 프로젝트 루트에서 실행합니다.

```bash
python -m pip install "Pillow>=10,<13"
python -m unittest discover -s tests -p "test_gen_*.py" -v
python -m gen.smoke --output outputs/gen_smoke
```

`outputs/gen_smoke/booth/four_cut.png`는 **연령 변환 결과가 아닌 합성 입력과 Mock 표시 이미지**입니다. `evaluation/results.csv`에는 3개 모드 × 목표 나이 3개의 9행이 기록됩니다. Mock의 나이 MAE·인물 유사도·얼굴 검출 실패율은 빈 값이며, 실행 상태는 `status.json`에 `real_model_validation=false`로 저장합니다. 재실행은 새로운 `--output` 경로를 지정합니다.

임의 입력 파일의 호출 흐름만 확인하려면:

```bash
python -m gen.booth --mock --image outputs/gen_smoke/synthetic.png --mock-age 25 --output outputs/booth_demo
```

실제 모델 경로 설정과 평가·연동 API는 [서버 실행 안내의 생성 항목](docs/server-setup.md#생성-파이프라인과-최소-연동)을 참고하세요.

## 나이 추정 실행

CUDA 지원 PyTorch와 가상환경을 준비한 뒤 실행합니다. AI Hub 이미지와 JSON은 `data/raw/aihub/`에 별도로 배치합니다.

```bash
pip install -r requirements.txt
python -m age.setup_models
python -m age.prep --source aihub --raw-root data/raw/aihub --device cpu
python -m age.train --device cuda
```

학습 결과는 `outputs/age/aihub/`에 저장됩니다. 데이터·가중치·학습 결과는 Git에 포함하지 않습니다.

## 테스트

나이 추정까지 포함한 전체 테스트는 기본 의존성 설치 후 실행합니다. 위의 `test_gen_*.py`는 별도로 Pillow만 사용합니다.

```bash
python -m unittest discover -s tests -v
python -m age.smoke
```

`tests/`는 코드 동작을 검사하며, 모델·데이터가 필요한 일부 검사는 파일이 없으면 건너뜁니다. `age.smoke`는 합성 입력으로 학습·ONNX 변환을 검증합니다. 실제 나이 추정 성능은 학습 후 별도 test 데이터로 평가합니다. smoke 재실행 시에는 새로운 `--output` 경로를 지정합니다.

자세한 설치·평가·ONNX 사용법은 [서버 실행 안내](docs/server-setup.md), 전체 설계는 [설계 문서](docs/superpowers/specs/2026-09-27-age-photobooth-models-design.md)를 참고하세요.
