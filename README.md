# 2026-2-HAI-ProjectTeam3

얼굴 사진에서 나이를 추정하고, 같은 인물의 다양한 연령대 모습을 생성하는 **연령 변환 포토부스** 프로젝트입니다.

## 주요 기능

| 기능 | 모델·학습 방식 | 현재 상태 |
|---|---|---|
| 나이 추정 | MobileNetV3-Large를 AI Hub 데이터로 전체 파인튜닝 | 전처리·학습·평가·ONNX 코드 구현, 실제 학습 전 |
| 연령 구간 추정 | MobileNetV3-Large를 로컬 FFHQ-Aging의 원본 10개 구간으로 파인튜닝 | 5 epoch 실학습 완료, test 100장 정확도 64%·macro-F1 0.640; 정확한 나이 회귀와 별개 |
| 연령 변환 | Realistic Vision + IP-Adapter FaceID + Age LoRA | 단계별 학습·캡션 비교 완료, 연령 변환 품질 개선 미확인 |
| 최소 연동 | 나이 추정 → 목표 나이 → 생성 3장 + 원본 | 호출 흐름·2×2 합성 구현, Mock으로 확인 |

전체 흐름: **얼굴 검출·정렬 → 나이 추정 → 목표 연령 설정 → 연령 변환 이미지 생성**

## Vast.ai 개발 환경

네 프로젝트 브랜치에서 사용할 GPU 개발 이미지는 [Vast.ai 설정 안내](docs/vast-ai.md)를 참고하세요. GitHub Actions에서 GHCR에 이미지를 게시하는 과정부터 Vast 템플릿, SSH 접속, 브랜치별 환경 선택, 모델·데이터 전송까지 설명합니다. 실제 게시 여부와 고정 이미지 주소는 해당 Actions 실행 결과로 확인합니다.

## 파일 구조

```text
age/                         # 나이 추정 파이프라인
  acquire.py                 # 공개 데이터 다운로드 유틸리티 (AI Hub 학습에는 불필요)
  setup_models.py            # 얼굴 검출·정렬용 모델 다운로드
  prep.py, face.py            # 데이터 준비·얼굴 검출·정렬
  model.py, dataset.py        # MobileNet 모델·학습 데이터 로더
  train.py, evaluate.py       # 학습·성능 평가
  ffhq.py, ffhq_data.py       # 로컬 FFHQ 원본 연령 구간 학습·평가·무결성 검사
  export_onnx.py, infer.py    # ONNX 내보내기·나이 추론
  smoke.py                   # 합성 입력으로 학습·ONNX 흐름 검증
  common.py, __init__.py      # 공통 유틸리티·패키지 초기화
gen/                         # 생성·연동 파이프라인
  pipeline.py                # 모델 로딩·age_shift·Mock
  faces.py                   # 주 얼굴 선택·마스크·원본 합성·ArcFace 임베딩
  eval.py                    # 3개 ablation·평가 CSV·실패/시간 기록
  compare_lora.py             # 고정 validation의 체크포인트 비교
  quality_experiment.py       # 조건별 생성 실험·중단/재개
  analyze_quality.py          # 짝 비교 지표·비교 이미지
  record_command.py           # 실행 명령·로그·종료 상태 기록
  validate_lora_data.py       # 준비된 FFHQ 이미지·캡션·출처 검사
  audit_lora_split.py         # 분할 간 동일인 후보 조사
  prepare_lora_stages.py      # 단계별 실험용 학습 복사본 준비
  prepare_quality_data.py     # 캡션·샘플링·crop 비교 자료 준비
  prepare_diagnostic_masks.py # 진단용 편집·보호 마스크 준비
  train_lora.py               # 공식 Diffusers LoRA 학습기 실행·검증 래퍼
  lora.smoke.json             # RTX 4060 8GB 소규모 학습 설정
  booth.py, smoke.py          # 원본+생성 3장 연동·모델 없는 종단 검증
  config.example.json        # 실제 실행용 로컬 모델 경로 예시
tests/                       # 나이 추정·얼굴 전처리·생성/연동 테스트
docs/                        # 구현 상태·서버 안내·전체 설계
data/README.md               # 데이터 준비 안내 (데이터는 별도)
requirements.txt             # 기본 의존성
requirements-insightface.txt # 전체 InsightFace SDK 사용 시 선택 의존성
requirements-generation.txt # 실제 생성에만 필요한 선택 의존성
requirements-lora.txt        # 생성 환경에 추가하는 LoRA 학습 의존성
.gitignore                   # 데이터·가중치·실행 결과 제외
README.md                    # 프로젝트 소개·실행 안내
```

Git 저장소에는 데이터·모델 가중치·생성 결과를 포함하지 않습니다. HTTP 서버·UI·하드웨어 제어는 미구현입니다.

## 실험 현황 (2026-11-01)

- 10월 26일: 12-step 실행 검증 후 별도 실험에서 100/300-step LoRA와 기준선을 비교했습니다. 수정 전 400장, 얼굴 선택·합성 수정 후 예비 30장과 본 검증 200장을 생성했습니다.
- 11월 1일: 기존 캡션과 연령 구간 캡션으로 각각 2,100회 업데이트를 학습하고, 40개 입력·2개 목표·2개 seed·4개 조건의 640행을 비교했습니다. 실제 생성 480장과 원본 복사 대조 160장입니다.
- 새 LoRA의 일관된 연령 변환 개선은 확인하지 못해 운영 모델을 유지합니다. 충분한 나이 변화와 인물·표정 보존의 절충이 남아 있으며, 정식 나이 MAE와 사람 평가는 미완료입니다.

단계별 결과는 [학습 비교](docs/age-lora-stages.md)와 [얼굴 선택·배경 보존 수정](docs/age-lora-recovery.md)에 있습니다. 11월 1일 비교 자료는 로컬 `outputs/quality_improvement_20261101/caption_ablation/analysis/`에 보관합니다.

## Mock 실행

Pillow만 있으면 GPU·데이터셋·모델 다운로드 없이 실행할 수 있습니다. 프로젝트 루트에서 실행합니다.

```bash
python -m pip install "Pillow>=10,<13"
python -m gen.smoke --output outputs/gen_smoke
```

`outputs/gen_smoke/booth/four_cut.png`는 **연령 변환 결과가 아닌 합성 입력과 Mock 표시 이미지**입니다. `evaluation/results.csv`에는 3개 모드 × 목표 나이 3개의 9행이 기록됩니다. Mock의 나이 MAE·인물 유사도·얼굴 검출 실패율은 빈 값이며, 실행 상태는 `status.json`에 `real_model_validation=false`로 저장합니다. 재실행은 새로운 `--output` 경로를 지정합니다.

임의 입력 파일의 호출 흐름만 확인하려면:

```bash
python -m gen.booth --mock --image outputs/gen_smoke/synthetic.png --mock-age 25 --output outputs/booth_demo
```

실제 모델 경로 설정과 평가·연동 API는 [서버 실행 안내의 생성 항목](docs/server-setup.md#생성-파이프라인과-최소-연동)을 참고하세요.

## 나이 추정 실행

### 로컬 FFHQ-Aging 연령 구간 학습

로컬 `data/lora/`의 이미지를 그대로 사용하고, `data/reports/ffhq_aging_samples.jsonl`의 원본 `age_group`을 이미지 ID로 연결합니다. 캡션을 정수 나이로 바꾸지 않습니다. 기존 train/val/test 분할과 원본 파일은 수정하지 않으며 이미지 해시·디코딩·분할 간 중복을 검사합니다. 데이터가 이동했다면 `--data-root`와 `--provenance`로 실제 경로를 지정합니다.

```bash
python -m age.ffhq train --data-root data/lora --provenance data/reports/ffhq_aging_samples.jsonl --device cuda --epochs 5 --batch-size 64 --workers 4 --output outputs/age/ffhq_groups
python -m age.ffhq evaluate --data-root data/lora --provenance data/reports/ffhq_aging_samples.jsonl --checkpoint outputs/age/ffhq_groups/best.pt --device cuda --output outputs/eval/ffhq_groups
```

ImageNet V2 초기화, 학습 분포의 역빈도 가중 cross entropy, AdamW/cosine/AMP를 사용합니다. val macro-F1로 체크포인트를 선택하고 동률이면 val loss를 비교합니다. 별도 평가 명령은 test만 평가하며 정확도·macro-F1·구간별 지표·성별 지표·혼동행렬과 train 다수 클래스 기준선을 저장합니다. 모든 출력 디렉터리는 새 경로여야 합니다.

라벨은 `0-2`, `3-6`, `7-9`, `10-14`, `15-19`, `20-29`, `30-39`, `40-49`, `50-69`, `70-120`입니다. **이 모델은 10개 구간 분류기이며 정확한 나이 MAE나 CS@5를 제공하지 않습니다.** 기존 101-bin `age.infer`/`age.export_onnx` 및 포토부스의 숫자 나이 입력에 직접 연결할 수 없습니다. FFHQ 정렬 이미지에서 학습하므로 SCRFD 재정렬 모델과도 구분합니다. 이미지 ID·해시 분리는 확인하지만 인물 단위 분리는 보장하지 않습니다. 데이터의 비상업·저작자 표시·동일조건 공유 조건을 유지합니다.

2026-11-05 실제 실행: 기존 8,163/400/100 분할, ImageNet V2, 5 epoch, batch 64, LR 3e-4, seed 42, RTX 4060에서 optimizer 갱신 639회·AMP skip 1회. val macro-F1로 선택된 5 epoch 모델의 val 정확도는 71%, test 정확도는 64%, test macro-F1은 0.639989입니다. train 다수 클래스 기준선의 test 정확도는 10%입니다. 각 test 구간은 10장뿐이므로 일반화 성능을 확정하지 않습니다. 명령·원본 라벨 대조·체크포인트·지표·혼동행렬은 로컬 `outputs/ffhq_mobilenet_20261105/`에 보관하며 Git에는 포함하지 않습니다.

### 정확한 나이 라벨을 사용하는 AI Hub 경로

CUDA 지원 PyTorch와 가상환경을 준비한 뒤 실행합니다. AI Hub 이미지와 JSON은 `data/raw/aihub/`에 별도로 배치합니다.

```bash
pip install -r requirements.txt
python -m age.setup_models
python -m age.prep --source aihub --raw-root data/raw/aihub --device cpu
python -m age.train --device cuda
```

학습 결과는 `outputs/age/aihub/`에 저장됩니다. 데이터·가중치·학습 결과는 Git에 포함하지 않습니다.

## Age LoRA 데이터

로컬 FFHQ-Aging 준비본은 `data/lora/`에 배치했습니다. `train/` 8,163장, `val/` 400장, `test/` 100장이며 각 폴더에 `images/`와 `metadata.jsonl`이 있습니다. RGB 512×512 이미지와 연령 그룹·성별 캡션은 **연령 변환용 Age LoRA 학습**에 사용하며, AI Hub 기반 `age.train` 입력과는 별개입니다.

학습에는 `data/lora/train`만 사용하고, 설정 선택에는 val, 최종 평가에는 test를 남깁니다. 기존 생성 평가 코드용 목록은 `data/gen_eval/ffhq_val.csv`와 `data/gen_eval/ffhq_test.csv`이며 이미지를 복제하지 않고 참조합니다. `gen.train_lora`는 Diffusers 0.35.1 공식 예제를 해시 고정해 실행하며, 이미 있는 데이터와 Realistic Vision V5.1 로컬 모델만 읽습니다.

최초 12-step 학습·재개·재로딩 절차는 [실행 검증 보고서](docs/age-lora-validation.md)에 기록했습니다. 이후의 학습·비교 결과는 위 실험 현황을 참고하세요.

출처·디렉터리 계약·무결성 검사·동일인 누수 제한은 [데이터 안내](data/README.md#ffhq-aging-배치와-사용), 실행 방법은 [서버 안내](docs/server-setup.md#준비된-ffhq-aging-데이터)를 참고하세요. 이 데이터와 평가 목록은 Git에서 제외되므로 서버에 별도로 전송해야 합니다.

## 테스트

전체 테스트에는 NumPy·OpenCV 등 `requirements.txt`의 기본 의존성이 필요합니다. 실제 생성 런타임을 사용하는 검사는 `requirements-generation.txt`의 의존성도 필요합니다. Pillow만으로 실행 가능한 Mock CLI와 전체 테스트 환경은 구분합니다.

```bash
python -m unittest discover -s tests -v
python -m age.smoke
```

`tests/`는 코드 동작을 검사하며, 모델·데이터가 필요한 일부 검사는 파일이 없으면 건너뜁니다. `age.smoke`는 합성 입력으로 학습·ONNX 변환을 검증합니다. 실제 나이 추정 성능은 학습 후 별도 test 데이터로 평가합니다. smoke 재실행 시에는 새로운 `--output` 경로를 지정합니다.

자세한 설치·평가·ONNX 사용법은 [서버 실행 안내](docs/server-setup.md), 전체 설계는 [설계 문서](docs/superpowers/specs/2026-09-27-age-photobooth-models-design.md)를 참고하세요.
