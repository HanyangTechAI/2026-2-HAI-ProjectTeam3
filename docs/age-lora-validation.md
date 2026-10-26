# Age LoRA 로컬 GPU 검증 (2026-10-26)

## 결과와 해석

기존 FFHQ-Aging 준비본과 기존 실추론에 성공한 **동일한 Realistic Vision V5.1 로컬 모델**로 실제 학습을 실행했다. 10 optimization step 후 checkpoint-10에서 2 step을 재개해 **총 12 step**을 적용했다. 저장된 LoRA의 256개 텐서를 새 베이스 모델에 재로딩해 전부 일치하는지 검사했고, `gen.pipeline.AgeShiftPipeline(mode='faceid_age')`로 실제 RGB 512×512 이미지 1장을 생성했다.

**학습 파이프라인·저장·로딩 경로 검증일 뿐, 품질 좋은 Age LoRA나 연령 변환 정확도를 입증한 결과가 아니다.** 두 초기 오류와 스킵·미실행 항목도 아래에 구분한다. 데이터·기존 모델은 다시 다운로드하지 않았다. 설치한 학습 패키지와 공식 Python 예제 소스는 이미지 데이터 다운로드와 별개다.

## 데이터 검증

실제 경로: `/home/uzer/Programme/2026-2-HAI-ProjectTeam3/data/lora`.

| Split | images/·metadata.jsonl | 이미지 | JSONL 행 | 유효 파일 참조 | 비어 있지 않은 캡션 | 512×512 RGB 완전 디코딩 | 오류 |
|---|---|---:|---:|---:|---:|---:|---:|
| train | 있음 | 8,163 | 8,163 | 8,163 | 8,163 | 8,163 | 0 |
| val | 있음 | 400 | 400 | 400 | 400 | 400 | 0 |
| test | 있음 | 100 | 100 | 100 | 100 | 100 | 0 |

`gen.validate_lora_data`는 파일 경로 이탈, 중복 참조, 미참조 이미지, 이미지 ID·파일 SHA-256 중복, 비정상 캡션, 손상·크기·모드도 검사한다. **수정한 데이터는 0건**이다. 기존 분할·캡션·이미지는 보존했다. `data/reports/ffhq_aging_samples.jsonl`의 8,663개 출처 기록도 파일 해시·split·캡션·저작자·라이선스, `age_group_confidence > 0.8` 및 `abs(head_yaw) < 30`을 대조해 통과했다. 인물 단위 중복 검사는 아니다.

10개 연령 그룹 × 2개 성별에 대해 val/test는 각각 셀당 20/5장이다. train의 상한 500장에 미달하는 셀은 다음과 같다. 나머지 6개 연령 그룹은 성별마다 500장이다.

| 원본 연령 그룹 | 남성 train | 여성 train | 500 대비 부족: 남성/여성 |
|---|---:|---:|---|
| 7~9 | 278 | 305 | 222 / 195 |
| 10~14 | 186 | 215 | 314 / 285 |
| 15~19 | 238 | 369 | 262 / 131 |
| 70~120 | 313 | 259 | 187 / 241 |

train 합계는 남성 4,015, 여성 4,148장이다. 균형 목표 10,000장보다 1,837장 부족하며, 복제로 채우지 않았다. val/test는 무결성 검사에만 읽었고 학습이나 이번 샘플 생성에는 사용하지 않았다.

## 환경과 설정

| 항목 | 실제 사용 값 |
|---|---|
| GPU | NVIDIA GeForce RTX 4060, nvidia-smi 총 8,188MiB |
| 드라이버 / PyTorch CUDA | 615.78.08 / 12.8 |
| Python / torch / torchvision | 3.11.16 / 2.8.0 / 0.23.0 |
| diffusers / transformers / peft | 0.35.1 / 4.57.6 / 0.17.1 |
| accelerate / datasets | 1.15.0 / 4.2.0 |
| 베이스 | `models/generation/realistic-vision-v5.1` |
| 베이스 출처 revision | `SG161222/Realistic_Vision_V5.1_noVAE`, `1e9f017a7b1eaefb63a1900ea6c5953d2739fd21` |
| 학습 입력 | `data/lora/train`, ImageFolder 8,163행; 모든 경로가 train/images 안인지 재검사 |
| 해상도 / rank / alpha | 512 / 16 / 16 (공식 예제의 alpha=rank) |
| batch / gradient accumulation | 1 / 4, 유효 batch 4 |
| precision / checkpointing | FP16 / 켬; 학습 대상 LoRA 텐서는 FP32 |
| attention | PyTorch SDPA `AttnProcessor2_0`; xformers 미설치·미사용 |
| optimizer / LR | AdamW / 1e-4, weight decay 0.01, max grad norm 1.0 |
| scheduler / warmup | constant / 0; 짧은 smoke용이며 기존 장기 cosine/warmup 설계를 대체하지 않음 |
| 학습 대상 | UNet attention의 to_q/to_k/to_v/to_out.0, 3,188,736개 파라미터 |
| 고정 대상 | UNet 기본 가중치, VAE, text encoder |
| 증강 / seed | center crop + random flip / 42 |

`requirements-lora.txt`는 생성 의존성과 `datasets==4.2.0`을 정의한다. 설치 시 `fsspec`가 2026.9.0에서 datasets 요구 범위인 2025.9.0으로 조정됐고, 기존 torch/diffusers/transformers/peft는 유지했다. `uv pip check --python .venv/bin/python`은 설치 전후 모두 통과했다. 전체 버전은 출력의 `packages.txt`에 있다. 8-bit Adam과 xformers를 추가 설치할 필요는 없었다.

## 실행 절차

프로젝트 루트에서 실행한다. 이미 검증한 CUDA `.venv`와 로컬 모델·데이터를 사용한다. 아래 출력 디렉터리는 이번 실행으로 이미 존재하므로 새 시험은 다른 경로를 사용한다. 일반 학습은 기존 출력 덮어쓰기를 거부하고, 재개만 명시적으로 같은 경로를 사용한다.

```bash
uv pip install --python .venv/bin/python -r requirements-lora.txt
uv pip check --python .venv/bin/python

.venv/bin/python -m gen.validate_lora_data \
  --root data/lora \
  --provenance data/reports/ffhq_aging_samples.jsonl \
  --output outputs/age_lora_validation_20261026/data

# 이미지나 모델이 아닌 공식 Python 예제 소스만 받아 해시를 검사한다.
.venv/bin/python -m gen.train_lora --prepare-script

# 실제 최초 시도: tracker 초기화 실패, 0 step. 로그 보존.
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m gen.train_lora \
  --config gen/lora.smoke.json \
  --output outputs/age_lora_validation_20261026/train

# 호환 처리 후 실제 재시도: 10 step와 checkpoint-5/10 저장.
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m gen.train_lora \
  --config gen/lora.smoke.json \
  --output outputs/age_lora_validation_20261026/train_retry

# 오프라인 재로딩 파일명 수정 후 실제 학습 상태 재개: 추가 2 step.
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m gen.train_lora \
  --config gen/lora.smoke.json \
  --output outputs/age_lora_validation_20261026/train_retry \
  --resume-from-checkpoint checkpoint-10 --max-train-steps 12
```

학습 중에는 `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`을 강제한다. ImageFolder의 `Downloading data` 진행 문구는 로컬 파일을 해석할 때도 나오는 datasets 공통 표시이며, 이 실행에서는 로컬 경로만 전달했다. 온라인 dataset 이름·Hub 업로드 옵션은 래퍼가 허용하지 않는다.

공식 소스: [Diffusers v0.35.1 train_text_to_image_lora.py](https://github.com/huggingface/diffusers/blob/v0.35.1/examples/text_to_image/train_text_to_image_lora.py). 원본 SHA-256은 `3dcc88d582bb9f20d4e72278f0a76749a1190ed43a6521b2cdb297efa512136a`이다. 래퍼는 원본 파일을 변경하지 않고 다음 두 지점만 메모리에서 수정하며 `command.json`에 기록한다.

- `lora_layers`의 일회용 `filter`를 목록으로 만든다. 원본에서는 AdamW 생성이 iterator를 소진해 이후 gradient clipping이 빈 목록에 적용될 수 있다.
- 최종 저장 전 전체 UNet의 FP32 변환을 GPU 대신 CPU에서 수행한다. 동결된 기본 가중치까지 GPU에서 확장하는 불필요한 VRAM 사용을 피한다.

그 외 호환 처리로 CLI의 `report_to="none"`을 Accelerator의 Python `log_with=None`으로 전달한다. 계산은 공식 예제의 noise-prediction MSE·UNet LoRA 경로를 사용한다. 관찰용 Accelerator는 CUDA 장치·trainable 텐서, 유한 loss·0보다 큰 gradient norm, AMP step skip 여부, 실제 가중치 변화량을 검사하고 JSONL로 남긴다. 검사용 첫 배치를 별도로 읽으므로 수정 없는 공식 실행과 난수 순서가 같다고 주장하지 않는다.

## 실제 결과

| 실행 | optimization step | 결과 |
|---|---:|---|
| `train/initial` | 0 | 실패. Accelerate 1.15가 문자열 `none` tracker를 거부. Python None 호환 처리 추가 |
| `train_retry/initial` | 1~10 | 학습·저장 성공. 이후 오프라인 LoRA 재로딩에 `weight_name`이 없어 검증 단계 실패. 해당 실행의 status=failed 기록 보존 |
| `train_retry/resume-12` | 11~12 | 성공. checkpoint-10의 model/optimizer/scheduler/sampler/GradScaler/RNG 상태 로드 후 2 step, 최종 LoRA 256개 텐서 정확한 재로딩 확인 |

- 총 **12개의 실제 optimizer update**, skip 0회, OOM 0회. 모든 관찰 loss와 gradient norm이 유한했다.
- 누적 step loss 범위 **0.039817~0.319463**, clipping 전 gradient norm **0.013722~0.045788**. 배치·noise timestep이 다르므로 이 범위로 수렴이나 품질을 주장하지 않는다.
- 첫 10 step에서 초기 LoRA 대비 최대 절댓값 변화 **0.001000895**; 재개 2 step에서 checkpoint-10 대비 추가 변화 **0.000205442**.
- 학습 직전 PyTorch 관점 여유 VRAM: 재시도 **6,716.69MiB**, 재개 **6,704.50MiB**. CUDA runtime이 보고하는 총 메모리와 nvidia-smi의 총 메모리는 서로 다르다.
- 학습 peak allocated **2,525.50MiB**, peak reserved **2,704MiB**. 0.5초 간격 nvidia-smi 최대 관측치는 **3,937MiB**이며 다른 애플리케이션을 포함한다. 후자는 연속 측정한 정확한 peak가 아니다.
- 재시도 전체 약 **26.77초**, 재개·최종 재로딩 약 **11.80초**. 모델 해시 계산·로딩·저장·CPU 재로딩을 포함하며 순수 학습 속도 벤치마크가 아니다.

최종 추론용 파일은 `outputs/age_lora_validation_20261026/train_retry/pytorch_lora_weights.safetensors`이다. SHA-256: `e87abcfc7c2c5fca982717c82b9f2be0fc292d511b47f56d320f897f6deeb022`. **루트 LoRA export는 12 step이고, 전체 재개용 상태는 checkpoint-5와 checkpoint-10에 있다.** checkpoint-12의 전체 optimizer 상태를 저장했다고 해석하면 안 된다. 공식 Accelerate 체크포인트는 동결된 UNet도 포함해 LoRA 파일보다 크다.

## faceid_age 샘플

기존 GPU 검증의 NASA Eileen Collins 공개 사진에서 로고·휘장이 없는 얼굴 crop을 재사용했다. 원본 조건과 SHA-256은 `faceid_age/sample_source.json`에 복사했다. 교육·비상업 연구 추론에만 사용했으며 FFHQ 학습 데이터에 추가하지 않았다. 출력은 AI 합성물이며 NASA의 실제 사진·보증이나 상업 초상 이용 허가를 의미하지 않는다.

실제 실행 명령:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python outputs/age_lora_validation_20261026/verify_faceid_age.py
```

이 로컬 실행 산출 스크립트는 `gen.pipeline`을 호출하고 원본 입력·베이스 모델·LoRA 해시를 대조한다. 추후 표준 CLI로 같은 설정을 사용하는 방법은 다음과 같다. 아래 CLI 예시는 별도로 실행하지 않았다.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m gen.pipeline \
  --config outputs/age_lora_validation_20261026/faceid_age/config.json \
  --mode faceid_age --image outputs/age_lora_validation_20261026/faceid_age/input.png \
  --gender female --targets 55 --output outputs/age_lora_replay
```

실제 활성 adapter는 `faceid`, `age` 두 개다. 목표 55, seed 42, steps 25, strength 0.6이므로 img2img denoising 반복은 15회였다. 모델 로딩 **4.47초**, 얼굴 전처리를 포함한 생성 **2.46초**. 추론 peak allocated **4,050.51MiB**, reserved **4,376MiB**. 출력은 RGB 512×512이며 `input.png`, `input_crop.png`, `age_55.png`, `comparison.png`로 확인할 수 있다. 학습과 추론을 합친 PyTorch allocated 최대는 추론의 4,050.51MiB다.

InsightFace ONNX는 기존 환경처럼 **CPU provider**로 실행됐고 diffusion은 CUDA로 실행됐다. 로컬 베이스에 safety checker가 없다는 기존 제한도 유지된다. 하나의 출력이 목표 나이 정확도·동일인 유지·인구집단 일반화를 검증하지 않는다.

## 출력과 테스트

모든 실행 자료는 Git 제외 경로 `outputs/age_lora_validation_20261026/` 아래에 있다.

| 경로 | 내용 |
|---|---|
| `summary.json`, `packages.txt` | 전체 결과·실패 이력·실행 명령·한계, 전체 패키지 버전 |
| `data/validation.json`, `data/images.json`, `data/groups.csv` | split 검사, 이미지별 해시·캡션, 셀별 분포 |
| `train/initial/` | 최초 0-step 실패의 train.log·summary.json·command.json·gpu.csv |
| `train_retry/config.json` | 최초 10-step 설정. 최종 12-step 설정은 resume-12/command.json에 기록 |
| `train_retry/initial/`, `train_retry/resume-12/` | 각 실행의 train.log·metrics.jsonl·summary.json·command.json·gpu.csv·packages.txt |
| `train_retry/checkpoint-5/`, `train_retry/checkpoint-10/` | 실제 저장한 전체 학습 상태와 각 시점 LoRA |
| `train_retry/pytorch_lora_weights.safetensors` | 최종 12-step 추론용 export |
| `faceid_age/` | config.json·run.json·inference.log·출처 기록·입출력·비교 이미지 |
| `tests_generation.log/json`, `tests_all.log/json` | 테스트별 결과·실패·스킵 사유 |
| `verify_faceid_age.py`, `run_tests.py`, `summarize.py` | 이번 로컬 실행·집계 스크립트 |

`gen/train_lora.py`, `gen/validate_lora_data.py`, 설정·최소 테스트·의존성·문서는 Git에 포함하지만 위 대용량 자료·모델·가상환경은 포함하지 않는다. 공식 예제 원본도 Git 제외 `outputs/age_lora_tools/`에 둔다.

실제로 `unittest` discovery를 출력 기록용 래퍼로 각각 실행했다:

```bash
PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python \
  outputs/age_lora_validation_20261026/run_tests.py --pattern 'test_gen_*.py' --name tests_generation
PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python \
  outputs/age_lora_validation_20261026/run_tests.py --pattern 'test_*.py' --name tests_all
```

- 생성 테스트: **39/39 통과**, 실패·오류·스킵 0. 기존 29개와 신규 데이터·학습 래퍼 계약 10개다.
- 전체 테스트: **52개 중 50 통과, 2 스킵**, 실패·오류 0.
- 스킵: `test_smoke_model_requires_explicit_override`는 기존 smoke ONNX가 없고, `test_training_and_inference_alignment_matches`는 준비된 실제 AI Hub 얼굴 fixture가 없다. FFHQ로 대체하거나 성공 처리하지 않았다.
- 표준 재실행 명령은 `.venv/bin/python -m unittest discover -s tests -p 'test_gen_*.py' -v` 및 `.venv/bin/python -m unittest discover -s tests -v`다. CPU·가짜 런타임 계약 테스트와 이번 실제 GPU 결과는 별도 증거다.

## 남은 한계와 다음 단계

1. 이미지 ID·파일 해시 중복 제거와 **동일인 분리**는 다르다. `person_id`가 없으므로 임베딩 기반 후보 조사와 적절한 검토 후 인물 단위 split을 검증해야 한다.
2. 청소년·고연령 표본 부족과 연령·성별별 성능 차이를 평가해야 한다. 원본 70~120세 그룹의 캡션은 `70s`이므로 정확한 70대 라벨로 해석하지 않는다. 이번에는 캡션을 임의로 수정하지 않았다.
3. 12 step은 수렴·장시간 안정성·학습 효과를 입증하지 않는다. 이 설정을 출발점으로 충분히 학습하고 val만으로 LR·epoch·alpha·adapter scale을 선택한 뒤 고정 설정으로 test를 평가해야 한다.
4. 실제 상태 재개는 성공했지만 데이터 순서까지 중단 없는 실행과 bitwise 동일함을 확인하지 않았다. 엄밀한 재현성이 필요하면 mid-epoch dataloader 재개 위치를 별도로 검증해야 한다.
5. val/test 전체 생성·MAE·ArcFace 품질 비교·공정성 평가·장기 OOM/열 안정성 검사는 **미실행**이다. 정식 나이 ONNX도 없으며 기존 임시 모델을 정확도 평가에 쓰지 않았다.
6. CC BY-NC-SA 4.0 데이터·메타데이터 조건과 이미지별 저작자·라이선스, FaceID/InsightFace 연구용 제약을 계속 준수한다. 개별 동의나 상업적 초상 이용 허가는 검증하지 않았다.
