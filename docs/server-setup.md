# GitHub 업로드와 서버 실행

현재 구현된 나이 추정과 생성·연동 코드를 서버로 옮기는 안내입니다. `gen/`은 중간 발표용 인터페이스·로딩·평가·orchestration과 Mock 검증을 구현했습니다. 2026-10-26 RTX 4060에서 `prompt_only`와 `faceid`를 검증한 뒤, FFHQ-Aging으로 12-step Age LoRA 학습·재로딩·`faceid_age` 생성 1장도 확인했습니다. 연령 변환 품질·정식 나이 정확도 평가는 아직 수행하지 않았습니다.

## GitHub에 올릴 파일

```text
.gitignore
README.md
requirements.txt
requirements-insightface.txt    # 향후 전체 InsightFace SDK 사용 시 참고
requirements-generation.txt   # 실제 생성 모델 실행 시 선택 의존성
age/                           # 전처리·모델·학습·평가·ONNX·모델 다운로드 코드
gen/                           # 생성·평가·4컷 연동 및 모델 없는 smoke 검증
tests/                         # 구현 검사
docs/                          # 설계·구현 상태·이 안내
data/README.md                 # 데이터 출처와 라벨 설명만
```

서버 실행에 필수인 것은 `age/`와 `requirements.txt`입니다. README·docs·tests는 팀 공유와 실행 검증을 위해 함께 올립니다. `requirements-insightface.txt`는 현재 나이 추정 실행에 설치하지 않아도 됩니다.

`.gitignore`는 `data/README.md`를 제외한 데이터 폴더 전체, 모델 캐시, 실행 결과, 가상환경, 학습 가중치와 압축 파일을 제외합니다. CSV/JSON에는 인물 ID와 원본 식별자가 있으므로 데이터 분할·전처리 보고서도 Git에 올리지 않습니다. 웹에서 파일을 직접 업로드할 때는 이 파일 목록을 따르세요. `.gitignore`는 웹 업로드를 차단하지 않습니다.

## 서버에서 별도로 준비할 파일

| 경로 | 준비 방법 |
|---|---|
| `data/raw/aihub/` | 승인된 전체 AI Hub 이미지·JSON을 별도 전송하거나 서버에서 다운로드·압축 해제 |
| `data/age_aihub/` | 전처리 코드가 정렬 이미지와 train/val/test CSV를 생성 |
| `data/lora/` | 로컬에 확보한 FFHQ-Aging 준비본의 train/val/test 폴더를 그대로 별도 전송. Age LoRA 학습용이며 AI Hub 입력과 별개 |
| `data/gen_eval/`, `data/reports/` | FFHQ val/test 상대 경로 CSV, 이동 검증 보고서·이미지별 출처·라이선스를 데이터와 함께 전송 |
| `models/insightface/` | `python -m age.setup_models`로 공식 소스·buffalo_l 가중치 다운로드 |
| `models/torch/` | 학습 시 ImageNet 사전학습 MobileNetV3 가중치 자동 다운로드 |
| `outputs/` | 학습 체크포인트·설정·시간 기록·평가·ONNX를 실행 중 생성 |

서버에서 인터넷 다운로드가 불가능하면, 이미 준비된 `models/insightface/`와 `models/torch/`를 GitHub와 별개로 전송하세요. 다운로드한 Python runtime 소스와 LICENSE·provenance도 함께 옮겨야 합니다. Windows의 `.venv`는 복사하지 않고 서버에서 새로 만듭니다.

## Linux 서버 첫 실행

서버의 NVIDIA 드라이버가 준비돼 있어야 합니다. 저장소를 복제하고 프로젝트 루트에서 실행합니다.

```bash
git clone https://github.com/HanyangTechAI/2026-2-HAI-ProjectTeam3.git
cd 2026-2-HAI-ProjectTeam3
```

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

먼저 [PyTorch 공식 설치 안내](https://pytorch.org/get-started/locally/)에서 서버에 맞는 CUDA 지원 torch·torchvision 설치 명령을 실행합니다. 그다음 프로젝트 의존성을 설치하고 GPU 접근을 확인합니다.

```bash
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__); assert torch.cuda.is_available(), 'CUDA GPU not available'; print(torch.cuda.get_device_name(0))"
python -m age.setup_models
python -m unittest discover -s tests -v
```

현재 requirements의 ONNX Runtime은 CPU용입니다. 얼굴 전처리는 CPU로 실행하고 MobileNet 학습은 PyTorch CUDA로 실행할 수 있습니다. 전처리도 GPU로 돌리려면 호환되는 GPU용 ONNX Runtime을 별도로 준비해야 합니다.

검사 중 모델/실제 데이터/시험용 ONNX가 필요한 항목은 파일이 없으면 skip됩니다. 데이터 없이 학습·ONNX 흐름을 먼저 확인하려면 아래를 실행합니다. 합성 입력으로 확인하므로 결과는 실제 나이 추정 성능이 아닙니다.

```bash
python -m age.smoke --pretrained --output outputs/server_smoke
```

## 실제 데이터 전처리·학습

이미 압축 해제된 이미지와 JSON이 `data/raw/aihub`에 있다면:

```bash
python -m age.prep --source aihub --raw-root data/raw/aihub --inspect-only
python -m age.prep --source aihub --raw-root data/raw/aihub --limit 50
python -m age.prep --source aihub --raw-root data/raw/aihub --device cpu
```

ZIP을 별도 폴더에 받았다면 첫 명령의 `--raw-root`를 실제 ZIP 폴더로 바꾸고 `--extract-to data/raw/aihub`를 추가합니다. 전체 데이터의 인물 수와 `age_past` 분포, 전처리 제외 사유, 인물별 분할 결과를 확인합니다. 현재 로컬에 있는 한 사람의 50장만으로는 val/test가 비어 정식 학습이 중단됩니다.

```bash
# 먼저 첫 epoch의 속도와 GPU 메모리 사용 확인
python -m age.train --device cuda --epochs 1 --workers 4 --output outputs/age/aihub_trial
# 이후 새 출력 폴더에서 기본 30 epoch 학습
python -m age.train --device cuda --workers 4 --output outputs/age/aihub
```

VRAM이 부족하면 `--batch-size 64` 등으로 줄입니다. 각 실험은 새로운 `--output`을 사용합니다. 학습 기본값은 AI Hub만 사용하고 ImageNet에서 시작하며, 기존 나이 추정 체크포인트가 필요하지 않습니다.

학습 완료 후 모델을 고정하고 test 평가와 ONNX 전달 검증을 실행합니다.

```bash
python -m age.evaluate --checkpoint outputs/age/aihub/best.pt --csv data/age_aihub/test.csv --output outputs/eval/aihub --device cuda --baseline
python -m age.export_onnx --checkpoint outputs/age/aihub/best.pt --csv data/age_aihub/val.csv --output outputs/age/age_estimator.onnx
```

나이 추정 학습을 검증한 환경은 Windows CPU입니다. Linux 서버의 나이 추정 GPU 학습은 첫 epoch로 확인해야 합니다. 아래 생성 GPU 검증은 이 학습 검증과 별개입니다.

## 생성 파이프라인과 최소 연동

### 중간 발표용 검증 범위

1차 Mock 검증에서는 2026-10-26 Linux/Python 3.14.8, Pillow 12.3.0에서 `test_gen_*.py` 29개와 `gen.smoke`를 실행했습니다. PyTorch·NumPy·Diffusers·ONNX Runtime 설치나 모델 다운로드 없이 확인했습니다. 실제 모델 로딩 API를 검사하는 테스트도 모의 런타임을 사용하므로 GPU·텐서 수치·실제 모델 호환성의 증거가 아닙니다. 이후 같은 날 수행한 실제 GPU 검증은 아래 별도 절에 기록합니다.

| 구분 | 상태 |
|---|---|
| 구현 완료 | 생성 입출력, 로컬 모델 로딩 코드, 캡션 매핑, 평가 루프/CSV, 원본+생성 3장 orchestration, 2×2 합성 |
| Mock 검증 완료 | 순서·원본 보존·중복 목표 나이·오류 전달·모드별 호출·평가 집계·CSV/JSON 출력 |
| 실제 모델 검증 | `prompt_only`·`faceid`·12-step LoRA를 사용한 `faceid_age` 각 1장 성공. 정식 나이 ONNX 미검증 |
| 실제 성능 평가 | 단일 입력의 GPU 시간·메모리·FaceID cosine 기록. 연령 MAE·모집단 정체성 보존·얼굴 검출률은 미평가 |
| 범위 밖 | HTTP API, 웹 UI, 카메라, 프린터, ControlNet, TensorRT, LCM |

1차 Mock 환경의 전체 `unittest` 실행은 `test_age`의 torch 및 `test_face`의 numpy import에서 중단됐습니다. 이후 실제 GPU 환경에서는 전체 42개 중 40개 통과·2개 스킵으로 재검증했습니다. 기존 나이 추정의 Windows 검증, 1차 생성 Mock 검증, 아래 실제 GPU 검증 기록을 구분하세요.

```bash
python -m unittest discover -s tests -p "test_gen_*.py" -v
python -m gen.smoke --output outputs/gen_smoke
```

smoke는 직접 그린 합성 그림과 고정 Mock 나이 25를 사용합니다. 얼굴 검출이나 연령 변환은 하지 않습니다. 모든 실행은 새 출력 디렉터리를 요구하며, 이미 존재하면 중단합니다. 산출물은 Git에서 제외합니다.

```text
outputs/gen_smoke/
  synthetic.png
  booth/{01_past.png,02_current.png,03_future.png,04_later.png,four_cut.png,metadata.json}
  evaluation/{results.csv,summary.json,prompt_only/,faceid/,faceid_age/}
  status.json
```

### 생성 인터페이스

```python
from gen.pipeline import MockPipeline, read_image

pipeline = MockPipeline()  # 명시적인 모의 실행. 실제 생성의 대체 결과가 아님.
try:
    images = pipeline.age_shift(read_image("outputs/gen_smoke/synthetic.png"), [15, 35, 45], seed=42)
finally:
    pipeline.close()
```

- 입력: RGB PIL 이미지 또는 NumPy `uint8 [H,W,3]`. 파일 입력은 `read_image`가 EXIF 방향과 RGB 변환을 처리합니다.
- 목표 나이: 0~100 정수의 비어 있지 않은 목록. 출력은 같은 순서·같은 개수의 RGB PIL 이미지입니다.
- 실제 생성: 기존 얼굴 검출·정렬을 재사용한 여유 있는 512px 크롭을 img2img 입력으로 사용합니다. 결과는 512px입니다.
- 캡션은 `gen.pipeline.age_prompt`의 단일 매핑을 사용합니다. 향후 LoRA 학습 데이터 준비도 이 함수를 재사용할 수 있습니다.
- `AgeShiftPipeline.load()`는 인스턴스마다 한 번 로드합니다. 요청마다 생성기를 다시 만들지 않습니다. `close()`는 자원을 해제합니다.
- 세 모드는 `prompt_only`(입력 이미지를 사용하는 기본 img2img), `faceid`, `faceid_age`입니다. 모드별 인스턴스를 분리해 어댑터 잔류를 피합니다.
- 얼굴 미검출은 `FaceNotFoundError`, 로컬 자산 누락은 `ModelUnavailableError`, 생성 실패는 `GenerationError`로 전달합니다. 실제 모드 실패를 Mock으로 자동 대체하지 않습니다.
- 한 인스턴스는 순차 요청용입니다. 서버에서 병렬 요청으로 공유하는 기능은 이번 범위에 없습니다.

### 준비된 FFHQ-Aging 데이터

2026-10-26에 외부 `ffhq-prepared/`를 프로젝트의 `data/lora/`로 이동했습니다. 기존 설계의 준비본 경로에 맞춘 것이며 `raw/`에 넣거나 `age.prep`로 다시 처리하지 않았습니다. 구성과 출처는 [데이터 안내](../data/README.md#ffhq-aging-배치와-사용)에 기록했습니다.

| 경로 | 장수 | 용도 |
|---|---:|---|
| `data/lora/train/` | 8,163 | Age LoRA 학습 입력. 이미지와 캡션을 사용하는 외부 학습 스크립트에 이 디렉터리만 전달 |
| `data/lora/val/` | 400 | 개발·설정·체크포인트 선택. `data/gen_eval/ffhq_val.csv`로 생성 평가에 연결 |
| `data/lora/test/` | 100 | 설정 고정 후 최종 평가. `data/gen_eval/ffhq_test.csv`로 연결하며 학습·튜닝에 사용하지 않음 |

각 split은 RGB 512×512 PNG와 `metadata.jsonl`의 `file_name`·`text`로 구성됩니다. 캡션은 `gen.pipeline.age_prompt`와 일치하며, FFHQ의 연령 그룹은 정확한 나이 정답이 아닙니다. 따라서 `age.train --data-dir`에 전달하거나 `gen/config.example.json`의 `age_lora`에 데이터 경로를 적으면 안 됩니다. 후자는 학습된 가중치 파일입니다.

`gen.train_lora`가 공식 Diffusers 0.35.1 LoRA 예제를 해시 고정해 실행합니다. `data/lora/train`만 읽으며, `data/lora/` 전체를 하나의 학습 폴더로 재귀 수집하면 val/test가 섞이므로 거부합니다. 데이터 이동 자체와 후속 학습 검증은 별개입니다. RTX 4060에서 전체 무결성 검사와 12-step 학습·재개·재로딩·`faceid_age` 1장 생성까지 수행한 절차는 [Age LoRA 검증 보고서](age-lora-validation.md)에 있습니다. 데이터를 다시 받지 않습니다.

데이터 파일 연결만 확인하려면 프로젝트 루트에서 val 이미지로 Mock을 실행합니다. 실제 얼굴 검출·연령 변환·정확도 검증이 아닙니다.

```bash
python -m gen.pipeline --mock --image data/lora/val/images/00033.png --targets 15 35 45 --output outputs/ffhq_val_image_mock
```

`gen.eval`은 JSONL 대신 별도 CSV를 읽으며 상대 경로는 CSV 디렉터리 기준입니다. 새 평가 CSV는 `../lora/val/images/...` 또는 `../lora/test/images/...`를 참조하므로 이미지 복사 없이 사용할 수 있습니다.

```bash
# val 전체의 입력·출력 연결 확인. 성능 지표는 계산하지 않음.
python -m gen.eval --mock --csv data/gen_eval/ffhq_val.csv --targets 15 35 45 --output outputs/ffhq_val_eval_mock
# Age LoRA·정식 나이 ONNX를 준비하고 설정을 고정한 뒤 최종 평가
python -m gen.eval --config gen/config.example.json --csv data/gen_eval/ffhq_test.csv --targets 15 35 45 --age-onnx outputs/age/age_estimator.onnx --output outputs/ffhq_test_eval
```

test 100장은 이미지 ID·파일 해시 기준으로 train/val과 겹치지 않지만, 인물 ID가 없어 **동일인 분리까지 보장하지는 않습니다**. 최종 성능 보고 전 별도 확인이 필요합니다. 모든 출력 경로는 새 디렉터리를 사용하세요.

### 실제 모델 준비와 실행

아래는 **필요한 가중치가 준비된 이후의 일반 실행 경로**입니다. 실제 검증에 사용한 버전·모델 출처·설정·명령은 아래 GPU 검증 절에 기록합니다. 서버에 맞는 CUDA PyTorch를 먼저 설치하고 생성 의존성을 준비합니다.

```bash
python -m pip install -r requirements-generation.txt
python -m age.setup_models --with-arcface
```

`--with-arcface`는 기존 검출·genderage 준비에 공식 v0.7 `arcface_onnx.py`와 buffalo_l의 `w600k_r50.onnx`를 추가합니다. 기존 `age.setup_models` 기본 동작은 유지합니다. 전체 InsightFace SDK/C++ 확장은 필요하지 않습니다. 이 준비 명령만 외부 모델·소스를 다운로드하며, 생성 로더 자체는 로컬 파일만 사용합니다.

`gen/config.example.json`은 경로와 기본값 예시입니다. 경로는 실행 디렉터리 기준이며 다음 자산은 별도로 준비해야 합니다.

| 설정 | 필요한 로컬 자산 |
|---|---|
| `base_model` | Realistic Vision V5.1의 완전한 Diffusers 디렉터리. config·tokenizer·각 모델 컴포넌트 포함. 단일 checkpoint 파일은 직접 지원하지 않음 |
| `image_encoder` | `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`에 대응하는 Transformers 모델·전처리 설정 디렉터리 |
| `ip_adapter` | `ip-adapter-faceid-plusv2_sd15.bin` 파일 |
| `faceid_lora` | 기본 빈 값: 체크포인트에 포함되어 Diffusers가 자동 로드한 `faceid_0` 사용. 별도 짝을 이루는 FaceID LoRA 파일을 지정하면 자동 어댑터를 제거하고 교체 |
| `age_lora` | 별도 학습된 SD1.5 Age LoRA 파일. 12-step 로컬 smoke export는 검증 보고서 참고; 가중치는 Git에 포함하지 않음 |
| `model_root` | `age.setup_models --with-arcface`로 준비한 InsightFace 디렉터리 |

`prompt_only`는 base와 검출·정렬 자산만 필요하며, `faceid`는 Age LoRA 없이 실행합니다. `faceid_age`만 Age LoRA까지 요구합니다. FaceID 모드는 CLIP·ArcFace 임베딩을 함께 사용하고 Plus v2의 shortcut을 활성화합니다. FaceID LoRA와 Age LoRA의 가중치는 별도로 적용합니다.

Diffusers 0.35.1 소스/API를 기준으로 구현했습니다. 기본 설정은 DPM++ 2M Karras, 25 step, strength 0.6, CFG 7.5입니다. 동일 요청의 각 목표 나이에 같은 seed를 다시 적용합니다. `--seed`의 기본값은 42이며 JSON의 seed보다 CLI 값을 우선합니다. 이 설정은 품질·속도 최적화 결과가 아닙니다.

```bash
# Age LoRA 없는 최초 실제 추론 경로. 동의받은 실제 얼굴 파일과 모델이 필요합니다.
python -m gen.pipeline --config gen/config.example.json --mode faceid --image data/demo/portrait.png --targets 15 35 45 --output outputs/gen_faceid_trial

# Age LoRA까지 준비한 이후
python -m gen.pipeline --config gen/config.example.json --mode faceid_age --image data/demo/portrait.png --targets 15 35 45 --output outputs/gen_full_trial
```

모델·데이터의 비상업 연구 및 원저작자 이용 조건을 확인해야 합니다. 아래 단일 샘플의 로딩·추론 검증은 제품 수준의 나이 정확도·정체성 보존 검증이 아닙니다.

### 실제 GPU 검증 기록 (2026-10-26)

이 절은 최초 `prompt_only`·`faceid` 검증 기록입니다. `feat/generation-midterm`에서 기존 추론 코드와 29개 생성 테스트를 수정하지 않고 실행했습니다. 다른 모델로 교체하거나 실패를 Mock으로 대체하지 않았습니다. 이 최초 실행에서는 FFHQ-Aging·AI Hub 데이터와 Age LoRA를 사용하지 않았고 학습도 수행하지 않았습니다. 후속 학습 결과는 [별도 보고서](age-lora-validation.md)에 기록했습니다.

#### 환경과 설치

| 항목 | 확인 결과 |
|---|---|
| GPU | NVIDIA GeForce RTX 4060, `nvidia-smi` 총 8,188 MiB; PyTorch가 보고한 사용 가능 총량 7,885.5 MiB |
| 드라이버 / CUDA | 615.78.08 / PyTorch CUDA runtime 12.8; 드라이버 표기의 CUDA UMD 13.4와 구분 |
| CUDA 동작 | `torch.cuda.is_available() == True`, 실제 GPU 행렬곱과 유한값 검사 성공 |
| Python | 시스템 3.14.8 대신 `.venv`의 Python 3.11.16 사용 |
| 메모리 / 디스크 | 시스템 RAM 31 GiB; 프로젝트 디스크 여유 약 134 GiB에서 준비 후 122.44 GiB |
| GPU 공존 | 시작 시 데스크톱 프로그램이 약 1.5 GiB 사용. 다른 프로세스를 종료하지 않음 |
| 얼굴 전처리 | ONNX Runtime CPU provider; diffusion UNet·VAE·FaceID CLIP은 CUDA FP16 |

사용 패키지는 torch **2.8.0+cu128**, torchvision **0.23.0+cu128**, diffusers **0.35.1**, transformers **4.57.6**, peft **0.17.1**, accelerate **1.15.0**, safetensors **0.8.0**, huggingface-hub **0.36.2**, onnxruntime **1.31.0**, numpy **2.4.6**, Pillow **12.3.0**, scikit-image **0.26.0**입니다. 전체 의존성 버전은 산출물의 `packages.txt`에 보관했고 `uv pip check`도 통과했습니다. 실행한 설치 명령은 다음과 같습니다.

```bash
python3.11 -m venv .venv
uv pip install --python .venv/bin/python 'torch==2.8.0' 'torchvision==0.23.0' -r requirements-generation.txt
.venv/bin/python -m age.setup_models --with-arcface
```

범위형 requirements를 나중에 다시 해석하면 하위 패키지가 달라질 수 있습니다. 동일 서버의 정확한 재설치에는 `outputs/gpu_validation_20261026/packages.txt`를 사용하세요.

#### 공식 모델과 이용 조건

| 구성 | 공식 배포처 / 고정 revision | 사용 파일·형식 / 조건 |
|---|---|---|
| Realistic Vision V5.1 | [SG161222/Realistic_Vision_V5.1_noVAE](https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE/tree/1e9f017a7b1eaefb63a1900ea6c5953d2739fd21), `1e9f017a7b1eaefb63a1900ea6c5953d2739fd21` | SD1.5의 완전한 Diffusers 디렉터리, UNet·text encoder·VAE는 Safetensors. CreativeML OpenRAIL-M |
| FaceID Plus v2 | [h94/IP-Adapter-FaceID](https://huggingface.co/h94/IP-Adapter-FaceID/tree/43907e6f44d079bf1a9102d9a6e56aef7a219bae), `43907e6f44d079bf1a9102d9a6e56aef7a219bae` | `ip-adapter-faceid-plusv2_sd15.bin`, PyTorch checkpoint. 모델 카드의 연구 전용·비상업 조건 |
| FaceID LoRA | 위와 동일한 배포처·revision | `ip-adapter-faceid-plusv2_sd15_lora.safetensors`, 51,059,544 bytes. 위 어댑터와 짝을 이루는 SD1.5 LoRA |
| CLIP vision encoder | [h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter/tree/018e402774aeeddd60609b4ecdb7e298259dc729/models/image_encoder), `018e402774aeeddd60609b4ecdb7e298259dc729` | 공식 `models/image_encoder/{config.json,model.safetensors}`. LAION ViT-H/14의 vision-only 배포본이며 SDXL용 bigG가 아님 |
| CLIP 전처리 | [laion/CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/tree/1c2b8495b28150b8a4922ee1c8edee224c284c0c), `1c2b8495b28150b8a4922ee1c8edee224c284c0c` | `preprocessor_config.json`을 위 encoder 디렉터리에 추가. upstream MIT, 모델 카드의 배포·감시 용도 제한 안내도 별도 확인 |
| 얼굴 검출·인식 | [InsightFace 공식 v0.7 buffalo_l.zip](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip) | `det_10g.onnx`, `genderage.onnx`, `w600k_r50.onnx`. 코드 MIT와 달리 사전학습 가중치는 비상업 연구 전용 |

base 저장소 이름에 `noVAE`가 있지만 해당 revision의 Diffusers 트리에는 `vae/config.json`과 `vae/diffusion_pytorch_model.safetensors`가 실제로 있습니다. 이를 그대로 사용했고 별도 VAE로 교체하지 않았습니다. 제작자가 권장한 별도 VAE fine-tune과 이 파일이 같은지는 검증하지 않았습니다. 루트의 단일 `.ckpt`·`.safetensors`, inpainting 및 SDXL 변형은 사용하지 않았습니다.

FaceID LoRA는 설정의 `faceid_lora`에 명시해 자동 로드된 `faceid_0`를 제거한 후 이름 `faceid`로 적용했습니다. 실제 활성 어댑터 목록은 `{"unet": ["faceid"]}`였습니다. `shortcut=True`, `ip_scale=0.8`, `faceid_lora_scale=1.0`입니다. Age LoRA 경로는 빈 문자열입니다. 다운로드 파일의 크기와 SHA-256은 `model_sources.json` 및 `models/insightface/provenance.json`에 기록했습니다.

FaceID에는 별도의 상세 가중치 라이선스 파일이 없어 [모델 카드의 연구용 제한](https://huggingface.co/h94/IP-Adapter-FaceID/blob/43907e6f44d079bf1a9102d9a6e56aef7a219bae/README.md)을 따릅니다. 일반 IP-Adapter 코드의 Apache-2.0을 FaceID 가중치의 상업적 사용 허가로 해석하면 안 됩니다. [OpenRAIL-M](https://github.com/CompVis/stable-diffusion/blob/main/LICENSE), [InsightFace 가중치 조건](https://github.com/deepinsight/insightface/blob/v0.7/python-package/README.md#license)도 각각 적용됩니다.

#### 입력 허가와 실제 실행

입력은 scikit-image 0.26.0에 포함된 `astronaut.png`입니다. [공식 scikit-image 설명](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut)과 [NASA 원출처](https://www.flickr.com/photos/nasacommons/16504233985/)에서 Eileen Collins의 NASA 사진 `S99-00858`, public domain / no known copyright restrictions임을 확인했습니다. 원본 SHA-256은 `88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5`입니다.

[NASA 이미지·AI 사용 지침](https://www.nasa.gov/nasa-brand-center/images-and-media/)에 따라 **모델에 입력하기 전에** `(125, 10, 335, 225)` 얼굴 영역만 잘라 NASA 로고·휘장을 제외했습니다. 교육·비상업 연구 검증에만 사용하며 상업적 초상 이용 허가를 주장하지 않습니다. 비교 이미지에는 원본 출처와 AI 생성물임을 구분했습니다. 결과는 이 프로젝트의 합성이며 NASA의 제작·보증이나 해당 나이의 실제 사진이 아닙니다.

사용 설정은 출력 512×512, seed 42, target 55, gender female, DPM++ 2M Karras, steps 25, strength 0.6, CFG 7.5입니다. img2img에서 실제 denoising은 15회였습니다. 측정 스크립트는 `AgeShiftPipeline.load()`와 `age_shift()`를 직접 호출하며 Mock 경로가 없습니다. 실행 명령:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. .venv/bin/python outputs/gpu_validation_20261026/verify.py --mode prompt_only --config outputs/gpu_validation_20261026/config.json --image outputs/gpu_validation_20261026/input_face.png --target 55 --output outputs/gpu_validation_20261026/prompt_only
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. .venv/bin/python outputs/gpu_validation_20261026/verify.py --mode faceid --config outputs/gpu_validation_20261026/config.json --image outputs/gpu_validation_20261026/input_face.png --target 55 --output outputs/gpu_validation_20261026/faceid
```

각 디렉터리는 이미 결과가 있으므로 재실행 시 새로운 `--output` 경로를 사용해야 합니다. `prepare.py`·`verify.py`는 이 서버의 실행 기록으로 `outputs/`에 보관하며 Git clone에는 포함되지 않습니다. 일반 생성 진입점은 기존 `.venv/bin/python -m gen.pipeline`이며 같은 설정·입력으로 `--mode faceid --targets 55 --gender female --output <새 경로>`를 사용할 수 있습니다.

#### 측정과 산출물

| 모드 | 실제 생성 | 초기 로딩 | 한 장 생성 | 생성 peak allocated / reserved | 생성 후 전체 GPU 사용량 |
|---|---|---:|---:|---:|---:|
| `prompt_only` | 성공, 512×512 PNG | 4.417초 | 2.201초 | 2,675.38 / 2,960 MiB | 4,611 MiB |
| `faceid` | 성공, 512×512 PNG | 4.021초 | 2.614초 | 4,044.43 / 4,372 MiB | 5,988 MiB |

각 모드는 별도 프로세스에서 워밍업 없이 한 번 측정했습니다. 타이머 경계에서 CUDA를 동기화했으며 생성 시간은 얼굴 전처리·임베딩을 포함하고 PNG 저장·후속 cosine 계산을 제외합니다. 모델 로딩은 `load()` 호출 시간이며 다운로드·가상환경 설치·사전 torch import는 제외합니다. OS 파일 캐시는 비우지 않았으므로 cold-disk 벤치마크나 평균 처리량이 아닙니다. peak는 PyTorch allocator 기준이고 전체 GPU 열은 다른 앱을 포함한 `nvidia-smi`의 **생성 후 스냅샷**, 전체 장치 peak가 아닙니다.

공통 경로는 `outputs/gpu_validation_20261026/`입니다.

```text
config.json, packages.txt, model_sources.json, sample_source.json
input_face.png, prepare.py, verify.py, summarize.py, validation.json
prompt_only/{age_55.png,input.png,input_crop.png,comparison.png,config.json,run.json}
faceid/{age_55.png,input.png,input_crop.png,comparison.png,config.json,run.json}
prepare.log, insightface_setup.log, prompt_only.log, faceid.log
tests_generation.log, tests_all.log, mock_smoke.log, mock_smoke/
```

비교 이미지를 열어 원본·정렬 입력·생성물이 정상인지 확인했습니다. 생성 얼굴의 헤어스타일·표정·피부 질감 변화가 보이지만, FaceID 결과를 정확한 55세 변환이나 동일인 보존의 증명으로 해석하지 않습니다. FaceID 출력에서 얼굴 검출에 성공했고 입력 대비 ArcFace cosine은 **0.655573**입니다. 단일 표본의 참고값이며 확률·정확도·검증된 판정 임계값이 아닙니다. 정식 나이 추정 ONNX가 없어 연령 MAE는 측정하지 않았습니다.

#### 재검증과 남은 범위

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_gen_*.py' -v
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m gen.smoke --output outputs/gpu_validation_20261026/mock_smoke
```

- 기존 생성 테스트 **29/29 통과**. 테스트 파일을 수정하지 않았습니다.
- 전체 테스트 **42개 중 40개 통과, 2개 스킵**, 실패·오류 0개. 스킵은 `Requires smoke ONNX`, `Requires actual prepared faces`이며 모두 통과했다고 보고하지 않습니다.
- 명시적 Mock smoke의 4컷·9개 평가 행도 정상 생성했습니다. 이 결과를 실제 추론 결과와 섞지 않았습니다.
- CUDA ONNX provider 부재, `torch_dtype` 및 scikit-image `estimate` 사용 중단 예고, FaceID LoRA의 text-encoder 키 부재 경고가 있었습니다. ONNX는 CPU 실행으로 확인했고 해당 LoRA는 UNet용입니다. 로딩·추론 실패나 OOM은 없었으며 API·모델 코드를 수정할 필요가 없었습니다.
- 제작자 base `model_index.json`에는 safety checker가 `null`입니다. 검증에서 이를 새로 끄거나 대체하지 않았고, 따라서 실제 출력의 안전 필터를 검증한 것으로 간주하지 않습니다. 공개 서비스 전 별도 정책·안전 검토가 필요합니다.
- 후속 12-step Age LoRA 학습·`faceid_age` 로딩 검증은 완료했습니다. 충분한 LoRA 학습·품질 평가, 정식 나이 ONNX, 실제 데이터 기반 4컷, 여러 인물의 연령 MAE·정체성 보존 평가는 남아 있습니다. 모델·입력·생성물·가상환경은 기존 `.gitignore`에 따라 Git에 넣지 않았습니다.

### D팀 연동

```python
from gen.booth import MockAgeEstimator, run_booth, compose_four_cut
from gen.pipeline import MockPipeline, read_image

generator = MockPipeline()
try:
    result = run_booth(read_image("outputs/gen_smoke/synthetic.png"), MockAgeEstimator(25), generator)
    sheet = compose_four_cut(result["frames"])
    metadata = result["metadata"]
finally:
    generator.close()
```

나이 추정기는 `PIL RGB -> {age, gender}` callable입니다. 실제 ONNX는 `gen.booth.make_age_estimator`가 기존 `age.infer.AgeEstimator`를 감쌉니다. 목표값은 기존 `age.common.target_ages`로 계산합니다.

0-based 슬롯은 `[과거, 현재, 미래, 더 먼 미래]`이며 현재 슬롯 1은 입력 원본의 복사본입니다. 슬롯 0·2·3만 생성합니다. clamp 때문에 같은 목표 나이가 여러 번 나와도 슬롯을 제거하지 않습니다. `compose_four_cut`은 비율을 유지한 2×2 합성 예시이며 프린터용 최종 규격은 아닙니다. 실패하면 예외를 호출자에게 전달하며 부분 성공을 완성된 4컷으로 저장하지 않습니다.

```bash
python -m gen.booth --mock --image outputs/gen_smoke/synthetic.png --mock-age 25 --output outputs/booth_demo
# 실제 모델과 정식 나이 ONNX가 준비된 이후
python -m gen.booth --config gen/config.example.json --image data/demo/portrait.png --age-onnx outputs/age/age_estimator.onnx --output outputs/booth_real_trial
```

### 평가 기록

`gen.eval`은 항상 세 ablation을 순차 실행합니다. 모드별로 모델을 한 번 로드하고 모든 입력을 처리한 뒤 해제합니다. 같은 입력·목표·seed·설정을 사용하며 모델 로드 시간과 생성 시간을 분리합니다. 이미지 저장과 지표 계산 시간은 생성 시간에서 제외합니다.

입력 CSV는 `path`와 선택적 `sample_id` 열입니다. 상대 경로는 **CSV가 있는 디렉터리 기준**입니다. 기존 `age/` manifest의 루트 기준 상대 경로와 혼동하지 마세요.

```csv
sample_id,path
portrait_01,images/portrait_01.png
```

```bash
# 위 형식의 별도 평가 manifest가 있을 때
python -m gen.eval --mock --csv data/gen_eval/samples.csv --targets 15 35 45 --output outputs/gen_eval_mock
python -m gen.eval --config gen/config.example.json --csv data/gen_eval/samples.csv --targets 15 35 45 --age-onnx outputs/age/age_estimator.onnx --output outputs/gen_eval_real
```

`results.csv`는 목표별 상태·추정 나이·절대 오차·ArcFace cosine·얼굴 실패 여부·시간을 기록합니다. `summary.json`은 모드별 MAE·평균 cosine·유효 표본 수·실패율·설정·평가기 정보를 기록합니다. 얼굴 실패율 분모는 명시적 얼굴 미검출 또는 얼굴 기반 지표 성공으로 검출 여부를 알 수 있는 시도이며, Mock 및 원인 불명 오류는 제외합니다. 누락된 지표는 0이 아닌 빈 값/`null`입니다.

Mock은 평가기를 호출하지 않으며 성능 수치를 만들지 않습니다. 실제 평가도 별도의 InsightFace genderage 교차 측정과 육안 검토를 대체하지 않습니다. 이번 구현에는 학습, 데이터 선별, 대규모 파라미터 그리드 실행을 포함하지 않았습니다.

API 확인에 사용한 공식 자료: [Diffusers 0.35.1 IP-Adapter](https://github.com/huggingface/diffusers/blob/v0.35.1/docs/source/en/using-diffusers/ip_adapter.md), [FaceID 모델 카드](https://huggingface.co/h94/IP-Adapter-FaceID), [InsightFace v0.7 ArcFace](https://github.com/deepinsight/insightface/blob/v0.7/python-package/insightface/model_zoo/arcface_onnx.py).
