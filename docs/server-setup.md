# GitHub 업로드와 서버 실행

현재 구현된 나이 추정과 생성·연동 코드를 서버로 옮기는 안내입니다. `gen/`은 중간 발표용 인터페이스·로딩·평가·orchestration과 Mock 검증까지 구현했습니다. 실제 생성 가중치를 사용한 실행은 아직 검증하지 않았습니다.

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

현재 검증한 환경은 Windows CPU입니다. Linux 서버의 GPU 실행은 첫 epoch로 확인해야 합니다.

## 생성 파이프라인과 최소 연동

### 중간 발표용 검증 범위

2026-10-26 Linux/Python 3.14.8, Pillow 12.3.0에서 `test_gen_*.py` 29개와 `gen.smoke`를 실행했습니다. PyTorch·NumPy·Diffusers·ONNX Runtime 설치나 모델 다운로드 없이 확인했습니다. 실제 모델 로딩 API를 검사하는 테스트도 모의 런타임을 사용하므로 GPU·텐서 수치·실제 모델 호환성의 증거가 아닙니다.

| 구분 | 상태 |
|---|---|
| 구현 완료 | 생성 입출력, 로컬 모델 로딩 코드, 캡션 매핑, 평가 루프/CSV, 원본+생성 3장 orchestration, 2×2 합성 |
| Mock 검증 완료 | 순서·원본 보존·중복 목표 나이·오류 전달·모드별 호출·평가 집계·CSV/JSON 출력 |
| 실제 모델 검증 | 미실행. 생성 가중치·정식 나이 ONNX·Age LoRA가 없음 |
| 실제 성능 평가 | 미실행. 연령 MAE·정체성 보존·얼굴 검출률·GPU 처리 시간 미측정 |
| 범위 밖 | HTTP API, 웹 UI, 카메라, 프린터, ControlNet, TensorRT, LCM |

기존 전체 `unittest` 실행은 `test_age`의 torch 및 `test_face`의 numpy import에서 중단됐습니다. 신규 생성 테스트 29개는 통과했지만 전체 테스트 통과라고 보고하지 않습니다. 기존 나이 추정의 Windows 검증 기록과 이번 생성 Mock 검증 기록을 구분하세요.

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

### 실제 모델 준비와 실행

아래는 **필요한 가중치가 준비된 이후의 실행 경로**이며, 이번 작업에서 실행한 명령이 아닙니다. 서버에 맞는 CUDA PyTorch를 먼저 설치하고 생성 의존성을 준비합니다.

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
| `age_lora` | 별도 학습된 SD1.5 Age LoRA. 현재 제공하지 않음 |
| `model_root` | `age.setup_models --with-arcface`로 준비한 InsightFace 디렉터리 |

`prompt_only`는 base와 검출·정렬 자산만 필요하며, `faceid`는 Age LoRA 없이 실행합니다. `faceid_age`만 Age LoRA까지 요구합니다. FaceID 모드는 CLIP·ArcFace 임베딩을 함께 사용하고 Plus v2의 shortcut을 활성화합니다. FaceID LoRA와 Age LoRA의 가중치는 별도로 적용합니다.

Diffusers 0.35.1 소스/API를 기준으로 구현했습니다. 기본 설정은 DPM++ 2M Karras, 25 step, strength 0.6, CFG 7.5입니다. 동일 요청의 각 목표 나이에 같은 seed를 다시 적용합니다. `--seed`의 기본값은 42이며 JSON의 seed보다 CLI 값을 우선합니다. 이 설정은 품질·속도 최적화 결과가 아닙니다.

```bash
# Age LoRA 없는 최초 실제 추론 경로. 동의받은 실제 얼굴 파일과 모델이 필요합니다.
python -m gen.pipeline --config gen/config.example.json --mode faceid --image data/demo/portrait.png --targets 15 35 45 --output outputs/gen_faceid_trial

# Age LoRA까지 준비한 이후
python -m gen.pipeline --config gen/config.example.json --mode faceid_age --image data/demo/portrait.png --targets 15 35 45 --output outputs/gen_full_trial
```

모델·데이터의 비상업 연구 및 원저작자 이용 조건을 확인해야 합니다. 실제 모델 로딩·GPU 메모리·생성 이미지 품질은 첫 실제 실행에서 별도로 검증합니다.

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
