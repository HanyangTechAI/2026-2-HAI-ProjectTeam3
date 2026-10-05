# GitHub 업로드와 서버 실행

현재 구현된 나이 추정 코드를 서버로 옮기는 안내입니다. 연령 변환은 설계 문서만 있으며 `gen/` 실행 코드는 아직 없습니다.

## GitHub에 올릴 파일

```text
.gitignore
README.md
requirements.txt
requirements-insightface.txt    # 향후 전체 InsightFace SDK 사용 시 참고
age/                           # 전처리·모델·학습·평가·ONNX·모델 다운로드 코드
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
