# 2026-2-HAI-ProjectTeam3

AI Hub 데이터로 MobileNetV3-Large 나이 추정 모델을 학습하는 프로젝트입니다.

기준: `docs/superpowers/specs/2026-09-27-data-prep-preprocessing-model-guide.md`의 데이터 준비·전처리·나이 추정 단계. UI와 생성 모델은 별도 단계입니다.

2026-10-05 결정: **AI Hub 데이터만으로 학습**합니다. ImageNet 사전학습 MobileNetV3에서 바로 시작하며 UTKFace·AFAD 학습이나 기존 나이 추정 체크포인트는 필요하지 않습니다.

GitHub 업로드 파일과 Linux 서버 준비 순서는 [서버 실행 안내](docs/server-setup.md)를 참고하세요. 코드와 문서를 Git에 올리고 데이터·모델 가중치·가상환경·실행 결과는 서버에서 별도로 준비합니다.

MobileNetV3-Large(ImageNet V2)의 classifier 마지막 Linear를 101개 logit으로 교체합니다. 정답은 σ=2인 0~100세 가우시안 분포, 손실은 `KL(target || prediction) + L1(expected_age, age)`입니다. 설계 문서와 실행 지침의 KL 방향이 서로 달라 실행 지침의 안정적인 API 방향을 채택했습니다. 예측은 softmax 확률의 나이 기댓값입니다.

## 환경

GPU-hai 서버에서는 CUDA에 맞는 PyTorch/torchvision을 먼저 설치하고 아래 의존성을 설치합니다. 현재 PC의 Python은 3.13, PyTorch는 CPU 버전입니다.

```powershell
python -m venv .venv --system-site-packages --without-pip
python -m pip --python .venv/Scripts/python.exe install -r requirements.txt
.venv/Scripts/python.exe -m age.setup_models
```

Linux에서는 `.venv/bin/python`을 사용합니다. Windows에서 `ensurepip`가 실패하는 환경을 피하도록 `--without-pip`로 만들고 외부 `python -m pip --python .venv/Scripts/python.exe ...`로 설치합니다.

이하 명령의 `python`은 가상환경의 Python입니다. PowerShell에서 `.\.venv\Scripts\Activate.ps1`로 활성화하거나, `python`을 `.venv/Scripts/python.exe`로 바꿔 실행하세요. Linux에서는 `source .venv/bin/activate`로 활성화합니다.

InsightFace 0.7.3 전체 SDK는 Windows에서 C++ Build Tools가 필요합니다. 이 프로젝트는 기본적으로 **동일한 공식 buffalo_l ONNX 가중치와 공식 v0.7 SCRFD·정렬·genderage Python 소스만** 내려받아 사용합니다. 3D mesh 확장을 빌드하지 않습니다. 원본 코드 수정 없이 별도의 빈 패키지 초기화 파일로 로드하며 출처·SHA-256·MIT LICENSE는 `models/insightface`에 보존합니다. 전체 SDK가 필요한 생성 단계에서는 `requirements-insightface.txt`를 사용하세요.

## 데이터 준비와 전처리

먼저 `data/README.md`에서 출처와 상태를 확인합니다. 다운로드·승인 없이 데이터를 임의로 생성하여 실제 학습에 넣지 않습니다.

```powershell
# 받은 AI Hub 전체 ZIP의 무결성·라벨 대응·인물 수 먼저 확인
python -m age.prep --source aihub --raw-root "C:\data\aihub_archives" --extract-to data/raw/aihub --inspect-only
# 50장 시험은 data/age_aihub/pilot_aihub_50에 별도 저장
python -m age.prep --source aihub --raw-root data/raw/aihub --limit 50
# 전체 전처리: data/age_aihub/{images,train.csv,val.csv,test.csv}
python -m age.prep --source aihub --raw-root data/raw/aihub --device cpu
```

AI Hub JSON의 `age_past`를 정답 나이, `id`를 인물 그룹으로 사용합니다. `age_now`는 정답으로 사용하지 않습니다. 현재 확보된 인물 1명 표본은 정상 train/val/test를 만들 수 없으므로, 여러 인물의 전체 데이터를 확보해야 실제 학습을 시작할 수 있습니다. 기본 의존성은 CPU용 ONNX Runtime이므로 얼굴 전처리는 `--device cpu`를 사용합니다. 전처리 GPU 사용은 호환되는 `onnxruntime-gpu` 환경을 별도로 준비한 뒤 선택합니다. MobileNet GPU 학습은 CUDA 지원 PyTorch를 사용합니다.

RGB 읽기 → EXIF 방향 적용 → 손상·크기·픽셀 중복 검사 → buffalo_l SCRFD 검출 → 5점 similarity 정렬 → 224×224 저장 → 출처별 고정 seed 8:1:1 분할 순서입니다. 학습 다중 얼굴은 제외하고 실시간 추론은 가장 큰 얼굴을 택합니다. 같은 인물과 중복 픽셀은 하나의 그룹으로 묶습니다. 작은 데이터는 비율을 근사하며 그룹 수가 3 미만이면 val/test가 비어 학습을 막습니다. 증강과 나이대 균형 샘플링은 train에만 적용합니다.

`age.face.FaceAligner`는 RGB uint8 입력·출력, 미검출/다중 얼굴/잘못된 랜드마크는 ValueError입니다. `crop(..., size=512, context=True)`로 생성팀용 여유 있는 크롭도 얻을 수 있습니다. 얼굴 이미지나 임베딩은 실패 로그에 넣지 않습니다.

## 학습·평가·전달

```powershell
# AI Hub 단독: ImageNet에서 시작, batch 128, lr 5e-4, 30 epoch, AdamW/cosine/AMP
python -m age.train --device cuda
# VRAM이 부족하면 --batch-size를 줄이고 변경값은 run.json에 기록
# 첫 epoch로 GPU에서 걸리는 시간 확인. 정식 30 epoch는 다른 output으로 시작
python -m age.train --device cuda --epochs 1 --output outputs/age/aihub_timing

# 체크포인트 결정 후 test를 한 번 평가, 동일 목록의 genderage 기준선 비교
python -m age.evaluate --checkpoint outputs/age/aihub/best.pt --csv data/age_aihub/test.csv --output outputs/eval/aihub --device cuda --baseline

# 실제 정렬된 val 이미지로 PyTorch/ORT 차이 < 0.01세 검증
python -m age.export_onnx --checkpoint outputs/age/aihub/best.pt --csv data/age_aihub/val.csv --output outputs/age/age_estimator.onnx
python -m age.infer --onnx outputs/age/age_estimator.onnx --image "촬영사진.jpg"
```

학습마다 새로운 `--output`을 지정합니다. 최고 val MAE 모델만 `best.pt`로 저장하고 `run.json`에 버전, 데이터 체크섬, 하이퍼파라미터, loss 방향을 남깁니다. test는 학습 모델 선택에 사용하지 않습니다. 평가 결과는 출처별·나이대별 MAE/CS@5/표본 수 CSV와 그래프입니다. 기준선 실패율은 `baseline_failures.csv`로 확인하고 성공한 동일 부분집합에서 모델 결과도 별도 저장합니다. 이미 평가한 체크포인트/test 조합은 같은 출력 폴더에서 반복 실행하지 않습니다.

기본 `--stage aihub`는 AI Hub 외의 출처가 섞인 CSV를 거부합니다. 첫 epoch 이후 예상 남은 시간을 출력하고, 학습·검증 소요 시간을 `history.csv`의 `epoch_seconds`에 기록합니다. 기존 공개 데이터 Stage 1/2는 명시적으로 선택할 때만 사용되는 이전 실험 경로입니다.

ONNX 입력은 float32 `[N,3,224,224]`, RGB 0~1 후 ImageNet mean/std 정규화이고 출력은 `[N]` 나이입니다. 정규화는 그래프 밖, softmax·기댓값은 그래프 안입니다. 전달 시 `.onnx`와 동명 `.json` 및 평가표를 함께 전달하세요. 실시간 추론은 원본·반전 평균, 목표 나이는 `[a-10,a,a+10,a+20]`을 정수 반올림 후 3~90으로 제한합니다. 경계에서 목표가 겹칠 수 있습니다.

## 구현 검증

```powershell
python -m unittest discover -s tests -v
python -m age.smoke
# ImageNet 초기화도 검증하려면 새 출력 폴더 사용
python -m age.smoke --pretrained --output outputs/smoke_aihub
```

`age.smoke`는 합성 RGB 배열로 AI Hub 학습 모드의 2개 학습 batch와 ONNX 수치 일치를 시험합니다. 체크포인트·ONNX에는 smoke 표시를 남기고 정식 평가를 거부합니다. 이 결과의 나이·MAE는 모델 성능이 아닙니다. 실제 학습 모델 완성은 전체 AI Hub 전처리, GPU 학습, 최종 test 평가를 마친 뒤에 판단합니다.

공식 참고: [torchvision MobileNetV3](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_large.html), [InsightFace 모델과 사용 조건](https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md), [SCRFD v0.7](https://github.com/deepinsight/insightface/blob/v0.7/python-package/insightface/model_zoo/scrfd.py).
