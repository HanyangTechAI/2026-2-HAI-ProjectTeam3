# 2026-2-HAI-ProjectTeam3

얼굴 사진에서 나이를 추정하고, 같은 인물의 다양한 연령대 모습을 생성하는 **연령 변환 포토부스** 프로젝트입니다.

## 주요 기능

| 기능 | 모델·학습 방식 | 현재 상태 |
|---|---|---|
| 나이 추정 | MobileNetV3-Large를 AI Hub 데이터로 전체 파인튜닝 | 전처리·학습·평가·ONNX 코드 구현, 실제 학습 전 |
| 연령 변환 | Realistic Vision + IP-Adapter FaceID + Age LoRA | 설계 완료, 구현 예정 |

전체 흐름: **얼굴 검출·정렬 → 나이 추정 → 목표 연령 설정 → 연령 변환 이미지 생성**

## 나이 추정 실행

CUDA 지원 PyTorch와 가상환경을 준비한 뒤 실행합니다. AI Hub 이미지와 JSON은 `data/raw/aihub/`에 별도로 배치합니다.

```bash
pip install -r requirements.txt
python -m age.setup_models
python -m age.prep --source aihub --raw-root data/raw/aihub --device cpu
python -m age.train --device cuda
```

학습 결과는 `outputs/age/aihub/`에 저장됩니다. 데이터·가중치·학습 결과는 Git에 포함하지 않습니다.

자세한 설치·평가·ONNX 사용법은 [서버 실행 안내](docs/server-setup.md), 전체 설계는 [설계 문서](docs/superpowers/specs/2026-09-27-age-photobooth-models-design.md)를 참고하세요.
