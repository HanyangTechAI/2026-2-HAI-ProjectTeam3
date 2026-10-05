# 2026-2-HAI-ProjectTeam3

AI Hub 얼굴 데이터로 나이를 추정하는 프로젝트입니다.

- **모델:** ImageNet 사전학습 MobileNetV3-Large 전체 파인튜닝
- **데이터:** AI Hub만 사용, 촬영 당시 나이(`age_past`)를 정답으로 사용
- **흐름:** 얼굴 검출·정렬 → 학습 → 평가 → ONNX 내보내기

## 실행

CUDA 지원 PyTorch와 가상환경을 준비한 뒤 실행합니다. AI Hub 이미지와 JSON은 `data/raw/aihub/`에 별도로 배치합니다.

```bash
pip install -r requirements.txt
python -m age.setup_models
python -m age.prep --source aihub --raw-root data/raw/aihub --device cpu
python -m age.train --device cuda
```

학습 결과는 `outputs/age/aihub/`에 저장됩니다. 데이터·가중치·학습 결과는 Git에 포함하지 않습니다.

현재 나이 추정 파이프라인을 구현했으며, 실제 데이터 학습과 연령변환 모델 구현은 아직 진행 전입니다.

자세한 설치·평가·ONNX 사용법은 [서버 실행 안내](docs/server-setup.md), 전체 설계는 [설계 문서](docs/superpowers/specs/2026-09-27-age-photobooth-models-design.md)를 참고하세요.
