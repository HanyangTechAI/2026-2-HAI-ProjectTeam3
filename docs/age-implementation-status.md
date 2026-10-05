# 나이 추정 구현·실행 상태 (2026-10-05)

문서의 나이 추정 준비 단계부터 학습·평가·ONNX 전달까지 구현했습니다. **정식 학습된 나이 추정 모델은 아직 없습니다.**

2026-10-05 사용자 결정으로 학습 경로를 **ImageNet MobileNetV3 → AI Hub 단독 학습**으로 변경했습니다. UTKFace·AFAD와 Stage 1 체크포인트가 필요하지 않습니다. 전처리·학습 기본 경로는 `data/age_aihub`, 모델 출력은 `outputs/age/aihub`입니다. 기존 데이터·시험 산출물은 보존합니다.

| 단계 | 실행 결과 |
|---|---|
| 원본 검사 | 제공된 AI Hub TS/TL ZIP 2개 CRC·SHA-256 검사 통과. 이미지와 JSON 50쌍 일치 |
| 라벨 변환 | 촬영 당시 나이 `age_past` 사용. 현재 나이 `age_now` 미사용. 인물 1명, 나이 1~29세 |
| 시험 전처리 | 50장 중 48장 RGB/EXIF 처리·SCRFD 검출·5점 224px 정렬 성공. 다중 얼굴 2장 제외 |
| 전체 표본 전처리 | AI Hub 단독 기본 경로 `data/age_aihub`에 48장 및 CSV 생성. 기존 `data/age_stage2`와 원본은 보존 |
| 인물별 분할 | train/val/test = 48/0/0. 동일인 사진을 나눠 평가하지 않음. 빈 검증 split 학습 거부 확인 |
| 모델 구현 | MobileNetV3-Large 1280→101, σ=2, KL(target‖prediction)+L1, softmax 기댓값 |
| 학습 구현 | AI Hub 단독 모드 기본, ImageNet 초기화, train 전용 증강·나이대 sampler, AdamW/cosine/AMP, 최저 val MAE 선택, epoch 시간 기록 |
| 평가 구현 | 출처별·나이대별 MAE/CS@5/표본 수, 그림, InsightFace genderage 기준선·실패/동일 부분집합 비교 |
| 합성 입력 실행 | AI Hub 기본 모드에서 ImageNet 초기화와 무작위 RGB 배열 2 batch 학습, 가중치 저장·재로드·ONNX opset 17 export 검증 |
| ONNX 실행 검증 | 합성 val 4장 및 실제 정렬 얼굴 20장 모두 PyTorch/ORT 최대 차이 0.01세 미만 |
| 검사 | 단위·실제 얼굴 정렬 검사 13개 통과. AI Hub 단독 기본값·다른 출처 거부, 미검출, 다중 얼굴 선택 규칙, 정렬 일치, 인물/중복 누수, 경계 라벨·gradient·지표·경로 검사 포함 |

검증 모델의 나이는 임의 출력입니다. `outputs/smoke/*.onnx`와 체크포인트는 합성 입력으로 2 batch만 학습한 시험용이고 실제 나이 성능을 보여주지 않습니다. ONNX 메타데이터에 `smoke_artifact=true`가 있으며 정식 평가와 기본 추론에서 사용을 거부합니다. 정식 MAE 표나 최종 `age_estimator.onnx`는 아직 생성하지 않았습니다.

현재 환경은 Windows/Python 3.13, PyTorch 2.14.1+cpu/torchvision 0.29.1+cpu입니다. 공식 InsightFace SDK의 C++ 확장 설치가 실패하여 공식 v0.7의 SCRFD·face_align·attribute 원본 Python 코드와 공식 buffalo_l의 det_10g/genderage ONNX 가중치를 내려받아 그대로 사용합니다. GPU 서버에서도 같은 코드를 사용하여 학습·추론 정렬 차이를 방지합니다. 사용 소스와 모델의 체크섬은 `models/insightface/provenance.json`에 있습니다.

남은 정식 학습 조건:

1. 여러 독립 인물의 승인된 AI Hub 전체 데이터를 확보·검사·전처리. `age_past`와 인물별 분할을 유지합니다.
2. GPU-hai의 사용 가능한 접속 경로 및 프로젝트·데이터 경로. 현재 PC에서는 CUDA GPU가 확인되지 않았습니다.
3. AI Hub 단독 학습·최저 val 모델 고정·최종 test 및 기준선 평가·ONNX 전달.

실행 명령은 [README](../README.md), 출처와 확보 상태는 [data/README](../data/README.md)를 참고하세요. 자료 부족을 임의 라벨·이미지·동일인 분할로 보완하지 않았습니다.
