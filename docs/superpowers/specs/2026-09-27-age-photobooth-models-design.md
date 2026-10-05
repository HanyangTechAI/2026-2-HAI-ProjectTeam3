# 나이 추정 & 연령 변환 포토부스 — 데이터/모델 설계

> 2026-10-05 변경 결정: 나이 추정은 **AI Hub 71415만으로 ImageNet 사전학습 MobileNetV3를 직접 학습**한다. 아래의 UTKFace+AFAD → AI Hub Stage 1/2 설명은 이전 설계이며 현재 선행 조건이 아니다. 모델·손실·전처리·인물별 분할은 유지한다. 최신 실행 경로와 명령은 저장소 루트 `README.md`를 따른다. 생성 모델 계획은 그대로 유지한다.

- 작성일: 2026-09-27
- 범위: 데이터셋 준비, 나이 추정 모델, 연령 변환 모델, 평가 (Phase 1, ~10/29 중간발표)
- 범위 밖: 라즈베리파이 GPIO/카메라/웹 UI/프린터 (Phase 2, 별도 스펙)

## 0. 결정 요약

| 항목 | 결정 | 이유 |
|---|---|---|
| 나이 추정 | MobileNetV3-Large 단일 모델, DLDL-v2, ONNX | 앙상블 없이 충분한 MAE, 경량 |
| 연령 변환 | Realistic Vision V5.1(SD1.5) img2img + IP-Adapter FaceID Plus v2 + Age LoRA 1개 | 학습은 LoRA만, 5주 내 현실적 |
| ControlNet | 기본 미사용 | 평가에서 구도 붕괴가 확인될 때만 depth 추가 |
| 얼굴 검출/정렬 | insightface `buffalo_l` (SCRFD + 5점) | IP-Adapter FaceID가 어차피 요구, 학습·추론 전처리 통일 |
| 나이 추정 데이터 | UTKFace + AFAD-Full → AI Hub 71415 파인튜닝 | AI Hub 승인 전에도 전체 파이프라인 가동 |
| LoRA 데이터 | FFHQ-Aging (+AI Hub 승인 후) | 1024px 고해상도 + 10개 연령 그룹 라벨 |
| CACD | 미사용 | 250×250 저해상도, 동일인 시간축 장점은 IP-Adapter 구조에서 불필요 |
| GPU | GPU-hai 서버 (24GB+) | 학습·데모 추론 모두 |

## 1. 데이터 준비

### 1.1 데이터셋

| 용도 | 데이터 | 규모 | 비고 |
|---|---|---|---|
| 나이 추정 | UTKFace | ~23k, 0–116세 | 파일명에 age/gender |
| 나이 추정 | AFAD-Full | ~164k, 아시아인 | 20–30대 편중 → 균형 샘플러 |
| 나이 추정 파인튜닝 | AI Hub 71415 | ~50k, 한국인 | **즉시 신청**. 라벨 포맷은 수령 후 변환기 추가 |
| Age LoRA | FFHQ-Aging | 70k 중 ~10k 사용 | 라벨 CSV(age_group, confidence, gender, head pose) |

### 1.2 전처리 규칙

- 모든 얼굴: insightface `buffalo_l`로 검출 → 5점 랜드마크 similarity transform 정렬. 이미 크롭된 UTKFace/AFAD도 동일하게 재정렬(추론 입력과 분포 일치).
- 검출 실패 이미지는 제외하고 데이터셋별 제외 수를 로그로 남긴다(중간발표 "전처리 결과"에 사용).
- 나이 추정용: 224px 크롭 → `data/age/{train,val,test}.csv` (`path,age,gender,source`), 8:1:1 랜덤 분할. AI Hub는 **인물 ID 기준 분할**.
- LoRA용: FFHQ-Aging에서 `age_group_confidence > 0.8`, `|yaw| < 30°` 필터 → 연령그룹(10) × 성별(2) 셀당 ~500장 → 512px 리사이즈 → `data/lora/metadata.jsonl` (`file_name`, `text`).

### 1.3 캡션 템플릿

`a photo of a {phrase}` 형식. 연령 그룹 → 문구 매핑 하나(코드 내 dict)를 학습과 추론이 공유한다.

| 그룹 | 남 / 여 |
|---|---|
| 0–2 | baby boy / baby girl |
| 3–6 | young boy / young girl |
| 7–9 | boy / girl |
| 10–14 | preteen boy / preteen girl |
| 15–19 | teenage boy / teenage girl |
| 20–29 | man in his 20s / woman in her 20s |
| 30–39 | man in his 30s / woman in her 30s |
| 40–49 | man in his 40s / woman in her 40s |
| 50–69 | middle-aged man in his 50s or 60s / middle-aged woman in her 50s or 60s |
| 70+ | elderly man in his 70s / elderly woman in her 70s |

- FFHQ 캡션에는 인종 표현을 넣지 않는다. `korean`은 AI Hub 이미지 캡션에만 붙인다.

### 1.4 디렉터리

```
data/raw/{utkface,afad,ffhq_aging,aihub}/
data/age/{images/, train.csv, val.csv, test.csv}
data/lora/{images/, metadata.jsonl}
```

### 1.5 리스크

- FFHQ 원본은 Google Drive 할당량 제한이 잦다 → 필요한 ~10k장만 받고 1–2일 여유.
- AI Hub 승인 지연 → 공개 데이터만으로 중간발표 가능하도록 설계됨. 승인 시 추가 파인튜닝.

## 2. 나이 추정 모델

### 2.1 모델

- torchvision `mobilenet_v3_large` (ImageNet 가중치), classifier 마지막 층을 `Linear(1280 → 101)`로 교체 (0–100세).
- DLDL-v2: 타깃 = 정답 나이 중심 가우시안(σ=2) 분포. 손실 = `KL(pred ‖ target) + L1(Σ pᵢ·i, age)`. 예측 = 기댓값.

### 2.2 학습

| 단계 | 데이터 | 설정 |
|---|---|---|
| Stage 1 | UTKFace + AFAD | 224px, batch 128, AdamW lr 5e-4, cosine, 30 epoch, AMP, 나이대 WeightedRandomSampler |
| Stage 2 | AI Hub | Stage 1에서 시작, lr 1e-4, 10 epoch |

증강: 좌우 반전, RandomResizedCrop(scale 0.8–1.0), ColorJitter.

### 2.3 평가

- 데이터셋별 test MAE, CS@5, 나이대별 MAE 막대그래프.
- 베이스라인: insightface 내장 genderage 나이 추정.
- 참고 목표치(보장 아님): UTKFace MAE ≈ 5, AFAD MAE ≈ 3.5.

### 2.4 ONNX

- softmax + 기댓값을 그래프에 포함해 출력이 나이 스칼라 하나가 되도록 export (opset 17).
- export 스크립트 내 검증: 동일 입력에서 PyTorch vs onnxruntime 차이 < 0.01세 (assert).
- 기본 실행: 서버 onnxruntime-gpu. 라즈베리파이 CPU 실행은 선택.

### 2.5 추론 흐름

1. 검출 → 가장 큰 얼굴 1개 선택
2. 5점 정렬 → 원본 + 좌우반전 예측 평균 = 현재 나이 `a`
3. 목표 나이 `[a−10, a, a+10, a+20]`, `[3, 90]`으로 clamp
4. 목표 나이 → 1.3의 연령 그룹 문구 (성별은 insightface genderage 출력 재사용)

## 3. 연령 변환 모델

### 3.1 파이프라인

- 베이스: Realistic Vision V5.1 (SD1.5). LoRA 학습과 추론에 동일 체크포인트 사용.
- 정체성: IP-Adapter FaceID Plus v2 (insightface 임베딩 + CLIP 이미지 임베딩) + 공식 FaceID LoRA.
- 나이: 자체 학습 Age LoRA. `set_adapters`로 FaceID LoRA / Age LoRA 가중치를 각각 제어.
- img2img: 얼굴 주변을 여유 있게 512px로 크롭한 이미지를 입력, strength ≈ 0.6 (옷·배경·구도 유지).
- 샘플러: DPM++ 2M Karras, 25 step. 한 세션 내 seed 고정.
- "현재" 컷: 원본 사진 그대로 사용 (흑백 열전사 출력이라 색감 차이는 문제되지 않음).
- 생성 시간 목표: 3장 합계 ~10초 이내 (24GB GPU).

### 3.2 Age LoRA 학습

- diffusers `examples/text_to_image/train_text_to_image_lora.py` 그대로 사용 (자체 학습 코드 없음).
- `--train_data_dir data/lora`, rank 16, lr 1e-4, 512px, batch 8, ~8k step, fp16, UNet만 학습.
- 1000 step마다 체크포인트 저장 + 10개 연령 그룹 검증 프롬프트 샘플링 → 최적 체크포인트 선택.

### 3.3 튜닝 노브

strength, IP-Adapter scale(+`s_scale`), Age LoRA 가중치. 정체성 보존과 나이 변화가 서로 반대로 당기므로 하드코딩하지 않고 3.4의 그리드로 결정한다.

### 3.4 평가

- 테스트 얼굴 ~50장: LoRA 학습에 쓰지 않은 FFHQ 이미지 + AI Hub + 동의받은 팀원 사진.
- **Age MAE**: 생성 이미지를 2장 나이 추정기로 측정한 나이 vs 목표 나이. 자기채점 편향 방지를 위해 insightface 나이로 교차 확인.
- **Identity**: 입력 vs 출력 ArcFace 코사인 유사도.
- **Ablation**: (a) 프롬프트만 → (b) +IP-Adapter → (c) +IP-Adapter+Age LoRA.
- **그리드**: strength {0.5, 0.6, 0.7} × IP scale {0.6, 0.8, 1.0} × LoRA 가중치 {0.6, 0.8, 1.0}. 결과 CSV.
- ControlNet(depth) 추가 조건: 그리드 결과에서 얼굴 위치·포즈가 원본에서 눈에 띄게 벗어날 때.

## 4. 서버 인터페이스 (Phase 2 연결점)

- `POST /booth` (이미지) → `{age, image: 4컷 합성 PNG}`
- 얼굴 미검출 → HTTP 422, 파이 UI에 "얼굴이 안 보여요" 표시.

## 5. 코드 구성

| 파일 | 역할 |
|---|---|
| `age/prep.py` | 검출·정렬, 나이 추정 CSV + LoRA metadata.jsonl 생성 |
| `age/train.py` | MobileNetV3 DLDL 학습·평가 (~150줄, argparse) |
| `age/export_onnx.py` | ONNX export + PyTorch/ORT 출력 일치 assert |
| `gen/pipeline.py` | 모델 1회 로드, `age_shift(img, targets) -> list[Image]` |
| `gen/eval.py` | ablation·그리드 실행, Age MAE / Identity CSV |
| (셸 명령) | diffusers LoRA 학습 |

설정 파일 없음. 모든 하이퍼파라미터는 argparse 인자.



단순화
                얼굴 사진
                   │
                   ▼
          SCRFD 얼굴 검출
                   │
                   ▼
            5-point 정렬
                   │
          ┌────────┴────────┐
          │                 │
          ▼                 ▼
 MobileNetV3           InsightFace
 + DLDL-v2            Face Embedding
          │                 │
          ▼                 │
      나이 = 23             │
                            │
사용자가 목표 나이 = 70 ────┤
                            │
                            ▼
                     Stable Diffusion
                     Realistic Vision
                            +
                   IP-Adapter FaceID
                            +
                       Age LoRA
                            │
                            ▼
                    70세 얼굴 생성
