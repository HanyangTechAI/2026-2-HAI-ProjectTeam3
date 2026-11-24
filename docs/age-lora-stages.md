# Age LoRA 단계별 학습·비교 (2026-10-26)

2026-10-26의 4차 실험 기록이다. 12-step 실행 검증과 구분하여, 동일한 Realistic Vision V5.1에서 학습량에 따른 validation 생성 결과를 비교했다. **FFHQ 라벨은 정확한 나이가 아닌 연령 구간이며, 평가 범위는 validation이다. 최종 Test 평가는 수행하지 않았다.**

새 실험에서 optimizer update 300회와 이미지 400장 비교를 완료했다. 100/300-step LoRA는 재로딩할 수 있었지만, **FaceID-only보다 나이 지시 반응이 약했다.** 1,000-step 연장은 하지 않았고, 100-step은 후속 실험용 후보로 보존했다.

**후속 수정:** 주 인물 오선택과 배경·옷·표정 변형에 대응하여 얼굴 선택 및 얼굴 영역 한정 편집을 수정했다. 별도 30장 pilot와 동일한 입력 20장의 200장 검증을 완료했다. 보호 마스크 밖 변경은 0픽셀이었지만 충분한 연령 변화와 자연스러움은 미해결로 남았다. 상세 결과와 당시 변경한 기본 동작은 [age-lora-recovery.md](age-lora-recovery.md)에 기록했다. 아래 400장 수치와 설정은 수정 전 실험이며 후속 실험과 별개다.

## 보존과 학습 전 결정

- 기존 `outputs/age_lora_validation_20261026/`는 수정하지 않았다. 이전 학습 폴더의 설정·로그·전체 체크포인트·12-step export를 `preserved_sha256.json`에 지문으로 기록했다.
- 원본 `data/lora/`와 출처 JSONL을 보존했다. 새 학습용 `outputs/age_lora_stages_20261026/data/train`은 8,163장의 독립 복사본이며 각 이미지 SHA-256을 원본과 대조했다. validation/test는 복사하거나 학습에 포함하지 않았다.
- 원본 `70-120` 그룹 572장의 학습 캡션만 `elderly man/woman aged 70 or older`로 수정했다. 원래 `in his/her 70s`는 해당 구간의 상한을 잘못 제한한다. `caption_changes.csv`에 파일별 전후 문구와 이유를 남겼다. 원본 이미지·연령 그룹·성별 라벨은 바꾸지 않았다.
- 50~69는 정확한 50대/60대 정보가 없으므로 분리하지 않았다. 동일 seed에서 50과 69는 같은 문구로 매핑되어, 이 구간 내 정밀한 나이 제어에는 한계가 있다.
- 부족한 train 그룹은 7~9(남 278/여 305), 10~14(186/215), 15~19(238/369), 70~120(313/259)이다. 나머지 그룹은 성별당 500장이다. 재표본화·복제로 채우지 않아 학습 빈도 불균형은 남는다.
- 새 학습은 step 0부터 시작했다. 이전 12-step에는 전체 optimizer 상태가 없고 checkpoint 10만 있으며, 캡션과 재개 샘플 순서도 보완했기 때문이다. **old12와 new100 비교에는 step 수 외에 이 변경들의 영향도 포함된다.** new100 이후는 같은 실험의 전체 상태를 재개했다.

## 설정과 재개

모델·해상도 512·rank/alpha 16·batch 1·gradient accumulation 4·FP16·gradient checkpointing·AdamW·LR 1e-4·constant/warmup 0·seed 42를 유지했다. 장기 cosine 설정으로 바꾸지 않았다. FaceID 생성 설정도 기존 strength 0.6, steps 25(실제 denoising 15), CFG 7.5, IP scale 0.8, FaceID/Age LoRA scale 1.0을 유지했다.

공식 Diffusers 0.35.1 소스 해시와 기존 clipping/CPU-save 패치를 유지했다. 새 패치는 mid-epoch 재개와 체크포인트 무결성 기록이다. `SeedableRandomSampler(seed42)`와 `set_epoch`, `skip_first_batches`를 사용하여 100-step 재개 시 이미 소비한 400 microbatch를 건너뛰었다. CPU 테스트와 Accelerate 1.15 체크포인트 왕복에서 순서를 검사했다. **random flip·noise를 포함한 중단 없는 GPU 실행과의 bitwise 동일성은 미검증이다.**

각 step의 MSE·LR·gradient norm·가중치 변화·CUDA 메모리·동기화된 경과 시간을 JSONL로 저장했다. step 시간에는 데이터 로딩·관찰 및 직전 체크포인트 저장 비용이 들어간다. 전체 실행 시간은 모델 해시·로딩·마지막 저장·CPU 재로딩까지 포함한다. nvidia-smi의 0.5초 간격 표본은 다른 프로그램을 포함하며 연속 peak와 다르다.

100-step마다 전체 상태와 LoRA export, 파일별 SHA-256을 저장하고 재개 시 무결성과 모델/메타데이터 지문을 확인하도록 설정했다. 보관 한도는 12로, 최대 1,000-step의 10개 체크포인트를 모두 보존하고 기존 모델이나 후보를 자동 삭제하지 않도록 했다. 시작 여유 공간은 약 115GiB였다.

## 사전 시간 산정

이전 10-step 로그의 학습 루프 표시 17초와 전체 26.7704초를 이용해 1.7~2.677초/step으로 계획했다. 이전 재개 2-step 전체 11.7967초는 초기화를 포함하므로 속도 추정에서 별도 기록했다. 이전 12개 update의 두 실행 합계는 38.5671초다.

| 목표 | 사전 예상 |
|---|---:|
| 100 | 170~268초 |
| 300 | 510~803초 |
| 1,000 | 1,700~2,677초 |

이는 검증 생성·CPU 중복 후보 검사 시간을 제외한 계획 범위이며 신뢰구간은 아니다. 실제 100-step 측정에 따른 보정값은 산출물 `training_summary.json`의 `estimate_after100`에 기록했다.

## 고정 평가 원칙

validation에서 10개 원본 연령 그룹 × 2개 메타데이터 성별, 셀당 1장으로 20장을 고정했다. 입력 선택에는 출력 품질을 사용하지 않았으며, 입력·목표 8/25/45/65/80·seed 42·생성 설정을 모든 체크포인트에서 유지했다. 동일인 여부는 후보 검사만으로 확정할 수 없다. test는 누수 후보 조사에만 읽고 학습·checkpoint 선택·생성 평가에는 사용하지 않았다.

FaceID-only, 기존 12-step, new100, 이후 추가한 체크포인트를 비교했다. **모든 조건은 수정된 동일한 70+ 추론 문구를 사용했다.** 관찰 대상은 넓은 외형 변화와 정체성 유지의 절충이며, 정확한 목표 나이 일치 여부는 평가하지 않았다.

ArcFace cosine, 출력 얼굴 검출률, 5점 landmark 구조 참고값, 생성 시간과 VRAM을 저장했다. InsightFace 나이 값은 이름에 `reference`를 붙인 참고값으로, MAE/정확도와 구분했다. 자동 geometry 수치와 AI의 시각적 검토도 별개다. 당시 정식 나이 ONNX 및 사람 평가 결과는 없었다.

## 당시 실행 명령

아래는 2026-10-26 실험의 실행 기록이며 현재 실행 안내가 아니다. 모든 경로는 프로젝트 루트 기준이다. 기존 출력은 보존했고 데이터와 모델은 다시 다운로드하지 않았다.

```bash
.venv/bin/python -m gen.prepare_lora_stages --output outputs/age_lora_stages_20261026
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m gen.train_lora \
  --config outputs/age_lora_stages_20261026/training_config.json \
  --output outputs/age_lora_stages_20261026/train
```

실제 실행 결과와 추가 학습 결정은 아래 실험 결과 절에 기록했다.

## 전체 누수 후보 검사

`gen.audit_lora_split`을 CPU 4 worker × ONNX 2 thread로 실행했다. 8,663장 전체 임베딩을 확보하는 데 1,181.76초가 걸렸다. 기존 640 검출 실패 후 **검사에 한하여** 320 크기로 재시도했고 검출 threshold 0.5는 유지했다. 생성용 검출기·파이프라인에는 이 fallback을 넣지 않았다.

| Split | 검사 수 | 기존 640 성공 | 320에서 추가 성공 | 최종 실패 |
|---|---:|---:|---:|---:|
| Train | 8,163 | 3,425 | 4,738 | 0 |
| Val | 400 | 170 | 230 | 0 |
| Test | 100 | 44 | 56 | 0 |

| 교차 분할 | cosine ≥ 0.5 후보 쌍 | 관련 이미지 수 | 후보 cosine 범위 |
|---|---:|---|---|
| Train–val | 213 | train 147, val 62 | 0.502764~0.926658 |
| Train–test | 19 | train 19, test 11 | 0.508691~0.837379 |
| Val–test | 0 | 0 | 해당 없음 |

**0.5는 보정되지 않은 탐색 threshold이며 표의 수치는 모두 미확정 후보 수다.** 실제 동일인 수·누수 확정 건수는 확인되지 않았다. 얼굴마다 다른 검출 크기가 정렬·점수에 영향을 줄 수 있고, 낮은 점수만으로 다른 인물임을 확정할 수도 없다. 인물 ID가 없고 가장 큰 얼굴만 검사하므로 인물 단위 분리 검증에는 한계가 있다. 자동 후보만으로 train을 삭제·재분할하지 않았다.

고정 20장은 모두 기존 640에서 검출되며 train 최대 cosine은 0.439013 이하이다. test 후보를 전부 제거하고 원본 목록 순서를 뒤집어도 같은 20장이 선택되는 것을 검사했다. manifest SHA-256은 `3f011e7859506dad43b51c0bc8a639dd60a0c96583751fcbd8b7389ac3f52e75`다. default 검출 성공 170/400 및 후보 제외 조건에 따른 **선택 편향**이 있다. 기본 검출에 실패한 val 230장으로 일반화할 수 없다.

## 100-Step 실제 학습

- 실제 optimizer update 100개, skip 0, 유한 loss/gradient, OOM·NaN·체크포인트 오류 0건.
- 전체 131.551초, update 간격 합 119.519초, median 1.18536초/step. 사전 예상 170~268초보다 짧았다. 100→300 추가 update 약 237.1초, 300→1,000 약 829.8초로 보정했다. 초기화·저장·검증 생성 시간은 별도다.
- peak allocated 2,525.498MiB, reserved 2,704MiB, 장치 표본 최대 3,581MiB.
- 학습 loss 평균 0.145655, 범위 0.019687~0.456063. 처음 20-step 평균 0.152150, 마지막 20-step 평균 0.139767. 서로 다른 배치와 noise timestep에서 측정한 값으로, 수렴·연령 정확도 판정에는 부족하다.
- `checkpoint-100` 전체 상태와 LoRA export를 저장했다. export의 256개 텐서를 새 베이스에 재로딩해 정확히 일치함을 확인했다. 최종 100-step export SHA-256은 `2e9bd058eebeb4dad27697765d3f4930040a3e6b84f4a7cb3d2d4a48c3d61d3b`다.

## 300-Step 재개 결과

100-step의 수치 안정성과 300장 생성을 확인한 다음, 연령 반응 저하가 지속되는지 확인하는 **제한된 추가 실험**으로 300까지 진행했다. `decision_after100.json`에 그 시점의 결과·가설·결정을 먼저 기록했다. 100에서 1,000으로 바로 연장하거나 중간에 생성 설정을 바꾸지 않았다.

| 항목 | 초기 0→100 | 재개 100→300 | 새 실험 합계 |
|---|---:|---:|---:|
| 실제 optimizer update | 100 | 200 | 300 |
| update skip | 0 | 0 | 0 |
| 전체 학습 실행 초 | 131.551 | 231.731 | 363.282 |
| update 간격 합 초 | 119.519 | 219.397 | 338.916 |
| step 시간 중앙값 초 | 1.1854 | 1.0727 | 실행별 값 |
| peak allocated MiB | 2,525.498 | 2,525.498 | 최대 2,525.498 |
| peak reserved MiB | 2,704 | 2,704 | 최대 2,704 |
| nvidia-smi 표본 peak MiB | 3,581 | 4,210 | 최대 4,210 |
| loss 평균 | 0.145655 | 0.135955 | noise/배치별 변동 있음 |
| 첫 20 update loss 평균 | 0.152150 | 0.120527 | 단순 수렴 판정 불가 |
| 마지막 20 update loss 평균 | 0.139767 | 0.154393 | 단순 수렴 판정 불가 |

모델·optimizer·scheduler·sampler·GradScaler·모든 RNG 상태의 로딩 성공 로그가 있다. checkpoint 100 무결성을 검증했고 400 microbatch를 건너뛰어 101부터 300까지 학습했다. LR은 모든 step에서 1e-4다. 재개 loss 범위는 0.009487~0.417214이며 OOM·NaN·학습/체크포인트 오류는 없었다. 300-step export도 CPU의 새 베이스에서 256개 텐서를 비교하여 재로딩했다. SHA-256은 `19055a138b804b1db7918df809411764ce58afb782e10625c47c0914c76a21bf`다.

새 300 update는 유효 batch 4로 1,200회 image exposure, 8,163장의 약 14.7%/epoch에 해당하는 초기 학습이다. 기존 12-step과는 별도로 step 0부터 학습했으므로 누적 312-step 모델이 아니다. 보조 launcher까지 포함한 300-step 재개 wall time은 232.453초이며 위 231.731초는 학습 wrapper의 측정 구간이다.

## 고정 생성 비교

입력 20장 × 목표 5개 × 조건 4개 = **400/400장 생성·저장**. 같은 파일 해시·prompt·seed 42·GPU/CPU runtime·모델 지문·추론 코드·설정의 일치를 확인한 후 합쳤다. 기존 12-step도 다른 조건과 동일한 새 70+ 추론 문구를 사용했다. 입력별 6열 비교표 20개를 `comparison300/sheets/`에 저장했다.

| 조건 | ArcFace 평균 (유효 수) | 얼굴 검출 | 다중 얼굴 출력 | 8→80 참고값 변화 폭 (쌍 수) | 생성 초/장 |
|---|---:|---:|---:|---:|---:|
| FaceID-only | 0.633271 (100) | 100/100 | 5/100 | 15.65 (20) | 2.3432 |
| 기존 12-step | 0.650502 (100) | 100/100 | 5/100 | 10.35 (20) | 2.4183 |
| 새 100-step | 0.687510 (100) | 100/100 | 10/100 | 5.60 (20) | 2.3090 |
| 새 300-step | 0.690530 (99) | 99/100 | 12/100 | 5.7895 (19) | 2.1419 |

참고값 변화 폭은 **같은 입력에서 목표 80 출력의 InsightFace 나이 추정값 − 목표 8 출력의 추정값**을 평균한 것이다. 실제 나이·MAE·정확도·육안 평가 점수와 구분되는 보조 지표이며, 큰 폭만으로 품질을 판단할 수 없다. 300-step의 누락 1건을 0으로 채우지 않았으며 조건별 유효 수가 다르다. 연령 참고값이 목표 순서에 따라 비감소한 사례는 baseline 8/20, old12 3/20, 100-step 4/20, 300-step 4/19다. 이 비율도 참고값의 반응을 나타내며 나이 정확도와는 다르다.

전체 paired identity delta는 old12 +0.017231/100쌍, 100-step +0.054239/100쌍, 300-step +0.055966/99쌍이다. **100-step과 300-step을 동일한 99쌍으로 직접 비교하면 300−100은 −0.00002084**다. 서로 다른 분모의 0.6875→0.6905와 달리, 같은 쌍에서는 identity 이득이 관찰되지 않았다. 통계적 유의성과 일반화는 미검증이다.

| 목표 | old12−baseline ArcFace | 100−baseline ArcFace | 300−baseline ArcFace | 100−baseline 나이 참고값 |
|---|---:|---:|---:|---:|
| 8 | +0.023804 | +0.046028 | +0.050028 (19쌍) | +2.95 |
| 25 | +0.014861 | +0.046239 | +0.047993 | −1.15 |
| 45 | +0.011738 | +0.039514 | +0.041233 | −1.55 |
| 65 | +0.021712 | +0.069676 | +0.066811 | −4.25 |
| 80 | +0.014039 | +0.069739 | +0.073467 | −7.10 |

별도 표기가 없으면 20쌍이다. 높은 목표 연령에서 원본 유사도는 높아지고 노화 반응은 낮아지는 절충이 관찰된다. 어린 목표에서도 더 나이 든 참고값이 나오는 등 양끝 목표가 원본 쪽으로 축소되는 경향이다.

동기화된 생성 호출 시간 합은 baseline 234.323초, old12 241.833초, 100-step 230.899초, 300-step 214.195초다. 합 921.250초는 PNG 저장·얼굴 채점·모델 로딩을 제외한다. 300-step 평가 전체 launcher 시간은 265.317초였다. 순차 실행·캐시·기기 부하가 달라 checkpoint별 속도 효과를 분리할 수 없다. 추론 PyTorch peak allocated는 약 4,044~4,051MiB, reserved는 약 4,372~4,382MiB이다. 이는 CPU로 실행된 InsightFace나 장치 전체 VRAM을 포함하지 않는다.

## 이미지 관찰과 실패

아래는 **AI의 비블라인드 시각적 검토**다. `visual_review.json`에 입력 20장 중 검토한 sheet index 0/3/4/6/9/11/12/15/17/19와 관찰을 기록했다. 10개 원본 연령 구간을 포함하며, 사람 평가 결과는 없다.

- `0000`, `0003`, `0004`: baseline은 일부 고령 목표에서 주름·턱선·목·수염 같은 변화가 있으나 모든 연령 변환이 자연스럽지는 않다. 100/300-step에서는 아동의 눈·코·입과 유아형 윤곽이 65/80 목표에서도 대부분 유지되어 연령 지시가 약해진다.
- `0006`, `0009`: 여러 목표에서 눈·코·입의 배열과 표정이 유지되지만 나이대 간 차이는 매우 작다. 피부 평활화와 표정/배경 재구성도 관찰된다.
- `0015`, `0017`, `0019`: 성인·고령 입력도 100/300-step에서 목표별 분리가 약하다. 배경 얼굴이 더 뚜렷해지고, 피부 질감·머리·안경 주변과 작은 어두운 점 등이 바뀐다. 다중 얼굴 수 증가에는 실제 배경 얼굴 보존/생성과 검출 변화가 섞여 있어 기형 얼굴 생성과 구분이 필요하다.
- Procrustes 5점 오차 평균은 baseline 0.049023, old12 0.044989, 100-step 0.037662, 300-step 0.036325(99건). 원본 5점 배치와의 차이를 나타내는 참고값이며, 눈동자·치아·얼굴 구조의 자연스러움은 별도 평가가 필요하다.

**입력 선택 오류를 별도로 확인했다.** `crop_audit/succeeded/`에 20개의 실제 512 crop과 bbox 오버레이를 로컬 보존했다. 공개 기록의 익명 사례 A는 중앙의 큰 얼굴을 놓치고 화면 면적 0.1053%의 배경 얼굴을 선택해 확대했다. 익명 사례 B는 중앙 얼굴이 아닌 오른쪽 얼굴을 선택해 검은 경계와 두 얼굴이 포함됐다. 원래 표본 ID는 이 공개 사본에서 생략했다. 이는 모든 조건에 공통인 기존 검출/선택 동작에서 발생했다. 이 두 입력의 ArcFace는 의도한 중앙 인물 보존을 평가하지 못한다. 나머지 다중 얼굴 출력도 동일인 채점에 주의가 필요하다.

원본 입력 20장/400행 결과는 바꾸지 않았다. 사후 입력 18장 분석에서도 참고 나이 변화 폭은 16.444→11.500→6.167→5.889로 감소하고, identity는 0.6431→0.6666→0.7097→0.7083이었다. 잘못된 두 crop의 영향을 확인하기 위한 참고 분석으로, 원래 고정 세트 평가와 별도로 기록했다.

300-step의 익명 사례 A, target 8은 이미지 생성·저장 뒤 얼굴 검출만 실패했다. `metric_error`와 null cosine/나이 값을 유지했으며 검출률 분모 100에 포함했다. 원래 지표에는 재시도 이미지나 fallback 점수를 반영하지 않았다. 별도 crop 진단 스크립트의 최초 API 오류(`FaceAligner.faces` 없음)는 실패 로그를 보존하고 `detector.detect` API로 수정하여 별도 경로에 성공 결과를 저장했다. 학습 오류와 이 진단 오류는 구분한다.

## 당시 선정과 후속 과제

후속 실험용 후보로 100-step을 보존했다. 300-step과 matched identity 성능이 거의 같고 100/100 검출, 더 적은 학습 비용, 전체 재개 상태를 갖는다. 연령 변환 개선은 확인하지 못해 최종 모델은 선정하지 않았다. 당시 기준선은 FaceID-only로 유지하고 운영 설정에 새 LoRA를 적용하지 않았다.

1,000-step까지 700 update 추가에는 100-step 측정 기준 약 830초와 저장/로딩, 검증 약 265초가 더 필요할 것으로 추정했다. 중단 이유는 시간 부족이 아니라 100→300에서 연령 반응이 회복되지 않고 검출 실패가 생겼으며 matched identity 이득도 없었기 때문이다. 300-step을 넘는 장기 학습의 효과는 미검증으로 남았다.

같은 연령 이미지의 denoising 재구성 학습은 cross-age 편집/identity-conditioned 편집을 직접 최적화하지 않는다. 강한 FaceID 및 img2img strength 0.6과의 상호작용, 학습량·caption·불균형 등이 반응을 약화시켰을 가능성이 있지만 이번 고정 설정 실험만으로 각 원인의 기여를 분리하지 못했다.

실험 종료 시점에 정리한 후속 과제는 다음과 같다.

1. 의도한 주 피사체/다중 얼굴 검증, 지나치게 작은 crop 거부 및 검출 크기 전략을 별도 변경으로 검증한다. 기존 결과를 사후 덮어쓰지 않는다.
2. validation에서 strength/IP scale/Age LoRA scale 및 연령 conditioning/목적함수 가설을 한 번에 하나씩 비교한 뒤 추가 학습을 결정한다.
3. 여러 seed와 더 넓은 고정 validation, 사람의 독립적인 연령 변화·정체성·자연스러움 평가가 필요하다. 실제 나이 모델이 준비되기 전 MAE는 null로 둔다.
4. 자동 유사도 후보를 검토하고 인물 단위 분할을 강화한다. Test 100장은 설정·후보가 확정될 때까지 최종 품질 평가에 쓰지 않는다.

## 실험 요약

- RTX 4060 8GB에서 누적 300회 실제 optimization update, 100→300 상태 재개, integrity 검사, step별 loss/LR/VRAM/time 계측, 독립 LoRA 재로딩을 완료했다.
- 원본 8,663장 보존·전체 유사도 후보 검사·70+ 캡션 수정 이력·train-only 학습 경로를 확보했다.
- 원본 validation 20장, 목표 5개, 조건 4개의 400장 실측 비교표/CSV/JSON을 확보했다. 테스트 100장은 품질 선택에 사용하지 않았다.
- 학습·저장·재로딩은 완료했으며, identity 수치 증가와 함께 연령 지시 반응 저하가 관찰됐다.
- 검출 성공이 의도한 인물 선택 성공을 의미하지 않는 실제 실패 사례와 재현 가능한 원인 자료를 확보했다.

## 산출물과 보관

전체 산출물 루트는 `outputs/age_lora_stages_20261026/`이며 대용량/이미지 데이터는 Git에 추가하지 않는다.

| 경로 | 내용 |
|---|---|
| `preflight/` | 전체 provenance·유사도 후보·원본 검출 실패·embedding cache·선정 근거·고정 입력 20장 |
| `data/train/`, `caption_changes.csv`, `preparation.json` | 독립 학습 복사본과 572개 수정 근거 |
| `training_config.json`, `time_estimate_before.json` | 설정과 실행 전 소요 시간 추정 |
| `train/checkpoint-100`, `checkpoint-200`, `checkpoint-300` | 모델/optimizer/scheduler/scaler/RNG 재개 상태, 각 LoRA, integrity manifest |
| `train/initial/`, `train/resume-300/` | phase별 설정·환경·로그·step JSONL·GPU 표본·summary |
| `training_metrics.csv`, `training_summary.json` | 300개 update 계측 및 100-step 기반 시간 보정 |
| `eval100/`, `eval300/` | 원본 설정·입력·모델 지문, 400개 PNG, 생성 JSONL, 개별 summary/CSV |
| `comparison300/` | 입력별 20개 비교표, 400행 결과, paired 비교 CSV/JSON, 통합 summary |
| `metrics100.json`, `metrics300.json` | 보조 변화 폭·분모·matched/checkpoint 차이와 층별 요약 |
| `crop_audit/`, `visual_review.json` | 실제 선택 crop 20개, AI 시각적 검토 범위/한계 |
| `decision_after100.json`, `final_decision.json` | 시점별 진행/중단 및 후보 결정 |
| `preserved_sha256.json`, `tests_all.json`, `tests_all.log` | 기존 결과 지문 및 실제 테스트 기록 |

checkpoint 100/200/300과 기존 12-step을 모두 보존했다. 각 전체 상태는 약 1.65GiB이며 LoRA는 약 12.2MiB다. 당시 마지막 저장 공간 확인에서 약 107GiB가 남아 보관 정리는 필요하지 않았다. root `train/pytorch_lora_weights.safetensors`는 마지막 300-step export이며, 100-step 후보의 경로는 `train/checkpoint-100/`다.

다음은 당시 소스 기준 실행 기록과 재검사 예시다. 이후 얼굴 선택·합성 정책이 변경되어 현재 코드 실행 결과와는 구분한다.

```bash
# 전체 CPU 사전검사: 새로운 코드 지문으로 실행할 때는 새 출력 경로 사용
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m gen.audit_lora_split \
  --workers 4 --threads 2 --fallback-size 320 \
  --output outputs/age_lora_preflight_recheck

# 100 전체 상태에서 300까지 재개 (기존 resume-300 출력이 있으면 덮어쓰지 않음)
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m gen.train_lora \
  --config outputs/age_lora_stages_20261026/training_config.json \
  --output outputs/age_lora_stages_20261026/train \
  --resume-from-checkpoint checkpoint-100 --max-train-steps 300

# 아래는 최초 실험의 실제 실행 명령 기록이다. 당시 소스는 original_inference_source/에 보존했다.
# 현재 코드는 얼굴 선택/합성 정책이 달라졌으므로 기존 결과 재현으로 취급하지 않는다.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TQDM_DISABLE=1 \
.venv/bin/python -m gen.compare_lora \
  --manifest outputs/age_lora_stages_20261026/preflight/manifest.json \
  --config outputs/age_lora_validation_20261026/faceid_age/config.json \
  --baseline \
  --checkpoint old12=outputs/age_lora_validation_20261026/train_retry/pytorch_lora_weights.safetensors \
  --checkpoint step100=outputs/age_lora_stages_20261026/train/checkpoint-100 \
  --insightface-reference --output outputs/age_lora_stages_20261026/eval100

# 동일 protocol의 300 생성은 --baseline 없이 --checkpoint step300=.../checkpoint-300으로 실행
.venv/bin/python -m gen.compare_lora --combine \
  outputs/age_lora_stages_20261026/eval100 outputs/age_lora_stages_20261026/eval300 \
  --output outputs/age_lora_stages_20261026/comparison300
```

당시 최종 전체 테스트는 111개 실행, 109개 통과, 2개 선택적 통합 테스트 스킵이다. 스킵 사유는 smoke ONNX 및 해당 통합 테스트용 준비 얼굴 경로 부재이며, 본 실험의 실제 GPU 학습·생성 및 얼굴 선택 검증과 구분한다.
