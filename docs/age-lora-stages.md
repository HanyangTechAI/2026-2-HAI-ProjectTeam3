# Age LoRA 단계별 학습·비교 (2026-10-26)

4차 실험. 12-step 실행 검증과 구분하여, 동일한 Realistic Vision V5.1에서 학습량에 따른 validation 생성 결과를 비교한다. **FFHQ 연령 구간은 정확한 나이 정답이 아니며, 이 실험은 최종 Test 평가가 아니다.**

**결론: 새로300 optimizer update와400장 생성 비교를 완료했다. 100/300-step LoRA는 재로딩 가능한 유효한 학습 산출물이지만, 현재 설정에서 FaceID-only 대비 연령 변환을 개선했다는 근거는 없다. 오히려 나이 지시 반응이 약해졌다. 1,000-step 연장은 하지 않았고,100-step은 후속 실험용 후보로만 보존한다.**

**후속 수정:** 사용자가 지적한 주 인물 오선택과 배경·옷·표정 변형에 대응하여 얼굴 선택 및 얼굴 영역 한정 편집을 수정했다. 별도 30장 pilot와 동일 20입력의 200장 검증을 완료했다. 보호 마스크 밖 변경은 0픽셀이지만 연령 변환 품질은 여전히 미완성이다. 상세 결과와 새 기본 동작은 [age-lora-recovery.md](age-lora-recovery.md)에 기록했다. 아래 400장 수치와 설정은 수정 전 실험이며 후속 실험과 섞지 않는다.

## 보존과 학습 전 결정

- 기존 `outputs/age_lora_validation_20261026/`는 수정하지 않는다. 이전 학습 폴더의 설정·로그·전체 체크포인트·12-step export를 `preserved_sha256.json`에 지문으로 기록했다.
- 원본 `data/lora/`와 출처 JSONL을 보존한다. 새 학습용 `outputs/age_lora_stages_20261026/data/train`은 8,163장의 독립 복사본이며 각 이미지 SHA-256을 원본과 대조했다. validation/test를 복사하거나 학습에 포함하지 않는다.
- 원본 `70-120` 그룹 572장의 학습 캡션만 `elderly man/woman aged 70 or older`로 수정했다. 원래 `in his/her 70s`는 해당 구간의 상한을 잘못 제한한다. `caption_changes.csv`에 파일별 전후 문구와 이유를 남겼다. 원본 이미지·연령 그룹·성별 라벨은 바꾸지 않았다.
- 50~69는 정확한 50대/60대 정보가 없으므로 분리하지 않았다. 동일 seed에서 50과 69는 같은 문구로 매핑되므로 정밀한 나이 제어를 주장할 수 없다.
- 부족한 train 그룹은 7~9(남278/여305), 10~14(186/215), 15~19(238/369), 70~120(313/259)이다. 나머지 그룹은 성별당500장이다. 재표본화·복제로 채우지 않아 학습 빈도 불균형은 남는다.
- 새 학습은 step0부터 시작한다. 이전12에는 전체 optimizer 상태가 없고 checkpoint10만 있으며, 캡션과 재개 샘플 순서를 보완하기 때문이다. **old12와 new100의 차이는 순수한 step 수 효과만이 아니다.** new100 이후는 같은 실험의 전체 상태를 재개한다.

## 설정과 재개

모델·해상도512·rank/alpha16·batch1·gradient accumulation4·FP16·gradient checkpointing·AdamW·LR1e-4·constant/warmup0·seed42를 유지한다. 장기 cosine 설정으로 바꾸지 않았다. FaceID 생성 설정도 기존 strength0.6, steps25(실제 denoising15), CFG7.5, IP scale0.8, FaceID/Age LoRA scale1.0을 유지한다.

공식 Diffusers0.35.1 소스 해시와 기존 clipping/CPU-save 패치를 유지한다. 새 패치는 mid-epoch 재개와 체크포인트 무결성 기록이다. `SeedableRandomSampler(seed42)`와 `set_epoch`, `skip_first_batches`를 사용하여100-step 재개 시 이미 소비한400 microbatch를 건너뛴다. CPU 테스트와 Accelerate1.15 체크포인트 왕복에서 순서를 검사했다. **random flip·noise까지 중단 없는 GPU 실행과 bitwise 동일하다는 주장은 하지 않는다.**

각 step의 MSE·LR·gradient norm·가중치 변화·CUDA 메모리·동기화된 경과 시간을 JSONL로 저장한다. step 시간에는 데이터 로딩·관찰 및 직전 체크포인트 저장 비용이 들어가며 순수 GPU 연산 시간은 아니다. 전체 실행 시간은 모델 해시·로딩·마지막 저장·CPU 재로딩까지 포함한다. nvidia-smi0.5초 표본은 다른 프로그램을 포함하며 연속 peak와 다르다.

100-step마다 전체 상태와 LoRA export, 파일별 SHA-256을 저장한다. 다음 재개에서 무결성과 모델/메타데이터 지문을 확인한다. 보관 한도12로 최대1,000-step의10개 체크포인트를 모두 보존하며 기존 성공 모델이나 후보를 자동 삭제하지 않는다. 시작 여유 공간은 약115GiB였다.

## 사전 시간 산정

이전10-step 로그의 학습 루프 표시17초와 전체26.7704초를 이용해1.7~2.677초/step으로 계획했다. 이전 재개2-step 전체11.7967초는 초기화 포함으로 속도 추정에서 별도 기록했다. 이전12개 update의 두 실행 합계는38.5671초다.

| 목표 | 사전 예상 |
|---|---:|
| 100 | 170~268초 |
| 300 | 510~803초 |
| 1,000 | 1,700~2,677초 |

이는 신뢰구간이 아니고 검증 생성·CPU 중복 후보 검사 시간은 제외한다. 실제100-step 측정은 산출물 `training_summary.json`의 `estimate_after100`으로 보정한다.

## 고정 평가 원칙

validation에서10개 원본 연령 그룹 ×2개 메타데이터 성별, 셀당1장으로20장을 고정한다. 입력 선택에는 출력 품질을 사용하지 않으며, 입력·목표8/25/45/65/80·seed42·생성 설정을 모든 체크포인트에서 유지한다. 후보 검사는 자동 동일인 확정이 아니며, test는 누수 후보 조사에만 읽고 학습·checkpoint 선택·생성 평가에는 쓰지 않는다.

FaceID-only, 기존12-step, new100, 이후 허용한 체크포인트를 비교한다. **모든 조건은 수정된 동일70+ 추론 문구를 사용한다.** 정확한 목표 나이 일치가 아니라 넓은 외형 변화와 정체성 유지의 절충을 관찰한다.

ArcFace cosine, 출력 얼굴 검출률,5점 landmark 구조 참고값, 생성 시간과 VRAM을 저장한다. 선택한 InsightFace 나이 값은 이름에 `reference`를 붙인 참고값이며 MAE/정확도로 변환하지 않는다. 자동 geometry 수치와 이미지에 대한 관찰자의 육안 평가는 별개다. 실제 정식 나이 ONNX 및 사람 평가 결과는 없다.

## 재현 명령

모든 경로는 프로젝트 루트 기준이며 기존 출력은 덮어쓰지 않는다. 데이터와 모델은 다시 다운로드하지 않는다.

```bash
.venv/bin/python -m gen.prepare_lora_stages --output outputs/age_lora_stages_20261026
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m gen.train_lora \
  --config outputs/age_lora_stages_20261026/training_config.json \
  --output outputs/age_lora_stages_20261026/train
```

실제 실행 결과와 추가 학습 결정은 아래 실험 결과 절에 기록한다.

## 전체 누수 후보 검사

`gen.audit_lora_split`을 CPU4 worker ×ONNX2 thread로 실행했다. 8,663장 전체 임베딩을 확보하는 데1,181.76초가 걸렸다. 기존640 검출 실패 후 **검사에 한하여**320 크기로 재시도했고 검출 threshold0.5는 유지했다. 생성용 검출기·파이프라인에는 이 fallback을 넣지 않았다.

| Split | 검사 수 | 기존640 성공 | 320에서 추가 성공 | 최종 실패 |
|---|---:|---:|---:|---:|
| Train | 8,163 | 3,425 | 4,738 | 0 |
| Val | 400 | 170 | 230 | 0 |
| Test | 100 | 44 | 56 | 0 |

| 교차 분할 | cosine≥0.5 후보 쌍 | 관련 이미지 수 | 후보 cosine 범위 |
|---|---:|---|---|
| Train–val | 213 | train147, val62 | 0.502764~0.926658 |
| Train–test | 19 | train19, test11 | 0.508691~0.837379 |
| Val–test | 0 | 0 | 해당 없음 |

**0.5는 보정되지 않은 탐색 threshold이며 모든 쌍은 미확정 후보다.** 실제 동일인 수·누수 확정 건수가 아니다. 얼굴마다 다른 검출 크기가 정렬·점수에 영향을 줄 수 있고, 낮은 점수도 다른 인물임을 증명하지 않는다. 인물ID가 없고 가장 큰 얼굴만 검사하므로 완전한 인물 단위 분리를 보장하지 않는다. 자동 후보만으로 train을 삭제·재분할하지 않았다.

고정20장은 모두 기존640에서 검출되며 train 최대 cosine이0.439013 이하이다. test 후보를 전부 제거하고 원본 목록 순서를 뒤집어도 같은20장이 선택되는 것을 검사했다. manifest SHA-256은 `3f011e7859506dad43b51c0bc8a639dd60a0c96583751fcbd8b7389ac3f52e75`다. default 검출 성공170/400 및 후보 제외 조건으로 인한 **선택 편향**을 결과에 포함해야 한다. 기본 검출에 실패한230장의 val로 일반화할 수 없다.

## 100-Step 실제 학습

- 실제 optimizer update100개, skip0, 유한 loss/gradient, OOM·NaN·체크포인트 오류0건.
- 전체131.551초, update 간격 합119.519초, median1.18536초/step. 사전 예상170~268초보다 짧았다.100→300 추가 update 약237.1초,300→1,000 약829.8초로 보정했다. 여기에 초기화·저장·검증 생성 시간은 별도다.
- peak allocated2,525.498MiB, reserved2,704MiB, 장치 표본 최대3,581MiB.
- 학습 loss 평균0.145655, 범위0.019687~0.456063. 처음20-step 평균0.152150, 마지막20-step0.139767. 다른 배치와 noise timestep이므로 이 차이로 수렴·연령 정확도를 주장하지 않는다.
- `checkpoint-100` 전체 상태와 LoRA export를 저장했다. export256개 텐서를 새 베이스에 재로딩해 정확히 일치함을 확인했다. 최종100-step export SHA-256은 `2e9bd058eebeb4dad27697765d3f4930040a3e6b84f4a7cb3d2d4a48c3d61d3b`다.

## 300-Step 재개 결과

100-step의 수치 안정성과300장 생성 성공을 확인한 다음, 연령 반응 저하가 지속되는지 확인하는 **제한된 추가 실험**으로300까지 진행했다. `decision_after100.json`에 그 시점의 결과·가설·결정을 먼저 기록했다.100에서1,000으로 바로 연장하거나 중간에 생성 설정을 바꾸지 않았다.

| 항목 | 초기0→100 | 재개100→300 | 새 실험 합계 |
|---|---:|---:|---:|
| 실제 optimizer update | 100 | 200 | 300 |
| update skip | 0 | 0 | 0 |
| 전체 학습 실행 초 | 131.551 | 231.731 | 363.282 |
| update 간격 합 초 | 119.519 | 219.397 | 338.916 |
| step 시간 중앙값 초 | 1.1854 | 1.0727 | 실행별 값 |
| peak allocated MiB | 2,525.498 | 2,525.498 | 최대2,525.498 |
| peak reserved MiB | 2,704 | 2,704 | 최대2,704 |
| nvidia-smi 표본 peak MiB | 3,581 | 4,210 | 최대4,210 |
| loss 평균 | 0.145655 | 0.135955 | noise/배치별 변동 있음 |
| 첫20 update loss 평균 | 0.152150 | 0.120527 | 단순 수렴 판정 불가 |
| 마지막20 update loss 평균 | 0.139767 | 0.154393 | 단순 수렴 판정 불가 |

모델·optimizer·scheduler·sampler·GradScaler·모든 RNG 상태의 로딩 성공 로그가 있다. checkpoint100 무결성을 검증했고400 microbatch를 건너뛰어101부터300까지 학습했다. LR은 전 step1e-4다. 재개 loss 범위는0.009487~0.417214이며 OOM·NaN·학습/체크포인트 오류는 없었다.300-step export도 CPU 새 베이스에서256개 텐서를 비교하여 재로딩했다. SHA-256은 `19055a138b804b1db7918df809411764ce58afb782e10625c47c0914c76a21bf`다.

새300 update는 batch4로1,200 image exposure,8,163장의 약14.7%/epoch에 해당하는 초기 학습이다. 기존12를 더하여312-step 모델이라고 부르지 않는다. 보조 launcher까지 포함한300 재개 wall time은232.453초이며 위231.731초는 학습 wrapper의 측정 구간이다.

## 고정 생성 비교

20입력 ×5목표 ×4조건 = **400/400장 생성·저장**. 같은 파일 해시·prompt·seed42·GPU/CPU runtime·모델 지문·추론 코드·설정의 일치를 확인한 후 합쳤다. 원본12도 다른 조건과 동일한 새70+ 추론 문구를 사용했다. 입력별6열 비교표20개를 `comparison300/sheets/`에 저장했다.

| 조건 | ArcFace 평균 (유효 수) | 얼굴 검출 | 다중 얼굴 출력 | 8→80 참고값 변화 폭 (쌍 수) | 생성 초/장 |
|---|---:|---:|---:|---:|---:|
| FaceID-only | 0.633271 (100) | 100/100 | 5/100 | 15.65 (20) | 2.3432 |
| 기존12-step | 0.650502 (100) | 100/100 | 5/100 | 10.35 (20) | 2.4183 |
| 새100-step | 0.687510 (100) | 100/100 | 10/100 | 5.60 (20) | 2.3090 |
| 새300-step | 0.690530 (99) | 99/100 | 12/100 | 5.7895 (19) | 2.1419 |

참고값 변화 폭은 **같은 입력에서 목표80 출력의 InsightFace 나이 추정값 − 목표8 출력의 추정값**을 평균한 것이다. 실제나이·MAE·정확도·육안 평가 점수가 아니다. 큰 폭이 무조건 좋은 것도 아니다.300의누락1건을0으로 채우지 않았으며 조건별 유효 수가 다르다. 연령 참고값이 목표 순서에 비감소한 사례는 baseline8/20, old12 3/20,100 4/20,300 4/19다. 이것도 보조 관찰이지 성능 정확도가 아니다.

전체 paired identity delta는 old12 +0.017231/100쌍,100 +0.054239/100쌍,300 +0.055966/99쌍이다. **100과300을 동일한99쌍으로 직접 비교하면300−100은−0.00002084**이므로, 서로 다른 분모의0.6875→0.6905를 개선이라고 해석하면 안 된다. 통계적 유의성이나 일반화는 주장하지 않는다.

| 목표 | old12−baseline ArcFace | 100−baseline ArcFace | 300−baseline ArcFace | 100−baseline 나이 참고값 |
|---|---:|---:|---:|---:|
| 8 | +0.023804 | +0.046028 | +0.050028 (19쌍) | +2.95 |
| 25 | +0.014861 | +0.046239 | +0.047993 | −1.15 |
| 45 | +0.011738 | +0.039514 | +0.041233 | −1.55 |
| 65 | +0.021712 | +0.069676 | +0.066811 | −4.25 |
| 80 | +0.014039 | +0.069739 | +0.073467 | −7.10 |

별도 표기가 없으면20쌍이다. 높은 목표 연령에서 원본 유사도는 높아지고 노화 반응은 낮아지는 절충이 관찰된다. 어린 목표에서도 더 나이 든 참고값이 나오는 등 양끝 목표가 원본 쪽으로 축소되는 경향이다.

동기화된 생성 호출 시간 합은 baseline234.323초, old12 241.833초,100 230.899초,300 214.195초다. 합921.250초는 PNG저장·얼굴 채점·모델 로딩을 제외한다.300 평가 전체 launcher 시간은265.317초였다. 순차 실행·캐시·기기 부하 차이 때문에 checkpoint별 속도 개선을 주장하지 않는다. 추론 PyTorch peak allocated는 약4,044~4,051MiB, reserved는 약4,372~4,382MiB이다. 이는 CPU로 실행된 InsightFace나 장치 전체 VRAM을 포함하지 않는다.

## 이미지 관찰과 실패

아래는 **AI의 비블라인드 시각적 검토**이며 사람 평가가 아니다. `visual_review.json`에 검토한 sheet index0/3/4/6/9/11/12/15/17/19와 관찰을 기록했다.10개 원본 연령 구간을 포함하지만20장 전부에 대한 사람 평가 점수는 없다.

- `0000`, `0003`, `0004`: baseline은 일부 고령 목표에서 주름·턱선·목·수염 같은 변화가 있으나 모든 연령 변환이 자연스럽지는 않다.100/300에서는 아동의 눈·코·입과 유아형 윤곽이65/80 목표에서도 대부분 유지되어 연령 지시가 약해진다.
- `0006`, `0009`: 여러 목표에서 눈·코·입의 배열과 표정이 유지되지만 나이대 간 차이는 매우 작다. 피부 평활화와 표정/배경 재구성이 있으며 정확한 나이 또는 무변형을 뜻하지 않는다.
- `0015`, `0017`, `0019`: 성인·고령 입력도100/300에서 목표별 분리가 약하다. 배경 얼굴이 더 뚜렷해지고, 피부 질감·머리·안경 주변과 작은 어두운 점 등이 바뀐다. 다중 얼굴 수 증가에는 실제 배경 얼굴 보존/생성과 검출 변화가 섞여 있으므로 모두 기형 얼굴 생성으로 분류하지 않는다.
- Procrustes5점 오차 평균은 baseline0.049023, old12 0.044989,100 0.037662,300 0.036325(99건). 원본5점 배치에 가까워졌다는 참고값일 뿐 눈동자·치아·얼굴 구조의 자연스러움을 보장하지 않는다.

**실제 입력 선택 오류를 별도로 확인했다.** `crop_audit/succeeded/`에20개의 실제512 crop과 bbox 오버레이를 보존했다. `ffhq_12875`는 중앙의 큰 얼굴을 놓치고 화면 면적0.1053%의 배경 얼굴을 선택해 확대했다. `ffhq_29139`는 중앙 얼굴이 아닌 오른쪽 얼굴을 선택해 검은 경계와 두 얼굴이 포함됐다. 이는 모든 조건에 공통인 기존 검출/선택 동작이며 LoRA로 생긴 문제라고 단정하지 않는다. 이 두 입력의 ArcFace는 의도한 중앙 인물 보존을 평가하지 못한다. 나머지 다중 얼굴 출력도 동일인 채점의 주의가 필요하다.

원본20입력/400행 결과는 바꾸지 않았다. 설명용 사후18입력 분석에서도 참고 나이 변화 폭은16.444→11.500→6.167→5.889로 감소하고, identity는0.6431→0.6666→0.7097→0.7083이었다. 이는 고정 세트 대체나 유리한 점수 선택이 아니라, 잘못된 두 crop만으로 결론이 생겼는지 확인한 참고 분석이다.

300-step의 `ffhq_12875`, target8은 이미지 생성·저장 성공 뒤 얼굴 검출만 실패했다. `metric_error`와 null cosine/나이 값을 유지했으며 분모100에서 제거하지 않았다. 재시도로 유리한 이미지를 골라 대체하거나 fallback 점수를 원래 지표에 섞지 않았다. 별도 crop 진단 스크립트의 최초API 오류(`FaceAligner.faces` 없음)는 실패 로그를 보존하고 실제 `detector.detect` API로 수정하여 별도 경로에 성공 결과를 저장했다. 학습 오류와 이 진단 오류는 구분한다.

## 선정과 다음 작업

**가장 유망한 후속 실험용 후보는100-step**이다.300과 matched identity 성능이 거의 같고100/100검출, 더 적은 학습 비용, 전체 재개 상태를 갖는다. 단,100은 연령 변환 품질이 검증된 최종 모델이 아니다. **연령 편집 목적의 승자는 선정하지 않았다. 현재 기준선은 FaceID-only로 유지하고 운영 설정에 새 LoRA를 적용하지 않았다.**

1,000-step까지700 update 추가는100 측정 기준약830초와 저장/로딩, 검증약265초가 더 필요하다. 시간이 없어서 중단한 것이 아니라100→300에서 연령 반응이 회복되지 않고 검출 실패가 생겼으며 matched identity 이득도 없기 때문에 비용 대비 근거가 부족하다.300-step 하나로 장기 학습 전체가 무효하다고 결론짓지는 않는다.

가설과 확정 사실을 구분해야 한다. 같은 연령 이미지의 denoising 재구성 학습은 cross-age 편집/identity-conditioned 편집을 직접 최적화하지 않는다. 강한 FaceID 및 img2img strength0.6과의 상호작용, 학습량·caption·불균형 등이 반응을 약화시켰을 가능성이 있지만 이번 고정 설정 실험만으로 각 원인의 기여를 분리하지 못했다.

1. 의도한 주 피사체/다중 얼굴 검증, 지나치게 작은 crop 거부 및 검출 크기 전략을 별도 변경으로 검증한다. 기존 결과를 사후 덮어쓰지 않는다.
2. validation에서 strength/IP scale/Age LoRA scale 및 연령 conditioning/목적함수 가설을 한 번에 하나씩 비교한 뒤 추가 학습을 결정한다.
3. 여러 seed와 더 넓은 고정 validation, 사람의 독립적인 연령 변화·정체성·자연스러움 평가가 필요하다. 실제 나이 모델이 준비되기 전 MAE는 null로 둔다.
4. 자동 유사도 후보를 검토하고 인물 단위 분할을 강화한다. Test100은 설정·후보가 확정될 때까지 최종 품질 평가에 쓰지 않는다.

## 중간발표용 성과

- RTX4060 8GB에서 누적300 실제 optimization update,100→300 상태 재개, integrity 검사, step별 loss/LR/VRAM/time 계측, 독립 LoRA 재로딩을 완료했다.
- 원본8,663장 보존·전체 유사도 후보 검사·70+ 캡션 수정 이력·train-only 학습 경로를 확보했다.
- 원본 validation20장,5목표,4조건의400장 실측 비교표/CSV/JSON을 확보했다. 테스트100장은 품질 선택에 사용하지 않았다.
- "학습이 실행된다"와 "연령 변환 품질이 좋아진다"를 구분하여, identity 수치 향상만으로 성공을 주장할 수 없음을 확인했다.
- 검출 성공이 의도한 인물 선택 성공을 의미하지 않는 실제 실패 사례와 재현 가능한 원인 자료를 확보했다.

## 산출물과 보관

전체 산출물 루트는 `outputs/age_lora_stages_20261026/`이며 대용량/이미지 데이터는 Git에 추가하지 않는다.

| 경로 | 내용 |
|---|---|
| `preflight/` | 전체 provenance·유사도 후보·원본검출 실패·embedding cache·선정근거·고정20입력 |
| `data/train/`, `caption_changes.csv`, `preparation.json` | 독립 학습 복사본과572개 수정 근거 |
| `training_config.json`, `time_estimate_before.json` | 설정과실행 전 소요 시간 추정 |
| `train/checkpoint-100`, `checkpoint-200`, `checkpoint-300` | 모델/optimizer/scheduler/scaler/RNG 재개 상태, 각LoRA, integrity manifest |
| `train/initial/`, `train/resume-300/` | phase별 설정·환경·로그·step JSONL·GPU 표본·summary |
| `training_metrics.csv`, `training_summary.json` |300개update 계측 및100-step 기반 시간 보정 |
| `eval100/`, `eval300/` | 원본 설정·입력·모델 지문,400개PNG,생성 JSONL,개별summary/CSV |
| `comparison300/` | 입력별20개비교표,400행결과,paired 비교 CSV/JSON,통합summary |
| `metrics100.json`, `metrics300.json` | 보조변화폭·분모·matched/checkpoint 차이와층별요약 |
| `crop_audit/`, `visual_review.json` | 실제 선택crop20개,수동AI관찰 범위/한계 |
| `decision_after100.json`, `final_decision.json` | 시점별 진행/중단 및 후보 결정 |
| `preserved_sha256.json`, `tests_all.json`, `tests_all.log` | 기존 결과 지문 및실제테스트기록 |

checkpoint100/200/300과기존12를 모두 보존하고 삭제하지 않았다. 각 전체 상태는 약1.65GiB이며 LoRA는 약12.2MiB다. 마지막 저장공간 확인에서약107GiB가남아 보관 정리는 필요하지 않았다. root `train/pytorch_lora_weights.safetensors`는 마지막300 export이므로100 후보는 반드시 `train/checkpoint-100/` 경로로 식별한다.

```bash
# 전체 CPU 사전검사: 새로운 코드 지문으로 실행할 때는 새 출력 경로 사용
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m gen.audit_lora_split \
  --workers 4 --threads 2 --fallback-size 320 \
  --output outputs/age_lora_preflight_recheck

#100 전체 상태에서300까지 재개 (기존 resume-300 출력이 있으면 덮어쓰지 않음)
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

# 동일protocol의300 생성은 --baseline 없이 --checkpoint step300=.../checkpoint-300으로실행
.venv/bin/python -m gen.compare_lora --combine \
  outputs/age_lora_stages_20261026/eval100 outputs/age_lora_stages_20261026/eval300 \
  --output outputs/age_lora_stages_20261026/comparison300
```

최종 전체 테스트는111개 실행,109개 통과,2개 선택적 통합 테스트 스킵이다. 스킵 사유는 smoke ONNX 및 해당 통합 테스트용 준비 얼굴 경로 부재이며, 본 실험의 실제 GPU 학습·생성 및 얼굴 선택 검증과 구분한다.
