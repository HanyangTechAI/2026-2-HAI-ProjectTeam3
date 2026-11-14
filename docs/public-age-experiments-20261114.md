# 공개 데이터 다운로드 및 나이 추정 실험 기록 (2026-11-13~14)

## 1. 범위와 결론

이 문서는 공개 얼굴 데이터 확보, 검증, 공개 데이터 전용 나이 추정 학습, 정렬 실패 조사, padding 재시도, 정렬 데이터 재학습과 고정 좌우 반전 평가를 기록한다. 2026-11-14 작성 시점의 로컬 JSON/CSV와 소스 코드를 대조했다. **최종 제품 배포나 상업 이용 승인 문서가 아니라 연구 결과와 재현 한계의 기록**이다. 문서 작성 과정에서는 test 이미지를 열람하지 않았고, 새 학습·추론·평가도 실행하지 않았다. 기존 test 평가의 저장된 집계만 인용한다.

- 최초 `public` 모델은 비정렬 test 20,223장에서 MAE **3.332714세**, CS@5 **0.783514**였다. 이 결과는 최초 모델에만 해당한다.
- 비정렬 val 24,313장의 학습 중 최저 MAE는 `public` 3.465184, `balanced` 3.371056, `weak_crop` 3.418837이다. 각 실험은 ImageNet에서 새로 시작했으며 순차 fine-tuning이 아니다.
- 첫 정렬은 train 162,823장 중 29,455장, val 24,313장 중 7,475장만 채택했다. AFAD val이 44장만 남아 비교 모집단이 심하게 달라졌다.
- 미검출에만 검은 padding을 적용한 재시도는 train **160,590장**, val **23,838장**을 채택했다. 검출 복구는 정렬 품질 보증이 아니며 특히 AFAD의 검은 출력 영역이 크다.
- 같은 retry val 23,838장과 `flip=true`에서 정렬 crop 모델은 **3.237303세 / 0.794236**, 정렬 weak 모델은 **3.276775세 / 0.788741**였다. 이 기준의 잠정 연구 후보는 **`public_aligned_crop_retry_v1/best.pt`, epoch 24**다.
- 후속 모델의 독립 test 결과는 없다. 최초 test MAE와 후속 val MAE를 직접 비교하여 test 성능 개선을 주장하지 않는다. 동일인·근접 중복 누수, 추정 나이 라벨, 소수 고령 표본과 이용 조건 제한이 남아 있다.
- 문서만 새로 추가했다. Git 변경 명령·commit·push·원격 업로드는 이 문서 작성 작업에서 수행하지 않았다. 향후 공개 대상은 별도 검토해야 하며 모델 및 출력의 무조건 공개를 승인하는 기록이 아니다.

MAE는 평균 절대 나이 오차(세), CS@5는 절대 오차가 5세 이하인 비율이다. 비율을 정확도나 신뢰도로 해석하지 않는다. 표의 소수는 표시 자리에서 반올림했으며 원본 전체 정밀도는 아래 근거 파일에 있다.

Git 업로드 범위, 실제 공유 링크, 데이터 메타데이터 제외 및 대용량 파일 검토는 [업로드 준비 기록](age-experiment-upload-scope-20261114.md)에 별도로 정리했다. 사용자 승인 전에는 아직 업로드 완료 상태가 아니다.

## 2. 경로와 근거

| 구분 | 실제 경로 / 의미 |
|---|---|
| 코드 작업 디렉터리 | `/workspace/hai/branches/main` |
| 공유 데이터 루트 | `/workspace/hai/shared/data` |
| 다운로드 기록 | `<데이터 루트>/{aaf,megaage_asian,afad,fgnet}/acquisition.json` 및 각 출처·이용 조건·검증 사본 |
| 통합 인벤토리 | `<데이터 루트>/reports/public_datasets_20261113/` |
| 비정렬 공개 학습 manifest | `<데이터 루트>/age_public/{train,val,test}.csv`, `summary.json` |
| 첫 정렬 | `<데이터 루트>/age_public_aligned_v1/` |
| padding retry 정렬 | `<데이터 루트>/age_public_aligned_retry_v1/` |
| 학습 기록 | `/workspace/hai/outputs/age/<실험명>/{run.json,history.csv,best.pt}` |
| 최초 test 평가 | `/workspace/hai/outputs/eval/public_initial_20261114/` |
| 대응 val 비교 | `/workspace/hai/outputs/age/compare_aligned_val_balanced_flip_v1/`, `compare_aligned_val_weak_crop_v1/`, `compare_aligned_retry_balanced_flip_v1/` |
| 정렬 재학습 flip 평가 | `/workspace/hai/outputs/age/eval_aligned_flip_{crop,weak}_v1/` |
| 얼굴 모델 / 런타임 | `/workspace/hai/shared/models/insightface/{provenance.json,python_runtime/,models/buffalo_l/}` |

저장소의 `data`·`models`·`outputs` 아래 경로는 공유 실파일을 가리킬 수 있다. Git에 디렉터리 이름이나 심볼릭 링크가 존재하는 것과 실파일이 전송되는 것은 다르다. manifest의 `path`는 이 실험에서는 절대 경로이며 서버 이동 시 그대로 유효하다는 보장은 없다. 데이터별 `samples.csv`는 해당 데이터셋 디렉터리 기준 상대 경로이고, 통합 인벤토리 CSV는 공유 데이터 루트 기준이다. 기존 문서의 FFHQ/AI Hub 과거 준비 기록과 현재 서버의 확보 상태를 혼동하지 않는다.

주요 구현 근거는 `age/prepare_public.py`, `age/catalog.py`, `age/train.py`, `age/train_public_balanced.py`, `age/train_public_weak_crop.py`, `age/dataset.py`, `age/model.py`, `age/evaluate.py`, `age/face.py`, `age/align_public.py`, `age/align_public_retry.py`, `age/compare_aligned_val.py`, `age/infer.py`다. 이 문서 작성에서는 이 파일들을 수정하지 않았다.

## 3. 공개 데이터 다운로드와 검증

### 3.1 확보 출처와 사용 판단

| 데이터셋 | 확보량 / 라벨 범위 | 원본 출처 | 이번 사용 |
|---|---|---|---|
| All-Age-Faces (AAF) | 13,322 publisher serial, original/정렬 각 13,322장, 합계 26,644 표본 이미지, 2~80세 | <https://github.com/JingchunCheng/All-Age-Faces-Dataset>, 공식 Dropbox ZIP | 명시적 데이터 이용 허가 미확인으로 학습 제외 |
| MegaAge-Asian | train 40,000 + test 3,945 = 43,945 JPEG, 전부 178x218, 0~69세 | <https://github.com/zyx2012/Age_estimation_BMVC2017>, <https://mmlab.ie.cuhk.edu.hk/projects/MegaAge/>, 공식 Dropbox ZIP | 비상업 연구용 공개 학습/검증 및 기존 test 평가 |
| AFAD-Full | 165,501 JPEG, 실제 디렉터리 라벨 15~72세 | <https://afad-dataset.github.io/>, <https://github.com/John-niu-07/tarball> | 학술 연구용 공개 학습/검증 및 기존 test 평가 |
| FG-NET | 1,002장, 82명, 0~69세, 68점 landmark 1,002개, 서로 다른 나이의 동일인 무순서 쌍 5,794개 | <https://yanweifu.github.io/FG_NET_data/>, 해당 공개 미러의 `FGNET.zip` | 평가/참조 후보만 확보; 이번 학습과 나이 평가에는 사용하지 않음 |

AAF 공식 다운로드는 `https://www.dropbox.com/s/a0lj1ddd54ns8qy/All-Age-Faces%20Dataset.zip?dl=1`, MegaAge-Asian은 `https://www.dropbox.com/s/x0gyfp12tozivjh/megaage_asian.zip?dl=1`이다. 일시적 redirect URL을 영구 출처로 취급하지 않는다. AFAD의 고정 원본 revision은 `0511da136747c31c7f9a7efdde44e0c8f902457b`이며 `AFAD-Full.tar.xzaa`부터 `AFAD-Full.tar.xzbt`까지 46개 조각을 순서대로 결합했다. 모든 조각의 바이트 수·SHA-256·publisher Git blob SHA-1은 `afad/acquisition.json`에 보존되어 있다.

### 3.2 아카이브 식별

| 아카이브 | bytes | SHA-256 |
|---|---:|---|
| AAF `All-Age-Faces Dataset.zip` | 365,164,027 | `0dcebf92031e43b987d78fb3621ab7f0bcb3af62aaf8871de060681c1f08db77` |
| MegaAge-Asian `megaage_asian.zip` | 290,421,152 | `bb2abe5efa0572ffc1bdb5810eabd6177b95b5d8972b5c3aa221bea49adddf2c` |
| AFAD 결합 `archives/AFAD-Full.tar.xz` | 1,539,051,212 | `5a11a3ee8759c959f15819ce4a56fe3d530631fa0a6530ce1c79e95ce402be39` |
| FG-NET `FGNET.zip` | 46,221,653 | `228acddeba96e469f2ded9475ee87f4ab16398de27a2f3ec35a266a46ef2f7fc` |

2026-11-13의 저장된 검증 기록은 다음을 확인했다. 이는 당시 전체 디코딩 기록을 인용한 것이며, 이번 문서 작성에서 test 이미지를 다시 열거나 재검증한 것이 아니다.

- AAF: ZIP CRC 전 항목 통과, 경로 이탈·symlink·특수 파일 거부, Pillow verify 후 reopen/full load. 최종 검증은 추출 파일 39,971개와 368,268,657 bytes의 원본 대응, 표본 26,644행과 전체 디코딩 26,644건을 확인했다. 아카이브 내 유효 이미지 26,646개 중 2개는 보조 이미지로 표본에서 제외된다. 손상·미상 라벨·annotation error는 모두 0이다.
- MegaAge-Asian: ZIP CRC·경로·symlink 검사 통과, 전체 43,945장 디코딩 실패 0, name/age/dis 목록의 행 대응 보존. 최종 검증에서 원본 파일 CRC 43,953개와 이미지 해시 43,945개를 확인했다. 원래 70열 분포 파일은 수정 없이 보존했고 유한·비음수·행 합 허용 오차 검사를 통과했다.
- AFAD: 46개 조각이 고정 publisher blob과 일치하고 결합 XZ 무결성이 통과했다. 이미지 165,501장 전체 디코딩 실패 0. 배포본 내 `AFAD-Full.txt`의 165,515항목 중 이미지 165,501개, 비이미지 14개와 대조했다.
- FG-NET: 이미지 1,002장과 point 파일 1,002개 대응·디코딩 통과, landmark error 0. 좌표 extent warning 1건은 원값 그대로 보존했다. byte 및 decoded RGB 동일 이미지 그룹은 모두 0. 같은 나이 조합 14개를 나이가 다른 쌍에서 제외했고, 쌍 행을 독립 사람 수나 자동 학습 표본 수로 세지 않는다.

### 3.3 배포 설명과 실제 파일의 차이

AFAD 소개 페이지는 164,432장, 15~40세, female 63,680 / male 100,752를 설명하지만 실제 Full 배포본은 165,501장, 15~72세, female 63,982 / male 101,519다. 이를 오류로 간주해 임의 relabel·삭제하지 않고 실제 배포본 기준으로 기록했다. 고령 라벨이 있다고 충분하거나 신뢰할 만한 고령 커버리지가 확보된 것은 아니다. 통합 인벤토리의 AFAD 50~69세는 318장, 70세 이상은 42장뿐이다.

MegaAge 프로젝트 설명의 40,000장과 실제 43,945장의 차이는 공식 train 40,000장 외 test 3,945장이 별도 포함되기 때문이다. 원래 split과 세 목록을 유지했다. MegaAge 나이는 human-in-the-loop 나이 비교로 얻은 **추정 나이**이며 출생일로 검증한 실제 나이가 아니다. 이번 DLDL 학습은 원래 `*_age.txt`의 scalar를 중심으로 Gaussian target을 만들며, 보존한 `*_dis.txt`의 원래 posterior를 직접 target으로 사용한 실험이 아니다.

AAF의 publisher serial 13,322개는 검증된 서로 다른 실제 사람 13,322명을 뜻하지 않는다. 서로 다른 serial의 동일 사진이 확인되어 원 설명의 uniqueness를 그대로 전제할 수 없다. original과 `aglined faces`라는 원본 오타 디렉터리의 정렬본을 독립 사람 두 명으로 세지 않는다.

### 3.4 중복과 통합 인벤토리

| 범위 | 확인된 중복 / 충돌 | 해석 |
|---|---|---|
| AAF | byte/pixel 중복 각 374그룹, 중복 파일 768개, 초과 파일 394개; 다른 serial 중복 374그룹, 공식 split 교차 192그룹, 나이 충돌 324그룹 | 원본은 보존, 학습 보류 |
| MegaAge-Asian | unique byte SHA 43,197개, 중복 650그룹, 공식 train/test 교차 중복 152그룹 | 공식 원본 보존과 학습 manifest의 누수 방지는 별도 |
| AFAD | byte 중복 408그룹, 초과 이미지 473개, 인물 ID 없음 | 파일 중복 제외만으로 인물 분리 입증 불가 |
| 통합 4종 | 이미지 행 237,092개, 중복 해시 1,432그룹, 나이 충돌 1,190그룹, `hashes_verified=true` | 데이터별 그룹 수를 단순 합산하여 독립 사람 수로 쓰지 않음 |

통합 보고서 `reports/public_datasets_20261113/summary.json`의 후보 CSV 행 수는 generation 21,316, age supervision 150,057, evaluation-only 4,947, review/excluded 28,527이다. generation 후보는 최소 변 256 이상, age-supervision 후보는 최소 변 64 이상이며 upscale하지 않았다. usage 분류는 permission review 26,644, research training candidate 203,618, conflicting labels review 1,694, heldout duplicate excluded 31, exact duplicate excluded 158, evaluation-only 4,947이다. 이들은 각각의 정책으로 분류된 인벤토리이며 후보 CSV끼리 단순 합산하거나 아래 최종 학습 수와 동일시하지 않는다. 특히 이번 `age.prepare_public`은 AFAD/MegaAge의 원래 `samples.csv`에서 별도로 준비했으므로 최소 변 64의 후보 CSV만 학습한 실험이 아니다.

## 4. 공개 학습 분할과 라벨

`age/prepare_public.py`는 AFAD와 MegaAge-Asian만 읽는다. 같은 SHA-256 그룹에 다른 나이가 있으면 그룹 전체를 제외한다. 나이가 같은 그룹은 공식 test를 우선하는 한 행만 유지하여 해당 사진과 복제본을 train/val에서 제거한다. 원본 이미지를 변경하지 않고 `age_public/sources/` 링크와 새 CSV를 만든다.

| 출처 | 원본 이미지 | train | val | test | 최종 유지 |
|---|---:|---:|---:|---:|---:|
| AFAD | 165,501 | 131,756 | 16,416 | 16,482 | 164,654 |
| MegaAge-Asian | 43,945 | 31,067 | 7,897 | 3,741 | 42,705 |
| 합계 | 209,446 | **162,823** | **24,313** | **20,223** | **207,359** |

`age_public/summary.json`의 나이 충돌 제외 **1,885장**과 중복 초과 제외 **202장**으로 `209,446 - 1,885 - 202 = 207,359`가 맞는다. 데이터별 유지 수는 위 split 수를 합한 값이다. 통합 카탈로그의 1,694/158 등 다른 제외 수와 정책·범위가 달라 혼용하지 않는다.

split key는 `age-public-v1:<원본 SHA-256>`의 SHA-256 앞 8자리 정수를 `%100`한 bucket이다. AFAD는 `<10` test, `<20` val, 나머지 train이고 MegaAge의 공식 test가 아닌 행은 `<20` val, 나머지 train이다. 이는 목표 비율에 가까운 결정적 이미지 해시 분할이지, 나이 층화 또는 인물 분할이 아니다. 공식 MegaAge test는 정리 후 3,741장으로 남는다. seed 문자열 `age-public-v1`과 학습 RNG seed `42`는 별개다.

AFAD는 출처 제공의 reported chronological age이며 독립 검증된 생년월일 정답은 아니다. MegaAge는 estimated age다. `person_id`는 두 출처 모두 확보되지 않았다. 따라서 path/SHA 교차 검사 통과는 **동일인 비중복을 뜻하지 않는다**. 같은 사람의 다른 사진, resize/re-encode된 사진, cross-dataset 동일인, 모델 사전학습 데이터와의 겹침은 배제하지 못한다. 성별·나이도 민감한 얼굴 관련 메타데이터로 보호해야 한다.

원래 val의 연령 구성이 AFAD 10대 2,407 / 20대 9,828 / 30대 3,952 / 40대 189 / 50대 22 / 60대 14 / 70대 4장, MegaAge 0~9세 887 / 10대 918 / 20대 2,048 / 30대 1,549 / 40대 1,235 / 50대 849 / 60대 411장이다. 전체 MAE는 이 관측 분포를 따른 sample-weighted 평균이며 source/연령별 macro 평균이 아니다. 전체 수치가 좋아도 소수 고령군의 품질이 좋다는 뜻은 아니다.

## 5. 학습 계약과 실험 순서

### 5.1 공통 설정

| 설정 | 값 / 의미 |
|---|---|
| 모델 | MobileNetV3-Large, 최종 classifier를 101 logits(0~100세)로 교체 |
| 초기화 | torchvision `IMAGENET1K_V2`, 모든 실험 ImageNet에서 새로 시작 |
| 예측 | `softmax(logits)`와 0~100의 내적, 정수 argmax가 아님 |
| 손실 | `KL(target \|\| prediction) + L1(expected_age, scalar_age)`, 각 가중치 1 |
| target | 0~100에서 정규화한 Gaussian, sigma=2 |
| 최적화 | AdamW, initial lr=0.0005, weight decay=0.01, cosine scheduler, 30 epochs |
| 실행 | batch=128, workers=4, seed=42, CUDA, AMP=true, `smoke_steps=0`, 사전학습 사용 |
| 검증 | 224x224 resize, ImageNet mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`; 학습 중 flip TTA 없음 |
| checkpoint 선택 | 각 epoch의 전체 val MAE가 엄격히 개선되고 optimizer update가 있을 때 `best.pt` 갱신 |
| 기록 | `history.csv`는 source/age group별 행; 같은 epoch의 loss/time/step가 여러 행에 반복되므로 합산 금지 |

AMP scale 하락으로 건너뛴 gradient update는 `amp_skipped_steps`에 기록된다. 손실 유한성, 실제 optimizer step, scheduler 진행을 구분한다. 30 epoch 완료가 모든 batch의 optimizer update 성공이나 최종 epoch 채택을 뜻하지 않는다. 여러 번의 독립 seed 학습·오차 막대·통계적 유의성 검정은 없다.

### 5.2 증강과 sampler

- 최초 `public`: 출처를 합친 decade 빈도의 역수 `1/n(decade)`를 가중치로 replacement sampling. epoch당 train 행 수만큼 추출한다. 원래 데이터 분포 그대로 한 번씩 학습하는 방식이 아니다.
- `balanced`: `(source, decade)` 셀 빈도의 역제곱근 `n^-0.5`를 구하고 sample-weight의 평균 5배에서 cap한다. replacement=true, epoch당 train 행 수 유지. 정확한 source/연령 균등 sampler가 아니라 완만한 보정이다. val/test 분포는 변경하지 않는다.
- crop 증강: `RandomResizedCrop(224, scale=(0.8,1.0))`, random horizontal flip, ColorJitter(brightness/contrast/saturation=0.2, hue=0.05), normalize.
- weak 증강: random crop만 제거하고 `Resize((224,224))`, 동일 flip/jitter/normalize를 적용한다. 이름은 `weak_crop`이지만 실제로는 no-crop 실험이다. 여러 요인의 효과가 섞이지 않도록 balanced sampler를 유지한다.

### 5.3 학습 결과

| 실험 디렉터리 | 입력 train / val | sampler / 증강 | 최저 val epoch | 최저 MAE | 해당 CS@5 | epoch 30 MAE |
|---|---:|---|---:|---:|---:|---:|
| `public` | 162,823 / 24,313 | decade inverse / crop | 25 | 3.465184 | 0.767779 | 3.488746 |
| `public_balanced_v1` | 162,823 / 24,313 | source-decade soft balance / crop | 26 | 3.371056 | 0.778226 | 3.379022 |
| `public_weak_crop_v1` | 162,823 / 24,313 | 동일 balanced / weak | 17 | 3.418837 | 0.774154 | 3.427366 |
| `public_aligned_crop_retry_v1` | 160,590 / 23,838 | 동일 balanced / crop | 24 | 3.267767 | 0.791006 | 3.296452 |
| `public_aligned_weak_retry_v1` | 160,590 / 23,838 | 동일 balanced / weak | 18 | 3.320102 | 0.783581 | 3.325873 |

각 행의 `run.json`과 `history.csv`가 근거다. 비정렬 balanced는 최초보다 MAE가 0.094127 낮고, 비정렬 weak는 balanced보다 0.047781 높다. 이는 한 seed, 해당 관측 val 분포에서의 결과다. 아래 정렬 재학습 두 모델은 서로 같은 데이터로 비교할 수 있지만, 비정렬 전체 val 24,313장과 정렬 val 23,838장은 동일 모집단이 아니므로 위 표만으로 정렬의 순수 효과를 계산하지 않는다.

sampler 변경의 가설은 희소 연령대의 같은 사진을 역빈도로 지나치게 반복하면 과적합 위험이 커질 수 있다는 것이었다. 최초 학습은 25 epoch 이후 train loss가 더 줄어도 val MAE가 좋아지지 않았다. 그러나 서로 다른 목적의 train loss와 val MAE만으로 과적합 또는 데이터 부족을 확정할 수는 없다. 그래서 모델 구조와 나머지 조건은 유지하고 sampler를 완화하는 비교를 먼저 했다. weak 실험은 얼굴의 눈가·이마·턱 등 나이 단서가 랜덤 크롭으로 잘릴 가능성을 확인하려는 것이었지만, 결과적으로 전체 val에서는 balanced보다 나빴다.

각 best epoch의 source MAE는 public AFAD/MegaAge **3.282754 / 3.844411**, balanced **3.197201 / 3.732460**, weak **3.329508 / 3.604531**, aligned crop **3.194018 / 3.429930**, aligned weak **3.248366 / 3.477841**다. source 간 라벨 의미와 분포도 달라 source MAE의 우열만으로 실제 연령 정확성을 확정하지 않는다.

## 6. 최초 고정 test 평가

기존 산출물 `/workspace/hai/outputs/eval/public_initial_20261114/model_metrics.csv`와 `evaluated_5826a6b053af3b2d_e1c9406b4333b3bd.json`을 대조했다. 체크포인트는 최초 `public/best.pt`, `flip=false`, test CSV SHA-256은 `e1c9406b4333b3bd23204b6d13a03528e27178c29e5db52921f004ad89df7360`이다. test 이미지는 비정렬 입력을 224x224 resize했다. InsightFace baseline 산출물은 이 디렉터리에 없으므로 해당 baseline 성능을 주장하지 않는다.

| 출처 / 연령 | n | MAE | CS@5 |
|---|---:|---:|---:|
| 전체 | **20,223** | **3.332714** | **0.783514** |
| AFAD 전체 | 16,482 | 3.303222 | 0.787768 |
| AFAD 10~19 | 2,471 | 2.265871 | 0.884662 |
| AFAD 20~29 | 9,685 | 2.923548 | 0.830356 |
| AFAD 30~39 | 4,076 | 4.586276 | 0.647694 |
| AFAD 40~49 | 215 | 6.247269 | 0.497674 |
| AFAD 50~59 | 19 | 11.631280 | 0.263158 |
| AFAD 60~69 | 11 | 21.202478 | 0.090909 |
| AFAD 70~79 | 5 | 7.823368 | 0.600000 |
| MegaAge 전체 | 3,741 | 3.462652 | 0.764769 |
| MegaAge 0~9 | 1,024 | 1.991360 | 0.915039 |
| MegaAge 10~19 | 650 | 3.193437 | 0.800000 |
| MegaAge 20~29 | 1,048 | 3.944337 | 0.699427 |
| MegaAge 30~39 | 667 | 4.195339 | 0.685157 |
| MegaAge 40~49 | 177 | 5.174613 | 0.587571 |
| MegaAge 50~59 | 87 | 6.715302 | 0.482759 |
| MegaAge 60~69 | 88 | 4.622708 | 0.772727 |

`age.train` 초기 경로는 test manifest를 읽어 나이·경로 존재·split 누수를 확인하고 해시를 기록하지만, 학습 중 test inference는 하지 않는다. 후속 balanced/weak 학습과 정렬 도구는 train/val만 사용한다. 최초 test 결과가 이미 알려진 뒤 수행한 후속 연구이므로 완전한 test-blind 연구 설계였다고 과장하지 않는다. 후속 모델의 선택과 진단 수치는 val에 기반하며, test 재평가는 수행하지 않았다.

## 7. 첫 정렬 실패와 전처리 불일치

### 7.1 정렬 계약과 실패 집계

`age.face.FaceAligner`는 InsightFace v0.7 buffalo_l의 SCRFD(`det_10g.onnx`)와 속성 모델(`genderage.onnx`)을 사용한다. 검출 입력 640x640, threshold 0.5, 학습용 `training=True`에서 정확히 한 얼굴만 허용한다. landmark 5점으로 ArcFace similarity transform을 구해 원래 RGB 이미지에서 224x224 crop한다. 검출된 `face.age`를 학습 정답으로 덮어쓰지 않는다.

| split / 출처 | 입력 | 기본 정렬 채택 | 채택률 |
|---|---:|---:|---:|
| train AFAD | 131,756 | 414 | 0.31% |
| train MegaAge | 31,067 | 29,041 | 93.48% |
| train 전체 | 162,823 | **29,455** | 18.09% |
| val AFAD | 16,416 | 44 | 0.27% |
| val MegaAge | 7,897 | 7,431 | 94.10% |
| val 전체 | 24,313 | **7,475** | 30.74% |

`age_public_aligned_v1/summary.json` 기준 train 실패 133,368, val 실패 16,838로 총 **150,206건**이다. `failures.csv` 실제 행 수와 맞는다. train 미검출 131,491 / 다중 얼굴 1,877, val 미검출 16,408 / 다중 얼굴 430이다. 합계 사유는 미검출 **147,899**, 다중 얼굴 **2,307**이다. 정렬 픽셀 train/val 교차 중복은 0그룹이고 `cross_split_duplicates.csv`는 헤더뿐이다. 전체 경과 2,584.971초, 입력 처리율 72.394장/초는 성공 이미지 전용 처리율이 아니다.

기존 조사에서는 AFAD의 타이트한 얼굴 crop과 검출기의 크기·맥락 조건 불일치가 유력했다. 공식 SCRFD 런타임은 긴 변을 640으로 resize하고 남는 쪽만 padding하므로 정사각형 타이트 crop에는 사방 맥락이 추가되지 않는다. 연령과 출처 구성이 얽혀 있으므로 단순 연령별 성공률만 보고 검출기의 연령 편향으로 단정하지 않는다. 이번 문서 작성은 그 조사나 이미지 관찰을 새로 수행하지 않았다.

전체 실행 전 GPU pilot `/tmp/kilo/age-align-gpu-pilot-100`에서는 원래 manifest 앞 100장씩을 처리해 train 28/100, val 39/100이 성공했고 133건은 미검출이었다. 처음 행만 고른 표본이므로 이 성공률을 전체에 외삽하지 않았다.

첫 정렬 결과는 AFAD가 거의 소실된 데이터이므로 이 데이터만으로 전체 재학습을 진행한 결과는 없다. 다음 비교는 **기존 비정렬 학습 모델에 입력만 정렬본으로 바꾼 진단**이지 정렬 학습 모델의 성능이 아니다.

### 7.2 동일 표본 원본/정렬 진단

`age.compare_aligned_val`은 `(source, original_id)` 대응, 원래 manifest 인덱스, 나이·성별·인물 ID 동일성, 정렬 val 경로 규칙을 검사하고 같은 표본의 원본/정렬 이미지를 각각 평가한다.

| 산출물 / 모델 | 표본 | flip | 원본 MAE / CS@5 | 정렬 MAE / CS@5 |
|---|---:|---|---|---|
| `compare_aligned_val_balanced_flip_v1`, balanced | 7,475 | true | 3.641269 / 0.737525 | 5.762984 / 0.563077 |
| `compare_aligned_val_weak_crop_v1`, 당시 weak checkpoint | 7,475 | false | 3.626250 / 0.741405 | 7.097245 / 0.473311 |
| `compare_aligned_retry_balanced_flip_v1`, balanced | 23,838 | true | 3.316739 / 0.785343 | 4.293198 / 0.694563 |

첫 7,475장에는 AFAD 44, MegaAge 7,431장만 있다. 첫 balanced 진단의 source MAE는 원본/정렬 AFAD 3.950966/5.422927, MegaAge 3.639435/5.764997이다. weak 진단은 AFAD 4.362463/7.931885, MegaAge 3.621891/7.092303이다. 서로 다른 flip 및 checkpoint 조건인 balanced와 weak의 위 행을 직접 비교하여 증강 우열을 주장하지 않는다.

**weak 비교 기록의 체크포인트 불일치:** `compare_aligned_val_weak_crop_v1/summary.json`은 checkpoint SHA-256 `3f88245ccadafd108bb9fc211014007868afcddcabced1e7f2ceb59182ed256b`를 기록한다. 문서 작성 시 현재 `public_weak_crop_v1/best.pt` 실파일은 `5b654618491bd33dde0218dd696ca486fbd5b378ed1b3ab8c5496210341d2fec`다. 비교 당시 checkpoint와 현재 최종 best는 동일하지 않다. 정확한 변경 원인·당시 epoch를 이 요약만으로 확정할 수 없으며, 현재 경로를 그대로 실행해 당시 수치가 재현된다고 주장하지 않는다. 결과 자체는 당시 저장된 진단 결과로 보존한다. 다른 네 학습 checkpoint의 저장된 평가 해시는 현재 실파일과 일치했다.

## 8. Padding Retry와 품질 감사

### 8.1 재시도 규칙과 pilot

별도 `age/align_public_retry.py`는 기존 `age.align_public`이나 원본을 덮어쓰지 않는 도구다. 기본 `align(training=True)`가 **정확히 `ValueError('face_not_detected')`**인 경우에만 재시도한다. 다른 예외나 기본 다중 얼굴은 재시도하지 않는다.

원본 높이/너비의 각각 절반을 내림한 `dy=h//2`, `dx=w//2`만큼 위·아래/좌·우에 RGB 검정 여백을 넣는다. 재검출에서도 단일 얼굴을 요구하며 bbox와 landmark에서 padding offset을 뺀 다음 **원본 RGB에서 crop**한다. padding 이미지를 그대로 학습 이미지로 저장하는 방식이 아니다. 원본 밖 warp 영역은 여전히 검게 채워질 수 있다.

저장된 CPU pilot `/tmp/kilo/age-align-retry-pilot-final-20261114/summary.json`은 split별·출처별 manifest 앞 8장씩 총 32장, elapsed 5.918초다. AFAD train/val 각 8장은 모두 retry 성공, MegaAge train/val 각 8장은 모두 base 성공, 오류와 교차 픽셀 중복은 0이었다. AFAD retry의 검은 픽셀 평균은 train 21.9525%, val 24.3396%, 최대는 val 39.9613%다. 임의 대표 표본이나 모집단 복구율 추정치가 아니다. 기존 작업 기록의 합성 단위 테스트는 7개 통과였으며 이번 문서 작성에서 재실행하지 않았다.

추가 GPU pilot `/tmp/kilo/age-align-retry-gpu-pilot-100`은 split별·출처별 앞 100장씩 총 400장을 처리했다. AFAD train/val은 각각 기본 1장과 retry 99장으로 모두 성공했고, MegaAge는 train 기본 95장+retry 1장, val 기본 96장이 성공했다. MegaAge의 나머지 8장은 다중 얼굴로 제외했다. AFAD retry 검은 픽셀 평균은 두 split 모두 약 26%, 최대 약 56~57%였다. CPU pilot과 같은 선택 편향이 있으며 검출 성공을 품질 보증으로 삼지 않았다.

### 8.2 전체 실행 결과

전체 retry는 2026-11-14 **10:50:28.819253 UTC**에 시작했고 6,932.943초가 걸렸다. `limit_per_source_per_split=0`, source filter 없음, CUDA였다. 같은 원본 train/val 총 187,136건을 처리해 audit 실제 187,136행, accepted 184,428행, failures 2,708행을 확인했다.

| split / 출처 | 입력 | base 성공 | retry 성공 | 실패 | 최종 채택 |
|---|---:|---:|---:|---:|---:|
| train AFAD | 131,756 | 414 | 131,036 | 306 | 131,450 |
| train MegaAge | 31,067 | 29,041 | 99 | 1,927 | 29,140 |
| train 전체 | **162,823** | **29,455** | **131,135** | **2,233** | **160,590** |
| val AFAD | 16,416 | 44 | 16,342 | 30 | 16,386 |
| val MegaAge | 7,897 | 7,431 | 21 | 445 | 7,452 |
| val 전체 | **24,313** | **7,475** | **16,363** | **475** | **23,838** |

AFAD는 train 실패가 retry 미검출 260 / retry 다중 얼굴 27 / base 다중 얼굴 19, val이 각각 27 / 1 / 2다. MegaAge는 train base 다중 얼굴 1,858 / retry 미검출 69, val 각각 428 / 17이다. 실패 2,708건은 모두 이 사유에 포함된다. 원래 기본 성공 수는 유지되고, 무조건 threshold를 낮추거나 가장 큰 얼굴을 강제 채택하지 않았다.

정렬 픽셀 train/val 교차 중복은 이번에도 0그룹이다. 픽셀 SHA는 224x224 RGB uint8의 row-major bytes를 기준으로 하며 PNG 파일 bytes의 해시와 다르다. 정책상 교차 중복이 발견되면 양쪽 manifest에서 제외하지만 PNG는 디스크에 남으므로 `images/` 전체를 무차별 재귀 학습하면 안 된다. 현재 0그룹이라는 사실은 test·동일인·근접 중복 검증이 아니다.

| 출처 / 경로 | split | 검은 픽셀 평균 | 중앙값 | 최대 |
|---|---|---:|---:|---:|
| AFAD base | train | 11.2058% | 0% | 73.9676% |
| AFAD retry | train | 25.7299% | 24.6612% | 71.0639% |
| AFAD base | val | 8.7072% | 0% | 52.2082% |
| AFAD retry | val | 25.6368% | 24.5137% | 64.2518% |
| MegaAge base | train | 0.0808% | 0% | 43.7719% |
| MegaAge retry | train | 19.9247% | 10.6186% | 72.4669% |
| MegaAge base | val | 0.0934% | 0% | 47.2716% |
| MegaAge retry | val | 10.6037% | 0% | 55.9072% |

검은 픽셀은 출력에서 `R=G=B=0`인 비율로, 원래 이미지의 검정 영역도 포함한다. 얼굴 잘림·랜드마크 오류·유효 얼굴 정보 손실을 직접 측정한 품질 점수나 새 제외 threshold가 아니다. 채택 이미지에 검은 영역 상한 필터를 추가한 실험도 아니다. CPU/CUDA의 작은 부동소수 차이로 근접한 crop의 픽셀 해시가 달라질 수 있어 해시만으로 근접 중복을 검출할 수 없다.

`audit.csv`는 split/index, 원래 path/source/original_id/age, base/retry/error status, 실패 stage/reason/type, 검은 픽셀 비율, 정렬 SHA, 최종 disposition을 보존한다. 민감한 행별 원자료를 문서에 복사하지 않고 집계만 기록한다. 감사 CSV와 summary는 정상 완료 시 생성되며 중단 디렉터리에 이어쓰기하는 방식이 아니다.

## 9. 정렬 재학습과 고정 Flip 평가

기존 비정렬 balanced 모델에 retry 정렬 입력을 넣으면 동일 23,838장에서 MAE가 원본 **3.316739**에서 정렬 **4.293198**로 악화됐다. source별로 원본/정렬 AFAD 3.162457/3.619216, MegaAge 3.655986/5.775198이다. 따라서 학습·추론 입력 분포를 맞추기 위해 retry 정렬 train/val로 crop 및 weak 모델을 **각각 ImageNet부터** 학습했다.

두 모델은 학습 중 `flip=false` val MAE로 best epoch를 정하고, **그 checkpoint를 고정한 뒤 둘 다 동일한 `flip=true`**로 평가했다. 뒤집기 TTA는 원본 예측 나이와 수평 반전 예측 나이를 산술 평균한다. 뒤집은 label을 만들거나 나이 분포의 argmax를 평균하는 방식이 아니다. flip로 모든 epoch를 재탐색한 결과는 아니다.

| 동일 retry val 23,838장 / flip=true | 전체 MAE | 전체 CS@5 | AFAD MAE / CS@5 | MegaAge MAE / CS@5 |
|---|---:|---:|---|---|
| 비정렬 balanced, 대응 원본 입력 | 3.316739 | 0.785343 | 3.162457 / 0.807763 | 3.655986 / 0.736044 |
| 비정렬 balanced, 대응 정렬 입력 | 4.293198 | 0.694563 | 3.619216 / 0.754730 | 5.775198 / 0.562265 |
| 정렬 crop, 정렬 입력, epoch 24 | **3.237303** | **0.794236** | **3.158800 / 0.806298** | **3.409922 / 0.767713** |
| 정렬 weak, 정렬 입력, epoch 18 | 3.276775 | 0.788741 | 3.203195 / 0.803552 | 3.438568 / 0.756173 |

crop은 weak보다 MAE **0.039472세** 낮고 CS@5가 **0.005495** 높다. 대응 원본 입력 balanced 대비 crop MAE는 **0.079436세** 낮다. 후자는 같은 val ID와 flip를 맞췄지만 학습 데이터 수·전처리·학습 epoch도 바뀐 end-to-end 파이프라인 비교이므로 정렬 하나의 인과 효과로 해석하지 않는다. crop의 AFAD CS@5는 원본 balanced보다 소폭 낮아 모든 source/지표가 개선된 것도 아니다.

### 9.1 동일 조건 연령별 결과

| 출처 / 연령 | n | crop MAE | weak MAE | crop CS@5 | weak CS@5 |
|---|---:|---:|---:|---:|---:|
| AFAD 10~19 | 2,400 | 2.525940 | 2.496555 | 0.878333 | 0.870417 |
| AFAD 20~29 | 9,808 | 2.611993 | 2.595861 | 0.863581 | 0.869698 |
| AFAD 30~39 | 3,949 | 4.648711 | 4.884164 | 0.638389 | 0.617625 |
| AFAD 40~49 | 189 | 6.162335 | 6.365565 | 0.544974 | 0.518519 |
| AFAD 50~59 | 22 | 15.108088 | 13.270071 | 0.181818 | 0.227273 |
| AFAD 60~69 | 14 | 13.072607 | 13.760364 | 0.214286 | 0.357143 |
| AFAD 70~79 | 4 | 10.394086 | 15.095194 | 0.750000 | 0.250000 |
| MegaAge 0~9 | 873 | 1.872851 | 1.866977 | 0.926690 | 0.914089 |
| MegaAge 10~19 | 883 | 3.193183 | 3.293724 | 0.793884 | 0.778029 |
| MegaAge 20~29 | 1,956 | 2.980975 | 2.980262 | 0.831288 | 0.829755 |
| MegaAge 30~39 | 1,457 | 3.954397 | 4.024200 | 0.695951 | 0.682224 |
| MegaAge 40~49 | 1,140 | 4.207915 | 4.381522 | 0.667544 | 0.640351 |
| MegaAge 50~59 | 771 | 3.885090 | 3.849058 | 0.715953 | 0.706874 |
| MegaAge 60~69 | 372 | 4.224152 | 3.846139 | 0.693548 | 0.693548 |

근거는 각 `eval_aligned_flip_*_v1/{summary.json,model_metrics.csv}`이며 두 파일의 수치가 일치한다. crop이 모든 연령대의 우승 모델은 아니다. 특히 AFAD 50대 22장, 60대 14장, 70대 4장의 변동은 크고 실제 고령 성능의 일반화 근거가 부족하다.

### 9.2 모델 선택과 배포 미완료

선택 기준은 첫째 동일 retry val와 flip 조건에서 전체 MAE가 낮은지, 둘째 source/연령별 실패가 숨겨지지 않는지, 셋째 독립 test와 실제 추론 계약의 검증 여부다. 앞의 두 모델 중 **잠정 연구 기준 모델**은 `/workspace/hai/outputs/age/public_aligned_crop_retry_v1/best.pt`다. 이는 val에서 선택된 후보라는 의미이며 상업 서비스·인구집단 일반화·의학적/법적 나이 판정에 적합한 모델이라는 뜻이 아니다.

실제 `age.infer.AgeEstimator`는 `training=False`로 가장 큰 얼굴을 선택하고 flip 평균을 사용한다. 연구용 정렬은 `training=True`로 다중 얼굴을 제외하고 padding retry까지 적용한다. 현재 기본 추론 경로에는 이 retry가 자동 추가된 것이 아니다. 따라서 정렬 val에서의 고정 flip 평가는 실제 제품 end-to-end 성공률, 검출 실패율, 다중 얼굴 정책, padding 정책까지 검증한 배포 평가가 아니다. 이번 결과를 위해 ONNX 최종 export·서비스 연결·새 test 평가를 완료했다는 주장은 하지 않는다.

## 10. 런타임과 재현 식별자

### 10.1 환경과 실패 과정

학습 `run.json`에 기록된 버전은 torch **2.8.0+cu128**, torchvision **0.23.0+cu128**, numpy **2.4.6**, Pillow **12.3.0**다. 학습 기록 자체에는 모든 패키지 pin, GPU 모델, 드라이버 및 Python executable가 없으므로 완전한 hermetic 재현 환경으로 간주하지 않는다.

얼굴 정렬 기본 환경 `/venv/main/bin/python`에는 CPU ONNX Runtime 1.31.0이 있었다. CUDA용으로 `/workspace/hai/.venv-age-ort-gpu`를 준비했다. `pyvenv.cfg`의 Python은 **3.11.14**, `include-system-site-packages=true`, base executable은 `/venv/main/bin/python3.11`이다. **이 venv는 완전 독립 환경이 아니며 base site-packages 변경의 영향을 받는다.** 이름이나 `pip freeze`만으로 이식 가능성을 보장할 수 없다.

첫 CUDA 검증에서는 provider 목록에 CUDA가 나타나도 라이브러리 탐색 실패로 실제 세션이 CPU로 fallback한 과정이 있었다. 기존 실행 기록에서는 FaceAligner 생성 전에 torch를 import해 PyTorch CUDA 라이브러리를 먼저 로드한 후 양쪽 세션을 확인했다. `age.align_public*`의 CUDA 경로도 torch를 먼저 import하고 실제 detector/attribute session provider를 확인해 조용한 CPU fallback을 거부한다. 일부 보조 노드가 CPU에 배치될 수 있어 CUDA provider 사용이 순수 GPU-only 실행을 뜻하지 않는다.

전체 retry `summary.json`에는 패키지 metadata로 `onnxruntime=1.31.0`, `onnxruntime-gpu=1.23.2`가 함께 보이지만 **실제 import된 `onnxruntime_loaded_version`은 1.23.2**다. detector와 attribute 모두 `CUDAExecutionProvider`, `CPUExecutionProvider` 순서였다. OpenCV headless 4.14.0.94, scikit-image 0.26.0도 기록되어 있다. 같은 `onnxruntime` namespace를 제공하는 CPU/GPU 배포판 metadata와 실제 로드 모듈을 혼동하지 않는다. 기록 보존 없이 기존 base 환경에 재설치하여 이를 맞추려 해서는 안 된다.

### 10.2 Manifest와 checkpoint SHA-256

다음 train/val 해시와 현재 checkpoint 해시는 문서 작성 중 이미지 열람 없이 실파일 bytes로 대조했다. test 해시는 이미 저장된 초기 run/evaluation marker의 값이며 test manifest나 이미지를 새 평가한 결과가 아니다.

| 파일 | SHA-256 |
|---|---|
| `age_public/train.csv` | `200bbe99d4c74cb41b2bfbbd555da8ba0e981f3a5966d7cb6b3ab50bab0a1461` |
| `age_public/val.csv` | `4f2214a79d9b4ebc187df1f2e2ad64ddf0a644da60968b622964e04425ba309c` |
| `age_public/test.csv` (기존 기록) | `e1c9406b4333b3bd23204b6d13a03528e27178c29e5db52921f004ad89df7360` |
| `age_public_aligned_v1/train.csv` | `56cc65a3a41e1b63dafd7ab51cab4d7187c86fdd9f13816914cedf0ba4db7660` |
| `age_public_aligned_v1/val.csv` | `0d507630628e1e45ea95a2fb61c6755affaf7d375cf7a1f81a4223aca3cb3f82` |
| `age_public_aligned_retry_v1/train.csv` | `b21c6d114eda563b2a0f235676597880583b353f8cc5f5b8148b88ec3615de6e` |
| `age_public_aligned_retry_v1/val.csv` | `d888ea21c354a47eb0e2ee04f21956452a534573d79a143c4cb19993434c0bd8` |
| `public/best.pt` | `5826a6b053af3b2d16cc9949da22b276a1158de9796c6846718fa4a20fe98f43` |
| `public_balanced_v1/best.pt` | `3130cf88a8129cf905204d9337ee66b2741ca21169b3dc105f1f659f8cae53ad` |
| `public_weak_crop_v1/best.pt` (현재) | `5b654618491bd33dde0218dd696ca486fbd5b378ed1b3ab8c5496210341d2fec` |
| `public_aligned_crop_retry_v1/best.pt` | `962220d547834a1c9b557e80b9a048418783b05c1fe5e411a673ec90a615d200` |
| `public_aligned_weak_retry_v1/best.pt` | `67cf1fffdbedd1703fac9522a066e73e6c529b2ae861e59196c87967c453709f` |

InsightFace provenance의 archive SHA-256은 `80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f`다. 사용 가중치는 detector `5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91`, attribute `4fde69b1c810857b88c64a335084f1c3fe8f01246c9a191b48c7bb756d6652fb`로 첫 정렬과 retry 보고서가 같다.

retry 실행 시 코드 해시는 `align_public_retry.py=95f316e9d096050e8887bc7ed83799ff5676bf4834276eda0879f3da0a87a7be`, `face.py=3c95e9d8bde3785629c7efc3f2af52239fb84611e6cff6ef0811f1b650372ef8`, `common.py=e4bd319657fc67fa88da9377dd4667ee5e27d663d62b0adabd64258e218c4dc6`다. 공식 runtime 해시는 `scrfd.py=64b0b787e4c235ff80c1141614a77f2b2c24c90ba5b920f1bc91fde80cbef9f9`, `attribute.py=90311f46c4315db31eb60cd52adc4e8819c35ccf5b814b04b3d0858f108861f7`, `face_align.py=a8b34ddd06716f5ca7e7501415cbd3789ccbce18db9e557b35e537695290d922`다. 경로가 같아도 소스나 의존성이 바뀌면 재현 조건이 같지 않다.

## 11. 재현 명령 기록

**아래는 과거 실행 설정을 보존·재구성한 참고 명령이며 이번 문서 작성에서 실행하지 않았다.** 모든 Python 모듈 명령의 작업 디렉터리는 `/workspace/hai/branches/main`이다. `*_repro`는 별도 새 출력 경로 예시이며 기존 v1 산출물을 덮어쓰지 않는다. 현재 test 접근/새 평가 금지 상태에서는 실행하지 않는다. 권한·목적·출력 경로가 별도로 승인된 향후 재현 작업에서만 사용한다. seed만 같아도 GPU 연산과 패키지·데이터·경로 변경으로 bitwise 재현되지 않을 수 있다.

### 11.1 다운로드 확인과 데이터 준비

원본 취득의 URL, 시각, 응답, SHA-256은 각 `acquisition.json`에 보존되어 있다. AAF/MegaAge는 위 공식 ZIP URL, AFAD는 고정 revision의 46개 조각, FG-NET은 위 미러 ZIP을 새 다운로드 디렉터리에 받아야 한다. 현재 자료만으로 당시 네 취득 스크립트의 shell history 전체가 확정되는 것은 아니므로 임의의 공통 downloader가 있었다고 만들지 않는다. 재취득 후 각 원본 취득·카탈로그 절차의 보안 검사와 이용 조건을 다시 확인한다.

이미지 디코딩이나 평가 없이 원본 아카이브 bytes의 식별을 확인하는 예시는 다음과 같다.

```bash
sha256sum "/workspace/hai/shared/data/aaf/All-Age-Faces Dataset.zip" \
  /workspace/hai/shared/data/megaage_asian/megaage_asian.zip \
  /workspace/hai/shared/data/afad/archives/AFAD-Full.tar.xz \
  /workspace/hai/shared/data/fgnet/FGNET.zip

/venv/main/bin/python -m age.prepare_public \
  --data-root /workspace/hai/shared/data \
  --output /workspace/hai/shared/data/age_public_repro
```

마지막 준비 명령은 원본 카탈로그에서 test manifest도 생성하므로 이번에 실행한 확인 명령이 아니다. `prepare_public`의 출력 manifest에는 출력 디렉터리의 절대 경로가 들어가므로 `_repro`로 출력하면 기존 CSV의 내용 해시도 바뀐다. 경로만 바뀐 경우와 표본·라벨이 바뀐 경우를 구분해 비교해야 한다.

### 11.2 학습

```bash
/venv/main/bin/python -m age.train \
  --stage public --data-dir /workspace/hai/shared/data/age_public \
  --output /workspace/hai/outputs/age/public_repro \
  --batch-size 128 --epochs 30 --lr 0.0005 --workers 4 --seed 42 --device cuda

/venv/main/bin/python -m age.train_public_balanced \
  --data-dir /workspace/hai/shared/data/age_public \
  --output /workspace/hai/outputs/age/public_balanced_repro \
  --batch-size 128 --epochs 30 --lr 0.0005 --workers 4 --seed 42 --device cuda

/venv/main/bin/python -m age.train_public_weak_crop \
  --data-dir /workspace/hai/shared/data/age_public \
  --output /workspace/hai/outputs/age/public_weak_crop_repro \
  --batch-size 128 --epochs 30 --lr 0.0005 --workers 4 --seed 42 --device cuda
```

첫 명령은 코드상 test manifest도 읽는다. 후속 두 명령은 train/val만 읽는다. 기존 `run.json`은 public/ balanced의 절대 경로와 weak의 `data/age_public` 상대 경로 표기를 각각 보존하고 있으며 위 명령은 유효 경로를 명시한 설정 재구성이다.

### 11.3 얼굴 모델과 정렬

공식 모델을 새 위치에 준비할 때의 진입점은 다음과 같다. 설치나 다운로드를 이번 문서 작업에서 반복하지 않았다. 전체 SDK/C++ 확장을 필수로 설치하는 경로가 아니라 공식 v0.7 Python runtime subset과 detector/attribute ONNX를 준비하는 경로다.

```bash
/venv/main/bin/python -m age.setup_models \
  --model-root /workspace/hai/shared/models/insightface_repro

/workspace/hai/.venv-age-ort-gpu/bin/python -m age.align_public \
  --data-dir /workspace/hai/shared/data/age_public \
  --output /workspace/hai/shared/data/age_public_aligned_repro \
  --model-root /workspace/hai/shared/models/insightface --device cuda

OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
/venv/main/bin/python -m age.align_public_retry \
  --data-dir /workspace/hai/shared/data/age_public \
  --output /tmp/kilo/age-align-retry-pilot-repro \
  --model-root /workspace/hai/shared/models/insightface --device cpu \
  --source afad --source megaage_asian --limit-per-source 8 --progress-every 8

/workspace/hai/.venv-age-ort-gpu/bin/python -m age.align_public_retry \
  --data-dir /workspace/hai/shared/data/age_public \
  --output /workspace/hai/shared/data/age_public_aligned_retry_repro \
  --device cuda --progress-every 1000
```

마지막 전체 retry의 원래 명령은 `age_public_aligned_retry_v1/summary.json`의 `command` 배열에 Python executable까지 저장되어 있고 output만 위 예시와 다르다. 해당 원래 명령은 `--model-root`를 생략해 저장소 기본 `models/insightface`를 사용했으며 summary의 resolved root는 공유 경로다. CPU pilot은 별도 `summary.json`의 command와 환경 변수를 근거로 했다.

### 11.4 정렬 재학습 및 대응 비교

```bash
/venv/main/bin/python -m age.train_public_balanced \
  --data-dir /workspace/hai/shared/data/age_public_aligned_retry_v1 \
  --output /workspace/hai/outputs/age/public_aligned_crop_retry_repro \
  --batch-size 128 --epochs 30 --lr 0.0005 --workers 4 --seed 42 --device cuda

/venv/main/bin/python -m age.train_public_weak_crop \
  --data-dir /workspace/hai/shared/data/age_public_aligned_retry_v1 \
  --output /workspace/hai/outputs/age/public_aligned_weak_retry_repro \
  --batch-size 128 --epochs 30 --lr 0.0005 --workers 4 --seed 42 --device cuda

/venv/main/bin/python -m age.compare_aligned_val \
  --original-csv /workspace/hai/shared/data/age_public/val.csv \
  --aligned-csv /workspace/hai/shared/data/age_public_aligned_v1/val.csv \
  --checkpoint /workspace/hai/outputs/age/public_balanced_v1/best.pt \
  --output /workspace/hai/outputs/age/compare_aligned_val_balanced_flip_repro \
  --device cuda --batch-size 128 --workers 4 --flip

/venv/main/bin/python -m age.compare_aligned_val \
  --original-csv /workspace/hai/shared/data/age_public/val.csv \
  --aligned-csv /workspace/hai/shared/data/age_public_aligned_retry_v1/val.csv \
  --checkpoint /workspace/hai/outputs/age/public_balanced_v1/best.pt \
  --output /workspace/hai/outputs/age/compare_aligned_retry_balanced_flip_repro \
  --device cuda --batch-size 128 --workers 4 --flip
```

weak의 첫 정렬 비교는 같은 첫 비교 명령에서 `--flip` 없이 당시 weak checkpoint를 사용한 설정이다. 위 7.2절의 당시 SHA가 일치하는 checkpoint가 별도로 보존되어 있지 않으면 과거 수치의 정확한 재현은 불가능하다.

### 11.5 고정 val Flip 평가의 재구성

`age.evaluate` CLI는 checkpoint에 저장된 train/val과 평가 입력의 누수를 막으므로, 학습에 사용한 val을 `--csv`로 주면 거부한다. 이 보호를 끄거나 평가 marker를 지우는 방식으로 val 평가를 반복하면 안 된다. 아래는 저장된 고정 checkpoint의 **연구용 val 평가 절차를 명시적으로 재구성**한 코드이며 과거 shell 명령 자체를 복원한 것이라는 주장은 하지 않는다. 새 평가이므로 이번에는 실행하지 않았다.

```bash
/venv/main/bin/python - <<'PY'
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from age.common import save_json, sha256
from age.dataset import AgeDataset
from age.evaluate import predict, write_metrics
from age.model import build_model

torch.set_num_threads(8)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
csv_path = Path('/workspace/hai/shared/data/age_public_aligned_retry_v1/val.csv')
for kind in ('crop', 'weak'):
    checkpoint_path = Path(f'/workspace/hai/outputs/age/public_aligned_{kind}_retry_v1/best.pt')
    output = Path(f'/workspace/hai/outputs/age/eval_aligned_flip_{kind}_repro')
    if output.exists():
        raise FileExistsError(output)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    run = checkpoint['run']
    assert not run.get('smoke_steps') and not run.get('no_pretrained')
    assert run['data_sha256']['val'] == sha256(csv_path)
    dataset = AgeDataset(csv_path)
    model = build_model(False).to('cuda')
    model.load_state_dict(checkpoint['model'])
    rows = predict(model, DataLoader(dataset, batch_size=128, num_workers=4), 'cuda', flip=True)
    output.mkdir(parents=True, exist_ok=False)
    metrics = write_metrics(output, rows)
    save_json(output / 'summary.json', dict(
        checkpoint=str(checkpoint_path), checkpoint_sha256=sha256(checkpoint_path),
        val_csv_sha256=sha256(csv_path), flip=True, epoch=checkpoint['epoch'], metrics=metrics))
PY
```

이 코드는 직접 관리하는 신뢰된 checkpoint 전용이며 임의의 제3자 `.pt`를 읽는 예제가 아니다. `weights_only=False`의 pickle load는 코드 실행을 수반할 수 있으므로 출처 불명 파일에는 실행하지 않는다.

최초 test 실행의 구성은 `age.evaluate --checkpoint /workspace/hai/outputs/age/public/best.pt --csv /workspace/hai/shared/data/age_public/test.csv --output /workspace/hai/outputs/eval/public_initial_20261114 --device cuda --batch-size 128`이며 `--flip`, `--baseline` 없는 조건이다. **이미 평가된 checkpoint/test 쌍이고 현재 재실행 금지**다. marker는 해당 output에서의 중복 실행 방어이지 다른 output으로 우회할 수 있는 평가 허가나 전역적인 test 봉인 장치는 아니다.

## 12. 라이선스, 개인정보와 향후 업로드

### 12.1 데이터와 가중치 조건

- **AAF:** 공식 README의 연구 인용 요청은 명시적 재사용 라이선스나 상업 허가가 아니다. `terms_snapshot.json` 및 `source_README.txt`에 근거를 보존했으며 허가 확인 전 학습 대상에 넣지 않았다. 얼굴 당사자의 개별 동의·생체정보 활용 허가가 확보되었다고 주장하지 않는다.
- **MegaAge-Asian:** 비상업 연구 전용 custom agreement다. 이미지뿐 아니라 derived data의 상업 이용을 제한하고, 데이터의 추가 복제·공개·배포를 금지하며 동일 조직 단일 site 내부 사용 예외가 있다. 원 저장소 GPL-3.0은 데이터 이용 계약의 대체 허가가 아니다. 학습 checkpoint·행별 예측 등 파생 산출물 공개 가능성도 별도 검토해야 한다.
- **AFAD:** 학술 연구 목적만 확인되었다. 상업 이용이나 공개 재배포가 허용되었다고 확정하지 않는다. 실제 배포본의 라벨 차이를 임의 정정하지 않았다.
- **FG-NET:** 미러의 연구용 안내를 보존했지만 미러가 원 권리자라는 근거, 표준 라이선스, 상업·재배포 허가는 없다. 인물 단위 분리 없는 실제 나이 쌍 평가에 사용해서도 안 된다.
- **InsightFace:** 코드의 MIT와 사전학습 가중치의 비상업 연구 제한은 서로 다르다. 모델 출처는 공식 v0.7 release의 `buffalo_l.zip`이며 Python runtime 원본과 `LICENSE`, `provenance.json`을 보존한다. 코드가 MIT라고 ONNX·ZIP까지 자유롭게 공개 재배포할 수 있는 것은 아니다.
- **MobileNet / ImageNet:** torchvision 코드 라이선스만으로 모든 사전학습 가중치·원 학습 데이터·본 데이터 파생 checkpoint의 상업적 사용/재배포 권리가 해결되었다고 간주하지 않는다. 사용한 `IMAGENET1K_V2`의 출처·조건과 본 공개 데이터 이용 조건을 함께 검토해야 한다.
- 이 실험에는 이전 FFHQ-Aging 구간 학습, 생성 LoRA, AI Hub 준비본을 합치지 않았다. 기존 AI Hub 기본 학습 정책이나 FFHQ split을 바꾼 실험이 아니다. 이전 문서의 라이선스나 모델을 이번 나이 추정 모델의 권리 근거로 전용하지 않는다.

필수 인용 근거는 AFAD의 Niu et al., *Ordinal Regression with a Multiple Output CNN for Age Estimation*, CVPR 2016; MegaAge의 Yunxuan Zhang, Li Liu, Cheng Li, Chen Change Loy, *Quantifying Facial Age by Posterior of Age Comparisons*, BMVC 2017; AAF의 Cheng et al., *Exploiting effective facial patches for robust gender recognition*, Tsinghua Science and Technology 24(3):333-345 (2019)다. 출처별 원문 조건과 인용 안내의 보존 위치를 우선하며, 인용만으로 이용 허가가 발생하지 않는다.

### 12.2 모델·출력에도 데이터가 포함될 수 있음

**`best.pt`를 가중치 tensor만 있는 파일로 취급하면 안 된다.** 현재 저장 코드는 `model` 외에 `run`, `split_rows={'train': train.rows, 'val': val.rows}`, optimizer/scheduler state, epoch, val MAE, optimizer step를 저장한다. `split_rows`에는 전체 train/val의 절대 경로, 나이, 성별, source, person_id, original_id, SHA-256이 포함될 수 있고 `run`에도 로컬 경로가 포함된다. 이미지 자체가 없어도 얼굴·나이·성별과 결합 가능한 훈련 데이터 메타데이터다. 이 위험은 코드 구조로 확인했으며 이번 문서 작업에서 checkpoint pickle을 새로 로드한 것은 아니다.

`model_predictions.csv`, 대응 비교 `predictions.csv`, audit/failures, source catalogue와 각 JSON에도 행별 경로·원 라벨·예측값·샘플 식별자가 들어갈 수 있다. test 예측 CSV 역시 이미지가 아니라고 해서 무조건 공개 가능한 파일이 아니다. URL redirect·로그·절대 경로는 내부 환경 정보를 노출할 수 있다. 본 문서에는 재현에 필요한 연구 경로와 집계·해시만 기재하고 실제 개별 sample 행은 싣지 않았다.

향후 Git 업로드는 다음 기준으로 구분해야 한다.

1. 코드와 이 집계 문서는 연구 재현 기록이다. 공개 전에도 내부 경로·불필요한 식별 정보·라이선스 설명을 검토한다.
2. 데이터 이미지·압축파일·manifest·행별 라벨/예측·audit와 원형 checkpoint는 데이터 제외 정책과 별도로 충돌할 수 있으므로 자동 포함하지 않는다. `.gitignore`의 `data/*`, `models/`, `outputs/`, `*.pt`, `*.onnx` 등의 기존 제외 규칙을 이 문서를 위해 변경하지 않는다.
3. tensor-only 또는 메타데이터 제거본을 향후 만들더라도 원형 checkpoint를 덮어쓰지 않고 별도 파일·SHA와 provenance를 관리해야 한다. 메타데이터 제거는 개인정보 노출 완화일 뿐 데이터/가중치 이용 조건을 해제하거나 얼굴 학습 정보의 암기를 배제하는 조치가 아니다. 이번 작업에서 변환·정제본은 생성하지 않았다.
4. Git LFS나 외부 저장소는 대용량 전송 수단일 뿐 데이터 공개·재배포 허가가 아니다. 심볼릭 링크의 공유 대상 전체를 무차별 스냅샷하거나 강제 추가하지 않는다.
5. 실제 배포 전에는 권리 검토, 인물·근접 중복 분리, 후속 모델의 독립 평가 계획, 실패 정책 및 추론 전처리 일치, 취약 집단과 고령·아동 표본 한계를 검토해야 한다. 이번 val 연구 결과만으로 배포 준비가 완료되었다고 선언하지 않는다.

## 13. 이번 문서 작성의 검증 범위

train/val manifest 실제 행 수는 비정렬 162,823/24,313, 첫 정렬 29,455/7,475, retry 160,590/23,838을 `rg` 행 집계로 대조했다. retry audit 187,136, failures 2,708, accepted 184,428 및 첫 정렬 failures 150,206도 대조했다. 교차 split 중복 CSV 두 개는 헤더만 있는 것을 확인했다. `history.csv`의 전체 val 30 epoch 행과 best/마지막 값, evaluation JSON과 metrics CSV, input/checkpoint 해시를 대조했다.

문서 작성 중에는 test 이미지, train/val 이미지 자체, 저장된 그래프 이미지를 열지 않았다. 새로운 모델 평가·전처리·unit test·다운로드·패키지 설치·checkpoint 변환을 실행하지 않았다. 과거 기록의 단위 테스트나 CPU/CUDA 조사 결과와 이번 문서 검증을 구분했다. 별도 기존 문서나 코드·데이터·모델·출력 파일은 수정하지 않았으며 새 Markdown 파일 하나만 추가했다.

문서 집필의 근거 대조와 별개로, 업로드 준비 단계에서는 기존 얼굴/나이 테스트와 신규 실험 테스트 45개를 실행해 43개 통과, 준비된 AI Hub 얼굴 및 smoke ONNX 부재에 따른 2개 생략을 확인했다. 명령과 공개 범위 점검은 위 업로드 준비 기록에 남겼다. 이 회귀 테스트는 새로운 본 학습이나 test 재평가가 아니다.
