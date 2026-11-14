# 2026-11-14 나이 추정 실험 공개 업로드 준비

이 문서는 최초 업로드 전 조사 결과와 당시 제안 범위를 보존한다. 이후 사용자는 제안 브랜치와 외부 가중치 재다운로드 방식을 승인했다. 실제 공개용 사본의 정제 방식은 [스냅샷 사용 안내](public-snapshot-usage-20261114.md)에 기록한다. 준비 문서 작성 자체는 푸시 완료를 뜻하지 않는다.

## 저장소와 브랜치

- 대상 원격: `https://github.com/HanyangTechAI/2026-2-HAI-ProjectTeam3.git`
- 조사 당시 원격은 공개 저장소다.
- 새 브랜치도 같은 공개 범위를 따른다. main이 아닌 브랜치나 LFS를 쓰는 것으로 파일이 비공개가 되지는 않는다.
- 현재 로컬 및 원격 `main` 기준 커밋: `2457c3c8fa27b7598060272e75cefd79599ffc3d`.
- 제안 브랜치: `feat/age-estimation-experiments-20261114`.
- `main`에 직접 커밋하거나 푸시하지 않는다. 기존 로컬 작업과 공유 데이터는 보존한다.
- 별도 업로드 작업 트리에 현재 main의 코드 및 문서 변경을 반영하고, 승인받은 산출물의 실파일을 복사하는 방식이 적합하다.
- PR 생성, main 병합, force push는 요청 범위에 포함되지 않는다.

## 링크와 실제 위치

현재 main 작업 디렉터리의 링크만 커밋하면 다른 머신에서 파일 내용을 사용할 수 없다.

| 로컬 항목 | 실제 위치와 주의점 |
|---|---|
| `branches/main/models` | `/workspace/hai/shared/models`를 참조한다. |
| `branches/main/outputs` | `/workspace/hai/outputs/main`을 참조한다. 이번 나이 학습의 `/workspace/hai/outputs/age`와 평가는 이 링크 아래에 있지 않다. |
| 이번 전체 실험 출력 | `/workspace/hai/outputs` 전체를 확인해야 한다. main의 outputs 링크만 복사하면 학습 기록과 모델 대부분이 누락된다. |
| `branches/main/data/*` | 공유 데이터셋 링크다. 데이터 제외 요청에 따라 원본과 전처리 이미지, 라벨 및 분할 목록을 따라가며 복사하지 않는다. 기존 추적 문서 `data/README.md`는 데이터셋 자체가 아니다. |
| `branches/main/third_party/SAM` | `/opt/hai/third_party/SAM` 링크. 현재 커밋은 `c1895aef275e702fba7560284dc16df60d65210e`. 최상위 MIT 외에 일부 구성요소는 별도 조건을 가지므로 공개 브랜치에는 예시 이미지/소스 복제 대신 upstream submodule 참조만 포함한다. |

## 조사 당시 산출물 용량

공유 models와 전체 outputs에서 일반 파일 48개를 확인했다. 합계는 883,725,169 bytes, 약 842.8 MiB다. 별도 심볼릭 링크 하나는 이 일반 파일 수와 바이트 합계에 포함되지 않는다. 이것은 공개용으로 정제하기 전의 크기다.

| 범위 | 일반 파일 수 | 바이트 |
|---|---:|---:|
| `/workspace/hai/shared/models` | 12 | 329,021,709 |
| `/workspace/hai/outputs` | 36 | 554,703,460 |

일반 Git blob의 100 MiB를 초과하는 파일은 다음과 같다. 바이너리를 포함할 경우 Git LFS 사용을 제안하며, LFS 저장 공간과 다운로드 전송량은 원격 계정의 할당량 적용을 받는다. LFS는 접근 제어나 데이터 익명화 수단이 아니다.

| 파일 | 바이트 |
|---|---:|
| `models/insightface/buffalo_l.zip` | 288,621,354 |
| `outputs/age/public/best.pt` | 108,665,891 |
| `outputs/age/public_balanced_v1/best.pt` | 108,665,891 |
| `outputs/age/public_weak_crop_v1/best.pt` | 108,665,891 |
| `outputs/age/public_aligned_crop_retry_v1/best.pt` | 106,177,571 |
| `outputs/age/public_aligned_weak_retry_v1/best.pt` | 106,177,635 |

## 데이터 제외는 data 디렉터리만 제외하는 것이 아님

다섯 `best.pt`는 순수 가중치 파일이 아니다. `model`, `run`, `split_rows`, `optimizer`, `scheduler`와 실험 선택 정보를 포함한다. `split_rows`에는 학습 및 검증 이미지의 절대 경로, 나이, 성별, 출처, 원본 식별자와 해시가 들어 있다. `person_id`가 비어 있다는 사실은 나머지 정보가 데이터 목록이 아니라는 뜻이 아니다.

- 첫 세 실험은 각각 train 162,823행 및 val 24,313행을 포함한다.
- 정렬 두 실험은 각각 train 160,590행 및 val 23,838행을 포함한다.
- 모델을 공개하려면 원본 체크포인트를 그대로 수정하는 대신 별도 공개용 사본에서 데이터 목록을 제거하는 방안을 먼저 승인받는다.
- 공개용 사본은 원본과 해시가 달라진다. 원본의 해시와 정제 내용을 별도 기록하고, 원본 실험 체크포인트와 동일 파일이라고 표현하지 않는다.
- `split_rows`를 제거하면 기존 `age.evaluate`의 체크포인트 기반 누수 검사에 그대로 사용할 수 없다. 재평가에는 로컬 원본 체크포인트 또는 별도로 관리하는 분할 정보가 필요하다.

다음 이미지별 결과 파일에도 원래 나이 라벨과 경로 또는 식별자가 포함된다. 데이터 제외 요청과 공개 범위 검토 없이 그대로 올리지 않는다.

```text
outputs/age/compare_aligned_val_weak_crop_v1/predictions.csv
outputs/age/compare_aligned_val_balanced_flip_v1/predictions.csv
outputs/age/compare_aligned_retry_balanced_flip_v1/predictions.csv
outputs/age/eval_aligned_flip_crop_v1/model_predictions.csv
outputs/age/eval_aligned_flip_weak_v1/model_predictions.csv
outputs/eval/public_initial_20261114/model_predictions.csv
```

집계 지표, epoch별 이력, 집계 그래프는 이미지 자체나 전체 정답 라벨 목록과 구별한다. `run.json`과 `summary.json`의 서버 절대 경로도 공개용 사본에서 확인한다. 기존 산출물을 생략하거나 정제한다면 범위와 이유를 공개 기록에 남긴다.

## 외부 모델과 이용 조건

- MegaAge 보존 이용 조건에는 데이터베이스 일부의 외부 공개 및 배포를 금지하는 조항이 있다. 경로를 바꾸거나 LFS에 저장해도 원본 라벨 재배포 문제가 없어지지 않는다.
- InsightFace Python 런타임은 MIT지만 buffalo_l 사전학습 가중치는 별도의 비상업 연구 조건을 따른다. 코드 라이선스가 가중치나 원본 얼굴 데이터의 권리를 대체하지 않는다.
- `buffalo_l.zip`에는 이번 정렬에 사용하는 검출 및 속성 모델 외에 얼굴 인식과 랜드마크 모델도 포함돼 있다. 정렬에 실제 쓰인 두 ONNX만 있다고 가정하면 안 된다.
- ImageNet 사전학습 MobileNetV3 가중치 또한 코드 라이선스만으로 재배포 권한이 모두 확인됐다고 주장하지 않는다.
- TorchVision의 공식 [사전학습 모델 안내](https://github.com/pytorch/vision/blob/main/README.md#pre-trained-models)는 가중치에 학습 데이터에서 유래한 별도 라이선스와 조건이 있을 수 있으며 사용자가 권한을 확인해야 한다고 명시한다.
- 외부 사전학습 파일의 동봉 여부와 조건을 최종 승인 범위에 포함한다. 동봉하지 않기로 한다면 공식 재다운로드 경로, 버전, 해시와 설정 코드를 남긴다.
- 직접 학습한 모델이라도 학습 데이터 조건, 얼굴 정보 및 파생물의 이용 제한을 자동으로 벗어나는 것은 아니다. 이 실험 기록은 상업 이용 허가나 독립적인 권리 검토를 뜻하지 않는다.

## 검증 범위

업로드 전 읽기 전용 조사에서 일반 파일의 주요 비밀키 및 토큰 패턴, 텍스트 자격증명 패턴을 확인했으며 발견되지 않았다. 패턴 검사 통과가 모든 민감정보의 부재를 보장하지 않는다.

코드 검증은 다음 두 명령으로 합계 45개 테스트를 실행했다. 43개 통과, 2개 조건부 생략이다.

```bash
python -m unittest age.test_align_public age.test_align_public_retry \
  tests.test_prepare_public tests.test_catalog tests.test_compare_aligned_val \
  tests.test_train_public_balanced tests.test_train_public_weak_crop

python -m unittest tests.test_face tests.test_age
```

생략된 두 테스트는 기존 AI Hub 준비 얼굴과 smoke ONNX 파일 부재 때문이다. 이번 준비 과정에서 본 학습을 다시 수행하거나 test 데이터를 다시 평가한 것은 아니다.

## 최종 확인 후 할 일

1. 브랜치 이름, LFS 사용 및 외부 사전학습 파일 동봉 여부를 확인받는다.
2. 로컬 원본은 보존하고, 데이터 제외 범위에 맞는 공개용 실파일 스냅샷을 만든다.
3. 데이터셋 이미지·라벨·분할 목록·비밀정보·가상환경·캐시·중첩 Git 메타데이터가 커밋에 포함되지 않는지 검사한다.
4. LFS pointer와 실제 LFS 파일 대상, 변경 diff와 커밋 목록을 검토한다.
5. 승인받은 새 원격 브랜치만 푸시하고 브랜치 URL 및 제외/정제 내역을 보고한다.
