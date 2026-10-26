# 데이터 출처와 현재 상태

2026-10-26 현재 이 작업 공간에는 **FFHQ-Aging의 Age LoRA용 준비본**이 있습니다. 원본·얼굴 이미지·메타데이터·평가 목록·모델 파일은 Git에서 제외합니다. 아래 AI Hub 전처리 결과는 2026-10-05의 이전 작업 기록이며, 해당 파일들은 현재 작업 공간에 없습니다.

2026-10-05 사용자 결정에 따라 **현재 나이 추정 학습은 AI Hub만 사용**합니다. UTKFace·AFAD는 다운로드·학습 선행 조건에서 제외했으며, 아래의 공개 데이터 항목은 출처 기록입니다.

| 데이터 | 출처 / 사용 범위 | 현재 확보 상태 | 라벨 |
|---|---|---|---|
| UTKFace | https://susanqq.github.io/UTKFace/ ; 비상업 연구 목적 | 아직 미확보. 2026-10-05 공식 Aligned&Cropped Drive 폴더 조회가 HTTP 404로 실패. 접근 가능한 공식 아카이브 또는 파일 URL 필요 | 파일명 age, gender(0 남/1 여). 101세 이상은 제외 |
| AFAD-Full | https://github.com/John-niu-07/tarball ; 학술 연구 범위는 기준 md 참고, 다운로드 시 제작자 조건 재확인 | 아직 미확보. `age.acquire`로 분할 파일 다운로드·바이너리 병합 가능 | `나이/111(남) 또는 112(여)/이미지` |
| AI Hub 71415 | https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71415 ; 계정 소유자가 승인받은 이용 범위 적용 | 이전 작업에서 `TS_0001.zip`, `TL_0001.zip`의 50장, 인물 1명 확인. 현재 이 작업 공간에는 없음 | JSON `age_past` = 사진 촬영 당시 나이, `age_now` = 현재 나이, `id` = 인물 ID, `gender` = male/female |
| FFHQ-Aging | https://github.com/royorel/FFHQ-Aging-Dataset ; 데이터·메타데이터 CC BY-NC-SA 4.0, 이미지별 조건도 준수 | `data/lora/`에 RGB 512×512 PNG 8,663장. train/val/test = 8,163/400/100 | 연령 그룹·성별 캡션. 정확한 나이 정답이 아니며 `age.train`에 사용하지 않음 |

## FFHQ-Aging 배치와 사용

2026-10-26에 `/home/uzer/Programme/FFHQ-Aging-Dataset/datasets/ffhq-prepared/`를 이 프로젝트의 `data/lora/`로 **이동**했습니다. 복사본이나 심볼릭 링크가 아니며, 이전 준비본 경로는 더 이상 없습니다. 외부 저장소의 다운로드 캐시·원본 라벨·선별 CSV·준비 스크립트는 이동하지 않았습니다.

```text
data/
  lora/
    train/{images/,metadata.jsonl}  # 8,163장: Age LoRA 학습 전용
    val/{images/,metadata.jsonl}    # 400장: 체크포인트·설정 선택
    test/{images/,metadata.jsonl}   # 100장: 설정 고정 후 최종 평가
  gen_eval/
    ffhq_val.csv                   # val 이미지를 참조하는 생성 평가 목록
    ffhq_test.csv                  # test 이미지를 참조하는 생성 평가 목록
  reports/
    ffhq_aging_import.json         # 출처·분포·체크섬·이동 검증 요약
    ffhq_aging_samples.jsonl        # 이미지별 원래 라벨·SHA-256·저작자·라이선스
    ffhq_aging_LICENSE.txt          # 원본 저장소의 라이선스 사본
```

- **`raw/`가 아니라 `lora/`:** 원본이 아닌 이미지·캡션 준비본이며, 기존 프로젝트의 `data/lora/train/metadata.jsonl` 계약에 맞춥니다. 중간에 `ffhq-prepared/` 폴더를 더 넣지 않습니다. 출처는 평가 목록과 보고서의 `ffhq`/`ffhq_aging` 이름으로 구분합니다.
- **분할과 식별자 보존:** 기존 train/val/test를 다시 섞지 않았습니다. `00004.png` 같은 5자리 FFHQ 이미지 ID와 캡션을 그대로 유지해 원래 라벨·이미지별 이용 조건을 추적할 수 있습니다.
- **LoRA 학습:** 학습 입력 디렉터리는 `data/lora/train`만 사용합니다. `data/lora` 전체를 재귀적으로 학습에 넣거나 val/test를 train에 합치지 않습니다. `gen.train_lora`로 공식 Diffusers 0.35.1 예제를 실행합니다. 2026-10-26 실제 12-step 시험은 학습·로딩 경로 검증일 뿐, 정식 학습이나 품질 검증 완료를 의미하지 않습니다.
- **나이 추정과 분리:** `age.train`은 AI Hub의 정확한 나이 CSV를 사용합니다. FFHQ의 연령 그룹을 임의의 정수 나이로 바꿔 넣지 않습니다. `gen/config.example.json`의 `age_lora`도 데이터 경로가 아니라 **학습된 가중치 파일 경로**이므로 변경하지 않았습니다.

각 `metadata.jsonl`의 경로는 해당 split 디렉터리 기준입니다. 이동 후에도 수정 없이 유효합니다.

```json
{"file_name":"images/00004.png","text":"a photo of a woman in her 20s"}
```

`gen.eval`은 이 JSONL을 직접 읽지 않으므로 별도 CSV를 생성했습니다. CSV의 `path`는 **CSV 디렉터리 기준 상대 경로**이며 이미지를 복제하지 않습니다. 서버로 옮길 때 `data/lora/`, `data/gen_eval/`, `data/reports/`를 함께 전송합니다.

```csv
sample_id,path
ffhq_00033,../lora/val/images/00033.png
```

실행 예시는 [서버 안내의 준비된 FFHQ-Aging 데이터](../docs/server-setup.md#준비된-ffhq-aging-데이터)를 따릅니다. 개발·검증은 val로 하고 test는 최종 평가용으로 남겨 둡니다.

### 확인한 품질과 제한

- 전체 8,663장의 PNG 검증·완전 디코딩, RGB·512×512 확인, JSONL과 이미지의 일대일 대응 검사를 통과했습니다. 이동 전후 이미지와 JSONL 총 8,666개 파일의 SHA-256·바이트 크기가 같습니다. 준비본의 합계 크기는 3,593,357,802 bytes입니다.
- 이미지 ID 중복·동일 파일 SHA-256 중복·누락·메타데이터에 없는 이미지가 없습니다. 모든 캡션이 `gen.pipeline.age_prompt`의 매핑과 일치합니다. 외부 `ffhq-selection.csv` 및 원래 `ffhq_aging_labels.csv`와 분할·연령 그룹·성별·confidence·yaw를 대조했습니다.
- 데이터 이동 당시 생성 평가 CSV 500행의 상대 경로와 분할 대응을 확인했고, val 이미지 1장으로 `gen.eval`의 세 모드 × 세 목표 나이 = 9행 Mock 실행 및 기존 생성 테스트 29개를 통과했습니다. 이 이동 검증과 후속 GPU 학습 시험은 별개의 기록이며, 모델 성능은 아직 검증하지 않았습니다.
- 선택된 표본은 `age_group_confidence > 0.8`, `abs(head_yaw) < 30`을 만족합니다. 10개 연령 그룹 × 2개 성별에 대해 val은 셀당 20장, test는 셀당 5장입니다. train은 셀당 186~500장으로 불균형하며, 부족한 셀을 복제로 채우지 않았습니다.
- **인물 단위 분리는 검증되지 않았습니다.** `person_id`가 없으므로 서로 다른 이미지 ID·파일 해시라도 같은 인물일 수 있습니다. 최종 성능 보고 전 동일인 누수를 별도로 검토해야 합니다.
- 7~9세·10~14세·15~19세 및 70~120세 train 그룹은 셀당 500장에 미달합니다. 또한 70~120세 원본 그룹을 캡션은 `70s`로 표현하므로 정확한 70대 정답으로 해석하지 않습니다. 분포와 부족 수는 [후속 학습 검증 보고서](../docs/age-lora-validation.md)에 기록했습니다.
- 외부 `prepare_ffhq_aging.py`는 Hugging Face의 `NUS-SRI-2025/FFHQ-Aging-Dataset` 미러에서 받은 1024px 이미지를 RGB·512px로 LANCZOS 리사이즈하도록 작성돼 있습니다. 이번에는 준비본을 검증했으며 원래 1024px 파일과의 대응이나 프로젝트의 InsightFace 재정렬을 검증·실행하지 않았습니다.
- 데이터·메타데이터는 **CC BY-NC-SA 4.0** 조건과 원 논문 인용 의무를 따르며, 이미지별 원저작자·라이선스도 준수해야 합니다. 선택된 이미지의 출처 URL·저작자·라이선스를 원본 `ffhq-dataset-v2.json`에서 `ffhq_aging_samples.jsonl`에 보존했습니다. 상업 이용 허가나 얼굴 당사자의 개별 동의를 확인한 것은 아닙니다.

인용: Roy Or-El, Soumyadip Sengupta, Ohad Fried, Eli Shechtman, Ira Kemelmacher-Shlizerman, *Lifespan Age Transformation Synthesis*, ECCV 2020, https://arxiv.org/pdf/2003.09764.pdf. 원 이미지 데이터셋: NVIDIA [Flickr-Faces-HQ](https://github.com/NVlabs/ffhq-dataset).

후속 `gen.validate_lora_data` 전체 재검사도 8,663장 및 출처 기록의 confidence/yaw·캡션·해시 대조를 통과했습니다. 데이터 재다운로드·재분할·수정은 하지 않았습니다. 결과는 `outputs/age_lora_validation_20261026/data/validation.json`, 이미지별 해시는 `images.json`, 20개 연령·성별 셀의 split별 분포는 `groups.csv`에 있습니다.

## AI Hub 이전 준비 기록 (2026-10-05)

이 절은 이전 작업 환경의 기록입니다. 현재 작업 공간에는 아래 AI Hub 이미지·CSV·보고서가 없으므로 별도로 준비해야 합니다.

당시 AI Hub 원본은 Downloads 폴더에 그대로 보존하고 `data/raw/aihub`에 검증 후 압축 해제했습니다. 받은 원본의 정확한 바이트 크기·SHA-256·ZIP CRC 검사 결과와 나이별 수는 당시 `reports/aihub_inventory.json`에 기록했습니다. ZIP 파일명의 선행 `/`는 데이터 배포 포맷에 맞춰 제거하되 경로 이탈 항목은 거부합니다.

당시 AI Hub 표본은 나이 1~29세의 **동일 인물** 사진 50장입니다. 나이가 다양해도 독립적인 50명으로 취급할 수 없습니다. 인물 단위 분할 결과 val/test가 비어 있으면 학습 코드가 중단합니다. 여러 인물의 전체 데이터를 확보한 뒤 ImageNet 사전학습 MobileNetV3에서 바로 AI Hub 단독 학습을 시작합니다.

전처리·학습 기본 경로는 `data/age_aihub`입니다. 이전 작업에서 이 경로에 실제 표본을 전처리해 train/val/test = 48/0/0을 생성했습니다. 당시 `data/age_stage2`와 pilot 산출물도 보존했습니다.

50장 전처리 시험과 전체 표본 처리를 실행했습니다. 48장 정렬 성공, 다중 얼굴 2장 제외이며 `data/age_stage2/{train,val,test}.csv`의 크기는 48/0/0입니다. 세부 제외 사유는 `reports/aihub_preprocess_failures.csv`, 분할·요약은 `reports/{split_summary,dataset_summary}.csv`에 있습니다. `data/age/pilot_aihub_50`에는 별도 시험 결과를 보존했습니다.

허용된 교육·비상업 연구 프로젝트 범위에서만 사용합니다. 상업 서비스 사용 가능 여부는 확인되지 않았습니다. InsightFace 공식 사전학습 가중치도 비상업 연구용이며 코드(MIT) 조건과 다릅니다.

전처리 CSV의 `path`는 프로젝트 내부일 때 프로젝트 루트 기준 상대 경로입니다. GPU 서버로 프로젝트를 옮겨도 같은 디렉터리 구조로 실행할 수 있습니다. `person_id`가 없는 공개 데이터는 동일인 누수 여부를 완전히 확인할 수 없다는 한계를 평가에 남깁니다.
