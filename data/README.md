# 데이터 출처와 현재 상태

2026-10-05에 확인한 상태입니다. 원본·얼굴 이미지·모델 파일은 Git에서 제외합니다.

2026-10-05 사용자 결정에 따라 **현재 나이 추정 학습은 AI Hub만 사용**합니다. UTKFace·AFAD는 다운로드·학습 선행 조건에서 제외했으며, 아래의 공개 데이터 항목은 출처 기록입니다.

| 데이터 | 출처 / 사용 범위 | 현재 확보 상태 | 라벨 |
|---|---|---|---|
| UTKFace | https://susanqq.github.io/UTKFace/ ; 비상업 연구 목적 | 아직 미확보. 2026-10-05 공식 Aligned&Cropped Drive 폴더 조회가 HTTP 404로 실패. 접근 가능한 공식 아카이브 또는 파일 URL 필요 | 파일명 age, gender(0 남/1 여). 101세 이상은 제외 |
| AFAD-Full | https://github.com/John-niu-07/tarball ; 학술 연구 범위는 기준 md 참고, 다운로드 시 제작자 조건 재확인 | 아직 미확보. `age.acquire`로 분할 파일 다운로드·바이너리 병합 가능 | `나이/111(남) 또는 112(여)/이미지` |
| AI Hub 71415 | https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71415 ; 계정 소유자가 승인받은 이용 범위 적용 | 제공된 `TS_0001.zip`, `TL_0001.zip`만 확보. 50장, 인물 1명. 승인 내역·전체 데이터 버전은 이 폴더만으로 확인 불가 | JSON `age_past` = 사진 촬영 당시 나이, `age_now` = 현재 나이, `id` = 인물 ID, `gender` = male/female |
| FFHQ-Aging | https://github.com/royorel/FFHQ-Aging-Dataset | 나이 추정 학습 대상 아님. LoRA 단계에서 별도 준비 | 이번 나이 추정 구현 범위 밖 |

AI Hub 원본은 Downloads 폴더에 그대로 보존하고 `data/raw/aihub`에 검증 후 압축 해제했습니다. 받은 원본의 정확한 바이트 크기·SHA-256·ZIP CRC 검사 결과와 나이별 수는 `reports/aihub_inventory.json`에 있습니다. ZIP 파일명의 선행 `/`는 데이터 배포 포맷에 맞춰 제거하되 경로 이탈 항목은 거부합니다.

현재 AI Hub 표본은 나이 1~29세의 **동일 인물** 사진 50장입니다. 나이가 다양해도 독립적인 50명으로 취급할 수 없습니다. 인물 단위 분할 결과 val/test가 비어 있으면 학습 코드가 중단합니다. 여러 인물의 전체 데이터를 확보한 뒤 ImageNet 사전학습 MobileNetV3에서 바로 AI Hub 단독 학습을 시작합니다.

현재 전처리·학습 기본 경로는 `data/age_aihub`입니다. 이 경로에 실제 표본을 전처리해 train/val/test = 48/0/0을 생성했습니다. 앞서 만든 `data/age_stage2`와 pilot 산출물은 보존합니다.

50장 전처리 시험과 전체 표본 처리를 실행했습니다. 48장 정렬 성공, 다중 얼굴 2장 제외이며 `data/age_stage2/{train,val,test}.csv`의 크기는 48/0/0입니다. 세부 제외 사유는 `reports/aihub_preprocess_failures.csv`, 분할·요약은 `reports/{split_summary,dataset_summary}.csv`에 있습니다. `data/age/pilot_aihub_50`에는 별도 시험 결과를 보존했습니다.

허용된 교육·비상업 연구 프로젝트 범위에서만 사용합니다. 상업 서비스 사용 가능 여부는 확인되지 않았습니다. InsightFace 공식 사전학습 가중치도 비상업 연구용이며 코드(MIT) 조건과 다릅니다.

전처리 CSV의 `path`는 프로젝트 내부일 때 프로젝트 루트 기준 상대 경로입니다. GPU 서버로 프로젝트를 옮겨도 같은 디렉터리 구조로 실행할 수 있습니다. `person_id`가 없는 공개 데이터는 동일인 누수 여부를 완전히 확인할 수 없다는 한계를 평가에 남깁니다.
