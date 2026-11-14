# 공개용 실험 스냅샷 사용 안내

## 승인된 범위

2026-11-14 사용자 확인 후 `feat/age-estimation-experiments-20261114` 브랜치를 만들었다. main에 직접 커밋하거나 푸시하지 않는다. 데이터셋을 제외하고 코드·실험 문서·직접 학습한 다섯 모델·실험 출력의 공개용 사본을 포함한다. 외부 사전학습 가중치는 동봉하지 않고 공식 재다운로드 방식으로 대체한다. 로컬 원본 데이터·모델·실험 출력은 수정하지 않았다.

실험 경과는 [전체 실험 기록](public-age-experiments-20261114.md), 초기 조사와 범위 판단은 [업로드 준비 기록](age-experiment-upload-scope-20261114.md)에 있다. 이 문서가 실제 공개 사본의 정책을 보완한다.

## 모델과 결과의 차이

- 다섯 `outputs/age/*/best.pt`는 Git LFS로 저장한다. `git lfs pull`로 바이너리를 받아야 한다.
- 원본의 `model` tensor와 optimizer/scheduler, epoch, val MAE, optimizer step 정보는 유지했다. 전체 train/val의 이미지 경로·정답 나이·성별·식별자·해시를 담은 `split_rows`는 제거했다. run의 서버 경로는 저장소 기준 표기로 바꿨다.
- `public_export` 필드는 원본 checkpoint SHA-256과 제거 내역을 담는다. 공개 사본은 원본과 해시가 다르며, 실험 보고서의 원본 checkpoint 해시를 현재 공개 파일 해시로 오해하면 안 된다.
- 여섯 행별 예측 CSV는 `source`, 선택적인 `version`, `prediction`만 남겼다. 원래 나이·ID·경로를 제거하고 공개 열 기준으로 정렬해 원본 manifest의 행 순서도 남기지 않았다. 예측값과 개수는 유지했지만 표본 대응 정보는 제거됐다.
- 과거 FFHQ 문서의 소수 실제 표본 ID와 CLI/스키마 예시도 익명 사례 A/B 및 `EXAMPLE_ID`로 바꿨다. 실험 수치와 실패 설명은 유지했으며 기존 main 문서는 수정하지 않았다.
- 따라서 공개 CSV만으로 MAE를 다시 계산하거나, 공개 checkpoint만으로 평가 표본이 학습/검증과 독립적인지 증명할 수 없다. 기존 `age.evaluate`의 split 검증을 우회하지 말고, 승인된 로컬 원본 checkpoint·분할 목록 또는 새로 적법하게 준비한 평가 계약을 사용해야 한다.
- `history.csv`, `metrics.csv`, `model_metrics.csv`와 집계 PNG는 원본과 동일하다. JSON의 서버 경로 표기만 정리했으며 과거 평가의 원본 해시와 수치는 유지했다. 변환 대상으로 조사한 기존 산출물 49개의 원본/공개 해시와 바이트 수, 제외/정제 사유는 [산출물 명세](public-artifact-manifest-20261114.json)에 기록했다. 새 코드·문서·라이선스·다운로드 안내는 이 49개에 포함되지 않으며 Git에서 별도로 추적한다.
- 메타데이터 제거는 학습 데이터의 이용 조건을 해제하거나 모델의 원본 암기를 배제하는 익명성 보증이 아니다. 비상업 연구용 조건, 동일인 누수, 검은 배경 및 고령 표본 한계는 그대로 남는다.

공개용 사본 생성 도구는 `tools/package_age_research.py`다. 별도 Git 작업 트리에만 출력하고 기존 산출물 디렉터리를 덮어쓰지 않는다. 원본 데이터와 라벨 목록을 읽어 가중치를 재학습하는 도구가 아니며, 기존 로컬 산출물에서 승인된 공개용 사본을 만든다. 최종 승인된 업로드용 `.gitignore`와 `.gitattributes`는 별도로 검토했다.

## 외부 의존성

`models/EXTERNAL_WEIGHTS.json`은 제외한 ImageNet 초기 가중치, InsightFace ZIP/ONNX의 공식 URL·해시·준비 명령을 제공한다. 외부 가중치를 사용하기 전에 해당 이용 조건을 확인해야 한다.

```bash
git lfs pull
python -m age.setup_models --model-root models/insightface
```

직접 학습한 모델의 state dict를 `age.model.build_model(False)`로 불러오면 ImageNet 초기 가중치 다운로드는 필요 없다. 학습을 새로 시작할 때는 기존 코드가 `models/torch` 캐시에 초기 가중치를 다운로드한다. 이 브랜치의 `data/`에는 문서만 있으므로, 데이터 이용 승인을 받고 별도로 데이터를 준비하지 않으면 학습·평가 명령을 바로 실행할 수 없다.

SAM은 source snapshot 대신 공식 저장소의 고정 커밋 `c1895aef275e702fba7560284dc16df60d65210e`를 가리키는 Git submodule이다. upstream에는 예시 얼굴 이미지와 서로 다른 이용 조건의 구성요소가 있어 이 저장소에 복제하지 않았다. 이번 나이 추정 실험에는 SAM이 필요하지 않다. 다른 생성 실험에서 필요할 때만 upstream 라이선스를 검토한 후 별도로 가져온다.

```bash
git submodule update --init third_party/SAM
```

기존 outputs의 LoRA 학습 스크립트 symlink는 고정 v0.35.1 실파일로 바꾸고 Apache-2.0 LICENSE와 NOTICE를 동봉했다. InsightFace Python 런타임은 MIT LICENSE 및 원본 provenance와 함께 포함한다. `.venv`, 캐시, 자격증명, 중첩 `.git`, 데이터 이미지·원래 라벨·분할 manifest는 포함하지 않는다.

## 검증

공개용 변환 전 기존/신규 테스트 45개 중 43개 통과, 2개 조건부 생략을 확인했다. 공개용 정제 도구의 추가 테스트 3개도 통과했다. 공개 작업 트리에서는 이 중 관련 33개를 다시 실행해 모두 통과했다.

별도 사후 감사에서 모델별 312개 tensor와 optimizer/scheduler, epoch·지표가 원본과 정확히 일치함을 확인했다. 변환 대상 manifest의 원본 49개 및 공개 파일 45개 해시·크기, 예측 CSV 6개의 값·개수 보존과 라벨·경로·원본 행 순서 제거, 집계 14개의 원본 해시 일치도 확인했다. 다섯 staged LFS pointer의 OID·size는 워크트리 및 실제 LFS object와 일치했고 `git lfs fsck --objects`, `git diff --cached --check`가 통과했다. 최종 검사에서 데이터 이미지, 외부 사전학습 바이너리, symlink, 중첩 Git 디렉터리, 주요 자격증명 패턴은 발견하지 않았다. 이는 알려진 검사 범위의 결과이지 법적 배포 권한·완전한 익명성·모델 일반화에 대한 보증은 아니다. 공개용 변환은 새로운 test 평가가 아니다.
