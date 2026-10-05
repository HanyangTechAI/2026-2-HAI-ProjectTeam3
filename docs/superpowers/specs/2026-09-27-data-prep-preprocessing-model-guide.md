# 포토부스 데이터 준비·전처리·모델 개발 실행 지침

> 2026-10-05 변경 결정: 나이 추정은 **AI Hub 71415 단독 학습**으로 진행한다. UTKFace·AFAD 다운로드·Stage 1 학습은 필수가 아니다. 승인된 AI Hub 전체 데이터의 검증·공통 얼굴 전처리·인물별 분할 후 ImageNet MobileNetV3에서 바로 시작한다(batch 128, lr 5e-4, 최대 30 epoch). 아래의 공개 데이터 선행 단계와 Stage 2 체크포인트 조건은 이전 계획으로 남겨 두며, 최신 실행은 루트 `README.md`를 따른다. 생성 모델 단계는 그대로 유지한다.

- 작성일: 2026-09-27
- 기준 문서: [나이 추정 & 연령 변환 포토부스 — 데이터/모델 설계](2026-09-27-age-photobooth-models-design.md)
- 대상: 데이터·나이 추정 담당 B팀, 생성 모델 담당 C팀, 결과를 연동할 D팀
- 범위: 10/29 중간발표까지의 데이터, 모델, 평가. 카메라·버튼·프린터·웹 UI는 별도 Phase 2.

## 1. 지금 무엇부터 할 것인가

현재 저장소에는 설계 문서만 있고 데이터·학습 코드·체크포인트는 없다. 데이터가 들어오기를 기다리지 말고 다음 순서로 진행한다.

| 순서 | 담당 | 작업 | 끝났다고 판단할 산출물 |
|---|---|---|---|
| 1 | B | AI Hub 71415 접근 신청, UTKFace·AFAD-Full·FFHQ-Aging의 원본/라벨 확보 경로와 사용 조건 확인 | `data/README.md`의 출처·승인 상태·사용 범위 기록 |
| 2 | B | 소량의 공개 데이터로 얼굴 검출·정렬과 라벨 변환을 먼저 시험 | 20~50장 처리 결과, 실패 사유 로그 |
| 3 | B | 전체 공개 데이터 전처리, 분할, 품질 통계 | `data/age/*.csv`, `data/lora/train/metadata.jsonl`, 데이터 요약표 |
| 4 | B·C 병행 | B: 나이 추정 Stage 1. C: LoRA 학습용 데이터 검수와 생성 파이프라인 최소 실행 | MAE 표·ONNX 후보, 첫 연령 변환 예시 |
| 5 | C | Age LoRA 학습·FaceID 결합·평가 | 선택한 LoRA, 4컷용 생성 이미지 3장, 평가 CSV |
| 6 | B·C·D | 모델 출력 연결 및 종단 간 측정 | 원본+생성 3장의 샘플, 처리 시간과 실패 사례 |

**AI Hub 접근 신청·승인만으로 데이터가 자동으로 이 작업 공간에 내려오지는 않는다.** 신청과 본인 인증은 계정 소유자가 진행한다. 승인 후에는 실제 다운로드 권한, 공식 다운로드 도구 또는 API 키, 네트워크 접근, 저장 공간을 확인해야 한다. 이 환경에 승인된 다운로드 경로가 제공되면 다운로드·무결성 검사·압축 해제·전처리를 자동화할 수 있다. 키나 비밀번호를 문서나 채팅에 적지 않는다. AI Hub 승인이 늦어지면 공개 데이터로 Stage 1과 발표를 진행한다. 승인 이후에만 AI Hub 변환기와 Stage 2 파인튜닝을 추가한다. 승인 전 접근을 전제로 학습 일정을 잡지 않는다.

## 2. 데이터 확보와 관리

### 2.1 각 데이터의 용도

| 데이터 | 목적 | 먼저 확인할 것 |
|---|---|---|
| UTKFace | 나이 추정 Stage 1, 여러 나이대의 초기 학습 | 파일명에서 나이·성별 파싱 가능 여부, 손상 파일, 허용된 사용 범위 |
| AFAD-Full | 나이 추정 Stage 1, 아시아인 데이터 보강 | 라벨 포맷, 나이대 편중, 허용된 사용 범위 |
| AI Hub 71415 | 승인 시 한국인 데이터로 Stage 2, 별도 평가 | 접근 승인, 실제 라벨 구조, 인물 ID 유무, 이용 약관 |
| FFHQ-Aging | Age LoRA 학습과 학습에 쓰지 않을 평가 이미지 | `age_group`, confidence, 성별, pose 컬럼의 실제 이름·값·사용 조건 |

실행 절차:

1. 원본 파일은 `data/raw/{utkface,afad,ffhq_aging,aihub}/`에 출처별로 보관한다. 원본을 덮어쓰지 않는다.
2. `data/README.md`에 출처 URL, 받은 날짜, 데이터 버전/파일 수, 접근 승인 상태, 라이선스·동의 범위, 라벨 설명, 체크섬 또는 아카이브 크기를 기록한다. 허용 범위가 확인되지 않은 데이터는 학습에 넣지 않는다.
3. 이미지와 얼굴 임베딩은 Git에 올리지 않는다. 팀원 사진은 명시적으로 동의받은 평가용으로만 사용하고 별도 제한된 위치에 둔다.
4. FFHQ-Aging은 필요한 규모인 약 1만 장부터 확보한다. 다만 공식 다운로드 스크립트는 임의의 1만 장만 받는 옵션이 없으므로 아래 2.2절의 선별 다운로드 절차가 먼저 필요하다. 확보한 라벨과 이미지의 대응 여부를 검사한 뒤 추가 다운로드를 결정한다.
5. UTKFace·AFAD-Full·FFHQ-Aging은 AI Hub 승인과 별개다. 각각의 배포처에서 접근 조건과 다운로드 가능 여부를 확인한다.

### 2.2 데이터셋별 실제 확보 방법과 자동화 가능 범위

**신청·승인과 다운로드는 별개다.** 현재 저장소에는 다운로드 프로그램도 데이터도 없고, 이 작업 환경의 외부 네트워크는 제한되어 있다. 아래의 ‘자동화 가능’은 **다운로드가 허용된 네트워크와 저장 공간이 준비되었을 때** 스크립트로 받을 수 있다는 뜻이며, 지금 다운로드가 완료되었다는 뜻이 아니다.

| 데이터 | 공식 입수 경로 | 사람의 선행 작업 | 다운로드 자동화 | 현재 확인한 제약 |
|---|---|---|---|---|
| UTKFace | [제작자 데이터 페이지](https://susanqq.github.io/UTKFace/)의 Google Drive ZIP | 비상업 연구 사용 조건 확인 | 공개 ZIP을 받을 수 있는 환경에서는 스크립트화 가능. **공식 자동 다운로드 스크립트는 확인되지 않음** | Google Drive의 실제 파일 접근·할당량은 다운로드 환경에서 확인 필요 |
| AFAD-Full | [제작자 페이지](https://github.com/John-niu-07/afad-dataset.github.io/blob/master/index.html)의 [AFAD-Full 분할 저장소](https://github.com/John-niu-07/tarball) | 학술 연구 목적의 사용 범위 확인 | 공개 Git 저장소를 받으면 분할 파일 병합·압축 해제를 스크립트화 가능 | 여러 `AFAD-Full.tar.xz*` 조각을 빠짐없이 받아야 함 |
| FFHQ-Aging | [제작자 저장소](https://github.com/royorel/FFHQ-Aging-Dataset)의 라벨 CSV와 공식 다운로드 스크립트; 실제 얼굴 원본은 FFHQ의 Google Drive | 비상업 조건과 이미지별 원저작자 조건 확인. Google Drive 할당량 초과 시 본인 Google 인증이 필요할 수 있음 | 전체 데이터 자동 다운로드는 공식 스크립트로 가능. **선별 1만 장은 스크립트 수정 또는 별도 선별 다운로드 코드 필요** | 기본 `.bat`는 256px 다운로드와 세그멘테이션까지 실행함. 그대로 실행하면 1만 장만 받지 않음 |
| AI Hub 71415 | [안면 인식 에이징 이미지 데이터 페이지](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71415) | 회원가입·본인 인증·데이터 이용 신청 및 승인. 페이지에는 내국인만 신청 가능하다고 표시됨 | 승인 후 웹 다운로드 또는 [공식 aihubshell](https://aihub.or.kr/static/pdf/aihubshell_%EA%B0%80%EC%9D%B4%EB%93%9C.pdf)의 API 키 기반 CLI 다운로드 가능 | 승인과 API 키/접근 경로가 제공되기 전에는 자동 다운로드 불가 |

실행 세부 사항:

1. **UTKFace:** 제작자 페이지의 `Aligned&Cropped Faces` ZIP(페이지 표기 107MB)을 먼저 시험한다. 원본 이미지가 필요하면 `In-the-wild Faces` ZIP(페이지 표기 1.3GB)을 받는다. ZIP을 `data/raw/utkface/` 아래에 풀고 이미지 수와 `[age]_[gender]_[race]_...jpg` 파일명 파싱 결과를 기록한다. 설계대로 이후 `buffalo_l`로 다시 정렬한다. 공식 Google Drive 링크가 이 환경에서 실제로 내려받아지는지는 아직 검증하지 않았다.
2. **AFAD-Full:** 제작자 링크는 현재 `John-niu-07/tarball` 공개 저장소로 연결된다. 저장소의 모든 `AFAD-Full.tar.xz*` 조각을 받은 뒤, 제공된 `restore.sh`처럼 **파일명 순서대로 바이너리 병합**해 `AFAD-Full.tar.xz`를 만들고 압축을 푼다. Windows에서는 Git Bash의 스크립트 또는 Python 바이너리 스트림을 사용한다. 텍스트 읽기/쓰기 명령으로 합치면 아카이브가 손상될 수 있다. 병합 후 아카이브 검사와 이미지·라벨 수 확인을 한다.
3. **FFHQ-Aging:** 제작자 저장소의 `ffhq_aging_labels.csv`를 먼저 받고 confidence·pose 조건으로 후보 이미지 ID를 선정한다. 공식 `download_ffhq_aging.py --debug --resolution 512`는 **50장 파이프라인 시험용**이다. 본 실행에서는 제작자 스크립트의 다운로드 목록(`specs`)을 선정 ID로 제한하는 작은 수정이 필요하다. 수정하지 않으면 전체 목록을 처리한다. 구형 `.bat`를 그대로 실행하면 `--resolution 256`과 세그멘테이션 단계가 함께 실행되므로 LoRA용 512px 확보 지침으로 사용하지 않는다. Google Drive 할당량 문제가 생기면 제작자 README의 PyDrive 인증 경로를 검토한다. 자격 증명 파일을 저장소에 넣지 않는다.
4. **AI Hub:** 데이터셋 71415에 사용 신청하고 승인 상태를 확인한다. 승인 후 웹에서 필요한 파일을 선택하거나, `aihubshell`의 `list`로 datasetkey/filekey를 확인한 뒤 필요한 파일을 다운로드한다. 공식 가이드는 전체 다운로드 후 병합·압축 해제를 수행하며 원본 크기의 2~3배 저장 공간을 권장한다. API 키는 채팅·md·소스 코드에 저장하지 않는다. 승인된 계정으로 실제 접근되는 것을 확인한 뒤에만 전처리 작업에 넣는다.

**자동화 경계:** UTKFace·AFAD-Full은 공개 링크를 사용하는 다운로드 스크립트를 만들 수 있다. FFHQ-Aging은 소량 선별 코드가 추가로 필요하다. AI Hub는 계정 소유자의 신청·인증·승인이 먼저다. 어떤 데이터도 신청만 하면 자동으로 이 저장소에 나타나지 않는다.

### 2.3 디렉터리와 파일 계약

```text
data/
  README.md
  raw/{utkface,afad,ffhq_aging,aihub}/
  age/images/
  age/{train,val,test}.csv
  lora/train/images/
  lora/train/metadata.jsonl
  lora/val/images/
  lora/val/metadata.jsonl
  reports/{dataset_summary.csv,preprocess_failures.csv,split_summary.csv}
outputs/{age/,age_lora/,eval/}
```

- `data/age/*.csv`: `path,age,gender,source`를 필수 열로 한다. `path`는 실제 파일을 가리키고 `age`는 0~100 정수로 검증한다. 원본의 인물 ID가 있으면 별도 `person_id` 열로 보존한다.
- `data/lora/*/metadata.jsonl`: 한 줄에 `{"file_name":"images/000001.png","text":"a photo of a woman in her 30s"}` 형식으로 저장한다. 실제 `file_name` 경로와 이미지 파일이 일치해야 한다.
- 실패 로그: 원본 식별자·출처·실패 단계·사유만 기록한다. 디버깅용 얼굴 이미지나 임베딩을 로그에 넣지 않는다.
- `outputs/`에는 모델 가중치·평가 결과를 두고, 최종 선택 모델의 데이터 버전·학습 설정·체크포인트 경로를 함께 기록한다.

## 3. 공통 전처리: B팀이 한 번 구현해 두 모델에 제공

### 3.1 입력 검사와 얼굴 정렬

1. 이미지를 읽고 EXIF 방향을 적용한 뒤 RGB로 통일한다. 읽기 실패, 비정상 크기, 중복 파일을 집계한다.
2. `insightface buffalo_l`의 SCRFD 검출 결과에서 사용할 얼굴을 선택한다. 학습 이미지에서 얼굴이 여러 개면 정답 라벨이 누구의 것인지 불명확하므로 자동 채택하지 않고 제외하거나 수동 확인한다. 실시간 추론에서는 가장 큰 얼굴 하나를 선택한다.
3. 5점 랜드마크로 similarity transform을 계산해 정렬한다. 이미 얼굴이 크롭된 UTKFace·AFAD도 동일한 검출·정렬 절차를 거친다.
4. 나이 추정용은 정렬된 얼굴을 224×224로 저장한다. LoRA용은 얼굴 주변과 머리 윤곽이 남도록 여유 있는 512×512 크롭을 만든다. 두 크롭은 원본 이미지 ID로 연결 가능하게 한다.
5. 검출 실패·랜드마크 실패·이미지 읽기 실패를 원본 데이터별로 집계한다. 임의의 기본 얼굴이나 임의 나이로 대체하지 않는다.

전처리 코드는 `age/prep.py`로 시작하되, 검출·정렬 함수를 생성 추론에서도 호출할 수 있게 분리한다. **학습과 실제 촬영 사진의 얼굴 정렬 방식이 같아야 한다.** B팀이 C팀에 함수의 입력 RGB/BGR 순서, 출력 크기, 얼굴 선택 규칙, 실패 반환값을 문서로 전달한다.

### 3.2 나이 추정 데이터 생성과 분할

1. 원본 라벨을 읽어 `age`, `gender`, `source`, 가능하면 `person_id`로 정규화한다. 라벨 누락·형식 오류·모델 범위 밖 나이(0~100)는 제외하고 수를 보고한다. 범위 밖 나이를 조용히 100으로 바꾸지 않는다.
2. **분할을 먼저 확정하고 그다음 증강**한다. 같은 이미지의 좌우 반전·크롭이 train과 test에 나뉘지 않도록 한다.
3. 기본 비율은 데이터셋별 train/val/test = 8:1:1, 고정 seed를 사용한다. 인물 ID가 제공되면 ID 단위로 묶어 분할한다. AI Hub는 반드시 인물 ID 기준으로 분할한다. ID가 없는 데이터의 동일인 중복 위험은 평가 제한으로 기록한다.
4. 원본별·나이대별 개수와 제외 비율을 확인한다. AFAD의 특정 연령 편중은 **train에서만** 나이대별 `WeightedRandomSampler`로 보완한다. val/test는 원래 분포를 유지한다.
5. train·val·test 사이에 파일 경로나 인물 ID가 겹치지 않는지 검사하고 `split_summary.csv`에 결과를 남긴다.

### 3.3 Age LoRA 데이터 생성

1. FFHQ-Aging 라벨의 실제 컬럼을 확인한다. 설계의 기준인 `age_group_confidence > 0.8`, `|yaw| < 30°`를 적용하되 컬럼 이름이나 단위가 다르면 변환 규칙을 기록한다.
2. 설계의 10개 연령 그룹 × 2개 성별 셀당 약 500장을 목표로 샘플링한다. 부족한 셀은 개수를 그대로 보고하고, 같은 파일을 복제해 숫자만 맞추지 않는다.
3. **평가 얼굴을 인물 단위로 먼저 분리**하고 그 인물의 모든 이미지를 LoRA 학습에서 제외한다. 별도 val 이미지와 팀원 동의 사진도 train에 섞지 않는다.
4. 캡션은 기준 문서 1.3절의 단일 `연령 그룹 → 문구` 매핑으로 만든다. 학습과 생성에 같은 매핑을 사용한다. FFHQ 캡션에 근거 없는 인종 표현을 넣지 않는다.
5. `metadata.jsonl`의 각 `file_name`이 존재하고 `text`가 비어 있지 않은지 검사한다. 연령 그룹·성별별 실제 장수와 제외 수를 저장한다.

## 4. 나이 추정 모델: B팀

### 4.1 모델과 학습

- 백본: ImageNet 사전학습 `torchvision mobilenet_v3_large`.
- 출력: 마지막 classifier를 101개 logit(0~100세)으로 교체한다. 학습 전에 실제 라이브러리 버전의 classifier 구조와 교체 위치를 출력해 확인한다.
- 타깃: 정답 나이를 중심으로 표준편차 2세인 이산 가우시안 분포를 만들고 0~100 범위에서 정규화한다.
- 예측 나이: 101개 확률에 나이 인덱스를 곱한 기댓값 `sum(p_i × i)`.
- 손실: 분포 손실 + 예측 나이의 L1 손실. `torch.nn.functional.kl_div(log_softmax(logits), target)`를 쓰면 계산되는 방향은 `KL(target || pred)`이다. 기준 문서의 `KL(pred || target)` 표기와 다르므로 구현에서 **한 방향을 명시하고 실험 기록에도 적는다**. 기본 구현은 전자의 안정적인 API 형태로 고정한다. 두 손실의 계수도 기록한다.
- Stage 1: UTKFace+AFAD, 224px, batch 128을 시작값으로 AdamW `lr=5e-4`, cosine scheduler, 최대 30 epoch, AMP. GPU 메모리에 맞춰 batch를 바꾸면 기록한다.
- 증강은 **train에만** 좌우 반전·RandomResizedCrop(0.8~1.0)·ColorJitter를 적용한다. val/test에는 결정적인 224px 입력 변환만 적용한다.
- 매 epoch 데이터셋별 val MAE를 저장하고 최저 val MAE 체크포인트를 선택한다. test는 체크포인트와 하이퍼파라미터가 정해진 뒤 한 번 평가한다.
- Stage 2는 AI Hub 승인·전처리가 끝난 뒤 Stage 1 체크포인트에서 시작한다. 시작값은 `lr=1e-4`, 최대 10 epoch. 공개 데이터 test 결과와 AI Hub test 결과를 구분해 보고한다.

### 4.2 평가와 ONNX 전달

1. UTKFace·AFAD·승인 시 AI Hub 각각의 test MAE, CS@5(오차 5세 이내 비율), 나이대별 MAE와 샘플 수를 기록한다. `insightface genderage`의 나이 추정을 같은 test 목록에서 비교한다.
2. 실패 사례를 어린이·고령·측면 얼굴 등으로 모아 확인한다. 특정 나이대의 표본이 적으면 평균만으로 성능을 주장하지 않는다.
3. softmax와 기댓값까지 포함한 ONNX를 opset 17로 export해 출력이 이미지당 나이 스칼라 하나가 되게 한다.
4. 같은 정렬 이미지에서 PyTorch와 ONNX Runtime 출력 차이가 0.01세 미만인지 검사한다. 생성 파이프라인에 전달할 `age_estimator.onnx`, 입력 정규화 규칙, 학습 데이터 버전, 평가표를 함께 제공한다.
5. 실시간 예측은 원본과 좌우 반전 예측을 평균한다. 목표 나이 4개는 `[a-10, a, a+10, a+20]`를 `[3,90]`에 맞춰 제한한다. 현재 컷은 원본 사진이고 생성 대상은 나머지 3개다. 같은 값으로 겹치는 목표가 생기면 UI 표기 규칙을 D팀과 정한다.

## 5. 연령 변환 모델: C팀

### 5.1 모델 구성과 최초 실행

1. 기준 문서의 SD 1.5 계열 Realistic Vision V5.1을 학습·추론에서 **동일한 모델 버전**으로 사용한다. 사용 가능한 파일 형식이 Diffusers 학습 예제의 `from_pretrained` 입력과 맞는지 먼저 확인하고, 필요하면 호환 형식으로 준비한다. 모델 파일 출처·버전·사용 조건을 기록한다.
2. Age LoRA 학습 전, 베이스 img2img와 FaceID Plus v2만으로 사진 1장·목표 나이 1개를 생성해 이미지 입출력, GPU 메모리, seed 재현성을 확인한다. FaceID용 가중치와 함께 제공되는 LoRA의 짝이 맞는지도 확인한다.
3. Age LoRA는 Diffusers의 `examples/text_to_image/train_text_to_image_lora.py`를 사용한다. 시작 설정은 rank 16, 512px, `lr=1e-4`, batch 8, fp16, 약 8,000 step, 1,000 step마다 체크포인트 저장이다. GPU 부족 시 batch/gradient accumulation을 조정하고 유효 batch를 기록한다.
4. 학습 스크립트가 `data/lora/train/metadata.jsonl`의 `file_name`·`text`를 실제로 읽는지 **소량 데이터로 10~20 step 시험**한 뒤 전체 학습을 시작한다. 배포 버전에 따라 CLI 인자 이름이 달라질 수 있으므로 실행에 사용한 Diffusers 버전과 명령을 저장한다.
5. 10개 연령 그룹의 고정 검증 프롬프트·고정 seed로 각 체크포인트를 비교한다. 최종 체크포인트는 훈련 손실만이 아니라 목표 나이 표현과 인물 유사도 평가를 보고 고른다.

### 5.2 추론 계약

- `gen/pipeline.py`는 모델을 요청마다 재로드하지 않고 한 번 로드한다.
- 입력은 RGB 얼굴 사진과 정수 목표 나이 목록. 얼굴 미검출은 명시적인 실패로 반환한다.
- 입력 얼굴에서 FaceID 임베딩과 필요한 이미지 임베딩을 얻고, 원본 구도 유지를 위한 여유 있는 512px img2img 크롭을 사용한다.
- 생성에는 같은 세션 seed를 사용하고, strength 약 0.6·25 step부터 시작한다. FaceID LoRA와 Age LoRA 가중치는 별도로 제어한다.
- 출력은 목표 나이별 RGB 이미지 3장과 각 목표 나이이다. 4컷 합성은 D팀의 책임으로 분리한다.
- ControlNet depth는 기본 파이프라인에 넣지 않는다. 평가에서 구도·포즈 붕괴가 반복 확인될 때만 추가한다.

### 5.3 생성 품질 평가

- LoRA train에 포함되지 않은 인물 약 50명을 평가 목록으로 고정한다. AI Hub는 접근 허용 범위 안에서만 사용하고, 팀원 사진은 동의받은 것만 사용한다.
- 같은 입력·목표·seed로 (a) 프롬프트만, (b) +FaceID, (c) +FaceID+Age LoRA를 비교한다.
- 출력 나이와 목표 나이의 MAE, 입력·출력 ArcFace 임베딩 코사인 유사도, 얼굴 검출 실패율, 3장 생성 시간을 함께 기록한다. B팀 모델로 측정한 나이만 단독 근거로 쓰지 말고 `insightface genderage` 결과와 사람의 육안 검토를 병행한다.
- strength `{0.5,0.6,0.7}` × IP scale `{0.6,0.8,1.0}` × Age LoRA 가중치 `{0.6,0.8,1.0}` 그리드의 결과를 CSV로 남긴다. 조건이 많으므로 작은 고정 검증 세트에서 후보를 좁힌 뒤 전체 평가 세트로 확인한다.
- 목표인 **24GB GPU에서 생성 3장 합계 약 10초**를 실측한다. 모델 로드 시간과 요청별 추론 시간을 구분해 보고한다. 달성하지 못해도 측정값과 설정을 기록한다.

## 6. 팀 간 전달과 완료 기준

| 전달 | 내용 | 받을 때 확인할 것 |
|---|---|---|
| B → C | 공통 검출·정렬 함수, LoRA train/val 데이터, 캡션 매핑 | 이미지 경로·RGB/BGR·크기·실패 처리·train/val 인물 중복 없음 |
| B → D | ONNX, 입력 정규화, 목표 나이 산출 규칙 | ONNX 추론값 검증, 얼굴 미검출 처리 |
| C → D | `age_shift(img, targets)` 또는 동등한 함수, 생성 이미지 3장, 목표 나이 | 반환 순서·이미지 형식·오류·GPU 자원 사용량 |
| D → A | 원본+생성 3장을 합성한 PNG와 API 계약 | 출력 크기·방향·프린터 입력 규격 |

중간발표까지 최소 완료 기준:

- 데이터셋별 원본·사용·제외 개수와 제외 사유, 나이 분포가 재현 가능하다.
- 나이 추정 모델의 데이터셋별 MAE/CS@5와 ONNX 동작 검증 결과가 있다.
- 동일 인물의 원본+연령 변환 3장 샘플, ablation 표, 나이 변화·인물 유사도 평가가 있다.
- 4컷 구성과 요청별 처리 시간, 실패 사례가 정리되어 있다.
- 실제 사용한 데이터·모델의 출처와 사용 범위가 문서화되어 있다.

## 7. 구현할 때 참고할 공식 자료

- [Hugging Face Diffusers: LoRA 학습](https://huggingface.co/docs/diffusers/training/lora), [공식 학습 스크립트](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py)
- [Hugging Face Datasets: 로컬 이미지와 metadata 파일](https://huggingface.co/docs/datasets/create_dataset)
- [IP-Adapter FaceID 공식 모델 카드](https://huggingface.co/h94/IP-Adapter-FaceID/blob/main/README.md)
- [InsightFace 공식 모델 목록·사용 조건](https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md)
