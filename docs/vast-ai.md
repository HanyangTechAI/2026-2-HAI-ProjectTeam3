# Vast.ai 개발 이미지: 빌드부터 접속까지

이 이미지는 **나이 추정·생성·LoRA 학습, 기존 웹 연동 브랜치, M1 실험을 위한 NVIDIA GPU 개발 환경**입니다. GitHub Actions가 이미지를 빌드해 GHCR에 올리고, Vast.ai는 그 이미지를 내려받아 실행합니다. 부팅하면 SSH 개발 환경과 선택한 프로젝트 작업 폴더를 준비합니다. 데이터·모델 가중치는 별도로 옮기며, 포토부스 API나 학습을 자동으로 시작하지 않습니다.

이미지 이름은 `ghcr.io/hanyangtechai/hai-photobooth`입니다. **Actions의 해당 실행이 성공해야 사용할 이미지가 게시된 것입니다.** 이 문서나 Dockerfile이 GitHub에 있다는 사실만으로 이미지 게시가 완료된 것은 아닙니다.

### 첫 게시 확인: 2026-11-12

[첫 GitHub Actions 실행](https://github.com/HanyangTechAI/2026-2-HAI-ProjectTeam3/actions/runs/43229954790)이 성공했고 다음 이미지를 GHCR에 게시했습니다. Vast 템플릿에 복사할 고정 버전 주소입니다.

```text
ghcr.io/hanyangtechai/hai-photobooth:sha-beaec52a80c781ab90f1f93bbbea3f0ce51d4134-43229954790-1
```

이미지 내용의 정확한 digest:

```text
ghcr.io/hanyangtechai/hai-photobooth@sha256:e9b64d074a0a72bb9acb1992f64f711da4f4554be1b8185d509d7a7f24ba72ed
```

작업 폴더·자산 보존 테스트 24개가 통과했고, 이미지 내부의 기존 main 테스트 236개 중 232개가 통과하고 4개가 건너뛰어졌습니다. 세 환경의 import, 네 브랜치의 CLI, 생성 Mock, FastAPI Mock·QR·사진 다운로드, Pi UI·직렬 출력 변환의 하드웨어 Mock, 공식 FLUX.3 학습기 CLI도 통과했습니다. 실제 GPU·모델 가중치·Pi 하드웨어 실행은 검증하지 않았습니다. 최초 사용 전 아래 GHCR 공개 설정을 확인합니다.

## 1. 이미지에 포함되는 환경

Ubuntu 22.04·CUDA 12.8.1·Python 3.11을 기반으로 `linux/amd64`용 이미지를 만듭니다. 기본 이미지의 정확한 digest는 [Dockerfile](../docker/Dockerfile)에, Python 의존성은 [docker 디렉터리](../docker/)의 고정 파일에 기록합니다. NVIDIA 드라이버는 Vast 호스트에서 제공하며 이미지에 설치하지 않습니다.

| 환경 | 용도 | 선택 방법 |
| --- | --- | --- |
| `main`, `/venv/main` | 현재 나이 추정·Realistic Vision/FaceID 생성·Age LoRA 학습·SAM 교사 도구 | `main`, `feat/generation-midterm`에서 자동 선택 |
| `web`, `/venv/web` | 기존 FastAPI·라즈베리파이 클라이언트 브랜치의 의존성 | `feat/raspi-fastapi-pipeline`, `feat/raspi-client`에서 자동 선택 |
| `m1`, `/venv/m1` | FLUX.3 기반 M1 연구·개발용 최신 모델 라이브러리 | 실행 시 `--env m1`로 선택 |

M1은 새로운 프로젝트 브랜치 이름이 아니라 별도 Python 환경입니다. SAM 소스와 LoRA 학습기 소스는 고정해 넣지만 학습 가중치와 데이터는 별도로 필요합니다. 이 이미지는 새로운 M1 학습 파이프라인이나 나이 변환 모델의 품질을 완성하는 작업까지 포함하지 않습니다. SSH, tmux, zsh, git, GitHub CLI, uv, micro, Kilo CLI도 함께 준비합니다.

브랜치별 코드는 아래 커밋을 이미지 안에 고정해 둡니다. 서버를 켤 때 GitHub의 최신 커밋을 자동으로 받아 덮어쓰지 않습니다. 기준을 바꾸려면 [branches.json](../docker/branches.json)을 수정하고 이미지를 다시 빌드합니다.

| 프로젝트 브랜치 | 고정 커밋 | 기본 환경 |
| --- | --- | --- |
| `main` | `2457c3c8fa27b7598060272e75cefd79599ffc3d` | `main` |
| `feat/generation-midterm` | `229a223d5f416465b4e92d471ef8f2e219498fa9` | `main` |
| `feat/raspi-fastapi-pipeline` | `c4b337a26afbdcc497a0229d5d9eeb69d525f188` | `web` |
| `feat/raspi-client` | `38630791ad9926e17650fc6dd8da5b33973c974c` | `web` |

라즈베리파이용 브랜치를 읽고 수정할 수 있지만, 이 x86 NVIDIA 이미지를 Pi에 설치하거나 원격 GPU 서버에서 Pi의 USB 카메라·프린터를 직접 구동하는 구성은 아닙니다.

## 2. GitHub에서 이미지 빌드·업로드

빌드 파일은 [build-vast-image.yml](../.github/workflows/build-vast-image.yml)입니다. `feat/vast-dev-image` 또는 `main`에 Docker 관련 변경을 push하면 **Build HAI Vast image**가 실행됩니다. 이미지 제작 코드와 테스트만 바뀐 경우에 실행되도록 경로를 제한했습니다. 문서만 바꾼 commit은 자동 빌드를 시작하지 않습니다.

1. [저장소 Actions](https://github.com/HanyangTechAI/2026-2-HAI-ProjectTeam3/actions)에서 **Build HAI Vast image**의 해당 커밋 실행을 엽니다.
2. `Check workspace and asset helpers`와 `Build, verify on CPU, and publish`가 성공했는지 확인합니다.
3. 실행의 Summary에서 **Version tag**와 **Fixed digest**를 복사합니다. 이 값이 Vast에 넣을 실제 이미지 주소입니다.

첫 실행은 기능 브랜치의 push로 시작합니다. 워크플로 파일이 기본 브랜치 `main`에 들어가기 전에는 **Run workflow** 버튼이 없을 수 있습니다. 기본 브랜치에 반영한 이후에는 Actions → Build HAI Vast image → Run workflow에서 `main` 또는 `feat/vast-dev-image`를 골라 수동으로 다시 빌드할 수 있습니다. [GitHub의 수동 실행 조건](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow)을 따릅니다.

| 게시되는 참조 | 의미 |
| --- | --- |
| `:sha-<전체 커밋 SHA>-<실행 ID>-<시도 번호>` | 빌드마다 새로 생기는 버전 태그. 재실행해도 이전 태그를 재사용하지 않음 |
| `:dev` | `feat/vast-dev-image`에서 마지막으로 성공한 빌드 |
| `:main` | `main`에서 마지막으로 성공한 빌드. main에 반영하고 빌드하기 전에는 없음 |
| `@sha256:...` | 해당 이미지 내용에 고정된 digest. 같은 환경을 재현할 때 기록할 값 |

Vast 템플릿에는 성공한 **버전 태그**를 넣는 것을 권장합니다. `dev`는 편하지만 다음 빌드 때 가리키는 이미지가 바뀝니다. 워크플로는 `latest` 태그를 만들지 않습니다.

GHCR 로그인에는 저장소의 `GITHUB_TOKEN`을 사용하므로 별도 Docker Hub 계정이나 개인 토큰을 Actions secret에 추가할 필요가 없습니다. 권한은 소스 읽기와 패키지 쓰기로 제한합니다. 공식 Actions는 커밋 SHA로 고정했고, 빌드 내부의 CPU 검증에 실패하면 이미지를 게시하지 않습니다. 일반 GitHub runner에는 GPU가 없으므로 실제 GPU 동작은 6절에서 확인합니다.

빌드 캐시는 최종 이미지에 inline으로 저장합니다. 별도의 대용량 GitHub Actions 캐시를 만들거나 이미지를 runner의 Docker 엔진에 다시 로드하지 않습니다. CUDA와 세 Python 환경을 포함하므로 최종 용량·빌드 시간은 실제 빌드 결과로 확인합니다.

### 처음 게시한 패키지의 공개 설정

GitHub 저장소가 공개여도 **처음 올린 GHCR 패키지는 기본적으로 비공개**입니다. Vast에서 인증정보 없이 받으려면 게시 후 패키지 공개 상태를 확인해야 합니다.

1. [HanyangTechAI의 Packages](https://github.com/orgs/HanyangTechAI/packages)에서 `hai-photobooth`를 엽니다.
2. **Package settings → Danger Zone → Change visibility → Public**으로 설정합니다.
3. 패키지가 저장소 `HanyangTechAI/2026-2-HAI-ProjectTeam3`에 연결되어 있는지 확인합니다. 기존 동명 패키지 때문에 Actions가 push 권한 오류를 내면 **Manage Actions access**에서 이 저장소에 쓰기 접근을 부여해야 합니다.

공개 전환은 GitHub에서 다시 비공개로 되돌릴 수 없으므로 이 패키지에는 공개 소스와 환경 파일만 포함합니다. 사진·데이터·모델 가중치·사용자 토큰은 빌드에 넣지 않습니다. 조직 정책 때문에 공개 설정이 막힌 경우 조직 관리자가 패키지 설정을 확인해야 합니다. 근거: [GHCR 게시·접근](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry), [패키지 공개 범위 설정](https://docs.github.com/en/packages/learn-github-packages/configuring-a-packages-access-control-and-visibility).

## 3. Vast.ai 템플릿 만들기

먼저 Vast의 **Keys → SSH Keys**에 로컬 PC의 SSH 공개키를 등록합니다. 기존에 Vast에서 쓰던 키가 있으면 같은 키를 사용할 수 있습니다. 개인키 파일은 PC에 둡니다.

[Vast Templates](https://cloud.vast.ai/templates)에서 **+ New**를 누르거나 기존 NVIDIA CUDA 템플릿의 연필 아이콘을 누르고, **Config** 탭에 아래 값을 넣습니다.

| 화면 항목 | 입력값 |
| --- | --- |
| Template Name | `HAI PhotoBooth Dev` |
| Image Path:Tag | Actions Summary에서 복사한 `ghcr.io/hanyangtechai/hai-photobooth:sha-...` 전체 주소 |
| Version Tag | 위에서 선택한 실제 태그. `[Automatic]`을 선택하지 않음 |
| Launch Mode | **docker ENTRYPOINT** |
| Entrypoint arguments | 비워 둠 |
| Ports | **22 / TCP**. 기본 개발 접속에는 8000을 열 필요 없음 |
| Environment Variables | `HAI_BRANCH=main`, `HAI_WORKSPACE=/workspace/hai` |
| On-start Script | 사용하지 않음. Entrypoint 모드에서는 별도 입력란이 없음 |

`HAI_BRANCH`는 부팅할 때 먼저 준비할 프로젝트 브랜치입니다. 웹 브랜치부터 작업하려면 `feat/raspi-fastapi-pipeline`을 넣습니다. 이후에는 `hai` 명령으로 다른 브랜치를 준비하고 환경을 선택할 수 있습니다.

**이 이미지의 초기화와 SSH 서비스를 실행하려면 docker ENTRYPOINT 모드를 유지합니다.** Vast의 SSH/Jupyter launch mode는 원래 ENTRYPOINT를 교체하므로 위 설정과 서로 바꿔 쓰지 않습니다. 기존 템플릿에서 복사했다면 예전 `PROVISIONING_SCRIPT`, `BOOT_SCRIPT`, on-start 설치 명령도 제거해 이 이미지에 다른 초기화를 덧씌우지 않도록 합니다.

저장만 하려면 **Create**, 저장 후 GPU 목록으로 이동하려면 **Create & Use**를 누릅니다. 실제 임대는 목록에서 별도로 **Rent**를 눌러야 시작됩니다. UI 항목의 근거는 [Vast 템플릿 생성](https://docs.vast.ai/guides/templates/creating-templates)과 [템플릿 설정](https://docs.vast.ai/guides/templates/template-settings)입니다.

## 4. GPU와 저장공간 선택

NVIDIA GPU가 있고 **직접 연결할 공개 포트가 있는 `amd64` 호스트**를 고릅니다. 호스트의 Direct Ports 지원을 확인합니다. 이 템플릿의 자체 SSH 서버는 내부 22/TCP에 연결된 공개 포트를 사용하며 Vast의 SSH proxy 제공을 전제로 하지 않습니다. CUDA 12.8 이미지와 PyTorch가 요구하는 드라이버·GPU 아키텍처 조건도 충족해야 합니다. CUDA 주 버전만 같다는 이유로 모든 GPU·드라이버 조합의 동작을 보장하지는 않습니다. [NVIDIA 호환성 문서](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)에는 PTX와 새 기능의 추가 조건도 설명되어 있습니다.

VRAM은 사용할 작업에 맞춰 선택합니다. 기존 SD1.5 생성 성공 기록과 M1 학습 요구량은 서로 다른 기준입니다. 이미지가 설치된다는 사실이 특정 GPU에서 모든 학습·생성이 가능하다는 뜻은 아닙니다. 모델을 준비한 뒤 작은 실행으로 실제 최대 사용량을 측정합니다.

저장할 모델·데이터·체크포인트를 보존하려면 템플릿의 **Disk Space (Container + Volume) → Add recommended volume settings**에서 local volume을 추가하고 **설치 경로를 `/workspace`로 지정**합니다. 모델, 학습 데이터, 여러 실험의 체크포인트와 작업 여유를 합산해 볼륨 크기를 선택합니다. 컨테이너 공간에는 이미지와 설치 환경을 수용할 여유가 별도로 필요합니다.

| 저장 방식 | Stop | 인스턴스 Destroy | 다른 물리 서버에서 사용 |
| --- | --- | --- | --- |
| 기본 컨테이너 저장소 | 유지, 저장 비용 계속 발생 | 삭제 | 먼저 별도 복사 필요 |
| 별도로 만든 local volume | 유지, 저장 비용 계속 발생 | 볼륨 자체를 삭제하지 않으면 유지 | 같은 볼륨을 직접 옮길 수 없고 데이터 복사 필요 |

`/workspace`라는 폴더 이름만으로 영구 저장되는 것은 아닙니다. **볼륨을 실제로 연결했는지** 확인합니다. Vast local volume은 같은 물리 호스트에 묶이고, 컨테이너와 볼륨 크기는 생성 후 늘릴 수 없습니다. 이미 있는 볼륨은 **Storage → 해당 볼륨 → Rent instance using this volume**에서 같은 호스트의 새 인스턴스에 연결합니다. 근거: [Storage Types](https://docs.vast.ai/guides/instances/storage/types), [Volumes](https://docs.vast.ai/guides/instances/storage/volumes).

## 5. SSH 접속과 작업 폴더

이미지 다운로드와 초기화가 끝나면 인스턴스의 **IP Port Info**에서 내부 **22/TCP**에 연결된 공개 IP와 외부 포트를 확인합니다. 공개 포트는 임대할 때 배정되므로 임의로 `22`를 사용하지 않습니다. 로컬 PC에서 아래 값을 실제 정보로 바꿔 실행합니다.

```bash
ssh -p SSH_PORT root@SERVER_IP
```

등록한 공개키와 짝을 이루는 개인키가 기본 키 경로에 없으면 `-i /path/to/private_key`도 지정합니다. **Connect**에 이 직접 연결 명령이 표시되면 그대로 사용할 수 있습니다. Entrypoint 모드에서는 자동 생성되는 접속 명령이나 SSH proxy가 항상 제공되는 것은 아닙니다. 근거: [Vast 접속 방식](https://docs.vast.ai/guides/instances/connect/overview).

서버에 접속한 뒤:

```bash
hai info
hai setup --all
hai doctor --branch main
hai doctor --branch main --gpu
```

`setup --all`은 이미지에 포함된 네 브랜치의 작업 폴더를 준비합니다. Python 패키지를 인터넷에서 다시 설치하는 명령이 아닙니다. 기존 작업 폴더와 사용자가 수정한 파일을 자동으로 초기화하지 않습니다.

| 경로 | 내용 |
| --- | --- |
| `/workspace/hai/branches/main` | main 작업 폴더 |
| `/workspace/hai/branches/feat--generation-midterm` | 생성 브랜치 작업 폴더 |
| `/workspace/hai/branches/feat--raspi-fastapi-pipeline` | API 브랜치 작업 폴더 |
| `/workspace/hai/branches/feat--raspi-client` | Pi 클라이언트 브랜치 작업 폴더 |
| `/workspace/hai/shared/models` | 브랜치가 공유하는 모델 가중치 |
| `/workspace/hai/shared/data` | 브랜치가 공유하는 데이터 |
| `/workspace/hai/outputs/<브랜치 폴더명>` | 브랜치별 체크포인트·로그·생성 결과 |
| `HF_HOME`가 가리키는 경로 | Hugging Face 캐시. 기존 값이 없을 때 hai의 기본 경로는 `/workspace/hai/cache/huggingface` |
| `/workspace/hai/shared/models/torch` | 기본 PyTorch 모델 캐시 |

이미 설정된 `HF_HOME`·`TORCH_HOME`은 우선 적용됩니다. Vast 기본 초기화가 `HF_HOME=/workspace/.hf_home`을 제공한 경우 그 경로를 사용합니다. 위 절차처럼 `/workspace` 전체를 볼륨에 연결하면 이 캐시도 같은 볼륨에 저장됩니다.

작업할 셸을 여는 방법:

```bash
hai shell main
```

웹 브랜치나 M1 환경을 사용할 때는 다음 중 해당 명령을 실행합니다.

```bash
hai shell feat/raspi-fastapi-pipeline
hai shell main --env m1
```

각 명령은 해당 브랜치 폴더와 Python 환경을 선택합니다. 다른 환경으로 이동하려면 셸을 `exit`로 나온 뒤 다시 선택합니다. 네 브랜치의 requirements를 한 환경에 차례로 설치하지 않습니다.

한 명령만 실행하려면 `hai run`을 사용합니다. 예를 들어 모델 없는 4컷 Mock 검사:

```bash
hai run main -- python -m gen.smoke --output outputs/vast_mock_001
```

이 결과는 합성 입력과 Mock 이미지입니다. 같은 출력 폴더로 재실행하지 말고 `_002`처럼 새 이름을 사용합니다.

## 6. 모델·데이터 전송과 실제 GPU 확인

기존 PC/서버의 프로젝트에서 아래 자료를 옮기면 됩니다. `.venv`나 설치된 Python 환경은 복사하지 않습니다.

| 기존 프로젝트의 파일 | 새 서버의 배치 |
| --- | --- |
| `models/` | `/workspace/hai/shared/models/` |
| `data/lora/`, `data/gen_eval/`, `data/reports/` 및 필요한 다른 데이터 | `/workspace/hai/shared/data/` 아래에 같은 구조로 복사 |
| 기존 `outputs/`의 체크포인트·실험 기록 | 해당 브랜치의 `/workspace/hai/outputs/<브랜치 폴더명>/` |
| SAM 등 별도 실험 모델 | `/workspace/hai/shared/models/` 아래에 배치하고 해당 실험 설정에서 경로 지정 |

예를 들어 **로컬 PC의 기존 프로젝트 루트**에서 전송할 때, 아래 두 변수의 IP·포트를 Vast IP Port Info의 실제 값으로 바꿉니다. PC에도 rsync가 설치되어 있어야 합니다.

```bash
HAI_SSH_DEST='root@SERVER_IP'
HAI_SSH_PORT='SSH_PORT'
rsync -avP -e "ssh -p ${HAI_SSH_PORT}" models/ "${HAI_SSH_DEST}:/workspace/hai/shared/models/"
rsync -avP -e "ssh -p ${HAI_SSH_PORT}" data/ "${HAI_SSH_DEST}:/workspace/hai/shared/data/"
rsync -avP -e "ssh -p ${HAI_SSH_PORT}" outputs/ "${HAI_SSH_DEST}:/workspace/hai/outputs/main/"
```

없는 로컬 폴더에 해당하는 명령은 생략합니다. SSH 명령에 별도 `-i` 키 경로가 필요했다면 rsync의 `-e` 안에도 같은 옵션을 넣습니다. rsync의 source 뒤 `/`는 폴더 자체가 아니라 그 내용을 대상에 복사한다는 뜻입니다. [rsync 공식 매뉴얼](https://download.samba.org/pub/rsync/rsync.1)을 따릅니다.

### 기존 가중치가 없을 때 다운로드

고정된 공식 모델 revision은 [모델 목록](../docker/assets/models.json)에 기록합니다. 먼저 내려받을 목록과 위치를 확인한 뒤, 필요한 프로필만 명시적으로 다운로드합니다.

```bash
hai-models plan baseline
hai-models download baseline
```

`baseline`은 Realistic Vision의 Diffusers 디렉터리, FaceID Plus v2와 짝을 이루는 LoRA, CLIP vision encoder와 전처리 설정을 준비합니다. 기존 파일의 내용이 다르면 덮어쓰지 않고 중단합니다. 직접 학습한 Age LoRA와 나이 추정기·SAM 체크포인트, 학습 데이터는 이 명령으로 받을 수 없으므로 기존 작업에서 전송합니다.

얼굴 검출·ArcFace 자산은 별도 준비합니다.

```bash
hai run main -- python -m age.setup_models --with-arcface
```

위 다운로드 명령들은 외부 파일을 내려받습니다. 이미지 빌드와 부팅 중에는 실행되지 않습니다. 단일 SD checkpoint 파일만 `base_model`에 지정할 수는 없습니다. 필요한 파일과 모델 이용 조건은 [서버 실행 안내](server-setup.md#생성-파이프라인과-최소-연동)를 확인합니다.

준비 후 `hai shell main`으로 들어갑니다. 다운로드 도구를 사용했다면 `models/generation/hai-faceid.json`이 생성되어 있습니다. 직접 옮긴 기존 모델을 사용한다면 `gen/config.example.json`을 복사해 경로를 맞춘 설정 파일을 사용합니다. 첫 실제 생성은 Age LoRA 없이 확인할 수 있는 `faceid`부터 실행합니다.

```bash
python -m gen.pipeline --config models/generation/hai-faceid.json --mode faceid --image data/demo/portrait.png --targets 35 --output outputs/vast_faceid_001
```

`portrait.png`는 직접 준비한 입력 사진입니다. 실제 얼굴 검출, 가중치 로드, CUDA 생성, 저장 이미지까지 확인한 뒤 여러 목표 나이와 LoRA 학습으로 확대합니다. 설치 검사와 Mock 통과는 이 실제 GPU 검사를 대신하지 않습니다.

FLUX.3 M1 실험에 필요한 파일은 별도 프로필로 준비합니다.

```bash
hai-models plan m1
hai-models download m1
```

모델 접근 승인이 필요한 경우 Hugging Face에서 이용 조건에 동의한 계정의 인증을 실행 환경에 제공해야 합니다. `HF_TOKEN`은 공개 템플릿이나 이미지에 저장하지 않습니다. 이 프로필은 모델 파일을 준비하며, 특정 VRAM에서의 추론·학습 성공이나 M1 학습 구현 완료를 뜻하지 않습니다.

## 7. 기존 API 브랜치의 Mock 연결 확인

현재 고정한 API 브랜치는 기본적으로 Mock 생성이며, 모델 경로 설정과 정식 나이 추정기 연결을 추가로 정리해야 합니다. ONNX가 없으면 루트 상태 응답에는 미로딩으로 표시되고, 생성 경로에서는 입력값이 없을 때 나이 25를 사용합니다. `/api/v1/predict-age`는 모델 없이 정상 추론할 수 없습니다.

개발 확인용 API는 서버의 localhost에 바인딩합니다.

```bash
hai run feat/raspi-fastapi-pipeline -- env USE_MOCK_GEN=true GEN_DEVICE=cpu python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

로컬 PC의 다른 터미널에서 IP Port Info의 실제 SSH 정보로 터널을 엽니다.

```bash
ssh -p SSH_PORT -N -L 8000:127.0.0.1:8000 root@SERVER_IP
```

이후 PC 브라우저에서 `http://127.0.0.1:8000/`로 상태를 확인하고 `http://127.0.0.1:8000/docs`에서 API를 확인합니다. 기존 API에는 인증 처리가 없으므로 이 기본 절차에서는 외부에 8000 포트를 공개하지 않습니다. 태블릿/부스에서 직접 접근하는 배포는 실제 생성 연결과 접근 경로를 정리한 뒤 별도로 구성합니다. Uvicorn 옵션은 [공식 설정 문서](https://github.com/Kludex/uvicorn/blob/main/docs/settings.md), 터널 옵션은 [OpenSSH 매뉴얼](https://man.openbsd.org/ssh)에 따릅니다.

## 8. 서버 이동과 다음 이미지 버전

작업을 마치면 변경한 코드는 Git에 저장하고, 모델·학습 결과·데이터는 별도로 보존합니다. 기존 서버를 다른 물리 호스트로 바꿀 때는 이미지 주소만 복사하는 것으로 충분하지 않습니다. 위 공유 데이터·모델·출력 폴더를 새 서버로 전송해야 합니다. 인증 토큰을 포함할 수 있는 Hugging Face 홈 전체를 공개 공유하지 않습니다.

기존 작업 폴더는 새 이미지를 사용해도 자동으로 리셋하지 않습니다. 새 이미지가 어떤 프로젝트 커밋을 포함하는지는 `hai info`와 이미지 안의 `/opt/hai/config/branches.json`으로 확인합니다. Python 의존성이나 고정 소스 기준을 바꾼 경우 Docker 관련 파일을 수정해 빌드하고, 새 버전 태그를 템플릿에 적용합니다.

## 9. 문제가 생겼을 때

| 증상 | 확인할 곳 |
| --- | --- |
| Actions가 아예 실행되지 않음 | workflow가 허용한 브랜치·경로의 변경인지, 조직의 Actions 사용/청구 제한이 있는지 확인 |
| Actions의 Run workflow 버튼이 없음 | 워크플로가 아직 기본 브랜치 main에 없을 수 있음. 최초 기능 브랜치 push 실행을 확인 |
| GHCR push가 permission denied | 워크플로의 package 쓰기 권한, 조직 정책, 기존 패키지의 Manage Actions access 확인 |
| Vast의 image pull에서 denied/unauthorized | 패키지 공개 상태 확인. 공개 저장소와 공개 패키지는 별개 |
| `manifest unknown` | 성공한 Actions 실행의 실제 버전 태그인지 확인. 아직 없는 main/latest를 쓰지 않았는지 확인 |
| SSH 접속이 안 됨 | Launch Mode, 등록 공개키, 22/TCP 포트 매핑, 인스턴스 초기화 로그 확인 |
| `hai doctor --gpu` 실패 | GPU 임대/노출 상태와 호스트 드라이버 확인. 이미지 안에 드라이버를 설치하지 않음 |
| 패키지 버전 충돌·import 오류 | `hai info`와 선택한 main/web/m1 환경 확인. 임의로 모든 requirements를 한 환경에 설치하지 않음 |
| 모델 파일이 없다는 오류 | 공유 모델 폴더, 설정 파일의 상대 경로, 전체 모델 디렉터리 구조 확인 |
| API가 응답하지만 실제 변환이 안 됨 | `mock_generation_mode`, 모델·나이 추정기 연결 상태 확인. 현재 Mock 검증과 구분 |

GitHub Actions의 CPU 검증 범위, 최초 실제 GPU 검증 결과, 최종 이미지 digest는 서로 구분해 기록합니다.
