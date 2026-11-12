# syntax=docker/dockerfile:1
# Linux/amd64 only. Both the registry index and its amd64 manifest are recorded
# in build_helpers/sources.json; the tag is descriptive, the digest is decisive.
FROM vastai/base-image:cuda-12.8.1-cudnn-devel-ubuntu22.04-py311@sha256:cf12789c494089adac60f8a9a7042cea80eb96b13aff66d0f49bcf7c732b18ce

USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

LABEL org.opencontainers.image.title="HAI Team 3 Vast.ai development environment" \
      org.opencontainers.image.description="Pinned SD1.5/SAM, API, and FLUX.3 development environments; external models and data" \
      org.opencontainers.image.source="https://github.com/HanyangTechAI/2026-2-HAI-ProjectTeam3"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_PYTHON_DOWNLOADS=never \
    HAI_CONFIG=/opt/hai/config/branches.json \
    HAI_WORKSPACE=/workspace/hai \
    HAI_BRANCH=main \
    PATH=/venv/main/bin:/usr/local/bin:/opt/instance-tools/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin

RUN test "$(uname -m)" = x86_64 \
    && /venv/main/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 11)' \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        build-essential ca-certificates cmake curl fontconfig fonts-nanum \
        gh git git-lfs htop jq less libgl1 libglib2.0-0 libgomp1 \
        micro openssh-client openssh-server pkg-config procps ripgrep \
        rsync tmux unzip wget zip zsh \
    && test -f /usr/share/fonts/truetype/nanum/NanumSquareRoundB.ttf \
    && test -f /usr/share/fonts/truetype/nanum/NanumGothic.ttf \
    && rm -f /etc/ssh/ssh_host_* \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY docker/build_helpers/ /opt/hai/build_helpers/
RUN /usr/bin/python3 /opt/hai/build_helpers/bootstrap_tools.py \
    && /usr/local/bin/uv --version \
    && /usr/local/bin/kilo --version

COPY docker/requirements/ /opt/hai/requirements/

# The base's /venv/main is a Conda Python 3.11 prefix. Keep that interpreter and
# its Vast activation contract; never install into /usr/bin/python3. Additional
# venvs are isolated. Identical wheel files are hardlinked in this ONE layer;
# deleting the download cache afterward does not delete their installed links.
RUN export UV_CACHE_DIR=/tmp/hai-uv-cache \
    && /usr/local/bin/uv venv --python /venv/main/bin/python /venv/web \
    && /usr/local/bin/uv venv --python /venv/main/bin/python /venv/m1 \
    && for environment in main web m1; do \
         /usr/local/bin/uv pip sync \
           --python "/venv/${environment}/bin/python" \
           --require-hashes --only-binary :all: --torch-backend cu128 \
           --link-mode hardlink --strict \
           "/opt/hai/requirements/${environment}.lock" \
         && /usr/local/bin/uv pip check --python "/venv/${environment}/bin/python" \
         || exit 1; \
       done \
    && /usr/local/bin/uv cache clean \
    && rm -rf /tmp/hai-uv-cache

COPY docker/branches.json /opt/hai/config/branches.json
RUN /venv/main/bin/python /opt/hai/build_helpers/seed_sources.py

COPY docker/runtime/hai.py /usr/local/bin/hai
COPY docker/runtime/80-hai-init.sh /etc/vast_boot.d/80-hai-init.sh
COPY docker/runtime/hai-sshd.sh /usr/local/bin/hai-sshd
COPY docker/runtime/authorize_ssh.py /usr/local/lib/hai/authorize_ssh.py
COPY docker/runtime/sshd.conf /etc/supervisor/conf.d/hai-sshd.conf
COPY docker/runtime/00-hai-sshd.conf /etc/ssh/sshd_config.d/00-hai.conf
COPY docker/assets/ /opt/hai/assets/
COPY docker/assets/models.json /opt/hai/config/models.json
COPY docker/build_helpers/hai-models.sh /usr/local/bin/hai-models
COPY docker/verify_image.py /opt/hai/verify_image.py

RUN chmod 0755 /usr/local/bin/hai /usr/local/bin/hai-sshd /usr/local/bin/hai-models /etc/vast_boot.d/80-hai-init.sh \
    && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       /venv/main/bin/python /opt/hai/verify_image.py

WORKDIR /workspace
EXPOSE 22

# Inherit Vast's /opt/instance-tools/bin/entrypoint.sh. The image starts its
# development services; model downloads, GPU training, and API serving are
# explicit hai commands, never build steps or automatic startup jobs.
