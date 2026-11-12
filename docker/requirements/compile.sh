#!/usr/bin/env bash
# Run from any directory with uv 0.13.6. Build installs the committed lockfiles.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if [[ "$(uv --version)" != uv\ 0.13.6* ]]; then
    echo "Use uv 0.13.6 to regenerate these locks." >&2
    exit 1
fi
for environment in main web m1; do
    uv pip compile "docker/requirements/${environment}.in" \
        --python-version 3.11 \
        --python-platform x86_64-manylinux_2_35 \
        --generate-hashes --only-binary :all: --torch-backend cu128 \
        --emit-index-url --quiet \
        --output-file "docker/requirements/${environment}.lock"
done
