#!/usr/bin/env bash
set -euo pipefail
exec /venv/main/bin/python /opt/hai/assets/prepare_models.py "$@"
