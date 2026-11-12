#!/usr/bin/env bash
# This file may be sourced by the Vast boot process. Do not change shell
# options, call exit, activate a venv, or start a training/API process here.
if command -v hai >/dev/null 2>&1; then
    if ! hai setup --branch "${HAI_BRANCH:-main}"; then
        printf '%s\n' 'HAI workspace setup failed; existing files were preserved. Run hai setup manually to inspect the error.' >&2
    fi
else
    printf '%s\n' 'HAI runtime is missing from PATH; SSH startup will continue.' >&2
fi
# In particular, remain safe when sourced from a parent running with set -e.
true
