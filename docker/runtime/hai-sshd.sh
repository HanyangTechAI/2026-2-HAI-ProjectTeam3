#!/usr/bin/env bash
# Runs as a Supervisor service, not as a sourced Vast startup hook.
set -euo pipefail

# Generate only missing host keys on this instance. Image layers contain none.
ssh-keygen -A
mkdir -p /run/sshd
chmod 0755 /run/sshd

# Vast's earlier boot hooks may already have populated authorized_keys. This
# helper only appends a genuinely new, explicitly injected public key.
python3 /usr/local/lib/hai/authorize_ssh.py

# Command-line values also protect against a conflicting base-image drop-in.
exec /usr/sbin/sshd -D -e -p 22 \
    -o PubkeyAuthentication=yes \
    -o PasswordAuthentication=no \
    -o KbdInteractiveAuthentication=no \
    -o AuthenticationMethods=publickey \
    -o PermitRootLogin=prohibit-password
