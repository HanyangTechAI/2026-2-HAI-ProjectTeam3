#!/usr/bin/env python3
"""Append an explicitly supplied public key without replacing existing access."""

import base64
import fcntl
import os
from pathlib import Path
import shlex
import stat
import struct
import sys


def key_identity(line):
    """Recognize the key after optional authorized_keys restrictions/comments."""
    try:
        fields = shlex.split(line, comments=True)
    except ValueError:
        return None
    for index in range(len(fields) - 1):
        try:
            data = base64.b64decode(fields[index + 1], validate=True)
            length, = struct.unpack(">I", data[:4])
            if 1 <= length <= 128 and data[4:4 + length].decode("ascii") == fields[index]:
                return fields[index], fields[index + 1]
        except (ValueError, UnicodeError, struct.error):
            continue
    return None


def authorize(directory, payload):
    lines = [line.strip() for line in payload.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if not lines:
        return 0
    identities = [key_identity(line) for line in lines]
    if any(identity is None for identity in identities):
        raise ValueError("SSH_PUBLIC_KEY must contain OpenSSH public-key lines; no keys were added")
    directory = Path(directory)
    path = directory / "authorized_keys"
    if directory.is_symlink() or path.is_symlink():
        # Vast or the user may intentionally share this file. Preserve their
        # setup instead of appending through a symlink to an unrelated file.
        print("hai-sshd: existing SSH directory/key symlink preserved; using its current keys", file=sys.stderr)
        return 0
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory.chmod(0o700)
    flags = os.O_CREAT | os.O_RDWR | os.O_APPEND | os.O_NOFOLLOW | os.O_NONBLOCK
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ValueError("Existing authorized_keys is not a regular file; preserved")
        handle.seek(0)
        existing = handle.read()
        known = {key_identity(line) for line in existing.splitlines()}
        additions = []
        for line, identity in zip(lines, identities):
            if identity not in known:
                additions.append(line)
                known.add(identity)
        if additions:
            prefix = "\n" if existing and not existing.endswith("\n") else ""
            handle.write(prefix + "\n".join(additions) + "\n")
            handle.flush()
        os.fchmod(handle.fileno(), 0o600)
        return len(additions)


def main():
    try:
        authorize(Path("/root/.ssh"), os.environ.get("SSH_PUBLIC_KEY", ""))
    except (OSError, ValueError) as error:
        print(f"hai-sshd: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
