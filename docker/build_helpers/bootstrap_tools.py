"""Install hash-checked standalone tools; never modify system Python packages."""

import base64
import hashlib
import hmac
import http.client
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import zipfile


SOURCES = json.loads(Path(__file__).with_name("sources.json").read_text())


def download(record, destination):
    algorithm = "sha256" if "sha256" in record else "sha512"
    expected = (bytes.fromhex(record[algorithm]) if algorithm in record else
                base64.b64decode(record["integrity"].removeprefix("sha512-"), validate=True))
    for attempt in range(3):
        digest = hashlib.new(algorithm)
        try:
            with urllib.request.urlopen(record["url"], timeout=120) as response, destination.open("wb") as out:
                while chunk := response.read(1024 * 1024):
                    digest.update(chunk)
                    out.write(chunk)
            break
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise
        except (urllib.error.URLError, TimeoutError, http.client.IncompleteRead):
            if attempt == 2:
                raise
        time.sleep(2 ** attempt)
    if not hmac.compare_digest(digest.digest(), expected):
        destination.unlink()
        raise RuntimeError(f"Downloaded artifact hash mismatch: {record['url']}")


def install_uv(archive, destination):
    with zipfile.ZipFile(archive) as wheel:
        for name in ("uv", "uvx"):
            member = f"uv-{SOURCES['uv']['version']}.data/scripts/{name}"
            path = destination / name
            with wheel.open(member) as src, path.open("wb") as out:
                shutil.copyfileobj(src, out)
            path.chmod(0o755)


def install_kilo(archive, destination, bindir=Path("/usr/local/bin")):
    # The platform package contains the standalone executable and its assets.
    # Only ordinary files/directories are accepted; no archive path may escape.
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            parts = PurePosixPath(member.name).parts
            if not parts or parts[0] != "package" or ".." in parts or member.name.startswith("/"):
                raise RuntimeError(f"Unsafe platform archive entry: {member.name}")
            relative = Path(*parts[1:])
            target = destination / relative
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as src, target.open("wb") as out:
                    shutil.copyfileobj(src, out)
                target.chmod(0o755 if member.mode & 0o111 else 0o644)
            else:
                raise RuntimeError(f"Unsupported platform archive entry: {member.name}")
    binary = destination / "bin/kilo"
    if not binary.is_file():
        raise RuntimeError("Kilo platform package has no bin/kilo executable")
    binary.chmod(0o755)
    for name in ("kilo", "kilocode"):
        link = bindir / name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(binary)


def main():
    with tempfile.TemporaryDirectory(prefix="hai-tools-") as temp:
        directory = Path(temp)
        uv_archive, kilo_archive = directory / "uv.whl", directory / "kilo.tar.gz"
        download(SOURCES["uv"], uv_archive)
        install_uv(uv_archive, Path("/usr/local/bin"))
        download(SOURCES["kilo"], kilo_archive)
        install_kilo(kilo_archive, Path("/opt/hai/kilo"))
    print(f"Installed uv {SOURCES['uv']['version']} and Kilo {SOURCES['kilo']['version']}")


if __name__ == "__main__":
    main()
