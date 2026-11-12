"""Fetch only pinned public source code. No weights, datasets, or credentials."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import urllib.request


SOURCES = json.loads(Path(__file__).with_name("sources.json").read_text())


def git(directory, *arguments):
    subprocess.run(["git", "-C", str(directory), *arguments], check=True)


def verified_commit(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError(f"Expected a full Git commit SHA: {value!r}")
    return value


def init_repository(directory, url):
    if directory.exists():
        raise FileExistsError(f"Refusing to replace a source checkout: {directory}")
    directory.mkdir(parents=True)
    git(directory, "init", "--quiet")
    git(directory, "remote", "add", "origin", url)


def fetch_commit(directory, commit):
    commit = verified_commit(commit)
    git(directory, "fetch", "--no-tags", "--depth", "1", "origin", commit)
    actual = subprocess.check_output(
        ["git", "-C", str(directory), "rev-parse", "FETCH_HEAD"], text=True).strip()
    if actual != commit:
        raise RuntimeError(f"Git returned {actual}, expected {commit}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("/opt/hai/config/branches.json"))
    parser.add_argument("--tools", type=Path, default=Path("/opt/hai/tools"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    seed = Path(config["source_seed"])
    init_repository(seed, config["repository_url"])
    for branch, item in config["branches"].items():
        git(seed, "check-ref-format", f"refs/heads/{branch}")
        commit = verified_commit(item["commit"])
        fetch_commit(seed, commit)
        git(seed, "update-ref", f"refs/heads/{branch}", commit)
        git(seed, "update-ref", f"refs/remotes/origin/{branch}", commit)
    git(seed, "checkout", "--quiet", config["default_branch"])
    git(seed, "symbolic-ref", "refs/remotes/origin/HEAD", f"refs/remotes/origin/{config['default_branch']}")

    sam = SOURCES["sam"]
    destination = Path(sam["destination"])
    init_repository(destination, sam["repository_url"])
    fetch_commit(destination, sam["commit"])
    git(destination, "checkout", "--quiet", "--detach", sam["commit"])

    args.tools.mkdir(parents=True, exist_ok=True)
    for script in SOURCES["scripts"]:
        with urllib.request.urlopen(script["url"], timeout=60) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != script["sha256"]:
            raise RuntimeError(f"Official script hash mismatch: {script['name']}")
        (args.tools / script["name"]).write_bytes(content)
    (args.tools / "sources.json").write_text(json.dumps(SOURCES, indent=2) + "\n")
    print("Pinned project branches, SAM source, and official training scripts are ready.")


if __name__ == "__main__":
    main()
