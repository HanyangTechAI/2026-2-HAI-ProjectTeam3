#!/usr/bin/env python3
"""Offline workspace and environment launcher for the HAI development image.

This module intentionally uses only Python's standard library. It never installs
packages, downloads models, updates Git refs, or starts an application on boot.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile


DEFAULT_CONFIG = "/opt/hai/config/branches.json"
DEFAULT_WORKSPACE = "/workspace/hai"
DATA_DIRECTORIES = (
    "raw", "age", "age_aihub", "lora", "gen_eval", "reports", "demo",
    "processed", "interim",
)


class HaiError(RuntimeError):
    """An actionable failure, without a Python traceback in the normal CLI."""


def absolute_path(value: str, label: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise HaiError(f"{label} must be a nonempty absolute path")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or path == Path("/"):
        raise HaiError(f"{label} must be an absolute non-root path without '..': {value}")
    return path


def branch_slug(name: str) -> str:
    # Names come from the image manifest, never from an arbitrary path argument.
    if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", name)
            or any(part in ("", ".", "..") or part.endswith(".lock") for part in name.split("/"))
            or ".." in name or name.endswith(".") or "//" in name):
        raise HaiError(f"Unsafe branch name in manifest: {name!r}")
    return name.replace("/", "--")


class Configuration:
    def __init__(self, value: dict):
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise HaiError("Unsupported branches.json schema; expected schema_version 1")
        self.value = value
        self.repository_url = value.get("repository_url", "")
        if (not isinstance(self.repository_url, str) or not self.repository_url
                or self.repository_url.startswith("-")
                or any(c.isspace() for c in self.repository_url)):
            raise HaiError("repository_url must be a nonempty Git remote URL without whitespace")
        self.source_seed = absolute_path(value.get("source_seed"), "source_seed")
        self.branches = value.get("branches")
        self.environments = value.get("environments")
        self.assets = value.get("assets", {})
        if not isinstance(self.branches, dict) or not self.branches:
            raise HaiError("branches must be a nonempty object")
        if not isinstance(self.environments, dict) or not self.environments:
            raise HaiError("environments must be a nonempty object")
        for name, directory in self.environments.items():
            if not re.fullmatch(r"[a-z][a-z0-9_-]*", name):
                raise HaiError(f"Unsafe environment name: {name!r}")
            absolute_path(directory, f"environment {name}")
        slugs = set()
        for name, branch in self.branches.items():
            slug = branch_slug(name)
            if slug in slugs:
                raise HaiError(f"Branch workspace name collision: {name}")
            slugs.add(slug)
            if not isinstance(branch, dict) or not re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", branch.get("commit", "")):
                raise HaiError(f"Branch {name} must specify a full commit SHA")
            if branch.get("environment") not in self.environments:
                raise HaiError(f"Branch {name} has an unknown environment")
        self.default_branch = value.get("default_branch", "main")
        self.branch(self.default_branch)
        if not isinstance(self.assets, dict):
            raise HaiError("assets must be an object")
        for name in ("sam_repository", "lora_training_script"):
            if name not in self.assets:
                raise HaiError(f"Missing image asset configuration: {name}")
            absolute_path(self.assets[name], name)

    @classmethod
    def load(cls, path: Path):
        try:
            return cls(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError) as error:
            raise HaiError(f"Cannot read configuration {path}: {error}") from error

    def branch(self, name: str) -> dict:
        if name not in self.branches:
            raise HaiError(f"Unknown branch {name!r}; choose: {', '.join(self.branches)}")
        return self.branches[name]

    def environment(self, branch: str, override: str | None = None) -> tuple[str, Path]:
        name = override or self.branch(branch)["environment"]
        if name not in self.environments:
            raise HaiError(f"Unknown environment {name!r}; choose: {', '.join(self.environments)}")
        return name, Path(self.environments[name])


def git(arguments: list[str], cwd: Path | None = None) -> str:
    command = ["git", "-c", "core.hooksPath=/dev/null"]
    if cwd is not None:
        command += ["-C", str(cwd)]
    try:
        result = subprocess.run(command + arguments, text=True, capture_output=True, check=False)
    except OSError as error:
        raise HaiError(f"Cannot run Git: {error}") from error
    if result.returncode:
        message = result.stderr.strip() or result.stdout.strip()
        raise HaiError(f"Git {' '.join(arguments[:2])} failed: {message}")
    return result.stdout.strip()


def exists(path: Path) -> bool:
    """Include broken symlinks; they are user assets too."""
    return os.path.lexists(path)


def managed_directory(path: Path) -> None:
    """Do not put checkouts or runtime state through a pre-existing symlink."""
    if path.is_symlink():
        raise HaiError(f"Refusing to use symlink as a managed directory: {path}")
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise HaiError(f"Cannot create directory {path}: {error}") from error
    if not path.is_dir():
        raise HaiError(f"Expected a directory: {path}")


def link_if_absent(destination: Path, target: Path, warnings: list[str]) -> bool:
    """Create an absent link, retaining both real files and existing symlinks."""
    if exists(destination):
        if destination.is_symlink() and destination.resolve() == target.resolve():
            return True
        warnings.append(f"Preserved existing asset: {destination} (expected link to {target})")
        return False
    try:
        destination.symlink_to(target, target_is_directory=target.is_dir())
    except FileExistsError:
        warnings.append(f"Preserved concurrently created asset: {destination}")
        return False
    except OSError as error:
        raise HaiError(f"Cannot link {destination} to {target}: {error}") from error
    return True


class Runtime:
    def __init__(self, config: Configuration, workspace: Path):
        self.config = config
        # An explicitly selected root may itself be a volume symlink. Descendant
        # checkout and state directories must be ordinary directories.
        self.workspace = absolute_path(str(workspace), "HAI_WORKSPACE").resolve()
        self.warnings: list[str] = []

    def branch_path(self, branch: str) -> Path:
        self.config.branch(branch)
        return self.workspace / "branches" / branch_slug(branch)

    @contextmanager
    def setup_lock(self):
        managed_directory(self.workspace)
        lock = self.workspace / ".setup.lock"
        try:
            fd = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        except OSError as error:
            raise HaiError(f"Cannot lock workspace {self.workspace}: {error}") from error
        with os.fdopen(fd, "a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def prepare_layout(self):
        for relative in ("branches", "shared", "shared/models", "shared/data", "outputs", "cache"):
            managed_directory(self.workspace / relative)
        for name in DATA_DIRECTORIES:
            # Do not modify user-provided shared directories or links. Creating
            # missing empty containers does not download or invent a dataset.
            path = self.workspace / "shared/data" / name
            if not exists(path):
                path.mkdir()
            elif not path.is_dir():
                raise HaiError(f"Shared dataset location is not a directory: {path}")

    def create_checkout(self, branch: str, destination: Path):
        seed = self.config.source_seed
        commit = self.config.branch(branch)["commit"]
        if not seed.is_dir():
            raise HaiError(f"Image source seed is missing: {seed}")
        if git(["cat-file", "-t", commit], seed) != "commit":
            raise HaiError(f"Pinned source object is not a commit: {commit}")
        # Stage locally and rename after checkout. Interrupted initialization
        # never leaves a half-populated directory at the user-facing path.
        with tempfile.TemporaryDirectory(prefix=f".pending-{branch_slug(branch)}-", dir=destination.parent) as temporary:
            staging = Path(temporary)
            git(["clone", "--quiet", "--no-checkout", "--no-hardlinks", "--local", str(seed), str(staging)])
            # Only a NEW, private staging clone is reset to the manifest pin.
            # Existing user checkouts never reach this code path.
            git(["checkout", "--quiet", "-B", branch, commit], staging)
            git(["remote", "set-url", "origin", self.config.repository_url], staging)
            # Keep a normal upstream for a later, explicit user git pull/push.
            git(["config", f"branch.{branch}.remote", "origin"], staging)
            git(["config", f"branch.{branch}.merge", f"refs/heads/{branch}"], staging)
            if exists(destination):
                raise HaiError(f"Checkout destination appeared during setup; preserved: {destination}")
            staging.rename(destination)

    def verify_existing_checkout(self, branch: str, path: Path):
        if path.is_symlink() or not path.is_dir() or not exists(path / ".git"):
            raise HaiError(f"Existing workspace is not an ordinary Git checkout; preserved: {path}")
        top = Path(git(["rev-parse", "--show-toplevel"], path)).resolve()
        if top != path.resolve():
            raise HaiError(f"Existing workspace belongs to another Git checkout; preserved: {path}")
        current = git(["branch", "--show-current"], path)
        if current != branch:
            self.warnings.append(
                f"Workspace {branch} is currently on {current or 'a detached HEAD'}; left unchanged: {path}"
            )

    def link_workspace_assets(self, branch: str, checkout: Path):
        shared = self.workspace / "shared"
        output = self.workspace / "outputs" / branch_slug(branch)
        managed_directory(output)
        link_if_absent(checkout / "models", shared / "models", self.warnings)
        linked_output = link_if_absent(checkout / "outputs", output, self.warnings)

        data = checkout / "data"
        if data.is_symlink() or (exists(data) and not data.is_dir()):
            self.warnings.append(f"Preserved existing data path; no child links added: {data}")
        else:
            data.mkdir(exist_ok=True)
            tracked = set(git(["ls-files", "--", "data"], checkout).splitlines())
            for source in sorted((shared / "data").iterdir()):
                if source.name.startswith(".") or not source.is_dir():
                    continue
                relative = f"data/{source.name}"
                if any(p == relative or p.startswith(relative + "/") for p in tracked):
                    self.warnings.append(f"Preserved tracked dataset path: {checkout / relative}")
                    continue
                link_if_absent(data / source.name, source, self.warnings)

        third_party = checkout / "third_party"
        if third_party.is_symlink() or (exists(third_party) and not third_party.is_dir()):
            self.warnings.append(f"Preserved existing third_party path; SAM link not added: {third_party}")
        else:
            third_party.mkdir(exist_ok=True)
            link_if_absent(third_party / "SAM", Path(self.config.assets["sam_repository"]), self.warnings)

        # Never write through an existing custom outputs link/directory. Users
        # keeping one can pass gen.train_lora --script explicitly instead.
        if linked_output:
            tools = output / "age_lora_tools"
            if tools.is_symlink() or (exists(tools) and not tools.is_dir()):
                self.warnings.append(f"Preserved existing LoRA tool directory: {tools}")
            else:
                tools.mkdir(exist_ok=True)
                link_if_absent(tools / "train_text_to_image_lora.py",
                               Path(self.config.assets["lora_training_script"]), self.warnings)

    def setup(self, branches: list[str]) -> list[Path]:
        # Validate every requested name before any workspace mutation.
        for branch in branches:
            self.config.branch(branch)
        paths = []
        with self.setup_lock():
            self.prepare_layout()
            for branch in branches:
                path = self.branch_path(branch)
                if exists(path):
                    self.verify_existing_checkout(branch, path)
                else:
                    self.create_checkout(branch, path)
                self.link_workspace_assets(branch, path)
                paths.append(path)
        return paths

    def extension_key(self, env_name: str, env_root: Path) -> str:
        # JIT extensions are not a portable model cache. Isolate them by
        # environment, interpreter/package metadata, and container/GPU identity.
        evidence = [str(env_root), socket.gethostname(), os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                    os.environ.get("NVIDIA_VISIBLE_DEVICES", "")]
        metadata = [env_root / "pyvenv.cfg"]
        metadata.extend(sorted(env_root.glob("lib/python*/site-packages/torch-*.dist-info/METADATA")))
        for path in metadata:
            try:
                evidence.append(str(path) + "\n" + path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError):
                evidence.append(str(path) + ":unavailable")
        override = os.environ.get("HAI_TORCH_EXTENSIONS_KEY")
        if override:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", override) or ".." in override:
                raise HaiError("HAI_TORCH_EXTENSIONS_KEY must be a simple cache key, not a path")
            evidence.append(override)
        return env_name + "-" + hashlib.sha256("\n".join(evidence).encode()).hexdigest()[:20]

    def environment(self, branch: str, override: str | None = None) -> dict[str, str]:
        env_name, env_root = self.config.environment(branch, override)
        interpreter = env_root / "bin/python"
        if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
            raise HaiError(f"Environment interpreter is missing or not executable: {interpreter}")
        environment = dict(os.environ)
        old_venv = environment.get("VIRTUAL_ENV")
        excluded = {str(Path(p) / "bin") for p in self.config.environments.values()}
        if old_venv:
            excluded.add(str(Path(old_venv) / "bin"))
        old_path = [p for p in environment.get("PATH", os.defpath).split(os.pathsep) if p and p not in excluded]
        environment["PATH"] = os.pathsep.join([str(env_root / "bin"), *old_path])
        environment.update(VIRTUAL_ENV=str(env_root), VIRTUAL_ENV_PROMPT=env_name,
                           HAI_ENV=env_name, HAI_BRANCH=branch, HAI_WORKSPACE=str(self.workspace),
                           HAI_BRANCH_DIR=str(self.branch_path(branch)), PYTHONNOUSERSITE="1")
        # A stale user/system Python search path must not silently cross venvs.
        environment.pop("PYTHONHOME", None)
        environment.pop("PYTHONPATH", None)
        environment.setdefault("HF_HOME", str(self.workspace / "cache/huggingface"))
        environment.setdefault("TORCH_HOME", str(self.workspace / "shared/models/torch"))
        environment.setdefault("XDG_CACHE_HOME", str(self.workspace / "cache"))
        # Deliberately replace the generic inherited TORCH_EXTENSIONS_DIR: a
        # cache shared across environments may contain incompatible binaries.
        environment["TORCH_EXTENSIONS_DIR"] = str(
            self.workspace / "cache/torch_extensions" / self.extension_key(env_name, env_root)
        )
        return environment

    def show_warnings(self):
        for warning in dict.fromkeys(self.warnings):
            print(f"hai: warning: {warning}", file=sys.stderr)

    def info(self):
        print(f"Workspace: {self.workspace}\nSource seed: {self.config.source_seed}")
        print("Checkouts are initialized at the pinned SHA; existing work is never reset or updated.")
        for name, specification in self.config.branches.items():
            path = self.branch_path(name)
            state = "not initialized"
            if path.is_dir() and not path.is_symlink() and exists(path / ".git"):
                try:
                    current = git(["branch", "--show-current"], path) or "detached"
                    head = git(["rev-parse", "HEAD"], path)[:12]
                    state = f"{current} @ {head}"
                except HaiError as error:
                    state = str(error)
            print(f"\n{name}\n  environment: {specification['environment']}"
                  f"\n  image pin: {specification['commit']}\n  checkout: {path}\n  state: {state}")
        print("\nEnvironments:")
        for name, directory in self.config.environments.items():
            print(f"  {name}: {directory}")
        print(f"\nShared models: {self.workspace / 'shared/models'}"
              f"\nShared data: {self.workspace / 'shared/data'}"
              f"\nIsolated outputs: {self.workspace / 'outputs'}")


DOCTOR_SCRIPT = r'''
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

expected_root, environment_name, require_gpu = sys.argv[1:]
errors = []
if Path(sys.prefix).resolve() != Path(expected_root).resolve():
    errors.append(f"Interpreter prefix {sys.prefix} does not match selected environment {expected_root}")
modules = {
    "torch": "torch", "torchvision": "torchvision", "numpy": "numpy", "Pillow": "PIL",
    "opencv-python-headless": "cv2", "onnxruntime": "onnxruntime", "onnx": "onnx",
    "diffusers": "diffusers", "transformers": "transformers", "peft": "peft",
    "accelerate": "accelerate", "datasets": "datasets",
}
if environment_name == "web":
    modules.update({"fastapi": "fastapi", "uvicorn": "uvicorn", "python-multipart": "python_multipart",
                    "qrcode": "qrcode", "requests": "requests", "Flask": "flask", "pyserial": "serial"})
for distribution, module in modules.items():
    try:
        importlib.import_module(module)
        print(f"OK {distribution}=={importlib.metadata.version(distribution)}")
    except Exception as error:
        errors.append(f"{distribution}: {type(error).__name__}: {error}")
if "torch" in sys.modules:
    import torch
    print(f"Torch CUDA build: {torch.version.cuda}; GPU available: {torch.cuda.is_available()}")
    if require_gpu == "yes":
        if not torch.cuda.is_available():
            errors.append("GPU check requested, but CUDA is unavailable. No CPU/Mock fallback.")
        else:
            try:
                for index in range(torch.cuda.device_count()):
                    value = torch.ones((2, 2), device=f"cuda:{index}")
                    if (value @ value).sum().item() != 8:
                        raise RuntimeError("GPU arithmetic check returned the wrong result")
                    torch.cuda.synchronize(index)
                    print(f"OK GPU {index}: {torch.cuda.get_device_name(index)}, capability {torch.cuda.get_device_capability(index)}")
            except Exception as error:
                errors.append(f"GPU check failed: {type(error).__name__}: {error}")
elif require_gpu == "yes":
    errors.append("GPU check requested, but PyTorch could not be imported")
for error in errors:
    print("ERROR " + error, file=sys.stderr)
print("Models and datasets were not loaded; this is not a real-model quality or readiness check.")
sys.exit(1 if errors else 0)
'''


def doctor(runtime: Runtime, branch: str, override: str | None, gpu: bool) -> int:
    name, root = runtime.config.environment(branch, override)
    environment = runtime.environment(branch, override)
    missing = []
    for key, location in runtime.config.assets.items():
        if not Path(location).exists():
            missing.append(f"Missing image asset {key}: {location}")
    if not runtime.config.source_seed.is_dir():
        missing.append(f"Missing image source seed: {runtime.config.source_seed}")
    for error in missing:
        print(f"ERROR {error}", file=sys.stderr)
    print(f"Checking {name}: {root}", flush=True)
    try:
        result = subprocess.run([str(root / "bin/python"), "-c", DOCTOR_SCRIPT, str(root), name,
                                 "yes" if gpu else "no"], env=environment, cwd="/", check=False)
    except OSError as error:
        raise HaiError(f"Cannot run environment check: {error}") from error
    return 1 if missing else result.returncode


def shell_command(environment: dict[str, str]) -> list[str]:
    # Do not run Vast's startup profiles again: they may activate /venv/main
    # after the user explicitly selected web or m1.
    preferred = environment.get("SHELL", "/bin/bash")
    if Path(preferred).name == "zsh" and Path(preferred).is_file():
        return [preferred, "-f", "-i"]
    bash = shutil.which("bash", path=environment["PATH"])
    if not bash:
        raise HaiError("Bash is required for an isolated interactive shell")
    return [bash, "--noprofile", "--norc", "-i"]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="action", required=True)
    setup = commands.add_parser("setup", help="Initialize pinned offline workspaces; preserve all existing work")
    targets = setup.add_mutually_exclusive_group()
    targets.add_argument("--branch")
    targets.add_argument("--all", action="store_true")
    for name in ("shell", "run"):
        sub = commands.add_parser(name, help="Open a shell" if name == "shell" else "Run a command after --")
        sub.add_argument("branch", nargs="?")
        sub.add_argument("--env", dest="environment")
    commands.add_parser("info", help="Show pins, workspace state, and environment mappings")
    check = commands.add_parser("doctor", help="Check imports; --gpu requires working CUDA")
    check.add_argument("--branch")
    check.add_argument("--env", dest="environment")
    check.add_argument("--gpu", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    command: list[str] = []
    if arguments and arguments[0] == "run" and "--" in arguments:
        split = arguments.index("--")
        command, arguments = arguments[split + 1:], arguments[:split]
    cli = parser()
    args = cli.parse_args(arguments)
    if args.action == "run" and not command:
        cli.error("hai run requires a command after -- (example: hai run main -- python -m gen.smoke)")
    try:
        config_path = absolute_path(os.environ.get("HAI_CONFIG", DEFAULT_CONFIG), "HAI_CONFIG")
        config = Configuration.load(config_path)
        runtime = Runtime(config, absolute_path(os.environ.get("HAI_WORKSPACE", DEFAULT_WORKSPACE), "HAI_WORKSPACE"))
        branch = getattr(args, "branch", None) or os.environ.get("HAI_BRANCH") or config.default_branch
        if args.action == "info":
            runtime.info()
            return 0
        if args.action == "setup":
            names = list(config.branches) if args.all else [branch]
            paths = runtime.setup(names)
            runtime.show_warnings()
            for name, path in zip(names, paths):
                print(f"{name}: {path} [environment={config.branch(name)['environment']}]")
            return 0
        config.branch(branch)
        if args.action == "doctor":
            return doctor(runtime, branch, args.environment, args.gpu)
        environment = runtime.environment(branch, args.environment)
        path, = runtime.setup([branch])
        runtime.show_warnings()
        if args.action == "shell":
            environment["PS1"] = f"[hai:{environment['HAI_ENV']} {branch}] \\w $ "
            environment["PROMPT"] = f"[hai:{environment['HAI_ENV']} {branch}] %~ %# "
            command = shell_command(environment)
        sys.stdout.flush()
        sys.stderr.flush()
        os.chdir(path)
        os.execvpe(command[0], command, environment)
    except (HaiError, OSError) as error:
        print(f"hai: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
