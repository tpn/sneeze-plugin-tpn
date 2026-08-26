# Copyright (c) 2026 Trent Nelson
# SPDX-License-Identifier: Apache-2.0

"""Lightweight orchestration for the isolated local image upscaler."""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import shutil
import stat
import subprocess
import tempfile
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..ocr_path_security import (
    TrustedPathError,
    prepare_trusted_directory,
    validate_trusted_file,
    validate_trusted_tree,
)

MODEL_FILENAME = "HAT-L_SRx2_ImageNet-pretrain.pth"
MODEL_UPSTREAM_REPOSITORY = "XPixelGroup/HAT"
MODEL_UPSTREAM_URL = "https://github.com/XPixelGroup/HAT"
MODEL_UPSTREAM_CHECKPOINT_URL = (
    "https://drive.google.com/file/d/"
    "16xtMezHvckdWEuSiOxcO-dgOlsI0rEUg/view?usp=sharing"
)
MODEL_MIRROR_REPOSITORY = "jaideepsingh/upscale_models"
MODEL_MIRROR_REVISION = "867e7c0ad519aba5d36b5bb4ef4a4a91781914c4"
MODEL_MIRROR_PATH = f"HAT/{MODEL_FILENAME}"
MODEL_SHA256 = (
    "2818c7ca8d72ec4cc5f31c93203d55252a662dd35cda34ce1a69661f97dcd38f"
)
MODEL_SIZE_BYTES = 165_182_573
MODEL_SHA256_PROVENANCE = (
    "independently matched the official Google Drive artifact to the "
    "immutable mirror revision"
)
MODEL_URL = (
    f"https://huggingface.co/{MODEL_MIRROR_REPOSITORY}/resolve/"
    f"{MODEL_MIRROR_REVISION}/{MODEL_MIRROR_PATH}?download=true"
)

PYTHON_VERSION = "3.12.13"
PYTORCH_INDEX_URL = "https://download.pytorch.org/whl/cu130"
PYPI_INDEX_URL = "https://pypi.org/simple"
TORCH_PACKAGES = ("torch==2.11.0", "torchvision==0.26.0")
RUNTIME_PACKAGES = (
    "einops==0.8.2",
    "numpy==2.5.2",
    "pillow==12.3.0",
    "safetensors==0.8.0",
    "spandrel==0.4.2",
)

_SECRET_ENV_MARKERS = (
    "AUTH",
    "COOKIE",
    "CREDENTIAL",
    "KEY",
    "PASSWORD",
    "SECRET",
    "TOKEN",
)
_ISOLATED_ENV_NAMES = {
    "ALL_PROXY",
    "CURL_CA_BUNDLE",
    "FTP_PROXY",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "VIRTUAL_ENV",
    "VIRTUAL_ENV_PROMPT",
    "__PYVENV_LAUNCHER__",
}
_ISOLATED_ENV_PREFIXES = (
    "CUDA_",
    "DYLD_",
    "LD_",
    "NVIDIA_",
    "PIP_",
    "PYTHON",
    "PYTORCH",
    "TORCH",
    "TRITON_",
    "UV_",
)
_PROC_STATUS = Path("/proc/self/status")
_SETUP_UMASK = 0o077


class ImageUpscaleError(ValueError):
    """Raised when the upscaler cannot be set up or run safely."""


def default_runtime_dir() -> Path:
    """Return the isolated runtime location without creating it."""

    data_home = os.environ.get("XDG_DATA_HOME")
    base = (
        Path(data_home).expanduser()
        if data_home
        else Path.home() / ".local/share"
    )
    return (base / "sneeze/tpn/upscale/runtime").resolve()


def default_model_cache() -> Path:
    """Return the model cache location without creating it."""

    cache_home = os.environ.get("XDG_CACHE_HOME")
    base = (
        Path(cache_home).expanduser()
        if cache_home
        else Path.home() / ".cache"
    )
    return (base / "sneeze/tpn/upscale/models").resolve()


def resolve_runtime_dir(value: str | os.PathLike[str] | None) -> Path:
    return _absolute_path(value) if value else default_runtime_dir()


def resolve_model_cache(value: str | os.PathLike[str] | None) -> Path:
    return _absolute_path(value) if value else default_model_cache()


def resolve_model_path(value: str | os.PathLike[str] | None) -> Path:
    return resolve_model_cache(value) / MODEL_FILENAME


def default_output_path(
    source: str | os.PathLike[str],
    target_width: int = 3840,
) -> Path:
    source_path = _destination_path(source)
    suffix = "4k" if target_width == 3840 else f"{target_width}w"
    return source_path.with_name(f"{source_path.stem}-{suffix}.png")


def runtime_python(runtime_dir: str | os.PathLike[str]) -> Path:
    return _absolute_path(runtime_dir) / "bin/python"


def runtime_script() -> Path:
    return Path(__file__).with_name("image_upscale_runtime.py").resolve()


def build_setup_commands(
    *,
    runtime_dir: str | os.PathLike[str],
    uv_executable: str,
    include_venv: bool,
) -> list[list[str]]:
    """Build the exact environment commands used by setup."""

    runtime_path = _absolute_path(runtime_dir)
    python_path = runtime_python(runtime_path)
    commands: list[list[str]] = []
    if include_venv:
        commands.append(
            [
                uv_executable,
                "--no-config",
                "venv",
                "--managed-python",
                "--python",
                PYTHON_VERSION,
                str(runtime_path),
            ]
        )
    commands.extend(
        [
            [
                uv_executable,
                "--no-config",
                "pip",
                "install",
                "--python",
                str(python_path),
                "--default-index",
                PYTORCH_INDEX_URL,
                *TORCH_PACKAGES,
            ],
            [
                uv_executable,
                "--no-config",
                "pip",
                "install",
                "--python",
                str(python_path),
                "--default-index",
                PYPI_INDEX_URL,
                *RUNTIME_PACKAGES,
            ],
        ]
    )
    return commands


def setup_upscaler(
    *,
    runtime_dir: str | os.PathLike[str] | None = None,
    model_cache: str | os.PathLike[str] | None = None,
    uv_executable: str = "uv",
    cuda_visible_devices: str = "0",
    dry_run: bool = False,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = (
        subprocess.run
    ),
) -> dict[str, Any]:
    """Create the pinned runtime, fetch the model, and probe Blackwell."""

    runtime_path = resolve_runtime_dir(runtime_dir)
    model_path = resolve_model_path(model_cache)
    python_path = runtime_python(runtime_path)
    located_uv = shutil.which(uv_executable)
    if not located_uv:
        raise ImageUpscaleError(
            f"Could not find uv executable: {uv_executable}"
        )
    try:
        uv_path = str(Path(located_uv).resolve(strict=True))
    except OSError as exc:
        raise ImageUpscaleError(
            f"Could not resolve uv executable: {located_uv}"
        ) from exc
    _validate_setup_tool(Path(uv_path))

    include_venv = not python_path.is_file()
    commands = build_setup_commands(
        runtime_dir=runtime_path,
        uv_executable=uv_path,
        include_venv=include_venv,
    )
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "kind": "sneeze_tpn_image_upscale_setup",
        "dry_run": bool(dry_run),
        "runtime": {
            "directory": str(runtime_path),
            "python": str(python_path),
            "python_version": PYTHON_VERSION,
            "torch_packages": list(TORCH_PACKAGES),
            "runtime_packages": list(RUNTIME_PACKAGES),
            "pytorch_index_url": PYTORCH_INDEX_URL,
            "pypi_index_url": PYPI_INDEX_URL,
        },
        "model": model_identity(model_path),
        "commands": commands,
    }
    if dry_run:
        return receipt

    try:
        prepare_trusted_directory(
            runtime_path,
            label="upscale runtime",
        )
    except TrustedPathError as exc:
        raise ImageUpscaleError(
            f"Upscale runtime is not trusted: {exc}"
        ) from exc
    try:
        prepare_trusted_directory(
            model_path.parent,
            label="upscale model cache",
        )
    except TrustedPathError as exc:
        raise ImageUpscaleError(
            f"Upscale model cache is not trusted: {exc}"
        ) from exc
    setup_environment = _runtime_environment(cuda_visible_devices)
    if not include_venv:
        _secure_runtime_lock(runtime_path)
        _validate_runtime_trust(runtime_path, python_path)
    for index, command in enumerate(commands):
        completed = command_runner(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=setup_environment,
            umask=_SETUP_UMASK,
        )
        if completed.returncode != 0:
            output = (completed.stdout or "").strip()
            raise ImageUpscaleError(
                "Runtime setup command failed with exit status "
                f"{completed.returncode}: {_quote_command(command)}"
                + (f"\n{output}" if output else "")
            )
        if include_venv and index == 0:
            _secure_runtime_lock(runtime_path)
            _validate_runtime_trust(runtime_path, python_path)

    _secure_runtime_lock(runtime_path)
    _ensure_model(model_path)
    script_path = runtime_script()
    _validate_upscale_trust(
        runtime_path=runtime_path,
        python_path=python_path,
        model_path=model_path,
        script_path=script_path,
    )
    probe_command = [str(python_path), "-P", str(script_path), "--probe"]
    completed = command_runner(
        probe_command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_runtime_environment(cuda_visible_devices),
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise ImageUpscaleError(
            "Upscale runtime probe failed" + (f": {detail}" if detail else "")
        )
    try:
        probe_receipt = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ImageUpscaleError(
            "Upscale runtime probe returned invalid JSON"
        ) from exc
    _validate_runtime_probe(probe_receipt)
    receipt["probe"] = probe_receipt
    receipt["model"] = model_identity(model_path, verify=True)
    return receipt


def run_upscale(
    *,
    source: str | os.PathLike[str],
    output: str | os.PathLike[str] | None = None,
    target_width: int = 3840,
    runtime_dir: str | os.PathLike[str] | None = None,
    model_cache: str | os.PathLike[str] | None = None,
    cuda_visible_devices: str = "0",
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    tile_size: int = 0,
    tile_overlap: int = 64,
    force: bool = False,
    receipt_path: str | os.PathLike[str] | None = None,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = (
        subprocess.run
    ),
) -> dict[str, Any]:
    """Run the pinned model in the isolated runtime and return its receipt."""

    source_path = _absolute_path(source)
    if not source_path.is_file():
        raise ImageUpscaleError(f"Source image does not exist: {source_path}")
    source_hash = file_sha256(source_path)
    output_path = (
        _destination_path(output)
        if output
        else default_output_path(source, target_width)
    )
    if _paths_alias(output_path, source_path):
        raise ImageUpscaleError("Source and output paths must be different")
    _reject_symlink_destination(output_path, "Output")
    if output_path.suffix.lower() != ".png":
        raise ImageUpscaleError("Upscale output must use a .png suffix")
    if output_path.is_dir():
        raise ImageUpscaleError(
            f"Upscale output must not be a directory: {output_path}"
        )
    if os.path.lexists(output_path) and not force:
        raise ImageUpscaleError(
            "Output already exists; pass --force to replace it: "
            f"{output_path}"
        )
    if target_width <= 0:
        raise ImageUpscaleError("Target width must be greater than zero")
    if tile_size < 0 or tile_overlap < 0:
        raise ImageUpscaleError("Tile size and overlap cannot be negative")
    if tile_size and tile_overlap >= tile_size:
        raise ImageUpscaleError("Tile overlap must be smaller than tile size")
    if dtype not in {"bfloat16", "float32"}:
        raise ImageUpscaleError("Dtype must be bfloat16 or float32")

    runtime_path = resolve_runtime_dir(runtime_dir)
    python_path = runtime_python(runtime_path)
    if not python_path.is_file():
        raise ImageUpscaleError(
            "Upscale runtime is missing; run `sne-tpn-upscale setup` "
            f"first: {python_path}"
        )
    model_path = resolve_model_path(model_cache)
    if not model_path.is_file():
        raise ImageUpscaleError(
            "Upscale model is missing; run `sne-tpn-upscale setup` "
            f"first: {model_path}"
        )
    model_identity(model_path, verify=True)
    script_path = runtime_script()
    _validate_upscale_trust(
        runtime_path=runtime_path,
        python_path=python_path,
        model_path=model_path,
        script_path=script_path,
    )
    implementation_hash = file_sha256(script_path)

    receipt_output = None
    if receipt_path is not None:
        receipt_output = _destination_path(receipt_path)
        _reject_symlink_destination(receipt_output, "Receipt")
        protected_paths = {
            source_path,
            output_path,
            model_path,
            python_path,
            script_path,
        }
        if any(
            _paths_alias(receipt_output, protected)
            for protected in protected_paths
        ):
            raise ImageUpscaleError(
                "Receipt path must be separate from the source, output, "
                "model, and runtime"
            )
        if os.path.lexists(receipt_output) and not force:
            raise ImageUpscaleError(
                "Receipt already exists; pass --force to replace it: "
                f"{receipt_output}"
            )

    command = [
        str(python_path),
        "-P",
        str(script_path),
        "--source",
        str(source_path),
        "--output",
        str(output_path),
        "--model",
        str(model_path),
        "--model-sha256",
        MODEL_SHA256,
        "--target-width",
        str(target_width),
        "--device",
        device,
        "--dtype",
        dtype,
        "--tile-size",
        str(tile_size),
        "--tile-overlap",
        str(tile_overlap),
    ]
    if force:
        command.append("--force")
    completed = command_runner(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_runtime_environment(cuda_visible_devices),
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise ImageUpscaleError(
            "Image upscale failed" + (f": {detail[-4000:]}" if detail else "")
        )
    unverified_output = f"; unverified output may remain at: {output_path}"
    try:
        receipt = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ImageUpscaleError(
            "Image upscale runtime returned invalid JSON" + unverified_output
        ) from exc
    if not isinstance(receipt, Mapping):
        raise ImageUpscaleError(
            "Image upscale runtime returned a non-object receipt"
            + unverified_output
        )
    implementation_receipt = receipt.get("implementation", {})
    if (
        not isinstance(implementation_receipt, Mapping)
        or implementation_receipt.get("path") != str(script_path)
        or implementation_receipt.get("sha256") != implementation_hash
        or file_sha256(script_path) != implementation_hash
    ):
        raise ImageUpscaleError(
            "Image upscale runtime implementation identity does not match"
            + unverified_output
        )
    source_receipt = receipt.get("source", {})
    output_receipt = receipt.get("output", {})
    if not isinstance(source_receipt, Mapping) or not isinstance(
        output_receipt,
        Mapping,
    ):
        raise ImageUpscaleError(
            "Image upscale runtime returned malformed image identities"
            + unverified_output
        )
    if source_receipt.get("path") != str(source_path):
        raise ImageUpscaleError(
            "Image upscale receipt reported another source"
            + unverified_output
        )
    if (
        source_receipt.get("sha256") != source_hash
        or file_sha256(source_path) != source_hash
    ):
        raise ImageUpscaleError(
            "Source image changed during upscale or its receipt is invalid"
            + unverified_output
        )
    if output_receipt.get("path") != str(output_path):
        raise ImageUpscaleError(
            "Image upscale receipt reported another output"
            + unverified_output
        )
    if not output_path.is_file():
        raise ImageUpscaleError(
            f"Image upscale did not create its output: {output_path}"
        )
    actual_output_hash = file_sha256(output_path)
    if output_receipt.get("sha256") != actual_output_hash:
        raise ImageUpscaleError(
            "Image upscale receipt output SHA-256 does not match the file"
            + unverified_output
        )
    output_dimensions = output_receipt.get("dimensions")
    parameters_receipt = receipt.get("parameters", {})
    target_height = (
        parameters_receipt.get("target_height")
        if isinstance(parameters_receipt, Mapping)
        else None
    )
    if (
        not isinstance(output_dimensions, list)
        or len(output_dimensions) != 2
        or output_dimensions[0] != target_width
        or type(target_height) is not int
        or target_height <= 0
        or output_dimensions[1] != target_height
    ):
        raise ImageUpscaleError(
            "Image upscale receipt output dimensions do not match the targets"
            + unverified_output
        )

    if receipt_output is not None:
        receipt["receipt_path"] = str(receipt_output)
        try:
            _write_json_atomic(receipt_output, receipt, force=force)
        except (ImageUpscaleError, OSError) as exc:
            raise ImageUpscaleError(
                f"{exc}; completed upscale output remains at: {output_path}"
            ) from exc
    return receipt


def model_identity(
    path: str | os.PathLike[str],
    *,
    verify: bool = False,
) -> dict[str, Any]:
    model_path = _absolute_path(path)
    identity: dict[str, Any] = {
        "architecture": "HAT-L",
        "native_scale": 2,
        "upstream": {
            "repository": MODEL_UPSTREAM_REPOSITORY,
            "url": MODEL_UPSTREAM_URL,
            "checkpoint_url": MODEL_UPSTREAM_CHECKPOINT_URL,
            "code_license": "Apache-2.0",
            "checkpoint_terms": "not separately stated upstream",
        },
        "download_mirror": {
            "repository": MODEL_MIRROR_REPOSITORY,
            "revision": MODEL_MIRROR_REVISION,
            "repository_path": MODEL_MIRROR_PATH,
            "url": MODEL_URL,
        },
        "path": str(model_path),
        "sha256": MODEL_SHA256,
        "expected_size_bytes": MODEL_SIZE_BYTES,
        "sha256_provenance": MODEL_SHA256_PROVENANCE,
    }
    if verify:
        actual = file_sha256(model_path)
        if actual != MODEL_SHA256:
            raise ImageUpscaleError(
                f"Model SHA-256 mismatch for {model_path}: "
                f"expected {MODEL_SHA256}, got {actual}"
            )
        identity["size_bytes"] = model_path.stat().st_size
    return identity


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_model(model_path: Path) -> None:
    if model_path.is_file():
        model_identity(model_path, verify=True)
        return
    if model_path.exists():
        raise ImageUpscaleError(
            f"Model path is not a regular file: {model_path}"
        )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        MODEL_URL,
        headers={"User-Agent": "sneeze-plugin-tpn/0.1"},
    )
    temporary_path: str | None = None
    digest = hashlib.sha256()
    downloaded_size = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=model_path.parent,
            prefix=f".{model_path.name}.",
            suffix=".download",
            delete=False,
        ) as destination:
            temporary_path = destination.name
            with urllib.request.urlopen(request, timeout=60) as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    downloaded_size += len(chunk)
                    if downloaded_size > MODEL_SIZE_BYTES:
                        raise ImageUpscaleError(
                            "Downloaded model exceeded expected size of "
                            f"{MODEL_SIZE_BYTES} bytes"
                        )
                    destination.write(chunk)
                    digest.update(chunk)
            destination.flush()
            os.fsync(destination.fileno())
        if downloaded_size != MODEL_SIZE_BYTES:
            raise ImageUpscaleError(
                "Downloaded model size mismatch: "
                f"expected {MODEL_SIZE_BYTES}, got {downloaded_size}"
            )
        actual = digest.hexdigest()
        if actual != MODEL_SHA256:
            raise ImageUpscaleError(
                "Downloaded model SHA-256 mismatch: "
                f"expected {MODEL_SHA256}, got {actual}"
            )
        os.chmod(
            temporary_path,
            _file_mode_from_umask() & ~0o022,
        )
        os.replace(temporary_path, model_path)
        temporary_path = None
    except (OSError, http.client.HTTPException) as exc:
        raise ImageUpscaleError(
            f"Could not download upscale model: {exc}"
        ) from exc
    finally:
        if temporary_path:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass


def _runtime_environment(cuda_visible_devices: str) -> dict[str, str]:
    if not cuda_visible_devices or any(
        char.isspace() for char in cuda_visible_devices
    ):
        raise ImageUpscaleError(
            "CUDA visible devices must not be empty or spaced"
        )
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.upper() not in _ISOLATED_ENV_NAMES
        and not key.upper().startswith(_ISOLATED_ENV_PREFIXES)
        and not any(marker in key.upper() for marker in _SECRET_ENV_MARKERS)
    }
    environment.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    return environment


def _validate_runtime_probe(probe_receipt: Any) -> None:
    if not isinstance(probe_receipt, Mapping):
        raise ImageUpscaleError("Upscale runtime probe returned a non-object")
    if not probe_receipt.get("cuda_available"):
        raise ImageUpscaleError("Upscale runtime cannot access CUDA")
    gpu = probe_receipt.get("gpu")
    if not isinstance(gpu, Mapping):
        raise ImageUpscaleError("Upscale runtime probe omitted its GPU")
    if not gpu.get("bfloat16_supported"):
        raise ImageUpscaleError(
            "Upscale runtime GPU does not support the default BF16 path"
        )
    expected_versions = {
        "python": PYTHON_VERSION,
        "torch": "2.11.0+cu130",
        "torch_cuda": "13.0",
        "torchvision": "0.26.0+cu130",
        "spandrel": "0.4.2",
        "pillow": "12.3.0",
        "numpy": "2.5.2",
    }
    for name, expected in expected_versions.items():
        actual = probe_receipt.get(name)
        matches = isinstance(actual, str) and actual == expected
        if not matches:
            raise ImageUpscaleError(
                f"Upscale runtime probe expected {name} {expected}, "
                f"got {actual}"
            )


def _absolute_path(value: str | os.PathLike[str]) -> Path:
    return Path(value).expanduser().resolve()


def _destination_path(value: str | os.PathLike[str]) -> Path:
    """Resolve destination parents without following the final component."""

    lexical = Path(os.path.abspath(Path(value).expanduser()))
    return lexical.parent.resolve() / lexical.name


def _paths_alias(first: Path, second: Path) -> bool:
    if first == second:
        return True
    if not os.path.lexists(first) or not os.path.lexists(second):
        return False
    try:
        return os.path.samefile(first, second)
    except OSError:
        return False


def _reject_symlink_destination(path: Path, label: str) -> None:
    if path.is_symlink():
        raise ImageUpscaleError(
            f"{label} path must not be a symbolic link: {path}"
        )


def _file_mode_from_umask() -> int:
    try:
        for line in _PROC_STATUS.read_text(encoding="ascii").splitlines():
            name, separator, value = line.partition(":")
            if name == "Umask" and separator:
                current_umask = int(value.strip(), 8)
                break
        else:
            raise ValueError("process status omitted Umask")
    except (OSError, UnicodeError, ValueError):
        return 0o600
    return (0o666 & ~current_umask) | 0o600


def _secure_runtime_lock(runtime_path: Path) -> None:
    lock_path = runtime_path / ".lock"
    if not os.path.lexists(lock_path):
        return
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = None
    try:
        descriptor = os.open(lock_path, flags)
        descriptor_metadata = os.fstat(descriptor)
        path_metadata = lock_path.lstat()
        if (
            not stat.S_ISREG(descriptor_metadata.st_mode)
            or descriptor_metadata.st_uid != os.geteuid()
            or descriptor_metadata.st_nlink != 1
            or (descriptor_metadata.st_dev, descriptor_metadata.st_ino)
            != (path_metadata.st_dev, path_metadata.st_ino)
        ):
            raise ImageUpscaleError(
                "Upscale runtime lock must be a private regular file: "
                f"{lock_path}"
            )
        os.fchmod(descriptor, 0o600)
    except ImageUpscaleError:
        raise
    except OSError as exc:
        raise ImageUpscaleError(
            f"Could not secure upscale runtime lock: {lock_path}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _validate_setup_tool(uv_path: Path) -> None:
    try:
        validate_trusted_file(
            uv_path,
            label="uv executable",
            executable=True,
        )
    except TrustedPathError as exc:
        raise ImageUpscaleError(
            f"Upscale setup tool is not trusted: {exc}"
        ) from exc


def _runtime_python_identity(
    runtime_path: Path,
    python_path: Path,
) -> tuple[Path, Path]:
    if not python_path.is_relative_to(runtime_path):
        raise ImageUpscaleError(
            f"Upscale Python must remain within its runtime: {python_path}"
        )
    if not python_path.is_symlink():
        raise ImageUpscaleError(
            f"Upscale Python must be the uv-managed environment link: "
            f"{python_path}"
        )
    try:
        raw_target = Path(os.readlink(python_path))
        target = Path(
            os.path.abspath(
                raw_target
                if raw_target.is_absolute()
                else python_path.parent / raw_target
            )
        )
        resolved_target = target.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ImageUpscaleError(
            f"Could not resolve upscale Python target: {python_path}"
        ) from exc
    if resolved_target != target:
        raise ImageUpscaleError(
            f"Upscale Python target must be canonical: {target}"
        )
    validate_trusted_file(
        target,
        label="upscale Python interpreter",
        executable=True,
        resolve_symlinks=False,
    )
    python_base = target.parent.parent
    python_series = PYTHON_VERSION.rpartition(".")[0]
    managed_prefix = f"cpython-{PYTHON_VERSION}-"
    if (
        target.name != f"python{python_series}"
        or target.parent.name != "bin"
        or not python_base.name.startswith(managed_prefix)
        or not python_base.name.removeprefix(managed_prefix)
        or python_base.parent.name != "python"
    ):
        raise ImageUpscaleError(
            "Upscale Python must target an exact uv-managed Python "
            f"{PYTHON_VERSION} installation: {target}"
        )
    validate_trusted_file(
        python_base / "BUILD",
        label="upscale managed Python build identity",
        resolve_symlinks=False,
    )
    return python_base, target


def _validate_runtime_python_config(
    runtime_path: Path,
    python_target: Path,
) -> None:
    config_path = runtime_path / "pyvenv.cfg"
    validate_trusted_file(
        config_path,
        label="upscale Python configuration",
        resolve_symlinks=False,
    )

    try:
        config_lines = config_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ImageUpscaleError(
            f"Could not read upscale Python configuration: {config_path}"
        ) from exc
    homes = [
        line.partition("=")[2].strip()
        for line in config_lines
        if line.partition("=")[0].strip() == "home"
    ]
    expected_home = str(python_target.parent)
    if homes != [expected_home]:
        raise ImageUpscaleError(
            f"Upscale Python configuration must name canonical home "
            f"{expected_home}: {config_path}"
        )


def _validate_symlink_scope(
    root: Path,
    *,
    allowed_external: Mapping[Path, Path] | None = None,
) -> None:
    allowed = allowed_external or {}

    def raise_walk_error(error: OSError) -> None:
        raise ImageUpscaleError(
            f"Could not scan upscale trusted tree: {root}"
        ) from error

    for directory, directory_names, file_names in os.walk(
        root,
        followlinks=False,
        onerror=raise_walk_error,
    ):
        for name in (*directory_names, *file_names):
            path = Path(directory) / name
            if not path.is_symlink():
                continue
            try:
                raw_target = Path(os.readlink(path))
            except OSError as exc:
                raise ImageUpscaleError(
                    f"Could not inspect upscale symlink: {path}"
                ) from exc
            target = Path(
                os.path.abspath(
                    raw_target
                    if raw_target.is_absolute()
                    else path.parent / raw_target
                )
            )
            if target.is_relative_to(root) or allowed.get(path) == target:
                continue
            raise ImageUpscaleError(
                f"Upscale symlink leaves its trusted tree: {path}"
            )


def _validate_runtime_trust(runtime_path: Path, python_path: Path) -> None:
    try:
        python_base, python_target = _runtime_python_identity(
            runtime_path,
            python_path,
        )
        _validate_symlink_scope(
            runtime_path,
            allowed_external={python_path: python_target},
        )
        validate_trusted_tree(
            runtime_path,
            label="upscale runtime",
            symlink_root=runtime_path.anchor,
        )
        _validate_runtime_python_config(runtime_path, python_target)
        validate_trusted_tree(
            python_base,
            label="upscale Python base",
            symlink_root=python_base,
        )
    except (ImageUpscaleError, TrustedPathError) as exc:
        raise ImageUpscaleError(
            f"Upscale execution path is not trusted: {exc}"
        ) from exc


def _validate_upscale_trust(
    *,
    runtime_path: Path,
    python_path: Path,
    model_path: Path,
    script_path: Path,
) -> None:
    _validate_runtime_trust(runtime_path, python_path)
    try:
        validate_trusted_file(
            script_path,
            label="upscale runtime implementation",
        )
        validate_trusted_file(
            model_path,
            label="upscale model checkpoint",
            resolve_symlinks=False,
        )
    except TrustedPathError as exc:
        raise ImageUpscaleError(
            f"Upscale execution path is not trusted: {exc}"
        ) from exc


def _write_json_atomic(
    path: Path,
    payload: Mapping[str, Any],
    *,
    force: bool,
) -> None:
    _reject_symlink_destination(path, "Receipt")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = handle.name
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, _file_mode_from_umask())
        if force:
            _reject_symlink_destination(path, "Receipt")
            # Replacing the directory entry does not follow a final symlink
            # that appears after the check, so its target remains untouched.
            os.replace(temporary_path, path)
        else:
            try:
                os.link(temporary_path, path)
            except FileExistsError as exc:
                raise ImageUpscaleError(
                    f"Receipt appeared while upscaling: {path}"
                ) from exc
            os.unlink(temporary_path)
        temporary_path = None
    finally:
        if temporary_path:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass


def _quote_command(command: Sequence[str]) -> str:
    import shlex

    return " ".join(shlex.quote(str(value)) for value in command)
