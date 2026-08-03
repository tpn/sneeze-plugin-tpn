"""Setup, receipt, provider-spec, and doctor helpers for OCR backends.

The helpers here construct commands and inspect installed environments without
importing heavyweight inference packages.  Provider execution remains owned by
``ocr_provider`` and benchmark execution remains owned by ``ocr_benchmark``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .ocr_backends import OcrBackendDefinition

SCHEMA_VERSION = 1
DEFAULT_TORCH_BACKEND = "cu130"
DEFAULT_CUDA_130_VLLM_WHEEL_INDEX = (
    "https://wheels.vllm.ai/nightly/cu130/"
)
_PYTORCH_WHEEL_INDEX = "https://download.pytorch.org/whl/{backend}"
_SECRET_FIELD_RE = re.compile(
    r"(?:^|_)(?:api_?key|auth|bearer|cookie|credential|key|password|"
    r"private|secret|token)(?:_|$)",
    re.IGNORECASE,
)
_SECRET_VALUE_RES = (
    re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\bBearer\s+\S+", re.IGNORECASE),
    re.compile(r"-----BEGIN [A-Z ]+ PRIVATE KEY-----"),
    re.compile(
        r"https?://[^/@\s:]+:[^/@\s]+@",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:api[_-]?key|authorization|password|secret|token)="
        r"[^&\s]+",
        re.IGNORECASE,
    ),
)


class BackendSetupError(ValueError):
    """Raised when backend setup inputs or installed state are invalid."""


def json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    """Serialize JSON deterministically with a final newline."""

    if pretty:
        text = json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    else:
        text = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    return (text + "\n").encode("utf-8")


def sha256_json(value: Any) -> str:
    """Hash canonical JSON without formatting whitespace."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_receipt_safe(value: Any, path: str = "receipt") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise BackendSetupError(
                    f"{path} contains a non-string JSON field"
                )
            if _SECRET_FIELD_RE.search(key):
                raise BackendSetupError(
                    f"refusing secret-bearing receipt field: {path}.{key}"
                )
            _assert_receipt_safe(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_receipt_safe(item, f"{path}[{index}]")
    elif isinstance(value, str) and any(
        pattern.search(value) for pattern in _SECRET_VALUE_RES
    ):
        raise BackendSetupError(
            f"refusing credential-like value in {path}"
        )


def assert_command_plan_safe(commands: Sequence[Sequence[str]]) -> None:
    for command_index, command in enumerate(commands):
        for argument_index, argument in enumerate(command):
            _assert_receipt_safe(
                argument,
                f"command[{command_index}][{argument_index}]",
            )


def make_receipt(kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Create a secret-free JSON receipt with its canonical SHA-256."""

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        **dict(payload),
    }
    _assert_receipt_safe(receipt)
    receipt["receipt_sha256"] = sha256_json(receipt)
    return receipt


def receipt_text(receipt: Mapping[str, Any]) -> str:
    """Render a receipt for CLI output."""

    return json_bytes(dict(receipt), pretty=True).decode("utf-8").rstrip()


def resolve_manager(value: str | None) -> str:
    """Resolve an explicit environment manager or find mamba/conda."""

    if value:
        return value
    for candidate in ("mamba", "conda"):
        path = shutil.which(candidate)
        if path:
            return path
    raise BackendSetupError("Could not find mamba or conda on PATH")


def build_environment_commands(
    backend: OcrBackendDefinition,
    *,
    manager: str,
    env_name: str,
    python_version: str,
    torch_backend: str = DEFAULT_TORCH_BACKEND,
    editable_root: str | None = None,
    package_spec: str = "sneeze-plugin-tpn[pdf,ocr]",
) -> list[list[str]]:
    """Build a deterministic environment bootstrap plan for a backend."""

    if not env_name or not python_version:
        raise BackendSetupError(
            "environment name and Python version cannot be empty"
        )
    if backend.environment_kind == "mineru25_transformers":
        if torch_backend != "cu130":
            raise BackendSetupError(
                "the qualified MinerU backend requires torch backend cu130"
            )
        commands = _mineru_environment_commands(
            manager=manager,
            env_name=env_name,
            python_version=python_version,
        )
    elif backend.environment_kind == "deepseek_vllm":
        commands = _deepseek_environment_commands(
            manager=manager,
            env_name=env_name,
            python_version=python_version,
            torch_backend=torch_backend,
            editable_root=editable_root,
            package_spec=package_spec,
        )
    else:
        raise BackendSetupError(
            f"unsupported environment kind: {backend.environment_kind}"
        )
    assert_command_plan_safe(commands)
    return commands


def _base_create_command(
    *,
    manager: str,
    env_name: str,
    python_version: str,
    poppler: bool,
) -> list[str]:
    command = [
        manager,
        "create",
        "-n",
        env_name,
        "-y",
        "-c",
        "conda-forge",
        f"python={python_version}",
        "pip=26.1.1",
        "uv=0.11.17",
    ]
    if poppler:
        command.append("poppler=26.05.0")
    return command


def _manager_run(manager: str, env_name: str, *argv: str) -> list[str]:
    return [manager, "run", "-n", env_name, *argv]


def _deepseek_environment_commands(
    *,
    manager: str,
    env_name: str,
    python_version: str,
    torch_backend: str,
    editable_root: str | None,
    package_spec: str,
) -> list[list[str]]:
    if torch_backend != "cu130":
        raise BackendSetupError(
            "the qualified DeepSeek backend requires torch backend cu130"
        )
    pytorch_index = _PYTORCH_WHEEL_INDEX.format(backend=torch_backend)
    commands = [
        _base_create_command(
            manager=manager,
            env_name=env_name,
            python_version=python_version,
            poppler=True,
        ),
        _manager_run(
            manager,
            env_name,
            "uv",
            "pip",
            "install",
            "huggingface-hub==1.17.0",
            "pdf2image==1.17.0",
            "pillow==12.2.0",
            "tqdm==4.67.3",
            "transformers==5.9.0",
        ),
        _manager_run(
            manager,
            env_name,
            "uv",
            "pip",
            "install",
            "torch==2.11.0",
            "torchvision==0.26.0",
            "torchaudio==2.11.0+cu130",
            "--index-url",
            pytorch_index,
        ),
    ]
    vllm_command = _manager_run(
        manager,
        env_name,
        "uv",
        "pip",
        "install",
    )
    vllm_command.append("vllm==0.21.1rc1.dev438+g4ff865c38")
    vllm_command.extend(
        ["--extra-index-url", DEFAULT_CUDA_130_VLLM_WHEEL_INDEX]
    )
    commands.append(vllm_command)
    install_target = editable_root or package_spec
    install_command = _manager_run(
        manager,
        env_name,
        "python",
        "-m",
        "pip",
        "install",
    )
    if editable_root:
        install_command.append("-e")
    install_command.append(install_target)
    commands.append(install_command)
    return commands


def _mineru_environment_commands(
    *,
    manager: str,
    env_name: str,
    python_version: str,
) -> list[list[str]]:
    pytorch_index = _PYTORCH_WHEEL_INDEX.format(backend="cu130")
    return [
        _base_create_command(
            manager=manager,
            env_name=env_name,
            python_version=python_version,
            poppler=False,
        ),
        _manager_run(
            manager,
            env_name,
            "uv",
            "pip",
            "install",
            "torch==2.11.0+cu130",
            "torchvision==0.26.0+cu130",
            "--index-url",
            pytorch_index,
        ),
        _manager_run(
            manager,
            env_name,
            "uv",
            "pip",
            "install",
            "accelerate==1.14.0",
            "huggingface-hub==0.36.2",
            "mineru-vl-utils==1.0.5",
            "pillow==12.2.0",
            "transformers==4.57.6",
        ),
    ]


def environment_plan_receipt(
    backend: OcrBackendDefinition,
    *,
    manager: str,
    env_name: str,
    python_version: str,
    torch_backend: str,
    commands: Sequence[Sequence[str]],
    dry_run: bool,
) -> dict[str, Any]:
    command_plan = [list(command) for command in commands]
    return make_receipt(
        "ocr_backend_environment",
        {
            "backend_id": backend.backend_id,
            "status": "planned" if dry_run else "completed",
            "manager": manager,
            "environment": {
                "name": env_name,
                "python": python_version,
                "torch_backend": torch_backend,
            },
            "model": {
                "requested": backend.model_id,
                "revision": backend.model_revision,
            },
            "workflow": {
                "standalone_adapter": (
                    backend.execution_mode == "provider_process"
                ),
                "model_fetch_separate": True,
                "provider_spec_generation_separate": (
                    backend.execution_mode == "provider_process"
                ),
            },
            "command_plan": command_plan,
            "command_plan_sha256": sha256_json(command_plan),
        },
    )


_FETCH_CODE = (
    "import sys\n"
    "from huggingface_hub import snapshot_download\n"
    "model, revision, cache_dir = sys.argv[1:4]\n"
    "print(snapshot_download(repo_id=model, revision=revision, "
    "cache_dir=cache_dir))\n"
)


def build_model_fetch_command(
    backend: OcrBackendDefinition,
    *,
    manager: str | None,
    env_name: str | None,
    cache_dir: str | os.PathLike[str],
) -> list[str]:
    """Build an exact-revision Hugging Face snapshot command."""

    cache_path = _absolute_path(cache_dir, "model cache")
    command = [
        "python",
        "-c",
        _FETCH_CODE,
        backend.model_id,
        backend.model_revision,
        str(cache_path),
    ]
    if env_name:
        command = _manager_run(resolve_manager(manager), env_name, *command)
    assert_command_plan_safe([command])
    return command


def expected_snapshot_path(
    backend: OcrBackendDefinition,
    cache_dir: str | os.PathLike[str],
) -> Path:
    cache_path = _absolute_path(cache_dir, "model cache")
    repo_cache_name = "models--" + backend.model_id.replace("/", "--")
    return cache_path / repo_cache_name / "snapshots" / backend.model_revision


def model_fetch_receipt(
    backend: OcrBackendDefinition,
    *,
    cache_dir: str | os.PathLike[str],
    command: Sequence[str],
    dry_run: bool,
) -> dict[str, Any]:
    snapshot = expected_snapshot_path(backend, cache_dir)
    if not dry_run and not snapshot.is_dir():
        raise BackendSetupError(
            "model fetch completed without the pinned snapshot at "
            f"{snapshot}"
        )
    return make_receipt(
        "ocr_backend_model",
        {
            "backend_id": backend.backend_id,
            "status": "planned" if dry_run else "completed",
            "model": {
                "requested": backend.model_id,
                "revision": backend.model_revision,
                "snapshot_path": str(snapshot),
                "snapshot_present": snapshot.is_dir(),
            },
            "cache_dir": str(_absolute_path(cache_dir, "model cache")),
            "qualification_artifacts": {
                "snapshot_identity_sha256": (
                    backend.snapshot_identity_sha256
                ),
                "weights_sha256": backend.weights_sha256,
            },
            "command": list(command),
            "command_sha256": sha256_json(list(command)),
        },
    )


def _absolute_path(
    path: str | os.PathLike[str],
    label: str,
) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        raise BackendSetupError(f"{label} must be an absolute path")
    return candidate.resolve()


def _require_executable(path: str | os.PathLike[str]) -> Path:
    candidate = _absolute_path(path, "provider Python executable")
    try:
        mode = candidate.stat().st_mode
    except OSError as exc:
        raise BackendSetupError(
            f"provider Python executable does not exist: {candidate}"
        ) from exc
    if not stat.S_ISREG(mode) or not os.access(candidate, os.X_OK):
        raise BackendSetupError(
            f"provider Python path is not executable: {candidate}"
        )
    return candidate


def _adapter_path(backend: OcrBackendDefinition) -> Path:
    if not backend.adapter_relative_path:
        raise BackendSetupError(
            f"backend {backend.backend_id} has no provider adapter"
        )
    candidate = (
        Path(__file__).resolve().parent / backend.adapter_relative_path
    ).resolve()
    if not candidate.is_file():
        raise BackendSetupError(
            f"provider adapter does not exist: {candidate}"
        )
    return candidate


def build_provider_spec(
    backend: OcrBackendDefinition,
    *,
    python_executable: str | os.PathLike[str],
    cache_dir: str | os.PathLike[str],
    device: str | None = None,
    attention_backend: str | None = None,
    cuda_visible_devices: str = "0",
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    """Build a strict local-only provider spec for a built-in backend."""

    if backend.execution_mode != "provider_process":
        raise BackendSetupError(
            f"backend {backend.backend_id} runs in process and has no spec"
        )
    python_path = _require_executable(python_executable)
    cache_path = _absolute_path(cache_dir, "model cache")
    adapter_path = _adapter_path(backend)
    adapter_digest = file_sha256(adapter_path)
    if (
        backend.qualified_adapter_sha256
        and adapter_digest != backend.qualified_adapter_sha256
    ):
        raise BackendSetupError(
            "installed provider adapter does not match the qualified "
            f"SHA-256 for {backend.backend_id}"
        )
    timeout = timeout_seconds or backend.default_timeout_seconds
    if timeout <= 0:
        raise BackendSetupError("provider timeout must be positive")
    if not cuda_visible_devices:
        raise BackendSetupError(
            "CUDA visible-device selection cannot be empty"
        )
    device_value = device or backend.default_device
    attention_value = (
        attention_backend or backend.default_attention_backend
    )
    return {
        "schema_version": 1,
        "provider_id": backend.provider_id,
        "argv": [
            str(python_path),
            str(adapter_path),
            "--model-id",
            backend.model_id,
            "--revision",
            backend.model_revision,
            "--device",
            device_value,
            "--attention-backend",
            attention_value,
        ],
        "egress": "none",
        "model": {
            "requested": backend.model_id,
            "revision": backend.model_revision,
        },
        "lanes": {
            backend.default_lane: {
                "prompt": backend.provider_prompt,
                "settings": dict(backend.provider_settings),
                "include_raw_text": False,
            }
        },
        "timeout_seconds": timeout,
        "secret_env_names": [],
        "env": {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
            "HF_HUB_CACHE": str(cache_path),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
    }


def _atomic_write_new_bytes(
    path: Path,
    payload: bytes,
    *,
    validate: Callable[[Path], None] | None = None,
) -> None:
    """Publish complete bytes without replacing an existing path."""

    if not path.is_absolute():
        raise BackendSetupError("output path must be absolute")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".candidate",
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if validate:
            validate(temporary_path)
        try:
            os.link(temporary_path, path)
        except FileExistsError as exc:
            raise BackendSetupError(
                f"refusing to replace existing file: {path}"
            ) from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def write_provider_spec(
    backend: OcrBackendDefinition,
    *,
    output_path: str | os.PathLike[str],
    python_executable: str | os.PathLike[str],
    cache_dir: str | os.PathLike[str],
    device: str | None = None,
    attention_backend: str | None = None,
    cuda_visible_devices: str = "0",
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    """Validate and atomically write a local provider specification."""

    from .ocr_provider import load_local_provider_spec

    destination = _absolute_path(output_path, "provider spec output")
    spec = build_provider_spec(
        backend,
        python_executable=python_executable,
        cache_dir=cache_dir,
        device=device,
        attention_backend=attention_backend,
        cuda_visible_devices=cuda_visible_devices,
        timeout_seconds=timeout_seconds,
    )
    payload = json_bytes(spec, pretty=True)
    _atomic_write_new_bytes(
        destination,
        payload,
        validate=lambda candidate: load_local_provider_spec(candidate),
    )
    return make_receipt(
        "ocr_backend_provider_spec",
        {
            "backend_id": backend.backend_id,
            "status": "completed",
            "output_path": str(destination),
            "spec_sha256": hashlib.sha256(payload).hexdigest(),
            "adapter_sha256": file_sha256(_adapter_path(backend)),
            "qualified_adapter_sha256": (
                backend.qualified_adapter_sha256
            ),
            "model": {
                "requested": backend.model_id,
                "revision": backend.model_revision,
            },
            "qualification_artifacts": {
                "snapshot_identity_sha256": (
                    backend.snapshot_identity_sha256
                ),
                "weights_sha256": backend.weights_sha256,
            },
        },
    )


def _check(
    name: str,
    passed: bool,
    detail: str,
) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _argv_option(argv: Sequence[str], name: str) -> str | None:
    try:
        index = argv.index(name)
    except ValueError:
        return None
    if index + 1 >= len(argv):
        return None
    return argv[index + 1]


_PACKAGE_PROBE_CODE = (
    "import json, sys\n"
    "from importlib.metadata import PackageNotFoundError, version\n"
    "result = {}\n"
    "for name in sys.argv[1:]:\n"
    "    try:\n"
    "        result[name] = version(name)\n"
    "    except PackageNotFoundError:\n"
    "        result[name] = None\n"
    "print(json.dumps(result, sort_keys=True))\n"
)

_MODEL_PROBE_CODE = (
    "import json, sys\n"
    "snapshot, device, attention = sys.argv[1:4]\n"
    "try:\n"
    "    import torch\n"
    "    from mineru_vl_utils import MinerUClient\n"
    "    from transformers import (\n"
    "        AutoProcessor, Qwen2VLForConditionalGeneration,\n"
    "    )\n"
    "    if not torch.cuda.is_available():\n"
    "        raise RuntimeError('CUDA is unavailable')\n"
    "    model = Qwen2VLForConditionalGeneration.from_pretrained(\n"
    "        snapshot, dtype='bfloat16', device_map=device,\n"
    "        attn_implementation=attention, local_files_only=True,\n"
    "    )\n"
    "    processor = AutoProcessor.from_pretrained(\n"
    "        snapshot, use_fast=True, local_files_only=True,\n"
    "    )\n"
    "    client = MinerUClient(\n"
    "        backend='transformers', model=model, processor=processor,\n"
    "        image_analysis=False, batch_size=1, use_tqdm=False,\n"
    "    )\n"
    "    result = {\n"
    "        'ok': True,\n"
    "        'model_class': type(model).__name__,\n"
    "        'processor_class': type(processor).__name__,\n"
    "        'client_class': type(client).__name__,\n"
    "        'dtype': 'bfloat16',\n"
    "        'device': device,\n"
    "        'attention_backend': attention,\n"
    "        'backend': 'transformers',\n"
    "        'image_analysis': False,\n"
    "        'batch_size': 1,\n"
    "    }\n"
    "except BaseException as exc:\n"
    "    result = {'ok': False, 'error_type': type(exc).__name__}\n"
    "print(json.dumps(result, sort_keys=True))\n"
    "raise SystemExit(0 if result['ok'] else 1)\n"
)

_MAX_PROBE_OUTPUT_BYTES = 64 * 1024


def _run_bounded_probe(
    argv: Sequence[str],
    *,
    cwd: str | None,
    environment: Mapping[str, str],
    timeout_seconds: float,
) -> dict[str, Any]:
    """Run a probe with secret-scrubbed env and bounded retained output."""

    with (
        tempfile.TemporaryFile() as stdout,
        tempfile.TemporaryFile() as stderr,
    ):
        try:
            completed = subprocess.run(
                list(argv),
                cwd=cwd,
                env=dict(environment),
                check=False,
                stdout=stdout,
                stderr=stderr,
                timeout=timeout_seconds,
            )
            returncode = completed.returncode
            error_type = None
        except subprocess.TimeoutExpired:
            returncode = None
            error_type = "TimeoutExpired"
        except OSError as exc:
            returncode = None
            error_type = type(exc).__name__
        stdout.seek(0)
        stdout_bytes = stdout.read(_MAX_PROBE_OUTPUT_BYTES + 1)
        stderr.seek(0)
        stderr_bytes = stderr.read(_MAX_PROBE_OUTPUT_BYTES + 1)
    return {
        "returncode": returncode,
        "error_type": error_type,
        "stdout": stdout_bytes[:_MAX_PROBE_OUTPUT_BYTES],
        "stderr": stderr_bytes[:_MAX_PROBE_OUTPUT_BYTES],
        "stdout_truncated": len(stdout_bytes) > _MAX_PROBE_OUTPUT_BYTES,
        "stderr_truncated": len(stderr_bytes) > _MAX_PROBE_OUTPUT_BYTES,
    }


def _probe_runtime_packages(
    spec: Mapping[str, Any],
    backend: OcrBackendDefinition,
) -> tuple[dict[str, str | None], str | None]:
    from .ocr_provider import build_local_provider_environment

    names = [name for name, _ in backend.runtime_packages]
    if not names:
        return {}, None
    completed = _run_bounded_probe(
        [spec["argv"][0], "-c", _PACKAGE_PROBE_CODE, *names],
        cwd=spec.get("cwd"),
        environment=build_local_provider_environment(spec),
        timeout_seconds=120,
    )
    if completed["error_type"]:
        return {}, completed["error_type"]
    if completed["returncode"] != 0:
        return {}, (
            "provider interpreter package probe exited with status "
            f"{completed['returncode']}"
        )
    try:
        value = json.loads(completed["stdout"])
    except (UnicodeDecodeError, ValueError):
        return {}, "provider interpreter returned invalid package JSON"
    if not isinstance(value, dict) or any(
        not isinstance(key, str)
        or (item is not None and not isinstance(item, str))
        for key, item in value.items()
    ):
        return {}, "provider interpreter returned invalid package data"
    return value, None


def _probe_model_initialization(
    spec: Mapping[str, Any],
    snapshot: Path,
) -> dict[str, Any]:
    from .ocr_provider import build_local_provider_environment

    device = _argv_option(spec["argv"], "--device")
    attention = _argv_option(spec["argv"], "--attention-backend")
    if not device or not attention:
        return {
            "passed": False,
            "error_type": "InvalidProviderArguments",
            "result": None,
        }
    completed = _run_bounded_probe(
        [
            spec["argv"][0],
            "-c",
            _MODEL_PROBE_CODE,
            str(snapshot),
            device,
            attention,
        ],
        cwd=spec.get("cwd"),
        environment=build_local_provider_environment(spec),
        timeout_seconds=min(float(spec.get("timeout_seconds", 900)), 900.0),
    )
    result = None
    error_type = completed["error_type"]
    try:
        decoded = json.loads(completed["stdout"])
        if isinstance(decoded, dict):
            result = decoded
            if not decoded.get("ok"):
                candidate = decoded.get("error_type")
                if isinstance(candidate, str) and candidate:
                    error_type = candidate
    except (UnicodeDecodeError, ValueError):
        if error_type is None:
            error_type = "InvalidProbeOutput"
    expected = {
        "ok": True,
        "model_class": "Qwen2VLForConditionalGeneration",
        "client_class": "MinerUClient",
        "dtype": "bfloat16",
        "device": device,
        "attention_backend": attention,
        "backend": "transformers",
        "image_analysis": False,
        "batch_size": 1,
    }
    passed = (
        completed["returncode"] == 0
        and error_type is None
        and isinstance(result, dict)
        and all(result.get(key) == value for key, value in expected.items())
    )
    if not passed and error_type is None:
        error_type = "ProbeContractMismatch"
    safe_result = None
    if isinstance(result, dict):
        allowed = {
            "ok",
            "model_class",
            "processor_class",
            "client_class",
            "dtype",
            "device",
            "attention_backend",
            "backend",
            "image_analysis",
            "batch_size",
        }
        safe_result = {
            key: value for key, value in result.items() if key in allowed
        }
    return {
        "passed": passed,
        "error_type": error_type,
        "result": safe_result,
        "output_truncated": bool(
            completed["stdout_truncated"]
            or completed["stderr_truncated"]
        ),
    }


def _snapshot_integrity(
    snapshot: Path,
    backend: OcrBackendDefinition,
) -> dict[str, Any]:
    if not snapshot.is_dir():
        return {
            "files_sha256": None,
            "weights_sha256": None,
            "error_type": "SnapshotMissing",
        }
    try:
        from .providers.mineru25 import snapshot_identity

        identity = snapshot_identity(snapshot, backend.model_revision)
        weights = identity["files"].get("model.safetensors", {})
        return {
            "files_sha256": identity["files_sha256"],
            "weights_sha256": weights.get("sha256"),
            "error_type": None,
        }
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return {
            "files_sha256": None,
            "weights_sha256": None,
            "error_type": type(exc).__name__,
        }


def doctor_provider_spec(
    spec_path: str | os.PathLike[str],
    *,
    load_model: bool = False,
) -> dict[str, Any]:
    """Qualify a built-in local provider and optionally load its model."""

    from .ocr_backends import get_backend_for_provider
    from .ocr_provider import load_local_provider_spec

    source = Path(spec_path).expanduser().resolve(strict=True)
    spec = load_local_provider_spec(source)
    backend = get_backend_for_provider(spec["provider_id"])
    expected_adapter = _adapter_path(backend)
    adapter_digest = file_sha256(expected_adapter)
    expected_snapshot = expected_snapshot_path(
        backend,
        spec.get("env", {}).get("HF_HUB_CACHE", ""),
    )
    snapshot_integrity = _snapshot_integrity(expected_snapshot, backend)
    checks = [
        _check(
            "model_id",
            spec["model"]["requested"] == backend.model_id,
            "provider model matches the built-in backend pin",
        ),
        _check(
            "model_revision",
            spec["model"]["revision"] == backend.model_revision,
            "provider revision is the immutable qualified revision",
        ),
        _check(
            "adapter_path",
            len(spec["argv"]) > 1
            and Path(spec["argv"][1]).resolve() == expected_adapter,
            "provider uses the installed built-in adapter",
        ),
        _check(
            "adapter_sha256",
            adapter_digest == backend.qualified_adapter_sha256,
            "installed adapter matches the qualified SHA-256",
        ),
        _check(
            "argv_model_id",
            _argv_option(spec["argv"], "--model-id") == backend.model_id,
            "adapter argv carries the pinned model ID",
        ),
        _check(
            "argv_revision",
            _argv_option(spec["argv"], "--revision")
            == backend.model_revision,
            "adapter argv carries the pinned model revision",
        ),
        _check(
            "offline",
            spec.get("env", {}).get("HF_HUB_OFFLINE") == "1"
            and spec.get("env", {}).get("TRANSFORMERS_OFFLINE") == "1",
            "provider is configured for offline model access",
        ),
        _check(
            "snapshot_present",
            expected_snapshot.is_dir(),
            f"pinned snapshot exists at {expected_snapshot}",
        ),
        _check(
            "snapshot_identity_sha256",
            snapshot_integrity["files_sha256"]
            == backend.snapshot_identity_sha256,
            (
                "complete snapshot-file identity matches qualification"
                if snapshot_integrity["error_type"] is None
                else "snapshot identity could not be computed: "
                + snapshot_integrity["error_type"]
            ),
        ),
        _check(
            "model_weights_sha256",
            snapshot_integrity["weights_sha256"]
            == backend.weights_sha256,
            "model.safetensors matches the qualified SHA-256",
        ),
    ]
    packages, package_error = _probe_runtime_packages(spec, backend)
    if package_error:
        checks.append(_check("runtime_packages", False, package_error))
    else:
        expected_packages = dict(backend.runtime_packages)
        mismatches = {
            name: {
                "expected": expected,
                "actual": packages.get(name),
            }
            for name, expected in expected_packages.items()
            if packages.get(name) != expected
        }
        checks.append(
            _check(
                "runtime_packages",
                not mismatches,
                (
                    "runtime package versions match the qualified tuple"
                    if not mismatches
                    else "runtime package tuple does not match"
                ),
            )
        )
    model_probe = {
        "requested": bool(load_model),
        "passed": None,
        "error_type": None,
        "result": None,
        "output_truncated": False,
    }
    if load_model:
        model_probe.update(
            _probe_model_initialization(spec, expected_snapshot)
        )
        checks.append(
            _check(
                "model_initialization",
                bool(model_probe["passed"]),
                (
                    "qualified model and MinerU client initialized"
                    if model_probe["passed"]
                    else "model initialization probe failed with "
                    + str(model_probe["error_type"])
                ),
            )
        )
    all_pass = all(item["passed"] for item in checks)
    return make_receipt(
        "ocr_backend_doctor",
        {
            "backend_id": backend.backend_id,
            "status": "pass" if all_pass else "fail",
            "all_pass": all_pass,
            "spec_path": str(source),
            "spec_sha256": file_sha256(source),
            "adapter_sha256": adapter_digest,
            "qualified_adapter_sha256": (
                backend.qualified_adapter_sha256
            ),
            "model": {
                "requested": backend.model_id,
                "revision": backend.model_revision,
                "snapshot_path": str(expected_snapshot),
            },
            "runtime_packages": packages,
            "qualification_artifacts": {
                "expected_snapshot_identity_sha256": (
                    backend.snapshot_identity_sha256
                ),
                "observed_snapshot_identity_sha256": (
                    snapshot_integrity["files_sha256"]
                ),
                "expected_weights_sha256": backend.weights_sha256,
                "observed_weights_sha256": (
                    snapshot_integrity["weights_sha256"]
                ),
            },
            "model_probe": model_probe,
            "checks": checks,
        },
    )
