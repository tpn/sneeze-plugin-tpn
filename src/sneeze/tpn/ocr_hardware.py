"""Lightweight NVIDIA hardware discovery and OCR memory budgeting.

This module is safe to import during CLI discovery.  It deliberately imports
only the Python standard library and, when requested, probes PyTorch in a
separate isolated interpreter process.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import selectors
import shutil
import signal
import socket
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .ocr_memory import canonical_memory_budget
from .ocr_path_security import TrustedPathError, validate_trusted_file
from .ocr_process_lifecycle import isolated_subprocess_signal_guard

SCHEMA_VERSION = 1
GIB = 1024**3
MEMORY_TIER_IDS = ("25", "50", "max")
_MAX_PROBE_STREAM_BYTES = 64 * 1024
_PROBE_READ_BYTES = 8 * 1024
_SUBPROCESS_RUNNER = subprocess.run

_NVIDIA_QUERY_FIELDS = (
    "index",
    "uuid",
    "pci.bus_id",
    "pci.device_id",
    "name",
    "compute_cap",
    "memory.total",
    "memory.free",
    "driver_version",
)
_SECRET_FIELD_RE = re.compile(
    r"(?:^|_)(?:api_?key|auth|bearer|cookie|credential|password|"
    r"key|private|secret|token)(?:_|$)",
    re.IGNORECASE,
)
_SECRET_VALUE_RES = (
    re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\b(?:gh[pousr]_|github_pat_)[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bBearer\s+\S+", re.IGNORECASE),
    re.compile(r"-----BEGIN [A-Z ]+ PRIVATE KEY-----"),
    re.compile(r"https?://[^/@\s:]+:[^/@\s]+@", re.IGNORECASE),
    re.compile(
        r"(?:api[_-]?key|authorization|password|secret|token)="
        r"[^&\s]+",
        re.IGNORECASE,
    ),
)
_SAFE_PROBE_ENV_NAMES = frozenset(
    {
        "CUDA_DEVICE_ORDER",
        "CUDA_VISIBLE_DEVICES",
        "LANG",
        "LANGUAGE",
        "LC_ALL",
        "LC_CTYPE",
        "LD_LIBRARY_PATH",
        "NVIDIA_VISIBLE_DEVICES",
        "PATH",
        "TZ",
    }
)
_TORCH_PROBE_CODE = r"""
import json
import os
import platform

try:
    import torch

    count = torch.cuda.device_count()
    if not torch.cuda.is_available() or count != 1:
        raise RuntimeError("expected exactly one visible CUDA device")
    properties = torch.cuda.get_device_properties(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    raw_uuid = getattr(properties, "uuid", None)
    if isinstance(raw_uuid, bytes):
        raw_uuid = raw_uuid.hex()
    elif raw_uuid is not None:
        raw_uuid = str(raw_uuid)
    integrated = getattr(
        properties,
        "is_integrated",
        getattr(properties, "integrated", None),
    )
    if integrated is not None:
        integrated = bool(integrated)
    result = {
        "ok": True,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "visible_device_count": count,
        "device": {
            "name": str(properties.name),
            "uuid": raw_uuid,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": int(total_bytes),
            "free_memory_bytes": int(free_bytes),
            "integrated": integrated,
        },
    }
except BaseException as exc:
    result = {
        "ok": False,
        "error_type": type(exc).__name__,
    }

print(json.dumps(result, sort_keys=True, separators=(",", ":")))
raise SystemExit(0 if result["ok"] else 1)
""".strip()


class HardwareProbeError(ValueError):
    """Raised when hardware identity cannot be resolved unambiguously."""


@dataclass(frozen=True)
class MemoryBudgetPolicy:
    """Capacity fractions and safety reserves for OCR execution tiers."""

    quarter_fraction: float = 0.25
    half_fraction: float = 0.50
    dedicated_max_fraction: float = 0.90
    unified_max_fraction: float = 0.75
    dedicated_reserve_bytes: int = 2 * GIB
    unified_reserve_bytes: int = 16 * GIB

    def __post_init__(self) -> None:
        fractions = (
            self.quarter_fraction,
            self.half_fraction,
            self.dedicated_max_fraction,
            self.unified_max_fraction,
        )
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not 0 < value < 1
            for value in fractions
        ):
            raise HardwareProbeError(
                "memory budget fractions must be numbers between zero and one"
            )
        if not self.quarter_fraction < self.half_fraction:
            raise HardwareProbeError(
                "quarter_fraction must be less than half_fraction"
            )
        if (
            min(
                self.dedicated_max_fraction,
                self.unified_max_fraction,
            )
            < self.half_fraction
        ):
            raise HardwareProbeError(
                "max fractions cannot be smaller than half_fraction"
            )
        reserves = (
            self.dedicated_reserve_bytes,
            self.unified_reserve_bytes,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in reserves
        ):
            raise HardwareProbeError(
                "memory safety reserves must be nonnegative integers"
            )


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def content_sha256(value: Any) -> str:
    """Return the SHA-256 of canonical JSON content."""

    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _assert_secret_free(value: Any, path: str = "receipt") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise HardwareProbeError(
                    f"{path} contains a non-string JSON field"
                )
            if _SECRET_FIELD_RE.search(key):
                raise HardwareProbeError(
                    f"refusing secret-bearing receipt field: {path}.{key}"
                )
            _assert_secret_free(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_secret_free(item, f"{path}[{index}]")
    elif isinstance(value, str) and any(
        pattern.search(value) for pattern in _SECRET_VALUE_RES
    ):
        raise HardwareProbeError(f"refusing credential-like value in {path}")


def _make_receipt(kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        **dict(payload),
    }
    _assert_secret_free(receipt)
    receipt["receipt_sha256"] = content_sha256(receipt)
    return receipt


def _normalize_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.strip().split())
    return normalized or None


def _normalize_pci_device_id(value: str | None) -> str | None:
    normalized = _normalize_text(value)
    if normalized is None or normalized.casefold() in {"n/a", "[n/a]"}:
        return None
    match = re.fullmatch(r"(?:0x)?([0-9a-fA-F]{4,16})", normalized)
    if not match:
        return normalized.casefold()
    return "0x" + match.group(1).lower()


def _parse_compute_capability(value: str | None) -> list[int] | None:
    match = re.fullmatch(r"\s*(\d+)(?:\.(\d+))?\s*", value or "")
    if not match:
        return None
    return [int(match.group(1)), int(match.group(2) or "0")]


def _parse_mib(value: str | None) -> int | None:
    normalized = _normalize_text(value)
    if normalized is None or normalized.casefold() in {"n/a", "[n/a]"}:
        return None
    match = re.fullmatch(
        r"([0-9]+(?:\.[0-9]+)?)\s*(?:mib)?",
        normalized,
        re.IGNORECASE,
    )
    if not match:
        return None
    return int(float(match.group(1)) * 1024 * 1024)


def _safe_environment(source: Mapping[str, str]) -> dict[str, str]:
    environment = {
        name: value
        for name, value in source.items()
        if name in _SAFE_PROBE_ENV_NAMES
        and isinstance(value, str)
        and "\x00" not in value
    }
    environment["PYTHONNOUSERSITE"] = "1"
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    return environment


def _resolve_executable(value: str, label: str) -> str:
    candidate = shutil.which(value)
    if candidate is None:
        path = Path(value).expanduser()
        if path.is_absolute():
            candidate = str(path)
    if candidate is None:
        raise HardwareProbeError(f"could not find {label}: {value}")
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        path = Path(os.path.abspath(path))
    try:
        path = validate_trusted_file(
            path,
            label=label,
            executable=True,
        )
    except TrustedPathError as exc:
        raise HardwareProbeError(str(exc)) from exc
    return str(path)


def _run_bounded(
    argv: list[str],
    *,
    environment: Mapping[str, str],
    timeout_seconds: float,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> subprocess.CompletedProcess[str]:
    # Unit tests inject an already-completed runner.  Real probe execution
    # always takes the streaming Popen path below; an injected runner has
    # already decided how its output is collected before returning here.
    if runner is not _SUBPROCESS_RUNNER:
        try:
            completed = runner(
                argv,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=dict(environment),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise HardwareProbeError(
                f"hardware probe command failed: {type(exc).__name__}"
            ) from exc
        if (
            len(completed.stdout.encode("utf-8", errors="replace"))
            > _MAX_PROBE_STREAM_BYTES
        ):
            raise HardwareProbeError("hardware probe stdout exceeded 64 KiB")
        if (
            len(completed.stderr.encode("utf-8", errors="replace"))
            > _MAX_PROBE_STREAM_BYTES
        ):
            raise HardwareProbeError("hardware probe stderr exceeded 64 KiB")
        return completed

    with isolated_subprocess_signal_guard() as lifecycle:
        return _run_bounded_guarded(
            argv,
            environment=environment,
            timeout_seconds=timeout_seconds,
            lifecycle=lifecycle,
        )


def _run_bounded_guarded(
    argv: list[str],
    *,
    environment: Mapping[str, str],
    timeout_seconds: float,
    lifecycle: Any,
) -> subprocess.CompletedProcess[str]:
    process: subprocess.Popen[bytes] | None = None
    streams: dict[str, Any] = {}
    chunks: dict[str, list[bytes]] = {"stdout": [], "stderr": []}
    sizes = {"stdout": 0, "stderr": 0}
    selector = selectors.DefaultSelector()

    def stop_process() -> None:
        if process is None:
            return
        # The direct child may have exited after spawning a descendant that
        # still holds a pipe.  Kill the private process group even when poll()
        # already reports the direct child as complete.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            if process.poll() is None:
                try:
                    process.kill()
                except OSError:
                    pass
        try:
            process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass

    try:
        try:
            with lifecycle.spawn_window():
                process = subprocess.Popen(
                    argv,
                    env=dict(environment),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=True,
                )
        except OSError as exc:
            raise HardwareProbeError(
                f"hardware probe command failed: {type(exc).__name__}"
            ) from exc

        streams = {
            "stdout": process.stdout,
            "stderr": process.stderr,
        }
        if any(  # pragma: no cover - Popen PIPE invariant
            stream is None for stream in streams.values()
        ):
            raise HardwareProbeError("hardware probe pipes were not created")
        for name, stream in streams.items():
            assert stream is not None
            selector.register(stream, selectors.EVENT_READ, name)
        deadline = time.monotonic() + timeout_seconds
        while selector.get_map():
            lifecycle.checkpoint()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(argv, timeout_seconds)
            for key, _ in selector.select(timeout=min(remaining, 0.1)):
                name = key.data
                allowance = _MAX_PROBE_STREAM_BYTES - sizes[name]
                chunk = os.read(
                    key.fd,
                    min(_PROBE_READ_BYTES, allowance + 1),
                )
                if not chunk:
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
                    continue
                if len(chunk) > allowance:
                    raise HardwareProbeError(
                        f"hardware probe {name} exceeded 64 KiB"
                    )
                chunks[name].append(chunk)
                sizes[name] += len(chunk)

        remaining = deadline - time.monotonic()
        if process.poll() is None and remaining <= 0:
            raise subprocess.TimeoutExpired(argv, timeout_seconds)
        returncode = lifecycle.wait(
            process,
            timeout_seconds=max(0, remaining),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise HardwareProbeError(
            f"hardware probe command failed: {type(exc).__name__}"
        ) from exc
    finally:
        # Keep passive handlers installed through unconditional private-group
        # teardown so a signal cannot interrupt escalation or reaping.
        with lifecycle.cleanup_window():
            stop_process()
            selector.close()
            for stream in streams.values():
                if stream is not None and not stream.closed:
                    stream.close()

    return subprocess.CompletedProcess(
        argv,
        returncode,
        stdout=b"".join(chunks["stdout"]).decode("utf-8", errors="replace"),
        stderr=b"".join(chunks["stderr"]).decode("utf-8", errors="replace"),
    )


def _parse_nvidia_rows(output: str) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(output.splitlines(), 1):
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != len(_NVIDIA_QUERY_FIELDS):
            raise HardwareProbeError(
                "nvidia-smi returned an unexpected CSV row at line "
                f"{line_number}"
            )
        try:
            index = int(parts[0])
        except ValueError as exc:
            raise HardwareProbeError(
                "nvidia-smi returned an invalid GPU index at line "
                f"{line_number}"
            ) from exc
        uuid = _normalize_text(parts[1])
        pci_bus_id = _normalize_text(parts[2])
        name = _normalize_text(parts[4])
        if uuid is None or pci_bus_id is None or name is None:
            raise HardwareProbeError(
                f"nvidia-smi omitted GPU identity at line {line_number}"
            )
        rows.append(
            {
                "index": index,
                "uuid": uuid.upper(),
                "pci_bus_id": pci_bus_id.lower(),
                "pci_device_id": _normalize_pci_device_id(parts[3]),
                "product_name": name,
                "compute_capability": _parse_compute_capability(parts[5]),
                "total_memory_bytes": _parse_mib(parts[6]),
                "free_memory_bytes": _parse_mib(parts[7]),
                "driver_version": _normalize_text(parts[8]),
                "nvml_memory_available": (
                    _parse_mib(parts[6]) is not None
                    and _parse_mib(parts[7]) is not None
                ),
            }
        )
    if not rows:
        raise HardwareProbeError("nvidia-smi reported no NVIDIA GPUs")
    if len({row["index"] for row in rows}) != len(rows):
        raise HardwareProbeError("nvidia-smi returned duplicate GPU indexes")
    if len({row["uuid"] for row in rows}) != len(rows):
        raise HardwareProbeError("nvidia-smi returned duplicate GPU UUIDs")
    return rows


def _select_visible_device(
    rows: list[dict[str, Any]],
    *,
    visible_devices: str | None,
    cuda_device_order: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if visible_devices is None or not visible_devices.strip():
        if len(rows) != 1:
            raise HardwareProbeError(
                "multiple NVIDIA GPUs are visible; set CUDA_VISIBLE_DEVICES "
                "to exactly one GPU"
            )
        return rows[0], {
            "requested": None,
            "method": "sole_nvidia_device",
        }

    tokens = [token.strip() for token in visible_devices.split(",")]
    if len(tokens) != 1 or not tokens[0]:
        raise HardwareProbeError(
            "CUDA_VISIBLE_DEVICES must select exactly one NVIDIA GPU"
        )
    token = tokens[0]
    if token == "-1":
        raise HardwareProbeError("CUDA_VISIBLE_DEVICES disables all GPUs")
    if token.upper().startswith("MIG-"):
        raise HardwareProbeError(
            "MIG visible-device identities require a partition-specific "
            "hardware probe"
        )

    if token.isdigit():
        ordinal = int(token)
        if (
            len(rows) > 1
            and (cuda_device_order or "").upper() != "PCI_BUS_ID"
        ):
            raise HardwareProbeError(
                "numeric CUDA_VISIBLE_DEVICES is ambiguous with multiple "
                "GPUs unless CUDA_DEVICE_ORDER=PCI_BUS_ID"
            )
        ordered = (
            sorted(rows, key=lambda row: row["pci_bus_id"])
            if (cuda_device_order or "").upper() == "PCI_BUS_ID"
            else rows
        )
        if ordinal >= len(ordered):
            raise HardwareProbeError(
                "CUDA_VISIBLE_DEVICES numeric selection is out of range"
            )
        return ordered[ordinal], {
            "requested": token,
            "method": "cuda_pci_bus_ordinal",
        }

    normalized = token.upper()
    if normalized.startswith("GPU-"):
        matches = [row for row in rows if row["uuid"].startswith(normalized)]
        if len(matches) != 1:
            raise HardwareProbeError(
                "CUDA_VISIBLE_DEVICES GPU UUID selection did not resolve "
                "uniquely"
            )
        return matches[0], {
            "requested": token,
            "method": "cuda_gpu_uuid",
        }
    raise HardwareProbeError(
        "CUDA_VISIBLE_DEVICES uses an unsupported or unresolved selector"
    )


def _read_system_memory(path: str | os.PathLike[str]) -> dict[str, Any]:
    result = {
        "source": str(path),
        "total_memory_bytes": None,
        "available_memory_bytes": None,
        "status": "unavailable",
    }
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return result
    values = {}
    for line in text.splitlines():
        match = re.fullmatch(r"([^:]+):\s*(\d+)\s+kB\s*", line)
        if match:
            values[match.group(1)] = int(match.group(2)) * 1024
    result["total_memory_bytes"] = values.get("MemTotal")
    result["available_memory_bytes"] = values.get("MemAvailable")
    result["status"] = (
        "available"
        if result["total_memory_bytes"] is not None
        and result["available_memory_bytes"] is not None
        else "incomplete"
    )
    return result


def _read_cgroup_v2_memory(
    root: str | os.PathLike[str],
    membership_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Read the effective cgroup-v2 memory envelope without recording paths.

    A process cgroup can be unlimited while an intermediate ancestor carries
    the effective limit. Capacity and availability therefore use the most
    restrictive value observed from the process cgroup through the cgroup-v2
    root. ``current_bytes`` and ``peak_bytes`` remain diagnostics for the
    nearest readable cgroup.
    """

    result = {
        "status": "unavailable",
        "unlimited": None,
        "limit_bytes": None,
        "current_bytes": None,
        "peak_bytes": None,
        "available_bytes": None,
    }
    root_path = Path(root)
    relative: str | None = None
    try:
        for line in (
            Path(membership_path).read_text(encoding="utf-8").splitlines()
        ):
            if line.startswith("0::"):
                relative = line[3:].strip().lstrip("/")
                break
    except OSError:
        return result
    if relative is None:
        return result
    relative_path = Path(relative)
    if relative_path.is_absolute() or any(
        part in {".", ".."} for part in relative_path.parts
    ):
        return result
    directory = root_path.joinpath(*relative_path.parts)
    if directory != root_path and root_path not in directory.parents:
        return result

    directories = []
    while True:
        directories.append(directory)
        if directory == root_path:
            break
        directory = directory.parent
        if directory != root_path and root_path not in directory.parents:
            return result
    if not directories:
        return result

    def read_value(directory: Path, name: str) -> int | str | None:
        try:
            value = (directory / name).read_text(encoding="utf-8").strip()
        except OSError:
            return None
        if value == "max":
            return value
        try:
            number = int(value)
        except ValueError:
            return None
        return number if number >= 0 else None

    records = [
        {
            "limit": read_value(candidate, "memory.max"),
            "current": read_value(candidate, "memory.current"),
            "peak": read_value(candidate, "memory.peak"),
        }
        for candidate in directories
    ]
    records = [
        record
        for record in records
        if any(value is not None for value in record.values())
    ]
    if not records:
        return result
    finite_limits = [
        record["limit"]
        for record in records
        if isinstance(record["limit"], int)
    ]
    valid_limits = [
        record["limit"]
        for record in records
        if isinstance(record["limit"], int) or record["limit"] == "max"
    ]
    finite_records = [
        record for record in records if isinstance(record["limit"], int)
    ]
    available_values = (
        [
            max(0, record["limit"] - record["current"])
            for record in finite_records
        ]
        if finite_records
        and all(
            isinstance(record["current"], int)
            for record in finite_records
        )
        else []
    )
    nearest = records[0]
    return {
        "status": "available",
        "unlimited": (
            all(value == "max" for value in valid_limits)
            if valid_limits
            else None
        ),
        "limit_bytes": min(finite_limits) if finite_limits else None,
        "current_bytes": (
            nearest["current"]
            if isinstance(nearest["current"], int)
            else None
        ),
        "peak_bytes": (
            nearest["peak"]
            if isinstance(nearest["peak"], int)
            else None
        ),
        "available_bytes": (
            min(available_values) if available_values else None
        ),
    }


def _probe_torch(
    python_executable: str,
    *,
    environment: Mapping[str, str],
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> dict[str, Any]:
    executable = _resolve_executable(
        python_executable,
        "Torch probe Python interpreter",
    )
    argv = [executable, "-I", "-B", "-c", _TORCH_PROBE_CODE]
    completed = _run_bounded(
        argv,
        environment=environment,
        timeout_seconds=120,
        runner=runner,
    )
    stdout_bytes = completed.stdout.encode("utf-8", errors="replace")
    stderr_bytes = completed.stderr.encode("utf-8", errors="replace")
    result = None
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if lines:
        try:
            decoded = json.loads(lines[-1])
        except ValueError:
            decoded = None
        if isinstance(decoded, dict):
            result = decoded
    status = "available"
    error_type = None
    if (
        completed.returncode != 0
        or not result
        or result.get("ok") is not True
    ):
        status = "error"
        if isinstance(result, dict):
            candidate = result.get("error_type")
            if isinstance(candidate, str) and candidate:
                error_type = candidate
        error_type = error_type or "InvalidTorchProbeResult"
        result = None
    return {
        "status": status,
        "python_executable": executable,
        "returncode": completed.returncode,
        "error_type": error_type,
        "stdout_sha256": hashlib.sha256(stdout_bytes).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr_bytes).hexdigest(),
        "environment_sha256": content_sha256(
            dict(sorted(environment.items()))
        ),
        "result": result,
    }


def _validate_torch_identity(
    selected: Mapping[str, Any],
    torch_result: Mapping[str, Any],
) -> None:
    if torch_result.get("visible_device_count") != 1:
        raise HardwareProbeError(
            "Torch probe did not observe exactly one CUDA device"
        )
    device = torch_result.get("device")
    if not isinstance(device, Mapping):
        raise HardwareProbeError("Torch probe omitted CUDA device identity")
    selected_capability = selected.get("compute_capability")
    torch_capability = device.get("compute_capability")
    if (
        selected_capability is not None
        and torch_capability is not None
        and list(torch_capability) != list(selected_capability)
    ):
        raise HardwareProbeError(
            "Torch and nvidia-smi compute capabilities do not match"
        )
    torch_uuid = device.get("uuid")
    selected_uuid = selected.get("uuid")
    if (
        isinstance(torch_uuid, str)
        and torch_uuid.upper().startswith("GPU-")
        and torch_uuid.upper() != str(selected_uuid).upper()
    ):
        raise HardwareProbeError(
            "Torch and nvidia-smi GPU UUIDs do not match"
        )


def _effective_memory(
    selected: Mapping[str, Any],
    torch_probe: Mapping[str, Any] | None,
    system_memory: Mapping[str, Any],
    cgroup_memory: Mapping[str, Any],
) -> dict[str, Any]:
    torch_device = None
    if torch_probe and isinstance(torch_probe.get("result"), Mapping):
        candidate = torch_probe["result"].get("device")
        if isinstance(candidate, Mapping):
            torch_device = candidate

    nvml_total = selected.get("total_memory_bytes")
    nvml_free = selected.get("free_memory_bytes")
    torch_total = (
        torch_device.get("total_memory_bytes") if torch_device else None
    )
    torch_free = (
        torch_device.get("free_memory_bytes") if torch_device else None
    )
    integrated = torch_device.get("integrated") if torch_device else None
    if integrated is True:
        memory_model = "unified"
    elif integrated is False:
        memory_model = "dedicated"
    else:
        memory_model = "unknown"

    total_sources: dict[str, int] = {}
    free_sources: dict[str, int] = {}
    if isinstance(nvml_total, int) and nvml_total > 0:
        total_sources["nvidia_smi"] = nvml_total
    if isinstance(nvml_free, int) and nvml_free >= 0:
        free_sources["nvidia_smi"] = nvml_free
    if isinstance(torch_total, int) and torch_total > 0:
        total_sources["torch_cuda"] = torch_total
    if isinstance(torch_free, int) and torch_free >= 0:
        free_sources["torch_cuda"] = torch_free
    system_total = system_memory.get("total_memory_bytes")
    system_available = system_memory.get("available_memory_bytes")
    if memory_model == "unified":
        if isinstance(system_total, int) and system_total > 0:
            total_sources["system"] = system_total
        if isinstance(system_available, int) and system_available >= 0:
            free_sources["system"] = system_available
        cgroup_limit = cgroup_memory.get("limit_bytes")
        cgroup_available = cgroup_memory.get("available_bytes")
        finite_cgroup_limit = (
            isinstance(cgroup_limit, int) and cgroup_limit >= 0
        )
        if finite_cgroup_limit:
            total_sources["cgroup_v2"] = cgroup_limit
        if isinstance(cgroup_available, int) and cgroup_available >= 0:
            free_sources["cgroup_v2"] = cgroup_available
    else:
        finite_cgroup_limit = False

    if memory_model == "dedicated":
        total = total_sources.get("nvidia_smi")
        free = free_sources.get("nvidia_smi")
        if total is None:
            total = total_sources.get("torch_cuda")
        if free is None:
            free = free_sources.get("torch_cuda")
    elif memory_model == "unified":
        # Integrated CUDA may expose only a small currently mapped ATS window
        # through cudaMemGetInfo. Keep it as diagnostic telemetry, but derive
        # shared-memory capacity from system-level authorities.
        authoritative_total = [
            total_sources[name]
            for name in ("nvidia_smi", "system", "cgroup_v2")
            if name in total_sources
        ]
        total = min(authoritative_total) if authoritative_total else None
        if total is None:
            total = total_sources.get("torch_cuda")
        # Integrated GPUs share reclaimable system memory. cudaMemGetInfo may
        # report only the small currently mapped ATS window (GB10 commonly
        # reports less than 1 GiB while Linux has over 100 GiB available), so
        # it is diagnostic rather than the capacity authority.
        if finite_cgroup_limit and "cgroup_v2" not in free_sources:
            # A finite cgroup envelope without a readable current value has
            # unknown live headroom. Falling back to host-wide availability
            # could select a tier the process cannot actually sustain.
            free = None
        else:
            authoritative_free = [
                free_sources[name]
                for name in ("system", "cgroup_v2")
                if name in free_sources
            ]
            free = min(authoritative_free) if authoritative_free else None
            if free is None:
                free = free_sources.get("torch_cuda")
    else:
        # Without either NVML capacity or an explicit Torch integration flag,
        # do not infer a shared-memory topology from host RAM.  Diagnostic
        # Torch values remain in the source maps, but budgets fail closed.
        total = None
        free = None
    return {
        "model": memory_model,
        "total_memory_bytes": total,
        "free_memory_bytes": free,
        "total_sources": dict(sorted(total_sources.items())),
        "free_sources": dict(sorted(free_sources.items())),
        "nvml_memory_available": bool(selected.get("nvml_memory_available")),
        "system": dict(system_memory),
        "cgroup_v2": dict(cgroup_memory),
    }


def portable_hardware_selector(
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a selector without host- or device-local identity."""

    device = observation.get("device")
    memory = observation.get("memory")
    if not isinstance(device, Mapping) or not isinstance(memory, Mapping):
        raise HardwareProbeError(
            "hardware observation must contain device and memory objects"
        )
    selector = {
        "schema_version": SCHEMA_VERSION,
        "vendor": "nvidia",
        "product_name": _normalize_text(device.get("product_name")),
        "pci_device_id": _normalize_pci_device_id(
            device.get("pci_device_id")
        ),
        "compute_capability": device.get("compute_capability"),
        "memory_model": memory.get("model"),
        "total_memory_bytes": memory.get("total_memory_bytes"),
    }
    if selector["product_name"] is None:
        raise HardwareProbeError("hardware selector requires a product name")
    capability = selector["compute_capability"]
    if not (
        isinstance(capability, (list, tuple))
        and len(capability) == 2
        and all(
            isinstance(item, int) and not isinstance(item, bool) and item >= 0
            for item in capability
        )
    ):
        raise HardwareProbeError(
            "hardware selector requires a compute capability"
        )
    selector["compute_capability"] = list(capability)
    if selector["memory_model"] not in {"dedicated", "unified", "unknown"}:
        raise HardwareProbeError(
            "hardware selector has an invalid memory model"
        )
    total = selector["total_memory_bytes"]
    if total is not None and (
        not isinstance(total, int) or isinstance(total, bool) or total <= 0
    ):
        raise HardwareProbeError(
            "hardware selector total memory must be a positive integer "
            "or null"
        )
    _assert_secret_free(selector, "hardware_selector")
    return selector


def probe_nvidia_hardware(
    *,
    nvidia_smi: str = "nvidia-smi",
    torch_python: str | None = None,
    environment: Mapping[str, str] | None = None,
    visible_devices: str | None = None,
    cuda_device_order: str | None = None,
    meminfo_path: str | os.PathLike[str] = "/proc/meminfo",
    cgroup_root: str | os.PathLike[str] = "/sys/fs/cgroup",
    cgroup_membership_path: str | os.PathLike[str] = "/proc/self/cgroup",
    hostname: str | None = None,
    runner: Callable[
        ..., subprocess.CompletedProcess[str]
    ] = _SUBPROCESS_RUNNER,
) -> dict[str, Any]:
    """Observe exactly one NVIDIA GPU without importing a model runtime.

    Numeric CUDA selections on multi-GPU hosts require
    ``CUDA_DEVICE_ORDER=PCI_BUS_ID``.  UUID selections must resolve uniquely;
    this function never falls back to another device or all devices.
    """

    source_environment = os.environ if environment is None else environment
    safe_environment = _safe_environment(source_environment)
    if visible_devices is None:
        visible_devices = source_environment.get("CUDA_VISIBLE_DEVICES")
    if cuda_device_order is None:
        cuda_device_order = source_environment.get("CUDA_DEVICE_ORDER")
    if visible_devices is not None:
        safe_environment["CUDA_VISIBLE_DEVICES"] = visible_devices
    if cuda_device_order is not None:
        safe_environment["CUDA_DEVICE_ORDER"] = cuda_device_order

    executable = _resolve_executable(nvidia_smi, "nvidia-smi")
    argv = [
        executable,
        "--query-gpu=" + ",".join(_NVIDIA_QUERY_FIELDS),
        "--format=csv,noheader,nounits",
    ]
    completed = _run_bounded(
        argv,
        environment=safe_environment,
        timeout_seconds=30,
        runner=runner,
    )
    if completed.returncode != 0:
        raise HardwareProbeError(
            f"nvidia-smi exited with status {completed.returncode}"
        )
    rows = _parse_nvidia_rows(completed.stdout)
    selected, selection = _select_visible_device(
        rows,
        visible_devices=visible_devices,
        cuda_device_order=cuda_device_order,
    )

    torch_probe = None
    if torch_python is not None:
        torch_probe = _probe_torch(
            torch_python,
            environment=safe_environment,
            runner=runner,
        )
        result = torch_probe.get("result")
        if isinstance(result, Mapping):
            _validate_torch_identity(selected, result)

    system_memory = _read_system_memory(meminfo_path)
    cgroup_memory = _read_cgroup_v2_memory(
        cgroup_root,
        cgroup_membership_path,
    )
    memory = _effective_memory(
        selected,
        torch_probe,
        system_memory,
        cgroup_memory,
    )
    observation = {
        "schema_version": SCHEMA_VERSION,
        "vendor": "nvidia",
        "hostname": hostname or socket.gethostname(),
        "selection": selection,
        "device": {
            "index": selected["index"],
            "uuid": selected["uuid"],
            "pci_bus_id": selected["pci_bus_id"],
            "pci_device_id": selected["pci_device_id"],
            "product_name": selected["product_name"],
            "compute_capability": selected["compute_capability"],
            "driver_version": selected["driver_version"],
        },
        "memory": memory,
        "nvidia_smi": {
            "path": executable,
            "argv": argv,
            "returncode": completed.returncode,
            "stdout_sha256": hashlib.sha256(
                completed.stdout.encode("utf-8")
            ).hexdigest(),
            "stderr_sha256": hashlib.sha256(
                completed.stderr.encode("utf-8")
            ).hexdigest(),
            "discovered_device_count": len(rows),
        },
        "torch_probe": torch_probe,
    }
    selector = portable_hardware_selector(observation)
    observation["selector"] = selector
    observation["selector_sha256"] = content_sha256(selector)
    _assert_secret_free(observation, "hardware_observation")
    return observation


def _minimum_for_tier(
    minimum_required_bytes: int | Mapping[str, int],
    tier: str,
) -> int:
    if isinstance(minimum_required_bytes, Mapping):
        value = minimum_required_bytes.get(tier, 0)
    else:
        value = minimum_required_bytes
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise HardwareProbeError(
            "minimum required memory must be a nonnegative integer"
        )
    return value


def calculate_memory_tiers(
    observation: Mapping[str, Any],
    *,
    minimum_required_bytes: int | Mapping[str, int] = 0,
    policy: MemoryBudgetPolicy | None = None,
) -> dict[str, Any]:
    """Calculate 25%, 50%, and safe-max budgets with live availability."""

    policy = policy or MemoryBudgetPolicy()
    memory = observation.get("memory")
    if not isinstance(memory, Mapping):
        raise HardwareProbeError("hardware observation has no memory record")
    memory_model = memory.get("model")
    total = memory.get("total_memory_bytes")
    free = memory.get("free_memory_bytes")
    if memory_model == "unified":
        reserve = policy.unified_reserve_bytes
        max_fraction = policy.unified_max_fraction
    else:
        reserve = policy.dedicated_reserve_bytes
        max_fraction = policy.dedicated_max_fraction
    fraction_by_tier = {
        "25": policy.quarter_fraction,
        "50": policy.half_fraction,
        "max": max_fraction,
    }
    tiers = []
    for tier in MEMORY_TIER_IDS:
        fraction = fraction_by_tier[tier]
        required = _minimum_for_tier(minimum_required_bytes, tier)
        reasons = []
        capacity = None
        live_after_reserve = None
        if isinstance(total, int) and total > 0:
            capacity = canonical_memory_budget(total, fraction, reserve)
        else:
            reasons.append("total_memory_unknown")
        if isinstance(free, int) and free >= 0:
            live_after_reserve = max(0, free - reserve)
        else:
            reasons.append("live_free_memory_unknown")

        supported = (
            capacity is not None and capacity > 0 and capacity >= required
        )
        if capacity == 0:
            reasons.append("budget_zero_after_reserve")
        if capacity is not None and capacity < required:
            reasons.append("budget_below_minimum_required")
        full_budget_available = (
            supported
            and live_after_reserve is not None
            and capacity is not None
            and live_after_reserve >= capacity
        )
        minimum_fit_available = (
            supported
            and live_after_reserve is not None
            and live_after_reserve >= required
        )
        if (
            supported
            and live_after_reserve is not None
            and not full_budget_available
        ):
            reasons.append("live_capacity_below_tier_budget")
        if not reasons:
            reasons.append("supported_and_available")
        tiers.append(
            {
                "tier": tier,
                "target_fraction": fraction,
                "reserve_bytes": reserve,
                "total_memory_bytes": total,
                "capacity_budget_bytes": capacity,
                "minimum_required_bytes": required,
                "supported": supported,
                "live_free_memory_bytes": free,
                "live_memory_after_reserve_bytes": live_after_reserve,
                "minimum_fit_available": minimum_fit_available,
                "full_budget_available": full_budget_available,
                "reason_codes": reasons,
            }
        )
    selector = portable_hardware_selector(observation)
    return _make_receipt(
        "ocr_memory_tiers",
        {
            "hardware_selector": selector,
            "hardware_selector_sha256": content_sha256(selector),
            "memory_model": memory_model,
            "memory_evidence": {
                "total_memory_bytes": total,
                "free_memory_bytes": free,
                "total_sources": memory.get("total_sources", {}),
                "free_sources": memory.get("free_sources", {}),
                "nvml_memory_available": memory.get("nvml_memory_available"),
            },
            "policy": asdict(policy),
            "tiers": tiers,
        },
    )


def make_hardware_receipt(
    observation: Mapping[str, Any],
    *,
    memory_budgets: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a deterministic, secret-free receipt for one observation."""

    selector = portable_hardware_selector(observation)
    payload: dict[str, Any] = {
        "observation": dict(observation),
        "observation_sha256": content_sha256(observation),
        "hardware_selector": selector,
        "hardware_selector_sha256": content_sha256(selector),
    }
    if memory_budgets is not None:
        payload["memory_budgets"] = dict(memory_budgets)
        payload["memory_budgets_sha256"] = content_sha256(memory_budgets)
    return _make_receipt("ocr_hardware_observation", payload)


__all__ = [
    "GIB",
    "MEMORY_TIER_IDS",
    "HardwareProbeError",
    "MemoryBudgetPolicy",
    "calculate_memory_tiers",
    "content_sha256",
    "make_hardware_receipt",
    "portable_hardware_selector",
    "probe_nvidia_hardware",
]
