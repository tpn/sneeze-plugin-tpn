"""Benchmark-only JSONL adapter for grounded DeepSeek OCR tuning.

The production OCR command does not import or register this module.  It is a
fail-closed adapter for the benchmark protocol: every invocation must opt in
to tuning, every request must use the grounded OCR lane, and only execution
controls may vary.  The prompt and generation-quality controls remain fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import resource
import stat
import sys
import time
import traceback
from pathlib import Path
from typing import Any


class DeepSeekBenchmarkAdapterError(ValueError):
    """Raised when a benchmark request is unsafe or irreproducible."""


def _source_root_for_adapter(adapter_file: str | os.PathLike[str]) -> Path:
    """Resolve the package root for an adapter in the exact shipped layout."""

    try:
        adapter_path = Path(adapter_file).resolve(strict=True)
    except OSError as exc:
        raise DeepSeekBenchmarkAdapterError(
            "DeepSeek adapter source could not be resolved"
        ) from exc
    expected_suffix = ("sneeze", "tpn", "providers", "deepseek_vllm.py")
    if (
        not adapter_path.is_file()
        or len(adapter_path.parts) < len(expected_suffix) + 1
        or adapter_path.parts[-len(expected_suffix) :] != expected_suffix
    ):
        raise DeepSeekBenchmarkAdapterError(
            "DeepSeek adapter source has an unexpected package layout"
        )
    source_root = adapter_path.parents[3]
    required_sources = (
        source_root / "sneeze" / "tpn" / "__init__.py",
        source_root / "sneeze" / "tpn" / "ocr_backends.py",
        source_root / "sneeze" / "tpn" / "ocr_implementation.py",
        source_root / "sneeze" / "tpn" / "ocr_vllm.py",
        source_root / "sneeze" / "tpn" / "providers" / "__init__.py",
    )
    for source in required_sources:
        try:
            resolved = source.resolve(strict=True)
        except OSError as exc:
            raise DeepSeekBenchmarkAdapterError(
                "DeepSeek adapter source tree is incomplete"
            ) from exc
        if not resolved.is_file() or not resolved.is_relative_to(source_root):
            raise DeepSeekBenchmarkAdapterError(
                "DeepSeek adapter source tree is unsafe"
            )
    return source_root


def _bind_adapter_source_tree() -> Path:
    """Put this adapter's source tree first before importing sneeze.tpn."""

    source_root = _source_root_for_adapter(__file__)
    package_root = (source_root / "sneeze" / "tpn").resolve(strict=True)
    namespace_root = (source_root / "sneeze").resolve(strict=True)
    namespace = sys.modules.get("sneeze")
    if namespace is not None:
        namespace_paths = getattr(namespace, "__path__", ())
        try:
            resolved_paths = {
                Path(path).resolve(strict=True) for path in namespace_paths
            }
        except (OSError, TypeError) as exc:
            raise DeepSeekBenchmarkAdapterError(
                "DeepSeek adapter found an untraceable sneeze namespace"
            ) from exc
        if namespace_root not in resolved_paths:
            raise DeepSeekBenchmarkAdapterError(
                "DeepSeek adapter found a different sneeze namespace"
            )
    for name, module in tuple(sys.modules.items()):
        if name != "sneeze.tpn" and not name.startswith("sneeze.tpn."):
            continue
        module_file = getattr(module, "__file__", None)
        try:
            module_path = Path(module_file).resolve(strict=True)
        except (OSError, TypeError) as exc:
            raise DeepSeekBenchmarkAdapterError(
                "DeepSeek adapter found an untraceable preloaded module"
            ) from exc
        if not module_path.is_relative_to(package_root):
            raise DeepSeekBenchmarkAdapterError(
                "DeepSeek adapter found a preloaded different source tree"
            )
    source_root_text = os.fspath(source_root)
    sys.path[:] = [
        source_root_text,
        *(entry for entry in sys.path if entry != source_root_text),
    ]
    return source_root


def _verify_bound_module(
    source_root: Path, module_name: str, relative_path: str
) -> None:
    module = sys.modules.get(module_name)
    module_file = getattr(module, "__file__", None)
    expected = (source_root / relative_path).resolve(strict=True)
    try:
        observed = Path(module_file).resolve(strict=True)
    except (OSError, TypeError) as exc:
        raise DeepSeekBenchmarkAdapterError(
            "DeepSeek adapter imported an untraceable source module"
        ) from exc
    if observed != expected:
        raise DeepSeekBenchmarkAdapterError(
            "DeepSeek adapter imported a different source tree"
        )


_BOUND_SOURCE_ROOT = _bind_adapter_source_tree()
_ocr_vllm = importlib.import_module("sneeze.tpn.ocr_vllm")
_verify_bound_module(
    _BOUND_SOURCE_ROOT, "sneeze.tpn", "sneeze/tpn/__init__.py"
)
_verify_bound_module(
    _BOUND_SOURCE_ROOT,
    "sneeze.tpn.ocr_vllm",
    "sneeze/tpn/ocr_vllm.py",
)

DEFAULT_OCR_MODEL = _ocr_vllm.DEFAULT_OCR_MODEL
DEFAULT_OCR_MODEL_REVISION = _ocr_vllm.DEFAULT_OCR_MODEL_REVISION
DeepSeekOcrOfflineProcessor = _ocr_vllm.DeepSeekOcrOfflineProcessor
OcrPageTask = _ocr_vllm.OcrPageTask
ocr_image_dims = _ocr_vllm.ocr_image_dims
deepseek_jit_environment = _ocr_vllm.deepseek_jit_environment
QUALIFIED_DEEPSEEK_MOE_BACKEND = (
    _ocr_vllm.QUALIFIED_DEEPSEEK_MOE_BACKEND
)
validate_deepseek_moe_backend = _ocr_vllm.validate_deepseek_moe_backend

FIXED_INSTRUCTION = "Free OCR."
FIXED_QUALITY_SETTINGS = {
    "max_tokens": 8192,
    "ngram_size": 10,
    "window_size": 50,
    "temperature": 0.0,
}
EXECUTION_SETTING_KEYS = frozenset(
    {
        "attention_backend",
        "batch_size",
        "enable_chunked_prefill",
        "engine_overhead_headroom_bytes",
        "gpu_memory_utilization",
        "gpu_memory_utilization_denominator_bytes",
        "max_num_batched_tokens",
        "max_num_seqs",
        "memory_budget_bytes",
        "memory_tier",
        "mm_encoder_attn_backend",
        "moe_backend",
    }
)
REQUEST_SETTING_KEYS = EXECUTION_SETTING_KEYS | frozenset(
    FIXED_QUALITY_SETTINGS
)
MEMORY_TIERS = frozenset({"25", "50", "max"})
ATTENTION_BACKENDS = frozenset(
    {
        None,
        "auto",
        "FLASH_ATTN",
        "FLASHINFER",
        "FLEX_ATTENTION",
        "TORCH_SDPA",
        "XFORMERS",
    }
)
MM_ENCODER_ATTENTION_BACKENDS = frozenset(
    {None, "auto", "FLASH_ATTN", "TORCH_SDPA", "XFORMERS"}
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_REVISION_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_LINUX_MEMORY_KEYS = frozenset(
    {
        "MemTotal",
        "MemFree",
        "MemAvailable",
        "Buffers",
        "Cached",
        "SwapCached",
        "SwapTotal",
        "SwapFree",
        "Active",
        "Inactive",
        "Unevictable",
        "Mlocked",
    }
)


def implementation_record() -> dict[str, Any]:
    """Attest the installed source bundle and this exact adapter file."""

    from sneeze.tpn.ocr_implementation import (
        backend_implementation_record,
    )

    record = backend_implementation_record("deepseek-vllm")
    adapter_digest = record["components"].get("providers/deepseek_vllm.py")
    if adapter_digest != sha256_path(Path(__file__).resolve()):
        raise DeepSeekBenchmarkAdapterError(
            "executed DeepSeek adapter differs from installed implementation"
        )
    return record


_PROCESS_STATUS_KEYS = frozenset(
    {"VmPeak", "VmSize", "VmHWM", "VmRSS", "RssAnon", "RssFile"}
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests-jsonl", required=True)
    parser.add_argument("--responses-jsonl", required=True)
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--model-id", default=DEFAULT_OCR_MODEL)
    parser.add_argument("--revision", default=DEFAULT_OCR_MODEL_REVISION)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--tuning",
        action="store_true",
        required=True,
        help="Required acknowledgement that this adapter is benchmark-only.",
    )
    return parser.parse_args(argv)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise DeepSeekBenchmarkAdapterError(
                "request contains a duplicate JSON object key"
            )
        value[key] = item
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    values = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(
                line,
                object_pairs_hook=_strict_json_object,
            )
            if not isinstance(value, dict):
                raise DeepSeekBenchmarkAdapterError(
                    f"request at line {line_number} must be an object"
                )
            values.append(value)
    if not values:
        raise DeepSeekBenchmarkAdapterError("requests JSONL is empty")
    return values


def write_json(path: Path, value: Any) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        json.dump(
            value,
            handle,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        handle.write("\n")


def write_response(stream, response: dict[str, Any]) -> None:
    stream.write(json.dumps(response, ensure_ascii=False, default=str))
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())


def package_versions(names: tuple[str, ...]) -> dict[str, str]:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def installed_distributions_identity() -> dict[str, Any]:
    """Hash the complete normalized Python distribution manifest."""

    distributions = sorted(
        f"{distribution.metadata.get('Name', distribution.name)}=="
        f"{distribution.version}"
        for distribution in importlib.metadata.distributions()
    )
    payload = json.dumps(
        distributions,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "count": len(distributions),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "distributions": distributions,
    }


def startup_installed_distributions_identity() -> dict[str, Any]:
    """Capture the registered environment before runtime plugin imports."""

    identity = installed_distributions_identity()
    return {
        "count": identity["count"],
        "sha256": identity["sha256"],
    }


def _positive_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise DeepSeekBenchmarkAdapterError(
            f"request tuning {label} must be a positive integer"
        )
    return value


def validate_tuning_settings(settings: Any) -> dict[str, Any]:
    if not isinstance(settings, dict):
        raise DeepSeekBenchmarkAdapterError(
            "request tuning settings must be an object"
        )
    missing = REQUEST_SETTING_KEYS - set(settings)
    unexpected = set(settings) - REQUEST_SETTING_KEYS
    if missing:
        raise DeepSeekBenchmarkAdapterError(
            "request tuning settings are missing controls: "
            + ", ".join(sorted(missing))
        )
    if unexpected:
        raise DeepSeekBenchmarkAdapterError(
            "request tuning settings contain unsupported controls: "
            + ", ".join(sorted(unexpected))
        )
    normalized = dict(settings)
    normalized["batch_size"] = _positive_integer(
        settings["batch_size"], "batch_size"
    )
    normalized["max_num_seqs"] = _positive_integer(
        settings["max_num_seqs"], "max_num_seqs"
    )
    normalized["max_num_batched_tokens"] = _positive_integer(
        settings["max_num_batched_tokens"],
        "max_num_batched_tokens",
    )
    normalized["memory_budget_bytes"] = _positive_integer(
        settings["memory_budget_bytes"], "memory_budget_bytes"
    )
    normalized["engine_overhead_headroom_bytes"] = _positive_integer(
        settings["engine_overhead_headroom_bytes"],
        "engine_overhead_headroom_bytes",
    )
    normalized["gpu_memory_utilization_denominator_bytes"] = (
        _positive_integer(
            settings["gpu_memory_utilization_denominator_bytes"],
            "gpu_memory_utilization_denominator_bytes",
        )
    )
    if (
        normalized["engine_overhead_headroom_bytes"]
        >= normalized["memory_budget_bytes"]
    ):
        raise DeepSeekBenchmarkAdapterError(
            "request tuning engine overhead headroom must be smaller than "
            "the memory budget"
        )
    if normalized["batch_size"] > normalized["max_num_seqs"]:
        raise DeepSeekBenchmarkAdapterError(
            "request tuning batch_size cannot exceed max_num_seqs"
        )
    utilization = settings["gpu_memory_utilization"]
    if (
        not isinstance(utilization, (int, float))
        or isinstance(utilization, bool)
        or not 0.0 < float(utilization) <= 1.0
    ):
        raise DeepSeekBenchmarkAdapterError(
            "request tuning gpu_memory_utilization must be greater than "
            "zero and at most one"
        )
    normalized["gpu_memory_utilization"] = float(utilization)
    if settings["memory_tier"] not in MEMORY_TIERS:
        raise DeepSeekBenchmarkAdapterError(
            "request tuning memory_tier must be 25, 50, or max"
        )
    if settings["attention_backend"] not in ATTENTION_BACKENDS:
        raise DeepSeekBenchmarkAdapterError(
            "request tuning attention_backend is not allowlisted"
        )
    if (
        settings["mm_encoder_attn_backend"]
        not in MM_ENCODER_ATTENTION_BACKENDS
    ):
        raise DeepSeekBenchmarkAdapterError(
            "request tuning mm_encoder_attn_backend is not allowlisted"
        )
    if not isinstance(settings["enable_chunked_prefill"], bool):
        raise DeepSeekBenchmarkAdapterError(
            "request tuning enable_chunked_prefill must be boolean"
        )
    try:
        normalized["moe_backend"] = validate_deepseek_moe_backend(
            settings["moe_backend"]
        )
    except ValueError as exc:
        raise DeepSeekBenchmarkAdapterError(str(exc)) from exc
    if normalized["moe_backend"] != QUALIFIED_DEEPSEEK_MOE_BACKEND:
        raise DeepSeekBenchmarkAdapterError(
            "DeepSeek profile tuning requires moe_backend=triton"
        )
    for key, expected in FIXED_QUALITY_SETTINGS.items():
        value = settings[key]
        if isinstance(value, bool) or value != expected:
            raise DeepSeekBenchmarkAdapterError(
                f"request tuning {key} must remain fixed at {expected!r}"
            )
    if normalized["attention_backend"] == "auto":
        normalized["attention_backend"] = None
    if normalized["mm_encoder_attn_backend"] == "auto":
        normalized["mm_encoder_attn_backend"] = None
    return normalized


def _validate_instruction(request: dict[str, Any]) -> None:
    instruction = request.get("instruction")
    if instruction != FIXED_INSTRUCTION:
        raise DeepSeekBenchmarkAdapterError(
            "grounded OCR tuning requires the exact Free OCR. instruction"
        )
    expected = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
    if request.get("instruction_sha256") != expected:
        raise DeepSeekBenchmarkAdapterError(
            "request instruction SHA-256 does not match the instruction"
        )


def validate_request(request: dict[str, Any]) -> dict[str, Any]:
    attempt_id = request.get("attempt_id")
    if not isinstance(attempt_id, str) or not _SHA256_RE.fullmatch(
        attempt_id
    ):
        raise DeepSeekBenchmarkAdapterError(
            "request attempt_id must be a lowercase SHA-256"
        )
    if request.get("schema_version") != 1:
        raise DeepSeekBenchmarkAdapterError(
            "unsupported request schema_version"
        )
    if request.get("lane") != "grounded_ocr":
        raise DeepSeekBenchmarkAdapterError(
            "DeepSeek tuning adapter only supports lane=grounded_ocr"
        )
    _validate_instruction(request)
    input_record = request.get("input")
    if not isinstance(input_record, dict):
        raise DeepSeekBenchmarkAdapterError("request input must be an object")
    if input_record.get("raw_text") is not None:
        raise DeepSeekBenchmarkAdapterError(
            "grounded OCR tuning requests cannot include raw text"
        )
    image = input_record.get("image")
    if not isinstance(image, dict):
        raise DeepSeekBenchmarkAdapterError(
            "request input.image must be an object"
        )
    image_path = image.get("path")
    if not isinstance(image_path, str) or not Path(image_path).is_absolute():
        raise DeepSeekBenchmarkAdapterError(
            "request image path must be absolute"
        )
    if not Path(image_path).is_file():
        raise DeepSeekBenchmarkAdapterError(
            "request image path must be an existing file"
        )
    digest = image.get("sha256")
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise DeepSeekBenchmarkAdapterError(
            "request image SHA-256 is invalid"
        )
    dimensions = image.get("dimensions")
    if not isinstance(dimensions, dict) or not all(
        isinstance(dimensions.get(key), int)
        and not isinstance(dimensions[key], bool)
        and dimensions[key] > 0
        for key in ("w", "h")
    ):
        raise DeepSeekBenchmarkAdapterError(
            "request image dimensions are invalid"
        )
    return validate_tuning_settings(request.get("settings"))


def validate_request_batch(
    requests: list[dict[str, Any]],
) -> dict[str, Any]:
    seen = set()
    settings = []
    for request in requests:
        attempt_id = request.get("attempt_id")
        if attempt_id in seen:
            raise DeepSeekBenchmarkAdapterError(
                "requests contain duplicate attempt IDs"
            )
        seen.add(attempt_id)
        settings.append(validate_request(request))
    first = settings[0]
    if any(item != first for item in settings[1:]):
        raise DeepSeekBenchmarkAdapterError(
            "all requests in one provider process must use identical "
            "execution settings"
        )
    return first


def validate_device(device: str) -> int:
    if device != "cuda:0":
        raise DeepSeekBenchmarkAdapterError(
            "DeepSeek tuning device must be cuda:0; select a physical GPU "
            "with CUDA_VISIBLE_DEVICES"
        )
    return 0


def validate_image(request: dict[str, Any]) -> None:
    image = request["input"]["image"]
    path = Path(image["path"])
    if sha256_path(path) != image["sha256"]:
        raise DeepSeekBenchmarkAdapterError(
            "input image SHA-256 mismatch in provider"
        )
    if ocr_image_dims(str(path)) != image["dimensions"]:
        raise DeepSeekBenchmarkAdapterError(
            "input image dimensions mismatch in provider"
        )


def resolve_snapshot(model_id: str, revision: str) -> Path:
    if not _REVISION_RE.fullmatch(revision):
        raise DeepSeekBenchmarkAdapterError(
            "DeepSeek model revision must be an immutable commit hash"
        )
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_files_only=True,
        )
    ).resolve()
    if snapshot.name != revision:
        raise DeepSeekBenchmarkAdapterError(
            "resolved model snapshot does not match requested revision"
        )
    return snapshot


def _reject_snapshot_directory_symlinks(snapshot: Path) -> None:
    """Reject directory links without following them during snapshot scans."""

    def raise_walk_error(error: OSError) -> None:
        raise error

    try:
        for directory, directory_names, file_names in os.walk(
            snapshot,
            followlinks=False,
            onerror=raise_walk_error,
        ):
            root = Path(directory)
            for name in (*directory_names, *file_names):
                candidate = root / name
                if not stat.S_ISLNK(candidate.lstat().st_mode):
                    continue
                if stat.S_ISDIR(candidate.stat().st_mode):
                    raise DeepSeekBenchmarkAdapterError(
                        "model snapshot contains a directory symlink"
                    )
    except DeepSeekBenchmarkAdapterError:
        raise
    except OSError as exc:
        raise DeepSeekBenchmarkAdapterError(
            "could not inspect model snapshot directory tree"
        ) from exc


def snapshot_identity(snapshot: Path, revision: str) -> dict[str, Any]:
    from sneeze.tpn.ocr_backends import get_backend

    _reject_snapshot_directory_symlinks(snapshot)
    backend = get_backend("deepseek-vllm")
    required_files = backend.snapshot_required_files
    ignored_files = backend.snapshot_ignored_files
    materialized_files = {
        candidate.relative_to(snapshot).as_posix()
        for candidate in snapshot.rglob("*")
        if candidate.is_file()
    }
    missing = [
        relative
        for relative in required_files
        if relative not in materialized_files
    ]
    if missing:
        raise DeepSeekBenchmarkAdapterError(
            "resolved model snapshot is missing required files: "
            + ", ".join(missing)
        )
    unexpected = materialized_files - set(required_files) - set(
        ignored_files
    )
    if unexpected:
        raise DeepSeekBenchmarkAdapterError(
            "resolved model snapshot contains files outside the reviewed "
            "identity scope"
        )
    files = {}
    total_size = 0
    for relative in required_files:
        path = snapshot / relative
        resolved = path.resolve(strict=True)
        before = resolved.stat()
        digest = sha256_path(resolved)
        after = resolved.stat()
        if (
            before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise DeepSeekBenchmarkAdapterError(
                "resolved model snapshot changed while it was being hashed"
            )
        files[relative] = {
            "sha256": digest,
            "size_bytes": after.st_size,
        }
        total_size += after.st_size
    if not files:
        raise DeepSeekBenchmarkAdapterError(
            "resolved model snapshot contains no files"
        )
    return {
        "commit": revision,
        "path": str(snapshot),
        "file_count": len(files),
        "size_bytes": total_size,
        "files": files,
        "files_sha256": sha256_json(files),
    }


def _kilobytes_file(path: Path, allowed: frozenset[str]) -> dict[str, int]:
    values = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        label, separator, raw = line.partition(":")
        if not separator or label not in allowed:
            continue
        fields = raw.split()
        if not fields or not fields[0].isdigit():
            continue
        values[label] = int(fields[0]) * 1024
    return values


def linux_memory_telemetry() -> dict[str, int]:
    return _kilobytes_file(Path("/proc/meminfo"), _LINUX_MEMORY_KEYS)


def _cgroup_v2_directory(
    membership_path: Path,
    cgroup_root: Path,
) -> Path | None:
    try:
        lines = membership_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return None
    for line in lines:
        hierarchy, separator, remainder = line.partition(":")
        controllers, second_separator, relative = remainder.partition(":")
        if (
            separator
            and second_separator
            and hierarchy == "0"
            and controllers == ""
        ):
            parts = Path(relative.lstrip("/")).parts
            if any(part in {".", ".."} for part in parts):
                return None
            return cgroup_root.joinpath(*parts)
    return None


def _nonnegative_decimal_file(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None
    if len(value) > 64 or not value.isdecimal():
        return None
    try:
        return int(value)
    except ValueError:
        return None


def cgroup_v2_memory_telemetry(
    *,
    membership_path: Path = Path("/proc/self/cgroup"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> dict[str, Any]:
    """Sample cgroup v2 memory counters without exposing its local path."""

    result: dict[str, Any] = {
        "available": False,
        "memory_max_bytes": None,
        "memory_max_is_unlimited": None,
        "memory_available_bytes": None,
        "memory_current_bytes": None,
        "memory_peak_bytes": None,
    }
    directory = _cgroup_v2_directory(membership_path, cgroup_root)
    if directory is None:
        return result
    directories = []
    candidate = directory
    while True:
        directories.append(candidate)
        if candidate == cgroup_root:
            break
        candidate = candidate.parent
        if candidate != cgroup_root and cgroup_root not in candidate.parents:
            return result
    records: list[dict[str, int | str | None]] = []
    for candidate in directories:
        try:
            raw_maximum = (
                (candidate / "memory.max")
                .read_text(encoding="utf-8")
                .strip()
            )
        except (OSError, UnicodeError):
            maximum: int | str | None = None
        else:
            maximum = (
                raw_maximum
                if raw_maximum == "max"
                else (
                    int(raw_maximum)
                    if len(raw_maximum) <= 64
                    and raw_maximum.isdecimal()
                    else None
                )
            )
        records.append(
            {
                "maximum": maximum,
                "current": _nonnegative_decimal_file(
                    candidate / "memory.current"
                ),
                "peak": _nonnegative_decimal_file(
                    candidate / "memory.peak"
                ),
            }
        )
    maximums = [
        record["maximum"]
        for record in records
        if isinstance(record["maximum"], int)
        or record["maximum"] == "max"
    ]
    finite_maximums = [
        value for value in maximums if isinstance(value, int)
    ]
    if maximums:
        result["memory_max_bytes"] = (
            min(finite_maximums) if finite_maximums else None
        )
        result["memory_max_is_unlimited"] = not finite_maximums
        result["available"] = True
    finite_records = [
        record
        for record in records
        if isinstance(record["maximum"], int)
    ]
    headrooms = (
        [
            max(0, record["maximum"] - record["current"])
            for record in finite_records
        ]
        if finite_records
        and all(
            isinstance(record["current"], int)
            for record in finite_records
        )
        else []
    )
    if headrooms:
        result["memory_available_bytes"] = min(headrooms)
        result["available"] = True
    if records:
        nearest = records[0]
        for source, key in (
            ("current", "memory_current_bytes"),
            ("peak", "memory_peak_bytes"),
        ):
            value = nearest[source]
            if isinstance(value, int):
                result[key] = value
                result["available"] = True
    return result


def process_memory_telemetry() -> dict[str, Any]:
    status = _kilobytes_file(Path("/proc/self/status"), _PROCESS_STATUS_KEYS)
    self_usage = resource.getrusage(resource.RUSAGE_SELF)
    child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "pid": os.getpid(),
        "status_bytes": status,
        "self_max_rss_bytes": int(self_usage.ru_maxrss) * 1024,
        "children_max_rss_bytes": int(child_usage.ru_maxrss) * 1024,
    }


def cuda_memory_telemetry(torch_module) -> dict[str, Any]:
    cuda = torch_module.cuda
    if not cuda.is_available():
        raise DeepSeekBenchmarkAdapterError(
            "CUDA is not available to the DeepSeek tuning adapter"
        )
    device = cuda.current_device()
    properties = cuda.get_device_properties(device)
    free_bytes, total_bytes = cuda.mem_get_info(device)
    integrated = getattr(
        properties,
        "is_integrated",
        getattr(properties, "integrated", None),
    )
    if integrated is not None and bool(integrated):
        memory_topology = "unified"
    elif integrated is not None:
        memory_topology = "dedicated"
    else:
        memory_topology = "unknown"
    return {
        "device_index": device,
        "device_name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "memory_topology": memory_topology,
        "is_integrated": integrated,
        "device_total_memory_bytes": properties.total_memory,
        "cuda_free_memory_bytes": free_bytes,
        "cuda_total_memory_bytes": total_bytes,
        "torch_allocator": {
            "allocated_bytes": cuda.memory_allocated(device),
            "reserved_bytes": cuda.memory_reserved(device),
            "peak_allocated_bytes": cuda.max_memory_allocated(device),
            "peak_reserved_bytes": cuda.max_memory_reserved(device),
        },
    }


def memory_telemetry(torch_module) -> dict[str, Any]:
    return {
        "captured_at_monotonic_ns": time.monotonic_ns(),
        "cuda": cuda_memory_telemetry(torch_module),
        "linux": linux_memory_telemetry(),
        "cgroup_v2": cgroup_v2_memory_telemetry(),
        "process": process_memory_telemetry(),
    }


def reset_allocator_peaks(torch_module) -> None:
    try:
        torch_module.cuda.reset_peak_memory_stats(
            torch_module.cuda.current_device()
        )
    except (AttributeError, RuntimeError):
        return


def enforce_memory_budget(
    settings: dict[str, Any], total_memory_bytes: int
) -> dict[str, Any]:
    if total_memory_bytes <= 0:
        raise DeepSeekBenchmarkAdapterError(
            "CUDA device reported a non-positive memory total"
        )
    budget = settings["memory_budget_bytes"]
    if budget > total_memory_bytes:
        raise DeepSeekBenchmarkAdapterError(
            "request memory_budget_bytes exceeds CUDA device memory"
        )
    maximum_utilization = budget / total_memory_bytes
    requested = settings["gpu_memory_utilization"]
    requested_bytes = round(requested * total_memory_bytes)
    denominator = settings["gpu_memory_utilization_denominator_bytes"]
    if denominator < total_memory_bytes:
        raise DeepSeekBenchmarkAdapterError(
            "request gpu-memory utilization denominator is smaller than "
            "the runtime CUDA total"
        )
    planned_bytes = round(requested * denominator)
    rounding_allowance = max(1, round(total_memory_bytes * 1e-8))
    headroom = settings["engine_overhead_headroom_bytes"]
    if planned_bytes + headroom > budget + rounding_allowance:
        raise DeepSeekBenchmarkAdapterError(
            "request gpu_memory_utilization plus engine overhead headroom "
            "exceeds its memory budget cap"
        )
    return {
        **settings,
        "memory_budget_fraction": maximum_utilization,
        "requested_gpu_memory_bytes": requested_bytes,
        "planned_gpu_memory_bytes": planned_bytes,
        "engine_overhead_headroom_bytes": headroom,
        "utilization_rounding_allowance_bytes": rounding_allowance,
    }


def _artifact_stem(request: dict[str, Any], row_index: int) -> str:
    attempt_id = request.get("attempt_id")
    if isinstance(attempt_id, str) and _SHA256_RE.fullmatch(attempt_id):
        return f"{attempt_id}-{row_index:04d}"
    return f"invalid-{row_index:04d}-{time.monotonic_ns()}"


def _normalized_stop_reason(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _cgroup_v2_memory_summary(
    snapshots: list[dict[str, Any]],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    records = [
        item.get("cgroup_v2", {})
        for item in snapshots
        if isinstance(item.get("cgroup_v2"), dict)
    ]
    baseline_record = baseline.get("cgroup_v2", {})
    if not isinstance(baseline_record, dict):
        baseline_record = {}
    limits = [
        record["memory_max_bytes"]
        for record in records
        if isinstance(record.get("memory_max_bytes"), int)
        and record["memory_max_bytes"] >= 0
    ]
    unlimited_observations = [
        record["memory_max_is_unlimited"]
        for record in records
        if isinstance(record.get("memory_max_is_unlimited"), bool)
    ]
    capacity = min(limits) if limits else None
    capacity_is_unlimited = (
        False
        if limits
        else (
            True
            if unlimited_observations and all(unlimited_observations)
            else None
        )
    )
    current_values = [
        record["memory_current_bytes"]
        for record in records
        if isinstance(record.get("memory_current_bytes"), int)
        and record["memory_current_bytes"] >= 0
    ]
    baseline_current = baseline_record.get("memory_current_bytes")
    if not isinstance(baseline_current, int) or baseline_current < 0:
        baseline_current = current_values[0] if current_values else None
    maximum_current = max(current_values) if current_values else None
    current_delta = (
        max(0, maximum_current - baseline_current)
        if baseline_current is not None and maximum_current is not None
        else None
    )
    peak_values = [
        record["memory_peak_bytes"]
        for record in records
        if isinstance(record.get("memory_peak_bytes"), int)
        and record["memory_peak_bytes"] >= 0
    ]
    baseline_peak = baseline_record.get("memory_peak_bytes")
    if not isinstance(baseline_peak, int) or baseline_peak < 0:
        baseline_peak = peak_values[0] if peak_values else None
    maximum_peak = max(peak_values) if peak_values else None
    peak_delta = (
        max(0, maximum_peak - baseline_peak)
        if baseline_peak is not None and maximum_peak is not None
        else None
    )
    headroom_values = [
        record["memory_available_bytes"]
        for record in records
        if isinstance(record.get("memory_available_bytes"), int)
        and record["memory_available_bytes"] >= 0
    ]
    baseline_headroom = baseline_record.get("memory_available_bytes")
    if not isinstance(baseline_headroom, int) or baseline_headroom < 0:
        baseline_headroom = headroom_values[0] if headroom_values else None
    minimum_headroom = min(headroom_values) if headroom_values else None
    headroom_delta = (
        max(0, baseline_headroom - minimum_headroom)
        if baseline_headroom is not None and minimum_headroom is not None
        else None
    )
    return {
        "available": any(
            record.get("available") is True for record in records
        ),
        "capacity_bytes": capacity,
        "capacity_is_unlimited": capacity_is_unlimited,
        "baseline_current_bytes": baseline_current,
        "maximum_current_bytes": maximum_current,
        "observed_current_delta_bytes": current_delta,
        "baseline_peak_bytes": baseline_peak,
        "maximum_peak_bytes": maximum_peak,
        "observed_peak_delta_bytes": peak_delta,
        "baseline_headroom_bytes": baseline_headroom,
        "minimum_headroom_bytes": minimum_headroom,
        "observed_headroom_delta_bytes": headroom_delta,
        "memory_peak_semantics": (
            "cgroup_lifetime_high_water_mark_not_reset_by_adapter"
        ),
    }


def _gpu_memory_summary(
    telemetry: dict[str, Any],
    effective_settings: dict[str, Any],
) -> dict[str, Any]:
    snapshots = [
        value
        for value in telemetry.values()
        if isinstance(value, dict)
        and isinstance(value.get("cuda"), dict)
        and isinstance(value.get("process"), dict)
    ]
    snapshots.extend(
        value
        for value in telemetry.get("samples", [])
        if isinstance(value, dict)
        and isinstance(value.get("cuda"), dict)
        and isinstance(value.get("process"), dict)
    )
    if not snapshots:
        raise RuntimeError("memory telemetry contains no CUDA snapshots")
    baseline = telemetry.get("before_engine", snapshots[0])
    current = telemetry.get("batch_after", snapshots[-1])
    topology = current["cuda"]["memory_topology"]
    memory_kind = {
        "dedicated": "dedicated_vram",
        "unified": "unified_system",
    }.get(topology, "unknown")
    free_values = [
        item["cuda"]["cuda_free_memory_bytes"] for item in snapshots
    ]
    baseline_free = baseline["cuda"]["cuda_free_memory_bytes"]
    minimum_free = min(free_values)
    peak_reserved = max(
        item["cuda"]["torch_allocator"]["peak_reserved_bytes"]
        for item in snapshots
    )
    peak_allocated = max(
        item["cuda"]["torch_allocator"]["peak_allocated_bytes"]
        for item in snapshots
    )
    process_peak_values = []
    parent_peak_values = []
    children_peak_values = []
    for item in snapshots:
        process = item["process"]
        status = process.get("status_bytes", {})
        parent_values = (
            int(status.get("VmHWM", 0)),
            int(process.get("self_max_rss_bytes", 0)),
        )
        child_value = int(process.get("children_max_rss_bytes", 0))
        parent_peak_values.extend(parent_values)
        children_peak_values.append(child_value)
        process_peak_values.extend((*parent_values, child_value))
    parent_peak = max(parent_peak_values, default=0)
    children_peak = max(children_peak_values, default=0)
    linux_values = [
        item.get("linux", {}).get("MemAvailable")
        for item in snapshots
        if isinstance(item.get("linux"), dict)
        and isinstance(item.get("linux", {}).get("MemAvailable"), int)
        and item["linux"]["MemAvailable"] >= 0
    ]
    baseline_linux = baseline.get("linux", {}).get("MemAvailable")
    if not isinstance(baseline_linux, int) or baseline_linux < 0:
        baseline_linux = linux_values[0] if linux_values else None
    minimum_linux = min(linux_values) if linux_values else None
    observed_linux_delta = (
        max(0, baseline_linux - minimum_linux)
        if baseline_linux is not None and minimum_linux is not None
        else None
    )
    linux = current.get("linux", {})
    return {
        "memory_kind": memory_kind,
        "memory_tier": effective_settings["memory_tier"],
        "memory_budget_bytes": effective_settings["memory_budget_bytes"],
        "gpu_memory_utilization": effective_settings[
            "gpu_memory_utilization"
        ],
        "requested_gpu_memory_bytes": effective_settings[
            "requested_gpu_memory_bytes"
        ],
        "planned_gpu_memory_bytes": effective_settings[
            "planned_gpu_memory_bytes"
        ],
        "gpu_memory_utilization_denominator_bytes": effective_settings[
            "gpu_memory_utilization_denominator_bytes"
        ],
        "engine_overhead_headroom_bytes": effective_settings[
            "engine_overhead_headroom_bytes"
        ],
        "cuda_total_memory_bytes": current["cuda"]["cuda_total_memory_bytes"],
        "cuda_free_memory_bytes": current["cuda"]["cuda_free_memory_bytes"],
        "cuda_baseline_free_memory_bytes": baseline_free,
        "cuda_minimum_free_memory_bytes": minimum_free,
        "observed_cuda_used_delta_bytes": max(
            0, baseline_free - minimum_free
        ),
        "torch_peak_allocated_bytes": peak_allocated,
        "torch_peak_reserved_bytes": peak_reserved,
        "process_rss_bytes": current["process"]
        .get("status_bytes", {})
        .get("VmRSS"),
        "process_peak_rss_bytes": max(process_peak_values, default=0),
        "parent_peak_rss_bytes": parent_peak,
        "children_peak_rss_bytes": children_peak,
        "parent_plus_children_peak_rss_bytes": (parent_peak + children_peak),
        "parent_plus_children_peak_rss_semantics": (
            "sum_of_independent_parent_and_reaped_children_high_water_"
            "marks_not_simultaneous"
        ),
        "linux_mem_total_bytes": linux.get("MemTotal"),
        "linux_mem_available_bytes": linux.get("MemAvailable"),
        "linux_baseline_mem_available_bytes": baseline_linux,
        "linux_minimum_mem_available_bytes": minimum_linux,
        "observed_linux_mem_available_delta_bytes": observed_linux_delta,
        "cgroup_v2_memory": _cgroup_v2_memory_summary(
            snapshots,
            baseline,
        ),
    }


def _runtime_record(
    *,
    torch_module,
    generation_metadata: dict[str, Any],
    telemetry: dict[str, Any],
    effective_settings: dict[str, Any],
    startup_installed_distributions: dict[str, Any],
) -> dict[str, Any]:
    native_runtime = generation_metadata.get("runtime", {})
    packages = package_versions(
        (
            "huggingface-hub",
            "pdf2image",
            "pillow",
            "torch",
            "torchaudio",
            "torchvision",
            "tqdm",
            "transformers",
            "vllm",
            "tokenizers",
            "nvidia-ml-py",
        )
    )
    for name in ("torch", "vllm", "transformers", "tokenizers", "pillow"):
        value = native_runtime.get(name)
        if isinstance(value, str) and value:
            packages.setdefault(name, value)
    torch_version = getattr(torch_module, "__version__", None)
    if isinstance(torch_version, str) and torch_version:
        packages.setdefault("torch", torch_version)
    vllm_version = packages.get("vllm") or "unknown"
    attention_backend = native_runtime.get("attention_backend_selected")
    if not isinstance(attention_backend, str) or not attention_backend:
        raise DeepSeekBenchmarkAdapterError(
            "vLLM runtime did not report the selected attention backend"
        )
    dtype = native_runtime.get("dtype")
    if not isinstance(dtype, str) or not dtype:
        raise DeepSeekBenchmarkAdapterError(
            "vLLM runtime did not report the effective model dtype"
        )
    dtype = dtype.removeprefix("torch.")
    requested_moe_backend = native_runtime.get("moe_backend_requested")
    configured_moe_backend = native_runtime.get("moe_backend_configured")
    if (
        requested_moe_backend != effective_settings["moe_backend"]
        or configured_moe_backend != effective_settings["moe_backend"]
    ):
        raise DeepSeekBenchmarkAdapterError(
            "vLLM runtime did not attest the requested moe_backend"
        )
    expected_jit_environment = deepseek_jit_environment()
    if native_runtime.get("jit_environment") != expected_jit_environment:
        raise DeepSeekBenchmarkAdapterError(
            "vLLM runtime did not attest the configured JIT environment"
        )
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "engine": "vllm",
        "engine_version": vllm_version,
        "torch": str(torch_version or packages.get("torch") or "unknown"),
        "packages": packages,
        "installed_distributions": {
            "count": startup_installed_distributions["count"],
            "sha256": startup_installed_distributions["sha256"],
        },
        "installed_distributions_count": startup_installed_distributions[
            "count"
        ],
        "installed_distributions_sha256": startup_installed_distributions[
            "sha256"
        ],
        "attention_backend": str(attention_backend),
        "dtype": dtype,
        "mm_encoder_attention_backend": native_runtime.get(
            "mm_encoder_attention_backend_requested"
        ),
        "moe_backend_requested": requested_moe_backend,
        "moe_backend_configured": configured_moe_backend,
        "jit_environment": expected_jit_environment,
        "torch_cuda": str(
            getattr(getattr(torch_module, "version", None), "cuda", None)
        ),
        "device": telemetry["batch_after"]["cuda"],
        "gpu_memory": _gpu_memory_summary(telemetry, effective_settings),
        "memory": telemetry,
        "provider_adapter": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_path(Path(__file__).resolve()),
        },
        "implementation": implementation_record(),
    }


def _success_response(
    *,
    request: dict[str, Any],
    row_index: int,
    output_path: Path,
    generation_metadata: dict[str, Any],
    effective_settings: dict[str, Any],
    snapshot_record: dict[str, Any],
    model_id: str,
    revision: str,
    batch_wall_seconds: float,
    telemetry: dict[str, Any],
    artifacts_dir: Path,
    startup_installed_distributions: dict[str, Any],
) -> dict[str, Any]:
    if generation_metadata.get("status") != "completed":
        raise RuntimeError("DeepSeek processor did not complete generation")
    if not output_path.is_file():
        raise RuntimeError("DeepSeek processor did not create page output")
    output_text = output_path.read_text(encoding="utf-8")
    finish_reason = generation_metadata.get("finish_reason")
    if not isinstance(finish_reason, str) or not finish_reason:
        raise RuntimeError("DeepSeek processor omitted finish_reason")
    termination = generation_metadata.get("termination", {})
    reached_max = bool(
        finish_reason == "length"
        or termination.get("max_tokens")
        or termination.get("length")
    )
    native_eos = termination.get("eos")
    if not isinstance(native_eos, bool):
        raise RuntimeError("DeepSeek processor omitted termination.eos")
    if native_eos and reached_max:
        raise RuntimeError(
            "DeepSeek processor reported conflicting termination"
        )
    reached_eos = native_eos
    stop_reason = (
        "max_tokens"
        if reached_max
        else "eos"
        if reached_eos
        else _normalized_stop_reason(generation_metadata.get("stop_reason"))
    )
    artifact_stem = _artifact_stem(request, row_index)
    native_name = f"{artifact_stem}-native.json"
    native_path = artifacts_dir / native_name
    serializable_telemetry = {
        key: value
        for key, value in telemetry.items()
        if key != "torch_module"
    }
    native_record = {
        "schema_version": 1,
        "attempt_id": request["attempt_id"],
        "request": request,
        "model_snapshot": snapshot_record,
        "effective_execution_settings": effective_settings,
        "fixed_quality_settings": FIXED_QUALITY_SETTINGS,
        "generation_metadata": generation_metadata,
        "memory_telemetry": serializable_telemetry,
        "batch_wall_seconds": round(batch_wall_seconds, 6),
    }
    write_json(native_path, native_record)
    relative_output = output_path.relative_to(artifacts_dir).as_posix()
    return {
        "schema_version": 1,
        "attempt_id": request["attempt_id"],
        "status": "completed",
        "output": {"format": "markdown", "text": output_text},
        "effective_request": {
            "prompt": generation_metadata["prompt"],
            "prompt_mode": "plain_ocr",
            "input_artifacts": [
                {**request["input"]["image"], "transform": "original"}
            ],
        },
        "model": {
            "requested": model_id,
            "revision": revision,
            "resolved_snapshot": snapshot_record["path"],
            "resolved_revision": snapshot_record["commit"],
            "snapshot_file_count": snapshot_record["file_count"],
            "snapshot_files_sha256": snapshot_record["files_sha256"],
        },
        "runtime": _runtime_record(
            torch_module=telemetry["torch_module"],
            generation_metadata=generation_metadata,
            telemetry=serializable_telemetry,
            effective_settings=effective_settings,
            startup_installed_distributions=(startup_installed_distributions),
        ),
        "generation": {
            "params": {
                "execution_settings": effective_settings,
                "fixed_quality_settings": FIXED_QUALITY_SETTINGS,
                "sampling_params": generation_metadata.get(
                    "sampling_params", {}
                ),
                "logits_processor_extra_args": generation_metadata.get(
                    "logits_processor_extra_args", {}
                ),
                "engine_init_kwargs": generation_metadata.get(
                    "runtime", {}
                ).get("engine_init_kwargs", {}),
            },
            "input_tokens": generation_metadata.get("prompt_token_count", 0),
            "output_tokens": generation_metadata.get(
                "generated_token_count", 0
            ),
            "finish_reason": finish_reason,
            "stop_reason": stop_reason,
            "eos": reached_eos,
            "max_tokens": reached_max,
            "repetition": generation_metadata.get("repetition"),
        },
        "usage": {
            "wall_seconds": round(batch_wall_seconds, 6),
            "cost_usd": 0.0,
        },
        "artifacts": [
            {
                "relative_path": relative_output,
                "sha256": sha256_path(output_path),
            },
            {
                "relative_path": native_name,
                "sha256": sha256_path(native_path),
            },
        ],
    }


def error_response(
    request: dict[str, Any],
    *,
    row_index: int,
    artifacts_dir: Path,
    exc: Exception,
    phase: str,
    traceback_text: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact_name = f"{_artifact_stem(request, row_index)}-error.json"
    artifact_path = artifacts_dir / artifact_name
    write_json(
        artifact_path,
        {
            "schema_version": 1,
            "attempt_id": request.get("attempt_id"),
            "phase": phase,
            "error": {
                "type": exc.__class__.__name__,
                "message": str(exc) or repr(exc),
            },
            "traceback": traceback_text,
            "context": context or {},
        },
    )
    return {
        "schema_version": 1,
        "attempt_id": request.get("attempt_id"),
        "status": "error",
        "error": {
            "type": exc.__class__.__name__,
            "message": str(exc) or repr(exc),
            "phase": phase,
        },
        "output": {"format": "markdown", "text": ""},
        "artifacts": [
            {
                "relative_path": artifact_name,
                "sha256": sha256_path(artifact_path),
            }
        ],
    }


def _batches(values: list[Any], batch_size: int):
    for index in range(0, len(values), batch_size):
        yield values[index : index + batch_size]


def run_provider(
    *,
    requests_path: Path,
    responses_path: Path,
    artifacts_dir: Path,
    model_id: str,
    revision: str,
    tuning: bool,
    device: str = "cuda:0",
    processor_class=None,
    torch_module=None,
    snapshot_resolver=None,
) -> None:
    if not tuning:
        raise DeepSeekBenchmarkAdapterError(
            "DeepSeek benchmark adapter requires explicit tuning mode"
        )
    startup_installed_distributions = (
        startup_installed_distributions_identity()
    )
    requests = read_jsonl(requests_path)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    responses_path.parent.mkdir(parents=True, exist_ok=True)
    responses: list[dict[str, Any] | None] = [None] * len(requests)
    try:
        settings = validate_request_batch(requests)
    except Exception as exc:
        traceback_text = traceback.format_exc()
        for index, request in enumerate(requests):
            responses[index] = error_response(
                request,
                row_index=index,
                artifacts_dir=artifacts_dir,
                exc=exc,
                phase="request_validation",
                traceback_text=traceback_text,
            )
        _write_responses(responses_path, responses)
        return

    valid_rows = []
    for index, request in enumerate(requests):
        try:
            validate_image(request)
            valid_rows.append((index, request))
        except Exception as exc:
            responses[index] = error_response(
                request,
                row_index=index,
                artifacts_dir=artifacts_dir,
                exc=exc,
                phase="input_validation",
                traceback_text=traceback.format_exc(),
            )
    if not valid_rows:
        _write_responses(responses_path, responses)
        return

    try:
        if torch_module is None:
            import torch as torch_module
        device_index = validate_device(device)
    except Exception as exc:
        traceback_text = traceback.format_exc()
        for index, request in valid_rows:
            responses[index] = error_response(
                request,
                row_index=index,
                artifacts_dir=artifacts_dir,
                exc=exc,
                phase="runtime_initialization",
                traceback_text=traceback_text,
            )
        _write_responses(responses_path, responses)
        return
    if processor_class is None:
        processor_class = DeepSeekOcrOfflineProcessor
    if snapshot_resolver is None:
        snapshot_resolver = resolve_snapshot

    processor = None
    llm = None
    sampling_params = None
    snapshot_record = None
    engine_telemetry: dict[str, Any] = {"samples": []}
    pending_successes = []
    initialization_error = None
    try:
        if not torch_module.cuda.is_available():
            raise DeepSeekBenchmarkAdapterError(
                "CUDA is not available to the DeepSeek tuning adapter"
            )
        torch_module.cuda.set_device(device_index)
        engine_telemetry["before_engine"] = memory_telemetry(torch_module)
        engine_telemetry["samples"].append(
            {
                "label": "before_engine",
                **engine_telemetry["before_engine"],
            }
        )
        total_memory = engine_telemetry["before_engine"]["cuda"][
            "device_total_memory_bytes"
        ]
        effective_settings = enforce_memory_budget(settings, total_memory)
        expected_jit_environment = deepseek_jit_environment()
        observed_jit_environment = {
            name: os.environ.get(name) for name in expected_jit_environment
        }
        if observed_jit_environment != expected_jit_environment:
            raise DeepSeekBenchmarkAdapterError(
                "DeepSeek tuning requires provider-spec JIT controls; "
                "MAX_JOBS and FLASHINFER_NVCC_THREADS must match the "
                "fixed implementation values"
            )
        snapshot = snapshot_resolver(model_id, revision)
        snapshot_record = snapshot_identity(snapshot, revision)
        processor = processor_class(
            model=str(snapshot),
            model_revision=None,
            output_format="txt",
            prompt=FIXED_INSTRUCTION,
            prompt_mode="plain_ocr",
            keep_grounding=False,
            use_grounding=True,
            batch_size=effective_settings["batch_size"],
            enable_prefix_caching=False,
            enable_chunked_prefill=effective_settings[
                "enable_chunked_prefill"
            ],
            max_num_batched_tokens=effective_settings[
                "max_num_batched_tokens"
            ],
            max_num_seqs=effective_settings["max_num_seqs"],
            attention_backend=effective_settings["attention_backend"],
            mm_encoder_attn_backend=effective_settings[
                "mm_encoder_attn_backend"
            ],
            moe_backend=effective_settings["moe_backend"],
            gpu_memory_utilization=effective_settings[
                "gpu_memory_utilization"
            ],
            temperature=FIXED_QUALITY_SETTINGS["temperature"],
            max_tokens=FIXED_QUALITY_SETTINGS["max_tokens"],
            ngram_size=FIXED_QUALITY_SETTINGS["ngram_size"],
            window_size=FIXED_QUALITY_SETTINGS["window_size"],
            no_tqdm=True,
        )
        llm, sampling_params = processor.prepare_engine()
        engine_telemetry["after_engine"] = memory_telemetry(torch_module)
        engine_telemetry["samples"].append(
            {
                "label": "after_engine",
                **engine_telemetry["after_engine"],
            }
        )
        for batch in _batches(valid_rows, effective_settings["batch_size"]):
            reset_allocator_peaks(torch_module)
            batch_before = memory_telemetry(torch_module)
            engine_telemetry["samples"].append(
                {"label": "batch_before", **batch_before}
            )
            tasks = []
            output_paths = {}
            for row_index, request in batch:
                artifact_stem = _artifact_stem(request, row_index)
                output_path = artifacts_dir / (
                    f"{artifact_stem}-grounded-ocr.md"
                )
                if output_path.exists():
                    raise DeepSeekBenchmarkAdapterError(
                        "provider output artifact already exists"
                    )
                output_paths[row_index] = output_path
                tasks.append(
                    OcrPageTask(
                        image_path=request["input"]["image"]["path"],
                        output_path=str(output_path),
                    )
                )
            started = time.monotonic()
            processor.run(
                tasks,
                llm=llm,
                sampling_params=sampling_params,
            )
            batch_wall_seconds = time.monotonic() - started
            batch_after = memory_telemetry(torch_module)
            engine_telemetry["samples"].append(
                {"label": "batch_after", **batch_after}
            )
            batch_telemetry = {
                "batch_before": batch_before,
                "batch_after": batch_after,
                "batch_request_count": len(batch),
                "batch_wall_seconds": round(batch_wall_seconds, 6),
            }
            for row_index, request in batch:
                output_path = output_paths[row_index]
                metadata = processor.generation_metadata.get(str(output_path))
                if not isinstance(metadata, dict):
                    raise RuntimeError(
                        "DeepSeek processor omitted generation metadata"
                    )
                pending_successes.append(
                    {
                        "row_index": row_index,
                        "request": request,
                        "output_path": output_path,
                        "generation_metadata": metadata,
                        "batch_wall_seconds": batch_wall_seconds,
                        "batch_telemetry": batch_telemetry,
                        "effective_settings": effective_settings,
                    }
                )
    except Exception as exc:
        initialization_error = (exc, traceback.format_exc())
    finally:
        if processor is not None and llm is not None:
            try:
                processor.shutdown_engine(llm)
            except Exception as exc:  # pragma: no cover - best effort guard
                engine_telemetry["shutdown_error"] = {
                    "type": exc.__class__.__name__,
                    "message": str(exc) or repr(exc),
                }
        if torch_module is not None:
            try:
                engine_telemetry["after_shutdown"] = memory_telemetry(
                    torch_module
                )
                engine_telemetry["samples"].append(
                    {
                        "label": "after_shutdown",
                        **engine_telemetry["after_shutdown"],
                    }
                )
            except Exception as exc:  # pragma: no cover - telemetry guard
                engine_telemetry["after_shutdown_error"] = {
                    "type": exc.__class__.__name__,
                    "message": str(exc) or repr(exc),
                }

    if initialization_error is not None:
        exc, traceback_text = initialization_error
        pending_rows = {row_index for row_index, _request in valid_rows} - {
            item["row_index"] for item in pending_successes
        }
        for row_index in sorted(pending_rows):
            request = requests[row_index]
            responses[row_index] = error_response(
                request,
                row_index=row_index,
                artifacts_dir=artifacts_dir,
                exc=exc,
                phase="engine_or_batch_execution",
                traceback_text=traceback_text,
                context={"engine_telemetry": engine_telemetry},
            )

    if snapshot_record is not None:
        for pending in pending_successes:
            row_index = pending["row_index"]
            telemetry = {
                **engine_telemetry,
                **pending["batch_telemetry"],
                "torch_module": torch_module,
            }
            try:
                responses[row_index] = _success_response(
                    request=pending["request"],
                    row_index=row_index,
                    output_path=pending["output_path"],
                    generation_metadata=pending["generation_metadata"],
                    effective_settings=pending["effective_settings"],
                    snapshot_record=snapshot_record,
                    model_id=model_id,
                    revision=revision,
                    batch_wall_seconds=pending["batch_wall_seconds"],
                    telemetry=telemetry,
                    artifacts_dir=artifacts_dir,
                    startup_installed_distributions=(
                        startup_installed_distributions
                    ),
                )
            except Exception as exc:
                responses[row_index] = error_response(
                    pending["request"],
                    row_index=row_index,
                    artifacts_dir=artifacts_dir,
                    exc=exc,
                    phase="response_construction",
                    traceback_text=traceback.format_exc(),
                    context={"engine_telemetry": engine_telemetry},
                )

    for row_index, response in enumerate(responses):
        if response is None:
            exc = RuntimeError("provider produced no terminal response")
            responses[row_index] = error_response(
                requests[row_index],
                row_index=row_index,
                artifacts_dir=artifacts_dir,
                exc=exc,
                phase="response_construction",
                traceback_text="",
                context={"engine_telemetry": engine_telemetry},
            )
    _write_responses(responses_path, responses)


def _write_responses(
    path: Path, responses: list[dict[str, Any] | None]
) -> None:
    with open(path, "w", encoding="utf-8", newline="") as stream:
        for response in responses:
            if response is None:
                raise RuntimeError("attempted to write an empty response")
            write_response(stream, response)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_provider(
        requests_path=Path(args.requests_jsonl),
        responses_path=Path(args.responses_jsonl),
        artifacts_dir=Path(args.artifacts_dir),
        model_id=args.model_id,
        revision=args.revision,
        tuning=args.tuning,
        device=args.device,
    )


if __name__ == "__main__":
    main()
