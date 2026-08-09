from __future__ import annotations

import base64
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import signal
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import time
import urllib.parse
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .ocr_path_security import (
    TrustedPathError,
    prepare_private_directory,
    reject_protected_corpus_overlap,
)
from .ocr_process_lifecycle import isolated_subprocess_signal_guard
from .ocr_vllm import (
    analyze_markdown_quality,
    file_sha256,
    ocr_image_dims,
)

SCHEMA_VERSION = 1
_BASE_PROTOCOL_ARGUMENTS = (
    "--requests-jsonl",
    "--responses-jsonl",
    "--artifacts-dir",
)
_EXTERNAL_PROTOCOL_ARGUMENTS = (
    "--external-provider-spec",
    "--external-plan",
    "--external-approval",
    "--external-budget-ledger",
)
_PROTOCOL_ARGUMENTS = (
    *_BASE_PROTOCOL_ARGUMENTS,
    *_EXTERNAL_PROTOCOL_ARGUMENTS,
)
_GIT_PROVENANCE_PATHS = (
    "src",
    "tests",
    "benchmarks",
    ".gitignore",
    "AGENTS.md",
    "README.md",
    "environment-ocr.yml",
    "pyproject.toml",
)
_SOURCE_TREE_CACHE_DIRECTORIES = frozenset(
    {
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
    }
)
_SOURCE_TREE_CACHE_SUFFIXES = frozenset({".pyc", ".pyo"})
_SECRET_ENV_RE = re.compile(
    r"(?:AUTH|COOKIE|CREDENTIAL|KEY|PASS(?:WORD)?|SECRET|TOKEN)",
    re.IGNORECASE,
)
_CREDENTIAL_ENV_RE = re.compile(
    r"(?:^|_)(?:API_KEY|AUTH(?:ORIZATION)?|BEARER|COOKIE|"
    r"CREDENTIALS?|KEY|PASS(?:WORD)?|SECRET|TOKEN)(?:_|$)",
    re.IGNORECASE,
)
_CREDENTIAL_ARG_RE = re.compile(
    r"(?:^|_)(?:API_KEY|ACCESS_TOKEN|AUTH(?:ORIZATION)?|BEARER|"
    r"COOKIE|CREDENTIALS?|KEY|PASS(?:WORD)?|SECRET|TOKEN)(?:_|$)",
    re.IGNORECASE,
)
_CREDENTIAL_HEADER_RE = re.compile(
    r"(?:^|[=\s])(?:authorization|proxy-authorization|cookie|x-api-key)"
    r"\s*[:=]|(?:^|[=\s])bearer\s+\S+",
    re.IGNORECASE,
)
_PROXY_ENV_RE = re.compile(
    r"^(?:ALL|FTP|HTTP|HTTPS|NO)_PROXY$",
    re.IGNORECASE,
)
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_INFERENCE_ENV_RE = re.compile(
    r"^(?:(?:CUDA|CUBLAS|CUDNN|FLASHINFER|HF|MKL|NCCL|OMP|PADDLE|"
    r"PYTORCH|TOKENIZERS|TORCH|TRANSFORMERS|TRITON|VLLM)(?:_|$)|"
    r"PYTHONHASHSEED$)"
)
_REPRO_ENV_RE = re.compile(
    rf"(?:{_INFERENCE_ENV_RE.pattern}|LD_LIBRARY_PATH$|PATH$|PYTHONPATH$)"
)
_EXTERNAL_AMBIENT_ENV_ALLOWLIST = frozenset(
    {
        "LANG",
        "LANGUAGE",
        "LC_ALL",
        "LC_CTYPE",
        "TZ",
    }
)
_EXTERNAL_SPEC_ENV_ALLOWLIST = (
    _EXTERNAL_AMBIENT_ENV_ALLOWLIST | {"PYTHONHASHSEED"}
)
_EXTERNAL_FORCED_ENV = {
    "PATH": os.defpath,
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
    "PYTHONUNBUFFERED": "1",
}
_CREDENTIAL_REDACTION = "[REDACTED_CREDENTIAL]"
_PROVIDER_SPEC_KEYS = frozenset(
    {
        "schema_version",
        "provider_id",
        "argv",
        "cwd",
        "egress",
        "model",
        "lanes",
        "timeout_seconds",
        "secret_env_names",
        "env",
    }
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
_ALLOWED_METRICS = frozenset(
    {
        "quality.axis.terms",
        "quality.table_completion.raw_token_multiset_recall",
        "quality.table_completion.grounded_ocr_token_multiset_recall",
        "quality.table_completion.markdown_raw_word_ratio",
        "quality.table_completion.nonblank_lines",
        "quality.table_completion.pipe_table_rows",
        "quality.repetition.max_repeat.nonoverlap_occurrences",
        "quality.repetition.max_unsupported_repeat.nonoverlap_occurrences",
    }
)
_ZERO_WHEN_MISSING_METRICS = frozenset(
    {
        "quality.repetition.max_repeat.nonoverlap_occurrences",
        "quality.repetition.max_unsupported_repeat.nonoverlap_occurrences",
    }
)


class BenchmarkError(ValueError):
    """Raised when a benchmark input or provider response is unsafe."""


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise BenchmarkError("duplicate JSON field")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number is not allowed: {value}")
    return parsed


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise BenchmarkError(f"Could not read {path}: {exc}") from exc


def _resolve_source_path(
    path: str | os.PathLike[str],
    label: str,
) -> Path:
    try:
        return Path(path).expanduser().resolve(strict=True)
    except OSError as exc:
        raise BenchmarkError(
            f"Could not resolve {label} {path}: {exc}"
        ) from exc


def _decode_json(payload: bytes, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
            parse_float=_parse_finite_json_float,
        )
    except ValueError as exc:
        raise BenchmarkError(
            f"Could not read JSON from {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise BenchmarkError(f"Expected one JSON object in {path}")
    return value


def _read_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    source = Path(path)
    return _decode_json(_read_bytes(source), source)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        json.dump(
            value,
            handle,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        for value in values:
            handle.write(_json_bytes(value).decode("utf-8"))
            handle.write("\n")


def _decode_jsonl(payload: bytes, path: Path) -> list[dict[str, Any]]:
    values = []
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BenchmarkError(f"Could not decode {path}: {exc}") from exc
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(
                line,
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
                parse_float=_parse_finite_json_float,
            )
        except ValueError as exc:
            raise BenchmarkError(
                f"Invalid JSON at {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise BenchmarkError(
                f"Expected an object at {path}:{line_number}"
            )
        values.append(value)
    return values


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return _decode_jsonl(_read_bytes(path), path)


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise BenchmarkError(f"{label} must be a nonempty string")
    return value


def _safe_stem(value: Any, label: str) -> str:
    stem = _require_string(value, label)
    candidate = Path(stem)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise BenchmarkError(f"{label} must be a safe relative path")
    if candidate.suffix:
        raise BenchmarkError(f"{label} must not include a file extension")
    return candidate.as_posix()


def _safe_component(value: Any, label: str) -> str:
    component = _require_string(value, label)
    candidate = Path(component)
    if (
        candidate.is_absolute()
        or component in {".", ".."}
        or "/" in component
        or "\\" in component
        or "\x00" in component
        or any(
            ord(character) < 32 or ord(character) == 127
            for character in component
        )
    ):
        raise BenchmarkError(f"{label} must be a safe path component")
    return component


def _looks_credential_related(name: str) -> bool:
    return bool(_CREDENTIAL_ENV_RE.search(name.replace("-", "_")))


def _argv_contains_credentials(argv: list[str]) -> bool:
    for argument in argv[1:]:
        if argument.startswith("-"):
            option = argument.split("=", 1)[0]
            normalized = option.lstrip("-").replace("-", "_")
            if _CREDENTIAL_ARG_RE.search(normalized):
                return True
        if _CREDENTIAL_HEADER_RE.search(argument):
            return True
        if "://" not in argument:
            continue
        try:
            parsed = urllib.parse.urlsplit(argument)
        except ValueError:
            return True
        if parsed.username is not None or parsed.password is not None:
            return True
        for encoded_fields in (parsed.query, parsed.fragment):
            for name, value in urllib.parse.parse_qsl(
                encoded_fields,
                keep_blank_values=True,
            ):
                if value and _looks_credential_related(name):
                    return True
    return False


def _reject_embedded_secret_material(
    value: Any,
    label: str,
) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if _looks_credential_related(str(key)):
                raise BenchmarkError(
                    f"{label} contains a credential-like field"
                )
            _reject_embedded_secret_material(item, f"{label}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_embedded_secret_material(
                item,
                f"{label}[{index}]",
            )
        return
    if isinstance(value, str) and any(
        pattern.search(value) for pattern in _SECRET_VALUE_RES
    ):
        raise BenchmarkError(
            f"{label} contains a credential-like value"
        )


def _validate_sha256(value: Any, label: str) -> str:
    digest = _require_string(value, label)
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise BenchmarkError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def load_manifest(
    path: str | os.PathLike[str],
    *,
    _source_bytes: bytes | None = None,
) -> list[dict[str, Any]]:
    source = Path(path)
    manifest = (
        _read_jsonl(source)
        if _source_bytes is None
        else _decode_jsonl(_source_bytes, source)
    )
    if not manifest:
        raise BenchmarkError("Benchmark manifest is empty")
    seen = set()
    for index, case in enumerate(manifest, 1):
        prefix = f"manifest case {index}"
        if case.get("schema_version") != SCHEMA_VERSION:
            raise BenchmarkError(
                f"{prefix} has an unsupported schema_version"
            )
        case_id = _safe_component(case.get("case_id"), f"{prefix}.case_id")
        if case_id in seen:
            raise BenchmarkError(f"Duplicate case_id: {case_id}")
        seen.add(case_id)
        _safe_stem(case.get("stem"), f"{prefix}.stem")
        input_spec = case.get("input")
        if not isinstance(input_spec, dict):
            raise BenchmarkError(f"{prefix}.input must be an object")
        _validate_sha256(
            input_spec.get("image_sha256"),
            f"{prefix}.input.image_sha256",
        )
        _validate_sha256(
            input_spec.get("raw_text_sha256"),
            f"{prefix}.input.raw_text_sha256",
        )
        dimensions = input_spec.get("dimensions")
        if not isinstance(dimensions, dict) or not all(
            isinstance(dimensions.get(key), int) and dimensions[key] > 0
            for key in ("w", "h")
        ):
            raise BenchmarkError(
                f"{prefix}.input.dimensions must contain positive w and h"
            )
        reference = case.get("reference")
        if not isinstance(reference, dict):
            raise BenchmarkError(f"{prefix}.reference must be an object")
        for key in ("markdown_sha256", "grounded_ocr_sha256"):
            _validate_sha256(reference.get(key), f"{prefix}.reference.{key}")
        assertions = case.get("assertions")
        if not isinstance(assertions, list) or not assertions:
            raise BenchmarkError(
                f"{prefix}.assertions must be a nonempty list"
            )
        for assertion_index, assertion in enumerate(assertions, 1):
            _validate_assertion(
                assertion,
                f"{prefix}.assertions[{assertion_index}]",
            )
    return manifest


def _validate_assertion(assertion: Any, label: str) -> None:
    if not isinstance(assertion, dict):
        raise BenchmarkError(f"{label} must be an object")
    lanes = assertion.get("lanes")
    if lanes is not None:
        if not isinstance(lanes, list) or not lanes:
            raise BenchmarkError(
                f"{label}.lanes must be a nonempty unique list"
            )
        normalized_lanes = [
            _safe_component(lane, f"{label}.lanes[{index}]")
            for index, lane in enumerate(lanes, 1)
        ]
        if len(normalized_lanes) != len(set(normalized_lanes)):
            raise BenchmarkError(
                f"{label}.lanes must be a nonempty unique list"
            )
    kind = assertion.get("type") or assertion.get("kind")
    if kind == "quality_status":
        expected = assertion.get("equals", assertion.get("value", "pass"))
        if expected not in {"pass", "fail"}:
            raise BenchmarkError(f"{label}.equals must be pass or fail")
        return
    if kind == "literal_occurrences":
        _require_string(
            assertion.get("literal", assertion.get("text")),
            f"{label}.text",
        )
    elif kind == "regex_occurrences":
        pattern = _require_string(
            assertion.get("pattern"), f"{label}.pattern"
        )
        try:
            re.compile(pattern)
        except re.error as exc:
            raise BenchmarkError(
                f"{label}.pattern is invalid: {exc}"
            ) from exc
    elif kind == "metric_range":
        metric = _require_string(
            assertion.get("metric", assertion.get("path")),
            f"{label}.path",
        )
        if metric not in _ALLOWED_METRICS:
            raise BenchmarkError(
                f"{label}.metric is not allowlisted: {metric}"
            )
    else:
        raise BenchmarkError(f"{label}.type is unsupported: {kind!r}")
    if "min" not in assertion and "max" not in assertion:
        raise BenchmarkError(f"{label} must specify min and/or max")
    for bound in ("min", "max"):
        if bound in assertion and not _is_finite_number(assertion[bound]):
            raise BenchmarkError(f"{label}.{bound} must be numeric")


def load_provider_spec(
    path: str | os.PathLike[str],
    *,
    _source_bytes: bytes | None = None,
) -> dict[str, Any]:
    source = Path(path)
    spec = (
        _read_json(source)
        if _source_bytes is None
        else _decode_json(_source_bytes, source)
    )
    if spec.get("schema_version") != SCHEMA_VERSION:
        raise BenchmarkError(
            "Provider spec has an unsupported schema_version"
        )
    egress = spec.get("egress", "none")
    allowed_keys = set(_PROVIDER_SPEC_KEYS)
    if egress == "external":
        allowed_keys.add("external_policy")
    unexpected_keys = set(spec) - allowed_keys
    if unexpected_keys:
        raise BenchmarkError(
            "provider spec has unexpected fields; refusing to persist "
            "unvalidated data"
        )
    _require_string(spec.get("provider_id"), "provider_id")
    argv = spec.get("argv")
    if (
        not isinstance(argv, list)
        or not argv
        or not all(isinstance(item, str) and item for item in argv)
    ):
        raise BenchmarkError("provider argv must be a nonempty string list")
    if any("\x00" in item for item in argv):
        raise BenchmarkError("provider argv cannot contain a NUL byte")
    if _argv_contains_credentials(argv):
        raise BenchmarkError(
            "provider argv cannot contain credential-bearing options or URLs"
        )
    for index, argument in enumerate(argv):
        _reject_embedded_secret_material(argument, f"provider argv[{index}]")
    executable = Path(argv[0]).expanduser()
    if not executable.is_absolute():
        raise BenchmarkError("provider argv[0] must be an absolute path")
    executable = executable.resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise BenchmarkError(
            "provider argv[0] must be an existing executable file"
        )
    if any(
        argument == protocol_arg
        or argument.startswith(f"{protocol_arg}=")
        for argument in argv
        for protocol_arg in _PROTOCOL_ARGUMENTS
    ):
        raise BenchmarkError(
            "provider argv cannot supply protocol path arguments"
        )
    spec["argv"] = [str(executable), *argv[1:]]
    if "cwd" in spec and spec["cwd"] is not None:
        cwd = Path(_require_string(spec["cwd"], "cwd")).expanduser()
        if not cwd.is_absolute() or not cwd.is_dir():
            raise BenchmarkError(
                "provider cwd must be an existing absolute path"
            )
        spec["cwd"] = str(cwd.resolve())
    if egress not in {"none", "external"}:
        raise BenchmarkError("provider egress must be none or external")
    model = spec.get("model")
    if not isinstance(model, dict):
        raise BenchmarkError("provider model must be an object")
    if set(model) != {"requested", "revision"}:
        raise BenchmarkError(
            "provider model must contain exactly requested and revision"
        )
    _require_string(model.get("requested"), "model.requested")
    _require_string(model.get("revision"), "model.revision")
    _reject_embedded_secret_material(model, "model")
    lane_specs = spec.get("lanes")
    if not isinstance(lane_specs, dict) or not lane_specs:
        raise BenchmarkError("provider lanes must be a nonempty object")
    for lane, lane_spec in lane_specs.items():
        _safe_component(lane, "lane name")
        if not isinstance(lane_spec, dict):
            raise BenchmarkError(f"provider lane {lane} must be an object")
        if not set(lane_spec) <= {
            "prompt",
            "settings",
            "include_raw_text",
        }:
            raise BenchmarkError(
                f"provider lane {lane} has unexpected fields"
            )
        _require_string(lane_spec.get("prompt"), f"lanes.{lane}.prompt")
        if not isinstance(lane_spec.get("settings", {}), dict):
            raise BenchmarkError(f"lanes.{lane}.settings must be an object")
        if egress == "external" and "include_raw_text" not in lane_spec:
            raise BenchmarkError(
                "external provider lanes must explicitly set "
                "include_raw_text"
            )
        include_raw = lane_spec.get(
            "include_raw_text",
            lane == "best" if egress == "none" else False,
        )
        if not isinstance(include_raw, bool):
            raise BenchmarkError(
                f"lanes.{lane}.include_raw_text must be boolean"
            )
        if lane == "parity" and include_raw:
            raise BenchmarkError("the parity lane cannot include raw text")
        _reject_embedded_secret_material(
            lane_spec["prompt"],
            f"lanes.{lane}.prompt",
        )
        _reject_embedded_secret_material(
            lane_spec.get("settings", {}),
            f"lanes.{lane}.settings",
        )
    timeout = spec.get("timeout_seconds", 1800)
    if not isinstance(timeout, int) or timeout <= 0:
        raise BenchmarkError("timeout_seconds must be a positive integer")
    secrets = spec.get("secret_env_names", [])
    if not isinstance(secrets, list) or not all(
        isinstance(item, str) and _ENV_NAME_RE.fullmatch(item)
        for item in secrets
    ):
        raise BenchmarkError(
            "secret_env_names must contain valid environment names"
        )
    if len(secrets) != len(set(secrets)):
        raise BenchmarkError("secret_env_names must be unique")
    if egress == "none" and secrets:
        raise BenchmarkError(
            "providers without external egress cannot receive secrets"
        )
    if egress == "external" and any(
        not _looks_credential_related(name) for name in secrets
    ):
        raise BenchmarkError(
            "external secret_env_names must be credential-related"
        )
    overrides = spec.get("env", {})
    if not isinstance(overrides, dict) or not all(
        isinstance(key, str)
        and _ENV_NAME_RE.fullmatch(key)
        and isinstance(value, str)
        and "\x00" not in value
        for key, value in overrides.items()
    ):
        raise BenchmarkError(
            "provider env must map valid environment names to "
            "NUL-free strings"
        )
    for name, value in overrides.items():
        _reject_embedded_secret_material(value, f"provider env.{name}")
    secret_overrides = [
        key for key in overrides if _SECRET_ENV_RE.search(key)
    ]
    if secret_overrides:
        raise BenchmarkError(
            "provider env cannot contain secret-looking values; use "
            "secret_env_names"
        )
    if any(_PROXY_ENV_RE.fullmatch(key) for key in overrides):
        raise BenchmarkError("provider env cannot override proxy settings")
    if egress == "external" and any(
        key not in _EXTERNAL_SPEC_ENV_ALLOWLIST for key in overrides
    ):
        raise BenchmarkError(
            "external provider env contains a non-allowlisted setting"
        )
    if egress == "external":
        try:
            from .providers.external_api import (
                ExternalAdapterError,
                validate_external_provider_spec_record,
            )

            spec = validate_external_provider_spec_record(spec)
        except ExternalAdapterError as exc:
            raise BenchmarkError(
                "external provider spec is not a valid built-in profile"
            ) from exc
    return spec


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_destination(input_root: Path, dest_dir: Path) -> Path:
    resolved_input = input_root.resolve()
    resolved_dest = dest_dir.resolve()
    if _is_within(resolved_dest, resolved_input) or _is_within(
        resolved_input, resolved_dest
    ):
        raise BenchmarkError(
            "Benchmark destination must be separate from the input root"
        )
    try:
        reject_protected_corpus_overlap(
            resolved_dest,
            label="benchmark destination",
        )
        return prepare_private_directory(
            dest_dir,
            label="benchmark destination",
            require_empty=True,
        )
    except TrustedPathError as exc:
        raise BenchmarkError(str(exc)) from exc


def _png_dimensions(path: Path) -> dict[str, int] | None:
    dimensions = ocr_image_dims(str(path))
    if dimensions is not None:
        return dimensions
    try:
        with open(path, "rb") as handle:
            header = handle.read(24)
    except OSError:
        return None
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    width, height = struct.unpack(">II", header[16:24])
    return {"w": width, "h": height}


def _verified_file(path: Path, expected: str, label: str) -> None:
    actual = file_sha256(str(path))
    if actual != expected:
        raise BenchmarkError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
        )


def prepare_run(
    manifest: list[dict[str, Any]],
    input_root: str | os.PathLike[str],
    dest_dir: str | os.PathLike[str],
) -> list[dict[str, Any]]:
    input_path = Path(input_root).expanduser()
    dest_path = Path(dest_dir).expanduser()
    if not input_path.is_dir():
        raise BenchmarkError(f"Input root does not exist: {input_path}")
    dest_path = _validate_destination(input_path, dest_path)

    prepared = []
    for case in manifest:
        stem = case["stem"]
        paths = {
            "image": input_path / f"{stem}.png",
            "raw_text": input_path / f"{stem}.txt",
            "grounded_ocr": input_path / f"{stem}.ocr",
            "reference_markdown": input_path / f"{stem}.md",
        }
        _verified_file(
            paths["image"],
            case["input"]["image_sha256"],
            f"{case['case_id']} image",
        )
        _verified_file(
            paths["raw_text"],
            case["input"]["raw_text_sha256"],
            f"{case['case_id']} raw text",
        )
        _verified_file(
            paths["grounded_ocr"],
            case["reference"]["grounded_ocr_sha256"],
            f"{case['case_id']} grounded OCR",
        )
        _verified_file(
            paths["reference_markdown"],
            case["reference"]["markdown_sha256"],
            f"{case['case_id']} reference Markdown",
        )
        dimensions = _png_dimensions(paths["image"])
        if dimensions != case["input"]["dimensions"]:
            raise BenchmarkError(
                f"{case['case_id']} dimensions mismatch: expected "
                f"{case['input']['dimensions']}, got {dimensions}"
            )
        evaluator_bytes = {
            key: paths[key].read_bytes()
            for key in ("raw_text", "grounded_ocr", "reference_markdown")
        }
        evaluator_hashes = {
            "raw_text": case["input"]["raw_text_sha256"],
            "grounded_ocr": case["reference"]["grounded_ocr_sha256"],
            "reference_markdown": case["reference"]["markdown_sha256"],
        }
        for key, payload in evaluator_bytes.items():
            actual = hashlib.sha256(payload).hexdigest()
            if actual != evaluator_hashes[key]:
                raise BenchmarkError(
                    f"{case['case_id']} {key} changed during preparation"
                )
        try:
            evaluator_text = {
                key: payload.decode("utf-8")
                for key, payload in evaluator_bytes.items()
            }
        except UnicodeDecodeError as exc:
            raise BenchmarkError(
                f"{case['case_id']} evaluator text is not UTF-8: {exc}"
            ) from exc
        prepared.append(
            {
                "case": case,
                "source_paths": paths,
                "evaluator_bytes": evaluator_bytes,
                "evaluator_text": evaluator_text,
            }
        )

    inputs = dest_path / "inputs"
    for item in prepared:
        case = item["case"]
        stem = case["stem"]
        copied = {
            "image": inputs / f"{stem}.png",
            "raw_text": inputs / f"{stem}.txt",
        }
        for key, output_path in copied.items():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item["source_paths"][key], output_path)
        _verified_file(
            copied["image"],
            case["input"]["image_sha256"],
            f"{case['case_id']} copied image",
        )
        _verified_file(
            copied["raw_text"],
            case["input"]["raw_text_sha256"],
            f"{case['case_id']} copied raw text",
        )
        item["copied_paths"] = copied
    return prepared


def _read_text(path: Path) -> str:
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    except OSError as exc:
        raise BenchmarkError(
            f"Could not read text file {path}: {exc}"
        ) from exc


def _write_evaluator_evidence(
    dest_path: Path,
    prepared: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    root = dest_path / "evaluator"
    records = []
    suffixes = {
        "raw_text": ".txt",
        "grounded_ocr": ".ocr",
        "reference_markdown": ".md",
    }
    for item in prepared:
        case = item["case"]
        case_records = []
        for key, suffix in suffixes.items():
            relative = Path(f"{case['stem']}{suffix}")
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(item["evaluator_bytes"][key])
            case_records.append(
                {
                    "kind": key,
                    "path": relative.as_posix(),
                    "sha256": file_sha256(str(destination)),
                }
            )
        records.append(
            {"case_id": case["case_id"], "artifacts": case_records}
        )
    _write_json(root / "manifest.json", {"cases": records})
    return records


def _nested_value(value: dict[str, Any], dotted: str) -> Any:
    current: Any = value
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _has_nested(value: dict[str, Any], dotted: str) -> bool:
    current: Any = value
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]
    return True


def evaluate_assertions(
    markdown_text: str,
    quality_or_assertions: dict[str, Any] | list[dict[str, Any]],
    assertions: list[dict[str, Any]] | None = None,
    *,
    raw_text: str | None = None,
    grounded_ocr_text: str | None = None,
    lane: str | None = None,
) -> list[dict[str, Any]]:
    if assertions is None:
        assertions = quality_or_assertions
        quality = analyze_markdown_quality(
            markdown_text,
            ocr_text=grounded_ocr_text,
            raw_text=raw_text,
        )
    else:
        quality = quality_or_assertions
    if not isinstance(assertions, list) or not isinstance(quality, dict):
        raise BenchmarkError("Invalid assertion evaluation inputs")
    context = {"quality": quality}
    outcomes = []
    for assertion in assertions:
        lanes = assertion.get("lanes")
        if isinstance(lanes, list) and lane not in lanes:
            continue
        kind = assertion.get("type") or assertion.get("kind")
        if kind == "quality_status":
            expected = assertion.get("equals", assertion.get("value", "pass"))
            actual = quality["status"]
            passed = actual == expected
        elif kind == "literal_occurrences":
            haystack = markdown_text
            needle = assertion.get("literal", assertion.get("text"))
            if assertion.get("casefold", True):
                haystack = haystack.casefold()
                needle = needle.casefold()
            actual = haystack.count(needle)
            expected = {
                key: assertion[key]
                for key in ("min", "max")
                if key in assertion
            }
            passed = _in_range(actual, assertion)
        elif kind == "regex_occurrences":
            flags = 0
            if assertion.get("ignore_case", True):
                flags |= re.IGNORECASE
            if assertion.get("multiline", True):
                flags |= re.MULTILINE
            actual = len(
                re.findall(assertion["pattern"], markdown_text, flags)
            )
            expected = {
                key: assertion[key]
                for key in ("min", "max")
                if key in assertion
            }
            passed = _in_range(actual, assertion)
        else:
            metric = assertion.get("metric", assertion.get("path"))
            actual = _nested_value(context, metric)
            if actual is None and metric in _ZERO_WHEN_MISSING_METRICS:
                actual = 0
            expected = {
                key: assertion[key]
                for key in ("min", "max")
                if key in assertion
            }
            passed = isinstance(actual, (int, float)) and _in_range(
                actual, assertion
            )
        outcomes.append(
            {
                "type": kind,
                "passed": passed,
                "actual": actual,
                "expected": expected,
                "assertion": assertion,
            }
        )
    return outcomes


def _in_range(actual: int | float, assertion: dict[str, Any]) -> bool:
    return not (
        ("min" in assertion and actual < assertion["min"])
        or ("max" in assertion and actual > assertion["max"])
    )


def _build_request(
    item: dict[str, Any],
    spec: dict[str, Any],
    spec_hash: str,
    manifest_hash: str,
    lane: str,
    repeat: int,
    experiment_hash: str,
) -> dict[str, Any]:
    case = item["case"]
    lane_spec = spec["lanes"][lane]
    identity = {
        "provider_spec_sha256": spec_hash,
        "manifest_sha256": manifest_hash,
        "lane": lane,
        "repeat": repeat,
        "case_id": case["case_id"],
        "experiment_sha256": experiment_hash,
    }
    prompt = lane_spec["prompt"]
    raw_text = None
    include_raw_text = lane_spec.get(
        "include_raw_text",
        lane == "best" if spec.get("egress", "none") == "none" else False,
    )
    if include_raw_text:
        raw_text = {
            "path": str(item["copied_paths"]["raw_text"].resolve()),
            "sha256": case["input"]["raw_text_sha256"],
            "text": _read_text(item["copied_paths"]["raw_text"]),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "attempt_id": _sha256_json(identity),
        "experiment_sha256": experiment_hash,
        "case_id": case["case_id"],
        "lane": lane,
        "repeat": repeat,
        "input": {
            "image": {
                "path": str(item["copied_paths"]["image"].resolve()),
                "sha256": case["input"]["image_sha256"],
                "dimensions": case["input"]["dimensions"],
            },
            "raw_text": raw_text,
        },
        "instruction": prompt,
        "instruction_sha256": _sha256_text(prompt),
        "settings": lane_spec.get("settings", {}),
    }


def _provider_environment(spec: dict[str, Any]) -> dict[str, str]:
    allowed_secrets = set(spec.get("secret_env_names", []))
    explicit = spec.get("env", {})
    if spec.get("egress", "none") == "external":
        environment = {
            key: os.environ[key]
            for key in _EXTERNAL_AMBIENT_ENV_ALLOWLIST
            if key in os.environ
        }
        environment.update(explicit)
        environment.update(_EXTERNAL_FORCED_ENV)
        environment.update(
            {
                key: os.environ[key]
                for key in allowed_secrets
                if key in os.environ
            }
        )
        return environment
    environment = {
        key: value
        for key, value in os.environ.items()
        if not _SECRET_ENV_RE.search(key) or key in allowed_secrets
        if (
            not _INFERENCE_ENV_RE.search(key)
            or key in explicit
            or key in allowed_secrets
        )
    }
    environment.update(explicit)
    return environment


def _recorded_provider_environment(
    environment: dict[str, str],
    spec: dict[str, Any],
) -> dict[str, str]:
    recorded = {
        key: value
        for key, value in environment.items()
        if _REPRO_ENV_RE.search(key) and not _SECRET_ENV_RE.search(key)
    }
    recorded.update(spec.get("env", {}))
    return dict(sorted(recorded.items()))


def _credential_variants(values: tuple[str, ...]) -> tuple[str, ...]:
    variants = set()
    for value in values:
        if not value:
            continue
        encoded = value.encode("utf-8")
        standard_base64 = base64.b64encode(encoded).decode("ascii")
        urlsafe_base64 = base64.urlsafe_b64encode(encoded).decode("ascii")
        variants.add(value)
        variants.add(json.dumps(value, ensure_ascii=False)[1:-1])
        variants.add(urllib.parse.quote(value, safe=""))
        variants.add(standard_base64)
        variants.add(standard_base64.rstrip("="))
        variants.add(urlsafe_base64)
        variants.add(urlsafe_base64.rstrip("="))
    return tuple(sorted(variants, key=len, reverse=True))


def _credential_replacements(
    values: tuple[str, ...],
) -> tuple[tuple[bytes, bytes], ...]:
    marker = _CREDENTIAL_REDACTION.encode("utf-8")
    return tuple(
        (variant.encode("utf-8"), marker)
        for variant in _credential_variants(values)
        if variant
    )


def _redact_credential_bytes(
    payload: bytes,
    replacements: tuple[tuple[bytes, bytes], ...],
) -> tuple[bytes, int]:
    redactions = 0
    for needle, marker in replacements:
        occurrences = payload.count(needle)
        if occurrences:
            payload = payload.replace(needle, marker)
            redactions += occurrences
    return payload, redactions


def _redact_credential_text(
    value: str,
    credential_values: tuple[str, ...],
) -> tuple[str, int]:
    redactions = 0
    for needle in _credential_variants(credential_values):
        occurrences = value.count(needle)
        if occurrences:
            value = value.replace(needle, _CREDENTIAL_REDACTION)
            redactions += occurrences
    return value, redactions


def _redact_credential_file(
    path: Path,
    replacements: tuple[tuple[bytes, bytes], ...],
) -> tuple[int, bool]:
    if path.is_symlink():
        path.unlink(missing_ok=True)
        return 0, False
    if not path.is_file():
        return 0, True
    try:
        payload = path.read_bytes()
        redacted, count = _redact_credential_bytes(payload, replacements)
        if count:
            path.write_bytes(redacted)
    except OSError:
        return 0, False
    return count, True


def _unused_sanitized_path(path: Path) -> Path:
    candidate = path
    index = 1
    while os.path.lexists(candidate):
        candidate = path.with_name(f"{path.name}.redacted-{index}")
        index += 1
    return candidate


def _redact_artifact_tree(
    artifacts_dir: Path,
    credential_values: tuple[str, ...],
    replacements: tuple[tuple[bytes, bytes], ...],
) -> tuple[int, bool]:
    redactions = 0
    scan_complete = True
    for root, dirnames, filenames in os.walk(
        artifacts_dir,
        topdown=False,
        followlinks=False,
    ):
        root_path = Path(root)
        for name in filenames:
            path = root_path / name
            count, complete = _redact_credential_file(path, replacements)
            redactions += count
            scan_complete = scan_complete and complete
            if not os.path.lexists(path):
                continue
            sanitized_name, count = _redact_credential_text(
                name,
                credential_values,
            )
            redactions += count
            if sanitized_name != name:
                destination = _unused_sanitized_path(
                    path.with_name(sanitized_name)
                )
                try:
                    path.rename(destination)
                except OSError:
                    scan_complete = False
        for name in dirnames:
            path = root_path / name
            if path.is_symlink():
                try:
                    path.unlink()
                except OSError:
                    scan_complete = False
                else:
                    scan_complete = False
                continue
            sanitized_name, count = _redact_credential_text(
                name,
                credential_values,
            )
            redactions += count
            if sanitized_name != name:
                destination = _unused_sanitized_path(
                    path.with_name(sanitized_name)
                )
                try:
                    path.rename(destination)
                except OSError:
                    scan_complete = False
    return redactions, scan_complete


def _provider_credential_values(
    spec: dict[str, Any],
    environment: dict[str, str],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                environment[name]
                for name in spec.get("secret_env_names", [])
                if environment.get(name)
            },
            key=len,
            reverse=True,
        )
    )


def _persist_sanitized_capture(
    capture,
    path: Path,
    replacements: tuple[tuple[bytes, bytes], ...],
) -> tuple[int, bool]:
    try:
        capture.seek(0)
        payload = capture.read()
        redacted, count = _redact_credential_bytes(payload, replacements)
        path.write_bytes(redacted)
    except OSError:
        path.write_bytes(b"")
        return 0, False
    return count, True


def _sanitize_provider_evidence(
    *,
    credential_values: tuple[str, ...],
    replacements: tuple[tuple[bytes, bytes], ...],
    log_redactions: dict[str, int],
    log_scan_complete: bool,
    responses_path: Path,
    artifacts_dir: Path,
) -> dict[str, Any]:
    surfaces = dict(log_redactions)
    scan_complete = log_scan_complete
    count, complete = _redact_credential_file(
        responses_path,
        replacements,
    )
    surfaces["responses"] = count
    scan_complete = scan_complete and complete
    artifact_count, complete = _redact_artifact_tree(
        artifacts_dir,
        credential_values,
        replacements,
    )
    surfaces["artifacts"] = artifact_count
    scan_complete = scan_complete and complete
    return {
        "credential_leak_detected": any(surfaces.values()),
        "credential_redactions": dict(sorted(surfaces.items())),
        "credential_scan_complete": scan_complete,
    }


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_process_group(
    process_group: int,
    *,
    grace_seconds: float = 2.0,
) -> None:
    if not _process_group_exists(process_group):
        return
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if not _process_group_exists(process_group):
            return
        time.sleep(0.05)
    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _terminate_and_reap_provider_process(
    process: subprocess.Popen,
) -> None:
    _terminate_process_group(process.pid)
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        _terminate_process_group(process.pid, grace_seconds=0)
        process.wait()


def _run_provider_process(
    argv: list[str],
    *,
    cwd: str | None,
    environment: dict[str, str],
    stdout_handle,
    stderr_handle,
    timeout_seconds: int,
) -> tuple[int, str | None]:
    with isolated_subprocess_signal_guard() as lifecycle:
        return _run_provider_process_guarded(
            argv,
            cwd=cwd,
            environment=environment,
            stdout_handle=stdout_handle,
            stderr_handle=stderr_handle,
            timeout_seconds=timeout_seconds,
            lifecycle=lifecycle,
        )


def _run_provider_process_guarded(
    argv: list[str],
    *,
    cwd: str | None,
    environment: dict[str, str],
    stdout_handle,
    stderr_handle,
    timeout_seconds: int,
    lifecycle: Any,
) -> tuple[int, str | None]:
    process: subprocess.Popen | None = None
    try:
        try:
            with lifecycle.spawn_window():
                process = subprocess.Popen(
                    argv,
                    cwd=cwd,
                    env=environment,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    start_new_session=True,
                )
        except OSError as exc:
            return 127, f"could not execute provider: {exc}"
        assert process is not None
        try:
            returncode = lifecycle.wait(
                process,
                timeout_seconds=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return 124, (
                f"provider timed out after {timeout_seconds} seconds"
            )
        return returncode, None
    finally:
        # Cleanup remains inside the passive guard and is unconditional:
        # normal providers can leave helpers, and failures can occur anywhere
        # after the private process group is created.
        with lifecycle.cleanup_window():
            if process is not None:
                _terminate_and_reap_provider_process(process)


def _is_nonnegative_integer(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
    )


def _is_finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _is_nonnegative_number(value: Any) -> bool:
    return _is_finite_number(value) and value >= 0


def _provider_metadata_gaps(
    response: dict[str, Any],
    request: dict[str, Any],
    spec: dict[str, Any],
    artifacts_dir: Path,
) -> list[str]:
    required = (
        "effective_request.prompt",
        "effective_request.input_artifacts",
        "model.requested",
        "model.revision",
        "model.resolved_snapshot",
        "runtime.python",
        "runtime.engine",
        "runtime.engine_version",
        "runtime.packages",
        "runtime.attention_backend",
        "generation.params",
        "generation.input_tokens",
        "generation.output_tokens",
        "generation.finish_reason",
        "generation.stop_reason",
        "generation.eos",
        "generation.max_tokens",
        "usage.wall_seconds",
        "usage.cost_usd",
        "artifacts",
    )
    gaps = [path for path in required if not _has_nested(response, path)]
    effective_prompt = _nested_value(response, "effective_request.prompt")
    if not isinstance(effective_prompt, str) or not effective_prompt:
        gaps.append("effective_request.prompt")
    artifacts = _nested_value(response, "effective_request.input_artifacts")
    original = None
    if isinstance(artifacts, list):
        original = next(
            (
                artifact
                for artifact in artifacts
                if isinstance(artifact, dict)
                and artifact.get("transform") == "original"
                and artifact.get("sha256")
                == request["input"]["image"]["sha256"]
                and artifact.get("dimensions")
                == request["input"]["image"]["dimensions"]
            ),
            None,
        )
    if original is None:
        gaps.append("effective_request.original_image")
    elif original.get("path") != request["input"]["image"]["path"]:
        gaps.append("effective_request.original_image.path")
    if (
        _nested_value(response, "model.requested")
        != spec["model"]["requested"]
    ):
        gaps.append("model.requested_mismatch")
    if _nested_value(response, "model.revision") != spec["model"]["revision"]:
        gaps.append("model.revision_mismatch")
    for path in (
        "model.requested",
        "model.revision",
        "model.resolved_snapshot",
        "runtime.python",
        "runtime.engine",
        "runtime.engine_version",
        "generation.finish_reason",
    ):
        value = _nested_value(response, path)
        if not isinstance(value, str) or not value:
            gaps.append(path)
    packages = _nested_value(response, "runtime.packages")
    if not isinstance(packages, dict) or not all(
        isinstance(key, str) and key
        and isinstance(value, str) and value
        for key, value in packages.items()
    ):
        gaps.append("runtime.packages")
    attention_backend = _nested_value(response, "runtime.attention_backend")
    if not isinstance(attention_backend, str) or not attention_backend:
        gaps.append("runtime.attention_backend")
    params = _nested_value(response, "generation.params")
    if not isinstance(params, dict):
        gaps.append("generation.params")
    input_tokens = _nested_value(response, "generation.input_tokens")
    if not _is_nonnegative_integer(input_tokens):
        gaps.append("generation.input_tokens")
    if not _is_nonnegative_integer(
        _nested_value(response, "generation.output_tokens")
    ):
        gaps.append("generation.output_tokens")
    finish_reason = _nested_value(response, "generation.finish_reason")
    stop_reason = _nested_value(response, "generation.stop_reason")
    if stop_reason is not None and not isinstance(stop_reason, str):
        gaps.append("generation.stop_reason")
    eos = _nested_value(response, "generation.eos")
    max_tokens = _nested_value(response, "generation.max_tokens")
    if not isinstance(eos, bool):
        gaps.append("generation.eos")
    if not isinstance(max_tokens, bool):
        gaps.append("generation.max_tokens")
    if isinstance(eos, bool) and isinstance(max_tokens, bool):
        if eos and max_tokens:
            gaps.append("generation.termination_conflict")
        elif not eos and not max_tokens and not stop_reason:
            gaps.append("generation.termination_unknown")
    if finish_reason == "length" and max_tokens is not True:
        gaps.append("generation.length_without_max_tokens")
    wall_seconds = _nested_value(response, "usage.wall_seconds")
    if not _is_nonnegative_number(wall_seconds):
        gaps.append("usage.wall_seconds")
    cost_usd = _nested_value(response, "usage.cost_usd")
    if cost_usd is not None and not _is_nonnegative_number(cost_usd):
        gaps.append("usage.cost_usd")
    response_artifacts = response.get("artifacts")
    if not isinstance(response_artifacts, list) or not response_artifacts:
        gaps.append("artifacts")
        response_artifacts = []
    for index, artifact in enumerate(response_artifacts):
        if not isinstance(artifact, dict):
            gaps.append(f"artifacts.{index}")
            continue
        try:
            relative = Path(
                _safe_stem_with_suffix(artifact.get("relative_path"))
            )
        except BenchmarkError:
            gaps.append(f"artifacts.{index}.relative_path")
            continue
        expected = artifact.get("sha256")
        artifact_path = artifacts_dir / relative
        if (
            not isinstance(expected, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected)
            or not artifact_path.is_file()
            or not _is_within(
                artifact_path.resolve(), artifacts_dir.resolve()
            )
            or file_sha256(str(artifact_path)) != expected
        ):
            gaps.append(f"artifacts.{index}.sha256")
    return list(dict.fromkeys(gaps))


def _safe_stem_with_suffix(value: Any) -> str:
    relative = _require_string(value, "artifact relative_path")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise BenchmarkError("artifact relative_path is unsafe")
    return candidate.as_posix()


def _normalize_markdown(text: str) -> str:
    return "\n".join(
        line.rstrip() for line in text.replace("\r\n", "\n").split("\n")
    ).strip()


def _result_for_response(
    request: dict[str, Any],
    response: dict[str, Any] | None,
    item: dict[str, Any],
    spec: dict[str, Any],
    attempt_dir: Path,
    subprocess_record: dict[str, Any],
    response_error: str | None,
) -> dict[str, Any]:
    case = item["case"]
    status = "error"
    output_text = ""
    if response is not None:
        status = response.get("status", "error")
        if response.get("schema_version") != SCHEMA_VERSION:
            response_error = response_error or (
                "provider response has an unsupported schema_version"
            )
            status = "error"
        if status not in {"completed", "error"}:
            response_error = response_error or (
                f"provider response has invalid status: {status!r}"
            )
            status = "error"
        output = response.get("output")
        if isinstance(output, dict) and isinstance(output.get("text"), str):
            output_text = output["text"]
            if output.get("format") != "markdown":
                response_error = response_error or (
                    "provider output format must be markdown"
                )
                status = "error"
        elif status == "completed":
            response_error = (
                response_error or "completed response has no output text"
            )
            status = "error"
    output_path = attempt_dir / "outputs" / f"{case['case_id']}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        handle.write(output_text)

    raw_text = item["evaluator_text"]["raw_text"]
    grounded = item["evaluator_text"]["grounded_ocr"]
    quality = analyze_markdown_quality(
        output_text,
        ocr_text=grounded,
        raw_text=raw_text,
    )
    assertion_outcomes = evaluate_assertions(
        output_text,
        quality,
        case["assertions"],
        raw_text=raw_text,
        grounded_ocr_text=grounded,
        lane=request["lane"],
    )
    metadata_gaps = []
    if response is None:
        metadata_gaps.append("response")
    else:
        metadata_gaps = _provider_metadata_gaps(
            response,
            request,
            spec,
            attempt_dir / "artifacts",
        )
    if subprocess_record["returncode"] != 0:
        response_error = response_error or (
            f"provider exited with status {subprocess_record['returncode']}"
        )
        status = "error"
    passed = (
        status == "completed"
        and not response_error
        and not metadata_gaps
        and bool(assertion_outcomes)
        and all(outcome["passed"] for outcome in assertion_outcomes)
    )
    output_sha256 = _sha256_text(output_text)
    normalized_output_sha256 = _sha256_text(_normalize_markdown(output_text))
    effective_prompt = (
        _nested_value(response, "effective_request.prompt")
        if response
        else None
    )
    metadata_complete = not metadata_gaps
    return {
        "schema_version": SCHEMA_VERSION,
        "attempt_id": request["attempt_id"],
        "case_id": case["case_id"],
        "lane": request["lane"],
        "repeat": request["repeat"],
        "status": status,
        "passed": passed,
        "metadata_complete": metadata_complete,
        "error": response_error,
        "output_sha256": output_sha256,
        "normalized_output_sha256": normalized_output_sha256,
        "output_characters": len(output_text),
        "hashes": {
            "image_sha256": case["input"]["image_sha256"],
            "raw_text_sha256": case["input"]["raw_text_sha256"],
            "prompt_sha256": request["instruction_sha256"],
            "effective_prompt_sha256": (
                _sha256_text(effective_prompt)
                if isinstance(effective_prompt, str)
                else None
            ),
            "request_sha256": _sha256_json(request),
            "response_sha256": _sha256_json(response) if response else None,
            "output_sha256": output_sha256,
            "normalized_output_sha256": normalized_output_sha256,
        },
        "quality": quality,
        "assertions": assertion_outcomes,
        "provenance": {
            "complete": metadata_complete,
            "missing_or_invalid": metadata_gaps,
            "provider": response,
        },
        "subprocess": subprocess_record,
        "output_path": str(output_path),
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    def output_hash(item, normalized=False):
        key = "normalized_output_sha256" if normalized else "output_sha256"
        return item.get(key) or item.get("hashes", {}).get(key)

    def metadata_complete(item):
        return item.get(
            "metadata_complete",
            item.get("provenance", {}).get("complete", False),
        )

    def attempt_passed(item):
        if "passed" in item:
            return bool(item["passed"])
        return (
            item.get("status") == "completed"
            and metadata_complete(item)
            and all(
                outcome.get("passed", False)
                for outcome in item.get("assertions", [])
            )
        )

    def numeric_range(values):
        numbers = [
            value
            for value in values
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
        ]
        return {
            "min": min(numbers) if numbers else None,
            "max": max(numbers) if numbers else None,
        }

    def provider_value(item, dotted):
        return _nested_value(
            item.get("provenance", {}).get("provider", {}),
            dotted,
        )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[(result["case_id"], result["lane"])].append(result)
    groups = []
    for (case_id, lane), attempts in sorted(grouped.items()):
        exact = Counter(output_hash(item) for item in attempts)
        normalized = Counter(output_hash(item, True) for item in attempts)
        pass_count = sum(attempt_passed(item) for item in attempts)
        complete = all(metadata_complete(item) for item in attempts)
        repeat_count = len(attempts)
        all_pass = pass_count == repeat_count and complete
        modal_fraction = max(exact.values()) / repeat_count
        finish_reasons = Counter(
            value
            for value in (
                provider_value(item, "generation.finish_reason")
                for item in attempts
            )
            if isinstance(value, str)
        )
        costs = [provider_value(item, "usage.cost_usd") for item in attempts]
        groups.append(
            {
                "case_id": case_id,
                "lane": lane,
                "repeats": repeat_count,
                "repeat_count": repeat_count,
                "pass_count": pass_count,
                "all_pass": all_pass,
                "provenance_complete": complete,
                "exact_output_hashes": dict(sorted(exact.items())),
                "output_hash_counts": dict(sorted(exact.items())),
                "exact_unique_outputs": len(exact),
                "unique_output_hash_count": len(exact),
                "normalized_output_hashes": dict(sorted(normalized.items())),
                "normalized_unique_outputs": len(normalized),
                "modal_fraction": modal_fraction,
                "modal_output_fraction": modal_fraction,
                "termination": {
                    "finish_reason_counts": dict(
                        sorted(finish_reasons.items())
                    ),
                    "eos_count": sum(
                        provider_value(item, "generation.eos") is True
                        for item in attempts
                    ),
                    "max_tokens_count": sum(
                        provider_value(item, "generation.max_tokens") is True
                        for item in attempts
                    ),
                },
                "ranges": {
                    "output_characters": numeric_range(
                        item.get("output_characters") for item in attempts
                    ),
                    "output_tokens": numeric_range(
                        provider_value(item, "generation.output_tokens")
                        for item in attempts
                    ),
                    "provider_wall_seconds": numeric_range(
                        provider_value(item, "usage.wall_seconds")
                        for item in attempts
                    ),
                    "subprocess_wall_seconds": numeric_range(
                        item.get("subprocess", {}).get("wall_seconds")
                        for item in attempts
                    ),
                    "raw_token_multiset_recall": numeric_range(
                        _nested_value(
                            item,
                            "quality.table_completion.raw_token_multiset_recall",
                        )
                        for item in attempts
                    ),
                    "axis_terms": numeric_range(
                        _nested_value(item, "quality.axis.terms")
                        for item in attempts
                    ),
                    "repeat_occurrences": numeric_range(
                        _nested_value(
                            item,
                            "quality.repetition.max_repeat.nonoverlap_occurrences",
                        )
                        or 0
                        for item in attempts
                    ),
                    "unsupported_repeat_occurrences": numeric_range(
                        _nested_value(
                            item,
                            "quality.repetition.max_unsupported_repeat.nonoverlap_occurrences",
                        )
                        or 0
                        for item in attempts
                    ),
                },
                "cost_usd": {
                    "known_total": sum(
                        value
                        for value in costs
                        if isinstance(value, (int, float))
                        and not isinstance(value, bool)
                    ),
                    "unknown_count": sum(value is None for value in costs),
                },
            }
        )
    passed_attempts = sum(attempt_passed(item) for item in results)
    return {
        "schema_version": SCHEMA_VERSION,
        "result_count": len(results),
        "attempt_count": len(results),
        "passed_attempt_count": passed_attempts,
        "groups": groups,
        "all_pass": bool(groups)
        and all(group["all_pass"] for group in groups),
        "provenance_complete": bool(groups)
        and all(group["provenance_complete"] for group in groups),
    }


def _git_probe_environment() -> dict[str, str]:
    return {
        "PATH": os.defpath,
        "LC_ALL": "C",
        "TZ": "UTC",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
    }


def _git_probe_executable() -> str | None:
    candidate = shutil.which("git", path=os.defpath) or shutil.which("git")
    if candidate is None:
        return None
    try:
        resolved = Path(candidate).resolve(strict=True)
    except OSError:
        return None
    return str(resolved) if resolved.is_file() else None


def _git_repository_context(path: Path) -> dict[str, Any]:
    declared_root = path.resolve(strict=True)
    environment = _git_probe_environment()
    executable = _git_probe_executable()
    if executable is None:
        return {
            "available": False,
            "root": declared_root,
            "environment": environment,
            "executable": None,
            "unavailable_reason": "git_probe_launch_failed",
        }
    try:
        completed = subprocess.run(
            [
                executable,
                "--no-pager",
                "-c",
                "core.fsmonitor=false",
                "rev-parse",
                "--show-toplevel",
            ],
            cwd=declared_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return {
            "available": False,
            "root": declared_root,
            "environment": environment,
            "executable": executable,
            "unavailable_reason": "git_probe_launch_failed",
        }
    if completed.returncode != 0 or not completed.stdout.strip():
        return {
            "available": False,
            "root": declared_root,
            "environment": environment,
            "executable": executable,
            "unavailable_reason": "git_toplevel_unavailable",
        }
    try:
        discovered_root = Path(completed.stdout.strip()).resolve(strict=True)
    except OSError:
        discovered_root = None
    if discovered_root != declared_root:
        return {
            "available": False,
            "root": declared_root,
            "environment": environment,
            "executable": executable,
            "unavailable_reason": "git_toplevel_mismatch",
        }
    return {
        "available": True,
        "root": declared_root,
        "environment": environment,
        "executable": executable,
        "unavailable_reason": None,
    }


def _run_git_probe(
    context: dict[str, Any],
    args: list[str],
    *,
    text: bool,
) -> subprocess.CompletedProcess:
    if context.get("available") is not True:
        raise BenchmarkError("Git probe requires an exact repository root")
    try:
        return subprocess.run(
            [
                context["executable"],
                "--no-pager",
                "-c",
                "core.fsmonitor=false",
                *args,
            ],
            cwd=context["root"],
            env=context["environment"],
            check=False,
            capture_output=True,
            text=text,
        )
    except OSError:
        raise BenchmarkError(
            "Git provenance probe could not be launched"
        ) from None


def _git_metadata(
    path: Path,
    *,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    git_context = context or _git_repository_context(path)
    if git_context.get("available") is not True:
        return {
            "available": False,
            "unavailable_reason": git_context.get("unavailable_reason"),
            "sha": None,
            "branch": None,
            "status": None,
        }
    values: dict[str, Any] = {
        "available": True,
        "unavailable_reason": None,
        "status": None,
    }
    for key, args in (
        ("sha", ["rev-parse", "HEAD"]),
        ("branch", ["branch", "--show-current"]),
    ):
        completed = _run_git_probe(git_context, args, text=True)
        values[key] = (
            completed.stdout.strip() if completed.returncode == 0 else None
        )
    return values


def _snapshot_git_diff(
    evidence_root: Path,
    *,
    context: dict[str, Any],
) -> dict[str, Any]:
    diff_path = evidence_root / "harness" / "git-diff.patch"
    diff_path.parent.mkdir(parents=True, exist_ok=True)
    # Git worktree operations can invoke repository-configured clean/process
    # filters through .gitattributes.  Do not execute them in an evidence
    # collection path.  The deterministic raw source-tree snapshot below
    # binds tracked, dirty, ignored, and untracked in-scope files instead.
    payload = b""
    available = False
    unavailable_reason = (
        "working_tree_git_diff_disabled"
        if context.get("available") is True
        else context.get("unavailable_reason")
    )
    returncode: int | None = None
    diff_path.write_bytes(payload)
    return {
        "available": available,
        "unavailable_reason": unavailable_reason,
        "evidence_path": str(diff_path.relative_to(evidence_root)),
        "returncode": returncode,
        "size_bytes": diff_path.stat().st_size,
        "sha256": file_sha256(str(diff_path)),
    }


def _argv_file_metadata(
    argv: list[str], cwd: str | None
) -> list[dict[str, Any]]:
    base = Path(cwd) if cwd else Path.cwd()
    source_suffixes = {".json", ".py", ".sh", ".toml", ".yaml", ".yml"}
    artifacts = []
    for index, value in enumerate(argv):
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = base / path
        if not path.is_file() or (
            index != 0 and path.suffix.casefold() not in source_suffixes
        ):
            continue
        artifacts.append(
            {
                "argv_index": index,
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(str(path)),
            }
        )
    return artifacts


def _argv_file_identity(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "argv_index": record["argv_index"],
            "path": record["path"],
            "size_bytes": record["size_bytes"],
            "sha256": record["sha256"],
        }
        for record in records
    ]


def _require_unchanged_argv_files(
    expected: list[dict[str, Any]],
    spec: dict[str, Any],
) -> None:
    actual = _argv_file_metadata(spec["argv"], spec.get("cwd"))
    if _argv_file_identity(actual) != _argv_file_identity(expected):
        raise BenchmarkError(
            "provider argv file drift detected; refusing to mix code "
            "identities in one benchmark run"
        )


def _require_unchanged_external_executor(
    records: list[dict[str, Any]],
) -> None:
    for record in records:
        source = Path(record["source_path"])
        try:
            size_bytes = source.stat().st_size
        except OSError as exc:
            raise BenchmarkError(
                "external executor source disappeared before spawn"
            ) from exc
        if (
            size_bytes != record["size_bytes"]
            or file_sha256(str(source)) != record["sha256"]
        ):
            raise BenchmarkError(
                "external executor source drift detected before spawn"
            )


def _snapshot_file(
    source: Path,
    evidence_root: Path,
    relative_path: Path,
    *,
    source_bytes: bytes | None = None,
) -> dict[str, Any]:
    destination = evidence_root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source_bytes is None:
        shutil.copyfile(source, destination)
    else:
        destination.write_bytes(source_bytes)
    return {
        "source_path": str(source.resolve()),
        "evidence_path": str(relative_path),
        "size_bytes": destination.stat().st_size,
        "sha256": file_sha256(str(destination)),
    }


def _scoped_source_files(root: Path) -> list[Path]:
    resolved_root = root.resolve(strict=True)
    files: list[Path] = []

    def visit(path: Path) -> None:
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise BenchmarkError(
                "source-tree evidence changed during enumeration"
            ) from exc
        relative = path.relative_to(resolved_root)
        mode = metadata.st_mode
        if stat.S_ISLNK(mode):
            raise BenchmarkError(
                "source-tree evidence contains an unsafe symlink: "
                + relative.as_posix()
            )
        if stat.S_ISREG(mode):
            if path.suffix.casefold() not in _SOURCE_TREE_CACHE_SUFFIXES:
                files.append(path)
            return
        if stat.S_ISDIR(mode):
            if path.name in _SOURCE_TREE_CACHE_DIRECTORIES:
                return
            try:
                children = sorted(path.iterdir(), key=lambda item: item.name)
            except OSError as exc:
                raise BenchmarkError(
                    "could not enumerate source-tree evidence: "
                    + relative.as_posix()
                ) from exc
            for child in children:
                visit(child)
            return
        raise BenchmarkError(
            "source-tree evidence contains a special file: "
            + relative.as_posix()
        )

    for raw_scope in _GIT_PROVENANCE_PATHS:
        scope = resolved_root / raw_scope
        if not os.path.lexists(scope):
            continue
        visit(scope)
    return sorted(
        files,
        key=lambda item: item.relative_to(resolved_root).as_posix(),
    )


def _read_stable_source_file(path: Path, *, root: Path) -> bytes:
    resolved_root = root.resolve(strict=True)
    try:
        relative = path.relative_to(resolved_root)
    except ValueError as exc:
        raise BenchmarkError("source-tree evidence escaped its root") from exc
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise BenchmarkError(
                    "source-tree evidence is not a regular file: "
                    + relative.as_posix()
                )
            payload = handle.read()
            after = os.fstat(handle.fileno())
        current = path.lstat()
    except BenchmarkError:
        raise
    except OSError as exc:
        raise BenchmarkError(
            "could not read source-tree evidence: " + relative.as_posix()
        ) from exc
    signatures = {
        (
            item.st_dev,
            item.st_ino,
            item.st_size,
            item.st_mtime_ns,
        )
        for item in (before, after, current)
    }
    if len(signatures) != 1 or len(payload) != before.st_size:
        raise BenchmarkError(
            "source-tree evidence changed while being read: "
            + relative.as_posix()
        )
    return payload


def _snapshot_source_tree(
    root: Path,
    evidence_root: Path,
) -> dict[str, Any]:
    resolved_root = root.resolve(strict=True)
    resolved_evidence = evidence_root.resolve(strict=True)
    for raw_scope in _GIT_PROVENANCE_PATHS:
        scope = resolved_root / raw_scope
        if resolved_evidence == scope or resolved_evidence.is_relative_to(
            scope
        ):
            raise BenchmarkError(
                "source-tree evidence destination overlaps its source scope"
            )

    source_files = _scoped_source_files(resolved_root)
    snapshot_root = evidence_root / "source-tree" / "files"
    records = []
    for source in source_files:
        relative = source.relative_to(resolved_root)
        payload = _read_stable_source_file(source, root=resolved_root)
        destination = snapshot_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        records.append(
            {
                "path": relative.as_posix(),
                "evidence_path": destination.relative_to(
                    evidence_root
                ).as_posix(),
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )

    observed_again = _scoped_source_files(resolved_root)
    expected_paths = [
        item.relative_to(resolved_root).as_posix() for item in source_files
    ]
    observed_paths = [
        item.relative_to(resolved_root).as_posix() for item in observed_again
    ]
    if observed_paths != expected_paths:
        raise BenchmarkError(
            "source-tree evidence changed during snapshot creation"
        )
    for source, record in zip(observed_again, records, strict=True):
        payload = _read_stable_source_file(source, root=resolved_root)
        if (
            len(payload) != record["size_bytes"]
            or hashlib.sha256(payload).hexdigest() != record["sha256"]
        ):
            raise BenchmarkError(
                "source-tree evidence changed during snapshot creation"
            )

    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "scope_paths": list(_GIT_PROVENANCE_PATHS),
        "excluded_directory_names": sorted(_SOURCE_TREE_CACHE_DIRECTORIES),
        "excluded_file_suffixes": sorted(_SOURCE_TREE_CACHE_SUFFIXES),
        "file_count": len(records),
        "files": records,
    }
    inventory = {
        **unsigned,
        "aggregate_sha256": _sha256_json(unsigned),
    }
    inventory_path = evidence_root / "source-tree" / "inventory.json"
    _write_json(inventory_path, inventory)
    return {
        **inventory,
        "inventory_evidence_path": inventory_path.relative_to(
            evidence_root
        ).as_posix(),
        "inventory_file_sha256": file_sha256(str(inventory_path)),
    }


def _snapshot_run_evidence(
    dest_path: Path,
    manifest_path: Path,
    provider_spec_path: Path,
    spec: dict[str, Any],
    git_context: dict[str, Any],
    *,
    manifest_source_bytes: bytes,
    provider_spec_source_bytes: bytes,
) -> dict[str, Any]:
    evidence_root = dest_path / "evidence"
    manifest = _snapshot_file(
        manifest_path,
        evidence_root,
        Path("manifest.jsonl"),
        source_bytes=manifest_source_bytes,
    )
    provider_spec = _snapshot_file(
        provider_spec_path,
        evidence_root,
        Path("provider-spec.json"),
        source_bytes=provider_spec_source_bytes,
    )
    argv_files = _argv_file_metadata(spec["argv"], spec.get("cwd"))
    for record in argv_files:
        if record["argv_index"] == 0:
            continue
        source = Path(record["path"])
        relative = Path("provider") / (
            f"argv-{record['argv_index']:03d}-{source.name}"
        )
        snapshot = _snapshot_file(source, evidence_root, relative)
        record["evidence_path"] = snapshot["evidence_path"]
    harness_files = []
    for source in (
        Path(__file__).resolve(),
        Path(__file__).with_name("ocr_vllm.py").resolve(),
        Path(__file__).with_name("commands.py").resolve(),
    ):
        harness_files.append(
            _snapshot_file(
                source,
                evidence_root,
                Path("harness") / source.name,
            )
        )
    git_diff = _snapshot_git_diff(evidence_root, context=git_context)
    source_tree = _snapshot_source_tree(git_context["root"], evidence_root)
    return {
        "manifest": manifest,
        "provider_spec": provider_spec,
        "argv_files": argv_files,
        "harness_files": harness_files,
        "git_diff": git_diff,
        "source_tree": source_tree,
    }


def _experiment_identity_payload(
    *,
    manifest_file_sha256: str,
    provider_spec_file_sha256: str,
    environment: dict[str, str],
    argv_files: list[dict[str, Any]],
    harness_files: list[dict[str, Any]],
    git_metadata: dict[str, Any],
    git_diff: dict[str, Any],
    source_tree_sha256: str,
    external_identity: dict[str, Any] | None,
) -> dict[str, Any]:
    repository_identity = {
        "available": git_metadata.get("available"),
        "unavailable_reason": git_metadata.get("unavailable_reason"),
        "sha": git_metadata.get("sha"),
        "git_diff": {
            "available": git_diff.get("available"),
            "unavailable_reason": git_diff.get("unavailable_reason"),
            "returncode": git_diff.get("returncode"),
            "sha256": git_diff.get("sha256"),
        },
    }
    return {
        "manifest_file_sha256": manifest_file_sha256,
        "provider_spec_file_sha256": provider_spec_file_sha256,
        "environment": environment,
        "argv_files": _argv_file_identity(argv_files),
        "harness_files": [record["sha256"] for record in harness_files],
        "git_diff_sha256": git_diff["sha256"],
        "source_tree_sha256": source_tree_sha256,
        "repository": repository_identity,
        "external_identity": external_identity,
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _external_gate(
    *,
    spec: dict[str, Any],
    manifest: list[dict[str, Any]],
    manifest_source_bytes: bytes,
    plan_path: str | os.PathLike[str] | None,
    approval_path: str | os.PathLike[str] | None,
    selected_lanes: list[str],
    repeats: int,
) -> dict[str, Any]:
    if plan_path is None or approval_path is None:
        raise BenchmarkError(
            "external providers require both --external-plan and "
            "--external-approval"
        )
    try:
        from .ocr_external import (
            ExternalApprovalError,
            ExternalPlanError,
            build_external_execution_binding,
            validate_external_upload_approval,
            validate_external_upload_plan,
        )
        from .ocr_external_setup import (
            ExternalSetupError,
            external_executor_bundle_identity,
            external_provider_spec_sha256,
            validate_external_provider_spec,
        )

        plan_source = _resolve_source_path(plan_path, "external plan")
        approval_source = _resolve_source_path(
            approval_path,
            "external approval",
        )
        plan_source_bytes = _read_bytes(plan_source)
        approval_source_bytes = _read_bytes(approval_source)
        plan = validate_external_upload_plan(
            _decode_json(plan_source_bytes, plan_source)
        )
        now = datetime.now(UTC).replace(microsecond=0).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        approval = validate_external_upload_approval(
            plan,
            _decode_json(approval_source_bytes, approval_source),
            now=now,
        )
        normalized_spec = validate_external_provider_spec(spec)
        spec_hash = external_provider_spec_sha256(normalized_spec)
        bundle = external_executor_bundle_identity(
            normalized_spec["argv"][0]
        )
        if plan["executor"]["provider_spec_sha256"] != spec_hash:
            raise BenchmarkError(
                "external plan is bound to a different provider spec"
            )
        if (
            plan["executor"]["adapter_bundle_sha256"]
            != bundle["bundle_sha256"]
        ):
            raise BenchmarkError(
                "external executor or Python identity differs from plan"
            )
        if (
            normalized_spec["external_policy"][
                "executor_bundle_sha256"
            ]
            != bundle["bundle_sha256"]
        ):
            raise BenchmarkError(
                "external provider spec executor identity is stale"
            )
        if plan["manifest_sha256"] != hashlib.sha256(
            manifest_source_bytes
        ).hexdigest():
            raise BenchmarkError(
                "external plan is bound to a different manifest file"
            )
        plan_lanes = [lane["lane_id"] for lane in plan["lanes"]]
        if plan_lanes != selected_lanes:
            raise BenchmarkError(
                "external plan lanes do not exactly match the benchmark"
            )
        if plan["execution"]["repeats"] != repeats:
            raise BenchmarkError(
                "external plan repeats do not exactly match the benchmark"
            )
        expected_lanes = []
        for lane_id in selected_lanes:
            lane = normalized_spec["lanes"][lane_id]
            expected_lanes.append(
                {
                    "lane_id": lane_id,
                    "prompt": lane["prompt"],
                    "prompt_sha256": hashlib.sha256(
                        lane["prompt"].encode("utf-8")
                    ).hexdigest(),
                    "prompt_mode": lane["settings"]["prompt_mode"],
                    "include_raw_text": lane["include_raw_text"],
                }
            )
        if plan["lanes"] != expected_lanes:
            raise BenchmarkError(
                "external plan prompt or data scope differs from spec"
            )
        include_raw_text = any(
            lane["include_raw_text"] for lane in plan["lanes"]
        )
        expected_cases = []
        for case in manifest:
            expected = {
                "case_id": case["case_id"],
                "image_sha256": case["input"]["image_sha256"],
                "dimensions": case["input"]["dimensions"],
                "media_type": "image/png",
            }
            if include_raw_text:
                expected["raw_text_sha256"] = case["input"][
                    "raw_text_sha256"
                ]
            expected_cases.append(expected)
        if plan["cases"] != expected_cases:
            raise BenchmarkError(
                "external plan cases do not exactly match the manifest"
            )
        if plan["provider_profile"]["profile_id"] != spec["provider_id"]:
            raise BenchmarkError(
                "external plan profile differs from provider spec"
            )
        binding = build_external_execution_binding(
            plan,
            approval,
            now=now,
        )
    except BenchmarkError:
        raise
    except (
        ExternalApprovalError,
        ExternalPlanError,
        ExternalSetupError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        raise BenchmarkError(
            "external plan or approval validation failed"
        ) from exc
    return {
        "plan_source": plan_source,
        "plan_source_bytes": plan_source_bytes,
        "plan_file_sha256": hashlib.sha256(plan_source_bytes).hexdigest(),
        "approval_source": approval_source,
        "approval_source_bytes": approval_source_bytes,
        "approval_file_sha256": hashlib.sha256(
            approval_source_bytes
        ).hexdigest(),
        "plan": plan,
        "approval": approval,
        "binding": binding,
        "executor_bundle": bundle,
    }


def run_benchmark(
    *,
    manifest_path: str | os.PathLike[str],
    input_root: str | os.PathLike[str],
    provider_spec_path: str | os.PathLike[str],
    dest_dir: str | os.PathLike[str],
    expected_provider_spec_file_sha256: str | None = None,
    lanes: list[str] | tuple[str, ...] | None = None,
    repeats: int = 5,
    allow_external: bool = False,
    external_plan_path: str | os.PathLike[str] | None = None,
    external_approval_path: str | os.PathLike[str] | None = None,
    out=None,
) -> dict[str, Any]:
    if (
        not isinstance(repeats, int)
        or isinstance(repeats, bool)
        or repeats <= 0
    ):
        raise BenchmarkError("repeats must be a positive integer")
    manifest_source = _resolve_source_path(manifest_path, "manifest")
    provider_spec_source = _resolve_source_path(
        provider_spec_path,
        "provider specification",
    )
    manifest_source_bytes = _read_bytes(manifest_source)
    provider_spec_source_bytes = _read_bytes(provider_spec_source)
    if expected_provider_spec_file_sha256 is not None:
        if (
            not isinstance(expected_provider_spec_file_sha256, str)
            or not re.fullmatch(
                r"[0-9a-f]{64}", expected_provider_spec_file_sha256
            )
        ):
            raise BenchmarkError(
                "expected provider specification SHA-256 is invalid"
            )
        observed_provider_spec_hash = hashlib.sha256(
            provider_spec_source_bytes
        ).hexdigest()
        if observed_provider_spec_hash != expected_provider_spec_file_sha256:
            raise BenchmarkError(
                "provider specification file differs from the caller-bound "
                "SHA-256"
            )
    manifest = load_manifest(
        manifest_source,
        _source_bytes=manifest_source_bytes,
    )
    spec = load_provider_spec(
        provider_spec_source,
        _source_bytes=provider_spec_source_bytes,
    )
    selected_lanes = list(spec["lanes"] if lanes is None else lanes)
    if not selected_lanes:
        raise BenchmarkError("lanes must be unique and nonempty")
    selected_lanes = [
        _safe_component(lane, f"lanes[{index}]")
        for index, lane in enumerate(selected_lanes, 1)
    ]
    if len(set(selected_lanes)) != len(selected_lanes):
        raise BenchmarkError("lanes must be unique and nonempty")
    unknown_lanes = set(selected_lanes) - set(spec["lanes"])
    if unknown_lanes:
        raise BenchmarkError(
            "Provider does not define lanes: "
            f"{', '.join(sorted(unknown_lanes))}"
        )
    external = spec.get("egress", "none") == "external"
    if external and not allow_external:
        raise BenchmarkError(
            "External provider refused without explicit --allow-external"
        )
    if not external and (
        external_plan_path is not None
        or external_approval_path is not None
    ):
        raise BenchmarkError(
            "external plan and approval arguments require an external "
            "provider spec"
        )
    external_execution = None
    if external:
        external_execution = _external_gate(
            spec=spec,
            manifest=manifest,
            manifest_source_bytes=manifest_source_bytes,
            plan_path=external_plan_path,
            approval_path=external_approval_path,
            selected_lanes=selected_lanes,
            repeats=repeats,
        )
    dest_path = Path(dest_dir).expanduser()
    prepared = prepare_run(manifest, input_root, dest_path)
    manifest_hash = _sha256_json(manifest)
    spec_hash = _sha256_json(spec)
    manifest_file_hash = hashlib.sha256(manifest_source_bytes).hexdigest()
    spec_file_hash = hashlib.sha256(provider_spec_source_bytes).hexdigest()
    repo_root = Path(__file__).resolve().parents[3]
    git_context = _git_repository_context(repo_root)
    git_metadata = _git_metadata(repo_root, context=git_context)
    evidence = _snapshot_run_evidence(
        dest_path,
        manifest_source,
        provider_spec_source,
        spec,
        git_context,
        manifest_source_bytes=manifest_source_bytes,
        provider_spec_source_bytes=provider_spec_source_bytes,
    )
    external_protocol_paths = None
    external_identity = None
    if external_execution is not None:
        evidence_root = dest_path / "evidence"
        evidence["external_plan"] = _snapshot_file(
            external_execution["plan_source"],
            evidence_root,
            Path("external/upload-plan.json"),
            source_bytes=external_execution["plan_source_bytes"],
        )
        evidence["external_approval"] = _snapshot_file(
            external_execution["approval_source"],
            evidence_root,
            Path("external/upload-approval.json"),
            source_bytes=external_execution["approval_source_bytes"],
        )
        external_protocol_paths = {
            "provider_spec": (
                evidence_root / evidence["provider_spec"]["evidence_path"]
            ).resolve(),
            "plan": (
                evidence_root / evidence["external_plan"]["evidence_path"]
            ).resolve(),
            "approval": (
                evidence_root
                / evidence["external_approval"]["evidence_path"]
            ).resolve(),
            "budget_ledger": (
                dest_path / "external-budget-ledger.json"
            ).resolve(),
        }
        external_execution = _external_gate(
            spec=spec,
            manifest=manifest,
            manifest_source_bytes=manifest_source_bytes,
            plan_path=external_protocol_paths["plan"],
            approval_path=external_protocol_paths["approval"],
            selected_lanes=selected_lanes,
            repeats=repeats,
        )
        executor_sources = []
        for record in external_execution["executor_bundle"]["files"]:
            module = importlib.import_module(record["module"])
            source = Path(module.__file__).resolve(strict=True)
            if (
                source.stat().st_size != record["size_bytes"]
                or file_sha256(str(source)) != record["sha256"]
            ):
                raise BenchmarkError(
                    "external executor source changed during snapshot"
                )
            snapshot = _snapshot_file(
                source,
                evidence_root,
                Path("external/executor")
                / (record["module"].replace(".", "-") + ".py"),
            )
            executor_sources.append({**record, **snapshot})
        evidence["external_executor"] = {
            "bundle_sha256": external_execution["executor_bundle"][
                "bundle_sha256"
            ],
            "python_executable": external_execution[
                "executor_bundle"
            ]["python_executable"],
            "files": executor_sources,
        }
        external_identity = {
            "plan_sha256": external_execution["plan"]["plan_sha256"],
            "approval_sha256": external_execution["approval"][
                "approval_sha256"
            ],
            "provider_profile_sha256": external_execution["plan"][
                "provider_profile_sha256"
            ],
            "executor_bundle_sha256": external_execution[
                "executor_bundle"
            ]["bundle_sha256"],
            "plan_file_sha256": external_execution["plan_file_sha256"],
            "approval_file_sha256": external_execution[
                "approval_file_sha256"
            ],
        }
    # This is the first credential read. The external gate immediately above
    # rehashes the approved interpreter and executor before values enter the
    # child environment mapping.
    provider_environment = _provider_environment(spec)
    recorded_environment = _recorded_provider_environment(
        provider_environment,
        spec,
    )
    experiment_identity = _experiment_identity_payload(
        manifest_file_sha256=manifest_file_hash,
        provider_spec_file_sha256=spec_file_hash,
        environment=recorded_environment,
        argv_files=evidence["argv_files"],
        harness_files=evidence["harness_files"],
        git_metadata=git_metadata,
        git_diff=evidence["git_diff"],
        source_tree_sha256=evidence["source_tree"]["aggregate_sha256"],
        external_identity=external_identity,
    )
    experiment_hash = _sha256_json(experiment_identity)
    run_record = {
        "schema_version": SCHEMA_VERSION,
        "started_at": datetime.now(UTC).isoformat(),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
        },
        "repository": {
            "root": str(repo_root),
            **git_metadata,
        },
        "invocation": list(sys.argv),
        "evidence": evidence,
        "harness": {
            "package_version": _package_version("sneeze-plugin-tpn"),
            "manifest_path": str(manifest_source),
            "manifest_sha256": manifest_hash,
            "manifest_file_sha256": manifest_file_hash,
            "provider_spec_path": str(provider_spec_source),
            "provider_spec_sha256": spec_hash,
            "provider_spec_file_sha256": spec_file_hash,
            "input_root": str(Path(input_root).resolve()),
            "dest_dir": str(dest_path.resolve()),
            "lanes": selected_lanes,
            "repeats": repeats,
            "allow_external": allow_external,
            "external_identity": external_identity,
            "experiment_sha256": experiment_hash,
        },
        "provider": {
            "provider_id": spec["provider_id"],
            "argv": spec["argv"],
            "argv_files": evidence["argv_files"],
            "cwd": spec.get("cwd"),
            "egress": spec.get("egress", "none"),
            "external_execution": (
                external_execution["binding"]
                if external_execution is not None
                else None
            ),
            "environment": recorded_environment,
            "environment_sha256": _sha256_json(recorded_environment),
            "lanes": {
                lane: spec["lanes"][lane] for lane in selected_lanes
            },
            "model": spec["model"],
            "secret_env_names": spec.get("secret_env_names", []),
            "timeout_seconds": spec.get("timeout_seconds", 1800),
        },
    }
    _write_json(dest_path / "run.json", run_record)
    all_results = []
    for lane in selected_lanes:
        for repeat in range(1, repeats + 1):
            try:
                _require_unchanged_argv_files(
                    evidence["argv_files"], spec
                )
                if external_execution is not None:
                    _require_unchanged_external_executor(
                        evidence["external_executor"]["files"]
                    )
            except BenchmarkError as exc:
                _write_json(
                    dest_path / "provider-drift.json",
                    {
                        "detected_at": datetime.now(UTC).isoformat(),
                        "lane": lane,
                        "repeat": repeat,
                        "error": str(exc),
                    },
                )
                raise
            if out:
                out(
                    f"Running {spec['provider_id']} lane={lane} "
                    f"repeat={repeat}/{repeats}"
                )
            attempt_dir = dest_path / lane / f"repeat-{repeat:03d}"
            artifacts_dir = attempt_dir / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            requests = [
                _build_request(
                    item,
                    spec,
                    spec_hash,
                    (
                        external_execution["plan"]["manifest_sha256"]
                        if external_execution is not None
                        else manifest_hash
                    ),
                    lane,
                    repeat,
                    experiment_hash,
                )
                for item in prepared
            ]
            requests_path = attempt_dir / "requests.jsonl"
            responses_path = attempt_dir / "responses.jsonl"
            _write_jsonl(requests_path, requests)
            argv = [
                *spec["argv"],
                "--requests-jsonl",
                str(requests_path.resolve()),
                "--responses-jsonl",
                str(responses_path.resolve()),
                "--artifacts-dir",
                str(artifacts_dir.resolve()),
            ]
            if external_protocol_paths is not None:
                argv.extend(
                    [
                        "--external-provider-spec",
                        str(external_protocol_paths["provider_spec"]),
                        "--external-plan",
                        str(external_protocol_paths["plan"]),
                        "--external-approval",
                        str(external_protocol_paths["approval"]),
                        "--external-budget-ledger",
                        str(external_protocol_paths["budget_ledger"]),
                    ]
                )
            started = time.monotonic()
            returncode = -1
            execution_error = None
            stdout_path = attempt_dir / "stdout.log"
            stderr_path = attempt_dir / "stderr.log"
            credential_values = _provider_credential_values(
                spec,
                provider_environment,
            )
            replacements = _credential_replacements(credential_values)
            log_redactions: dict[str, int] = {}
            log_scan_complete = True
            with (
                tempfile.TemporaryFile() as stdout_capture,
                tempfile.TemporaryFile() as stderr_capture,
            ):
                returncode, execution_error = _run_provider_process(
                    argv,
                    cwd=spec.get("cwd"),
                    environment=provider_environment,
                    stdout_handle=stdout_capture,
                    stderr_handle=stderr_capture,
                    timeout_seconds=spec.get("timeout_seconds", 1800),
                )
                for label, capture, path in (
                    ("stdout", stdout_capture, stdout_path),
                    ("stderr", stderr_capture, stderr_path),
                ):
                    count, complete = _persist_sanitized_capture(
                        capture,
                        path,
                        replacements,
                    )
                    log_redactions[label] = count
                    log_scan_complete = log_scan_complete and complete
            if execution_error:
                execution_error, count = _redact_credential_text(
                    execution_error,
                    credential_values,
                )
                log_redactions["execution_error"] = count
            sanitization = _sanitize_provider_evidence(
                credential_values=credential_values,
                replacements=replacements,
                log_redactions=log_redactions,
                log_scan_complete=log_scan_complete,
                responses_path=responses_path,
                artifacts_dir=artifacts_dir,
            )
            try:
                _require_unchanged_argv_files(
                    evidence["argv_files"], spec
                )
                if external_execution is not None:
                    _require_unchanged_external_executor(
                        evidence["external_executor"]["files"]
                    )
            except BenchmarkError as exc:
                _write_json(
                    dest_path / "provider-drift.json",
                    {
                        "detected_at": datetime.now(UTC).isoformat(),
                        "lane": lane,
                        "repeat": repeat,
                        "error": str(exc),
                    },
                )
                raise
            subprocess_record = {
                "argv": argv,
                "returncode": returncode,
                "wall_seconds": round(time.monotonic() - started, 6),
                "error": execution_error,
                "stdout_sha256": file_sha256(str(stdout_path)),
                "stderr_sha256": file_sha256(str(stderr_path)),
                **sanitization,
            }
            responses: list[dict[str, Any]] = []
            parse_errors = [execution_error] if execution_error else []
            if sanitization["credential_leak_detected"]:
                parse_errors.append(
                    "provider credential leak detected; persisted evidence "
                    "was redacted"
                )
            if not sanitization["credential_scan_complete"]:
                parse_errors.append(
                    "provider evidence credential scan was incomplete"
                )
            if responses_path.exists():
                try:
                    responses = _read_jsonl(responses_path)
                except BenchmarkError as exc:
                    parse_errors.append(str(exc))
            else:
                responses_path.touch()
                parse_errors.append(
                    "provider did not create responses JSONL"
                )
            parse_error = "; ".join(parse_errors) or None
            subprocess_record["requests_jsonl_sha256"] = file_sha256(
                str(requests_path)
            )
            subprocess_record["responses_jsonl_sha256"] = file_sha256(
                str(responses_path)
            )
            by_id = {}
            duplicate_ids = set()
            invalid_ids = []
            for response_index, response in enumerate(responses, 1):
                attempt_id = response.get("attempt_id")
                if not isinstance(attempt_id, str) or not re.fullmatch(
                    r"[0-9a-f]{64}", attempt_id
                ):
                    invalid_ids.append(response_index)
                    continue
                if attempt_id in by_id:
                    duplicate_ids.add(attempt_id)
                else:
                    by_id[attempt_id] = response
            expected_ids = {request["attempt_id"] for request in requests}
            unexpected = set(by_id) - expected_ids
            structural_errors = [parse_error] if parse_error else []
            if invalid_ids:
                structural_errors.append(
                    "invalid response IDs at rows: "
                    + ", ".join(map(str, invalid_ids))
                )
            if duplicate_ids:
                response_ids = ", ".join(sorted(map(str, duplicate_ids)))
                structural_errors.append(
                    f"duplicate response IDs: {response_ids}"
                )
            if unexpected:
                response_ids = ", ".join(sorted(map(str, unexpected)))
                structural_errors.append(
                    f"unexpected response IDs: {response_ids}"
                )
            structural_error = "; ".join(structural_errors) or None
            for request, item in zip(requests, prepared, strict=True):
                response = by_id.get(request["attempt_id"])
                error = structural_error
                if response is None:
                    error = error or "provider response is missing"
                result = _result_for_response(
                    request,
                    response,
                    item,
                    spec,
                    attempt_dir,
                    subprocess_record,
                    error,
                )
                all_results.append(result)
    evaluator_evidence = _write_evaluator_evidence(dest_path, prepared)
    _write_jsonl(dest_path / "results.jsonl", all_results)
    summary = summarize_results(all_results)
    summary["provider_id"] = spec["provider_id"]
    summary["manifest_sha256"] = manifest_hash
    summary["manifest_file_sha256"] = manifest_file_hash
    summary["provider_spec_sha256"] = spec_hash
    summary["provider_spec_file_sha256"] = spec_file_hash
    summary["evaluator_evidence_sha256"] = _sha256_json(
        evaluator_evidence
    )
    summary["finished_at"] = datetime.now(UTC).isoformat()
    _write_json(dest_path / "summary.json", summary)
    return summary
