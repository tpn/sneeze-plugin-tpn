"""Portable, content-hashed OCR pipeline execution profiles.

The records in this module describe *qualified execution*, not model setup.
They intentionally contain no host paths, GPU UUIDs, hostnames, credentials,
or mutable model references.  A profile binds an immutable pipeline to one
portable hardware class, exact per-stage model/runtime selectors, and the
settings qualified for the 25%, 50%, and maximum-safe memory tiers.

This module is safe during CLI discovery: it imports only the standard library
at module import time.  Hardware probing is delegated lazily to
``sneeze.tpn.ocr_hardware`` when that optional helper is available.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path, PureWindowsPath
from typing import Any

from .ocr_path_security import (
    TrustedPathError,
    validate_trusted_directory,
    validate_trusted_file,
)
from .ocr_profile_trust import TRUSTED_QUALIFIED_PROFILE_SHA256S

SCHEMA_VERSION = 1
PIPELINE_KIND = "ocr_pipeline_definition"
PROFILE_KIND = "ocr_execution_profile"
MEMORY_TIERS = ("25", "50", "max")
QUALIFICATION_STATES = frozenset({"provisional", "qualified"})
BATCH_SEMANTICS = frozenset({"page_batch", "layout_block_batch"})
MINIMUM_QUALIFICATION_REPEATS = 5
EXECUTION_QUALIFICATION_POLICY = {
    "schema_version": 1,
    "policy_id": "local-ocr-gpu-execution-v1",
    "scope": "hardware-runtime-and-settings",
    "backend_model_quality_is_separate": True,
    "exact_model_snapshot_required": True,
    "exact_runtime_required": True,
    "exact_implementation_required": True,
    "effective_settings_attestation_required": True,
    "all_supported_tiers_require_passing_evidence": True,
    "minimum_qualification_repeats": MINIMUM_QUALIFICATION_REPEATS,
    "source_control_trust_required": True,
}
_GIB = 1024**3
_MAX_PROFILE_FILE_BYTES = 1024 * 1024
_MAX_PROFILE_DIRECTORY_ENTRIES = 256
_MEMORY_TIER_POLICIES = {
    "dedicated": {"max_fraction": 0.9, "reserve_bytes": 2 * _GIB},
    "unified": {"max_fraction": 0.75, "reserve_bytes": 16 * _GIB},
}
_ID_RE = re.compile(r"^[a-z0-9]+(?:[a-z0-9._-]*[a-z0-9])?$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_SECRET_FIELD_RE = re.compile(
    r"(?:^|_)(?:api_?key|auth(?:orization)?|bearer|cookie|credential|"
    r"password|private_?key|secret|access_?token)(?:_|$)",
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
_HOST_SPECIFIC_HARDWARE_FIELDS = frozenset(
    {
        "bus_id",
        "device_index",
        "hostname",
        "index",
        "pci_bus_id",
        "serial",
        "uuid",
    }
)
_HARDWARE_SELECTOR_KEYS = frozenset(
    {
        "schema_version",
        "vendor",
        "product_name",
        "pci_device_id",
        "compute_capability",
        "memory_model",
        "total_memory_bytes",
    }
)
_PROFILE_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "profile_id",
        "display_name",
        "pipeline_id",
        "pipeline_definition_sha256",
        "hardware_selector",
        "stages",
        "memory_tiers",
        "qualification",
        "profile_sha256",
    }
)
_STAGE_KEYS = frozenset(
    {
        "backend_id",
        "capability",
        "model",
        "batch_semantics",
        "runtime_selector",
    }
)
_MODEL_KEYS = frozenset(
    {"requested", "revision", "snapshot_files_sha256"}
)
_RUNTIME_SELECTOR_KEYS = frozenset(
    {
        "python",
        "engine",
        "engine_version",
        "torch",
        "torch_cuda",
        "packages",
        "attention_backend",
        "dtype",
        "implementation_sha256",
        "installed_distributions_count",
        "installed_distributions_sha256",
    }
)
_TIER_KEYS = frozenset(
    {
        "supported",
        "reason",
        "budget_fraction",
        "budget_bytes",
        "reserve_bytes",
        "stages",
    }
)
_TIER_STAGE_KEYS = frozenset(
    {
        "settings",
        "observed_peak_memory_bytes",
        "confirmed_pages_per_second",
    }
)
_DEEPSEEK_LEGACY_TIER_SETTING_KEYS = frozenset(
    {
        "attention_backend",
        "batch_size",
        "enable_chunked_prefill",
        "engine_overhead_headroom_bytes",
        "gpu_memory_utilization",
        "gpu_memory_utilization_denominator_bytes",
        "max_num_batched_tokens",
        "max_num_seqs",
        "max_tokens",
        "memory_budget_bytes",
        "memory_tier",
        "mm_encoder_attn_backend",
        "ngram_size",
        "temperature",
        "window_size",
    }
)
_DEEPSEEK_RUNTIME_CONTROL_KEYS = frozenset({"moe_backend"})
_DEEPSEEK_TIER_SETTING_KEYS = (
    _DEEPSEEK_LEGACY_TIER_SETTING_KEYS | _DEEPSEEK_RUNTIME_CONTROL_KEYS
)
_QUALIFIED_DEEPSEEK_MOE_BACKEND = "triton"
_DEEPSEEK_LEGACY_IMPLEMENTATION_SHA256 = (
    "1572663ce37a0735db51abb66454a0b1c5cbd8db7dc8bb96153698342f2e58be"
)
_MINERU_TIER_SETTING_KEYS = frozenset(
    {
        "backend",
        "batch_size",
        "image_analysis",
        "memory_budget_bytes",
        "memory_tier",
    }
)
_QUALIFICATION_KEYS = frozenset(
    {"state", "policy_sha256", "required_suites", "evidence"}
)
_EVIDENCE_KEYS = frozenset(
    {
        "suite_id",
        "memory_tier",
        "manifest_sha256",
        "summary_sha256",
        "experiment_sha256",
        "repeats",
        "all_pass",
        "provenance_complete",
    }
)

if not isinstance(TRUSTED_QUALIFIED_PROFILE_SHA256S, frozenset) or any(
    not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest)
    for digest in TRUSTED_QUALIFIED_PROFILE_SHA256S
):
    raise RuntimeError("trusted OCR profile registry contains invalid hashes")


class OcrProfileError(ValueError):
    """Raised when an OCR pipeline or execution profile is invalid."""


class OcrProfileNotFoundError(OcrProfileError):
    """Raised when no profile has the requested identifier."""


class OcrProfileSelectionError(OcrProfileError):
    """Raised when automatic profile selection cannot resolve one profile."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def content_sha256(value: Any) -> str:
    """Return the SHA-256 of canonical, whitespace-free JSON."""

    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise OcrProfileError(
                f"duplicate JSON field in OCR profile: {key}"
            )
        result[key] = value
    return result


def qualification_policy_sha256() -> str:
    """Return the digest of the source-controlled execution policy."""

    return content_sha256(EXECUTION_QUALIFICATION_POLICY)


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OcrProfileError(f"{label} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        message = "; ".join(details)
        raise OcrProfileError(f"{label} fields are invalid: {message}")


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OcrProfileError(f"{label} must be a nonempty string")
    if "\x00" in value:
        raise OcrProfileError(f"{label} cannot contain a NUL byte")
    return value


def _require_id(value: Any, label: str) -> str:
    identifier = _require_text(value, label)
    if not _ID_RE.fullmatch(identifier):
        raise OcrProfileError(f"{label} is not a portable identifier")
    return identifier


def _require_sha256(value: Any, label: str) -> str:
    digest = _require_text(value, label)
    if not _SHA256_RE.fullmatch(digest):
        raise OcrProfileError(f"{label} must be a lowercase SHA-256")
    return digest


def _require_positive_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise OcrProfileError(f"{label} must be a positive integer")
    return value


def _looks_absolute_path(value: str) -> bool:
    if value.startswith(("/", "~", "file://", "\\\\")):
        return True
    return PureWindowsPath(value).is_absolute()


def _validate_portable_json(value: Any, label: str) -> None:
    """Reject non-JSON, host-path, credential, and non-finite values."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise OcrProfileError(f"{label} has a non-string JSON field")
            if _SECRET_FIELD_RE.search(key):
                raise OcrProfileError(
                    f"{label}.{key} is a credential-bearing field"
                )
            _validate_portable_json(item, f"{label}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_portable_json(item, f"{label}[{index}]")
        return
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise OcrProfileError(f"{label} must be finite")
        return
    if isinstance(value, str):
        if _looks_absolute_path(value):
            raise OcrProfileError(f"{label} contains an absolute host path")
        if any(pattern.search(value) for pattern in _SECRET_VALUE_RES):
            raise OcrProfileError(f"{label} contains credential-like data")
        return
    raise OcrProfileError(f"{label} is not portable JSON")


def _pipeline_records() -> tuple[dict[str, Any], ...]:
    # Keep the module lightweight during command discovery.  ocr_backends is
    # itself lightweight, but importing it lazily also avoids an import cycle
    # while the registry grows profile-aware helpers.
    from .ocr_backends import get_backend

    deepseek = get_backend("deepseek-vllm")
    mineru = get_backend("mineru25-pro-2605")

    def stage(
        stage_id: str,
        capability: str,
        backend,
        output_format: str,
        batch_semantics: str,
    ) -> dict[str, Any]:
        return {
            "stage_id": stage_id,
            "capability": capability,
            "backend_id": backend.backend_id,
            "model": {
                "requested": backend.model_id,
                "revision": backend.model_revision,
            },
            "output_format": output_format,
            "batch_semantics": batch_semantics,
        }

    records = (
        {
            "schema_version": SCHEMA_VERSION,
            "kind": PIPELINE_KIND,
            "pipeline_id": "deepseek-vllm-full",
            "aliases": ["deepseek-full", "legacy-deepseek-full"],
            "display_name": "Legacy DeepSeek full OCR pipeline",
            "execution_order": "sequential",
            "stages": [
                stage(
                    "grounded_ocr",
                    "grounded_ocr",
                    deepseek,
                    "ocr",
                    "page_batch",
                ),
                stage(
                    "markdown",
                    "markdown",
                    deepseek,
                    "markdown",
                    "page_batch",
                ),
            ],
        },
        {
            "schema_version": SCHEMA_VERSION,
            "kind": PIPELINE_KIND,
            "pipeline_id": "local-mineru25",
            "aliases": ["deepseek-mineru25", "mineru25-pipeline"],
            "display_name": "DeepSeek grounded OCR plus MinerU Markdown",
            "execution_order": "sequential",
            "stages": [
                stage(
                    "grounded_ocr",
                    "grounded_ocr",
                    deepseek,
                    "ocr",
                    "page_batch",
                ),
                stage(
                    "markdown",
                    "markdown",
                    mineru,
                    "markdown",
                    "layout_block_batch",
                ),
            ],
        },
    )
    sealed = []
    for record in records:
        value = copy.deepcopy(record)
        value["definition_sha256"] = content_sha256(value)
        sealed.append(value)
    return tuple(sealed)


def list_pipeline_definitions() -> tuple[dict[str, Any], ...]:
    """Return isolated built-in pipeline definitions in stable ID order."""

    return tuple(
        copy.deepcopy(record)
        for record in sorted(
            _pipeline_records(), key=lambda item: item["pipeline_id"]
        )
    )


def get_pipeline_definition(name: str) -> dict[str, Any]:
    """Resolve a built-in pipeline ID or alias."""

    normalized = _require_text(name, "pipeline name").strip().lower()
    for record in _pipeline_records():
        if normalized in {record["pipeline_id"], *record["aliases"]}:
            return copy.deepcopy(record)
    choices = ", ".join(
        item["pipeline_id"] for item in list_pipeline_definitions()
    )
    raise OcrProfileNotFoundError(
        f"unknown OCR pipeline {name!r}; expected one of: {choices}"
    )


def _pipeline_stages(pipeline_id: str) -> dict[str, dict[str, Any]]:
    pipeline = get_pipeline_definition(pipeline_id)
    return {stage["stage_id"]: stage for stage in pipeline["stages"]}


def _validate_model(
    value: Any,
    label: str,
    *,
    expected: Mapping[str, Any] | None = None,
) -> None:
    model = _require_mapping(value, label)
    _require_exact_keys(model, _MODEL_KEYS, label)
    _require_text(model["requested"], f"{label}.requested")
    revision = _require_text(model["revision"], f"{label}.revision")
    if not _REVISION_RE.fullmatch(revision):
        raise OcrProfileError(
            f"{label}.revision must be an immutable model SHA"
        )
    _require_sha256(
        model["snapshot_files_sha256"],
        f"{label}.snapshot_files_sha256",
    )
    if expected is not None:
        pinned = {"requested": model["requested"], "revision": revision}
        if pinned != dict(expected):
            raise OcrProfileError(
                f"{label} does not match the pipeline model pin"
            )


def _validate_hardware_selector(value: Any) -> dict[str, Any]:
    selector = dict(_require_mapping(value, "hardware_selector"))
    _require_exact_keys(
        selector,
        _HARDWARE_SELECTOR_KEYS,
        "hardware_selector",
    )
    normalized_keys = {key.casefold() for key in selector}
    forbidden = normalized_keys & _HOST_SPECIFIC_HARDWARE_FIELDS
    if forbidden:
        raise OcrProfileError(
            "hardware_selector contains host-specific fields: "
            + ", ".join(sorted(forbidden))
        )
    if selector["schema_version"] != SCHEMA_VERSION:
        raise OcrProfileError(
            "hardware_selector has an unsupported schema_version"
        )
    if selector["vendor"] != "nvidia":
        raise OcrProfileError("hardware_selector.vendor must be nvidia")
    _require_text(selector["product_name"], "hardware_selector.product_name")
    pci_device_id = selector["pci_device_id"]
    if pci_device_id is not None:
        _require_text(pci_device_id, "hardware_selector.pci_device_id")
    capability = selector["compute_capability"]
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or not all(
            isinstance(item, int) and not isinstance(item, bool) and item >= 0
            for item in capability
        )
    ):
        raise OcrProfileError(
            "hardware_selector.compute_capability must be [major, minor]"
        )
    if selector["memory_model"] not in {"dedicated", "unified", "unknown"}:
        raise OcrProfileError("hardware_selector.memory_model is invalid")
    total_memory = selector["total_memory_bytes"]
    _require_positive_integer(
        total_memory,
        "hardware_selector.total_memory_bytes",
    )
    _validate_portable_json(selector, "hardware_selector")
    return selector


def hardware_selector_from_observation(
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract the portable class selector, discarding host/UUID identity.

    ``ocr_hardware.portable_hardware_selector`` is authoritative when present.
    Tests and callers may also pass an observation that already contains a
    ``selector`` mapping.  This keeps the module decoupled from probe details.
    """

    record = _require_mapping(observation, "hardware observation")
    if isinstance(record.get("selector"), Mapping):
        return _validate_hardware_selector(record["selector"])
    try:
        from .ocr_hardware import portable_hardware_selector
    except ImportError:
        portable_hardware_selector = None
    if portable_hardware_selector is not None:
        try:
            selector = portable_hardware_selector(record)
        except (KeyError, TypeError, ValueError) as exc:
            raise OcrProfileError(
                f"could not derive a portable hardware selector: {exc}"
            ) from exc
        return _validate_hardware_selector(selector)

    # Transitional fallback for callers that have a raw, single-device record
    # before ocr_hardware is installed.  Only portable class fields survive.
    selector = {}
    aliases = {
        "schema_version": ("schema_version",),
        "vendor": ("vendor",),
        "product_name": (
            "product_name",
            "device_name",
            "cuda_device",
            "name",
        ),
        "pci_device_id": ("pci_device_id",),
        "compute_capability": (
            "compute_capability",
            "cuda_capability",
        ),
        "memory_model": ("memory_model",),
        "total_memory_bytes": (
            "total_memory_bytes",
            "cuda_total_memory_bytes",
        ),
    }
    for output_key, candidates in aliases.items():
        for candidate in candidates:
            if candidate in record:
                selector[output_key] = copy.deepcopy(record[candidate])
                break
    selector.setdefault("schema_version", SCHEMA_VERSION)
    selector.setdefault("pci_device_id", None)
    selector.setdefault("memory_model", "unknown")
    if isinstance(selector.get("vendor"), str):
        selector["vendor"] = selector["vendor"].casefold()
    if isinstance(selector.get("compute_capability"), tuple):
        selector["compute_capability"] = list(selector["compute_capability"])
    return _validate_hardware_selector(selector)


def _validate_runtime_selector(value: Any, label: str) -> None:
    runtime = _require_mapping(value, label)
    if not runtime:
        raise OcrProfileError(f"{label} cannot be empty")
    required = _RUNTIME_SELECTOR_KEYS
    missing = required - set(runtime)
    if missing:
        raise OcrProfileError(
            f"{label} is missing exact runtime fields: "
            + ", ".join(sorted(missing))
        )
    for key in required - {
        "packages",
        "installed_distributions_count",
    }:
        _require_text(runtime[key], f"{label}.{key}")
    _require_sha256(
        runtime["implementation_sha256"],
        f"{label}.implementation_sha256",
    )
    _require_positive_integer(
        runtime["installed_distributions_count"],
        f"{label}.installed_distributions_count",
    )
    _require_sha256(
        runtime["installed_distributions_sha256"],
        f"{label}.installed_distributions_sha256",
    )
    packages = _require_mapping(runtime["packages"], f"{label}.packages")
    if not packages or not all(
        isinstance(key, str) and key and isinstance(item, str) and item
        for key, item in packages.items()
    ):
        raise OcrProfileError(
            f"{label}.packages must contain exact package versions"
        )
    _validate_portable_json(runtime, label)
    _require_exact_keys(runtime, _RUNTIME_SELECTOR_KEYS, label)


def _validate_stages(
    value: Any,
    pipeline_stages: Mapping[str, Mapping[str, Any]],
) -> None:
    stages = _require_mapping(value, "stages")
    if set(stages) != set(pipeline_stages):
        raise OcrProfileError(
            "profile stages must exactly match the pipeline stages"
        )
    for stage_id, expected in pipeline_stages.items():
        label = f"stages.{stage_id}"
        stage = _require_mapping(stages[stage_id], label)
        _require_exact_keys(stage, _STAGE_KEYS, label)
        for key in ("backend_id", "capability", "batch_semantics"):
            _require_text(stage[key], f"{label}.{key}")
        if stage["backend_id"] != expected["backend_id"]:
            raise OcrProfileError(f"{label}.backend_id differs from pipeline")
        if stage["capability"] != expected["capability"]:
            raise OcrProfileError(f"{label}.capability differs from pipeline")
        if stage["batch_semantics"] != expected["batch_semantics"]:
            raise OcrProfileError(
                f"{label}.batch_semantics differs from pipeline"
            )
        if stage["batch_semantics"] not in BATCH_SEMANTICS:
            raise OcrProfileError(f"{label}.batch_semantics is unsupported")
        _validate_model(
            stage["model"],
            f"{label}.model",
            expected=expected["model"],
        )
        _validate_runtime_selector(
            stage["runtime_selector"],
            f"{label}.runtime_selector",
        )


def _validate_tier_stage(
    value: Any,
    label: str,
    *,
    backend_id: str,
    implementation_sha256: str,
    memory_tier: str,
    budget_bytes: int,
    total_memory_bytes: int,
) -> None:
    stage = _require_mapping(value, label)
    _require_exact_keys(stage, _TIER_STAGE_KEYS, label)
    settings = _require_mapping(stage["settings"], f"{label}.settings")
    if backend_id == "deepseek-vllm":
        expected_keys = _DEEPSEEK_TIER_SETTING_KEYS
    elif backend_id == "mineru25-pro-2605":
        expected_keys = _MINERU_TIER_SETTING_KEYS
    else:
        raise OcrProfileError(
            f"{label} has no validated settings schema for {backend_id}"
        )
    legacy_deepseek_settings = (
        backend_id == "deepseek-vllm"
        and set(settings) == _DEEPSEEK_LEGACY_TIER_SETTING_KEYS
        and implementation_sha256
        == _DEEPSEEK_LEGACY_IMPLEMENTATION_SHA256
    )
    if legacy_deepseek_settings:
        # Existing sealed profiles remain readable. Their implementation
        # selector prevents reuse with runtime-control-aware source until
        # the GPU is requalified and a new profile is trusted.
        pass
    else:
        _require_exact_keys(settings, expected_keys, f"{label}.settings")
    batch_size = settings.get("batch_size")
    _require_positive_integer(batch_size, f"{label}.settings.batch_size")
    if settings["memory_tier"] != memory_tier:
        raise OcrProfileError(
            f"{label}.settings.memory_tier differs from its containing tier"
        )
    if settings["memory_budget_bytes"] != budget_bytes:
        raise OcrProfileError(
            f"{label}.settings.memory_budget_bytes differs from its tier"
        )
    if backend_id == "deepseek-vllm":
        for key in (
            "max_num_batched_tokens",
            "max_num_seqs",
            "max_tokens",
            "ngram_size",
            "window_size",
        ):
            _require_positive_integer(
                settings[key], f"{label}.settings.{key}"
            )
        if settings["max_num_seqs"] != batch_size:
            raise OcrProfileError(
                f"{label}.settings.max_num_seqs must equal batch_size"
            )
        utilization = settings["gpu_memory_utilization"]
        if (
            not isinstance(utilization, (int, float))
            or isinstance(utilization, bool)
            or not math.isfinite(float(utilization))
            or not 0 < float(utilization) <= 0.95
        ):
            raise OcrProfileError(
                f"{label}.settings.gpu_memory_utilization is invalid"
            )
        denominator = _require_positive_integer(
            settings["gpu_memory_utilization_denominator_bytes"],
            f"{label}.settings.gpu_memory_utilization_denominator_bytes",
        )
        if denominator < total_memory_bytes:
            raise OcrProfileError(
                f"{label}.settings utilization denominator is smaller than "
                "the profile memory capacity"
            )
        requested_bytes = round(float(utilization) * denominator)
        rounding_allowance = max(1, round(total_memory_bytes * 1e-8))
        headroom = _require_positive_integer(
            settings["engine_overhead_headroom_bytes"],
            f"{label}.settings.engine_overhead_headroom_bytes",
        )
        if headroom >= budget_bytes:
            raise OcrProfileError(
                f"{label}.settings engine overhead headroom exceeds its tier"
            )
        if requested_bytes + headroom > budget_bytes + rounding_allowance:
            raise OcrProfileError(
                f"{label}.settings.gpu_memory_utilization plus engine "
                "overhead headroom exceeds its tier"
            )
        if not isinstance(settings["enable_chunked_prefill"], bool):
            raise OcrProfileError(
                f"{label}.settings.enable_chunked_prefill must be boolean"
            )
        for key in ("attention_backend", "mm_encoder_attn_backend"):
            _require_text(settings[key], f"{label}.settings.{key}")
        if set(settings) == _DEEPSEEK_TIER_SETTING_KEYS:
            moe_backend = _require_text(
                settings["moe_backend"],
                f"{label}.settings.moe_backend",
            )
            if moe_backend != _QUALIFIED_DEEPSEEK_MOE_BACKEND:
                raise OcrProfileError(
                    f"{label}.settings.moe_backend must be triton"
                )
        if (
            settings["max_tokens"] != 8192
            or settings["ngram_size"] != 10
            or settings["window_size"] != 50
            or settings["temperature"] != 0.0
        ):
            raise OcrProfileError(
                f"{label}.settings changes fixed grounded-OCR quality "
                "controls"
            )
    else:
        if settings["backend"] != "official-two-step-transformers":
            raise OcrProfileError(
                f"{label}.settings.backend is not the qualified MinerU path"
            )
        if settings["image_analysis"] is not False:
            raise OcrProfileError(
                f"{label}.settings.image_analysis must be false"
            )
    _validate_portable_json(settings, f"{label}.settings")
    peak = _require_positive_integer(
        stage["observed_peak_memory_bytes"],
        f"{label}.observed_peak_memory_bytes",
    )
    if peak > budget_bytes:
        raise OcrProfileError(
            f"{label}.observed_peak_memory_bytes exceeds its tier budget"
        )
    throughput = stage["confirmed_pages_per_second"]
    if (
        not isinstance(throughput, (int, float))
        or isinstance(throughput, bool)
        or not math.isfinite(float(throughput))
        or throughput <= 0
    ):
        raise OcrProfileError(
            f"{label}.confirmed_pages_per_second must be positive and finite"
        )


def _validate_memory_tiers(
    value: Any,
    *,
    memory_model: str,
    total_memory_bytes: int,
    pipeline_stages: Mapping[str, Mapping[str, Any]],
    profile_stages: Mapping[str, Mapping[str, Any]],
) -> set[str]:
    tiers = _require_mapping(value, "memory_tiers")
    if set(tiers) != set(MEMORY_TIERS):
        raise OcrProfileError(
            "memory_tiers must contain exactly 25, 50, and max"
        )
    try:
        policy = _MEMORY_TIER_POLICIES[memory_model]
    except KeyError as exc:
        raise OcrProfileError(
            "hardware_selector.memory_model must be dedicated or unified "
            "for canonical memory tiers"
        ) from exc
    expected_reserve = min(
        policy["reserve_bytes"],
        total_memory_bytes - 1,
    )
    expected_fractions = {
        "25": 0.25,
        "50": 0.5,
        "max": policy["max_fraction"],
    }
    supported_tiers = set()
    for tier_name in MEMORY_TIERS:
        label = f"memory_tiers.{tier_name}"
        tier = _require_mapping(tiers[tier_name], label)
        _require_exact_keys(tier, _TIER_KEYS, label)
        if not isinstance(tier["supported"], bool):
            raise OcrProfileError(f"{label}.supported must be boolean")
        fraction = tier["budget_fraction"]
        if (
            not isinstance(fraction, (int, float))
            or isinstance(fraction, bool)
            or not math.isfinite(float(fraction))
            or not 0 < float(fraction) < 1
        ):
            raise OcrProfileError(
                f"{label}.budget_fraction must be between zero and one"
            )
        expected_fraction = expected_fractions[tier_name]
        if float(fraction) != expected_fraction:
            raise OcrProfileError(
                f"{label}.budget_fraction must be {expected_fraction}"
            )
        budget = _require_positive_integer(
            tier["budget_bytes"], f"{label}.budget_bytes"
        )
        reserve = _require_positive_integer(
            tier["reserve_bytes"], f"{label}.reserve_bytes"
        )
        if reserve != expected_reserve:
            raise OcrProfileError(
                f"{label}.reserve_bytes must be {expected_reserve} for "
                f"{memory_model} memory"
            )
        expected_budget = min(
            int(total_memory_bytes * expected_fraction),
            total_memory_bytes - expected_reserve,
        )
        if budget != expected_budget:
            raise OcrProfileError(
                f"{label}.budget_bytes must equal canonical budget "
                f"{expected_budget}"
            )
        stage_settings = _require_mapping(tier["stages"], f"{label}.stages")
        reason = tier["reason"]
        if tier["supported"]:
            supported_tiers.add(tier_name)
            if reason is not None:
                raise OcrProfileError(
                    f"{label}.reason must be null when supported"
                )
            if set(stage_settings) != set(pipeline_stages):
                raise OcrProfileError(
                    f"{label}.stages must cover the complete pipeline"
                )
            for stage_id in pipeline_stages:
                _validate_tier_stage(
                    stage_settings[stage_id],
                    f"{label}.stages.{stage_id}",
                    backend_id=pipeline_stages[stage_id]["backend_id"],
                    implementation_sha256=profile_stages[stage_id][
                        "runtime_selector"
                    ]["implementation_sha256"],
                    memory_tier=tier_name,
                    budget_bytes=budget,
                    total_memory_bytes=total_memory_bytes,
                )
        else:
            _require_text(reason, f"{label}.reason")
            if stage_settings:
                raise OcrProfileError(
                    f"{label}.stages must be empty when unsupported"
                )
    if not supported_tiers:
        raise OcrProfileError("at least one memory tier must be supported")
    return supported_tiers


def _validate_qualification(
    value: Any,
    *,
    supported_tiers: set[str],
) -> None:
    qualification = _require_mapping(value, "qualification")
    _require_exact_keys(
        qualification,
        _QUALIFICATION_KEYS,
        "qualification",
    )
    state = qualification["state"]
    if state not in QUALIFICATION_STATES:
        raise OcrProfileError(
            "qualification.state must be provisional or qualified"
        )
    _require_sha256(
        qualification["policy_sha256"],
        "qualification.policy_sha256",
    )
    if qualification["policy_sha256"] != qualification_policy_sha256():
        raise OcrProfileError(
            "qualification.policy_sha256 is not the current execution "
            "policy"
        )
    required_suites = qualification["required_suites"]
    if (
        not isinstance(required_suites, list)
        or not required_suites
        or not all(
            isinstance(item, str) and _ID_RE.fullmatch(item)
            for item in required_suites
        )
        or len(required_suites) != len(set(required_suites))
    ):
        raise OcrProfileError(
            "qualification.required_suites must contain unique suite IDs"
        )
    evidence = qualification["evidence"]
    if not isinstance(evidence, list):
        raise OcrProfileError("qualification.evidence must be a list")
    seen = set()
    passing_coverage = set()
    for index, item in enumerate(evidence):
        label = f"qualification.evidence[{index}]"
        record = _require_mapping(item, label)
        _require_exact_keys(record, _EVIDENCE_KEYS, label)
        suite_id = _require_id(record["suite_id"], f"{label}.suite_id")
        tier = record["memory_tier"]
        if tier not in MEMORY_TIERS:
            raise OcrProfileError(f"{label}.memory_tier is invalid")
        if tier not in supported_tiers:
            raise OcrProfileError(
                f"{label} cannot qualify an unsupported memory tier"
            )
        identity = (suite_id, tier)
        if identity in seen:
            raise OcrProfileError(
                f"duplicate qualification evidence for {suite_id}/{tier}"
            )
        seen.add(identity)
        for key in (
            "manifest_sha256",
            "summary_sha256",
            "experiment_sha256",
        ):
            _require_sha256(record[key], f"{label}.{key}")
        _require_positive_integer(record["repeats"], f"{label}.repeats")
        if (
            state == "qualified"
            and record["repeats"] < MINIMUM_QUALIFICATION_REPEATS
        ):
            raise OcrProfileError(
                f"{label}.repeats must be at least "
                f"{MINIMUM_QUALIFICATION_REPEATS} for qualification"
            )
        for key in ("all_pass", "provenance_complete"):
            if not isinstance(record[key], bool):
                raise OcrProfileError(f"{label}.{key} must be boolean")
        if record["all_pass"] and record["provenance_complete"]:
            passing_coverage.add(identity)
    if state == "qualified":
        required_coverage = {
            (suite_id, tier)
            for suite_id in required_suites
            for tier in supported_tiers
        }
        missing = required_coverage - passing_coverage
        if missing:
            rendered = ", ".join(
                f"{suite}/{tier}" for suite, tier in sorted(missing)
            )
            raise OcrProfileError(
                "qualified profile lacks passing evidence for " + rendered
            )


def _validate_profile_payload(
    value: Mapping[str, Any],
    *,
    require_hash: bool,
) -> dict[str, Any]:
    profile = copy.deepcopy(dict(value))
    expected_keys = (
        _PROFILE_KEYS if require_hash else _PROFILE_KEYS - {"profile_sha256"}
    )
    _require_exact_keys(profile, expected_keys, "profile")
    if profile["schema_version"] != SCHEMA_VERSION:
        raise OcrProfileError("profile has an unsupported schema_version")
    if profile["kind"] != PROFILE_KIND:
        raise OcrProfileError(f"profile.kind must be {PROFILE_KIND}")
    _require_id(profile["profile_id"], "profile.profile_id")
    _require_text(profile["display_name"], "profile.display_name")
    pipeline_id = _require_id(profile["pipeline_id"], "profile.pipeline_id")
    pipeline = get_pipeline_definition(pipeline_id)
    if pipeline_id != pipeline["pipeline_id"]:
        raise OcrProfileError(
            "profile.pipeline_id must use the canonical pipeline ID"
        )
    pipeline_digest = _require_sha256(
        profile["pipeline_definition_sha256"],
        "profile.pipeline_definition_sha256",
    )
    if pipeline_digest != pipeline["definition_sha256"]:
        raise OcrProfileError(
            "profile.pipeline_definition_sha256 differs from the "
            "built-in pipeline"
        )
    pipeline_stages = {
        stage["stage_id"]: stage for stage in pipeline["stages"]
    }
    selector = _validate_hardware_selector(profile["hardware_selector"])
    _validate_stages(profile["stages"], pipeline_stages)
    supported_tiers = _validate_memory_tiers(
        profile["memory_tiers"],
        memory_model=selector["memory_model"],
        total_memory_bytes=selector["total_memory_bytes"],
        pipeline_stages=pipeline_stages,
        profile_stages=profile["stages"],
    )
    _validate_qualification(
        profile["qualification"], supported_tiers=supported_tiers
    )
    _validate_portable_json(profile, "profile")
    if require_hash:
        digest = _require_sha256(
            profile["profile_sha256"], "profile.profile_sha256"
        )
        unhashed = dict(profile)
        unhashed.pop("profile_sha256")
        if digest != content_sha256(unhashed):
            raise OcrProfileError(
                "profile_sha256 does not match profile content"
            )
    return profile


def seal_execution_profile(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate profile payload and add its canonical content hash."""

    profile = _validate_profile_payload(value, require_hash=False)
    profile["profile_sha256"] = content_sha256(profile)
    return profile


def qualify_execution_profile(
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a qualified candidate from reviewed provisional evidence.

    The resulting hash is deliberately not trusted automatically. Production
    use still requires source review and inclusion in the plugin's trusted
    profile set.
    """

    value = validate_execution_profile(profile)
    if value["qualification"]["state"] != "provisional":
        raise OcrProfileError(
            "only a provisional profile can be promoted for review"
        )
    candidate = copy.deepcopy(value)
    candidate.pop("profile_sha256", None)
    candidate["qualification"]["state"] = "qualified"
    return seal_execution_profile(candidate)


def validate_execution_profile(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return an isolated sealed execution profile."""

    return _validate_profile_payload(value, require_hash=True)


def profile_receipt_sha256(value: Mapping[str, Any]) -> str:
    """Return the verified content digest of a sealed profile."""

    return validate_execution_profile(value)["profile_sha256"]


def load_execution_profile(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load one strict execution-profile JSON file."""

    source = Path(path).expanduser()
    try:
        source = validate_trusted_file(
            source,
            label="OCR execution profile",
            resolve_symlinks=False,
        )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NONBLOCK", 0)
        if not hasattr(os, "O_NOFOLLOW"):
            raise OcrProfileError(
                "OCR profile loading requires POSIX no-follow opens"
            )
        flags |= os.O_NOFOLLOW
        descriptor = os.open(source, flags)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise OcrProfileError(
                    f"OCR profile {source} is not a regular file"
                )
            if metadata.st_size > _MAX_PROFILE_FILE_BYTES:
                raise OcrProfileError(
                    f"OCR profile {source} exceeds "
                    f"{_MAX_PROFILE_FILE_BYTES} bytes"
                )
            chunks = []
            remaining = _MAX_PROFILE_FILE_BYTES + 1
            while remaining:
                chunk = os.read(descriptor, min(remaining, 64 * 1024))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            payload = b"".join(chunks)
            if len(payload) > _MAX_PROFILE_FILE_BYTES:
                raise OcrProfileError(
                    f"OCR profile {source} exceeds "
                    f"{_MAX_PROFILE_FILE_BYTES} bytes"
                )
        finally:
            os.close(descriptor)
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
        )
    except OcrProfileError:
        raise
    except (OSError, UnicodeError, ValueError, TrustedPathError) as exc:
        raise OcrProfileError(
            f"could not read OCR profile {source}: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise OcrProfileError(f"OCR profile {source} is not a JSON object")
    return validate_execution_profile(value)


def write_execution_profile(
    value: Mapping[str, Any],
    path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Atomically create, but never replace, a portable profile file."""

    profile = validate_execution_profile(value)
    destination = Path(path).expanduser()
    if not destination.is_absolute():
        raise OcrProfileError("execution profile output must be absolute")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            profile,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    temporary = None
    try:
        descriptor, name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".candidate",
        )
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise OcrProfileError(
                f"refusing to replace existing profile: {destination}"
            ) from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    return copy.deepcopy(profile)


def _profile_sources(
    *,
    profiles: Iterable[Mapping[str, Any]] | None,
    paths: Iterable[str | os.PathLike[str]] | None,
) -> list[dict[str, Any]]:
    values = []
    if profiles is not None:
        values.extend(validate_execution_profile(item) for item in profiles)
    for raw_path in paths or ():
        path = Path(raw_path).expanduser()
        if path.is_dir():
            try:
                path = validate_trusted_directory(
                    path,
                    label="OCR profile directory",
                    resolve_symlinks=False,
                )
            except TrustedPathError as exc:
                raise OcrProfileError(str(exc)) from exc
            try:
                candidates = []
                with os.scandir(path) as entries:
                    for entry_count, entry in enumerate(entries, 1):
                        if entry_count > _MAX_PROFILE_DIRECTORY_ENTRIES:
                            raise OcrProfileError(
                                f"OCR profile directory {path} exceeds "
                                f"{_MAX_PROFILE_DIRECTORY_ENTRIES} entries"
                            )
                        if entry.name.endswith(".json"):
                            candidates.append(path / entry.name)
                candidates.sort()
            except OcrProfileError:
                raise
            except OSError as exc:
                raise OcrProfileError(
                    f"could not scan OCR profile directory {path}: {exc}"
                ) from exc
        else:
            candidates = [path]
        values.extend(
            load_execution_profile(candidate) for candidate in candidates
        )
    by_id = {}
    for profile in values:
        profile_id = profile["profile_id"]
        existing = by_id.get(profile_id)
        if existing is None:
            by_id[profile_id] = profile
            continue
        if (
            existing["profile_sha256"] != profile["profile_sha256"]
            or existing != profile
        ):
            raise OcrProfileError(
                "duplicate OCR profile ID with conflicting content: "
                f"{profile_id}"
            )
    return [by_id[key] for key in sorted(by_id)]


def list_execution_profiles(
    profiles: Iterable[Mapping[str, Any]] | None = None,
    *,
    paths: Iterable[str | os.PathLike[str]] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Validate and list supplied profile records or JSON paths."""

    return tuple(
        copy.deepcopy(item)
        for item in _profile_sources(profiles=profiles, paths=paths)
    )


def show_execution_profile(
    profile_id: str,
    profiles: Iterable[Mapping[str, Any]] | None = None,
    *,
    paths: Iterable[str | os.PathLike[str]] | None = None,
) -> dict[str, Any]:
    """Return one supplied execution profile by exact ID."""

    wanted = _require_id(profile_id, "profile_id")
    for profile in _profile_sources(profiles=profiles, paths=paths):
        if profile["profile_id"] == wanted:
            return copy.deepcopy(profile)
    raise OcrProfileNotFoundError(f"unknown OCR execution profile: {wanted}")


def _selector_mismatches(
    selector: Mapping[str, Any],
    observation: Mapping[str, Any],
    prefix: str,
) -> list[str]:
    mismatches = []
    for key, expected in selector.items():
        path = f"{prefix}.{key}"
        if key not in observation:
            mismatches.append(f"{path} is missing")
            continue
        actual = observation[key]
        if isinstance(expected, Mapping):
            if not isinstance(actual, Mapping):
                mismatches.append(f"{path} differs")
            else:
                mismatches.extend(
                    _selector_mismatches(expected, actual, path)
                )
        elif actual != expected:
            mismatches.append(f"{path} differs")
    return mismatches


def _preflight_runtime_mismatches(
    *,
    selector: Mapping[str, Any],
    observation: Mapping[str, Any],
    tier_settings: Mapping[str, Any],
    prefix: str,
) -> list[str]:
    """Compare requested attention before load and selected after load."""

    selected_attention = selector.get("attention_backend")
    without_attention = {
        key: value
        for key, value in selector.items()
        if key != "attention_backend"
    }
    reasons = _selector_mismatches(
        without_attention,
        observation,
        prefix,
    )
    observed_any_attention = False
    if "attention_backend_requested" in observation:
        observed_any_attention = True
        expected_requested = tier_settings.get(
            "attention_backend", selected_attention
        )
        if observation["attention_backend_requested"] != expected_requested:
            reasons.append(f"{prefix}.attention_backend_requested differs")
    for key in ("attention_backend", "attention_backend_selected"):
        if key not in observation:
            continue
        observed_any_attention = True
        if observation[key] != selected_attention:
            reasons.append(f"{prefix}.{key} differs")
    if not observed_any_attention:
        reasons.append(f"{prefix}.attention backend observation is missing")
    return reasons


def profile_match_report(
    profile: Mapping[str, Any],
    observation: Mapping[str, Any],
    *,
    memory_tier: str = "max",
    require_qualified: bool = True,
    trusted_qualified_profile_sha256s: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Explain exact hardware, model, runtime, tier, and status matching."""

    value = validate_execution_profile(profile)
    if memory_tier not in MEMORY_TIERS:
        raise OcrProfileError(
            "memory_tier must be one of: " + ", ".join(MEMORY_TIERS)
        )
    observed = _require_mapping(observation, "execution observation")
    hardware_value = observed.get("hardware", observed)
    hardware = hardware_selector_from_observation(
        _require_mapping(hardware_value, "execution observation.hardware")
    )
    reasons = _selector_mismatches(
        value["hardware_selector"], hardware, "hardware"
    )
    tier = value["memory_tiers"][memory_tier]
    observed_stages = observed.get("stages")
    if not isinstance(observed_stages, Mapping):
        reasons.append("stages are missing")
        observed_stages = {}
    for stage_id, stage in value["stages"].items():
        observed_stage = observed_stages.get(stage_id)
        if not isinstance(observed_stage, Mapping):
            reasons.append(f"stages.{stage_id} is missing")
            continue
        if observed_stage.get("backend_id") != stage["backend_id"]:
            reasons.append(f"stages.{stage_id}.backend_id differs")
        if observed_stage.get("capability") != stage["capability"]:
            reasons.append(f"stages.{stage_id}.capability differs")
        model = observed_stage.get("model")
        if not isinstance(model, Mapping) or dict(model) != stage["model"]:
            reasons.append(f"stages.{stage_id}.model differs")
        runtime = observed_stage.get("runtime")
        if not isinstance(runtime, Mapping):
            reasons.append(f"stages.{stage_id}.runtime is missing")
        else:
            tier_stage = tier.get("stages", {}).get(stage_id, {})
            tier_settings = (
                tier_stage.get("settings", {})
                if isinstance(tier_stage, Mapping)
                else {}
            )
            reasons.extend(
                _preflight_runtime_mismatches(
                    selector=stage["runtime_selector"],
                    observation=runtime,
                    tier_settings=(
                        tier_settings
                        if isinstance(tier_settings, Mapping)
                        else {}
                    ),
                    prefix=f"stages.{stage_id}.runtime",
                )
            )
    if not tier["supported"]:
        reasons.append(f"memory tier {memory_tier} is unsupported")
    qualification_state = value["qualification"]["state"]
    if qualification_state == "provisional":
        if require_qualified:
            reasons.append("profile qualification is provisional")
    else:
        trusted = (
            TRUSTED_QUALIFIED_PROFILE_SHA256S
            if trusted_qualified_profile_sha256s is None
            else frozenset(trusted_qualified_profile_sha256s)
        )
        if value["profile_sha256"] not in trusted:
            reasons.append(
                "profile qualification is not trusted by this plugin build"
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "profile_id": value["profile_id"],
        "pipeline_id": value["pipeline_id"],
        "memory_tier": memory_tier,
        "matched": not reasons,
        "reasons": reasons,
        "profile_sha256": value["profile_sha256"],
    }


def profile_matches(
    profile: Mapping[str, Any],
    observation: Mapping[str, Any],
    *,
    memory_tier: str = "max",
    require_qualified: bool = True,
    trusted_qualified_profile_sha256s: Iterable[str] | None = None,
) -> bool:
    """Return whether a sealed profile exactly matches one observation."""

    return profile_match_report(
        profile,
        observation,
        memory_tier=memory_tier,
        require_qualified=require_qualified,
        trusted_qualified_profile_sha256s=(
            trusted_qualified_profile_sha256s
        ),
    )["matched"]


def auto_select_execution_profile(
    profiles: Iterable[Mapping[str, Any]],
    observation: Mapping[str, Any],
    *,
    pipeline_id: str,
    memory_tier: str = "max",
    require_qualified: bool = True,
    trusted_qualified_profile_sha256s: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Select the one exact profile or fail with actionable diagnostics."""

    canonical_pipeline = get_pipeline_definition(pipeline_id)["pipeline_id"]
    candidates = [
        profile
        for profile in list_execution_profiles(profiles)
        if profile["pipeline_id"] == canonical_pipeline
    ]
    reports = [
        profile_match_report(
            profile,
            observation,
            memory_tier=memory_tier,
            require_qualified=require_qualified,
            trusted_qualified_profile_sha256s=(
                trusted_qualified_profile_sha256s
            ),
        )
        for profile in candidates
    ]
    matched = [
        profile
        for profile, report in zip(candidates, reports, strict=True)
        if report["matched"]
    ]
    if len(matched) == 1:
        return copy.deepcopy(matched[0])
    if len(matched) > 1:
        identifiers = ", ".join(item["profile_id"] for item in matched)
        raise OcrProfileSelectionError(
            "multiple OCR execution profiles match exactly: " + identifiers
        )
    if not candidates:
        raise OcrProfileSelectionError(
            "no OCR profiles are registered for pipeline "
            f"{canonical_pipeline}; "
            "run the GPU tuning workflow to onboard this hardware"
        )
    diagnostics = "; ".join(
        f"{report['profile_id']}: {', '.join(report['reasons'])}"
        for report in reports
    )
    raise OcrProfileSelectionError(
        "no OCR execution profile matches this hardware/runtime/tier; "
        "run the GPU tuning workflow to onboard it. " + diagnostics
    )


__all__ = [
    "BATCH_SEMANTICS",
    "EXECUTION_QUALIFICATION_POLICY",
    "MEMORY_TIERS",
    "MINIMUM_QUALIFICATION_REPEATS",
    "OcrProfileError",
    "OcrProfileNotFoundError",
    "OcrProfileSelectionError",
    "PROFILE_KIND",
    "QUALIFICATION_STATES",
    "SCHEMA_VERSION",
    "TRUSTED_QUALIFIED_PROFILE_SHA256S",
    "auto_select_execution_profile",
    "content_sha256",
    "get_pipeline_definition",
    "hardware_selector_from_observation",
    "list_execution_profiles",
    "list_pipeline_definitions",
    "load_execution_profile",
    "profile_match_report",
    "profile_matches",
    "profile_receipt_sha256",
    "qualification_policy_sha256",
    "qualify_execution_profile",
    "seal_execution_profile",
    "show_execution_profile",
    "validate_execution_profile",
    "write_execution_profile",
]
