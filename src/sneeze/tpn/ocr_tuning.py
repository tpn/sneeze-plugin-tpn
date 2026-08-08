"""Bounded, resumable local OCR execution-profile tuning.

The tuning orchestrator deliberately sits above :mod:`ocr_benchmark`.  Every
candidate is run in a fresh benchmark directory and partial evidence is never
overwritten.  The three checked-in cuPhoton regressions are a hardware smoke
gate; they can produce a provisional execution profile, not a new
model-quality qualification.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .ocr_backend_setup import file_sha256, make_receipt, sha256_json
from .ocr_backends import get_backend
from .ocr_path_security import (
    TrustedPathError,
    prepare_private_directory,
    prepare_private_output_path,
    validate_trusted_file,
)

SCHEMA_VERSION = 1
MEMORY_TIERS = ("25", "50", "max")
DEFAULT_BATCH_SIZES = (1, 2, 4, 8)
GIB = 1024**3
DEFAULT_DEEPSEEK_ENGINE_OVERHEAD_HEADROOM_BYTES = (GIB, 2 * GIB, 4 * GIB)
MAX_BATCH_SIZE = 64
MAX_CANDIDATES = 96
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[a-z0-9]+(?:[a-z0-9.-]*[a-z0-9])?$")
_STAGE_BY_PIPELINE = {
    "local-mineru25": (
        ("grounded_ocr", "deepseek-vllm"),
        ("markdown", "mineru25-pro-2605"),
    ),
}


class TuningError(ValueError):
    """Raised when a tuning session is unsafe, incomplete, or inconsistent."""


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise TuningError("duplicate JSON field")
        value[key] = item
    return value


def _paths_overlap(first: Path, second: Path) -> bool:
    return (
        first == second
        or first.is_relative_to(second)
        or second.is_relative_to(first)
    )


def _reject_symlink_ancestry(path: Path, *, label: str) -> None:
    """Reject every existing symlink component before following it."""

    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if os.path.lexists(current) and current.is_symlink():
            raise TuningError(
                f"{label} must not traverse symlinks: {current}"
            )


def _validate_tuning_destination(
    plan: Mapping[str, Any],
    raw_path: str | os.PathLike[str],
) -> Path:
    destination = Path(raw_path).expanduser()
    if not destination.is_absolute():
        raise TuningError("tuning destination must be absolute")
    _reject_symlink_ancestry(destination, label="tuning destination")
    resolved = destination.resolve(strict=False)
    if resolved != destination:
        raise TuningError("tuning destination must be a canonical path")
    if destination.exists() and not destination.is_dir():
        raise TuningError("tuning destination must be a directory")

    input_root = Path(str(plan.get("input_root", ""))).resolve(strict=True)
    manifest = Path(
        str(plan.get("manifest", {}).get("path", ""))
    ).resolve(strict=True)
    for label, protected in (
        ("input root", input_root),
        ("manifest", manifest),
    ):
        if _paths_overlap(resolved, protected):
            raise TuningError(
                f"tuning destination must be separate from the {label}"
            )
    try:
        return prepare_private_directory(
            resolved,
            label="tuning destination",
        )
    except TrustedPathError as exc:
        raise TuningError(str(exc)) from exc


def _validate_profile_output_path(
    plan: Mapping[str, Any],
    raw_path: str | os.PathLike[str],
) -> Path:
    """Validate a profile output before the tuning session writes anything."""

    destination = Path(raw_path).expanduser()
    if not destination.is_absolute():
        raise TuningError("execution profile output must be absolute")
    _reject_symlink_ancestry(destination, label="execution profile output")
    resolved = destination.resolve(strict=False)
    if resolved != destination:
        raise TuningError("execution profile output must be a canonical path")
    if destination.exists() and not destination.is_file():
        raise TuningError("execution profile output must be a file")

    input_root = Path(str(plan.get("input_root", ""))).resolve(strict=True)
    manifest = Path(
        str(plan.get("manifest", {}).get("path", ""))
    ).resolve(strict=True)
    for label, protected in (
        ("input root", input_root),
        ("manifest", manifest),
    ):
        if _paths_overlap(resolved, protected):
            raise TuningError(
                "execution profile output must be separate from the " + label
            )
    try:
        return prepare_private_output_path(
            resolved,
            label="execution profile output",
        )
    except TrustedPathError as exc:
        raise TuningError(str(exc)) from exc


def _validate_session_path(path: Path, *, root: Path, label: str) -> None:
    if not path.is_relative_to(root):
        raise TuningError(f"{label} escaped the tuning destination")
    _reject_symlink_ancestry(path, label=label)


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_json_bytes(dict(value)))
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise TuningError(f"tuning JSON must be a regular file: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except (OSError, ValueError) as exc:
        raise TuningError(
            f"could not read tuning JSON {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise TuningError(f"tuning JSON must contain one object: {path}")
    return value


def _append_journal(path: Path, value: Mapping[str, Any]) -> None:
    if path.is_symlink() or not path.is_file():
        raise TuningError("tuning journal must be a regular file")
    payload = json.dumps(
        dict(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(payload + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _require_absolute_existing_file(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TuningError(f"{label} must be a nonempty absolute path")
    path = Path(value).expanduser()
    if not path.is_absolute() or not path.is_file():
        raise TuningError(f"{label} must be an existing absolute file")
    return str(path.resolve())


def _require_absolute_existing_dir(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TuningError(f"{label} must be a nonempty absolute path")
    path = Path(value).expanduser()
    if not path.is_absolute() or not path.is_dir():
        raise TuningError(f"{label} must be an existing absolute directory")
    return str(path.resolve())


def _validate_tiers(values: Sequence[str]) -> tuple[str, ...]:
    tiers = tuple(
        dict.fromkeys(str(value).strip().lower() for value in values)
    )
    if not tiers or any(tier not in MEMORY_TIERS for tier in tiers):
        raise TuningError("memory tiers must be selected from 25, 50, max")
    return tiers


def _validate_batches(values: Sequence[int]) -> tuple[int, ...]:
    batches = tuple(dict.fromkeys(int(value) for value in values))
    if (
        not batches
        or any(value <= 0 or value > MAX_BATCH_SIZE for value in batches)
        or tuple(sorted(batches)) != batches
    ):
        raise TuningError(
            f"batch sizes must be ascending integers from 1 through "
            f"{MAX_BATCH_SIZE}"
        )
    return batches


def _stage_lane(capability: str, backend_id: str) -> str:
    if capability == "markdown" and backend_id.startswith("mineru"):
        return "best"
    if capability == "grounded_ocr" and backend_id == "deepseek-vllm":
        return "grounded_ocr"
    raise TuningError(
        f"no tuning lane for capability {capability!r} on {backend_id!r}"
    )


def _manifest_summary(
    path: Path,
    *,
    exercised_lanes: frozenset[str],
) -> tuple[int, tuple[str, ...]]:
    from .ocr_benchmark import BenchmarkError, load_manifest

    try:
        manifest = load_manifest(path)
    except BenchmarkError as exc:
        raise TuningError(f"tuning manifest is invalid: {exc}") from exc

    completeness_metric = {
        "best": "quality.table_completion.raw_token_multiset_recall",
        "grounded_ocr": (
            "quality.table_completion.grounded_ocr_token_multiset_recall"
        ),
    }
    case_ids = []
    for case_index, case in enumerate(manifest, 1):
        case_ids.append(case["case_id"])
        assertions = case["assertions"]
        for assertion_index, assertion in enumerate(assertions, 1):
            scoped_lanes = assertion.get("lanes")
            if (
                isinstance(scoped_lanes, list)
                and exercised_lanes.isdisjoint(scoped_lanes)
            ):
                raise TuningError(
                    "tuning manifest assertion is not exercised by this "
                    f"pipeline in case {case_index}, assertion "
                    f"{assertion_index}"
                )
        for lane in sorted(exercised_lanes):
            applicable = [
                assertion
                for assertion in assertions
                if not isinstance(assertion.get("lanes"), list)
                or lane in assertion["lanes"]
            ]
            if not applicable:
                raise TuningError(
                    "tuning manifest case has no assertions for exercised "
                    f"lane {lane!r} in case {case_index}"
                )
            if not any(
                (assertion.get("type") or assertion.get("kind"))
                == "quality_status"
                for assertion in applicable
            ):
                raise TuningError(
                    "tuning manifest case has no quality-status gate for "
                    f"exercised lane {lane!r} in case {case_index}"
                )
            expected_metric = completeness_metric.get(lane)
            if expected_metric is not None and not any(
                (assertion.get("type") or assertion.get("kind"))
                == "metric_range"
                and assertion.get("metric", assertion.get("path"))
                == expected_metric
                and isinstance(assertion.get("lanes"), list)
                and lane in assertion["lanes"]
                for assertion in applicable
            ):
                raise TuningError(
                    "tuning manifest case has no lane-scoped completeness "
                    f"gate for exercised lane {lane!r} in case {case_index}"
                )
    return len(manifest), tuple(case_ids)


def _stage_records(pipeline_id: str) -> tuple[tuple[str, str], ...]:
    try:
        return _STAGE_BY_PIPELINE[pipeline_id]
    except KeyError as exc:
        choices = ", ".join(sorted(_STAGE_BY_PIPELINE))
        raise TuningError(
            f"unknown tuning pipeline {pipeline_id!r}; expected {choices}"
        ) from exc


def _portable_hardware_selector(
    hardware: Mapping[str, Any],
) -> dict[str, Any]:
    selector = hardware.get("selector", hardware)
    if not isinstance(selector, Mapping):
        raise TuningError("hardware selector must be an object")
    required = (
        "vendor",
        "product_name",
        "compute_capability",
        "memory_model",
        "total_memory_bytes",
    )
    missing = [key for key in required if selector.get(key) is None]
    missing.extend(key for key in required if selector.get(key) == "")
    missing = list(dict.fromkeys(missing))
    if missing:
        raise TuningError(
            "hardware selector is missing " + ", ".join(missing)
        )
    result = {
        "schema_version": int(selector.get("schema_version", 1)),
        **{key: selector[key] for key in required},
        "pci_device_id": selector.get("pci_device_id"),
    }
    if (
        not isinstance(result["total_memory_bytes"], int)
        or result["total_memory_bytes"] <= 0
    ):
        raise TuningError("hardware total_memory_bytes must be positive")
    if result["memory_model"] not in {"dedicated", "unified"}:
        raise TuningError("hardware memory_model is unsupported")
    return result


def _memory_budget_records(
    hardware: Mapping[str, Any],
    tiers: Sequence[str],
) -> dict[str, dict[str, Any]]:
    supplied = hardware.get("memory_budgets")
    live_attested = False
    if isinstance(supplied, Mapping) and isinstance(
        supplied.get("tiers"), list
    ):
        live_attested = True
        supplied = {
            record.get("tier"): {
                **dict(record),
                "budget_fraction": record.get("target_fraction"),
                "budget_bytes": record.get("capacity_budget_bytes"),
            }
            for record in supplied["tiers"]
            if isinstance(record, Mapping)
        }
    if isinstance(supplied, Mapping):
        records = {}
        for tier in tiers:
            record = supplied.get(tier)
            if not isinstance(record, Mapping):
                raise TuningError(f"hardware has no memory budget for {tier}")
            budget = record.get("budget_bytes")
            if not isinstance(budget, int) or budget <= 0:
                raise TuningError(f"memory budget {tier} is invalid")
            if live_attested and (
                record.get("supported") is not True
                or record.get("full_budget_available") is not True
            ):
                reason_codes = record.get("reason_codes")
                detail = (
                    ", ".join(str(item) for item in reason_codes)
                    if isinstance(reason_codes, list) and reason_codes
                    else "tier is not fully available"
                )
                raise TuningError(
                    f"memory tier {tier} is not available for tuning: "
                    + detail
                )
            fraction = record.get("budget_fraction")
            reserve = record.get("reserve_bytes")
            if (
                not isinstance(fraction, (int, float))
                or isinstance(fraction, bool)
                or not 0 < float(fraction) < 1
                or not isinstance(reserve, int)
                or isinstance(reserve, bool)
                or reserve < 0
            ):
                raise TuningError(f"memory budget {tier} policy is invalid")
            records[tier] = {
                "tier": tier,
                "budget_fraction": float(fraction),
                "budget_bytes": budget,
                "reserve_bytes": reserve,
                "policy": record.get("policy") or (
                    "unified_system_reserve"
                    if _portable_hardware_selector(hardware)["memory_model"]
                    == "unified"
                    else "dedicated_vram_reserve"
                ),
            }
        return records

    selector = _portable_hardware_selector(hardware)
    total = selector["total_memory_bytes"]
    reserve = min(
        (16 if selector["memory_model"] == "unified" else 2) * 1024**3,
        total - 1,
    )
    max_fraction = 0.75 if selector["memory_model"] == "unified" else 0.9
    fractions = {"25": 0.25, "50": 0.5, "max": max_fraction}
    return {
        tier: {
            "tier": tier,
            "budget_fraction": fractions[tier],
            "budget_bytes": min(
                math.floor(total * fractions[tier]),
                total - reserve,
            ),
            "reserve_bytes": reserve,
            "policy": (
                "unified_system_reserve"
                if selector["memory_model"] == "unified"
                else "dedicated_vram_reserve"
            ),
        }
        for tier in tiers
    }


def _validate_installations(
    stages: Sequence[tuple[str, str]],
    installations: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result = {}
    for _capability, backend_id in stages:
        if backend_id in result:
            continue
        value = installations.get(backend_id)
        if not isinstance(value, Mapping):
            raise TuningError(
                f"missing machine installation for backend {backend_id}"
            )
        identity = value.get("identity")
        expected_identity_keys = {
            "record_sha256",
            "python_executable_sha256",
            "installed_distributions_count",
            "installed_distributions_sha256",
            "hf_cache_identity_sha256",
            "snapshot_files_sha256",
            "snapshot_weights_sha256",
            "adapter_sha256",
            "implementation_sha256",
        }
        if not isinstance(identity, Mapping) or set(identity) != (
            expected_identity_keys
        ):
            raise TuningError(
                f"missing exact installation identity for {backend_id}"
            )
        distribution_count = identity["installed_distributions_count"]
        if (
            type(distribution_count) is not int
            or distribution_count <= 0
        ):
            raise TuningError(
                "installation identity "
                f"{backend_id}.installed_distributions_count is invalid"
            )
        for name, digest in identity.items():
            if name == "installed_distributions_count":
                continue
            if name in {"snapshot_weights_sha256", "adapter_sha256"} and (
                digest is None
            ):
                continue
            if not isinstance(digest, str) or not _SHA256_RE.fullmatch(
                digest
            ):
                raise TuningError(
                    f"installation identity {backend_id}.{name} is invalid"
                )
        configured_attention = value.get("attention_backend")
        if not isinstance(configured_attention, str) or not (
            configured_attention.strip()
        ):
            configured_attention = (
                "sdpa" if backend_id.startswith("mineru") else "auto"
            )
        configured_attention = configured_attention.strip()
        result[backend_id] = {
            "python_executable": _require_absolute_existing_file(
                value.get("python_executable"),
                f"{backend_id} python_executable",
            ),
            "cache_dir": _require_absolute_existing_dir(
                value.get("cache_dir"),
                f"{backend_id} cache_dir",
            ),
            "cuda_visible_devices": str(
                value.get("cuda_visible_devices", "0")
            ),
            "device": str(value.get("device", "cuda:0")),
            "attention_backend": configured_attention,
            "identity": dict(identity),
        }
        if not result[backend_id]["cuda_visible_devices"]:
            raise TuningError("CUDA visible-device selection cannot be empty")
    return result


def _candidate_settings(
    backend_id: str,
    *,
    tier: str,
    budget: Mapping[str, Any],
    batch_size: int,
    total_memory_bytes: int,
    attention_backend: str,
    engine_overhead_headroom_bytes: int | None = None,
) -> dict[str, Any]:
    budget_bytes = int(budget["budget_bytes"])
    if backend_id == "mineru25-pro-2605":
        return {
            "backend": "official-two-step-transformers",
            "batch_size": batch_size,
            "image_analysis": False,
            "memory_tier": tier,
            "memory_budget_bytes": budget_bytes,
        }
    if backend_id != "deepseek-vllm":
        raise TuningError(f"backend has no tuning strategy: {backend_id}")
    if (
        not isinstance(engine_overhead_headroom_bytes, int)
        or isinstance(engine_overhead_headroom_bytes, bool)
        or engine_overhead_headroom_bytes <= 0
        or engine_overhead_headroom_bytes >= budget_bytes
    ):
        raise TuningError("DeepSeek engine overhead headroom is invalid")
    engine_target_bytes = budget_bytes - engine_overhead_headroom_bytes
    utilization = min(engine_target_bytes / total_memory_bytes, 0.95)
    return {
        "batch_size": batch_size,
        "max_num_seqs": batch_size,
        "max_num_batched_tokens": 8192 if batch_size == 1 else 16384,
        "gpu_memory_utilization": round(utilization, 8),
        "gpu_memory_utilization_denominator_bytes": total_memory_bytes,
        "engine_overhead_headroom_bytes": (
            engine_overhead_headroom_bytes
        ),
        "attention_backend": attention_backend,
        "mm_encoder_attn_backend": "auto",
        "enable_chunked_prefill": True,
        "max_tokens": 8192,
        "ngram_size": 10,
        "window_size": 50,
        "temperature": 0.0,
        "memory_tier": tier,
        "memory_budget_bytes": budget_bytes,
    }


def build_tuning_plan(
    *,
    pipeline_id: str,
    hardware: Mapping[str, Any],
    installations: Mapping[str, Mapping[str, Any]],
    manifest_path: str | os.PathLike[str],
    input_root: str | os.PathLike[str],
    memory_tiers: Sequence[str] = MEMORY_TIERS,
    batch_sizes: Sequence[int] = DEFAULT_BATCH_SIZES,
    discovery_repeats: int = 1,
    confirmation_repeats: int = 5,
    timeout_seconds: int = 3600,
    deepseek_engine_overhead_headroom_bytes: Sequence[int] = (
        DEFAULT_DEEPSEEK_ENGINE_OVERHEAD_HEADROOM_BYTES
    ),
) -> dict[str, Any]:
    """Build a deterministic, secret-free tuning-session plan."""

    stages = _stage_records(pipeline_id)
    tiers = _validate_tiers(memory_tiers)
    batches = _validate_batches(batch_sizes)
    if any(
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
        for value in (discovery_repeats, confirmation_repeats)
    ):
        raise TuningError("tuning repeat counts must be positive")
    if (
        not isinstance(timeout_seconds, (int, float))
        or isinstance(timeout_seconds, bool)
        or timeout_seconds <= 0
    ):
        raise TuningError("tuning timeout must be a finite positive number")
    try:
        timeout_is_finite = math.isfinite(timeout_seconds)
    except OverflowError:
        timeout_is_finite = False
    if not timeout_is_finite:
        raise TuningError("tuning timeout must be a finite positive number")
    manifest = Path(manifest_path).expanduser().resolve(strict=True)
    inputs = Path(input_root).expanduser().resolve(strict=True)
    if not manifest.is_file() or not inputs.is_dir():
        raise TuningError("manifest and input root must exist")
    exercised_lanes = frozenset(
        _stage_lane(capability, backend_id)
        for capability, backend_id in stages
    )
    manifest_case_count, manifest_case_ids = _manifest_summary(
        manifest,
        exercised_lanes=exercised_lanes,
    )
    page_batches = tuple(
        dict.fromkeys(
            min(batch_size, manifest_case_count) for batch_size in batches
        )
    )
    selector = _portable_hardware_selector(hardware)
    budgets = _memory_budget_records(hardware, tiers)
    bindings = _validate_installations(stages, installations)
    headrooms = tuple(
        dict.fromkeys(deepseek_engine_overhead_headroom_bytes)
    )
    if not headrooms or any(
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
        for value in headrooms
    ):
        raise TuningError(
            "DeepSeek engine overhead headrooms must be positive integers"
        )
    torch_total = hardware.get("torch_cuda_total_memory_bytes")
    if not isinstance(torch_total, int) or isinstance(torch_total, bool):
        memory_budgets = hardware.get("memory_budgets")
        memory_evidence = (
            memory_budgets.get("memory_evidence")
            if isinstance(memory_budgets, Mapping)
            else None
        )
        total_sources = (
            memory_evidence.get("total_sources")
            if isinstance(memory_evidence, Mapping)
            else None
        )
        torch_total = (
            total_sources.get("torch_cuda")
            if isinstance(total_sources, Mapping)
            else None
        )
    if not isinstance(torch_total, int) or torch_total <= 0:
        torch_total = selector["total_memory_bytes"]
    utilization_total = max(selector["total_memory_bytes"], torch_total)
    candidates = []
    for capability, backend_id in stages:
        binding = bindings[backend_id]
        for tier in tiers:
            candidate_batches = (
                batches if backend_id.startswith("mineru") else page_batches
            )
            for batch_size in candidate_batches:
                candidate_headrooms = (
                    tuple(
                        value
                        for value in headrooms
                        if value < budgets[tier]["budget_bytes"]
                    )
                    if backend_id == "deepseek-vllm"
                    else (None,)
                )
                for headroom in candidate_headrooms:
                    settings = _candidate_settings(
                        backend_id,
                        tier=tier,
                        budget=budgets[tier],
                        batch_size=batch_size,
                        total_memory_bytes=utilization_total,
                        attention_backend=binding["attention_backend"],
                        engine_overhead_headroom_bytes=headroom,
                    )
                    identity = {
                        "pipeline_id": pipeline_id,
                        "capability": capability,
                        "backend_id": backend_id,
                        "model_revision": get_backend(
                            backend_id
                        ).model_revision,
                        "hardware_selector": selector,
                        "installation_identity": binding["identity"],
                        "tier": tier,
                        "settings": settings,
                    }
                    candidate_id = sha256_json(identity)
                    candidates.append(
                        {
                            "candidate_id": candidate_id,
                            **identity,
                            "batch_semantics": (
                                "layout_block_batch"
                                if backend_id.startswith("mineru")
                                else "page_batch"
                            ),
                        }
                    )
    if len(candidates) > MAX_CANDIDATES:
        raise TuningError(
            f"tuning plan has {len(candidates)} candidates; maximum is "
            f"{MAX_CANDIDATES}"
        )
    payload = {
        "pipeline_id": pipeline_id,
        "hardware": {
            "selector": selector,
            "torch_cuda_total_memory_bytes": torch_total,
            "utilization_denominator_bytes": utilization_total,
            "observation_sha256": sha256_json(
                {
                    "selector": selector,
                    "torch_cuda_total_memory_bytes": torch_total,
                }
            ),
        },
        "memory_budgets": budgets,
        "manifest": {
            "path": str(manifest),
            "sha256": file_sha256(manifest),
            "case_count": manifest_case_count,
            "case_ids": list(manifest_case_ids),
        },
        "input_root": str(inputs),
        "installations": bindings,
        "policy": {
            "memory_tiers": list(tiers),
            "batch_sizes": list(batches),
            "deepseek_engine_overhead_headroom_bytes": list(headrooms),
            "batch_bounds": {
                "manifest_case_count": manifest_case_count,
                "page_batch_max_size": manifest_case_count,
                "page_batch_sizes": list(page_batches),
                "layout_block_batch_sizes": list(batches),
            },
            "discovery_repeats": discovery_repeats,
            "confirmation_repeats": confirmation_repeats,
            "timeout_seconds": timeout_seconds,
            "quality_first": True,
            "output_limit_terminations_allowed": 0,
            "suite_role": "hardware_smoke",
            "suite_id": "cuphoton-three-page-smoke",
        },
        "candidates": candidates,
    }
    return make_receipt("ocr_profile_tuning_plan", payload)


def _adapter_path(backend_id: str) -> Path:
    name = (
        "mineru25.py"
        if backend_id == "mineru25-pro-2605"
        else "deepseek_vllm.py"
    )
    path = Path(__file__).resolve().parent / "providers" / name
    try:
        return validate_trusted_file(path, label="tuning provider adapter")
    except TrustedPathError as exc:
        raise TuningError(str(exc)) from exc


def build_candidate_provider_spec(
    plan: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    backend_id = candidate["backend_id"]
    backend = get_backend(backend_id)
    binding = plan["installations"][backend_id]
    adapter = _adapter_path(backend_id)
    argv = [
        binding["python_executable"],
        "-I",
        "-B",
        str(adapter),
        "--model-id",
        backend.model_id,
        "--revision",
        backend.model_revision,
        "--device",
        binding.get("device", "cuda:0"),
        "--tuning",
    ]
    if backend_id == "mineru25-pro-2605":
        argv.extend(
            [
                "--attention-backend",
                binding.get("attention_backend", "sdpa"),
            ]
        )
        lane = _stage_lane(candidate["capability"], backend_id)
        prompt = backend.provider_prompt
    else:
        if candidate.get("capability") != "grounded_ocr":
            raise TuningError(
                "DeepSeek profile tuning supports grounded OCR only"
            )
        lane = _stage_lane(candidate["capability"], backend_id)
        prompt = "Free OCR."
    return {
        "schema_version": 1,
        "provider_id": f"{backend_id}-profile-tuning",
        "argv": argv,
        "egress": "none",
        "model": {
            "requested": backend.model_id,
            "revision": backend.model_revision,
        },
        "lanes": {
            lane: {
                "prompt": prompt,
                "settings": candidate["settings"],
                "include_raw_text": False,
            }
        },
        "timeout_seconds": plan["policy"]["timeout_seconds"],
        "secret_env_names": [],
        "env": {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": binding["cuda_visible_devices"],
            "HF_HUB_CACHE": binding["cache_dir"],
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
    }


def _nested(value: Mapping[str, Any], dotted: str) -> Any:
    current: Any = value
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _read_results(path: Path) -> list[dict[str, Any]]:
    values = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    value = json.loads(
                        line,
                        object_pairs_hook=_unique_json_object,
                    )
                    if not isinstance(value, dict):
                        raise TuningError(
                            "benchmark result must be an object at line "
                            f"{line_number}"
                        )
                    values.append(value)
    except (OSError, ValueError) as exc:
        raise TuningError(f"could not read benchmark results: {exc}") from exc
    return values


def _generation_settings_evidence(
    *,
    candidate: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Prove that every provider response used the candidate settings."""

    backend_id = candidate["backend_id"]
    declared = dict(candidate["settings"])
    observed_digests = set()
    reasons: list[str] = []
    for result in results:
        provider = _nested(result, "provenance.provider")
        params = _nested(provider, "generation.params") if isinstance(
            provider, Mapping
        ) else None
        if not isinstance(params, Mapping):
            reasons.append("missing_effective_generation_settings")
            continue
        if backend_id == "deepseek-vllm":
            effective = params.get("execution_settings")
            sampling = params.get("sampling_params")
            extra_args = params.get("logits_processor_extra_args")
            engine_kwargs = params.get("engine_init_kwargs")
            if not all(
                isinstance(value, Mapping)
                for value in (
                    effective,
                    sampling,
                    extra_args,
                    engine_kwargs,
                )
            ):
                reasons.append("missing_effective_generation_settings")
                continue
            normalized = dict(declared)
            for key in ("attention_backend", "mm_encoder_attn_backend"):
                if normalized[key] == "auto":
                    normalized[key] = None
            if any(
                effective.get(key) != expected
                for key, expected in normalized.items()
            ):
                reasons.append("effective_generation_settings_mismatch")
            for key in (
                "enable_chunked_prefill",
                "gpu_memory_utilization",
                "max_num_batched_tokens",
                "max_num_seqs",
            ):
                if engine_kwargs.get(key) != normalized[key]:
                    reasons.append("effective_engine_settings_mismatch")
            for key in ("attention_backend", "mm_encoder_attn_backend"):
                expected = normalized[key]
                if expected is None:
                    if engine_kwargs.get(key) is not None:
                        reasons.append("effective_engine_settings_mismatch")
                elif engine_kwargs.get(key) != expected:
                    reasons.append("effective_engine_settings_mismatch")
            if any(
                sampling.get(key) != normalized[key]
                for key in ("max_tokens", "temperature")
            ):
                reasons.append("effective_sampling_settings_mismatch")
            if any(
                extra_args.get(key) != normalized[key]
                for key in ("ngram_size", "window_size")
            ):
                reasons.append("effective_logits_settings_mismatch")
            observed = {
                "execution_settings": {
                    key: effective.get(key) for key in sorted(normalized)
                },
                "engine_settings": {
                    key: engine_kwargs.get(key)
                    for key in (
                        "attention_backend",
                        "enable_chunked_prefill",
                        "gpu_memory_utilization",
                        "max_num_batched_tokens",
                        "max_num_seqs",
                        "mm_encoder_attn_backend",
                    )
                },
                "sampling_settings": {
                    key: sampling.get(key)
                    for key in ("max_tokens", "temperature")
                },
                "logits_settings": {
                    key: extra_args.get(key)
                    for key in ("ngram_size", "window_size")
                },
            }
        elif backend_id == "mineru25-pro-2605":
            request_settings = params.get("request_settings")
            pipeline = params.get("pipeline")
            memory_budget = _nested(provider, "runtime.memory_budget")
            if not all(
                isinstance(value, Mapping)
                for value in (request_settings, pipeline, memory_budget)
            ):
                reasons.append("missing_effective_generation_settings")
                continue
            if dict(request_settings) != declared:
                reasons.append("effective_generation_settings_mismatch")
            if (
                pipeline.get("batch_size") != declared["batch_size"]
                or pipeline.get("image_analysis")
                != declared["image_analysis"]
                or pipeline.get("backend") != "transformers"
            ):
                reasons.append("effective_pipeline_settings_mismatch")
            if (
                memory_budget.get("memory_tier")
                != declared["memory_tier"]
                or memory_budget.get("budget_bytes")
                != declared["memory_budget_bytes"]
                or memory_budget.get("allocator_cap_applied") is not True
            ):
                reasons.append("effective_memory_settings_mismatch")
            observed = {
                "request_settings": dict(request_settings),
                "pipeline_settings": {
                    "backend": pipeline.get("backend"),
                    "batch_size": pipeline.get("batch_size"),
                    "image_analysis": pipeline.get("image_analysis"),
                },
                "memory_budget": {
                    "memory_tier": memory_budget.get("memory_tier"),
                    "budget_bytes": memory_budget.get("budget_bytes"),
                    "allocator_cap_applied": memory_budget.get(
                        "allocator_cap_applied"
                    ),
                },
            }
        else:
            reasons.append("unsupported_generation_settings_backend")
            continue
        observed_digests.add(sha256_json(observed))
    if len(observed_digests) != 1:
        reasons.append("effective_generation_settings_drift")
    return {
        "effective_settings_sha256s": sorted(observed_digests),
        "settings_reasons": list(dict.fromkeys(reasons)),
    }


def _installed_distributions_evidence(
    value: Any,
) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    if set(value) != {"count", "sha256"}:
        return None
    count = value.get("count")
    digest = value.get("sha256")
    if (
        type(count) is not int
        or count <= 0
        or not isinstance(digest, str)
        or not _SHA256_RE.fullmatch(digest)
    ):
        return None
    return {"count": count, "sha256": digest}


def _candidate_installation_identity_reasons(
    *,
    candidate: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
) -> list[str]:
    expected = candidate.get("installation_identity")
    if not isinstance(expected, Mapping):
        return ["missing_planned_installation_identity"]
    expected_distributions = {
        "count": expected.get("installed_distributions_count"),
        "sha256": expected.get("installed_distributions_sha256"),
    }
    snapshot_identities = set()
    implementation_identities = set()
    distribution_identities: dict[str, dict[str, Any]] = {}
    reasons = []
    for result in results:
        provider = _nested(result, "provenance.provider")
        if not isinstance(provider, Mapping):
            reasons.append("missing_provider_installation_identity")
            continue
        snapshot = _nested(provider, "model.snapshot_files_sha256")
        if not isinstance(snapshot, str) or not _SHA256_RE.fullmatch(
            snapshot
        ):
            reasons.append("missing_model_snapshot_identity")
        else:
            snapshot_identities.add(snapshot)
        implementation = _nested(provider, "runtime.implementation.sha256")
        if (
            not isinstance(implementation, str)
            or not _SHA256_RE.fullmatch(implementation)
        ):
            reasons.append("missing_implementation_identity")
        else:
            implementation_identities.add(implementation)
        distribution = _installed_distributions_evidence(
            _nested(provider, "runtime.installed_distributions")
        )
        if distribution is None:
            reasons.append("invalid_installed_distributions_identity")
        else:
            distribution_identities[sha256_json(distribution)] = distribution
    expected_snapshot = expected.get("snapshot_files_sha256")
    if snapshot_identities != {expected_snapshot}:
        reasons.append("model_snapshot_identity_mismatch")
    expected_implementation = expected.get("implementation_sha256")
    if implementation_identities != {expected_implementation}:
        reasons.append("implementation_identity_mismatch")
    if (
        len(distribution_identities) != 1
        or next(iter(distribution_identities.values()), None)
        != expected_distributions
    ):
        reasons.append("installed_distributions_identity_mismatch")
    return list(dict.fromkeys(reasons))


def _benchmark_matrix_reasons(
    *,
    summary: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    run: Mapping[str, Any],
    expected_case_ids: Sequence[str],
    expected_repeats: int,
    expected_lane: str,
) -> list[str]:
    case_ids = tuple(expected_case_ids)
    if (
        not case_ids
        or len(case_ids) != len(set(case_ids))
        or any(
            not isinstance(case_id, str) or not case_id
            for case_id in case_ids
        )
        or not isinstance(expected_repeats, int)
        or isinstance(expected_repeats, bool)
        or expected_repeats <= 0
        or not isinstance(expected_lane, str)
        or not expected_lane
    ):
        raise TuningError("invalid expected benchmark evidence matrix")

    expected_keys = {
        (case_id, expected_lane, repeat)
        for case_id in case_ids
        for repeat in range(1, expected_repeats + 1)
    }
    actual_keys: set[tuple[Any, Any, Any]] = set()
    duplicate = False
    invalid_identity = False
    row_failure = False
    process_keys_by_repeat: dict[int, set[str]] = {
        repeat: set() for repeat in range(1, expected_repeats + 1)
    }
    for result in results:
        key = (
            result.get("case_id"),
            result.get("lane"),
            result.get("repeat"),
        )
        if (
            not isinstance(key[0], str)
            or not key[0]
            or not isinstance(key[1], str)
            or not key[1]
            or type(key[2]) is not int
        ):
            invalid_identity = True
        if key in actual_keys:
            duplicate = True
        actual_keys.add(key)
        if (
            result.get("status") != "completed"
            or result.get("passed") is not True
            or result.get("metadata_complete") is not True
        ):
            row_failure = True
        repeat = result.get("repeat")
        process = result.get("subprocess")
        process_key = (
            process.get("requests_jsonl_sha256")
            if isinstance(process, Mapping)
            else None
        )
        if (
            isinstance(repeat, int)
            and not isinstance(repeat, bool)
            and repeat in process_keys_by_repeat
            and isinstance(process_key, str)
            and _SHA256_RE.fullmatch(process_key)
        ):
            process_keys_by_repeat[repeat].add(process_key)

    expected_count = len(expected_keys)
    reasons: list[str] = []
    if len(results) != expected_count:
        reasons.append("result_matrix_count_mismatch")
    if duplicate:
        reasons.append("result_matrix_duplicate")
    if invalid_identity:
        reasons.append("result_matrix_identity_invalid")
    if actual_keys != expected_keys:
        reasons.append("result_matrix_coverage_mismatch")
    if row_failure:
        reasons.append("result_matrix_status_failure")
    if any(len(keys) != 1 for keys in process_keys_by_repeat.values()):
        reasons.append("benchmark_process_matrix_mismatch")
    if len(set().union(*process_keys_by_repeat.values())) != expected_repeats:
        reasons.append("benchmark_process_matrix_mismatch")

    if any(
        type(summary.get(key)) is not int
        or summary.get(key) != expected_count
        for key in ("result_count", "attempt_count", "passed_attempt_count")
    ):
        reasons.append("summary_count_mismatch")
    run_repeats = _nested(run, "harness.repeats")
    if type(run_repeats) is not int or run_repeats != expected_repeats:
        reasons.append("benchmark_repeat_count_mismatch")
    if _nested(run, "harness.lanes") != [expected_lane]:
        reasons.append("benchmark_lane_mismatch")
    return list(dict.fromkeys(reasons))


def _plan_case_ids(plan: Mapping[str, Any]) -> tuple[str, ...]:
    manifest = plan.get("manifest")
    case_ids = (
        manifest.get("case_ids") if isinstance(manifest, Mapping) else None
    )
    if (
        not isinstance(case_ids, list)
        or not case_ids
        or len(case_ids) != len(set(case_ids))
        or any(
            not isinstance(case_id, str) or not case_id
            for case_id in case_ids
        )
        or manifest.get("case_count") != len(case_ids)
    ):
        raise TuningError("tuning plan has no valid manifest case matrix")
    return tuple(case_ids)


def evaluate_candidate_evidence(
    *,
    candidate: Mapping[str, Any],
    benchmark_dir: str | os.PathLike[str],
    expected_case_ids: Sequence[str],
    expected_repeats: int,
) -> dict[str, Any]:
    root = Path(benchmark_dir)
    summary = _read_json(root / "summary.json")
    results = _read_results(root / "results.jsonl")
    run = _read_json(root / "run.json")
    budget = int(candidate["settings"]["memory_budget_bytes"])
    memory_records = []
    terminations = []
    processes: dict[str, dict[str, Any]] = {}
    inconsistent_process_telemetry = False
    for result in results:
        provider = _nested(result, "provenance.provider")
        if isinstance(provider, Mapping):
            memory = _nested(provider, "runtime.gpu_memory")
            if isinstance(memory, Mapping):
                memory_records.append(dict(memory))
            terminations.append(
                {
                    "eos": _nested(provider, "generation.eos"),
                    "max_tokens": _nested(provider, "generation.max_tokens"),
                }
            )
        process = result.get("subprocess")
        if isinstance(process, Mapping):
            key = process.get("requests_jsonl_sha256")
            if isinstance(key, str):
                process_record = dict(process)
                existing = processes.setdefault(key, process_record)
                if existing != process_record:
                    inconsistent_process_telemetry = True
    dedicated_peak_values: list[tuple[str, int]] = []
    unified_peak_values: list[tuple[str, int]] = []
    unified_aggregate_values: list[tuple[str, int]] = []
    memory_kinds = set()
    for memory in memory_records:
        memory_kind = memory.get("memory_kind")
        memory_kinds.add(memory_kind)
        for key in (
            "torch_peak_reserved_bytes",
            "observed_cuda_used_delta_bytes",
        ):
            value = memory.get(key)
            if type(value) is int and value >= 0:
                dedicated_peak_values.append((key, value))
                unified_peak_values.append((key, value))
        aggregate_rss = memory.get("parent_plus_children_peak_rss_bytes")
        if type(aggregate_rss) is not int or aggregate_rss < 0:
            aggregate_rss = memory.get("process_peak_rss_bytes")
        if type(aggregate_rss) is int and aggregate_rss >= 0:
            unified_peak_values.append(
                ("aggregate_process_rss", aggregate_rss)
            )
        linux_delta = memory.get("observed_linux_mem_available_delta_bytes")
        if type(linux_delta) is int and linux_delta >= 0:
            record = ("linux_mem_available_delta", linux_delta)
            unified_peak_values.append(record)
            unified_aggregate_values.append(record)
        cgroup = memory.get("cgroup_v2_memory")
        if isinstance(cgroup, Mapping):
            for key in (
                "observed_current_delta_bytes",
                "observed_peak_delta_bytes",
                "observed_headroom_delta_bytes",
            ):
                value = cgroup.get(key)
                if type(value) is int and value >= 0:
                    record = (f"cgroup_v2.{key}", value)
                    unified_peak_values.append(record)
                    unified_aggregate_values.append(record)
    if memory_kinds == {"unified_system"}:
        peak_values = unified_peak_values
    elif memory_kinds == {"dedicated_vram"}:
        peak_values = dedicated_peak_values
    else:
        peak_values = []
    peak = max((value for _source, value in peak_values), default=None)
    peak_sources = sorted(
        {source for source, value in peak_values if value == peak}
    )
    walls = [
        float(process["wall_seconds"])
        for process in processes.values()
        if isinstance(process.get("wall_seconds"), (int, float))
        and not isinstance(process["wall_seconds"], bool)
        and math.isfinite(float(process["wall_seconds"]))
        and process["wall_seconds"] > 0
    ]
    page_count = len(results)
    throughput = page_count / sum(walls) if walls else None
    reasons = []
    reasons.extend(
        _benchmark_matrix_reasons(
            summary=summary,
            results=results,
            run=run,
            expected_case_ids=expected_case_ids,
            expected_repeats=expected_repeats,
            expected_lane=_stage_lane(
                str(candidate["capability"]),
                str(candidate["backend_id"]),
            ),
        )
    )
    if inconsistent_process_telemetry:
        reasons.append("inconsistent_process_telemetry")
    if summary.get("all_pass") is not True:
        reasons.append("quality_or_protocol_failure")
    if summary.get("provenance_complete") is not True:
        reasons.append("incomplete_provenance")
    if (
        len(walls) != expected_repeats
        or throughput is None
        or not math.isfinite(throughput)
        or throughput <= 0
    ):
        reasons.append("missing_throughput_telemetry")
    if not memory_records or peak is None or peak <= 0:
        reasons.append("missing_memory_telemetry")
    elif peak > budget:
        reasons.append("memory_budget_exceeded")
    if memory_kinds == {"unified_system"} and not unified_aggregate_values:
        reasons.append("missing_unified_system_memory_telemetry")
    if any(item.get("max_tokens") is True for item in terminations):
        reasons.append("max_tokens_termination")
    if any(item.get("eos") is not True for item in terminations):
        reasons.append("non_eos_termination")
    if len(memory_kinds) != 1 or None in memory_kinds:
        reasons.append("inconsistent_memory_kind")
    settings_evidence = _generation_settings_evidence(
        candidate=candidate,
        results=results,
    )
    reasons.extend(settings_evidence["settings_reasons"])
    reasons.extend(
        _candidate_installation_identity_reasons(
            candidate=candidate,
            results=results,
        )
    )
    reasons = list(dict.fromkeys(reasons))
    return make_receipt(
        "ocr_profile_candidate_result",
        {
            "candidate_id": candidate["candidate_id"],
            "eligible": not reasons,
            "ineligible_reasons": reasons,
            "summary_sha256": file_sha256(root / "summary.json"),
            "results_sha256": file_sha256(root / "results.jsonl"),
            "run_sha256": file_sha256(root / "run.json"),
            "attempt_count": summary.get("attempt_count"),
            "passed_attempt_count": summary.get("passed_attempt_count"),
            "process_count": len(processes),
            "cold_process_wall_seconds": walls,
            "cold_pages_per_second": throughput,
            "peak_memory_bytes": peak,
            "peak_memory_sources": peak_sources,
            "memory_budget_bytes": budget,
            "memory_kinds": sorted(str(item) for item in memory_kinds),
            "termination_records": terminations,
            "declared_settings_sha256": sha256_json(
                candidate["settings"]
            ),
            "effective_settings_sha256s": settings_evidence[
                "effective_settings_sha256s"
            ],
        },
    )


def _runtime_and_model_evidence(
    *,
    candidate: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    configured_attention_backend: str,
) -> dict[str, Any]:
    """Extract one portable, exact runtime/model identity from responses."""

    backend = get_backend(candidate["backend_id"])
    installation_identity = candidate.get("installation_identity")
    if not isinstance(installation_identity, Mapping):
        installation_identity = {}
    expected_packages = dict(backend.runtime_packages)
    runtime_selector_candidates: dict[str, dict[str, Any]] = {}
    model_records: dict[str, dict[str, Any]] = {}
    observed_attention_backends = set()
    implementation_sha256s = set()
    reasons = []
    for result in results:
        provider = _nested(result, "provenance.provider")
        if not isinstance(provider, Mapping):
            reasons.append("missing_provider_evidence")
            continue
        runtime = provider.get("runtime")
        if not isinstance(runtime, Mapping):
            reasons.append("missing_runtime_evidence")
        else:
            observed_packages = runtime.get("packages")
            packages = {}
            if not isinstance(observed_packages, Mapping):
                reasons.append("missing_runtime_packages")
            else:
                missing_packages = []
                mismatched_packages = []
                for name, expected_version in expected_packages.items():
                    version = observed_packages.get(name)
                    if not isinstance(version, str) or not version:
                        missing_packages.append(name)
                    elif version != expected_version:
                        mismatched_packages.append(name)
                    else:
                        packages[name] = version
                if missing_packages:
                    reasons.append("missing_runtime_packages")
                    reasons.extend(
                        f"missing_runtime_package:{name}"
                        for name in missing_packages
                    )
                if mismatched_packages:
                    reasons.append("runtime_package_version_mismatch")
                    reasons.extend(
                        f"runtime_package_version_mismatch:{name}"
                        for name in mismatched_packages
                    )
            torch_version = runtime.get("torch")
            if not isinstance(torch_version, str) or not torch_version:
                torch_version = (
                    packages.get("torch")
                    if isinstance(packages, Mapping)
                    else None
                )
            selector = {
                "python": runtime.get("python"),
                "engine": runtime.get("engine"),
                "engine_version": runtime.get("engine_version"),
                "torch": torch_version,
                "torch_cuda": runtime.get("torch_cuda"),
                "packages": packages,
                "dtype": runtime.get("dtype"),
            }
            installed_distributions = _installed_distributions_evidence(
                runtime.get("installed_distributions")
            )
            if installed_distributions is None:
                reasons.append("invalid_installed_distributions_identity")
            else:
                installed_count = installed_distributions["count"]
                installed_sha256 = installed_distributions["sha256"]
                selector["installed_distributions_count"] = installed_count
                selector["installed_distributions_sha256"] = installed_sha256
                if (
                    installed_count
                    != installation_identity.get(
                        "installed_distributions_count"
                    )
                    or installed_sha256
                    != installation_identity.get(
                        "installed_distributions_sha256"
                    )
                ):
                    reasons.append(
                        "installed_distributions_identity_mismatch"
                    )
            expected_runtime = {
                "python": backend.default_python,
                "engine": backend.runtime_engine,
                "engine_version": expected_packages[
                    backend.runtime_engine_package
                ],
                "torch": backend.qualified_torch_version,
                "torch_cuda": backend.qualified_torch_cuda,
                "dtype": backend.qualified_dtype,
            }
            runtime_mismatches = [
                key
                for key, expected in expected_runtime.items()
                if selector.get(key) != expected
            ]
            if runtime_mismatches:
                reasons.append("runtime_identity_mismatch")
                reasons.extend(
                    f"runtime_identity_mismatch:{key}"
                    for key in runtime_mismatches
                )
            if len(packages) != len(expected_packages) or any(
                not isinstance(selector[key], str) or not selector[key]
                for key in (
                    "python",
                    "engine",
                    "engine_version",
                    "torch",
                    "torch_cuda",
                    "dtype",
                    "installed_distributions_sha256",
                )
            ) or runtime_mismatches:
                reasons.append("incomplete_runtime_identity")
            else:
                runtime_selector_candidates[sha256_json(selector)] = selector
            selected_attention = runtime.get("attention_backend")
            if isinstance(selected_attention, str) and selected_attention:
                observed_attention_backends.add(selected_attention)
            else:
                reasons.append("missing_selected_attention_backend")
            implementation = runtime.get("implementation")
            implementation_sha256 = (
                implementation.get("sha256")
                if isinstance(implementation, Mapping)
                else None
            )
            if implementation_sha256 != installation_identity.get(
                "implementation_sha256"
            ):
                reasons.append("implementation_identity_mismatch")
            else:
                implementation_sha256s.add(implementation_sha256)

        model = provider.get("model")
        if not isinstance(model, Mapping):
            reasons.append("missing_model_evidence")
            continue
        model_record = {
            "requested": model.get("requested"),
            "revision": model.get("revision"),
            "resolved_revision": model.get("resolved_revision"),
            "snapshot_files_sha256": model.get("snapshot_files_sha256"),
        }
        if model_record["requested"] != backend.model_id:
            reasons.append("model_id_mismatch")
        if model_record["revision"] != candidate["model_revision"]:
            reasons.append("model_revision_mismatch")
        if model_record["resolved_revision"] != candidate["model_revision"]:
            reasons.append("resolved_model_revision_mismatch")
        snapshot_files_sha256 = model_record["snapshot_files_sha256"]
        if not _SHA256_RE.fullmatch(str(snapshot_files_sha256 or "")):
            reasons.append("missing_model_snapshot_identity")
        elif snapshot_files_sha256 != installation_identity.get(
            "snapshot_files_sha256"
        ):
            reasons.append("model_snapshot_identity_mismatch")
        else:
            model_records[sha256_json(model_record)] = model_record

    selected_attention_backend = None
    if len(observed_attention_backends) != 1:
        reasons.append("selected_attention_backend_drift")
    else:
        selected_attention_backend = next(iter(observed_attention_backends))
        if (
            configured_attention_backend != "auto"
            and selected_attention_backend != configured_attention_backend
        ):
            reasons.append("selected_attention_backend_mismatch")
            selected_attention_backend = None

    runtime_selectors: dict[str, dict[str, Any]] = {}
    if selected_attention_backend is not None:
        for selector in runtime_selector_candidates.values():
            sealed_selector = {
                **selector,
                "attention_backend": selected_attention_backend,
                "implementation_sha256": (
                    next(iter(implementation_sha256s))
                    if len(implementation_sha256s) == 1
                    else None
                ),
            }
            runtime_selectors[sha256_json(sealed_selector)] = sealed_selector
    if len(runtime_selectors) != 1:
        reasons.append("runtime_identity_drift")
    if len(implementation_sha256s) != 1:
        reasons.append("implementation_identity_drift")
    if len(model_records) != 1:
        reasons.append("model_identity_drift")
    return {
        "runtime_selector": (
            next(iter(runtime_selectors.values()))
            if len(runtime_selectors) == 1
            else None
        ),
        "model_evidence": (
            next(iter(model_records.values()))
            if len(model_records) == 1
            else None
        ),
        "observed_attention_backends": sorted(observed_attention_backends),
        "identity_reasons": list(dict.fromkeys(reasons)),
    }


def _configured_attention_backend(
    plan: Mapping[str, Any], candidate: Mapping[str, Any]
) -> str:
    value = candidate.get("settings", {}).get("attention_backend")
    if not isinstance(value, str) or not value:
        value = plan["installations"][candidate["backend_id"]].get(
            "attention_backend"
        )
    if not isinstance(value, str) or not value:
        value = (
            "sdpa" if candidate["backend_id"].startswith("mineru") else "auto"
        )
    return value


def _confirmation_identity(
    *,
    plan: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    identity = {
        "plan_receipt_sha256": plan["receipt_sha256"],
        "candidate_id": selection["candidate_id"],
        "discovery_evidence_receipt_sha256": selection[
            "evidence_receipt_sha256"
        ],
        "manifest_sha256": plan["manifest"]["sha256"],
        "repeats": plan["policy"]["confirmation_repeats"],
        "suite_id": plan["policy"].get(
            "suite_id", "cuphoton-three-page-smoke"
        ),
    }
    return sha256_json(identity), identity


def _validate_confirmation_binding(
    *,
    plan: Mapping[str, Any],
    selection: Mapping[str, Any],
    confirmation: Mapping[str, Any],
) -> None:
    _validated_receipt_sha256(confirmation, "tuning confirmation")
    expected_id, _identity = _confirmation_identity(
        plan=plan,
        selection=selection,
    )
    candidates = {
        candidate.get("candidate_id"): candidate
        for candidate in plan.get("candidates", [])
        if isinstance(candidate, Mapping)
    }
    candidate = candidates.get(selection.get("candidate_id"))
    if candidate is None:
        raise TuningError("confirmation selection is not present in the plan")
    expected_fields = {
        "confirmation_id": expected_id,
        "candidate_id": selection.get("candidate_id"),
        "capability": selection.get("capability"),
        "backend_id": selection.get("backend_id"),
        "tier": selection.get("tier"),
        "repeats": plan["policy"]["confirmation_repeats"],
        "manifest_sha256": plan["manifest"]["sha256"],
        "settings_sha256": sha256_json(selection.get("settings")),
    }
    if any(
        confirmation.get(key) != expected
        for key, expected in expected_fields.items()
    ) or any(
        candidate.get(key) != selection.get(key)
        for key in ("capability", "backend_id", "tier", "settings")
    ):
        raise TuningError(
            "confirmation receipt does not match the current plan"
        )


def evaluate_confirmation_evidence(
    *,
    plan: Mapping[str, Any],
    candidate: Mapping[str, Any],
    selection: Mapping[str, Any],
    confirmation_id: str,
    benchmark_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Evaluate a repeated confirmation and bind its execution identity."""

    root = Path(benchmark_dir)
    candidate_result = evaluate_candidate_evidence(
        candidate=candidate,
        benchmark_dir=root,
        expected_case_ids=plan["manifest"]["case_ids"],
        expected_repeats=plan["policy"]["confirmation_repeats"],
    )
    benchmark_summary = _read_json(root / "summary.json")
    results = _read_results(root / "results.jsonl")
    run = _read_json(root / "run.json")
    execution = _runtime_and_model_evidence(
        candidate=candidate,
        results=results,
        configured_attention_backend=_configured_attention_backend(
            plan, candidate
        ),
    )
    reasons = list(candidate_result["ineligible_reasons"])
    reasons.extend(execution["identity_reasons"])
    repeats = _nested(run, "harness.repeats")
    expected_repeats = plan["policy"]["confirmation_repeats"]
    if repeats != expected_repeats:
        reasons.append("confirmation_repeat_count_mismatch")
    manifest_sha256 = _nested(run, "harness.manifest_file_sha256")
    if manifest_sha256 != plan["manifest"]["sha256"]:
        reasons.append("confirmation_manifest_mismatch")
    experiment_sha256 = _nested(run, "harness.experiment_sha256")
    if not _SHA256_RE.fullmatch(str(experiment_sha256 or "")):
        reasons.append("missing_experiment_identity")
        experiment_sha256 = None
    reasons = list(dict.fromkeys(reasons))
    return make_receipt(
        "ocr_profile_confirmation_result",
        {
            "confirmation_id": confirmation_id,
            "candidate_id": candidate["candidate_id"],
            "capability": candidate["capability"],
            "backend_id": candidate["backend_id"],
            "tier": candidate["tier"],
            "repeats": repeats,
            "eligible": not reasons,
            "ineligible_reasons": reasons,
            "all_pass": benchmark_summary.get("all_pass") is True,
            "provenance_complete": (
                benchmark_summary.get("provenance_complete") is True
            ),
            "manifest_sha256": manifest_sha256,
            "summary_sha256": candidate_result["summary_sha256"],
            "results_sha256": candidate_result["results_sha256"],
            "run_sha256": file_sha256(root / "run.json"),
            "experiment_sha256": experiment_sha256,
            "peak_memory_bytes": candidate_result.get("peak_memory_bytes"),
            "memory_budget_bytes": candidate_result.get(
                "memory_budget_bytes"
            ),
            "cold_pages_per_second": candidate_result.get(
                "cold_pages_per_second"
            ),
            "settings_sha256": sha256_json(candidate["settings"]),
            "runtime_selector": execution["runtime_selector"],
            "model_evidence": execution["model_evidence"],
            "observed_attention_backends": execution[
                "observed_attention_backends"
            ],
        },
    )


def _completed_receipt(
    root: Path,
    *,
    filename: str,
    identity_field: str,
    identity: str,
) -> dict[str, Any] | None:
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise TuningError(f"tuning receipt root is unsafe: {root}")
    matches = sorted(root.glob(f"attempt-*/{filename}"))
    for path in reversed(matches):
        if path.parent.is_symlink() or path.is_symlink():
            raise TuningError(f"tuning receipt path is unsafe: {path}")
        value = _read_json(path)
        if value.get(identity_field) != identity:
            continue
        digest = value.get("receipt_sha256")
        payload = dict(value)
        payload.pop("receipt_sha256", None)
        if digest != sha256_json(payload):
            raise TuningError(
                f"completed tuning receipt was tampered: {path}"
            )
        benchmark = path.parent / "benchmark"
        for field, artifact_name in (
            ("summary_sha256", "summary.json"),
            ("results_sha256", "results.jsonl"),
            ("run_sha256", "run.json"),
        ):
            expected = value.get(field)
            artifact = benchmark / artifact_name
            if not _SHA256_RE.fullmatch(str(expected or "")):
                raise TuningError(
                    f"completed tuning receipt has no {field}: {path}"
                )
            if not artifact.is_file() or artifact.is_symlink():
                raise TuningError(
                    f"completed tuning artifact is missing: {artifact}"
                )
            try:
                actual = file_sha256(artifact)
            except OSError as exc:
                raise TuningError(
                    f"could not verify completed tuning artifact: {artifact}"
                ) from exc
            if actual != expected:
                raise TuningError(
                    f"completed tuning artifact was tampered: {artifact}"
                )
        return value
    return None


def _completed_candidate_result(
    candidate_root: Path,
    candidate_id: str,
) -> dict[str, Any] | None:
    return _completed_receipt(
        candidate_root,
        filename="candidate-result.json",
        identity_field="candidate_id",
        identity=candidate_id,
    )


def _next_attempt_dir(candidate_root: Path) -> Path:
    if candidate_root.exists() and (
        candidate_root.is_symlink() or not candidate_root.is_dir()
    ):
        raise TuningError(
            f"tuning candidate root is unsafe: {candidate_root}"
        )
    candidate_root.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        path = candidate_root / f"attempt-{index:03d}"
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            index += 1
            continue
        return path


def _run_confirmations(
    *,
    plan: Mapping[str, Any],
    discovery: Mapping[str, Any],
    destination: Path,
    journal: Path,
    runner: Callable[..., Mapping[str, Any]],
    out: Callable[[str], None] | None,
) -> list[dict[str, Any]]:
    if discovery.get("complete") is not True:
        return []
    candidates = {
        candidate["candidate_id"]: candidate
        for candidate in plan["candidates"]
    }
    results = []
    for selection in discovery["selections"]:
        if selection.get("supported") is not True:
            continue
        candidate = candidates[selection["candidate_id"]]
        confirmation_id, identity = _confirmation_identity(
            plan=plan,
            selection=selection,
        )
        confirmation_root = destination / "confirmations" / confirmation_id
        _validate_session_path(
            confirmation_root,
            root=destination,
            label="tuning confirmation directory",
        )
        existing = _completed_receipt(
            confirmation_root,
            filename="confirmation-result.json",
            identity_field="confirmation_id",
            identity=confirmation_id,
        )
        if existing is not None:
            results.append(existing)
            if out:
                out(
                    "Reusing completed tuning confirmation "
                    f"{confirmation_id[:12]}."
                )
            continue
        attempt = _next_attempt_dir(confirmation_root)
        _write_new_json(
            attempt / "confirmation.json",
            {
                "schema_version": SCHEMA_VERSION,
                "confirmation_id": confirmation_id,
                **identity,
            },
        )
        _write_new_json(attempt / "candidate.json", candidate)
        spec = build_candidate_provider_spec(plan, candidate)
        spec_path = attempt / "provider-spec.json"
        _write_new_json(spec_path, spec)
        spec_file_sha256 = file_sha256(spec_path)
        benchmark_dir = attempt / "benchmark"
        lane = next(iter(spec["lanes"]))
        _append_journal(
            journal,
            {
                "event": "confirmation_started",
                "at": datetime.now(UTC).isoformat(),
                "confirmation_id": confirmation_id,
                "candidate_id": candidate["candidate_id"],
                "attempt": attempt.name,
            },
        )
        if out:
            out(
                f"Confirming {candidate['backend_id']} "
                f"tier={candidate['tier']} "
                f"batch={candidate['settings']['batch_size']} "
                f"repeats={plan['policy']['confirmation_repeats']}"
            )
        try:
            runner(
                manifest_path=plan["manifest"]["path"],
                input_root=plan["input_root"],
                provider_spec_path=str(spec_path),
                expected_provider_spec_file_sha256=spec_file_sha256,
                dest_dir=str(benchmark_dir),
                lanes=[lane],
                repeats=plan["policy"]["confirmation_repeats"],
                allow_external=False,
                external_plan_path=None,
                external_approval_path=None,
                out=out,
            )
            result = evaluate_confirmation_evidence(
                plan=plan,
                candidate=candidate,
                selection=selection,
                confirmation_id=confirmation_id,
                benchmark_dir=benchmark_dir,
            )
            _write_new_json(attempt / "confirmation-result.json", result)
            results.append(result)
            event = "confirmation_completed"
        except Exception as exc:
            event = "confirmation_abandoned"
            _write_new_json(
                attempt / "abandoned.json",
                make_receipt(
                    "ocr_profile_confirmation_abandoned",
                    {
                        "confirmation_id": confirmation_id,
                        "candidate_id": candidate["candidate_id"],
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                ),
            )
        _append_journal(
            journal,
            {
                "event": event,
                "at": datetime.now(UTC).isoformat(),
                "confirmation_id": confirmation_id,
                "candidate_id": candidate["candidate_id"],
                "attempt": attempt.name,
            },
        )
    return results


def summarize_onboarding_results(
    *,
    plan: Mapping[str, Any],
    discovery: Mapping[str, Any],
    confirmation_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Combine discovery and confirmation into a provisional onboarding."""

    plan_digest = _validated_receipt_sha256(plan, "tuning plan")
    discovery_digest = _validated_receipt_sha256(
        discovery,
        "tuning discovery summary",
    )
    if discovery.get("plan_receipt_sha256") != plan_digest:
        raise TuningError("tuning discovery summary belongs to another plan")
    for index, confirmation in enumerate(confirmation_results, 1):
        _validated_receipt_sha256(
            confirmation,
            f"tuning confirmation {index}",
        )
    confirmation_ids = [
        item.get("candidate_id") for item in confirmation_results
    ]
    confirmations = {
        item.get("candidate_id"): dict(item) for item in confirmation_results
    }
    expected_ids = {
        selection["candidate_id"]
        for selection in discovery["selections"]
        if selection.get("supported") is True
    }
    for selection in discovery["selections"]:
        if selection.get("supported") is not True:
            continue
        confirmation = confirmations.get(selection["candidate_id"])
        if confirmation is not None:
            _validate_confirmation_binding(
                plan=plan,
                selection=selection,
                confirmation=confirmation,
            )
    tiers = []
    for tier in plan["policy"]["memory_tiers"]:
        selections = [
            selection
            for selection in discovery["selections"]
            if selection["tier"] == tier
        ]
        stages = []
        status = "supported"
        reason = None
        for selection in selections:
            stage = {
                "capability": selection["capability"],
                "backend_id": selection["backend_id"],
                "candidate_id": selection.get("candidate_id"),
            }
            if selection.get("supported") is not True:
                stage["status"] = "unsupported"
                stage["reason"] = "no_eligible_discovery_candidate"
                status = "unsupported"
                reason = "no_eligible_discovery_candidate"
            else:
                confirmation = confirmations.get(selection["candidate_id"])
                if confirmation is None:
                    stage["status"] = "pending"
                    stage["reason"] = "confirmation_incomplete"
                    if status != "unsupported":
                        status = "pending"
                        reason = "confirmation_incomplete"
                elif confirmation.get("eligible") is not True:
                    stage["status"] = "unsupported"
                    stage["reason"] = "confirmation_failed"
                    stage["confirmation_receipt_sha256"] = confirmation[
                        "receipt_sha256"
                    ]
                    status = "unsupported"
                    reason = "confirmation_failed"
                else:
                    stage["status"] = "supported"
                    stage["reason"] = None
                    stage["confirmation_receipt_sha256"] = confirmation[
                        "receipt_sha256"
                    ]
            stages.append(stage)
        tiers.append(
            {
                "tier": tier,
                "status": status,
                "supported": status == "supported",
                "reason": reason,
                "stages": stages,
            }
        )
    actual_ids = set(confirmation_ids)
    confirmation_complete = (
        len(confirmation_ids) == len(actual_ids)
        and actual_ids == expected_ids
    )
    complete = discovery.get("complete") is True and confirmation_complete
    blockers = []
    runtime_by_stage: dict[str, set[str]] = {}
    if complete:
        for tier in tiers:
            if tier["supported"] is not True:
                continue
            for stage in tier["stages"]:
                confirmation = confirmations[stage["candidate_id"]]
                runtime_by_stage.setdefault(stage["capability"], set()).add(
                    sha256_json(confirmation["runtime_selector"])
                )
    for stage, identities in sorted(runtime_by_stage.items()):
        if len(identities) != 1:
            blockers.append(f"runtime_identity_drift:{stage}")
    if complete and not any(tier["supported"] for tier in tiers):
        blockers.append("no_supported_memory_tier")
    profile_ready = (
        complete and any(tier["supported"] for tier in tiers) and not blockers
    )
    return make_receipt(
        "ocr_profile_onboarding_summary",
        {
            "plan_receipt_sha256": plan_digest,
            "pipeline_id": plan["pipeline_id"],
            "suite_role": plan["policy"]["suite_role"],
            "suite_id": plan["policy"].get(
                "suite_id", "cuphoton-three-page-smoke"
            ),
            # The checked-in three-page gate is deliberately insufficient for
            # model-quality qualification, regardless of repeat count.
            "qualification_status": "provisional",
            "candidate_count": discovery["candidate_count"],
            "completed_candidate_count": discovery[
                "completed_candidate_count"
            ],
            "discovery_complete": discovery["complete"],
            "discovery_summary_sha256": discovery_digest,
            "discovery_result": dict(discovery),
            "selections": discovery["selections"],
            "expected_confirmation_count": len(expected_ids),
            "completed_confirmation_count": len(actual_ids & expected_ids),
            "confirmation_complete": confirmation_complete,
            "confirmation_results": [
                confirmations[key] for key in sorted(confirmations)
            ],
            "tiers": tiers,
            "complete": complete,
            "profile_ready": profile_ready,
            "profile_blockers": blockers,
        },
    )


def _validated_receipt_sha256(value: Mapping[str, Any], label: str) -> str:
    digest = value.get("receipt_sha256")
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise TuningError(f"{label} has no valid receipt SHA-256")
    payload = dict(value)
    payload.pop("receipt_sha256", None)
    if digest != sha256_json(payload):
        raise TuningError(f"{label} content hash does not match")
    return digest


def _archive_summary(destination: Path, value: Mapping[str, Any]) -> None:
    digest = _validated_receipt_sha256(value, "tuning summary")
    archive = destination / "summaries" / f"{digest}.json"
    if archive.exists():
        if _read_json(archive) != dict(value):
            raise TuningError("tuning summary content-hash collision")
        return
    _write_new_json(archive, value)


def _replace_json(path: Path, value: Mapping[str, Any]) -> None:
    index = 1
    while True:
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{index:03d}.tmp"
        )
        try:
            _write_new_json(temporary, value)
        except FileExistsError:
            index += 1
            continue
        break
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _store_current_summary(
    destination: Path,
    session: Mapping[str, Any],
) -> None:
    _archive_summary(destination, session)
    current = destination / "tuning-summary.json"
    if not current.exists():
        _write_new_json(current, session)
        return
    previous = _read_json(current)
    _archive_summary(destination, previous)
    if previous != dict(session):
        _replace_json(current, session)


def run_tuning_session(
    *,
    plan: Mapping[str, Any],
    dest_dir: str | os.PathLike[str],
    runner: Callable[..., Mapping[str, Any]] | None = None,
    out: Callable[[str], None] | None = None,
    profile_path: str | os.PathLike[str] | None = None,
    profile_id: str | None = None,
    profile_display_name: str | None = None,
) -> dict[str, Any]:
    """Run or resume while preserving immutable, content-hashed evidence."""

    _validated_receipt_sha256(plan, "tuning plan")
    _plan_case_ids(plan)
    profile_arguments = (
        profile_path,
        profile_id,
        profile_display_name,
    )
    if any(value is not None for value in profile_arguments) and not all(
        value is not None for value in profile_arguments
    ):
        raise TuningError(
            "profile_path, profile_id, and profile_display_name must be "
            "supplied together"
        )

    validated_profile_path = (
        _validate_profile_output_path(plan, profile_path)
        if profile_path is not None
        else None
    )
    destination = _validate_tuning_destination(plan, dest_dir)
    _validate_session_path(
        destination,
        root=destination,
        label="tuning destination",
    )
    plan_path = destination / "plan.json"
    if plan_path.exists():
        existing = _read_json(plan_path)
        if existing != dict(plan):
            raise TuningError("tuning destination belongs to another plan")
    else:
        if any(destination.iterdir()):
            raise TuningError(
                "new tuning destination must be empty before plan creation"
            )
        _write_new_json(plan_path, dict(plan))
    journal = destination / "journal.jsonl"
    if journal.exists() and (journal.is_symlink() or not journal.is_file()):
        raise TuningError("tuning journal path is unsafe")
    journal.touch(mode=0o600, exist_ok=True)
    if runner is None:
        from .ocr_benchmark import run_benchmark

        runner = run_benchmark
    results = []
    for candidate in plan["candidates"]:
        candidate_id = candidate["candidate_id"]
        candidate_root = destination / "candidates" / candidate_id
        _validate_session_path(
            candidate_root,
            root=destination,
            label="tuning candidate directory",
        )
        existing = _completed_candidate_result(candidate_root, candidate_id)
        if existing is not None:
            results.append(existing)
            if out:
                out(
                    f"Reusing completed tuning candidate {candidate_id[:12]}."
                )
            continue
        attempt = _next_attempt_dir(candidate_root)
        _write_new_json(attempt / "candidate.json", candidate)
        spec = build_candidate_provider_spec(plan, candidate)
        spec_path = attempt / "provider-spec.json"
        _write_new_json(spec_path, spec)
        spec_file_sha256 = file_sha256(spec_path)
        benchmark_dir = attempt / "benchmark"
        lane = next(iter(spec["lanes"]))
        _append_journal(
            journal,
            {
                "event": "candidate_started",
                "at": datetime.now(UTC).isoformat(),
                "candidate_id": candidate_id,
                "attempt": attempt.name,
            },
        )
        if out:
            out(
                f"Tuning {candidate['backend_id']} "
                f"tier={candidate['tier']} "
                f"batch={candidate['settings']['batch_size']}"
            )
        try:
            runner(
                manifest_path=plan["manifest"]["path"],
                input_root=plan["input_root"],
                provider_spec_path=str(spec_path),
                expected_provider_spec_file_sha256=spec_file_sha256,
                dest_dir=str(benchmark_dir),
                lanes=[lane],
                repeats=plan["policy"]["discovery_repeats"],
                allow_external=False,
                external_plan_path=None,
                external_approval_path=None,
                out=out,
            )
            result = evaluate_candidate_evidence(
                candidate=candidate,
                benchmark_dir=benchmark_dir,
                expected_case_ids=plan["manifest"]["case_ids"],
                expected_repeats=plan["policy"]["discovery_repeats"],
            )
            _write_new_json(attempt / "candidate-result.json", result)
            results.append(result)
            event = "candidate_completed"
        except Exception as exc:
            event = "candidate_abandoned"
            _write_new_json(
                attempt / "abandoned.json",
                make_receipt(
                    "ocr_profile_candidate_abandoned",
                    {
                        "candidate_id": candidate_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                ),
            )
        _append_journal(
            journal,
            {
                "event": event,
                "at": datetime.now(UTC).isoformat(),
                "candidate_id": candidate_id,
                "attempt": attempt.name,
            },
        )
    discovery = summarize_tuning_results(
        plan=plan,
        candidate_results=results,
    )
    if "confirmation_repeats" in plan["policy"]:
        confirmation_results = _run_confirmations(
            plan=plan,
            discovery=discovery,
            destination=destination,
            journal=journal,
            runner=runner,
            out=out,
        )
        session = summarize_onboarding_results(
            plan=plan,
            discovery=discovery,
            confirmation_results=confirmation_results,
        )
    else:
        session = discovery
    _store_current_summary(destination, session)
    if validated_profile_path is not None:
        write_provisional_execution_profile(
            plan=plan,
            tuning_summary=session,
            profile_id=str(profile_id),
            display_name=str(profile_display_name),
            path=validated_profile_path,
        )
    return session


def summarize_tuning_results(
    *,
    plan: Mapping[str, Any],
    candidate_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_id = {item.get("candidate_id"): item for item in candidate_results}
    selections = []
    for capability, backend_id in _stage_records(plan["pipeline_id"]):
        for tier in plan["policy"]["memory_tiers"]:
            candidates = [
                candidate
                for candidate in plan["candidates"]
                if candidate["capability"] == capability
                and candidate["backend_id"] == backend_id
                and candidate["tier"] == tier
            ]
            eligible = [
                (candidate, by_id[candidate["candidate_id"]])
                for candidate in candidates
                if candidate["candidate_id"] in by_id
                and by_id[candidate["candidate_id"]].get("eligible") is True
            ]
            eligible.sort(
                key=lambda pair: (
                    -float(pair[1].get("cold_pages_per_second") or 0.0),
                    int(pair[1].get("peak_memory_bytes") or 2**63),
                    int(pair[0]["settings"]["batch_size"]),
                )
            )
            if eligible:
                candidate, result = eligible[0]
                selections.append(
                    {
                        "capability": capability,
                        "backend_id": backend_id,
                        "tier": tier,
                        "supported": True,
                        "candidate_id": candidate["candidate_id"],
                        "settings": candidate["settings"],
                        "batch_semantics": candidate["batch_semantics"],
                        "evidence_receipt_sha256": result["receipt_sha256"],
                        "cold_pages_per_second": result.get(
                            "cold_pages_per_second"
                        ),
                        "peak_memory_bytes": result.get("peak_memory_bytes"),
                    }
                )
            else:
                selections.append(
                    {
                        "capability": capability,
                        "backend_id": backend_id,
                        "tier": tier,
                        "supported": False,
                        "candidate_id": None,
                        "settings": None,
                        "batch_semantics": (
                            "layout_block_batch"
                            if backend_id.startswith("mineru")
                            else "page_batch"
                        ),
                        "reason": "no eligible discovery candidate",
                    }
                )
    completed = len(by_id)
    expected = len(plan["candidates"])
    return make_receipt(
        "ocr_profile_tuning_summary",
        {
            "plan_receipt_sha256": plan["receipt_sha256"],
            "pipeline_id": plan["pipeline_id"],
            "suite_role": plan["policy"]["suite_role"],
            "qualification_status": "provisional",
            "candidate_count": expected,
            "completed_candidate_count": completed,
            "complete": completed == expected,
            "selections": selections,
        },
    )


def _profile_memory_budgets(
    plan: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    selector = plan["hardware"]["selector"]
    defaults = _memory_budget_records(
        {"selector": selector},
        MEMORY_TIERS,
    )
    supplied = plan.get("memory_budgets", {})
    records = {}
    for tier in MEMORY_TIERS:
        source = supplied.get(tier, defaults[tier])
        fallback = defaults[tier]
        fraction = source.get(
            "budget_fraction", source.get("target_fraction")
        )
        budget = source.get(
            "budget_bytes", source.get("capacity_budget_bytes")
        )
        reserve = source.get("reserve_bytes", fallback["reserve_bytes"])
        if not isinstance(fraction, (int, float)):
            raise TuningError(f"memory tier {tier} has no budget fraction")
        if not isinstance(budget, int) or budget <= 0:
            raise TuningError(f"memory tier {tier} has no positive budget")
        if not isinstance(reserve, int) or reserve <= 0:
            raise TuningError(f"memory tier {tier} has no positive reserve")
        records[tier] = {
            "budget_fraction": float(fraction),
            "budget_bytes": budget,
            "reserve_bytes": reserve,
        }
    return records


def build_provisional_execution_profile(
    *,
    plan: Mapping[str, Any],
    tuning_summary: Mapping[str, Any],
    profile_id: str,
    display_name: str,
) -> dict[str, Any]:
    """Build a sealed profile from complete three-page smoke evidence."""

    from .ocr_profiles import (
        get_pipeline_definition,
        qualification_policy_sha256,
        seal_execution_profile,
    )

    plan_digest = _validated_receipt_sha256(plan, "tuning plan")
    _plan_case_ids(plan)
    _validated_receipt_sha256(
        tuning_summary,
        "tuning onboarding summary",
    )
    if tuning_summary.get("kind") != "ocr_profile_onboarding_summary":
        raise TuningError(
            "a confirmation-complete onboarding summary is required"
        )
    if tuning_summary.get("plan_receipt_sha256") != plan_digest:
        raise TuningError("tuning summary belongs to another plan")
    discovery = tuning_summary.get("discovery_result")
    if not isinstance(discovery, Mapping):
        raise TuningError("tuning summary has no discovery receipt")
    discovery_digest = _validated_receipt_sha256(
        discovery,
        "tuning discovery summary",
    )
    if (
        discovery_digest != tuning_summary.get("discovery_summary_sha256")
        or discovery.get("plan_receipt_sha256") != plan_digest
        or discovery.get("selections") != tuning_summary.get("selections")
    ):
        raise TuningError("tuning discovery receipt does not match summary")
    if tuning_summary.get("qualification_status") != "provisional":
        raise TuningError("three-page smoke evidence must remain provisional")
    if tuning_summary.get("profile_ready") is not True:
        blockers = tuning_summary.get("profile_blockers") or [
            "onboarding is incomplete"
        ]
        raise TuningError(
            "tuning evidence cannot produce a profile: "
            + ", ".join(str(item) for item in blockers)
        )

    pipeline = get_pipeline_definition(plan["pipeline_id"])
    pipeline_stages = {
        stage["stage_id"]: stage for stage in pipeline["stages"]
    }
    confirmation_results = tuning_summary.get("confirmation_results")
    if not isinstance(confirmation_results, list):
        raise TuningError("tuning summary has no confirmation receipts")
    confirmation_ids = []
    for index, item in enumerate(confirmation_results, 1):
        if not isinstance(item, Mapping):
            raise TuningError(f"tuning confirmation {index} is invalid")
        _validated_receipt_sha256(item, f"tuning confirmation {index}")
        confirmation_ids.append(item.get("candidate_id"))
    if len(confirmation_ids) != len(set(confirmation_ids)):
        raise TuningError("tuning summary has duplicate confirmations")
    confirmations = {
        item["candidate_id"]: item for item in confirmation_results
    }
    selections = {
        (item["capability"], item["tier"]): item
        for item in tuning_summary["selections"]
        if item.get("supported") is True
    }
    tier_status = {item["tier"]: item for item in tuning_summary["tiers"]}
    for selection in selections.values():
        confirmation = confirmations.get(selection["candidate_id"])
        if confirmation is None:
            raise TuningError(
                "supported selection has no confirmation receipt"
            )
        _validate_confirmation_binding(
            plan=plan,
            selection=selection,
            confirmation=confirmation,
        )
    for tier in tuning_summary["tiers"]:
        for stage in tier.get("stages", []):
            candidate_id = stage.get("candidate_id")
            confirmation = confirmations.get(candidate_id)
            expected_receipt = stage.get("confirmation_receipt_sha256")
            if expected_receipt is None:
                continue
            if (
                confirmation is None
                or confirmation.get("receipt_sha256") != expected_receipt
            ):
                raise TuningError(
                    "tuning tier confirmation receipt does not match summary"
                )

    stages = {}
    supported_candidate_ids = {
        selection["candidate_id"]
        for (_capability, tier), selection in selections.items()
        if tier_status[tier].get("supported") is True
    }
    for stage_id, pipeline_stage in pipeline_stages.items():
        stage_confirmations = [
            item
            for item in tuning_summary["confirmation_results"]
            if item.get("candidate_id") in supported_candidate_ids
            if item.get("capability") == pipeline_stage["capability"]
            and item.get("backend_id") == pipeline_stage["backend_id"]
        ]
        if not stage_confirmations:
            raise TuningError(
                f"no confirmation identity for stage {stage_id}"
            )
        snapshot_hashes = set()
        expected_model = pipeline_stage["model"]
        for item in stage_confirmations:
            observed_model = item.get("model_evidence")
            if not isinstance(observed_model, Mapping):
                raise TuningError(
                    f"model identity is missing for stage {stage_id}"
                )
            snapshot_files_sha256 = observed_model.get(
                "snapshot_files_sha256"
            )
            if (
                observed_model.get("requested") != expected_model["requested"]
                or observed_model.get("revision")
                != expected_model["revision"]
                or observed_model.get("resolved_revision")
                != expected_model["revision"]
                or not isinstance(snapshot_files_sha256, str)
                or not _SHA256_RE.fullmatch(snapshot_files_sha256)
            ):
                raise TuningError(
                    f"model identity differs for stage {stage_id}"
                )
            snapshot_hashes.add(snapshot_files_sha256)
        if len(snapshot_hashes) != 1:
            raise TuningError(
                "model snapshot identity drift across tiers for stage "
                f"{stage_id}"
            )
        snapshot_files_sha256 = next(iter(snapshot_hashes))

        identities = []
        for tier in plan["policy"]["memory_tiers"]:
            status = tier_status[tier]
            if status.get("supported") is not True:
                continue
            selection = selections[(stage_id, tier)]
            confirmation = confirmations[selection["candidate_id"]]
            identities.append(confirmation)
        if not identities:
            raise TuningError(
                f"no supported confirmation identity for stage {stage_id}"
            )
        runtime_digests = {
            sha256_json(item["runtime_selector"]) for item in identities
        }
        if len(runtime_digests) != 1:
            raise TuningError(
                f"runtime identity drift across tiers for stage {stage_id}"
            )
        stages[stage_id] = {
            "backend_id": pipeline_stage["backend_id"],
            "capability": pipeline_stage["capability"],
            "model": {
                **dict(pipeline_stage["model"]),
                "snapshot_files_sha256": snapshot_files_sha256,
            },
            "batch_semantics": pipeline_stage["batch_semantics"],
            "runtime_selector": dict(identities[0]["runtime_selector"]),
        }

    budgets = _profile_memory_budgets(plan)
    memory_tiers = {}
    evidence = []
    requested_tiers = set(plan["policy"]["memory_tiers"])
    suite_id = plan["policy"].get("suite_id", "cuphoton-three-page-smoke")
    observed_confirmation_repeats = {
        confirmation.get("repeats") for confirmation in confirmations.values()
    }
    if observed_confirmation_repeats != {
        plan["policy"]["confirmation_repeats"]
    }:
        raise TuningError(
            "confirmation repeat evidence differs from the plan"
        )
    confirmation_repeats = next(iter(observed_confirmation_repeats))
    for tier in MEMORY_TIERS:
        budget = budgets[tier]
        status = tier_status.get(tier)
        supported = bool(status and status.get("supported") is True)
        if tier not in requested_tiers:
            reason = "tier_not_tuned"
        elif supported:
            reason = None
        else:
            reason = str((status or {}).get("reason") or "tier_not_supported")
        tier_stages = {}
        confirmation_by_stage = {}
        if supported:
            for stage_id in pipeline_stages:
                selection = selections[(stage_id, tier)]
                confirmation = confirmations[selection["candidate_id"]]
                if confirmation.get("eligible") is not True:
                    raise TuningError(
                        f"supported tier {tier} has failed confirmation"
                    )
                peak = confirmation.get("peak_memory_bytes")
                if type(peak) is not int or peak <= 0:
                    raise TuningError(
                        f"supported tier {tier} has invalid memory evidence"
                    )
                throughput = confirmation.get("cold_pages_per_second")
                if (
                    not isinstance(throughput, (int, float))
                    or isinstance(throughput, bool)
                    or not math.isfinite(float(throughput))
                    or throughput <= 0
                ):
                    raise TuningError(
                        f"supported tier {tier} has invalid throughput "
                        "evidence"
                    )
                tier_stages[stage_id] = {
                    "settings": dict(selection["settings"]),
                    "observed_peak_memory_bytes": peak,
                    "confirmed_pages_per_second": float(throughput),
                }
                confirmation_by_stage[stage_id] = confirmation
            evidence.append(
                {
                    "suite_id": suite_id,
                    "memory_tier": tier,
                    "manifest_sha256": plan["manifest"]["sha256"],
                    "summary_sha256": sha256_json(
                        {
                            stage_id: confirmation_by_stage[stage_id][
                                "receipt_sha256"
                            ]
                            for stage_id in sorted(confirmation_by_stage)
                        }
                    ),
                    "experiment_sha256": sha256_json(
                        {
                            stage_id: confirmation_by_stage[stage_id][
                                "experiment_sha256"
                            ]
                            for stage_id in sorted(confirmation_by_stage)
                        }
                    ),
                    "repeats": confirmation_repeats,
                    "all_pass": all(
                        item["all_pass"]
                        for item in confirmation_by_stage.values()
                    ),
                    "provenance_complete": all(
                        item["provenance_complete"]
                        for item in confirmation_by_stage.values()
                    ),
                }
            )
        memory_tiers[tier] = {
            "supported": supported,
            "reason": reason,
            **budget,
            "stages": tier_stages,
        }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "ocr_execution_profile",
        "profile_id": profile_id,
        "display_name": display_name,
        "pipeline_id": pipeline["pipeline_id"],
        "pipeline_definition_sha256": pipeline["definition_sha256"],
        "hardware_selector": dict(plan["hardware"]["selector"]),
        "stages": stages,
        "memory_tiers": memory_tiers,
        "qualification": {
            "state": "provisional",
            "policy_sha256": qualification_policy_sha256(),
            "required_suites": [suite_id],
            "evidence": evidence,
        },
    }
    return seal_execution_profile(payload)


def write_provisional_execution_profile(
    *,
    plan: Mapping[str, Any],
    tuning_summary: Mapping[str, Any],
    profile_id: str,
    display_name: str,
    path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Create, or safely reuse, one explicit provisional profile path."""

    from .ocr_profiles import (
        OcrProfileError,
        load_execution_profile,
        write_execution_profile,
    )

    destination = _validate_profile_output_path(plan, path)
    profile = build_provisional_execution_profile(
        plan=plan,
        tuning_summary=tuning_summary,
        profile_id=profile_id,
        display_name=display_name,
    )
    if destination.exists():
        try:
            existing = load_execution_profile(destination)
        except OcrProfileError as exc:
            raise TuningError(
                f"existing execution profile is invalid: {destination}"
            ) from exc
        if existing != profile:
            raise TuningError(
                f"refusing to replace different profile: {destination}"
            )
        return existing
    try:
        return write_execution_profile(profile, destination)
    except OcrProfileError as exc:
        raise TuningError(str(exc)) from exc
